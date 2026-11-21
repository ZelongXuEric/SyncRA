"""MiniCPM-o 4.5, loaded without its speech generator.

A clip enters the prompt as one-second chunks, each a frame followed by its
second of audio. The processor expands every chunk into a fixed number of video
and audio placeholders and reports where they sit, so the native intervals are
read directly from those tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..intervals import Intervals
from .base import Backbone, Example, count_target_tokens, mask_prompt, target_window, to_device

CHUNK_SECONDS = 1.0
SAMPLE_RATE = 16000
CHUNK_PLACEHOLDER = "<image>./</image><audio>./</audio>"
VIDEO_TOKENS = 64  # tokens per chunk after placeholder expansion
AUDIO_TOKENS = 10
MAX_TRAIN_CHUNKS = 180
MAX_EVAL_CHUNKS = 256
EOS_TOKEN_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>


def decode_chunks(video: Path, seconds: float | None = None) -> tuple[list[Any], list[np.ndarray]]:
    """Frames at 1 FPS and, for each, its second of mono 16 kHz audio.

    A frame is taken whenever the timestamps reach the next whole second. The
    clip is cut to the whole seconds present in both streams and, if given, to
    its first `seconds`.
    """
    import av

    with av.open(str(video)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError(f"{video} lacks a video or an audio stream")
        stream = container.streams.video[0]
        frames, next_time = [], 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            if time + 1e-6 >= next_time:
                frames.append(frame.to_image())
                next_time += CHUNK_SECONDS
            if seconds is not None and time > seconds:
                break
    with av.open(str(video)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        blocks = [
            block.to_ndarray().reshape(-1)
            for frame in container.decode(container.streams.audio[0])
            for block in resampler.resample(frame)
        ]
    waveform = np.concatenate(blocks).astype(np.float32) / 32768.0 if blocks else np.zeros(0, np.float32)

    samples = int(SAMPLE_RATE * CHUNK_SECONDS)
    num_chunks = min(len(frames), len(waveform) // samples)
    if seconds is not None:
        num_chunks = min(num_chunks, int(seconds / CHUNK_SECONDS))
    if num_chunks < 2:
        raise ValueError(f"{video} holds only {num_chunks} whole audio-video chunks")
    return frames[:num_chunks], [waveform[k * samples:(k + 1) * samples] for k in range(num_chunks)]


def subsample(num_chunks: int, limit: int) -> list[int]:
    """Indices of the chunks to keep: all of them, or `limit` uniformly spaced ones."""
    if num_chunks <= limit:
        return list(range(num_chunks))
    return [int((k + 0.5) * num_chunks / limit) for k in range(limit)]


def causal_mask(length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive causal attention mask of shape [1, 1, length, length]."""
    mask = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=device)
    return mask.triu_(diagonal=1).view(1, 1, length, length)


def chunk_intervals(inputs: dict[str, Any], num_chunks: int, prompt_length: int) -> Intervals:
    """One interval per chunk, from the processor's tables of placeholder positions.

    Every chunk must hold its video tokens and then its audio tokens, in clip
    order and inside the prompt, so a wrong layout cannot pass silently.
    """
    video = [(start, end) for start, end in inputs["image_bound"][0].tolist()]
    audio = [(start, end) for start, end in inputs["audio_bounds"][0].tolist()]
    if len(video) != num_chunks or len(audio) != num_chunks:
        raise ValueError(f"{len(video)} video and {len(audio)} audio spans for {num_chunks} chunks")
    media_end = 0
    for k, ((video_start, video_end), (audio_start, audio_end)) in enumerate(zip(video, audio)):
        in_order = media_end <= video_start and video_end <= audio_start
        if not in_order or video_end - video_start != VIDEO_TOKENS or audio_end - audio_start != AUDIO_TOKENS:
            raise ValueError(f"chunk {k} is not {VIDEO_TOKENS} video tokens followed by {AUDIO_TOKENS} audio tokens")
        media_end = audio_end
    if media_end >= prompt_length:
        raise ValueError("media placeholders overlap the assistant target")
    return Intervals(
        audio=[[span] for span in audio],
        video=[[span] for span in video],
        index=list(range(num_chunks)),
        seconds=CHUNK_SECONDS,
    )


class MiniCPMo(Backbone):
    """MiniCPM-o 4.5: full fine-tuning of the understanding model; native 1 s intervals.

    Every parameter trains except the audio encoder's sinusoidal position table,
    which the released model itself keeps frozen.

    Without an attention mask the text decoder runs its attention in the fused
    causal mode; with an additive mask it takes a different kernel, and in
    bfloat16 the two give substantially different logits on this model. Every
    decoder pass in training and in generation therefore carries an explicit
    causal mask, so that all of them share one numerical path.
    """

    repo = "openbmb/MiniCPM-o-4_5"
    revision = "503e754207c94da6bb26850b4469f367c9ea3582"
    hidden_size, num_blocks, layer = 4096, 36, 9

    def __init__(self, model_path: str | Path, device: torch.device, train: bool = False):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        # "sdpa" selects the attention of the text decoder and the audio encoder; the vision encoder runs eager.
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            init_tts=False, init_audio=True, init_vision=True,
        ).to(device)
        if train:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Activations of an encoder left without checkpointing would dominate memory.
            towers = (self.model.llm.model, self.model.vpm.encoder, self.model.apm)
            if not all(tower.gradient_checkpointing for tower in towers):
                raise RuntimeError("gradient checkpointing did not reach the decoder and both encoders")
            self.model.train()
        else:
            self.model.requires_grad_(False).eval()

    # ------------------------------------------------------------------ inputs
    def _render(self, num_chunks: int, text: str, target: str | None = None) -> str:
        """The chat prompt for `text` about `num_chunks` chunks, completed by `target` if given."""
        messages = [{"role": "user", "content": CHUNK_PLACEHOLDER * num_chunks + text}]
        if target is not None:
            messages.append({"role": "assistant", "content": target})
        # With reasoning disabled the template closes an empty reasoning block inside the prompt, so the
        # target is the answer itself. The speech template would end the prompt with a speech token that
        # a rendered assistant turn does not contain, and the target would no longer extend the prompt.
        return self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=target is None,
            enable_thinking=False, use_tts_template=False,
        )

    def _encode(self, text: str, frames: list[Any], audio: list[np.ndarray]) -> dict[str, Any]:
        """Expand the chunk placeholders of `text` and extract the image and audio features."""
        # The shared part id joins the segments into one track for the audio encoder, each frame
        # stays a single unsliced image without an index tag, and the sequence is never truncated.
        inputs = dict(self.processor(
            [text], [frames], [audio], [[0] * len(audio)],
            max_slice_nums=1, use_image_id=False, stream_input=False, return_tensors="pt", max_length=None,
        ))
        inputs.pop("image_sizes")  # preprocessing metadata, not a model input
        # The model reads positions from its inputs rather than deriving them.
        inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[-1]).unsqueeze(0)
        return inputs

    def prepare(self, video: Path, duration: float, question: str, target: str) -> Example:
        frames, audio = decode_chunks(video, duration)
        if len(frames) > MAX_TRAIN_CHUNKS:
            raise ValueError(f"{video} has {len(frames)} chunks; training supports at most {MAX_TRAIN_CHUNKS}")
        prompt = self._render(len(frames), question)
        full = self._render(len(frames), question, target)
        num_target_tokens, text_ids = count_target_tokens(self.processor.tokenizer, prompt, full)
        inputs = self._encode(full, frames, audio)
        labels = mask_prompt(inputs["input_ids"], num_target_tokens, text_ids)
        intervals = chunk_intervals(inputs, len(frames), labels.shape[-1] - num_target_tokens)
        return Example(inputs, labels, num_target_tokens, intervals)

    # ----------------------------------------------------------------- forward
    def loss(self, example: Example, layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = to_device(example.inputs, self.device)
        labels = target_window(example.labels.to(self.device), example.num_target_tokens)
        # Logits are computed for the scored positions only; over the full sequence they do not fit in memory.
        outputs = self.model(
            inputs, labels=labels, logits_to_keep=labels.shape[-1],
            attention_mask=causal_mask(inputs["input_ids"].shape[-1], self.model.dtype, self.device),
            use_cache=False, output_hidden_states=layer is not None,
        )
        if layer is None:
            return outputs.loss, None
        assert len(outputs.hidden_states) == self.num_blocks + 1
        return outputs.loss, outputs.hidden_states[layer][0]

    @torch.inference_mode()
    def generate(self, video: Path, prompt: str, max_new_tokens: int = 1280, answer_prefix: str | None = None) -> str:
        """Greedy answer to `prompt` about the clip; clips beyond 256 chunks are subsampled uniformly.

        `answer_prefix` opens the assistant turn and is returned in front of its
        continuation. The released checkpoint is evaluated with the prefix
        "Final answer:" and six new tokens, because it otherwise replies in prose
        that an answer parser cannot read; fine-tuned checkpoints use the defaults.
        """
        frames, audio = decode_chunks(video)
        keep = subsample(len(frames), MAX_EVAL_CHUNKS)
        frames, audio = [frames[k] for k in keep], [audio[k] for k in keep]
        prefix = answer_prefix or ""
        inputs = to_device(self._encode(self._render(len(frames), prompt) + prefix, frames, audio), self.device)
        tokens = self._greedy(inputs, max_new_tokens)
        return prefix + self.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def states(self, video: Path, duration: float, question: str, layers: Sequence[int]) -> tuple[list[torch.Tensor], Intervals]:
        frames, audio = decode_chunks(video, duration)
        inputs = self._encode(self._render(len(frames), question), frames, audio)
        intervals = chunk_intervals(inputs, len(frames), inputs["input_ids"].shape[-1])
        # No explicit mask: the block-selection scores reported in the paper come from the decoder's
        # default causal mode, and passing the mask would not reproduce them.
        outputs = self.model(to_device(inputs, self.device), use_cache=False, output_hidden_states=True)
        return [outputs.hidden_states[layer][0] for layer in layers], intervals

    def _greedy(self, inputs: dict[str, Any], max_new_tokens: int) -> list[int]:
        """Greedy continuation with a key-value cache, ending with the stop token if one is produced."""
        model, llm = self.model, self.model.llm
        embeds, _ = model.get_vllm_embedding(inputs)
        embeds = model.get_omni_embedding(inputs, input_embeddings=embeds, chunk_length=model.config.audio_chunk_length)
        length = embeds.shape[1]
        outputs = llm(
            inputs_embeds=embeds, position_ids=inputs["position_ids"],
            attention_mask=causal_mask(length, embeds.dtype, self.device), use_cache=True, logits_to_keep=1,
        )
        tokens = [int(outputs.logits[0, -1].argmax())]
        while tokens[-1] not in EOS_TOKEN_IDS and len(tokens) < max_new_tokens:
            position = length + len(tokens) - 1
            token = torch.tensor([[tokens[-1]]], device=self.device)
            # The new token may attend to everything cached, stated as an explicit all-zero mask.
            outputs = llm(
                inputs_embeds=llm.get_input_embeddings()(token),
                position_ids=torch.tensor([[position]], device=self.device),
                attention_mask=torch.zeros((1, 1, 1, position + 1), dtype=embeds.dtype, device=self.device),
                past_key_values=outputs.past_key_values, use_cache=True, logits_to_keep=1,
            )
            tokens.append(int(outputs.logits[0, -1].argmax()))
        return tokens
