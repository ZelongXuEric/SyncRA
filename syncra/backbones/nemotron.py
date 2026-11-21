"""Nemotron-3-Nano-Omni-30B-A3B: a hybrid Mamba2 / attention / MoE language model with vision and audio towers.

Audio and video are not interleaved. The processor writes one `<img>...</img>`
group per two-frame tubelet, each preceded by its timestamps, and then a single
`<so_start>...<so_end>` audio block holding one token per 80 ms. Interval
membership therefore follows time: a tubelet belongs to the 2 s interval that
contains its first frame, and interval k owns audio tokens [25k, 25k + 25).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..intervals import Intervals, Span
from .base import IGNORE, Backbone, Example, count_target_tokens, mask_prompt, target_window, to_device

MODEL_PACKAGE = "nemotron3_omni_remote"
RADIO_PACKAGE = "radio_remote"
RADIO_REPO = "nvidia/C-RADIOv4-H"  # code of the vision tower; its weights are part of the checkpoint
RADIO_REVISION = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"

# Ids in the released tokenizer.
IMAGE, IMAGE_START, IMAGE_END = 18, 19, 20  # <image>, <img>, </img>
AUDIO, AUDIO_START, AUDIO_END = 27, 28, 29  # <so_embedding>, <so_start>, <so_end>
EOS_TOKEN_IDS = [2, 11]  # </s>, <|im_end|>

MEDIA_PREFIX = "<video><so_embedding>\n"  # the processor expands these into the video and audio blocks
FPS, MIN_FRAMES, MAX_FRAMES = 2.0, 4, 256
FRAMES_PER_TUBELET = 2
SAMPLE_RATE = 16000
SECONDS_PER_INTERVAL = 2.0
AUDIO_TOKENS_PER_INTERVAL = 25
MAX_AUDIO_TOKENS = 5000  # position limit of the audio encoder, i.e. 400 s


class NemotronOmni(Backbone):
    """Routed experts frozen; routers, shared experts and every other module trained."""

    repo = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    revision = "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
    hidden_size, num_blocks, layer = 2688, 52, 26

    TRAINABLE_PARAMETERS = 3_640_738_752

    def __init__(self, model_path: str | Path, device: torch.device, train: bool = False):
        self.device = device
        # Processes load one at a time: each first holds the whole 62 GiB checkpoint in host memory.
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        for turn in range(world):
            if turn == rank:
                self.processor, self.tokenizer, self.model = self._load(Path(model_path))
            if world > 1:
                dist.barrier()
        if train:
            self._set_trainable()
            language_model = self.model.language_model
            language_model.config.use_cache = False
            # Every block supports checkpointing, although the class does not declare it.
            language_model.supports_gradient_checkpointing = True
            language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.vision_model.radio_model.model.set_grad_checkpointing(True)
            self.model.sound_encoder.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.set_training(True)
        else:
            self.model.requires_grad_(False).eval()

    def _load(self, model_path: Path) -> tuple[Any, Any, torch.nn.Module]:
        from transformers import AutoModel, AutoTokenizer

        # The checkpoint's code is imported directly: the dynamic-module loader of transformers
        # does not follow its nested relative imports.
        configuration, modeling, processing, image_processing = (
            _import_from(model_path, MODEL_PACKAGE, name) for name in ("configuration", "modeling", "processing", "image_processing")
        )
        radio = _import_from(_radio_code(), RADIO_PACKAGE, "hf_model")

        config = configuration.NemotronH_Nano_Omni_Reasoning_V3_Config.from_pretrained(model_path)
        # The wrapper has no attention layers of its own and accepts only "eager".
        config._attn_implementation = "eager"
        config.llm_config._attn_implementation = "sdpa"
        config.vision_config.use_flash_attn = False
        # Build the vision tower from the pinned code instead of the revision the hub currently serves.
        config.vision_config.auto_map = {}
        AutoModel.register(type(config.vision_config), radio.RADIOModel, exist_ok=True)

        # Passing the config keeps transformers from looking it up itself and asking to run the checkpoint's code.
        tokenizer = AutoTokenizer.from_pretrained(model_path, config=config)
        processor = processing.NemotronH_Nano_Omni_Reasoning_V3Processor(
            image_processor=image_processing.NemotronH_Nano_Omni_Reasoning_V3ImageProcessor.from_pretrained(model_path),
            tokenizer=tokenizer,
            chat_template=(model_path / "chat_template.jinja").read_text(encoding="utf-8"),
        )
        model = modeling.NemotronH_Nano_Omni_Reasoning_V3.from_pretrained(model_path, config=config, dtype=torch.bfloat16)
        model = model.eval().to(self.device)
        model.video_pruning_rate = 0.0  # keep every video token; pruning is an optional inference-time speed-up

        # The checkpoint does not store this index buffer and loading leaves it uninitialised,
        # while the vision tower indexes with it on every forward.
        radio_model = model.vision_model.radio_model
        teachers = model.vision_model.config.args["teachers"]
        radio_model.summary_idxs = torch.tensor(
            [i for i, teacher in enumerate(teachers) if teacher.get("use_summary", True)],
            dtype=torch.int64, device=radio_model.summary_idxs.device,
        )

        language_model = model.language_model
        if language_model.config._attn_implementation != "sdpa":
            language_model.set_attn_implementation("sdpa")
        # Without the CUDA kernels the Mamba2 mixers fall back to a path with different numerics.
        mamba_kernels = getattr(sys.modules[f"{MODEL_PACKAGE}.modeling_nemotron_h"], "is_fast_path_available", False)
        if language_model.config._attn_implementation != "sdpa" or not mamba_kernels:
            raise RuntimeError("the language model needs SDPA attention and the mamba-ssm / causal-conv1d kernels")
        return processor, tokenizer, model

    def _set_trainable(self) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(".mixer.experts." not in name)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable != self.TRAINABLE_PARAMETERS:
            raise RuntimeError(f"unexpected trainable parameter count {trainable:,}")

    def set_training(self, mode: bool) -> None:
        self.model.train(mode)
        if mode:
            # Checkpoints hold trainable parameters only. The BatchNorm statistics of the audio
            # encoder are buffers, so they must not move if a checkpoint is to reproduce the model.
            for module in self.model.modules():
                if getattr(module, "track_running_stats", False) and getattr(module, "running_mean", None) is not None:
                    module.eval()

    # ------------------------------------------------------------------ inputs
    def _render(self, text: str, target: str | None = None) -> str:
        """Chat prompt with reasoning disabled, optionally followed by the assistant target."""
        messages = [{"role": "user", "content": MEDIA_PREFIX + text}]
        if target is not None:
            messages.append({"role": "assistant", "content": target})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=target is None, enable_thinking=False
        )

    def _encode(self, text: str, clip: Clip) -> tuple[dict[str, Any], torch.Tensor]:
        """Run the processor on rendered `text`: the inputs of a forward pass, and the attention mask."""
        from transformers.video_utils import VideoMetadata

        # The processor prints each tubelet's timestamps from the source frame rate and indices.
        metadata = VideoMetadata(
            total_num_frames=len(clip.frames), fps=clip.source_fps, frames_indices=clip.frame_indices.tolist(),
            duration=clip.duration, video_backend=None,
        )
        encoded = self.processor(
            text=[text], videos=clip.frames, videos_kwargs={"video_metadata": metadata}, audio=[clip.waveform],
            return_tensors="pt",
        )
        # The audio features are computed from the waveform inside the forward pass, as in `generate`.
        inputs = {"input_ids": encoded["input_ids"], "pixel_values_videos": encoded["pixel_values_videos"], "waveform": clip.waveform}
        return inputs, encoded["attention_mask"]

    def prepare(self, video: Path, duration: float, question: str, target: str) -> Example:
        prompt, full = self._render(question), self._render(question, target)
        num_target_tokens, text_ids = count_target_tokens(self.tokenizer, prompt, full)
        clip = decode_clip(video, duration)
        inputs, _ = self._encode(full, clip)
        labels = mask_prompt(inputs["input_ids"], num_target_tokens, text_ids)
        intervals = locate_intervals(inputs["input_ids"], clip.frame_indices, clip.source_fps)
        return Example(inputs, labels, num_target_tokens, intervals)

    # ----------------------------------------------------------------- forward
    def _embed(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Token embeddings with the video and audio features written over their placeholders.

        The model's own `forward` accepts images only; this is the assembly performed by its
        `generate`, kept differentiable.
        """
        model = self.model
        embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
        flat, flat_ids = embeds.reshape(-1, embeds.shape[-1]), inputs["input_ids"].reshape(-1)
        video = model.extract_video_feature(inputs["pixel_values_videos"])  # [tubelets, tokens, d]
        flat = _scatter(flat, flat_ids == IMAGE, video)
        extractor = model.sound_feature_extractor
        features = extractor([inputs["waveform"]], sampling_rate=extractor.sampling_rate, return_tensors="pt")
        features = to_device((features.input_features, features.get("attention_mask")), self.device, torch.bfloat16)
        audio = model.extract_sound_feature(*features)  # [1, tokens, d]
        flat = _scatter(flat, flat_ids == AUDIO, audio)
        return flat.reshape(embeds.shape)

    def _forward(self, inputs: dict[str, Any], layers: Sequence[int], keep: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits of the last `keep` positions and the output states [seq, d] of the one-indexed blocks in `layers`."""
        embeds = self._embed(to_device(inputs, self.device, torch.bfloat16))
        language_model = self.model.language_model
        # Hooks keep only the requested block outputs; `output_hidden_states` would hold all 53 states.
        states: dict[int, torch.Tensor] = {}

        def capture(layer: int) -> Any:
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                states[layer] = output

            return hook

        hooks = [language_model.backbone.layers[layer - 1].register_forward_hook(capture(layer)) for layer in layers]
        try:
            logits = language_model(
                inputs_embeds=embeds,
                attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
                use_cache=False, output_hidden_states=False, logits_to_keep=keep,
            ).logits
        finally:
            for hook in hooks:
                hook.remove()
        assert all(states[layer].shape == embeds.shape for layer in layers)
        return logits, [states[layer][0] for layer in layers]

    def loss(self, example: Example, layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        labels = target_window(example.labels.to(self.device), example.num_target_tokens)
        # Only the scored positions are projected to the vocabulary.
        logits, states = self._forward(example.inputs, [] if layer is None else [layer], keep=labels.shape[-1])
        return answer_loss(logits, labels), states[0] if states else None

    @torch.no_grad()
    def generate(self, video: Path, prompt: str, max_new_tokens: int = 1280) -> str:
        from transformers import GenerationConfig

        inputs, attention_mask = self._encode(self._render(prompt), decode_clip(video))
        input_ids = inputs["input_ids"].to(self.device)
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask.to(self.device),
            pixel_values_videos=inputs["pixel_values_videos"].to(self.device),
            sound_clips=[inputs["waveform"]],
            generation_config=GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                eos_token_id=EOS_TOKEN_IDS, pad_token_id=EOS_TOKEN_IDS[1],
            ),
        )
        return self.tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True)

    @torch.no_grad()
    def states(self, video: Path, duration: float, question: str, layers: Sequence[int]) -> tuple[list[torch.Tensor], Intervals]:
        # Layer selection samples frames on the exact 2 FPS grid: nominally two tubelets per interval.
        clip = decode_clip(video, duration, on_grid=True)
        inputs, _ = self._encode(self._render(question), clip)
        intervals = locate_intervals(inputs["input_ids"], clip.frame_indices, clip.source_fps)
        _, states = self._forward(inputs, layers, keep=1)
        return states, intervals


def _import_from(directory: Path, package: str, module: str) -> Any:
    """Import `module` from a directory of model code, as part of a package so that relative imports resolve."""
    if package not in sys.modules:
        holder = types.ModuleType(package)
        holder.__path__ = [str(directory)]
        sys.modules[package] = holder
    return importlib.import_module(f"{package}.{module}")


def _radio_code() -> Path:
    """Directory with the vision tower's code at the pinned revision; RADIO_CODE_DIR avoids the hub."""
    if os.environ.get("RADIO_CODE_DIR"):
        return Path(os.environ["RADIO_CODE_DIR"])
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(RADIO_REPO, revision=RADIO_REVISION, allow_patterns=["*.py"]))


def _scatter(embeds: torch.Tensor, placeholders: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Write `features` [..., d] over the rows of `embeds` [seq, d] selected by `placeholders`."""
    # masked_scatter would accept surplus features and silently misplace the rest.
    if int(placeholders.sum()) != features.shape[:-1].numel():
        raise ValueError(f"{int(placeholders.sum())} placeholders for media features of shape {tuple(features.shape)}")
    return embeds.masked_scatter(placeholders.unsqueeze(-1), features.reshape(-1, embeds.shape[-1]).to(embeds.dtype))


def answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL over the target window; logit row j scores label j + 1."""
    return F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=IGNORE)


# ----------------------------------------------------------------------- media
@dataclass
class Clip:
    frames: list[Any]  # PIL images
    frame_indices: np.ndarray  # their positions in the source video
    source_fps: float
    duration: float  # of the source video, in seconds
    waveform: np.ndarray  # mono float32 at 16 kHz


def decode_clip(path: Path, max_seconds: float | None = None, on_grid: bool = False) -> Clip:
    """Frames at 2 FPS (see `sample_frames`) and the complete audio track of the first `max_seconds` (default: the whole file)."""
    import av
    import decord
    from PIL import Image

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.guessed_rate or 25)
        total_frames = int(stream.frames or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None and container.duration:
            duration = container.duration / 1e6
        if total_frames <= 0 and duration:
            total_frames = int(math.floor(duration * source_fps))
        if duration is None:
            duration = total_frames / source_fps
        if not container.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        blocks = [
            block.to_ndarray().reshape(-1)
            for frame in container.decode(container.streams.audio[0])
            for block in resampler.resample(frame)
        ]
    waveform = np.concatenate(blocks).astype(np.float32) / 32768.0 if blocks else np.zeros(0, np.float32)
    if max_seconds is not None:
        waveform = waveform[: int(max_seconds * SAMPLE_RATE)]
    if len(waveform) == 0 or audio_token_count(len(waveform)) > MAX_AUDIO_TOKENS:
        raise ValueError(f"{path}: the audio track is empty or exceeds the {MAX_AUDIO_TOKENS} positions of the audio encoder")

    frame_indices = sample_frames(total_frames, source_fps, max_seconds, on_grid)
    reader = decord.VideoReader(str(path), num_threads=2)
    # The two decoders can disagree on the frame count by a frame; repeat the last one if so.
    readable = [min(int(index), len(reader) - 1) for index in frame_indices]
    frames = [Image.fromarray(frame) for frame in reader.get_batch(readable).asnumpy()]
    return Clip(frames, frame_indices, source_fps, duration, waveform)


def sample_frames(total_frames: int, source_fps: float, max_seconds: float | None = None, on_grid: bool = False) -> np.ndarray:
    """Source-frame indices of a 2 FPS sample with an even count of at most 256.

    By default 4 or more frames are spread evenly from the first to the last frame
    of the clip. With `on_grid`, frame i instead sits at exactly i / 2 s, and the
    clip's remainder beyond the last sampled half-second is not covered.
    """
    if max_seconds is not None:
        total_frames = min(total_frames, int(math.floor(max_seconds * source_fps)))
    if total_frames < FRAMES_PER_TUBELET:
        raise RuntimeError(f"video too short: {total_frames} frames")
    if on_grid:
        count = min(int(math.floor(total_frames / source_fps * FPS)), MAX_FRAMES)
        count = int(math.floor(count / FRAMES_PER_TUBELET) * FRAMES_PER_TUBELET)
        if count < FRAMES_PER_TUBELET:
            raise RuntimeError("video shorter than one tubelet at 2 FPS")
        return np.minimum(np.round(np.arange(count) * source_fps / FPS).astype(int), total_frames - 1)
    count = min(max(total_frames / source_fps * FPS, MIN_FRAMES), MAX_FRAMES)
    count = int(math.floor(count / FRAMES_PER_TUBELET) * FRAMES_PER_TUBELET)
    count = max(FRAMES_PER_TUBELET, min(count, total_frames - total_frames % FRAMES_PER_TUBELET))
    return np.linspace(0, total_frames - 1, count).round().astype(int)


def audio_token_count(num_samples: int) -> int:
    """Audio tokens for a 16 kHz waveform: 10 ms feature frames, then three stride-2 convolutions."""
    length = 1 + num_samples // 160
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return max(1, length)


# ------------------------------------------------------------------- intervals
def tubelet_seconds(frame_indices: np.ndarray, source_fps: float) -> list[float]:
    """Timestamp of each tubelet's first frame, as printed by the processor (whole-millisecond frame period)."""
    frame_ms = int(1000.0 / source_fps)
    return [int(index) * frame_ms / 1000.0 for index in frame_indices[::FRAMES_PER_TUBELET]]


def locate_intervals(input_ids: torch.Tensor, frame_indices: np.ndarray, source_fps: float) -> Intervals:
    """The 2 s intervals that hold at least one tubelet and all 25 of their audio tokens."""
    ids = input_ids.reshape(-1)
    tubelets = _placeholder_runs(ids, IMAGE_START, IMAGE_END, IMAGE)
    audio_blocks = _placeholder_runs(ids, AUDIO_START, AUDIO_END, AUDIO)
    starts = tubelet_seconds(frame_indices, source_fps)
    if len(tubelets) != len(starts) or len(audio_blocks) != 1 or tubelets[-1][1] > audio_blocks[0][0]:
        raise ValueError("expected one token group per tubelet followed by a single audio block")
    audio_start, audio_end = audio_blocks[0]
    by_interval: dict[int, list[Span]] = {}
    for span, start in zip(tubelets, starts):
        by_interval.setdefault(int(math.floor(start / SECONDS_PER_INTERVAL)), []).append(span)
    complete = (audio_end - audio_start) // AUDIO_TOKENS_PER_INTERVAL
    keep = [k for k in sorted(by_interval) if k < complete]
    offsets = [audio_start + k * AUDIO_TOKENS_PER_INTERVAL for k in keep]
    return Intervals(
        audio=[[(offset, offset + AUDIO_TOKENS_PER_INTERVAL)] for offset in offsets],
        video=[by_interval[k] for k in keep],
        index=keep,
        seconds=SECONDS_PER_INTERVAL,
    )


def _placeholder_runs(ids: torch.Tensor, start: int, end: int, placeholder: int) -> list[Span]:
    """Spans of the placeholder runs enclosed by each `start` ... `end` pair, every position checked."""
    opens = (ids == start).nonzero(as_tuple=True)[0].tolist()
    closes = (ids == end).nonzero(as_tuple=True)[0].tolist()
    spans = [(opened + 1, closed) for opened, closed in zip(opens, closes)]
    if len(opens) != len(closes) or any(b <= a or bool((ids[a:b] != placeholder).any()) for a, b in spans):
        raise ValueError("malformed media block in input_ids")
    return spans
