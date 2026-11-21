"""Qwen2.5-Omni-7B (dense) and Qwen3-Omni-30B-A3B (MoE) Thinkers.

Both processors interleave audio and video tokens along a shared time index, so
interval membership is recovered by replaying that rule and checking the result
against the placeholder tokens actually present in `input_ids`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..intervals import Intervals, Span
from .base import Backbone, Example, count_target_tokens, mask_prompt, target_window, to_device

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech."
)
VIDEO = {
    "fps": 2.0,
    "min_frames": 4,
    "max_frames": 256,
    "min_pixels": 100352,
    "max_pixels": 602112,
    "total_pixels": 19267584,
}
SPATIAL_MERGE = 2

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")  # the video reader used for all reported runs


class QwenOmni(Backbone):
    """Shared input assembly, target-only loss and generation for the Qwen Thinkers."""

    vision_kwargs: dict[str, Any] = {}

    def __init__(self, model_path: str | Path, device: torch.device, train: bool = False):
        self.device = device
        self.processor, self.model = self._load(Path(model_path))
        tokenizer, processor = self.processor.tokenizer, self.processor
        self.token_ids = {
            name: tokenizer.convert_tokens_to_ids(getattr(processor, f"{name}_token"))
            for name in ("audio", "video", "vision_bos", "vision_eos", "audio_bos", "audio_eos")
        }
        # Project only the positions that are scored; full-sequence logits do not fit in memory.
        self._keep: int | None = None
        self.model.lm_head.register_forward_pre_hook(self._keep_scored_positions)
        if train:
            self.model.config.use_cache = False
            self._set_trainable()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()
        else:
            self.model.requires_grad_(False).eval()

    def _load(self, model_path: Path) -> tuple[Any, torch.nn.Module]:
        raise NotImplementedError

    def _set_trainable(self) -> None:
        """Dense backbones train the whole Thinker."""

    def _intervals(self, inputs: Any, fps: float) -> Intervals:
        raise NotImplementedError

    def _generation_kwargs(self) -> dict[str, Any]:
        return {}

    def _processor_kwargs(self) -> dict[str, Any]:
        """Extra processor arguments; built per call because the processor consumes nested dicts."""
        return {}

    def _keep_scored_positions(self, _module: torch.nn.Module, args: tuple[torch.Tensor]) -> Any:
        return None if self._keep is None else (args[0][:, -self._keep:, :],)

    # ------------------------------------------------------------------ inputs
    def _conversation(self, video: Path, text: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video", "video": str(video), **VIDEO}, {"type": "text", "text": text}]},
        ]

    def _render(self, conversation: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt, tokenize=False)
        return text[0] if isinstance(text, list) else text

    def _encode(self, conversation: list[dict[str, Any]], text: str) -> tuple[Any, float]:
        """Decode media at the fixed sampling policy and run the processor on `text`."""
        from qwen_omni_utils import process_audio_info, process_vision_info

        images, videos, video_kwargs = process_vision_info(conversation, return_video_kwargs=True, **self.vision_kwargs)
        fps = float(np.asarray(video_kwargs.pop("fps")).reshape(-1)[0])
        audios = process_audio_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
            use_audio_in_video=True, fps=fps, **video_kwargs, **self._processor_kwargs(),
        )
        # 16 kHz audio at a 10 ms hop: fewer feature frames means the extractor truncated the track.
        if abs(int(inputs["feature_attention_mask"].sum()) - len(audios[0]) // 160) > 100:
            raise RuntimeError("the feature extractor truncated the audio track")
        return inputs, fps

    def prepare(self, video: Path, duration: float, question: str, target: str) -> Example:
        conversation = self._conversation(video, question)
        answer_turn = {"role": "assistant", "content": [{"type": "text", "text": target}]}
        prompt = self._render(conversation, add_generation_prompt=True)
        full = self._render(conversation + [answer_turn], add_generation_prompt=False)
        num_target_tokens, text_ids = count_target_tokens(self.processor.tokenizer, prompt, full)
        inputs, fps = self._encode(conversation, full)
        labels = mask_prompt(inputs["input_ids"], num_target_tokens, text_ids)
        return Example(dict(inputs), labels, num_target_tokens, self._intervals(inputs, fps))

    # ----------------------------------------------------------------- forward
    def loss(self, example: Example, layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = to_device(example.inputs, self.device, torch.bfloat16)
        labels = target_window(example.labels.to(self.device), example.num_target_tokens)
        self._keep = labels.shape[-1]
        try:
            outputs = self.model(
                **inputs, labels=labels, use_audio_in_video=True, use_cache=False,
                output_hidden_states=layer is not None,
            )
        finally:
            self._keep = None
        if layer is None:
            return outputs.loss, None
        assert len(outputs.hidden_states) == self.num_blocks + 1
        return outputs.loss, outputs.hidden_states[layer][0]

    @torch.no_grad()
    def generate(self, video: Path, prompt: str, max_new_tokens: int = 1280) -> str:
        conversation = self._conversation(video, prompt)
        inputs, _ = self._encode(conversation, self._render(conversation, add_generation_prompt=True))
        inputs = to_device(dict(inputs), self.device, torch.bfloat16)
        generated = self.model.generate(
            **inputs, use_audio_in_video=True, do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, **self._generation_kwargs(),
        )
        new_tokens = generated[0, inputs["input_ids"].shape[-1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def states(self, video: Path, duration: float, question: str, layers: Sequence[int]) -> tuple[list[torch.Tensor], Intervals]:
        conversation = self._conversation(video, question)
        inputs, fps = self._encode(conversation, self._render(conversation, add_generation_prompt=True))
        intervals = self._intervals(inputs, fps)
        inputs = to_device(dict(inputs), self.device, torch.bfloat16)
        outputs = self.model(**inputs, use_audio_in_video=True, use_cache=False, output_hidden_states=True)
        return [outputs.hidden_states[layer][0] for layer in layers], intervals

    def _locate(self, input_ids: torch.Tensor, counts: list[tuple[int, int]], trailing_audio: int = 0) -> tuple[list[Span], list[Span]]:
        """Spans of each interval's video run and audio run inside the interleaved media block.

        `counts[k]` is the (video, audio) token count of interval k. Every position
        is checked against the placeholder id, so a wrong layout cannot pass silently.
        """
        ids = input_ids[0].tolist()
        tokens = self.token_ids
        (start,) = [i for i, token in enumerate(ids) if token == tokens["vision_bos"]]
        if ids[start + 1] != tokens["audio_bos"]:
            raise ValueError("expected an interleaved audio-video block")
        cursor = start + 2
        video_spans, audio_spans = [], []
        for num_video, num_audio in counts:
            for spans, count, token in ((video_spans, num_video, tokens["video"]), (audio_spans, num_audio, tokens["audio"])):
                if any(i != token for i in ids[cursor:cursor + count]):
                    raise ValueError(f"interval layout disagrees with input_ids near position {cursor}")
                spans.append((cursor, cursor + count))
                cursor += count
        cursor += trailing_audio
        if ids[cursor:cursor + 2] != [tokens["audio_eos"], tokens["vision_eos"]]:
            raise ValueError("interval layout does not span the whole audio-video block")
        return video_spans, audio_spans


def _valid(video_spans: list[Span], audio_spans: list[Span], seconds_per_interval: float) -> Intervals:
    keep = [k for k, (v, a) in enumerate(zip(video_spans, audio_spans)) if v[1] > v[0] and a[1] > a[0]]
    return Intervals(
        audio=[[audio_spans[k]] for k in keep],
        video=[[video_spans[k]] for k in keep],
        index=keep,
        seconds=seconds_per_interval,
    )


class Qwen25Omni(QwenOmni):
    """Qwen2.5-Omni-7B: full Thinker fine-tuning; native 2 s intervals."""

    repo = "Qwen/Qwen2.5-Omni-7B"
    revision = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
    hidden_size, num_blocks, layer = 3584, 28, 21

    POSITIONS_PER_SECOND = 25  # time-index units per second; also the audio token rate
    SECONDS_PER_INTERVAL = 2.0

    def _load(self, model_path: Path) -> tuple[Any, torch.nn.Module]:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        return processor, model.to(self.device)

    def _generation_kwargs(self) -> dict[str, Any]:
        return {"eos_token_id": self.model.generation_config.eos_token_id or self.model.config.eos_token_id}

    def _intervals(self, inputs: Any, fps: float) -> Intervals:
        grid_t, grid_h, grid_w = inputs["video_grid_thw"][0].tolist()
        tokens_per_grid = (grid_h // SPATIAL_MERGE) * (grid_w // SPATIAL_MERGE)
        seconds_per_grid = self.processor.video_processor.temporal_patch_size / fps
        video_times = [t * seconds_per_grid * self.POSITIONS_PER_SECOND for t in range(grid_t) for _ in range(tokens_per_grid)]
        frames = int(inputs["feature_attention_mask"].sum())
        num_audio = (((frames - 1) // 2 + 1) - 2) // 2 + 1
        per_interval = int(self.POSITIONS_PER_SECOND * self.SECONDS_PER_INTERVAL)
        video_chunks = _chunk_sizes(video_times, per_interval)
        audio_chunks = _chunk_sizes(list(range(num_audio)), per_interval)
        num_intervals = max(len(video_chunks), len(audio_chunks))
        counts = [
            (video_chunks[k] if k < len(video_chunks) else 0, audio_chunks[k] if k < len(audio_chunks) else 0)
            for k in range(num_intervals)
        ]
        return _valid(*self._locate(inputs["input_ids"], counts), self.SECONDS_PER_INTERVAL)


def _chunk_sizes(time_index: list[float], per_chunk: int) -> list[int]:
    """Sizes of the processor's chunks: a new chunk opens when the time index reaches the next multiple."""
    sizes, start, chunk = [], 0, 1
    for i, value in enumerate(time_index):
        if value >= chunk * per_chunk:
            sizes.append(i - start)
            start, chunk = i, chunk + 1
    sizes.append(len(time_index) - start)
    return sizes


class Qwen3Omni(QwenOmni):
    """Qwen3-Omni-30B-A3B: routed experts and router gates frozen; one interval per temporal grid."""

    repo = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    revision = "26291f793822fb6be9555850f06dfe95f2d7e695"
    hidden_size, num_blocks, layer = 2048, 48, 36

    POSITIONS_PER_SECOND = 13
    TEMPORAL_PATCH = 2
    TRAINABLE_PARAMETERS = 2_715_593_328
    EOS_TOKEN_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>

    vision_kwargs = {"image_patch_size": 16}

    def _load(self, model_path: Path) -> tuple[Any, torch.nn.Module]:
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        model, info = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map={"": self.device.index or 0}, output_loading_info=True,
        )
        if info["missing_keys"] or info["mismatched_keys"]:
            raise RuntimeError("the checkpoint does not cover the Thinker")
        return processor, model

    def _set_trainable(self) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(".mlp.experts." not in name and ".mlp.gate." not in name)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable != self.TRAINABLE_PARAMETERS:
            raise RuntimeError(f"unexpected trainable parameter count {trainable:,}")

    def _generation_kwargs(self) -> dict[str, Any]:
        return {"eos_token_id": self.EOS_TOKEN_IDS, "pad_token_id": self.EOS_TOKEN_IDS[1], "use_cache": True}

    def _processor_kwargs(self) -> dict[str, Any]:
        # The feature extractor would otherwise cut every audio track at 30 s.
        return {"audio_kwargs": {"truncation": False}}

    def _intervals(self, inputs: Any, fps: float) -> Intervals:
        grid_t, grid_h, grid_w = inputs["video_grid_thw"][0].tolist()
        tokens_per_grid = (grid_h // SPATIAL_MERGE) * (grid_w // SPATIAL_MERGE)
        # float64 on purpose: the processor merges on this value, not on its float32 `video_second_per_grid`.
        seconds_per_grid = self.TEMPORAL_PATCH / fps
        frames = int(inputs["feature_attention_mask"].sum())
        num_audio = ((((frames % 100 - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13
        # Audio token j follows the video tokens of grid t while grid_time[t] <= j < grid_time[t + 1].
        video_times = np.repeat(np.arange(grid_t), tokens_per_grid) * seconds_per_grid * self.POSITIONS_PER_SECOND
        grid_time = np.append(video_times[::tokens_per_grid], np.float64(grid_t) * seconds_per_grid * self.POSITIONS_PER_SECOND)
        bounds = np.searchsorted(np.arange(num_audio), grid_time, side="left")
        counts = [(tokens_per_grid, int(bounds[t + 1] - bounds[t])) for t in range(grid_t)]
        trailing = num_audio - int(bounds[-1])
        return _valid(*self._locate(inputs["input_ids"], counts, trailing), seconds_per_grid)
