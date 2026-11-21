"""Backbone interface shared by the trainer and the swap test."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..intervals import Intervals

IGNORE = -100


@dataclass
class Example:
    """One prepared training example (batch size 1, CPU tensors)."""

    inputs: dict[str, Any]
    labels: torch.Tensor  # [1, seq]; IGNORE everywhere except the assistant target
    num_target_tokens: int
    intervals: Intervals


class Backbone(ABC):
    """An omni-modal answering model with a native audio-video time grid.

    Subclasses load the released checkpoint, fix the trainable parameter scope,
    and expose three operations: build a teacher-forced example, compute its
    answer loss (optionally returning one block's hidden states), and generate.
    """

    repo: str  # Hugging Face identifier and the revision used in the paper
    revision: str
    hidden_size: int
    num_blocks: int
    layer: int  # default supervision block, one-indexed

    model: nn.Module
    device: torch.device

    @abstractmethod
    def prepare(self, video: Path, duration: float, question: str, target: str) -> Example:
        """Decode the clip and build inputs, labels and interval layout (CPU only)."""

    @abstractmethod
    def loss(self, example: Example, layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mean answer-token NLL and, if requested, the output states [seq, d] of `layer`."""

    @abstractmethod
    def generate(self, video: Path, prompt: str, max_new_tokens: int = 1280) -> str:
        """Greedy answer to `prompt` about the clip."""

    @abstractmethod
    def states(self, video: Path, duration: float, question: str, layers: Sequence[int]) -> tuple[list[torch.Tensor], Intervals]:
        """Prompt-only forward pass: output states [seq, d] of each block in `layers`, and the interval layout."""

    def set_training(self, mode: bool) -> None:
        self.model.train(mode)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Checkpoints hold trainable parameters only; the rest come from the released model."""
        names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        return {name: value for name, value in self.model.state_dict().items() if name in names}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        unexpected = self.model.load_state_dict(state, strict=False).unexpected_keys
        if unexpected:
            raise KeyError(f"checkpoint tensors not in the model: {unexpected[:5]}")


def count_target_tokens(tokenizer: Any, prompt: str, full: str) -> tuple[int, list[int]]:
    """Number of trailing tokens of `full` that belong to the assistant target.

    `full` must extend `prompt`. A token merging the last prompt character with
    the first target character counts as target. Media placeholders expand
    before the assistant turn, so the count carries over to the expanded sequence.
    """
    if not full.startswith(prompt):
        raise ValueError("the chat template does not render the target as a suffix of the prompt")
    encoded = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
    first = next((i for i, (start, end) in enumerate(offsets) if end > len(prompt) and end > start), None)
    if first is None:
        raise ValueError("the assistant target is empty")
    return len(ids) - first, ids


def mask_prompt(input_ids: torch.Tensor, num_target_tokens: int, text_ids: list[int]) -> torch.Tensor:
    """Labels that supervise only the last `num_target_tokens` positions."""
    if input_ids[0, -num_target_tokens:].tolist() != text_ids[-num_target_tokens:]:
        raise ValueError("media expansion changed the tokens of the assistant target")
    labels = input_ids.clone().long()
    labels[:, :-num_target_tokens] = IGNORE
    return labels


def target_window(labels: torch.Tensor, num_target_tokens: int) -> torch.Tensor:
    """The target labels preceded by one masked position.

    Logits are needed only where they are scored: position j predicts label j+1,
    so the last `num_target_tokens + 1` positions suffice.
    """
    window = labels[:, -(num_target_tokens + 1):]
    if window[0, 0] != IGNORE or int((window != IGNORE).sum()) != num_target_tokens:
        raise ValueError("labels do not end with exactly the assistant target")
    return window


def to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    """Move nested tensors to `device`, casting floating-point ones to `dtype`."""
    if torch.is_tensor(value):
        value = value.to(device, non_blocking=True)
        return value.to(dtype) if dtype is not None and value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device, dtype) for item in value)
    return value
