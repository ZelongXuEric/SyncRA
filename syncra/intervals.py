"""Native time intervals of a clip and interval-level pooling of hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Span = tuple[int, int]  # half-open range of sequence positions


@dataclass(frozen=True)
class Intervals:
    """The valid intervals of one clip, i.e. those holding both audio and video tokens.

    Intervals are the processor's own sampling units, located from token timing
    metadata. A modality may occupy several disjoint spans within one interval.
    """

    audio: list[list[Span]]
    video: list[list[Span]]
    index: list[int]  # position of each valid interval on the clip's full grid
    seconds: float  # interval length

    def __len__(self) -> int:
        return len(self.index)


def tokens(hidden: torch.Tensor, spans: list[Span]) -> torch.Tensor:
    """Token states [n, d] covered by `spans`."""
    if len(spans) == 1:
        start, end = spans[0]
        return hidden[start:end]
    return torch.cat([hidden[start:end] for start, end in spans], dim=0)


def pool(hidden: torch.Tensor, intervals: Intervals) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool token states [seq, d] per interval and modality: (audio [K, d], video [K, d])."""
    if not len(intervals):
        empty = hidden.new_empty((0, hidden.shape[-1]))
        return empty, empty
    audio = torch.stack([tokens(hidden, spans).mean(dim=0) for spans in intervals.audio])
    video = torch.stack([tokens(hidden, spans).mean(dim=0) for spans in intervals.video])
    return audio, video
