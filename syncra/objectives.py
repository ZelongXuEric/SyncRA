"""Auxiliary objectives on interval-pooled audio and video states.

An objective sees one clip: its audio and video states pooled per native
interval at the supervised block. It maps them through a bias-free projection
to unit vectors and returns a scalar float32 loss. SyncRA classifies the
co-occurring interval among all intervals of the same clip, in both
directions. The other objectives here are the controls and loss variants it is
compared against. Ports of prior methods live in `prior_objectives`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .intervals import Intervals

DIM = 64
TAU = 0.07
DERANGEMENT_SEED = 20260718


@dataclass
class Clip:
    """One training example as seen by an auxiliary objective."""

    audio: torch.Tensor  # [K, d] interval-pooled states of the supervised block
    video: torch.Tensor
    intervals: Intervals
    video_id: str
    hidden: torch.Tensor  # [seq, d] token states the pools were computed from
    step: int = 0  # optimizer steps completed so far
    total_steps: int = 1


def build_projection(hidden_size: int) -> nn.Linear:
    """The training-only head W shared by both modalities; it is discarded after training."""
    return nn.Linear(hidden_size, DIM, bias=False)


@dataclass(frozen=True)
class Objective:
    loss: Callable[[nn.Module, Clip], torch.Tensor]
    build_head: Callable[[int], nn.Module] = build_projection
    cross_clip: bool = False  # compares the clips held by different training processes


def embed(projection: nn.Linear, states: torch.Tensor) -> torch.Tensor:
    """Project in the head's dtype, then normalize in float32."""
    return F.normalize(projection(states.to(projection.weight.dtype)).float(), dim=-1, eps=1e-8)


def infonce(z_audio: torch.Tensor, z_video: torch.Tensor, positive: list[int]) -> torch.Tensor:
    """Symmetric InfoNCE where video interval `positive[k]` is the positive of audio interval k."""
    logits = z_audio @ z_video.T / TAU
    audio_to_video = torch.as_tensor(positive, device=logits.device)
    video_to_audio = torch.empty_like(audio_to_video)
    video_to_audio[audio_to_video] = torch.arange(len(positive), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, audio_to_video) + F.cross_entropy(logits.T, video_to_audio))


def syncra(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Within-video synchrony contrast: same-interval pairs against the clip's other intervals."""
    return infonce(embed(projection, clip.audio), embed(projection, clip.video), list(range(len(clip.audio))))


def derangement(video_id: str, size: int) -> list[int]:
    """A fixed-point-free permutation of `size` intervals, determined by the source video."""
    rng = random.Random(f"{DERANGEMENT_SEED}:{video_id}")
    permutation = list(range(size))
    for _ in range(10000):
        rng.shuffle(permutation)
        if all(i != p for i, p in enumerate(permutation)):
            return permutation
    raise RuntimeError(f"no derangement found for {video_id}")


def permuted_pairs(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Control: the same contrast with every positive assigned to a wrong interval."""
    positive = derangement(clip.video_id, len(clip.audio))
    return infonce(embed(projection, clip.audio), embed(projection, clip.video), positive)


def time_codes(seconds: torch.Tensor) -> torch.Tensor:
    """Fixed unit-norm sinusoidal codes with 32 log-spaced periods from 4 to 512 seconds."""
    periods = torch.logspace(math.log10(4.0), math.log10(512.0), DIM // 2, device=seconds.device, dtype=torch.float32)
    phases = 2 * math.pi * seconds[:, None] / periods[None]
    return torch.cat((phases.sin(), phases.cos()), dim=1) / math.sqrt(DIM // 2)


def fixed_time_codes(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Control: each modality separately classifies the time code of its interval.

    Both modalities share a temporal target but are never compared with each other.
    """
    intervals = clip.intervals
    seconds = torch.as_tensor(intervals.index, device=clip.audio.device, dtype=torch.float32) * intervals.seconds
    codes = time_codes(seconds)
    targets = torch.arange(len(seconds), device=clip.audio.device)
    z_audio, z_video = embed(projection, clip.audio), embed(projection, clip.video)
    return 0.5 * (F.cross_entropy(z_audio @ codes.T / TAU, targets) + F.cross_entropy(z_video @ codes.T / TAU, targets))


def clip_contrast(z_audio: torch.Tensor, z_video: torch.Tensor, sources: list[str]) -> torch.Tensor:
    """Symmetric clip-level contrast; clips of the same source video are all positives."""
    logits = z_audio @ z_video.T / TAU
    same = torch.tensor([[a == b for b in sources] for a in sources], device=logits.device)
    matched = logits.masked_fill(~same, -torch.inf)
    return 0.5 * (
        (torch.logsumexp(logits, dim=1) - torch.logsumexp(matched, dim=1)).mean()
        + (torch.logsumexp(logits, dim=0) - torch.logsumexp(matched, dim=0)).mean()
    )


def clip_level_av(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Control: global audio-video association across the clips held by the training processes.

    Every process evaluates the same loss. The differentiable gather sums the
    per-process contributions and DDP's gradient averaging divides them out again.
    """
    import torch.distributed as dist
    from torch.distributed.nn.functional import all_gather

    local = embed(projection, torch.stack((clip.audio.mean(0), clip.video.mean(0))))
    gathered = all_gather(local)
    sources: list = [None] * dist.get_world_size()
    dist.all_gather_object(sources, clip.video_id)
    z_audio = torch.stack([other[0] for other in gathered])
    z_video = torch.stack([other[1] for other in gathered])
    return clip_contrast(z_audio, z_video, sources)


def cosine_only(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Loss variant: attract same-interval pairs with no competing intervals.

    The 1/tau factor matches the positive-pair term of InfoNCE.
    """
    z_audio, z_video = embed(projection, clip.audio), embed(projection, clip.video)
    return (1.0 - (z_audio * z_video).sum(dim=-1)).mean() / TAU


def pairwise_ranking(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """Loss variant: rank each positive above every other interval independently."""
    logits = embed(projection, clip.audio) @ embed(projection, clip.video).T / TAU
    positive = logits.diagonal()
    negatives = ~torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    by_audio = F.softplus((logits - positive[:, None])[negatives]).mean()
    by_video = F.softplus((logits - positive[None, :])[negatives]).mean()
    return 0.5 * (by_audio + by_video)


OBJECTIVES = {
    "syncra": Objective(syncra),
    "permuted_pairs": Objective(permuted_pairs),
    "fixed_time_codes": Objective(fixed_time_codes),
    "clip_level_av": Objective(clip_level_av, cross_clip=True),
    "cosine_only": Objective(cosine_only),
    "pairwise_ranking": Objective(pairwise_ranking),
}
