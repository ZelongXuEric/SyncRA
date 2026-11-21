"""Auxiliary losses of prior audio-visual synchrony methods, ported to interval-pooled backbone states.

Each port keeps the loss of its source paper and changes only where it is
applied: the same block, intervals and loss weight as SyncRA. Projection sizes
and the CAV-MAE Sync temperature are not given by the sources and follow SyncRA.

Negatives from other videos are the intervals of the clips that the other
training processes hold in the same step, so `korbar` and `cav_mae_sync` need
at least two processes.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .intervals import tokens
from .objectives import DIM, TAU, Clip, Objective, embed

MARGIN = 0.99
HARD_FRACTION = 0.25  # share of same-video negatives once the curriculum admits them
MIN_GAP_SECONDS = 0.5  # minimum offset of a same-video negative
EASY_ONLY = (50, 90)  # fraction of training that uses other-video negatives only
SAMPLING_SEED = 20260718

Features = tuple[torch.Tensor, torch.Tensor]  # unit audio and video features, [K, DIM] each


def build_modality_heads(hidden_size: int) -> nn.ModuleDict:
    """One bias-free projection per modality."""
    return nn.ModuleDict({name: nn.Linear(hidden_size, DIM, bias=False) for name in ("audio", "video")})


def gather_others(z_audio: torch.Tensor, z_video: torch.Tensor, clip: Clip) -> tuple[list[Features], torch.Tensor]:
    """Interval features of the clips held by the other processes, with gradients.

    Clips of the same source video are left out. The gather is a collective
    operation in both directions, so every process must call it in every
    forward pass and add the returned zero-valued `tie` to its loss; otherwise
    a process whose loss ignores the gathered features would skip the backward.
    """
    from torch.distributed.nn.functional import all_gather

    anchor = clip.hidden[0, 0].float() * 0.0  # ties the sent block to the graph even when it is empty
    meta: list = [None] * dist.get_world_size()
    dist.all_gather_object(meta, (len(z_audio), clip.video_id))
    longest = max(1, max(size for size, _ in meta))
    padding = z_audio.new_zeros((longest - len(z_audio), DIM))
    gathered = all_gather(torch.cat([z_audio, padding, z_video, padding]) + anchor)
    tie = torch.stack([block.sum() for block in gathered]).sum() * 0.0
    others = [
        (gathered[rank][:size], gathered[rank][longest:longest + size])
        for rank, (size, video_id) in enumerate(meta)
        if rank != dist.get_rank() and size > 0 and video_id != clip.video_id
    ]
    return others, tie


def _local_features(audio_head: nn.Linear, video_head: nn.Linear, clip: Clip) -> Features:
    """Features of this clip; none if it has fewer than two intervals."""
    if len(clip.intervals) < 2:
        empty = clip.hidden.new_zeros((0, DIM), dtype=torch.float32)
        return empty, empty
    return embed(audio_head, clip.audio), embed(video_head, clip.video)


def korbar(heads: nn.ModuleDict, clip: Clip) -> torch.Tensor:
    """Korbar et al. (2018): margin contrastive loss on distances, one negative per positive pair.

    A negative pair replaces the audio or the video of an interval, either with
    an interval of another video (easy) or with an interval of the same video
    at least 0.5 s away (hard). Following the source curriculum, hard negatives
    enter after the first 50/90 of training, as a quarter of all negatives.
    """
    z_audio, z_video = _local_features(heads["audio"], heads["video"], clip)
    others, tie = gather_others(z_audio, z_video, clip)
    if not len(z_audio):
        return tie
    seed = hashlib.sha256(f"{SAMPLING_SEED}:{clip.video_id}:{clip.step}:{dist.get_rank()}".encode()).hexdigest()
    generator = torch.Generator().manual_seed(int(seed[:15], 16))

    def draw(bound: int) -> int:
        return int(torch.randint(bound, (), generator=generator))

    def chance() -> float:
        return float(torch.rand((), generator=generator))

    hard_fraction = HARD_FRACTION if clip.step * EASY_ONLY[1] >= clip.total_steps * EASY_ONLY[0] else 0.0
    grid = clip.intervals.index
    min_gap = 1 + max(0, math.ceil(MIN_GAP_SECONDS / clip.intervals.seconds - 1e-9))
    negative_audio, negative_video = [], []
    for i in range(len(z_audio)):
        hard = [j for j in range(len(z_audio)) if abs(grid[j] - grid[i]) >= min_gap]
        use_hard = bool(hard) and (not others or chance() < hard_fraction)
        replace_video = chance() < 0.5
        if use_hard:
            j = hard[draw(len(hard))]
            other_audio, other_video = z_audio[j], z_video[j]
        elif others:
            source_audio, source_video = others[draw(len(others))]
            j = draw(len(source_audio))
            other_audio, other_video = source_audio[j], source_video[j]
        else:
            continue
        negative_audio.append(z_audio[i] if replace_video else other_audio)
        negative_video.append(other_video if replace_video else z_video[i])
    terms = [(z_audio - z_video).pow(2).sum(-1)]
    if negative_audio:
        distance = (torch.stack(negative_audio) - torch.stack(negative_video)).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        terms.append((MARGIN - distance).clamp_min(0.0).pow(2))
    return torch.cat(terms).mean() + tie


def cav_mae_sync(projection: nn.Linear, clip: Clip) -> torch.Tensor:
    """CAV-MAE Sync (Araujo et al., 2025): frame-level InfoNCE anchored on the visual side.

    The negatives of a video interval are the audio intervals of other videos
    only; other moments of the same video are never contrasted.
    """
    z_audio, z_video = _local_features(projection, projection, clip)
    others, tie = gather_others(z_audio, z_video, clip)
    if not len(z_audio) or not others:
        return z_audio.sum() * 0.0 + tie
    negatives = torch.cat([other_audio for other_audio, _ in others])
    positive = (z_video * z_audio).sum(-1, keepdim=True)
    logits = torch.cat([positive, z_video @ negatives.T], dim=1) / TAU
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device)) + tie


def avaf_net(heads: nn.ModuleDict, clip: Clip) -> torch.Tensor:
    """AVAF-Net (Zhao et al., 2025): temporal alignment by matching similarity structure within a video.

    The audio-video similarity matrix over intervals is matched (L1) to the
    audio-audio and video-video ones. A third term matches audio-audio
    similarity to that of audio-attended video features, where each interval's
    video tokens are pooled with attention from its audio feature.
    """
    f_audio, f_video = embed(heads["audio"], clip.audio), embed(heads["video"], clip.video)
    dtype = heads["video"].weight.dtype
    raw_audio = heads["audio"](clip.audio.to(dtype)).float()
    attended = []
    for k, spans in enumerate(clip.intervals.video):
        patches = heads["video"](tokens(clip.hidden, spans).to(dtype)).float()
        attention = torch.softmax(patches @ raw_audio[k] / math.sqrt(DIM), dim=0)
        attended.append(attention @ patches)
    f_attended = F.normalize(torch.stack(attended), dim=-1, eps=1e-8)
    s_av, s_aa, s_vv, s_ss = f_audio @ f_video.T, f_audio @ f_audio.T, f_video @ f_video.T, f_attended @ f_attended.T
    return (s_av - s_aa).abs().mean() + (s_av - s_vv).abs().mean() + (s_aa - s_ss).abs().mean()


OBJECTIVES = {
    "korbar": Objective(korbar, build_modality_heads, cross_clip=True),
    "cav_mae_sync": Objective(cav_mae_sync, cross_clip=True),
    "avaf_net": Objective(avaf_net, build_modality_heads),
}
