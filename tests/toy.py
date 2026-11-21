"""A tiny CPU backbone with a synthetic audio-video grid, for testing the trainer end to end."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from syncra import backbones
from syncra.backbones.base import IGNORE, Backbone, Example, target_window
from syncra.intervals import Intervals

VOCAB, MEDIA_DIM = 64, 8
VIDEO_TOKENS, AUDIO_TOKENS = 3, 2


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, hidden_size)
        self.media = nn.Linear(MEDIA_DIM, hidden_size)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) for _ in range(num_blocks))
        self.lm_head = nn.Linear(hidden_size, VOCAB)

    def forward(self, tokens: torch.Tensor, features: torch.Tensor, is_media: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states = self.embed(tokens).clone()
        states[is_media] = self.media(features)
        steps = torch.arange(1, len(states) + 1, dtype=states.dtype)[:, None]
        hidden = []
        for block in self.blocks:
            states = states + block(states.cumsum(0) / steps)  # causal mixing
            hidden.append(states)
        return states, hidden


class ToyBackbone(Backbone):
    repo, revision = "toy", "0"
    hidden_size, num_blocks, layer = 32, 4, 3

    def __init__(self, model_path, device, train=False):
        self.device = device
        torch.manual_seed(1234)
        self.model = ToyModel(self.hidden_size, self.num_blocks)
        self.model.blocks[0].requires_grad_(False)  # a frozen subset, as on the MoE backbones

    def prepare(self, video: Path, duration: float, question: str, target: str) -> Example:
        generator = torch.Generator().manual_seed(_seed(video.stem))
        num_intervals = 1 + _seed(video.stem) % 6  # some clips have a single interval
        per_interval = VIDEO_TOKENS + AUDIO_TOKENS
        features = torch.randn(num_intervals * per_interval, MEDIA_DIM, generator=generator)
        text = [_seed(word) % VOCAB for word in question.split()[:6]]
        answer = [_seed(word) % VOCAB for word in target.split()[:4]]
        media_end = 1 + num_intervals * per_interval
        tokens = torch.tensor([1] + [0] * (media_end - 1) + text + answer)
        is_media = torch.zeros(len(tokens), dtype=torch.bool)
        is_media[1:media_end] = True
        labels = tokens.clone()[None]
        labels[:, :-len(answer)] = IGNORE
        starts = [1 + k * per_interval for k in range(num_intervals)]
        intervals = Intervals(
            video=[[(s, s + VIDEO_TOKENS)] for s in starts],
            audio=[[(s + VIDEO_TOKENS, s + per_interval)] for s in starts],
            index=list(range(num_intervals)), seconds=2.0,
        )
        return Example({"tokens": tokens, "features": features, "is_media": is_media}, labels, len(answer), intervals)

    def loss(self, example: Example, layer: int | None = None):
        states, hidden = self.model(**example.inputs)
        window = target_window(example.labels, example.num_target_tokens)
        logits = self.model.lm_head(states[-window.shape[-1]:-1])
        loss = nn.functional.cross_entropy(logits, window[0, 1:])
        return loss, None if layer is None else hidden[layer - 1]

    def generate(self, video, prompt, max_new_tokens=1280):
        return "A"

    @torch.no_grad()
    def states(self, video, duration, question, layers):
        example = self.prepare(video, duration, question, "none")
        _, hidden = self.model(**example.inputs)
        return [hidden[layer - 1] for layer in layers], example.intervals


def write_corpus(root: Path, train: int = 24, validation: int = 6) -> None:
    """A synthetic corpus in the released format, with empty placeholder media files."""
    (root / "media").mkdir(parents=True, exist_ok=True)
    for name, count, offset in (("train", train, 0), ("validation", validation, 1000)):
        with open(root / f"{name}.jsonl", "w") as handle:
            for i in range(count):
                pair = offset + i // 2  # rows 2p and 2p+1 are held by different processes in the same step
                video_id = f"video{pair}" if pair % 3 == 2 else f"video{pair}{'ab'[i % 2]}"
                (root / "media" / f"{video_id}.mp4").touch()
                row = {"id": f"q{offset + i}", "video_id": video_id, "duration": 60.0, "task": "toy",
                       "question": f"what happens in clip {i} ?", "options": ["red", "blue"] if i % 2 else None,
                       "answer": "A" if i % 2 else f"answer number {i}", "reasoning": f"because of {i}"}
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    # Entry point for the end-to-end tests: the regular trainer on the toy backbone.
    from syncra import train

    backbones.REGISTRY["toy"] = ("toy", "ToyBackbone")
    backbones.load = lambda name, path, device, train=False: ToyBackbone(path, device, train)
    train.main()
