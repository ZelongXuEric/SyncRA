"""Choose the supervised block of a backbone by probing its frozen states.

    python -m syncra.select_layer --backbone qwen3-omni --model-path /path/to/model \\
        --media-root /path/to/videos --output-dir runs/select_qwen3

For each source video, the first question defines one prompt-only forward
pass, and the interval-pooled audio and video states of every candidate block
are cached. A 64-dimensional shared projection is then fitted per candidate
with the SyncRA loss alone, and candidates are compared by same-interval
retrieval on held-out videos. Nothing in the backbone is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import backbones, data
from .backbones.base import Backbone
from .intervals import pool
from .objectives import build_projection, embed, infonce

Pools = tuple[torch.Tensor, torch.Tensor]  # interval-pooled audio and video states of one video, [K, d] each


def first_questions(rows: list[data.Row]) -> list[data.Row]:
    """The first question of every source video, in file order."""
    seen: set[str] = set()
    return [row for row in rows if not (row["video_id"] in seen or seen.add(row["video_id"]))]


def cache_states(backbone: Backbone, rows: list[data.Row], media_root: str, layers: list[int], max_seconds: float | None) -> list[list[Pools]]:
    """Pooled states of every usable video at every candidate block: result[video][candidate]."""
    cached = []
    for row in rows:
        duration = row["duration"] if max_seconds is None else min(row["duration"], max_seconds)
        try:
            states, intervals = backbone.states(data.media_path(media_root, row), duration, data.question_text(row), layers)
        except ValueError as error:  # the processor produced a layout without a recoverable time grid
            print(f"skipping {row['video_id']}: {error}", flush=True)
            continue
        if len(intervals) >= 2:
            cached.append([tuple(pooled.float().cpu() for pooled in pool(hidden, intervals)) for hidden in states])
    return cached


def fit_probe(videos: list[Pools], seed: int, epochs: int = 50) -> torch.nn.Linear:
    """Fit a shared projection with the within-video InfoNCE, one video per update in fixed order."""
    torch.manual_seed(seed)
    projection = build_projection(videos[0][0].shape[-1])
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
    for _ in range(epochs):
        for audio, video in videos:
            loss = infonce(embed(projection, audio), embed(projection, video), list(range(len(audio))))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return projection


@torch.no_grad()
def retrieval(projection: torch.nn.Linear, videos: list[Pools]) -> float:
    """Same-interval R@1 pooled over both directions and all queries of all videos."""
    hits = queries = 0
    for audio, video in videos:
        similarity = embed(projection, audio) @ embed(projection, video).T
        target = torch.arange(len(audio))
        hits += int((similarity.argmax(dim=1) == target).sum()) + int((similarity.argmax(dim=0) == target).sum())
        queries += 2 * len(audio)
    return hits / queries


def select(scores: dict[int, float], chance: float) -> int | None:
    """The best block with signal clearly above chance and headroom left to learn; None if there is none."""
    eligible = [layer for layer, score in scores.items() if chance + 0.05 <= score < 0.90]
    return max(eligible, key=scores.get) if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=sorted(backbones.REGISTRY))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", help="probe a fine-tuned checkpoint instead of the released weights")
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-file", default="data/omnivideo_10k/train.jsonl")
    parser.add_argument("--validation-file", default="data/omnivideo_10k/validation.jsonl")
    parser.add_argument("--layers", type=int, nargs="+", help="candidate blocks (default: quarter, half and three-quarter depth)")
    parser.add_argument("--fit-videos", type=int, default=200, help="training source videos used to fit the probes")
    parser.add_argument("--max-seconds", type=float, help="probe only the start of each clip")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "states.pt"
    if cache.exists():
        layers, fit, validation = torch.load(cache)
    else:
        backbone = backbones.load(args.backbone, args.model_path, torch.device("cuda", 0))
        if args.checkpoint:
            backbone.load_trainable_state(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        depth = backbone.num_blocks
        layers = args.layers or [depth // 4, depth // 2, 3 * depth // 4]
        fit_rows = first_questions(data.load_rows(args.train_file))[:args.fit_videos]
        validation_rows = first_questions(data.load_rows(args.validation_file))
        fit = cache_states(backbone, fit_rows, args.media_root, layers, args.max_seconds)
        validation = cache_states(backbone, validation_rows, args.media_root, layers, args.max_seconds)
        torch.save((layers, fit, validation), cache)

    # Expected R@1 of a random guess: one hit per K candidates for each of the 2K queries of a video.
    chance = len(validation) / sum(len(video[0][0]) for video in validation)
    scores = {}
    for index, layer in enumerate(layers):
        projection = fit_probe([video[index] for video in fit], args.seed)
        scores[layer] = retrieval(projection, [video[index] for video in validation])
        print(f"block {layer}: R@1 {100 * scores[layer]:.2f}%", flush=True)
    selected = select(scores, chance)
    print(f"chance {100 * chance:.2f}% | selected block: {selected}")
    summary = {"scores": scores, "chance": chance, "selected": selected, "fit_videos": len(fit), "validation_videos": len(validation)}
    (args.output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
