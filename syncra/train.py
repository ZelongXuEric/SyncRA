"""QA fine-tuning with an optional auxiliary objective on intermediate states.

Launch with torchrun; the paper uses two processes:

    torchrun --nproc_per_node 2 -m syncra.train \\
        --backbone qwen3-omni --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \\
        --media-root /path/to/videos --objective syncra --output-dir runs/qwen3_syncra

Examples are visited in file order. Each optimizer step consumes the next
`--batch-size` rows, split evenly across processes that accumulate gradients one
example at a time. `--objective none` is answer-only fine-tuning (Vanilla SFT).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from . import backbones, data, objectives, prior_objectives
from .backbones.base import Backbone, Example
from .intervals import pool
from .objectives import Clip, Objective

OBJECTIVES = {**objectives.OBJECTIVES, **prior_objectives.OBJECTIVES}


class TrainModule(nn.Module):
    """Backbone plus the auxiliary projection, so both live in one DDP graph."""

    def __init__(self, backbone: Backbone, objective: Objective | None, layer: int, total_steps: int):
        super().__init__()
        self.backbone = backbone
        self.model = backbone.model
        self.objective = objective
        self.layer = layer
        self.step, self.total_steps = 0, total_steps  # training progress, for objectives with a curriculum
        if objective is not None:
            parameter = next(backbone.model.parameters())
            self.projection = objective.build_head(backbone.hidden_size).to(parameter.device, parameter.dtype)

    def train(self, mode: bool = True) -> "TrainModule":
        super().train(mode)
        self.backbone.set_training(mode)
        return self

    def forward(self, example: Example, video_id: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Answer loss and auxiliary loss; the latter is None without an objective."""
        if self.objective is None:
            return self.backbone.loss(example)[0], None
        answer_loss, hidden = self.backbone.loss(example, self.layer)
        if self.training and not hidden.requires_grad:
            raise RuntimeError("the supervised block is detached from the graph")
        intervals = example.intervals
        if len(intervals) < 2 and not self.objective.cross_clip:
            return answer_loss, None  # nothing to contrast: answer supervision alone
        clip = Clip(*pool(hidden, intervals), intervals, video_id, hidden, self.step, self.total_steps)
        return answer_loss, self.objective.loss(self.projection, clip)


def prefetch(prepare: Callable[[Any], Any], items: Iterable[Any]) -> Iterator[tuple[Any, Any]]:
    """Yield (item, prepare(item)) in order, preparing one item ahead on a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None
        for item in items:
            future = executor.submit(prepare, item)
            if pending is not None:
                yield pending[0], pending[1].result()
            pending = (item, future)
        if pending is not None:
            yield pending[0], pending[1].result()


def all_sum(value: float, device: torch.device) -> float:
    """Sum of a scalar over all processes."""
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.item()


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
        self.output = Path(args.output_dir)
        self.train_rows = data.load_rows(args.train_file)
        self.validation_rows = data.load_rows(args.validation_file)
        if args.batch_size % self.world or (len(self.train_rows) % args.batch_size) % self.world:
            raise ValueError("every optimizer batch must split evenly across processes")
        total_steps = args.epochs * math.ceil(len(self.train_rows) / args.batch_size)
        objective = OBJECTIVES.get(args.objective)
        if objective is not None and objective.cross_clip and self.world < 2:
            raise ValueError(f"{args.objective} compares clips across processes; launch at least two")

        self.backbone = backbones.load(args.backbone, args.model_path, self.device, train=True)
        torch.manual_seed(args.seed)  # every process draws the same initial projection
        module = TrainModule(self.backbone, objective, args.layer or self.backbone.layer, total_steps)
        torch.manual_seed(args.seed + self.rank)
        module.train()
        self.model = DistributedDataParallel(
            module, device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True,
        )
        self.trainable = [p for p in module.parameters() if p.requires_grad]
        # ZeRO-1: optimizer state is sharded across processes; the update is plain AdamW.
        self.optimizer = ZeroRedundancyOptimizer(
            self.trainable, optimizer_class=torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=0.1, foreach=False,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: step / max(1, args.warmup_steps) if step < args.warmup_steps
            else max(0.0, (total_steps - step) / max(1, total_steps - args.warmup_steps)),
        )
        self.progress = {"epoch": 0, "cursor": 0, "step": 0, "validation_nll": []}

    def log(self, **record: Any) -> None:
        if self.rank == 0:
            print(json.dumps(record), flush=True)

    def prepare(self, row: data.Row) -> Example:
        video = data.media_path(self.args.media_root, row)
        return self.backbone.prepare(video, row["duration"], data.question_text(row), data.target_text(row, self.args.target))

    # ------------------------------------------------------------------ steps
    def train_step(self, rows: list[data.Row]) -> dict[str, float]:
        """One optimizer step on a global batch; this process handles every `world`-th row."""
        local = rows[self.rank::self.world]
        self.model.module.step = self.progress["step"]
        self.optimizer.zero_grad(set_to_none=True)
        answer_sum = aux_sum = 0.0
        for position, (row, example) in enumerate(prefetch(self.prepare, local)):
            # Gradients are synchronized once, on the last example of the step.
            sync = contextlib.nullcontext() if position + 1 == len(local) else self.model.no_sync()
            with sync:
                answer_loss, aux_loss = self.model(example, row["video_id"])
                loss = answer_loss if aux_loss is None else answer_loss + self.args.weight * aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss on {row['id']}")
                (loss / len(local)).backward()
            answer_sum += answer_loss.item()
            aux_sum += 0.0 if aux_loss is None else aux_loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.args.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        return {
            "answer_loss": all_sum(answer_sum, self.device) / len(rows),
            "aux_loss": all_sum(aux_sum, self.device) / len(rows),
            "grad_norm": grad_norm.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> float:
        """Mean negative log-likelihood per target token over the validation set."""
        module = self.model.module
        module.eval()
        nll = tokens = 0.0
        for _, example in prefetch(self.prepare, self.validation_rows[self.rank::self.world]):
            loss, _ = self.backbone.loss(example)
            nll += loss.item() * example.num_target_tokens
            tokens += example.num_target_tokens
        module.train()
        return all_sum(nll, self.device) / all_sum(tokens, self.device)

    # ------------------------------------------------------------ checkpoints
    def save_epoch(self, epoch: int) -> None:
        """Trainable backbone parameters; the projection is kept apart and never needed at inference."""
        if self.rank == 0:
            torch.save(self.backbone.trainable_state(), self.output / f"epoch_{epoch}.pt")
            if hasattr(self.model.module, "projection"):
                torch.save(self.model.module.projection.state_dict(), self.output / f"epoch_{epoch}_projection.pt")
        dist.barrier()

    def save_resume(self) -> None:
        staging, final = self.output / "resume.tmp", self.output / "resume"
        if self.rank == 0:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
        dist.barrier()
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        torch.save({"optimizer": self.optimizer.optim.state_dict(), "rng": rng}, staging / f"rank_{self.rank}.pt")
        dist.barrier()
        if self.rank == 0:
            module = self.model.module
            trainable = {name for name, p in module.named_parameters() if p.requires_grad}
            state = {name: value for name, value in module.state_dict().items() if name in trainable}
            torch.save({"model": state, "scheduler": self.scheduler.state_dict(), "progress": self.progress}, staging / "state.pt")
            shutil.rmtree(final, ignore_errors=True)
            staging.rename(final)
        dist.barrier()

    def load_resume(self) -> None:
        folder = self.output / "resume"
        state = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
        local = torch.load(folder / f"rank_{self.rank}.pt", map_location="cpu", weights_only=False)
        missing, unexpected = self.model.module.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"resume checkpoint has unknown parameters: {unexpected[:5]}")
        self.optimizer.optim.load_state_dict(local["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
            group["lr"] = lr
        random.setstate(local["rng"]["python"])
        np.random.set_state(local["rng"]["numpy"])
        torch.random.set_rng_state(local["rng"]["torch"])
        if "cuda" in local["rng"]:
            torch.cuda.set_rng_state(local["rng"]["cuda"], self.device)
        self.progress = state["progress"]
        self.log(event="resumed", **self.progress)

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        if self.rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        if (self.output / "resume" / "state.pt").exists():
            self.load_resume()
        args, progress = self.args, self.progress
        while progress["epoch"] < args.epochs:
            while progress["cursor"] < len(self.train_rows):
                started = time.time()
                rows = self.train_rows[progress["cursor"]:progress["cursor"] + args.batch_size]
                metrics = self.train_step(rows)
                progress["cursor"] += len(rows)
                progress["step"] += 1
                if progress["step"] % args.log_every == 0:
                    self.log(event="train", epoch=progress["epoch"] + 1, step=progress["step"], seconds=round(time.time() - started, 1), **metrics)
                if progress["step"] == args.max_steps:
                    return
                if args.save_every and progress["step"] % args.save_every == 0 and progress["cursor"] < len(self.train_rows):
                    self.save_resume()
            progress["epoch"] += 1
            progress["cursor"] = 0
            progress["validation_nll"].append(self.validate())
            self.log(event="validation", epoch=progress["epoch"], nll=progress["validation_nll"][-1])
            self.save_epoch(progress["epoch"])
            if args.save_every and progress["epoch"] < args.epochs:
                self.save_resume()
        # Lowest validation answer NLL; the earlier epoch wins a tie.
        nll = progress["validation_nll"]
        selected = min(range(len(nll)), key=lambda epoch: (nll[epoch], epoch)) + 1
        if self.rank == 0:
            summary = {"selected_epoch": selected, "validation_nll": nll, "checkpoint": f"epoch_{selected}.pt", "args": vars(args)}
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            shutil.rmtree(self.output / "resume", ignore_errors=True)
        self.log(event="done", selected_epoch=selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=sorted(backbones.REGISTRY))
    parser.add_argument("--model-path", required=True, help="local snapshot of the released checkpoint")
    parser.add_argument("--media-root", required=True, help="directory holding <video_id>.mp4")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-file", default="data/omnivideo_10k/train.jsonl")
    parser.add_argument("--validation-file", default="data/omnivideo_10k/validation.jsonl")
    parser.add_argument("--objective", default="syncra", choices=["none", *OBJECTIVES])
    parser.add_argument("--target", default="answer", choices=data.TARGETS, help="'reasoning' trains Text CoT-SFT")
    parser.add_argument("--layer", type=int, help="supervised block, one-indexed (default: the backbone's selected block)")
    parser.add_argument("--weight", type=float, default=0.1, help="auxiliary loss weight")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=12, help="examples per optimizer step, over all processes")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25, help="optimizer steps between resume checkpoints; 0 disables")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=6))
    try:
        seed = args.seed + dist.get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        Trainer(args).run()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
