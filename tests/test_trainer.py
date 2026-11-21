"""End-to-end trainer checks on a toy backbone: two CPU processes against a single-process reference."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

from syncra import data
from syncra import prior_objectives as P
from syncra.intervals import pool
from syncra.objectives import Clip, clip_contrast, embed
from syncra.train import OBJECTIVES
from tests.toy import ToyBackbone, write_corpus

ROOT = Path(__file__).resolve().parents[1]
BATCH, WARMUP, LR, WEIGHT, SEED = 12, 2, 1e-2, 0.1, 1
torch.set_num_threads(1)  # tiny models: thread pools only add contention


def launch(corpus: Path, out: Path, *extra: str) -> None:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(ROOT / "tests" / "toy.py"),
        "--backbone", "toy", "--model-path", "unused", "--media-root", str(corpus / "media"),
        "--train-file", str(corpus / "train.jsonl"), "--validation-file", str(corpus / "validation.jsonl"),
        "--output-dir", str(out), "--lr", str(LR), "--warmup-steps", str(WARMUP), "--seed", str(SEED), *extra,
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-3000:]


def cross_clip_loss(name: str, head: torch.nn.Module, left: Clip, right: Clip) -> torch.Tensor:
    """Summed auxiliary loss of the two clips that the two processes hold in the same step."""
    if name == "clip_level_av":
        z = [embed(head, torch.stack((clip.audio.mean(0), clip.video.mean(0)))) for clip in (left, right)]
        loss = clip_contrast(torch.stack((z[0][0], z[1][0])), torch.stack((z[0][1], z[1][1])), [left.video_id, right.video_id])
        return 2 * loss  # both processes evaluate the same loss
    heads = (head["audio"], head["video"]) if name == "korbar" else (head, head)
    features = [P._local_features(*heads, clip) for clip in (left, right)]
    total = 0.0
    for rank, (clip, other, other_features) in enumerate(((left, right, features[1]), (right, left, features[0]))):
        others = [other_features] if len(other_features[0]) and other.video_id != clip.video_id else []
        with mock.patch.object(P, "gather_others", lambda *_: (others, torch.zeros(()))), mock.patch.object(P.dist, "get_rank", lambda: rank):
            total = total + OBJECTIVES[name].loss(head, clip)
    return total


def reference(corpus: Path, name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Two epochs of plain single-process AdamW on the mean loss of each 12-example batch."""
    rows = data.load_rows(corpus / "train.jsonl")
    objective = OBJECTIVES[name]
    backbone = ToyBackbone(None, torch.device("cpu"), train=True)
    torch.manual_seed(SEED)
    head = objective.build_head(backbone.hidden_size)
    parameters = [p for p in [*backbone.model.parameters(), *head.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    total = 2 * len(rows) // BATCH
    for step in range(total):
        batch = rows[(step * BATCH) % len(rows):][:BATCH]
        answer, aux, clips = 0.0, 0.0, []
        for row in batch:
            example = backbone.prepare(Path(row["video_id"]), 60.0, data.question_text(row), data.target_text(row))
            loss, hidden = backbone.loss(example, backbone.layer)
            answer = answer + loss / len(batch)
            clips.append(Clip(*pool(hidden, example.intervals), example.intervals, row["video_id"], hidden, step, total))
        if objective.cross_clip:
            aux = sum(cross_clip_loss(name, head, left, right) for left, right in zip(clips[0::2], clips[1::2])) / len(batch)
        else:  # clips with a single interval receive answer supervision alone
            aux = sum(objective.loss(head, clip) for clip in clips if len(clip.intervals) >= 2) / len(batch)
        for group in optimizer.param_groups:
            group["lr"] = LR * (step / WARMUP if step < WARMUP else (total - step) / (total - WARMUP))
        optimizer.zero_grad()
        (answer + WEIGHT * aux).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return backbone.trainable_state(), head.state_dict()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    write_corpus(root)
    return root


@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_two_processes_match_single_process_reference(corpus, tmp_path, name):
    launch(corpus, tmp_path, "--objective", name, "--save-every", "0")
    state, head = reference(corpus, name)
    trained = torch.load(tmp_path / "epoch_2.pt")
    assert trained.keys() == state.keys()
    for key in state:
        assert torch.allclose(trained[key], state[key], rtol=1e-4, atol=1e-6), key
    trained_head = torch.load(tmp_path / "epoch_2_projection.pt")
    torch.manual_seed(SEED)
    initial = OBJECTIVES[name].build_head(32).state_dict()
    for key in head:
        assert torch.allclose(trained_head[key], head[key], rtol=1e-4, atol=1e-6), key
        assert not torch.equal(trained_head[key], initial[key])  # the head did train


def test_cross_clip_objectives_need_two_processes(corpus, tmp_path):
    command = [
        sys.executable, str(ROOT / "tests" / "toy.py"), "--backbone", "toy", "--model-path", "unused",
        "--media-root", str(corpus / "media"), "--train-file", str(corpus / "train.jsonl"),
        "--validation-file", str(corpus / "validation.jsonl"), "--output-dir", str(tmp_path), "--objective", "cav_mae_sync",
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
           "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29571"}
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode != 0 and "launch at least two" in result.stderr


def test_resume_reproduces_an_uninterrupted_run(corpus, tmp_path):
    launch(corpus, tmp_path / "straight", "--save-every", "0")
    launch(corpus, tmp_path / "resumed", "--save-every", "1", "--max-steps", "3")
    assert (tmp_path / "resumed" / "resume" / "state.pt").exists()
    launch(corpus, tmp_path / "resumed", "--save-every", "1")
    for name in ("epoch_1.pt", "epoch_2.pt", "epoch_2_projection.pt"):
        straight, resumed = torch.load(tmp_path / "straight" / name), torch.load(tmp_path / "resumed" / name)
        assert all(torch.equal(straight[key], resumed[key]) for key in straight)
    assert not (tmp_path / "resumed" / "resume").exists()
    summaries = [json.loads((tmp_path / run / "summary.json").read_text()) for run in ("straight", "resumed")]
    assert summaries[0]["validation_nll"] == summaries[1]["validation_nll"] and summaries[0]["selected_epoch"] in (1, 2)


def test_answer_only_training_saves_trainable_parameters_only(corpus, tmp_path):
    launch(corpus, tmp_path, "--objective", "none", "--save-every", "0")
    state = torch.load(tmp_path / "epoch_2.pt")
    assert state and not any(name.startswith("blocks.0.") for name in state)  # the frozen block is not stored
    assert not (tmp_path / "epoch_2_projection.pt").exists()
    backbone = ToyBackbone(None, torch.device("cpu"))
    untouched = backbone.model.blocks[0][0].weight.clone()
    backbone.load_trainable_state(state)
    assert torch.equal(backbone.model.blocks[0][0].weight, untouched)
    assert torch.equal(backbone.model.lm_head.weight, state["lm_head.weight"])
