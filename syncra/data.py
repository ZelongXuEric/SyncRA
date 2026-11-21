"""The 10K fine-tuning corpus: rows, prompts and supervision targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Row = dict[str, Any]

TARGETS = ("answer", "reasoning")


def load_rows(path: str | Path) -> list[Row]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_text(row: Row) -> str:
    """The question followed by its lettered options, if any."""
    options = row.get("options")
    if not options:
        return row["question"]
    lines = [f"{chr(ord('A') + i)}. {text}" for i, text in enumerate(options)]
    return row["question"] + "\n" + "\n".join(lines)


def target_text(row: Row, target: str = "answer") -> str:
    """Assistant target: the final answer alone, or the supplied reasoning followed by it."""
    if target == "answer":
        return row["answer"]
    if target == "reasoning":
        return f"{row['reasoning']}\nFinal answer: {row['answer']}"
    raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")


def media_path(media_root: str | Path, row: Row) -> Path:
    path = Path(media_root) / f"{row['video_id']}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
