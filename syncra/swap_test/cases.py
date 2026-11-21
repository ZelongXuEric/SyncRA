"""Swap-test cases: window geometry, questions, correct answers and answer parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

Case = dict[str, Any]

VERSIONS = ("O", "A", "V", "AV")  # original, audio exchanged, video exchanged, both exchanged
PARTNER = (2, 3, 0, 1)  # the window each window is exchanged with
LETTERS = "ABCD"


def load_cases(path: str | Path) -> list[Case]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def visible_scene(cue: int, version: str) -> int:
    """Scene shown while cue `cue` is spoken.

    Exchanging a single stream pairs the cue with its partner window's scene;
    exchanging both moves the original pairing to the partner window intact.
    """
    return cue if version in ("O", "AV") else PARTNER[cue]


def answer(case: Case, cue: int, version: str) -> str:
    return LETTERS[case["option_order"].index(visible_scene(cue, version))]


def prompt(case: Case, cue: int) -> str:
    """The question is identical across versions: no timestamp, window position or version label."""
    options = "\n".join(f"{letter}. {case['scenes'][scene]}" for letter, scene in zip(LETTERS, case["option_order"]))
    return (
        f'While the phrase "{case["cues"][cue]["text"]}" is being spoken, which of the following scenes is visible?\n'
        f"{options}\nAnswer with A, B, C, or D only."
    )


def questions(cases: list[Case]) -> Iterator[dict[str, Any]]:
    """All questions: every cue of every case under each of the four versions."""
    for case in cases:
        for cue in range(4):
            for version in VERSIONS:
                yield {
                    "id": f"{case['id']}/{cue}/{version}", "case": case["id"], "cue": cue, "version": version,
                    "prompt": prompt(case, cue), "answer": answer(case, cue, version),
                }


_DIRECT = re.compile(
    r"\A\s*(?:(?:answer|option)\s*:\s*|(?:the\s+)?(?:best|correct)\s+(?:answer|option)\s+is\s+)?"
    r"[\[(]?([ABCD])[\])]?[.,:]?\s*\Z",
    re.IGNORECASE,
)
_FINAL = re.compile(r"(?<![A-Za-z])final\s+answer\s*:\s*[\[(]?([ABCD])[\])]?[.,:]?(?=\s|$)", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    """A complete direct option answer, or the last explicit `Final answer:` field.

    Bare letters inside free text are never matched; anything else is unparsed
    and scored as incorrect.
    """
    if match := _DIRECT.fullmatch(text):
        return match.group(1).upper()
    if matches := _FINAL.findall(text):
        return matches[-1].upper()
    return None
