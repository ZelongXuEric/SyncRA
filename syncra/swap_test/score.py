"""Score swap-test predictions: AllFour, per-version accuracy and paired bootstrap intervals.

    python -m syncra.swap_test.score syncra.jsonl --baseline vanilla.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cases import VERSIONS, Case, load_cases, questions

BOOTSTRAP_SEED = 20260917


def correctness(cases: list[Case], predictions_file: str | Path) -> np.ndarray:
    """Boolean array [source, cue, version]; missing and unparsed answers are incorrect."""
    with open(predictions_file, encoding="utf-8") as handle:
        predicted = {row["id"]: row["prediction"] for row in map(json.loads, handle)}
    correct = [predicted.get(question["id"]) == question["answer"] for question in questions(cases)]
    return np.asarray(correct).reshape(len(cases), 4, len(VERSIONS))


def per_source(correct: np.ndarray) -> dict[str, np.ndarray]:
    """Each metric as one score per source video, the unit that is resampled."""
    scores = {"AllFour": correct.all(axis=2).mean(axis=1), "Overall": correct.mean(axis=(1, 2))}
    scores.update({version: correct[:, :, i].mean(axis=1) for i, version in enumerate(VERSIONS)})
    return scores


def paired_bootstrap(left: np.ndarray, right: np.ndarray, replicates: int = 10000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean per-source difference with a 95% percentile interval over resampled sources."""
    delta = np.asarray(left, float) - np.asarray(right, float)
    indices = np.random.default_rng(seed).integers(0, len(delta), size=(replicates, len(delta)))
    low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions")
    parser.add_argument("--cases", default="data/swap_test/cases.json")
    parser.add_argument("--baseline", help="predictions of a second checkpoint for paired differences")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    scores = per_source(correctness(cases, args.predictions))
    for name, values in scores.items():
        print(f"{name:8s} {100 * values.mean():6.2f}")
    # AllFour conditioned on answering the original version correctly.
    print(f"{'P(all|O)':8s} {100 * scores['AllFour'].mean() / scores['O'].mean():6.2f}")
    if args.baseline:
        baseline = per_source(correctness(cases, args.baseline))
        print("\ndifference to baseline, percentage points [95% interval]")
        for name in scores:
            difference, low, high = paired_bootstrap(scores[name], baseline[name])
            print(f"{name:8s} {100 * difference:+6.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")


if __name__ == "__main__":
    main()
