"""Swap-test geometry, questions, answer parsing and statistics."""

import json

import numpy as np
import pytest

from syncra.swap_test import cases as C
from syncra.swap_test import score as S
from syncra.swap_test.media import exchange

CASE = {
    "id": "case", "benchmark": "Daily-Omni", "video_id": "abc", "offset": 0.0, "duration": 60,
    "windows": [[4.0, 9.0], [15.5, 20.5], [31.0, 36.0], [48.0, 53.0]],
    "cues": [{"text": f"cue {i}"} for i in range(4)],
    "scenes": [f"scene {i}" for i in range(4)],
    "option_order": [2, 0, 3, 1],
}


def test_single_stream_exchanges_pair_a_cue_with_its_partner_scene():
    table = {cue: [C.visible_scene(cue, version) for version in C.VERSIONS] for cue in range(4)}
    assert table == {0: [0, 2, 2, 0], 1: [1, 3, 3, 1], 2: [2, 0, 0, 2], 3: [3, 1, 1, 3]}


def test_questions_are_identical_across_versions_and_answers_follow_option_order():
    rows = list(C.questions([CASE]))
    assert len(rows) == 16 and len({row["id"] for row in rows}) == 16
    first = rows[:4]
    assert len({row["prompt"] for row in first}) == 1
    assert first[0]["prompt"] == (
        'While the phrase "cue 0" is being spoken, which of the following scenes is visible?\n'
        "A. scene 2\nB. scene 0\nC. scene 3\nD. scene 1\nAnswer with A, B, C, or D only."
    )
    assert [row["answer"] for row in first] == ["B", "A", "A", "B"]  # O, A, V, AV


@pytest.mark.parametrize("text, expected", [
    ("C", "C"), (" b. ", "B"), ("(A)", "A"), ("Answer: D", "D"), ("The correct answer is B.", "B"),
    ("I think it shows a kitchen.\nFinal answer: A", "A"), ("Final answer: B ... no, Final answer: C.", "C"),
    ("A kitchen is visible.", None), ("B because the audio mentions it", None), ("", None),
])
def test_parse_answer(text, expected):
    assert C.parse_answer(text) == expected


def test_exchange_swaps_windows_and_keeps_everything_else():
    track = np.arange(60 * 30)
    swapped = exchange(track, CASE["windows"], 30)
    assert np.array_equal(swapped[120:270], track[930:1080]) and np.array_equal(swapped[930:1080], track[120:270])
    outside = np.ones(len(track), bool)
    for start, end in CASE["windows"]:
        outside[round(start * 30):round(end * 30)] = False
    assert np.array_equal(swapped[outside], track[outside])
    assert np.array_equal(exchange(swapped, CASE["windows"], 30), track)


def test_scores_credit_a_cue_only_when_all_four_versions_are_right(tmp_path):
    cases = [dict(CASE, id=f"case{i}") for i in range(3)]
    path = tmp_path / "predictions.jsonl"
    with open(path, "w") as handle:
        for question in C.questions(cases):
            wrong = question["case"] == "case0" and question["cue"] == 1 and question["version"] == "V"
            missing = question["case"] == "case2" and question["cue"] == 0 and question["version"] == "A"
            if not missing:
                handle.write(json.dumps({"id": question["id"], "prediction": None if wrong else question["answer"]}) + "\n")
    scores = S.per_source(S.correctness(cases, path))
    assert scores["AllFour"].tolist() == [0.75, 1.0, 0.75]
    assert scores["Overall"].tolist() == [15 / 16, 1.0, 15 / 16]
    assert scores["O"].tolist() == [1.0, 1.0, 1.0] and scores["V"].tolist() == [0.75, 1.0, 1.0]


def test_paired_bootstrap_resamples_sources():
    rng = np.random.default_rng(0)
    left, right = rng.random(100), rng.random(100)
    difference, low, high = S.paired_bootstrap(left, right)
    assert difference == pytest.approx((left - right).mean()) and low < difference < high
    assert S.paired_bootstrap(left, right) == (difference, low, high)
    assert S.paired_bootstrap(left, left) == (0.0, 0.0, 0.0)
