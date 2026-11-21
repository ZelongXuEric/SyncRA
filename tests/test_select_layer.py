"""Layer selection: state caching, probe fitting, retrieval score and the selection rule."""

import torch

from syncra import data
from syncra.select_layer import cache_states, first_questions, fit_probe, retrieval, select
from tests.toy import ToyBackbone, write_corpus

torch.set_num_threads(1)


def test_first_questions_keeps_one_row_per_source_video_in_order():
    rows = [{"video_id": v, "id": i} for i, v in enumerate("abacbd")]
    assert [row["id"] for row in first_questions(rows)] == [0, 1, 3, 5]


def test_cache_states_pools_every_candidate_block_and_drops_single_interval_clips(tmp_path):
    write_corpus(tmp_path)
    rows = first_questions(data.load_rows(tmp_path / "train.jsonl"))
    backbone = ToyBackbone(None, torch.device("cpu"))
    cached = cache_states(backbone, rows, tmp_path / "media", layers=[1, 3], max_seconds=None)
    assert 0 < len(cached) < len(rows)  # the toy corpus contains clips with a single interval
    for video in cached:
        assert len(video) == 2
        for audio, pooled_video in video:
            assert audio.shape == pooled_video.shape and audio.shape[0] >= 2 and audio.shape[1] == backbone.hidden_size


def test_probe_learns_a_shared_subspace_that_identifies_the_interval():
    generator = torch.Generator().manual_seed(0)
    mix_audio, mix_video = torch.randn(8, 48, generator=generator), torch.randn(8, 48, generator=generator)

    def video(intervals: int):
        latent = torch.randn(intervals, 8, generator=generator)  # what the two modalities share per interval
        noise = lambda: 0.3 * torch.randn(intervals, 48, generator=generator)
        return latent @ mix_audio + noise(), latent @ mix_video + noise()

    fit, held_out = [video(12) for _ in range(40)], [video(12) for _ in range(20)]
    torch.manual_seed(1)
    from syncra.objectives import build_projection

    untrained = retrieval(build_projection(48), held_out)
    trained = retrieval(fit_probe(fit, seed=1, epochs=20), held_out)
    assert untrained < 0.4 < 0.8 < trained


def test_selection_takes_the_best_block_inside_the_eligibility_band():
    chance = 0.02
    assert select({7: 0.11, 14: 0.16, 21: 0.20}, chance) == 21
    assert select({9: 0.95, 18: 0.74, 27: 0.36}, chance) == 18  # no headroom left at block 9
    assert select({12: 0.06, 24: 0.065}, chance) is None  # nothing clearly above chance
