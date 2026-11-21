"""Properties of the auxiliary objectives, checked against direct evaluations of their definitions."""

import math

import pytest
import torch
import torch.nn.functional as F

from syncra import objectives as O
from syncra.intervals import Intervals, pool

torch.set_num_threads(1)
K, D = 7, 48


def make_clip(audio, video, intervals, video_id="v"):
    return O.Clip(audio, video, intervals, video_id, hidden=torch.zeros(1, D))


@pytest.fixture
def clip():
    torch.manual_seed(0)
    intervals = Intervals(audio=[], video=[], index=[0, 1, 2, 4, 5, 8, 9], seconds=2.0)
    return O.build_projection(D), torch.randn(K, D), torch.randn(K, D), intervals


def logits(projection, audio, video):
    return O.embed(projection, audio) @ O.embed(projection, video).T / O.TAU


def test_embeddings_are_unit_vectors_in_float32(clip):
    projection, audio, _, _ = clip
    z = O.embed(projection.to(torch.bfloat16), audio.to(torch.bfloat16))
    assert z.dtype == torch.float32 and z.shape == (K, O.DIM)
    assert torch.allclose(z.norm(dim=-1), torch.ones(K), atol=1e-5)


def test_syncra_is_the_symmetric_interval_classification_loss(clip):
    projection, audio, video, intervals = clip
    s = logits(projection, audio, video)
    expected = -sum(s[k, k] - torch.logsumexp(s[k], 0) + s[k, k] - torch.logsumexp(s[:, k], 0) for k in range(K)) / (2 * K)
    assert torch.allclose(O.syncra(projection, make_clip(audio, video, intervals)), expected, atol=1e-5)


def test_collapsed_states_cost_log_k_under_syncra_but_nothing_under_cosine(clip):
    projection, audio, _, intervals = clip
    same = audio[:1].expand(K, D)
    assert torch.allclose(O.syncra(projection, make_clip(same, same, intervals)), torch.tensor(math.log(K)), atol=1e-4)
    assert torch.allclose(O.cosine_only(projection, make_clip(same, same, intervals)), torch.tensor(0.0), atol=1e-4)


def test_permuted_pairs_uses_a_fixed_derangement_as_positives(clip):
    projection, audio, video, intervals = clip
    pi = O.derangement("some_video", K)
    assert sorted(pi) == list(range(K)) and all(k != p for k, p in enumerate(pi))
    assert pi == O.derangement("some_video", K) and pi != O.derangement("another_video", K)
    s = logits(projection, audio, video)
    inverse = [pi.index(j) for j in range(K)]
    expected = 0.5 * (F.cross_entropy(s, torch.tensor(pi)) + F.cross_entropy(s.T, torch.tensor(inverse)))
    assert torch.allclose(O.permuted_pairs(projection, make_clip(audio, video, intervals, "some_video")), expected, atol=1e-5)


def test_time_codes_are_unit_norm_and_indexed_by_grid_time(clip):
    projection, audio, video, intervals = clip
    seconds = torch.tensor(intervals.index, dtype=torch.float32) * intervals.seconds
    codes = O.time_codes(seconds)
    assert codes.shape == (K, O.DIM) and torch.allclose(codes.norm(dim=-1), torch.ones(K), atol=1e-5)
    period = 4.0  # shortest period
    assert torch.allclose(codes[:, 0], torch.sin(2 * math.pi * seconds / period) / math.sqrt(O.DIM // 2), atol=1e-5)
    targets = torch.arange(K)
    expected = 0.5 * sum(F.cross_entropy(O.embed(projection, x) @ codes.T / O.TAU, targets) for x in (audio, video))
    assert torch.allclose(O.fixed_time_codes(projection, make_clip(audio, video, intervals)), expected, atol=1e-5)


def test_clip_contrast_treats_repeated_sources_as_positives():
    torch.manual_seed(0)
    z_audio, z_video = F.normalize(torch.randn(3, 16), dim=-1), F.normalize(torch.randn(3, 16), dim=-1)
    assert O.clip_contrast(z_audio, z_video, ["a", "a", "a"]).abs() < 1e-6
    s = z_audio @ z_video.T / O.TAU
    distinct = 0.5 * (F.cross_entropy(s, torch.arange(3)) + F.cross_entropy(s.T, torch.arange(3)))
    assert torch.allclose(O.clip_contrast(z_audio, z_video, ["a", "b", "c"]), distinct, atol=1e-5)


def test_pairwise_ranking_matches_its_double_sum(clip):
    projection, audio, video, intervals = clip
    s = logits(projection, audio, video)
    total = sum(F.softplus(s[i, j] - s[i, i]) + F.softplus(s[j, i] - s[i, i]) for i in range(K) for j in range(K) if i != j)
    assert torch.allclose(O.pairwise_ranking(projection, make_clip(audio, video, intervals)), total / (2 * K * (K - 1)), atol=1e-5)


def test_pooling_averages_each_modality_over_its_spans():
    hidden = torch.arange(40.0).reshape(20, 2)
    intervals = Intervals(audio=[[(0, 2)], [(6, 8)]], video=[[(2, 6)], [(8, 10), (12, 14)]], index=[0, 1], seconds=1.0)
    audio, video = pool(hidden, intervals)
    assert torch.equal(audio, torch.stack([hidden[0:2].mean(0), hidden[6:8].mean(0)]))
    assert torch.equal(video[1], torch.cat([hidden[8:10], hidden[12:14]]).mean(0))


def test_pooling_of_a_clip_without_intervals_is_empty():
    audio, video = pool(torch.zeros(5, 3), Intervals(audio=[], video=[], index=[], seconds=1.0))
    assert audio.shape == video.shape == (0, 3)


def test_avaf_net_is_zero_when_all_similarity_structures_agree():
    from syncra import prior_objectives as P

    torch.manual_seed(0)
    heads = P.build_modality_heads(D)
    heads["video"].load_state_dict(heads["audio"].state_dict())
    hidden = torch.randn(4, D).repeat_interleave(3, dim=0)  # four intervals of three identical tokens
    spans = [[(3 * k, 3 * k + 3)] for k in range(4)]
    intervals = Intervals(audio=spans, video=spans, index=list(range(4)), seconds=1.0)
    clip = O.Clip(*pool(hidden, intervals), intervals, "v", hidden)
    assert P.avaf_net(heads, clip).abs() < 1e-5
    clip.video = clip.video.flip(0)
    assert P.avaf_net(heads, clip) > 1e-3
