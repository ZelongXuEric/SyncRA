# SyncRA: Learning Temporal Correspondence in Omni-Modal Models

Code and data for the paper [SyncRA: Learning Temporal Correspondence in Omni-Modal Models](https://arxiv.org/abs/2609.34363).

SyncRA (Synchrony-Guided Representation Alignment) is a training-only auxiliary
objective for omni-modal question answering. At one intermediate block of the
answering model, audio and visual hidden states are mean-pooled within the
processor's native time intervals, mapped through a shared 64-dimensional
projection, and contrasted with a symmetric InfoNCE loss: states from the same
interval attract, states from other intervals of the same video repel. The
pairing signal comes from input timing alone, and the projection is discarded
after training, so inference is unchanged.

This repository contains

- the SyncRA objective with its controls and loss variants (`syncra/objectives.py`), and ports of three
  prior synchrony objectives (`syncra/prior_objectives.py`),
- QA fine-tuning for four backbones (`syncra/train.py`, `syncra/backbones/`),
- the probe that selects the supervised block (`syncra/select_layer.py`),
- the swap test that measures whether answers follow the audio-visual pairing in the input
  (`syncra/swap_test/`, cases in `data/swap_test/`),
- the 10K-question fine-tuning subset of OmniVideo-100K (`data/omnivideo_10k/`).

Model weights, checkpoints and result files are not part of the release.

## Setup

Python 3.11 and PyTorch 2.5.1 (CUDA 12.4). The backbones need different
`transformers` versions, so each gets its own environment:

```bash
pip install -r requirements/qwen3-omni.txt   # or qwen2.5-omni / minicpm-o-4.5 / nemotron-3-nano-omni
pip install -e .
```

| `--backbone` | Released checkpoint (revision) | Supervised block | Trained parameters |
| --- | --- | --- | --- |
| `qwen2.5-omni` | `Qwen/Qwen2.5-Omni-7B` (`ae9e169`) | 21 of 28 | full Thinker |
| `minicpm-o-4.5` | `openbmb/MiniCPM-o-4_5` (`503e754`) | 9 of 36 | full understanding model, TTS disabled |
| `qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (`26291f7`) | 36 of 48 | all but routed experts and router gates (8.56%) |
| `nemotron-3-nano-omni` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (`e5e9932`) | 26 of 52 | all but routed experts (11.03%) |

Download a checkpoint with `huggingface-cli download <repo> --revision <revision> --local-dir <dir>`
and pass the directory as `--model-path`. Nemotron additionally loads the code
of its vision tower (`nvidia/C-RADIOv4-H`, revision `0057b33`) from the hub;
on machines without network access, point `RADIO_CODE_DIR` at a local copy of
that revision's Python files.

## Data

`data/omnivideo_10k/` holds the questions, answers and reasoning text of the
fine-tuning subset: 9,500 training questions from 1,367 videos and 500
validation questions from 415 other videos (see its README for the format).
The videos are not redistributed. Obtain them from OmniVideo-100K and place
each clip at `<media-root>/<video_id>.mp4`.

## Training

Runs in the paper use two processes on two 141 GB GPUs, an effective batch of
12 examples, and two epochs (1,584 optimizer steps).

```bash
torchrun --nproc_per_node 2 -m syncra.train \
    --backbone qwen3-omni --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --media-root /path/to/omnivideo/videos \
    --objective syncra --output-dir runs/qwen3_syncra
```

`--objective` selects what is added to the answer loss, with weight `--weight` (0.1):

| `--objective` | Paper name | Auxiliary target |
| --- | --- | --- |
| `none` | Vanilla SFT | answer supervision only |
| `syncra` | SyncRA | same-interval audio-video pairs against other intervals of the video |
| `permuted_pairs` | Permuted pairs | the same contrast with every positive moved to a wrong interval |
| `fixed_time_codes` | Fixed time codes | each modality classifies a fixed code of its interval's time |
| `clip_level_av` | Clip-level AV | clip-level audio-video contrast across the clips in the current step |
| `cosine_only` | Cosine-only | attraction of same-interval pairs, no competing intervals |
| `pairwise_ranking` | Pairwise ranking | independent logistic ranking of each positive over each negative |
| `korbar` | Korbar et al. (2018) | margin loss with other-video negatives, later 25% same-video negatives |
| `cav_mae_sync` | CAV-MAE Sync | video-to-audio InfoNCE with other-video negatives only |
| `avaf_net` | AVAF-Net | consistency of cross-modal and within-modal similarity structure |

`clip_level_av`, `korbar` and `cav_mae_sync` take their other-video candidates
from the clips held by the other training processes, so they need at least two.

`--target reasoning` supervises the supplied reasoning followed by the answer
(Text CoT-SFT); `--layer` overrides the supervised block.

Examples are visited in file order. After each epoch the trainer evaluates the
mean answer-token negative log-likelihood on the validation set and writes
`epoch_<n>.pt`, which holds the trained backbone parameters only. `summary.json`
names the epoch with the lowest validation NLL; no benchmark score enters the
selection. An interrupted run resumes from `<output-dir>/resume` when relaunched
with the same command.

`--seed` fixes the initial projection and the dropout of backbones that train
with it (the Nemotron audio encoder). GPU runs are still not bitwise
reproducible across launches.

## Selecting the supervised block

The supervised block of each backbone is chosen once, before fine-tuning, from
its frozen released weights:

```bash
python -m syncra.select_layer --backbone qwen3-omni --model-path /path/to/model \
    --media-root /path/to/omnivideo/videos --output-dir runs/select_qwen3
```

The probe caches interval-pooled states at one-quarter, one-half and
three-quarter depth, fits a 64-dimensional shared projection per candidate with
the SyncRA loss alone (first 200 training videos), and scores same-interval
retrieval on the validation videos. It selects the best candidate whose R@1 is
at least five points above chance and below 90%. The projections fitted here
are not reused for training. `--max-seconds` restricts the probe to the start
of each clip; the paper probes the first 60 s for MiniCPM-o 4.5 and the first
120 s for Nemotron, and whole clips for the Qwen models.

## Swap test

Each case is one source clip with four spoken cues in four equal-length
windows and four scene descriptions. Exchanging windows 1<->3 and 2<->4 in the
audio stream, the video stream, or both gives four input versions (O, A, V,
AV). The question is the same in all of them:

> While the phrase "{cue}" is being spoken, which of the following scenes is visible?

A single-stream exchange changes the correct scene; exchanging both streams
moves the original pairing to another position. AllFour credits a cue only if
all four versions are answered correctly.

`data/swap_test/cases.json` defines the 100 source clips (81 from Daily-Omni,
19 from LVOmniBench; 400 cues, 1,600 questions per checkpoint). The source
videos are not redistributed.

```bash
# 1. Render the four versions of every case (needs ffmpeg with libx264).
python -m syncra.swap_test.media --output-dir media \
    --daily-omni /path/to/Daily-Omni --lvomnibench /path/to/LVOmniBench

# 2. Answer all questions with a checkpoint (omit --checkpoint for the released model).
python -m syncra.swap_test.run --backbone qwen3-omni --model-path /path/to/model \
    --checkpoint runs/qwen3_syncra/epoch_2.pt --media-dir media \
    --output predictions/qwen3_syncra.jsonl

# 3. AllFour, per-version accuracy, and paired bootstrap intervals against a baseline.
python -m syncra.swap_test.score predictions/qwen3_syncra.jsonl --baseline predictions/qwen3_vanilla.jsonl
```

Decoding is greedy with at most 1,280 new tokens. The released MiniCPM-o 4.5
checkpoint is the one exception: it is run with `--answer-prefix "Final answer:"
--max-new-tokens 6`. Answers are parsed with a fixed regular expression;
unparsed and missing answers count as incorrect.
Intervals are 95% percentile intervals over 10,000 resamples of source videos.

## Tests

```bash
pytest
```

The tests run on CPU with a toy backbone. They check that two-process training
with gradient accumulation matches a single-process reference for every
objective, including those that exchange features between processes, and that
a resumed run reproduces an uninterrupted one.

## Citation

```bibtex
@article{xu2026syncra,
  title   = {SyncRA: Learning Temporal Correspondence in Omni-Modal Models},
  author  = {Xu, Zelong and Li, Yan and Hu, Wenhe and Hu, Xiyang},
  journal = {arXiv preprint arXiv:2609.34363},
  year    = {2026}
}
```
