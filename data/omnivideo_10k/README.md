# OmniVideo-10K fine-tuning subset

A fixed, balanced subset of [OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K)
(revision `fc57fe1fdaf688a12bfea9cbc7683a89d68bc686`) used for every fine-tuning run.

| Split | File | Questions | Source videos |
| --- | --- | ---: | ---: |
| Training | `train.jsonl` | 9,500 | 1,367 |
| Validation | `validation.jsonl` | 500 | 415 |

Half of the questions are multiple-choice and half are open-ended; each of ten
task categories contributes 1,000 questions. Training and validation share no
source video. Clips last 60-180 seconds. Row order is the training order.

Each line is one question:

| Field | Meaning |
| --- | --- |
| `id` | question identifier in OmniVideo-100K |
| `video_id` | source video (YouTube ID); the clip is `<video_id>.mp4` |
| `duration` | clip length in seconds |
| `task` | task category |
| `question` | question text |
| `options` | option texts in order A, B, ... for multiple-choice questions, otherwise `null` |
| `answer` | option letter or free-text answer; the target of answer-only fine-tuning |
| `reasoning` | reasoning text supplied by OmniVideo-100K; used only by Text CoT-SFT |

The annotations are redistributed under the Apache-2.0 license of OmniVideo-100K.
Videos are not included; obtain them from the original release.
