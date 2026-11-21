# Swap-test cases

`cases.json` defines the 100 source clips of the swap test: 81 from Daily-Omni
(80 clips of 60 s and one of 53.5 s) and 19 from LVOmniBench (120 s each). No
source video appears in the fine-tuning data. Every clip has four spoken cues in
four non-overlapping 5 s windows, giving 400 cues and, with four input versions
each, 1,600 questions per checkpoint.

| Field | Meaning |
| --- | --- |
| `id` | case identifier |
| `benchmark`, `video_id` | source benchmark and its video identifier |
| `youtube_id` | YouTube ID of the source video |
| `offset`, `duration` | start and length, in seconds, of the clip inside the source video |
| `windows` | the four exchanged windows `[start, end]`, in clip time |
| `cues` | per window, the spoken phrase and its `[start, end]` in the original clip |
| `scenes` | per window, the scene visible while its cue is spoken in the original clip |
| `option_order` | scene indices in the order of options A-D |

Windows 1 and 3 and windows 2 and 4 are exchanged. Each phrase occurs once in
the clip and lies at least one second inside its window, and its scene is
visible for the whole phrase in every input version.

Videos are not included; obtain them from the Daily-Omni and LVOmniBench
releases and render the input versions with `python -m syncra.swap_test.media`.
