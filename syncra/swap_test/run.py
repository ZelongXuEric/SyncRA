"""Answer every swap-test question with one checkpoint.

    python -m syncra.swap_test.run --backbone qwen3-omni --model-path /path/to/model \\
        --checkpoint runs/qwen3_syncra/epoch_2.pt --media-dir media \\
        --output predictions/qwen3_syncra.jsonl

Without --checkpoint the released model is evaluated. Decoding is greedy and
each question is answered independently. Finished questions are skipped on
restart; --shard splits the work across jobs writing to separate files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import backbones
from .cases import load_cases, parse_answer, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=sorted(backbones.REGISTRY))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", help="trainable-parameter checkpoint written by syncra.train")
    parser.add_argument("--cases", default="data/swap_test/cases.json")
    parser.add_argument("--media-dir", required=True, type=Path, help="output directory of syncra.swap_test.media")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--answer-prefix", help="text that opens the answer, for backbones that support it")
    parser.add_argument("--shard", default="0/1", help="INDEX/COUNT")
    args = parser.parse_args()

    index, count = map(int, args.shard.split("/"))
    todo = list(questions(load_cases(args.cases)))[index::count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.read_text().splitlines()}
        todo = [question for question in todo if question["id"] not in done]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = backbones.load(args.backbone, args.model_path, torch.device("cuda", 0))
    if args.checkpoint:
        backbone.load_trainable_state(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    extra = {"answer_prefix": args.answer_prefix} if args.answer_prefix else {}
    with open(args.output, "a", encoding="utf-8") as out:
        for question in todo:
            video = args.media_dir / question["case"] / f"{question['version']}.mov"
            text = backbone.generate(video, question.pop("prompt"), args.max_new_tokens, **extra)
            out.write(json.dumps({**question, "output": text, "prediction": parse_answer(text)}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
