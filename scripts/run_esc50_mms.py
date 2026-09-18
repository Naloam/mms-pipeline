"""Train and apply the ESC-50 spectrogram MMS proxy without editing source assets."""

import argparse
import json

from mms_eval.utils import read_json
from mms_multimodal.esc50 import (build_reference, evaluate_generated, prepare,
                                  train_classifier)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare")
    split.add_argument("--dataset-dir", required=True)
    split.add_argument("--spectrogram-dir", required=True)
    split.add_argument("--out", required=True)
    train = commands.add_parser("train")
    train.add_argument("--splits", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--checkpoint", required=True)
    reference.add_argument("--splits", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--alpha", type=float, default=.05)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--generated-dir", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--count", type=int, default=5000)
    evaluate.add_argument("--source-id")
    evaluate.add_argument("--expected-total", type=int)
    evaluate.add_argument("--sample-seed", type=int, default=2026091713)
    evaluate.add_argument("--generator-checkpoint")
    for command in (reference, evaluate):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.dataset_dir, args.spectrogram_dir, args.out)
    elif args.command == "train":
        result = train_classifier(args.splits, args.out, config=read_json(args.config))
        result.pop("metadata", None)
    elif args.command == "reference":
        ref = build_reference(args.checkpoint, args.splits, args.out, device=args.device,
                              batch_size=args.batch_size, alpha=args.alpha)
        result = {"signature": ref["signature"], "real_audit": ref["real_audit"]}
    else:
        report = evaluate_generated(args.checkpoint, args.reference, args.splits,
                                    args.generated_dir, args.out, count=args.count,
                                    device=args.device, batch_size=args.batch_size,
                                    source_id=args.source_id, expected_total=args.expected_total,
                                    sample_seed=args.sample_seed,
                                    generator_checkpoint=args.generator_checkpoint)
        result = {"n": report["n"], "mms": report["mms"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
