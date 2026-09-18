"""Prepare, train, freeze and evaluate the AG News text MMS proxy (CPU only)."""

import argparse
import json

from mms_eval.utils import read_json
from mms_multimodal.agnews import (build_reference, evaluate_generated, prepare,
                                   train_classifier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare")
    split.add_argument("--train-csv", required=True)
    split.add_argument("--test-csv", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--validation-n", type=int, default=6000)
    train = commands.add_parser("train")
    train.add_argument("--splits", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--classifier-dir", required=True)
    reference.add_argument("--splits", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--alpha", type=float, default=.05)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--classifier-dir", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--generated", required=True,
                          help="JSONL manifest with sample_id+text, or a directory of .txt files")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--source-id")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.train_csv, args.test_csv, args.out, validation_n=args.validation_n)
    elif args.command == "train":
        result = train_classifier(args.splits, args.out, config=read_json(args.config))
    elif args.command == "reference":
        ref = build_reference(args.classifier_dir, args.splits, args.out, alpha=args.alpha)
        result = {"signature": ref["signature"], "real_audit": ref["real_audit"]}
    else:
        report = evaluate_generated(args.classifier_dir, args.reference, args.splits,
                                    args.generated, args.out, source_id=args.source_id)
        result = {"n": report["n"], "mms": report["mms"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
