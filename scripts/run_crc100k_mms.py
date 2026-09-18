"""CRC100K classifier, external real reference and generated-image MMS proxy."""

import argparse
import json

from mms_eval.utils import read_json
from mms_multimodal.crc100k import (build_reference, evaluate_generated,
                                     prepare_classifier_split, prepare_quality_inputs,
                                     train_classifier)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--out", required=True)
    quality = commands.add_parser("quality-prepare")
    quality.add_argument("--raw-dir", required=True)
    quality.add_argument("--generated-dir", required=True)
    quality.add_argument("--out", required=True)
    train = commands.add_parser("train")
    train.add_argument("--splits", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--checkpoint", required=True)
    reference.add_argument("--official-validation-dir", required=True)
    reference.add_argument("--official-zip-md5", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--alpha", type=float, default=.05)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--generated-dir", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--count", type=int, default=5000)
    for command in (reference, evaluate):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_classifier_split(args.raw_dir, args.out)
    elif args.command == "quality-prepare":
        result = prepare_quality_inputs(args.raw_dir, args.generated_dir, args.out)
    elif args.command == "train":
        result = train_classifier(args.splits, args.out, config=read_json(args.config))
        result.pop("metadata", None)
    elif args.command == "reference":
        result = build_reference(args.checkpoint, args.official_validation_dir, args.out,
                                 device=args.device, batch_size=args.batch_size,
                                 alpha=args.alpha, official_zip_md5=args.official_zip_md5)
        result = {"signature": result["signature"], "real_audit": result["real_audit"]}
    else:
        result = evaluate_generated(args.checkpoint, args.reference, args.generated_dir, args.out,
                                    device=args.device, batch_size=args.batch_size, count=args.count)
        result = {"n": result["n"], "mms": result["mms"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
