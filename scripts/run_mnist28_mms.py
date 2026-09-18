"""Freeze the MNIST real reference, then score generated native 28x28 PNGs."""

import argparse
import json

from mms_multimodal.mnist28_pipeline import build_reference, evaluate_generated


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--data-dir", required=True)
    reference.add_argument("--classifier-dir", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--alpha", type=float, default=.05)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--classifier-dir", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--generated-manifest", required=True)
    evaluate.add_argument("--out", required=True)
    for command in (reference, evaluate):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.command == "reference":
        result = build_reference(args.data_dir, args.classifier_dir, args.out,
                                 device=args.device, batch_size=args.batch_size, alpha=args.alpha)
        print(json.dumps({"signature": result["signature"], "real_audit": result["real_audit"]}))
    else:
        result = evaluate_generated(args.classifier_dir, args.reference, args.generated_manifest,
                                    args.out, device=args.device, batch_size=args.batch_size)
        print(json.dumps({"n": result["n"], "mms": result["mms"]}))


if __name__ == "__main__":
    main()
