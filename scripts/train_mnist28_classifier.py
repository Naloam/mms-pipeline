"""Train a digit evaluator from official MNIST IDX files on the MNIST GPU host."""

import argparse
import json

from mms_multimodal.mnist28 import train_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026091706)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = train_classifier(args.data_dir, args.out, epochs=args.epochs, batch_size=args.batch_size,
                              seed=args.seed, device=args.device)
    print(json.dumps({"checkpoint_sha256": result["checkpoint_sha256"], "test": result["test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
