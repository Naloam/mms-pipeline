"""Semantic-only MMS evaluation: Algorithm-1 outputs without quality metrics.

Runs the same frozen path as `mms_eval evaluate` (reference validation ->
checkpoint check -> predict_paths -> posterior_scores -> aggregate_mms ->
make_score_rows) but skips FID/KID/IS/PR backends, so it needs no GPU and no
Inception/VGG weights. Writes report.json + scores.jsonl in the pipeline's
own format; scripts/export_excel.py and scripts/report_intervals.py work on
the output unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

from mms_eval.evaluator import describe_checkpoint, predict_paths
from mms_eval.pipeline import code_signature, load_reference
from mms_eval.semantic import aggregate_mms, make_score_rows, posterior_scores
from mms_eval.utils import write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="CSV/JSONL manifest, image file, or image directory")
    parser.add_argument("--reference", required=True, help="frozen reference.json")
    parser.add_argument("--out", required=True, help="output directory (must be new)")
    parser.add_argument("--checkpoint", help="evaluator best.pt; default: reference path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    if reference["code_sha256"] != code_signature():
        raise SystemExit("Evaluation code differs from the frozen reference")
    checkpoint = args.checkpoint or reference["evaluator_checkpoint"]
    metadata = describe_checkpoint(checkpoint)
    if metadata["checkpoint_sha256"] != reference["checkpoint_sha256"]:
        raise SystemExit("Checkpoint differs from the frozen reference")

    from mms_eval.images import collect_images
    records = collect_images(args.input)
    for row in records:
        if row.get("sha256"):
            digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise SystemExit(f"input hash mismatch: {row['path']}")
    if not records:
        raise SystemExit("No input images found")

    temperature = reference["config"].get("temperature", 1)
    logits = predict_paths([row["path"] for row in records], checkpoint,
                           device=args.device, batch_size=args.batch_size)
    scores = posterior_scores(logits["logits"], temperature)
    mms = aggregate_mms(scores, reference["calibrations"], reference["classes"])
    rows = make_score_rows(records, logits["logits"], reference["calibrations"],
                           reference["classes"], temperature=temperature,
                           protocol_id=reference["protocol_id"],
                           evaluator_id=reference["evaluator_id"])

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "scores.jsonl", rows)
    write_json(out / "report.json", {
        "n": len(records), "protocol_id": reference["protocol_id"],
        "evaluator_id": reference["evaluator_id"],
        "evaluator_checkpoint_sha256": metadata["checkpoint_sha256"],
        "reference_signature": reference["signature"],
        "reference_path": str(Path(args.reference).resolve()),
        "calibrations": reference["calibrations"],
        "image_policy": reference.get("image_policy"),
        "mms": mms,
        "provenance": {"mode": "semantic_only",
                       "note": "quality metrics (FID/KID/IS/PR) not computed"},
    })
    print(json.dumps({
        "n": mms["n"], "candidate_count": mms["candidate_count"],
        "mms_value": mms["value"], "ci95_wilson": mms["ci95"],
        "mean_entropy": mms["mean_entropy"],
        "entropy_threshold_tau": reference["calibrations"]["entropy"]["threshold"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
