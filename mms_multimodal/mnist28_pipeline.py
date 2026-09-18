"""MNIST adapter for the shared calibrated semantic scoring module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from mms_eval.utils import read_json, read_jsonl, sha256_file

from .core import evaluate_logits, freeze_reference
from .mnist28 import CLASSES, NORMALIZATION, predict_logits, read_idx_gzip


def _profile(batch_size: int) -> dict:
    return {
        "protocol_id": "MNIST28_digit_entropy_clean_v1", "modality": "image",
        "dataset_version": "MNIST official 60k train / 10k test IDX; source hashes in reference",
        "category_mode": "exclusive", "classes": CLASSES,
        "semantic_definition": "One intended digit 0–9 per image; an uncertain digit is a candidate, not confirmed mixed semantics",
        "preprocessing": {"size": [28, 28], "mode": "L", "normalization": NORMALIZATION,
                          "inference_batch_size": batch_size},
        "temperature": 1.0,
    }


def _verify_training(data_dir: str | Path, classifier_dir: str | Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    source, classifier = Path(data_dir).resolve(), Path(classifier_dir).resolve()
    report = read_json(classifier / "report.json")
    split = read_json(classifier / "split.json")
    if report["status"] != "trained_and_tested" or report["classes"] != CLASSES:
        raise ValueError("Expected the frozen ten-class MNIST evaluator")
    if sha256_file(classifier / "best.pt") != report["checkpoint_sha256"] or \
            sha256_file(classifier / "split.json") != report["split_sha256"] or \
            sha256_file(classifier / "test_logits.npy") != report["test_logits_sha256"]:
        raise ValueError("Classifier checkpoint/split/logits differ from the training report")
    for name, expected in report["source_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"MNIST source file changed: {name}")
    images = read_idx_gzip(source / "t10k-images-idx3-ubyte.gz", images=True)
    labels = read_idx_gzip(source / "t10k-labels-idx1-ubyte.gz", images=False)
    calibration = np.asarray(split["calibration_test_indices"], dtype=int)
    audit = np.asarray(split["audit_test_indices"], dtype=int)
    if (len(calibration) != 5000 or len(audit) != 5000 or
            set(calibration) & set(audit) or set(np.r_[calibration, audit]) != set(range(10000))):
        raise ValueError("Official test calibration and audit partitions are not a 5k/5k disjoint cover")
    return report, split, images, labels


def _real_records(images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> list[dict]:
    return [{"sample_id": f"MNIST-official-test:{int(i):05d}",
             "sha256": hashlib.sha256(images[i].tobytes()).hexdigest(), "label": int(labels[i])}
            for i in indices]


def build_reference(data_dir: str | Path, classifier_dir: str | Path, output_dir: str | Path,
                    *, device: str = "cuda:0", batch_size: int = 256, alpha: float = .05) -> dict:
    report, split, images, labels = _verify_training(data_dir, classifier_dir)
    classifier = Path(classifier_dir).resolve()
    if batch_size != 256:
        raise ValueError("This frozen classifier test-logit protocol uses inference_batch_size=256")
    stored_logits = np.load(classifier / "test_logits.npy", allow_pickle=False)
    recomputed = predict_logits(classifier / "best.pt", images, device=device, batch_size=batch_size)
    if not np.array_equal(recomputed, stored_logits):
        raise ValueError("Reloaded checkpoint does not reproduce saved official-test logits")
    calibration = np.asarray(split["calibration_test_indices"], dtype=int)
    audit = np.asarray(split["audit_test_indices"], dtype=int)
    return freeze_reference(
        _profile(batch_size), report["checkpoint_sha256"],
        _real_records(images, labels, calibration), stored_logits[calibration],
        _real_records(images, labels, audit), stored_logits[audit],
        output_dir, alpha=alpha,
        source_hashes={**report["source_sha256"], "classifier_split_sha256": report["split_sha256"],
                       "classifier_test_logits_sha256": report["test_logits_sha256"]},
    )


def evaluate_generated(classifier_dir: str | Path, reference_path: str | Path,
                       generated_manifest: str | Path, output_dir: str | Path,
                       *, device: str = "cuda:0", batch_size: int = 256) -> dict:
    classifier = Path(classifier_dir).resolve()
    report = read_json(classifier / "report.json")
    if sha256_file(classifier / "best.pt") != report["checkpoint_sha256"]:
        raise ValueError("Evaluator checkpoint changed")
    manifest = Path(generated_manifest).resolve()
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("Generated manifest is empty")
    images, records = [], []
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Generated file changed: {path}")
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (28, 28):
                raise ValueError(f"Expected native 28x28 L image: {path}")
            pixels = np.asarray(image, dtype=np.uint8).copy()
        images.append(pixels)
        records.append({"sample_id": row["image_id"], "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    logits = predict_logits(classifier / "best.pt", np.stack(images), device=device, batch_size=batch_size)
    return evaluate_logits(
        _profile(batch_size), report["checkpoint_sha256"], records, logits,
        reference_path, output_dir,
        provenance={"generated_manifest_sha256": sha256_file(manifest),
                    "generated_setting_ids": sorted({str(row.get("setting_id", "")) for row in rows}),
                    "image_file_sha256_verified": True},
    )
