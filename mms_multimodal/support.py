"""Shared helpers for adapters built on the modality-agnostic core.

Only new adapters use this module. The frozen MNIST28/CRC100K/ESC-50 adapters
keep their own copies so their behavior cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mms_eval.semantic import wilson_interval
from mms_eval.utils import read_jsonl


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    probabilities = np.exp(z)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def ece_15_bins(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    bins = np.clip((confidence * 15).astype(int), 0, 14)
    gap = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            gap += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(gap)


def audit_metrics(labels, logits, classes: list[str]) -> dict:
    """Real-sample classification audit from raw [N,C] logits."""
    truth = np.asarray(labels, dtype=int)
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    correct = int((truth == predicted).sum())
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1
    by_class, recalls = {}, []
    for i, name in enumerate(classes):
        n = int((truth == i).sum())
        true_positive = int(matrix[i, i])
        false_positive = int(matrix[:, i].sum() - true_positive)
        false_negative = int(matrix[i, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        recalls.append(true_positive / n if n else 0.0)
        by_class[name] = {"n": n, "correct": true_positive,
                          "accuracy": true_positive / n if n else None,
                          "f1": 2 * true_positive / denominator if denominator else 0.0}
    return {
        "n": len(truth), "accuracy": correct / len(truth),
        "accuracy_wilson_95": wilson_interval(correct, len(truth)),
        "macro_f1": float(np.mean([v["f1"] for v in by_class.values()])),
        "mean_recall": float(np.mean(recalls)),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-300)).mean()),
        "ece_15_bins": ece_15_bins(probabilities, truth),
        "by_class": by_class,
        "confusion_matrix_true_x_predicted": matrix.tolist(),
    }


def fresh_output_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already contains files; use a fresh directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def verify_split_manifests(split_dir: str | Path, expected_name: str) -> tuple[Path, dict]:
    """Re-check the recorded manifest hashes of a prepared split directory."""
    root = Path(split_dir).resolve(strict=True)
    report = json.loads((root / "split.json").read_text(encoding="utf-8"))
    digests = report.get("manifest_sha256", {})
    if expected_name not in digests:
        raise ValueError(f"split.json does not record a '{expected_name}' manifest")
    for name, digest in digests.items():
        if hashlib.sha256((root / f"{name}.jsonl").read_bytes()).hexdigest() != digest:
            raise ValueError(f"Split manifest changed after preparation: {name}")
    return root, report


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_generated_texts(source: str | Path) -> list[dict]:
    """Read generated text samples from a JSONL manifest or a directory of .txt files.

    Manifest rows carry ``sample_id`` plus either inline ``text`` or a ``path``
    to a UTF-8 text file. Directories contribute one sample per sorted .txt
    file, identified by file stem. Content identity is always the SHA-256 of
    the exact UTF-8 text bytes.
    """
    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in generated directory: {path}")
        rows = []
        for file in files:
            rows.append({"sample_id": file.stem, "text": file.read_text(encoding="utf-8"),
                         "origin": str(file.resolve())})
    elif path.is_file():
        rows = read_jsonl(path)
        if not rows:
            raise ValueError("Generated text manifest is empty")
        for row in rows:
            if "path" in row:
                row["text"] = Path(row["path"]).resolve(strict=True).read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(path)
    records = []
    for row in rows:
        sample_id, text = row.get("sample_id"), row.get("text")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(text, str):
            raise ValueError("Each generated text needs a nonempty sample_id and text")
        if not text.strip():
            raise ValueError(f"Generated text is empty: {sample_id}")
        records.append({"sample_id": sample_id, "text": text,
                        "sha256": text_sha256(text)})
    return records
