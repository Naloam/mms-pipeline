"""ESC-50 spectrogram adapter for the shared MMS candidate score.

The generator emits grayscale mel-spectrogram PNGs, not audio waveforms. The
evaluator therefore works in that saved representation and makes no claim
about audible quality or verified mixtures of sound events.
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from mms_eval.evaluator import describe_checkpoint, predict_paths, train_evaluator
from mms_eval.semantic import wilson_interval
from mms_eval.utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl

from .core import evaluate_logits, freeze_reference


SPLIT_SEED = 2026091712
GENERATED_SAMPLE_SEED = 2026091713
EXPECTED_CLASSES = 50
EXPECTED_REAL = 2000
EXPECTED_GENERATED = 50000


def _image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (256, 256):
            raise ValueError(f"Expected 256x256 grayscale spectrogram: {path}")
    return sha256_file(path)


def _source(dataset_dir: str | Path, spectrogram_dir: str | Path) -> tuple[Path, list[str], list[dict]]:
    data = Path(dataset_dir).resolve(strict=True)
    spectra = Path(spectrogram_dir).resolve(strict=True)
    csv_path = data / "meta" / "esc50.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_REAL:
        raise ValueError(f"Expected {EXPECTED_REAL} official ESC-50 metadata rows")
    classes: list[str | None] = [None] * EXPECTED_CLASSES
    counts = collections.Counter()
    seen_names = set()
    records = []
    for row in rows:
        name = row["filename"]
        label, fold = int(row["target"]), int(row["fold"])
        if (Path(name).name != name or name in seen_names or
                not 0 <= label < EXPECTED_CLASSES or not 1 <= fold <= 5 or
                not name.startswith(f"{fold}-")):
            raise ValueError(f"Invalid ESC-50 metadata identity: {name}")
        if classes[label] not in (None, row["category"]):
            raise ValueError(f"Class {label} has inconsistent names")
        classes[label] = row["category"]
        wav = data / "audio" / name
        if not wav.is_file():
            raise FileNotFoundError(wav)
        png = spectra / f"{Path(name).stem}.png"
        records.append({
            "image_id": f"ESC-50:{name}", "path": str(png), "label": label,
            "sha256": _image(png), "fold": fold, "source_file": row["src_file"],
        })
        counts[(fold, label)] += 1
        seen_names.add(name)
    if any(name is None for name in classes) or any(counts[(fold, label)] != 8
            for fold in range(1, 6) for label in range(EXPECTED_CLASSES)):
        raise ValueError("ESC-50 official five-fold class balance changed")
    if len({r["sha256"] for r in records}) != EXPECTED_REAL:
        raise ValueError("Real spectrogram file content is duplicated")
    if len(list(spectra.glob("*.png"))) != EXPECTED_REAL:
        raise ValueError("Spectrogram directory must contain exactly 2,000 PNGs")
    return csv_path, [str(name) for name in classes], sorted(records, key=lambda r: r["image_id"])


def prepare(dataset_dir: str | Path, spectrogram_dir: str | Path,
            out_dir: str | Path, *, seed: int = SPLIT_SEED) -> dict:
    """Use folds 1-3 to fit, half of fold 4 for selection/calibration, fold 5 for audit."""
    csv_path, classes, records = _source(dataset_dir, spectrogram_dir)
    fit = [r for r in records if r["fold"] in (1, 2, 3)]
    audit = [r for r in records if r["fold"] == 5]
    audit_sources = {r["source_file"] for r in audit}
    quarantine = [r for r in records if r["fold"] == 4 and r["source_file"] in audit_sources]
    if len(quarantine) != 2:
        raise ValueError(f"Expected two cross-fold source collisions, found {len(quarantine)}")
    validation, calibration = [], []
    rng = np.random.default_rng(seed)
    for label in range(EXPECTED_CLASSES):
        group = [r for r in records if r["fold"] == 4 and r["label"] == label
                 and r["source_file"] not in audit_sources]
        order = rng.permutation(len(group))
        validation.extend(group[int(i)] for i in order[:4])
        calibration.extend(group[int(i)] for i in order[4:])
    parts = {name: sorted(part, key=lambda r: r["image_id"]) for name, part in
             (("fit", fit), ("validation", validation), ("calibration", calibration),
              ("audit", audit), ("quarantine", quarantine))}
    if [len(parts[k]) for k in parts] != [1200, 200, 198, 400, 2]:
        raise AssertionError("ESC-50 split count error")
    if len({r["image_id"] for part in parts.values() for r in part}) != EXPECTED_REAL:
        raise AssertionError("ESC-50 split identity overlap")
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh ESC-50 split directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        write_jsonl(out / f"{name}.jsonl", part)
    report = {
        "dataset": "official ESC-50 five-fold metadata and saved 256x256 grayscale mel spectrograms",
        "metadata_csv_sha256": sha256_file(csv_path),
        "classes": classes, "seed": seed,
        "fold_roles": {"fit": [1, 2, 3], "checkpoint_selection": [4],
                       "real_calibration": [4], "untouched_audit": [5],
                       "cross_fold_source_quarantine": [4]},
        "counts": {name: len(part) for name, part in parts.items()},
        "manifest_sha256": {name: sha256_file(out / f"{name}.jsonl") for name in parts},
        "fold5_source_files_disjoint_from_fit_and_used_fold4": not (
            {r["source_file"] for r in audit} &
            {r["source_file"] for name in ("fit", "validation", "calibration") for r in parts[name]}),
        "quarantined_images": [r["image_id"] for r in parts["quarantine"]],
        "validation_calibration_source_overlap": len(
            {r["source_file"] for r in validation} & {r["source_file"] for r in calibration}),
        "limitation": "Generator trained on all 2,000 real spectrograms; classifier audit is held out from evaluator fitting, not from GAN fitting.",
    }
    write_json(out / "split.json", report)
    return report


def _split(split_dir: str | Path) -> tuple[Path, dict]:
    root = Path(split_dir).resolve(strict=True)
    report = read_json(root / "split.json")
    for name, digest in report["manifest_sha256"].items():
        if sha256_file(root / f"{name}.jsonl") != digest:
            raise ValueError(f"ESC-50 {name} manifest changed")
    if report["counts"] != {"fit": 1200, "validation": 200, "calibration": 198,
                            "audit": 400, "quarantine": 2}:
        raise ValueError("ESC-50 split count changed")
    if not report["fold5_source_files_disjoint_from_fit_and_used_fold4"]:
        raise ValueError("ESC-50 audit source overlaps training/selection/calibration")
    return root, report


def train_classifier(split_dir: str | Path, out_dir: str | Path, *, config: dict) -> dict:
    root, split = _split(split_dir)
    if config.get("classes") != split["classes"] or config.get("horizontal_flip_probability") != 0:
        raise ValueError("Evaluator classes must match ESC-50 and time-axis flips must be disabled")
    return train_evaluator(read_jsonl(root / "fit.jsonl"), read_jsonl(root / "validation.jsonl"),
                           out_dir, config=config)


def _profile(classes: list[str], split: dict) -> dict:
    return {
        "protocol_id": "ESC50_mel_spectrogram_50class_entropy_v1", "modality": "audio",
        "dataset_version": f"ESC-50 official metadata SHA256:{split['metadata_csv_sha256']}",
        "category_mode": "exclusive", "classes": classes,
        "semantic_definition": (
            "A saved spectrogram is assigned one dominant ESC-50 sound-event class. "
            "High classifier uncertainty is only a candidate for cross-class ambiguity; "
            "neither a mixed-sound label nor an audible-quality judgment."
        ),
        "preprocessing": {
            "source": "44.1 kHz five-second ESC-50 WAVs converted by existing esc50_spec.py",
            "real_and_generated": "256x256 grayscale PNG; loaded as replicated RGB, bilinear resized 224x224, ImageNet normalized",
            "time_flip": False,
            "representation_limit": "GAN emits spectrogram PNGs, not audio waveforms",
        },
        "temperature": 1.0,
    }


def _predict(rows: list[dict], checkpoint: Path, device: str, batch_size: int):
    return predict_paths([r["path"] for r in rows], checkpoint,
                         device=device, batch_size=batch_size)["logits"]


def build_reference(checkpoint_path: str | Path, split_dir: str | Path, out_dir: str | Path,
                    *, device: str = "cuda:0", batch_size: int = 32, alpha: float = .05) -> dict:
    root, split = _split(split_dir)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    metadata = describe_checkpoint(checkpoint)
    if metadata["classes"] != split["classes"]:
        raise ValueError("Checkpoint class order changed")
    calibration = read_jsonl(root / "calibration.jsonl")
    audit = read_jsonl(root / "audit.jsonl")
    cal_logits = _predict(calibration, checkpoint, device, batch_size)
    audit_logits = _predict(audit, checkpoint, device, batch_size)
    identities = lambda rows: [
        {"sample_id": r["image_id"], "sha256": r["sha256"], "label": r["label"]} for r in rows]
    ref = freeze_reference(
        _profile(split["classes"], split), metadata["checkpoint_sha256"],
        identities(calibration), cal_logits, identities(audit), audit_logits,
        out_dir, alpha=alpha, source_hashes={"metadata_csv": split["metadata_csv_sha256"],
                                             "split_manifest": sha256_file(root / "split.json")})
    labels = np.asarray([r["label"] for r in audit])
    shifted = np.asarray(audit_logits, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    correct = int((labels == predicted).sum())
    per_class = {}
    f1 = []
    for i, name in enumerate(split["classes"]):
        true_positive = int(((labels == i) & (predicted == i)).sum())
        false_positive = int(((labels != i) & (predicted == i)).sum())
        false_negative = int(((labels == i) & (predicted != i)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
        per_class[name] = {"n": int((labels == i).sum()),
                           "correct": true_positive,
                           "accuracy": true_positive / int((labels == i).sum())}
    write_json(Path(out_dir) / "classifier_audit.json", {
        "split_role": "ESC-50 fold 5 held out from classifier fitting, checkpoint selection and threshold calibration",
        "n": len(audit), "accuracy": correct / len(audit),
        "accuracy_wilson_95": wilson_interval(correct, len(audit)),
        "macro_f1": float(np.mean(f1)),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(audit)), labels], 1e-300)).mean()),
        "by_class": per_class,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "reference_signature": ref["signature"],
        "gan_training_overlap": "All original ESC-50 spectrograms including fold 5 were used to train the GAN",
    })
    return ref


def selected_generated(generated_dir: str | Path, count: int = 5000,
                       seed: int = GENERATED_SAMPLE_SEED,
                       expected_total: int | None = None,
                       source_id: str | None = None) -> list[dict]:
    root = Path(generated_dir).resolve(strict=True)
    files = sorted(root.glob("*.png"))
    if expected_total is not None and len(files) != expected_total:
        raise ValueError(f"Expected {expected_total} generated PNGs, found {len(files)}")
    if not 0 < count <= len(files):
        raise ValueError("Generated selection count out of range")
    if source_id is None:
        source_id = root.name
    if not source_id or any(char in source_id for char in ("/", "\\", ":")):
        raise ValueError("source_id must be a nonempty path-safe identifier")
    indices = sorted(np.random.default_rng(seed).choice(len(files), size=count, replace=False).tolist())
    rows = []
    for index in indices:
        path = files[index]
        rows.append({"sample_id": f"ESC50-GAN:{source_id}:{path.name}", "path": str(path),
                     "sha256": _image(path)})
    return rows


def evaluate_generated(checkpoint_path: str | Path, reference_path: str | Path,
                       split_dir: str | Path, generated_dir: str | Path, out_dir: str | Path,
                       *, count: int = 5000, device: str = "cuda:0", batch_size: int = 32,
                       source_id: str | None = None, expected_total: int | None = None,
                       sample_seed: int = GENERATED_SAMPLE_SEED,
                       generator_checkpoint: str | Path | None = None) -> dict:
    _, split = _split(split_dir)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    metadata = describe_checkpoint(checkpoint)
    if metadata["classes"] != split["classes"]:
        raise ValueError("Checkpoint class order changed")
    rows = selected_generated(generated_dir, count=count, seed=sample_seed,
                              expected_total=expected_total, source_id=source_id)
    logits = _predict(rows, checkpoint, device, batch_size)
    provenance = {
        "representation": "saved 256x256 grayscale mel-spectrogram PNGs",
        "generated_dir": str(Path(generated_dir).resolve(strict=True)),
        "source_id": source_id or Path(generated_dir).name,
        "sampling_seed": sample_seed, "selection_count": count,
        "full_saved_cohort_count": len(list(Path(generated_dir).glob("*.png"))),
        "selected_identity_sha256": stable_hash([(r["sample_id"], r["sha256"]) for r in rows]),
    }
    if generator_checkpoint is not None:
        provenance["generator_checkpoint_sha256"] = sha256_file(generator_checkpoint)
    return evaluate_logits(
        _profile(split["classes"], split), metadata["checkpoint_sha256"],
        [{"sample_id": r["sample_id"], "sha256": r["sha256"]} for r in rows],
        logits, reference_path, out_dir,
        provenance=provenance,
    )
