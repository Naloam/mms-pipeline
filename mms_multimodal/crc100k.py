"""CRC100K tissue classifier inputs and frozen semantic reference utilities.

The source GAN and histology files are read only. All splits, checkpoints and
scores belong in a separate output directory selected by the caller.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from mms_eval.evaluator import describe_checkpoint, predict_paths, train_evaluator
from mms_eval.utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl

from .core import evaluate_logits, freeze_reference


CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TRAIN_SOURCE = "NCT-CRC-HE-100K Macenko-normalized PNG, 224x224, nine tissue labels"
VALIDATION_SOURCE = "CRC-VAL-HE-7K official Zenodo release, independent patients"
SPLIT_SEED = 2026091707
EXTERNAL_SPLIT_SEED = 2026091708
GENERATED_SAMPLE_SEED = 2026091709
QUALITY_REAL_SEED = 2026091710


def _validate_train_source(raw_dir: str | Path) -> tuple[Path, list[dict]]:
    root = Path(raw_dir).resolve(strict=True)
    metadata = read_json(root / "dataset.json")
    labels = metadata.get("labels")
    if not isinstance(labels, list) or len(labels) != 100000:
        raise ValueError("Expected exactly 100,000 labeled CRC training images")
    rows, seen = [], set()
    for name, label in labels:
        if (not isinstance(name, str) or Path(name).name != name or name in seen or
                not isinstance(label, int) or isinstance(label, bool) or
                not 0 <= label < len(CLASSES) or name.split("-", 1)[0] != CLASSES[label]):
            raise ValueError(f"Invalid CRC training label: {name!r}, {label!r}")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        rows.append({"image_id": f"NCT-CRC-HE-100K:{name}", "path": str(path), "label": label})
    rows.sort(key=lambda row: row["image_id"])
    return root, rows


def prepare_classifier_split(raw_dir: str | Path, out_dir: str | Path,
                             *, seed: int = SPLIT_SEED, validation_fraction: float = .10) -> dict:
    """Freeze a stratified internal split for checkpoint selection only.

    A random tile split may share patients between fit and validation, so the
    official independent-patient CRC-VAL-HE-7K set is reserved for audit.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    root, rows = _validate_train_source(raw_dir)
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new split directory: {out}")
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for label in range(len(CLASSES)):
        subset = [row for row in rows if row["label"] == label]
        order = rng.permutation(len(subset))
        n_validation = round(len(subset) * validation_fraction)
        selected = set(order[:n_validation].tolist())
        for i, row in enumerate(subset):
            (validation if i in selected else fit).append(row)
    fit.sort(key=lambda row: row["image_id"])
    validation.sort(key=lambda row: row["image_id"])
    report = {
        "dataset": TRAIN_SOURCE, "source_dataset_json_sha256": sha256_file(root / "dataset.json"),
        "classes": CLASSES, "seed": seed, "validation_fraction": validation_fraction,
        "split_role": "internal_checkpoint_selection_only; possible shared patients",
        "fit_count": len(fit), "validation_count": len(validation),
        "fit_by_class": dict(sorted(collections.Counter(CLASSES[r["label"]] for r in fit).items())),
        "validation_by_class": dict(sorted(collections.Counter(CLASSES[r["label"]] for r in validation).items())),
        "fit_identity_sha256": stable_hash([(r["image_id"], r["label"]) for r in fit]),
        "validation_identity_sha256": stable_hash([(r["image_id"], r["label"]) for r in validation]),
    }
    out.mkdir(parents=True)
    write_jsonl(out / "fit.jsonl", fit)
    write_jsonl(out / "validation.jsonl", validation)
    report["fit_manifest_sha256"] = sha256_file(out / "fit.jsonl")
    report["validation_manifest_sha256"] = sha256_file(out / "validation.jsonl")
    write_json(out / "split.json", report)
    return report


def train_classifier(split_dir: str | Path, out_dir: str | Path, *, config: dict) -> dict:
    split = Path(split_dir).resolve(strict=True)
    metadata = read_json(split / "split.json")
    if metadata["classes"] != CLASSES:
        raise ValueError("CRC class order changed")
    for part in ("fit", "validation"):
        if sha256_file(split / f"{part}.jsonl") != metadata[f"{part}_manifest_sha256"]:
            raise ValueError(f"CRC {part} manifest changed")
    if config.get("classes") != CLASSES:
        raise ValueError("Evaluator class order must match the dataset")
    return train_evaluator(read_jsonl(split / "fit.jsonl"),
                           read_jsonl(split / "validation.jsonl"), out_dir, config=config)


def _profile(batch_size: int) -> dict:
    return {
        "protocol_id": "CRC100K_nine_tissue_entropy_v1", "modality": "image",
        "dataset_version": f"{TRAIN_SOURCE}; {VALIDATION_SOURCE}",
        "category_mode": "exclusive", "classes": CLASSES,
        "semantic_definition": (
            "One dominant named tissue class per patch. High classifier uncertainty is a "
            "candidate for cross-class ambiguity, not proof of mixed tissue or pathology."
        ),
        "preprocessing": {
            "real": "official 224x224 RGB patches", "generated": "saved 256x256 RGB JPEG",
            "classifier": "RGB whole-image bilinear resize to 224x224 and ImageNet normalization",
            "inference_batch_size": batch_size,
        },
        "temperature": 1.0,
    }


def _class_records(folder: str | Path) -> list[dict]:
    root = Path(folder).resolve(strict=True)
    rows = []
    for label, class_name in enumerate(CLASSES):
        matches = sorted(root.rglob(f"{class_name}-*.tif"))
        matches += sorted(root.rglob(f"{class_name}-*.tiff"))
        if not matches:
            raise ValueError(f"Missing {class_name} official validation images under {root}")
        for path in matches:
            rows.append({"image_id": f"CRC-VAL-HE-7K:{path.name}", "path": str(path),
                         "label": label, "sha256": sha256_file(path)})
    if len(rows) != 7180 or len({row["image_id"] for row in rows}) != 7180:
        raise ValueError(f"Expected 7,180 distinct official validation images, found {len(rows)}")
    return sorted(rows, key=lambda row: row["image_id"])


def _external_split(rows: list[dict], seed: int = EXTERNAL_SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    calibration, audit = [], []
    for label in range(len(CLASSES)):
        group = [row for row in rows if row["label"] == label]
        order = rng.permutation(len(group))
        selected = set(order[:len(group) // 2].tolist())
        for i, row in enumerate(group):
            (calibration if i in selected else audit).append(row)
    return (sorted(calibration, key=lambda row: row["image_id"]),
            sorted(audit, key=lambda row: row["image_id"]))


def build_reference(classifier_checkpoint: str | Path, official_validation_dir: str | Path,
                    out_dir: str | Path, *, device: str = "cuda:0", batch_size: int = 32,
                    alpha: float = .05, official_zip_md5: str | None = None) -> dict:
    if official_zip_md5 != "2fd1651b4f94ebd818ebf90ad2b6ce06":
        raise ValueError("The official CRC-VAL-HE-7K archive MD5 must be verified first")
    checkpoint = Path(classifier_checkpoint).resolve(strict=True)
    metadata = describe_checkpoint(checkpoint)
    if metadata["classes"] != CLASSES:
        raise ValueError("Expected a nine-class CRC evaluator")
    all_rows = _class_records(official_validation_dir)
    calibration, audit = _external_split(all_rows)
    from mms_eval.evaluator import PredictionSession
    session = PredictionSession()
    cal_logits = predict_paths([row["path"] for row in calibration], checkpoint,
                               device=device, batch_size=batch_size, session=session)["logits"]
    audit_logits = predict_paths([row["path"] for row in audit], checkpoint,
                                 device=device, batch_size=batch_size, session=session)["logits"]
    output = Path(out_dir).resolve()
    ref = freeze_reference(
        _profile(batch_size), metadata["checkpoint_sha256"],
        [{"sample_id": r["image_id"], "sha256": r["sha256"], "label": r["label"]} for r in calibration],
        cal_logits,
        [{"sample_id": r["image_id"], "sha256": r["sha256"], "label": r["label"]} for r in audit],
        audit_logits, output, alpha=alpha,
        source_hashes={"official_crc_val_zip_md5": official_zip_md5},
    )
    write_json(output / "external_split.json", {
        "source": VALIDATION_SOURCE, "seed": EXTERNAL_SPLIT_SEED,
        "calibration_count": len(calibration), "audit_count": len(audit),
        "calibration_by_class": dict(collections.Counter(CLASSES[r["label"]] for r in calibration)),
        "audit_by_class": dict(collections.Counter(CLASSES[r["label"]] for r in audit)),
        "calibration_ids_sha256": stable_hash([r["image_id"] for r in calibration]),
        "audit_ids_sha256": stable_hash([r["image_id"] for r in audit]),
    })
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from scipy.special import softmax
    labels = np.asarray([row["label"] for row in audit])
    probabilities = softmax(audit_logits, axis=1)
    write_json(output / "classifier_audit.json", {
        "official_independent_patient_set": VALIDATION_SOURCE,
        "split_role": "audit half untouched by classifier training and threshold selection",
        "n": len(audit), "accuracy": float(accuracy_score(labels, probabilities.argmax(axis=1))),
        "macro_f1": float(f1_score(labels, probabilities.argmax(axis=1), average="macro")),
        "nll": float(log_loss(labels, probabilities, labels=list(range(len(CLASSES))))),
        "by_class": {name: {"n": int((labels == i).sum()),
                             "accuracy": float((probabilities.argmax(axis=1)[labels == i] == i).mean())}
                     for i, name in enumerate(CLASSES)},
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "reference_signature": ref["signature"],
    })
    return ref


def selected_generated(generated_dir: str | Path, count: int = 5000,
                       seed: int = GENERATED_SAMPLE_SEED) -> list[dict]:
    root = Path(generated_dir).resolve(strict=True)
    info = read_json(root / "GEN_INFO.json")
    if info.get("n_written") != 50000:
        raise ValueError("Expected the complete saved 50,000-image GAN cohort")
    if not 0 < count <= 50000:
        raise ValueError("Generated selection count out of range")
    indices = sorted(np.random.default_rng(seed).choice(50000, size=count, replace=False).tolist())
    rows = []
    for i in indices:
        path = root / f"{i // 1000:05d}" / f"{i:06d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"sample_id": f"CRC-GAN-seed:{i}", "image_id": f"CRC-GAN-seed:{i}",
                     "path": str(path), "sha256": sha256_file(path), "setting_id": "StyleGAN2-ADA_CRC100K"})
    return rows


def prepare_quality_inputs(raw_dir: str | Path, generated_dir: str | Path,
                           out_dir: str | Path, *, real_count: int = 10000,
                           generated_count: int = 5000) -> dict:
    """Freeze saved-image and training-source cohorts before quality scoring."""
    if not 0 < real_count <= 100000:
        raise ValueError("Real quality cohort size out of range")
    root, real_pool = _validate_train_source(raw_dir)
    selected = sorted(np.random.default_rng(QUALITY_REAL_SEED).choice(
        len(real_pool), size=real_count, replace=False).tolist())
    real = [{**real_pool[i], "sha256": sha256_file(real_pool[i]["path"])} for i in selected]
    generated = selected_generated(generated_dir, count=generated_count)
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new quality-input directory: {out}")
    out.mkdir(parents=True)
    write_jsonl(out / "real_manifest.jsonl", real)
    write_jsonl(out / "generated_manifest.jsonl", generated)
    report = {
        "dataset": TRAIN_SOURCE,
        "source_dataset_json_sha256": sha256_file(root / "dataset.json"),
        "generated_info_sha256": sha256_file(Path(generated_dir) / "GEN_INFO.json"),
        "real_sampling_seed": QUALITY_REAL_SEED,
        "generated_sampling_seed": GENERATED_SAMPLE_SEED,
        "real_count": len(real), "generated_count": len(generated),
        "real_by_class": dict(sorted(collections.Counter(CLASSES[r["label"]] for r in real).items())),
        "real_manifest_sha256": sha256_file(out / "real_manifest.jsonl"),
        "generated_manifest_sha256": sha256_file(out / "generated_manifest.jsonl"),
        "interpretation": "The real quality reference is sampled from the generator's training source.",
    }
    write_json(out / "selection.json", report)
    return report


def evaluate_generated(classifier_checkpoint: str | Path, reference_path: str | Path,
                       generated_dir: str | Path, out_dir: str | Path,
                       *, device: str = "cuda:0", batch_size: int = 32,
                       count: int = 5000) -> dict:
    checkpoint = Path(classifier_checkpoint).resolve(strict=True)
    metadata = describe_checkpoint(checkpoint)
    rows = selected_generated(generated_dir, count=count)
    logits = predict_paths([row["path"] for row in rows], checkpoint,
                           device=device, batch_size=batch_size)["logits"]
    return evaluate_logits(
        _profile(batch_size), metadata["checkpoint_sha256"],
        [{"sample_id": row["sample_id"], "sha256": row["sha256"]} for row in rows],
        logits, reference_path, out_dir,
        provenance={"generator": read_json(Path(generated_dir) / "GEN_INFO.json"),
                    "selection_seed": GENERATED_SAMPLE_SEED, "selection_count": count,
                    "selected_identity_sha256": stable_hash([(r["sample_id"], r["sha256"]) for r in rows])},
    )
