"""One scoring and real-reference calibration contract for every modality.

Adapters provide identified samples and raw [N,C] classifier logits. They own
text/audio/image decoding and any modality-specific quality metrics. This
module owns only the shared semantic candidate rule and its audit trail.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from mms_eval.semantic import (MMS_ALGORITHM_VERSION, SCORE_NAMES, aggregate_mms,
                               apply_threshold, posterior_scores, rank_calibrate)
from mms_eval.utils import read_json, sha256_file, stable_hash, write_json, write_jsonl


SCHEMA_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _profile(profile: dict) -> dict:
    if not isinstance(profile, dict):
        raise ValueError("Task profile must be an object")
    required = ("protocol_id", "modality", "dataset_version", "classes", "semantic_definition", "preprocessing")
    if any(not profile.get(key) for key in required):
        raise ValueError(f"Task profile requires {required}")
    if profile["modality"] not in ("image", "text", "audio"):
        raise ValueError("modality must be image, text, or audio")
    classes = profile["classes"]
    if (not isinstance(classes, list) or len(classes) < 2 or
            any(not isinstance(name, str) or not name for name in classes) or len(set(classes)) != len(classes)):
        raise ValueError("MMS requires distinct, mutually exclusive class names")
    if profile.get("category_mode") != "exclusive":
        raise ValueError("Current entropy MMS applies only to exclusive classes")
    return profile


def _records(records: list[dict], *, real: bool, class_count: int) -> list[dict]:
    if not records:
        raise ValueError("Sample records cannot be empty")
    seen_ids, seen_hashes, identities = set(), set(), []
    for record in records:
        sample_id, content_sha = record.get("sample_id"), record.get("sha256")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(content_sha, str) or not _HASH.fullmatch(content_sha):
            raise ValueError("Each sample needs a nonempty sample_id and SHA-256")
        if sample_id in seen_ids or (real and content_sha in seen_hashes):
            raise ValueError(f"Duplicate sample identity/content: {sample_id}")
        seen_ids.add(sample_id)
        seen_hashes.add(content_sha)
        identity = {"sample_id": sample_id, "sha256": content_sha}
        if real:
            label = record.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < class_count:
                raise ValueError(f"Real sample {sample_id} needs a class label")
            identity["label"] = label
        identities.append(identity)
    return identities


def _logits(logits, n: int, class_count: int) -> np.ndarray:
    array = np.asarray(logits)
    if array.shape != (n, class_count) or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"Expected finite logits of shape ({n}, {class_count})")
    return np.asarray(array, dtype="<f4", order="C")


def _array_hash(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _core_hash() -> str:
    return sha256_file(__file__)


def _summary(scores: dict, calibrations: dict, classes: list[str]) -> dict:
    summary = aggregate_mms(scores, calibrations, classes)
    summary["ci_assumptions"] = "fixed evaluator and reference; independent sample draws assumed; not semantic truth"
    if summary["status"] == "single_image_only":
        summary["status"] = "single_sample_only"
        summary["reason"] = "One sample receives a candidate flag, not a model prevalence claim"
    return summary


def freeze_reference(profile: dict, checkpoint_sha256: str, calibration_records: list[dict],
                     calibration_logits, audit_records: list[dict], audit_logits,
                     output_dir: str | Path, *, alpha: float = .05, source_hashes: dict | None = None) -> dict:
    """Freeze real-sample thresholds and report a disjoint real-sample audit."""
    profile = _profile(profile)
    if not _HASH.fullmatch(checkpoint_sha256):
        raise ValueError("A frozen evaluator checkpoint SHA-256 is required")
    classes = profile["classes"]
    calibration = _records(calibration_records, real=True, class_count=len(classes))
    audit = _records(audit_records, real=True, class_count=len(classes))
    if ({row["sample_id"] for row in calibration} & {row["sample_id"] for row in audit} or
            {row["sha256"] for row in calibration} & {row["sha256"] for row in audit}):
        raise ValueError("Calibration and audit real samples overlap")
    cal_logits = _logits(calibration_logits, len(calibration), len(classes))
    audit_logits = _logits(audit_logits, len(audit), len(classes))
    cal_scores = posterior_scores(cal_logits, float(profile.get("temperature", 1)))
    calibrations = {name: rank_calibrate(cal_scores[name], alpha) for name in SCORE_NAMES}
    audit_scores = posterior_scores(audit_logits, float(profile.get("temperature", 1)))
    real_audit = _summary(audit_scores, calibrations, classes)
    labels = np.asarray([row["label"] for row in audit])
    real_audit["classification_accuracy"] = float((audit_scores["class_order"][:, 0] == labels).mean())
    real_audit["interpretation"] = "Candidate rate on untouched real samples; not a human-MM false-positive guarantee"
    identities = {"calibration": calibration, "audit": audit}
    contract = {
        "schema_version": SCHEMA_VERSION, "profile": profile, "checkpoint_sha256": checkpoint_sha256,
        "score_algorithm": MMS_ALGORITHM_VERSION, "core_sha256": _core_hash(),
        "calibration_logits_sha256": _array_hash(cal_logits), "audit_logits_sha256": _array_hash(audit_logits),
        "real_identities_sha256": stable_hash(identities), "source_hashes": source_hashes or {},
        "alpha": alpha,
    }
    signature = stable_hash(contract)
    out = Path(output_dir).resolve()
    ref_path = out / "reference.json"
    if out.exists() and any(out.iterdir()):
        prior = _load_reference(ref_path)[0] if ref_path.exists() else None
        if prior is None or prior.get("signature") != signature:
            raise FileExistsError("Reference output already differs; use a new directory")
        return prior
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "calibration_ids.jsonl", calibration)
    write_jsonl(out / "audit_ids.jsonl", audit)
    reference = {
        **contract, "signature": signature, "classes": classes,
        "calibrations": calibrations, "real_audit": real_audit,
        "counts": {"calibration": len(calibration), "audit": len(audit)},
        "identity_files": {"calibration": "calibration_ids.jsonl", "audit": "audit_ids.jsonl"},
        "identity_file_sha256": {"calibration": sha256_file(out / "calibration_ids.jsonl"),
                                 "audit": sha256_file(out / "audit_ids.jsonl")},
        "semantic_status": "human_confusion_validity_pending",
    }
    reference["integrity_sha256"] = stable_hash(reference)
    write_json(ref_path, reference)
    return reference


def _load_reference(path: str | Path) -> tuple[dict, Path]:
    ref_path = Path(path).resolve()
    ref = read_json(ref_path)
    integrity = ref.pop("integrity_sha256", None)
    if integrity != stable_hash(ref):
        raise ValueError("Reference integrity SHA-256 mismatch")
    ref["integrity_sha256"] = integrity
    if ref["core_sha256"] != _core_hash() or ref["score_algorithm"] != MMS_ALGORITHM_VERSION:
        raise ValueError("Scoring code differs from the frozen reference")
    for part, name in ref["identity_files"].items():
        if sha256_file(ref_path.parent / name) != ref["identity_file_sha256"][part]:
            raise ValueError(f"Frozen {part} identities were changed")
    return ref, ref_path


def evaluate_logits(profile: dict, checkpoint_sha256: str, generated_records: list[dict],
                    logits, reference_path: str | Path, output_dir: str | Path,
                    *, provenance: dict | None = None) -> dict:
    """Evaluate modality-agnostic generated logits against a frozen real reference."""
    profile = _profile(profile)
    reference, ref_path = _load_reference(reference_path)
    if (stable_hash(profile) != stable_hash(reference["profile"]) or
            checkpoint_sha256 != reference["checkpoint_sha256"]):
        raise ValueError("Task profile/evaluator differs from the frozen reference")
    records = _records(generated_records, real=False, class_count=len(profile["classes"]))
    real_hashes, real_ids = set(), set()
    for name in reference["identity_files"].values():
        from mms_eval.utils import read_jsonl
        for row in read_jsonl(ref_path.parent / name):
            real_hashes.add(row["sha256"])
            real_ids.add(row["sample_id"])
    if ({row["sample_id"] for row in records} & real_ids or
            {row["sha256"] for row in records} & real_hashes):
        raise ValueError("Generated samples overlap frozen real calibration/audit samples")
    array = _logits(logits, len(records), len(profile["classes"]))
    score = posterior_scores(array, float(profile.get("temperature", 1)))
    summary = _summary(score, reference["calibrations"], profile["classes"])
    summary["duplicate_content_count"] = len(records) - len({row["sha256"] for row in records})
    flags = {name: apply_threshold(score[name], reference["calibrations"][name]) for name in SCORE_NAMES}
    rows = []
    for i, record in enumerate(records):
        order = score["class_order"][i]
        row = {**record, "top1_class": profile["classes"][order[0]],
               "top2_class": profile["classes"][order[1]],
               "probabilities": score["probabilities"][i].tolist()}
        row.update({name: float(score[name][i]) for name in SCORE_NAMES})
        row.update({f"flag_{name}": bool(flags[name][i]) for name in SCORE_NAMES})
        rows.append(row)
    request = {"reference_signature": reference["signature"],
               "generated_identities_sha256": stable_hash(records),
               "generated_logits_sha256": _array_hash(array),
               "provenance": provenance or {}}
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        prior = read_json(out / "evaluation_request.json") if (out / "evaluation_request.json").exists() else None
        if prior != request or not (out / "report.json").exists():
            raise FileExistsError("Evaluation output already differs; use a new directory")
        prior_report = read_json(out / "report.json")
        if sha256_file(out / "scores.jsonl") != prior_report["per_sample_scores_sha256"]:
            raise ValueError("Saved per-sample scores changed")
        return prior_report
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "evaluation_request.json", request)
    write_jsonl(out / "scores.jsonl", rows)
    report = {
        "protocol_id": profile["protocol_id"], "modality": profile["modality"],
        "reference_signature": reference["signature"], "reference_path": str(ref_path),
        "checkpoint_sha256": checkpoint_sha256, "n": len(records), "mms": summary,
        "provenance": provenance or {},
        "per_sample_scores": "scores.jsonl", "per_sample_scores_sha256": sha256_file(out / "scores.jsonl"),
        "interpretation": "Calibrated classifier-uncertainty candidates; human semantic confusion remains unverified",
    }
    write_json(out / "report.json", report)
    return report
