"""AG News text adapter for the shared MMS candidate score.

Real data are the official AG News CSV splits (4 topics). The evaluator is a
CPU-trained scikit-learn TF-IDF + logistic regression pipeline, so every step
here runs without a GPU. A future transformer evaluator only needs a new
checkpoint directory and a new frozen reference; this adapter's contract stays
unchanged. High classifier uncertainty marks a cross-topic candidate, never a
confirmed mixed-topic article.
"""

from __future__ import annotations

import csv
import pickle
import platform
from pathlib import Path

import numpy as np

from mms_eval.utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl

from .core import evaluate_logits, freeze_reference
from .support import (audit_metrics, fresh_output_dir, read_generated_texts,
                      text_sha256, verify_split_manifests)


CLASSES = ["World", "Sports", "Business", "SciTech"]
SPLIT_SEED = 2026091801
EXPECTED_TRAIN_ROWS = 120_000
EXPECTED_TEST_ROWS = 7_600
EXPECTED_PER_CLASS = {"train": 30_000, "test": 1_900}
ARCHITECTURE = "sklearn_tfidf_logreg_v1"
EVALUATOR_FILE = "evaluator.pkl"


def _read_csv(path: str | Path, *, expected_rows: int, part: str) -> tuple[list[dict], dict]:
    source = Path(path).resolve(strict=True)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} official AG News {part} rows, found {len(rows)}")
    counts = [0] * len(CLASSES)
    records, seen = [], {}
    dropped = []
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(f"AG News {part} row {index} does not have three CSV fields")
        label = int(row[0]) - 1
        if not 0 <= label < len(CLASSES):
            raise ValueError(f"AG News {part} row {index} has an unknown class index: {row[0]}")
        text = " ".join(part_text.strip() for part_text in row[1:] if part_text.strip())
        if not text:
            raise ValueError(f"AG News {part} row {index} has an empty text")
        digest = text_sha256(text)
        record = {"sample_id": f"agnews-{part}:{index:06d}", "label": label, "text": text,
                  "sha256": digest, "row_index": index}
        if digest in seen:
            dropped.append({"kept": seen[digest], "dropped": record["sample_id"]})
        else:
            seen[digest] = record["sample_id"]
            records.append(record)
        counts[label] += 1
    if counts != [EXPECTED_PER_CLASS[part]] * len(CLASSES):
        raise ValueError(f"AG News {part} class balance changed: {counts}")
    return records, {"source": str(source), "sha256": sha256_file(source),
                     "rows": expected_rows, "duplicate_texts_dropped": dropped}


def prepare(train_csv: str | Path, test_csv: str | Path, out_dir: str | Path,
            *, seed: int = SPLIT_SEED, validation_n: int = 6000) -> dict:
    """Split official train rows into fit/validation and test rows into calibration/audit."""
    if validation_n % len(CLASSES) or not 0 < validation_n < EXPECTED_TRAIN_ROWS // 2:
        raise ValueError("validation_n must be a positive multiple of four below half the train set")
    train, train_info = _read_csv(train_csv, expected_rows=EXPECTED_TRAIN_ROWS, part="train")
    test, test_info = _read_csv(test_csv, expected_rows=EXPECTED_TEST_ROWS, part="test")
    if {r["sha256"] for r in train} & {r["sha256"] for r in test}:
        raise ValueError("The two CSV files share identical texts; they are not the official AG News splits")
    rng = np.random.default_rng(seed)
    per_class_validation = validation_n // len(CLASSES)
    parts: dict[str, list[dict]] = {"fit": [], "validation": [], "calibration": [], "audit": []}
    for label in range(len(CLASSES)):
        for source, (first_name, second_name), first_size in (
                (train, ("validation", "fit"), per_class_validation),
                (test, ("calibration", "audit"), None)):
            group = [r for r in source if r["label"] == label]
            order = rng.permutation(len(group))
            if first_size is None:
                first_size = len(group) // 2
            for i in order[:first_size]:
                parts[first_name].append(group[int(i)])
            for i in order[first_size:]:
                parts[second_name].append(group[int(i)])
    parts = {name: sorted(part, key=lambda r: r["sample_id"]) for name, part in parts.items()}
    counts = {name: len(part) for name, part in parts.items()}
    if counts["fit"] + counts["validation"] != len(train) or counts["calibration"] + counts["audit"] != len(test):
        raise AssertionError("AG News split does not cover the source rows")
    if len({r["sample_id"] for part in parts.values() for r in part}) != len(train) + len(test):
        raise AssertionError("AG News split identity overlap")
    out = fresh_output_dir(out_dir)
    for name, part in parts.items():
        write_jsonl(out / f"{name}.jsonl", part)
    report = {
        "dataset": "AG News official CSV splits (title + description, class index 1-4)",
        "classes": CLASSES, "seed": seed, "validation_n": validation_n,
        "train_csv": train_info, "test_csv": test_info,
        "counts": counts,
        "roles": {"fit": "logistic-regression gradient fitting",
                  "validation": "C selection by mean negative log likelihood",
                  "calibration": "entropy threshold calibration",
                  "audit": "untouched real-sample audit"},
        "duplicate_texts_dropped": {"train": len(train_info["duplicate_texts_dropped"]),
                                    "test": len(test_info["duplicate_texts_dropped"])},
        "manifest_sha256": {name: sha256_file(out / f"{name}.jsonl") for name in parts},
    }
    write_json(out / "split.json", report)
    return report


def _split(split_dir: str | Path) -> tuple[Path, dict]:
    root, report = verify_split_manifests(split_dir, expected_name="calibration")
    if report["classes"] != CLASSES:
        raise ValueError("AG News split class list changed")
    return root, report


def _vectorizer(config: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    settings = config["vectorizer"]
    return TfidfVectorizer(lowercase=settings["lowercase"],
                           ngram_range=tuple(settings["ngram_range"]),
                           min_df=settings["min_df"], sublinear_tf=settings["sublinear_tf"])


def _train_one(config: dict, texts: list[str], labels: np.ndarray, c_value: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    regression = config["logistic_regression"]
    return Pipeline([("tfidf", _vectorizer(config)),
                     ("logreg", LogisticRegression(C=c_value, max_iter=regression["max_iter"],
                                                   solver=regression["solver"],
                                                   random_state=regression["random_state"]))]).fit(texts, labels)


def _config_check(config: dict, split: dict) -> dict:
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Expected evaluator architecture {ARCHITECTURE}")
    if config.get("classes") != split["classes"]:
        raise ValueError("Evaluator classes must match the AG News split")
    settings, regression = config.get("vectorizer"), config.get("logistic_regression")
    if not isinstance(settings, dict) or not isinstance(regression, dict):
        raise ValueError("Configuration requires vectorizer and logistic_regression sections")
    ngram = settings.get("ngram_range")
    if (not isinstance(settings.get("lowercase"), bool) or
            not isinstance(ngram, list) or len(ngram) != 2 or
            not all(isinstance(v, int) for v in ngram) or not 1 <= ngram[0] <= ngram[1] or
            not isinstance(settings.get("min_df"), int) or settings["min_df"] < 1 or
            not isinstance(settings.get("sublinear_tf"), bool)):
        raise ValueError("Invalid vectorizer settings")
    candidates = regression.get("Cs")
    if (not isinstance(candidates, list) or not candidates or
            not all(isinstance(v, (int, float)) and v > 0 for v in candidates) or
            not isinstance(regression.get("max_iter"), int) or regression["max_iter"] < 1 or
            regression.get("solver") != "lbfgs" or
            not isinstance(regression.get("random_state"), int)):
        raise ValueError("Invalid logistic regression settings")
    return config


def train_classifier(split_dir: str | Path, out_dir: str | Path, *, config: dict) -> dict:
    """Fit the TF-IDF + logistic regression evaluator and select C on validation NLL."""
    import sklearn
    root, split = _split(split_dir)
    _config_check(config, split)
    fit = read_jsonl(root / "fit.jsonl")
    validation = read_jsonl(root / "validation.jsonl")
    calibration = read_jsonl(root / "calibration.jsonl")
    audit = read_jsonl(root / "audit.jsonl")
    fit_texts = [r["text"] for r in fit]
    fit_labels = np.asarray([r["label"] for r in fit], dtype=int)
    val_texts = [r["text"] for r in validation]
    val_labels = np.asarray([r["label"] for r in validation], dtype=int)
    out = fresh_output_dir(out_dir)
    candidates = []
    for c_value in config["logistic_regression"]["Cs"]:
        model = _train_one(config, fit_texts, fit_labels, c_value)
        probabilities = model.predict_proba(val_texts)
        nll = float(-np.log(np.maximum(probabilities[np.arange(len(val_labels)), val_labels],
                                       1e-300)).mean())
        candidates.append({"C": float(c_value), "validation_nll": nll, "model": model})
    best = min(candidates, key=lambda item: item["validation_nll"])
    with (out / EVALUATOR_FILE).open("wb") as handle:
        pickle.dump(best["model"], handle, protocol=4)
    metrics = {}
    for name, rows in (("validation", validation), ("calibration_half", calibration),
                       ("audit_half", audit)):
        logits = best["model"].decision_function([r["text"] for r in rows])
        metrics[name] = audit_metrics([r["label"] for r in rows], logits, CLASSES)
    report = {
        "status": "trained_and_evaluated", "architecture": ARCHITECTURE, "classes": CLASSES,
        "vectorizer": config["vectorizer"],
        "logistic_regression": {**config["logistic_regression"],
                                "Cs": [c["C"] for c in candidates],
                                "selected_C": best["C"],
                                "selection_nll": best["validation_nll"]},
        "selection": "minimum_validation_mean_negative_log_likelihood",
        "temperature_applied": False,
        "counts": {"fit": len(fit), "validation": len(validation),
                   "calibration": len(calibration), "audit": len(audit)},
        "metrics": metrics,
        "train_csv_sha256": split["train_csv"]["sha256"],
        "test_csv_sha256": split["test_csv"]["sha256"],
        "split_manifest_sha256": {name: digest for name, digest in split["manifest_sha256"].items()},
        "evaluator_sha256": sha256_file(out / EVALUATOR_FILE),
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__},
        "model_selection": ("Validation texts select the regularization strength only; "
                            "calibration and audit texts are never used for fitting or selection."),
        "limitation": ("A linear bag-of-words evaluator captures topic-word confusion, not full "
                       "natural-language semantics; candidate rates inherit this boundary."),
    }
    write_json(out / "report.json", report)
    return report


def _load_classifier(classifier_dir: str | Path) -> tuple[object, dict, Path]:
    classifier = Path(classifier_dir).resolve(strict=True)
    report = read_json(classifier / "report.json")
    if report.get("status") != "trained_and_evaluated" or report.get("classes") != CLASSES:
        raise ValueError("Expected a frozen four-class AG News evaluator")
    if sha256_file(classifier / EVALUATOR_FILE) != report["evaluator_sha256"]:
        raise ValueError("AG News evaluator checkpoint changed")
    with (classifier / EVALUATOR_FILE).open("rb") as handle:
        return pickle.load(handle), report, classifier


def _predict(model, texts: list[str]) -> np.ndarray:
    logits = np.asarray(model.decision_function(texts), dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(len(texts), 1)
    if logits.shape != (len(texts), len(CLASSES)) or not np.isfinite(logits).all():
        raise ValueError("Evaluator returned invalid logits")
    return logits


def _profile(report: dict, split: dict) -> dict:
    vectorizer, regression = report["vectorizer"], report["logistic_regression"]
    return {
        "protocol_id": "AGNews_4class_entropy_tfidf_logreg_v1", "modality": "text",
        "dataset_version": (f"AG News official CSVs; train SHA256:{split['train_csv']['sha256']} "
                            f"test SHA256:{split['test_csv']['sha256']}"),
        "category_mode": "exclusive", "classes": CLASSES,
        "semantic_definition": (
            "One dominant news topic (World/Sports/Business/Sci-Tech) per article. High classifier "
            "uncertainty is a candidate for cross-topic feature mixing, not a confirmed mixed-topic article."
        ),
        "preprocessing": {
            "representation": "UTF-8 text of the article title and description; identity is the SHA-256 of the exact text bytes",
            "features": (f"TfidfVectorizer(lowercase={vectorizer['lowercase']}, "
                         f"ngram_range={vectorizer['ngram_range']}, min_df={vectorizer['min_df']}, "
                         f"sublinear_tf={vectorizer['sublinear_tf']}) fitted on the fit split only"),
            "classifier": (f"multinomial logistic regression (lbfgs, C={regression['selected_C']}, "
                           f"max_iter={regression['max_iter']}), selected by validation NLL"),
            "inference": "deterministic CPU scoring; no batching or device effects",
        },
        "temperature": 1.0,
    }


def build_reference(classifier_dir: str | Path, split_dir: str | Path, out_dir: str | Path,
                    *, alpha: float = .05) -> dict:
    """Freeze text-entropy thresholds on held-out official AG News test rows."""
    model, report, _ = _load_classifier(classifier_dir)
    root, split = _split(split_dir)
    calibration = read_jsonl(root / "calibration.jsonl")
    audit = read_jsonl(root / "audit.jsonl")
    cal_logits = _predict(model, [r["text"] for r in calibration])
    audit_logits = _predict(model, [r["text"] for r in audit])
    identities = lambda rows: [
        {"sample_id": r["sample_id"], "sha256": r["sha256"], "label": r["label"]} for r in rows]
    ref = freeze_reference(
        _profile(report, split), report["evaluator_sha256"],
        identities(calibration), cal_logits, identities(audit), audit_logits,
        out_dir, alpha=alpha,
        source_hashes={"train_csv": split["train_csv"]["sha256"],
                       "test_csv": split["test_csv"]["sha256"],
                       "split_manifest": sha256_file(root / "split.json")})
    write_json(Path(out_dir) / "classifier_audit.json", {
        **audit_metrics([r["label"] for r in audit], audit_logits, CLASSES),
        "split_role": "Official-test audit half; never used for fitting, selection, or calibration",
        "checkpoint_sha256": report["evaluator_sha256"],
        "reference_signature": ref["signature"],
    })
    return ref


def evaluate_generated(classifier_dir: str | Path, reference_path: str | Path,
                       split_dir: str | Path, generated: str | Path, out_dir: str | Path,
                       *, source_id: str | None = None) -> dict:
    """Score generated AG News texts against the frozen real reference."""
    model, report, _ = _load_classifier(classifier_dir)
    _, split = _split(split_dir)
    source = Path(generated)
    records = read_generated_texts(source)
    logits = _predict(model, [r["text"] for r in records])
    provenance = {
        "representation": "generated news-article texts (title + description style)",
        "generated_input": str(source.resolve()),
        "input_kind": "directory" if source.is_dir() else "jsonl_manifest",
        "source_id": source_id or source.name,
        "n": len(records),
        "selected_identity_sha256": stable_hash([(r["sample_id"], r["sha256"]) for r in records]),
    }
    if source.is_file():
        provenance["manifest_sha256"] = sha256_file(source)
    report = evaluate_logits(
        _profile(report, split), report["evaluator_sha256"], records,
        logits, reference_path, out_dir, provenance=provenance)
    # The shared core records identities and scores only; keep a readable copy
    # of the evaluated texts beside them for manual candidate review.
    write_jsonl(Path(out_dir) / "generated_input.jsonl",
                [{k: v for k, v in record.items() if k != "origin"} for record in records])
    return report
