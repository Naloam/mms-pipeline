"""End-to-end AG News adapter tests on synthetic CSVs with the official row counts."""

import csv
import json
from pathlib import Path

import pytest

from mms_multimodal import agnews
from mms_eval.utils import read_json, read_jsonl

CLASS_WORDS = {
    0: "war diplomat treaty government election minister",
    1: "match score team season coach victory league",
    2: "market stocks profit company economy merger shares",
    3: "software research science technology data experiment",
}
SHARED = "news report said year"


def _write_csv(path: Path, per_class: int, part: str, *, start: int = 0) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        counter = start
        for label in range(4):
            for _ in range(per_class):
                writer.writerow([label + 1, f"{part} item {counter}",
                                 f"{CLASS_WORDS[label]} {SHARED} number {counter}"])
                counter += 1


def _tiny_config() -> dict:
    return {
        "architecture": "sklearn_tfidf_logreg_v1", "classes": list(agnews.CLASSES),
        "vectorizer": {"lowercase": True, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
        "logistic_regression": {"Cs": [1.0], "max_iter": 50, "solver": "lbfgs",
                                "random_state": 7},
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("agnews")
    train_csv, test_csv = root / "train.csv", root / "test.csv"
    _write_csv(train_csv, 30_000, "train", start=0)
    _write_csv(test_csv, 1_900, "test", start=1_000_000)
    splits = root / "splits"
    split = agnews.prepare(train_csv, test_csv, splits)
    classifier = root / "classifier"
    report = agnews.train_classifier(splits, classifier, config=_tiny_config())
    reference_dir = root / "reference"
    reference = agnews.build_reference(classifier, splits, reference_dir)
    return {"root": root, "train_csv": train_csv, "test_csv": test_csv, "splits": splits,
            "split": split, "classifier": classifier, "report": report,
            "reference": reference, "reference_dir": reference_dir,
            "reference_json": reference_dir / "reference.json"}


def test_prepare_counts_and_manifests(trained):
    split = trained["split"]
    assert split["counts"] == {"fit": 114_000, "validation": 6_000,
                               "calibration": 3_800, "audit": 3_800}
    assert split["duplicate_texts_dropped"] == {"train": 0, "test": 0}
    for name in ("fit", "validation", "calibration", "audit"):
        rows = read_jsonl(trained["splits"] / f"{name}.jsonl")
        assert len(rows) == split["counts"][name]
        assert all(r["text"] and r["sha256"] == agnews.text_sha256(r["text"]) for r in rows[:5])


def test_prepare_rejects_reuse_and_bad_sources(trained, tmp_path):
    with pytest.raises(FileExistsError):
        agnews.prepare(trained["train_csv"], trained["test_csv"], trained["splits"])
    short = tmp_path / "short.csv"
    _write_csv(short, 100, "train")
    with pytest.raises(ValueError, match="120000"):
        agnews.prepare(short, trained["test_csv"], tmp_path / "splits1")
    shared = tmp_path / "shared.csv"
    _write_csv(shared, 1_900, "train", start=0)
    with pytest.raises(ValueError, match="share identical texts"):
        agnews.prepare(trained["train_csv"], shared, tmp_path / "splits2")


def test_train_report_and_audit_metrics(trained):
    report = trained["report"]
    assert report["status"] == "trained_and_evaluated"
    assert report["counts"] == {"fit": 114_000, "validation": 6_000,
                                "calibration": 3_800, "audit": 3_800}
    assert report["metrics"]["audit_half"]["accuracy"] > .9
    assert Path(trained["classifier"] / "evaluator.pkl").is_file()
    audit = read_json(trained["reference_dir"] / "classifier_audit.json")
    assert audit["n"] == 3_800 and audit["accuracy"] > .9
    assert audit["reference_signature"] == trained["reference"]["signature"]


def test_reference_contents(trained):
    reference = trained["reference"]
    assert reference["profile"]["modality"] == "text"
    assert reference["counts"] == {"calibration": 3_800, "audit": 3_800}
    assert reference["calibrations"]["entropy"]["alpha"] == .05
    assert reference["calibrations"]["entropy"]["threshold"] != "infinity"


def test_evaluate_generated_manifest_and_directory(trained, tmp_path):
    texts = [f"synthetic generated article {i} {CLASS_WORDS[i % 4]} {SHARED}" for i in range(24)]
    manifest = tmp_path / "generated.jsonl"
    manifest.write_text("".join(
        json.dumps({"sample_id": f"gen:{i:03d}", "text": text}) + "\n"
        for i, text in enumerate(texts)), encoding="utf-8")
    out = tmp_path / "evaluation"
    report = agnews.evaluate_generated(trained["classifier"], trained["reference_json"],
                                       trained["splits"], manifest, out)
    assert report["n"] == 24 and report["modality"] == "text"
    assert report["mms"]["n"] == 24
    rows = read_jsonl(out / "scores.jsonl")
    assert len(rows) == 24 and "flag_entropy" in rows[0]
    snapshot = read_jsonl(out / "generated_input.jsonl")
    assert len(snapshot) == 24 and snapshot[0]["text"].startswith("synthetic generated")
    again = agnews.evaluate_generated(trained["classifier"], trained["reference_json"],
                                      trained["splits"], manifest, out)
    assert again["mms"] == report["mms"]
    directory = tmp_path / "texts"
    directory.mkdir()
    for i, text in enumerate(texts[:6]):
        (directory / f"{i:03d}.txt").write_text(text, encoding="utf-8")
    report_dir = agnews.evaluate_generated(trained["classifier"], trained["reference_json"],
                                           trained["splits"], directory, tmp_path / "evaluation_dir",
                                           source_id="dir_model")
    assert report_dir["n"] == 6
    assert report_dir["provenance"]["input_kind"] == "directory"
    assert report_dir["provenance"]["source_id"] == "dir_model"


def test_evaluate_rejects_real_overlap_and_bad_rows(trained, tmp_path):
    real = read_jsonl(trained["splits"] / "calibration.jsonl")[0]
    manifest = tmp_path / "overlap.jsonl"
    manifest.write_text(json.dumps({"sample_id": "leak:0", "text": real["text"]}) + "\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        agnews.evaluate_generated(trained["classifier"], trained["reference_json"],
                                  trained["splits"], manifest, tmp_path / "eval_overlap")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"sample_id": "x", "text": "   "}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        agnews.evaluate_generated(trained["classifier"], trained["reference_json"],
                                  trained["splits"], bad, tmp_path / "eval_bad")


def test_read_generated_texts_manifest_paths(trained, tmp_path):
    target = tmp_path / "one.txt"
    target.write_text("market stocks profit economy article", encoding="utf-8")
    manifest = tmp_path / "paths.jsonl"
    manifest.write_text(json.dumps({"sample_id": "p:0", "path": str(target)}) + "\n",
                        encoding="utf-8")
    rows = agnews.read_generated_texts(manifest)
    assert rows[0]["sample_id"] == "p:0"
    assert rows[0]["sha256"] == agnews.text_sha256("market stocks profit economy article")
    with pytest.raises(FileNotFoundError):
        agnews.read_generated_texts(tmp_path / "missing.jsonl")


def test_config_validation(trained, tmp_path):
    config = _tiny_config()
    config["architecture"] = "something_else"
    with pytest.raises(ValueError, match="architecture"):
        agnews.train_classifier(trained["splits"], tmp_path / "c1", config=config)
    config = _tiny_config()
    config["classes"] = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="classes"):
        agnews.train_classifier(trained["splits"], tmp_path / "c2", config=config)
