import hashlib

import numpy as np
import pytest

from mms_multimodal.core import evaluate_logits, freeze_reference


def _sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_same_scoring_interface_for_text_and_audio(tmp_path, modality):
    profile = {"protocol_id": f"toy-{modality}-v1", "modality": modality,
               "dataset_version": "fixed-test", "category_mode": "exclusive",
               "classes": ["a", "b"], "semantic_definition": "One of two subjects",
               "preprocessing": "frozen adapter"}
    real = [{"sample_id": f"real-{i}", "sha256": _sha(i), "label": i % 2} for i in range(30)]
    calibration = np.array([[4.0, 0.0] if row["label"] == 0 else [0.0, 4.0] for row in real[:20]])
    audit = np.array([[4.0, 0.0] if row["label"] == 0 else [0.0, 4.0] for row in real[20:]])
    checkpoint = _sha("checkpoint")
    reference = freeze_reference(profile, checkpoint, real[:20], calibration, real[20:], audit,
                                 tmp_path / "reference", source_hashes={"dataset": _sha("dataset")})
    assert reference["counts"] == {"calibration": 20, "audit": 10}
    assert reference["real_audit"]["classification_accuracy"] == 1
    assert freeze_reference(profile, checkpoint, real[:20], calibration, real[20:], audit,
                            tmp_path / "reference", source_hashes={"dataset": _sha("dataset")})["signature"] == reference["signature"]

    generated = [{"sample_id": f"generated-{i}", "sha256": _sha(f"generated-{i // 2}")} for i in range(4)]
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [0.1, 0.1], [0.2, 0.2]])
    result = evaluate_logits(profile, checkpoint, generated, logits,
                             tmp_path / "reference" / "reference.json", tmp_path / "evaluation")
    assert result["n"] == 4
    assert result["mms"]["duplicate_content_count"] == 2
    assert result["mms"]["candidate_count"] >= 1
    assert evaluate_logits(profile, checkpoint, generated, logits,
                           tmp_path / "reference" / "reference.json", tmp_path / "evaluation") == result
    with pytest.raises(ValueError, match="differs"):
        evaluate_logits(profile, _sha("wrong"), generated, logits,
                        tmp_path / "reference" / "reference.json", tmp_path / "other")
    with pytest.raises(ValueError, match="overlap"):
        evaluate_logits(profile, checkpoint, [{"sample_id": "new", "sha256": real[0]["sha256"]}],
                        [[0.0, 1.0]], tmp_path / "reference" / "reference.json", tmp_path / "other")


def test_reference_rejects_real_overlap_and_multilabel(tmp_path):
    profile = {"protocol_id": "toy", "modality": "text", "dataset_version": "one",
               "category_mode": "exclusive", "classes": ["a", "b"],
               "semantic_definition": "one topic", "preprocessing": "tokenizer-v1"}
    record = {"sample_id": "x", "sha256": _sha("x"), "label": 0}
    with pytest.raises(ValueError, match="overlap"):
        freeze_reference(profile, _sha("checkpoint"), [record], [[1, 0]], [record], [[1, 0]], tmp_path)
    profile["category_mode"] = "multilabel"
    with pytest.raises(ValueError, match="exclusive"):
        freeze_reference(profile, _sha("checkpoint"), [record], [[1, 0]], [record], [[1, 0]], tmp_path)
