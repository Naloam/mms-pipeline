"""Registered domain and evaluator configs must pass their own contracts."""

from pathlib import Path

import pytest

from mms_eval.domains import validate_domain_contract
from mms_eval.evaluator import _normalise_config
from mms_eval.utils import read_json

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_cifar10_domain_contract_passes_semantic_mode():
    config = read_json(CONFIGS / "cifar10.json")
    verdict = validate_domain_contract(config, mode="semantic",
                                       evaluator_config=config["evaluator_initialization"])
    assert verdict["classes"] == config["classes"]
    assert verdict["mode"] == "semantic"


def test_cifar10_evaluator_config_normalises():
    protocol, operational = _normalise_config(read_json(CONFIGS / "evaluator_cifar10.json"))
    assert protocol["architecture"] == "resnet50"
    assert protocol["weights"] == "IMAGENET1K_V1"
    assert protocol["classes"] == read_json(CONFIGS / "cifar10.json")["classes"]
    assert operational["device"] == "cuda:0"


def test_cifar10_domain_rejects_pretrained_from_scratch_conflict():
    config = read_json(CONFIGS / "cifar10.json")
    broken = {**config, "pretraining_audit": {"status": "from_scratch",
                                              "evidence": "declared", "calibration_source": "x"}}
    with pytest.raises(ValueError, match="from_scratch"):
        validate_domain_contract(broken, mode="semantic",
                                 evaluator_config=config["evaluator_initialization"])


def test_imagenet_template_stays_locked_until_registered():
    config = read_json(CONFIGS / "imagenet_template.json")
    with pytest.raises(ValueError, match="Record the actual dataset"):
        validate_domain_contract(config, mode="semantic", evaluator_config=None)
    with pytest.raises(ValueError, match="Record the actual dataset"):
        validate_domain_contract({**config, "registration_status": "registered",
                                  "classes": ["a", "b"]}, mode="semantic",
                                 evaluator_config=None)


def test_celeba_multilabel_only_allows_quality_mode():
    config = read_json(CONFIGS / "celeba_template.json")
    with pytest.raises(ValueError, match="Record the actual dataset"):
        validate_domain_contract(config, mode="semantic")
    filled = {**config, "dataset_version": "celeba_aligned_crops_release"}
    with pytest.raises(ValueError, match="exclusive"):
        validate_domain_contract(filled, mode="semantic")
    verdict = validate_domain_contract(filled, mode="quality_only")
    assert verdict["MMS"] == "unavailable"
