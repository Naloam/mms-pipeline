"""Numerical checks against official torch-fidelity, without downloading weights."""

import json

import numpy as np
import pytest

from mms_eval.distribution import (
    FidelityExtractor, compute_distribution_metrics, fid_from_features,
    inception_score_from_logits, kid_from_features, polynomial_mmd_unbiased,
    precision_recall_from_features,
)


@pytest.mark.parametrize("shape", [(30, 5), (7, 30), (4, 1)])
def test_fid_matches_torch_fidelity(shape):
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("torch_fidelity.metric_fid")
    rng = np.random.RandomState(97)
    a = rng.normal(size=shape)
    b = rng.normal(size=(shape[0] + 2, shape[1])) * 1.2 + 0.4
    expected = reference.fid_statistics_to_metric(
        reference.fid_features_to_statistics(torch.from_numpy(a)),
        reference.fid_features_to_statistics(torch.from_numpy(b)), False,
    )["frechet_inception_distance"]
    assert fid_from_features(a, b) == pytest.approx(expected, abs=2e-6, rel=2e-6)


def test_fid_identity_constant_and_known_shift():
    a = np.random.RandomState(4).normal(size=(30, 5))
    assert fid_from_features(a, a) == pytest.approx(0, abs=1e-7)
    assert fid_from_features(a, a + 2) == pytest.approx(20, abs=1e-7)
    assert fid_from_features(np.zeros((4, 8)), np.ones((5, 8))) == pytest.approx(8)


def test_kid_matches_official_subsets_and_std():
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("torch_fidelity.metric_kid")
    rng = np.random.RandomState(4)
    a, b = rng.normal(size=(35, 7)), rng.normal(size=(40, 7)) + 0.3
    actual = kid_from_features(a, b, subsets=8, subset_size=12, seed=73)
    expected = reference.kid_features_to_metric(
        torch.from_numpy(a), torch.from_numpy(b), kid_subsets=8,
        kid_subset_size=12, rng_seed=73, verbose=False,
    )
    assert actual["mean"] == pytest.approx(expected["kernel_inception_distance_mean"], abs=1e-12)
    assert actual["std"] == pytest.approx(expected["kernel_inception_distance_std"], abs=1e-12)
    assert "not a confidence interval" in actual["std_kind"]


def test_kid_unbiased_estimate_may_be_negative():
    features = np.array([[-1.0], [1.0]])
    assert polynomial_mmd_unbiased(features, features) < 0


@pytest.mark.parametrize("shuffle", [True, False])
def test_is_matches_official_including_unequal_split_sizes(shuffle):
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("torch_fidelity.metric_isc")
    logits = np.random.RandomState(11).normal(size=(37, 12)) * 2
    actual = inception_score_from_logits(logits, splits=4, seed=55, shuffle=shuffle)
    expected = reference.isc_features_to_metric(torch.from_numpy(logits), splits=4, rng_seed=55, shuffle=shuffle)
    assert actual["mean"] == pytest.approx(expected["inception_score_mean"], abs=1e-12)
    assert actual["std"] == pytest.approx(expected["inception_score_std"], abs=1e-12)
    assert actual["split_sizes"] == [9, 9, 9, 10]


def test_is_extreme_logits_are_finite():
    actual = inception_score_from_logits(np.array([[1e4, -1e4], [-1e4, 1e4]]), splits=1)
    assert actual["mean"] == pytest.approx(2)


def _dense_ball_definition(real, generated, k):
    from scipy.spatial.distance import cdist

    real_radius = np.sort(cdist(real, real), axis=1)[:, k]
    gen_radius = np.sort(cdist(generated, generated), axis=1)[:, k]
    cross = cdist(generated, real)
    return float((cross <= real_radius).any(axis=1).mean()), float((cross.T <= gen_radius).any(axis=1).mean())


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 64])
def test_chunked_pr_matches_dense_published_manifold_definition(chunk_size):
    rng = np.random.RandomState(5)
    a, b = rng.normal(size=(19, 4)), rng.normal(size=(23, 4)) + 0.75
    actual = precision_recall_from_features(a, b, k=3, chunk_size=chunk_size)
    precision, recall = _dense_ball_definition(a, b, k=3)
    assert actual["precision"] == precision
    assert actual["recall"] == recall


def test_pr_orientation_has_known_asymmetry():
    # Generated support is concentrated, with some generated points far away.
    real = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    generated = np.array([[0.0], [0.1], [0.2], [9.0]])
    actual = precision_recall_from_features(real, generated, k=1, chunk_size=2)
    assert actual["precision"] == pytest.approx(3 / 4)
    # The generated outlier at 9 has radius 8.8, covering [0.2,17.8];
    # the other balls cover 0 and 0.1. All five real points are covered.
    assert actual["recall"] == pytest.approx(1.0)


def test_pr_identity_and_duplicate_points():
    real = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0]])
    actual = precision_recall_from_features(real, real, k=3, chunk_size=2)
    assert actual["precision"] == 1
    assert actual["recall"] == 1


def test_single_image_is_unavailable_for_every_collection_metric():
    actual = compute_distribution_metrics(np.zeros((12, 8)), np.ones((1, 8)), np.ones((1, 1008)))
    for metric in ("fid", "kid", "is", "precision_recall"):
        assert actual[metric]["status"] == "unavailable"
        assert actual[metric]["value"] is None
    json.dumps(actual, allow_nan=False)


def test_small_n_does_not_silently_change_subset_or_splits():
    rng = np.random.RandomState(2)
    actual = compute_distribution_metrics(rng.randn(9, 4), rng.randn(9, 4), rng.randn(9, 10))
    assert actual["fid"]["status"] == "ok"
    assert actual["fid"]["small_sample_check"]
    assert actual["kid"]["status"] == "unavailable"
    assert actual["kid"]["subset_size"] == 1000
    assert actual["is"]["status"] == "unavailable"
    assert actual["is"]["splits"] == 10


def test_pipeline_all_metrics_and_reproducible_pr_subsample():
    rng = np.random.RandomState(2)
    real, generated, logits = rng.randn(31, 4), rng.randn(37, 4), rng.randn(37, 10)
    config = {"kid_subset_size": 8, "kid_subsets": 3, "isc_splits": 3,
              "pr_max_samples": 15, "pr_chunk_size": 5, "rng_seed": 14}
    first = compute_distribution_metrics(real, generated, logits, config=config)
    second = compute_distribution_metrics(real, generated, logits, config=config)
    assert first == second
    assert all(first[key]["status"] == "ok" for key in ("fid", "kid", "is", "precision_recall"))
    assert first["precision_recall"]["n_real"] == 15
    assert len(set(first["precision_recall"]["real_indices"])) == 15
    json.dumps(first, allow_nan=False)


def test_is_does_not_require_real_references():
    actual = compute_distribution_metrics(None, None, np.zeros((30, 1008)))
    assert actual["is"]["mean"] == pytest.approx(1)
    assert actual["fid"]["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["nan", "shape", "feature_mismatch", "logit_count"])
def test_invalid_caches_fail_loudly(kind):
    a, b, logits = np.zeros((8, 4)), np.zeros((8, 4)), np.zeros((8, 10))
    if kind == "nan":
        a[0, 0] = np.nan
    elif kind == "shape":
        a = np.zeros(8)
    elif kind == "feature_mismatch":
        b = np.zeros((8, 5))
    else:
        logits = np.zeros((7, 10))
    with pytest.raises(ValueError):
        compute_distribution_metrics(a, b, logits)


def test_extractor_mixed_sizes_preserves_order_and_no_padding(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    # No pretrained download: exercise batching/IO with a deterministic probe model.
    class ProbeModel:
        def __init__(self):
            self.shapes = []

        def __call__(self, tensor):
            self.shapes.append(tuple(tensor.shape))
            means = tensor.float().mean((1, 2, 3))
            return means[:, None].repeat(1, 2048), means[:, None].repeat(1, 1008)

    paths = []
    for index, (width, height) in enumerate([(28, 28), (64, 32), (28, 28), (8, 19)]):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (width, height), (index * 40,) * 3).save(path)
        paths.append(path)
    extractor = FidelityExtractor.__new__(FidelityExtractor)
    extractor.device = torch.device("cpu")
    extractor.batch_size = 4
    extractor.load_image = None
    extractor.metadata = {"test_probe": True}
    extractor.model = ProbeModel()
    result = extractor.extract_paths(paths)
    np.testing.assert_array_equal(result["features"][:, 0], [0, 40, 80, 120])
    assert extractor.model.shapes == [(2, 3, 28, 28), (1, 3, 32, 64), (1, 3, 19, 8)]
    assert result["metadata"]["input_size_counts"] == {"28x28": 2, "64x32": 1, "8x19": 1}
