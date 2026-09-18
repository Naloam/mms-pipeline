import numpy as np
import pytest

from mms_eval.distribution import compute_distribution_metrics, precision_recall_from_features


def test_vgg_pr_uses_separate_features_and_reports_its_protocol():
    rng = np.random.RandomState(1)
    real, gen, logits = rng.normal(size=(12, 3)), rng.normal(size=(10, 3)), rng.normal(size=(10, 4))
    rv, gv = rng.normal(size=(12, 5)), rng.normal(size=(10, 5)) + 4
    config = {'kid_subset_size': 5, 'kid_subsets': 2, 'isc_splits': 2,
              'pr_feature_backend': 'vgg16', 'pr_max_samples': None}
    report = compute_distribution_metrics(real, gen, logits, config=config,
                                         real_pr_features=rv, generated_pr_features=gv)
    expected = precision_recall_from_features(rv, gv)
    assert report['precision_recall']['precision'] == expected['precision']
    assert report['precision_recall']['recall'] == expected['recall']
    assert 'VGG' in report['precision_recall']['method']
    with pytest.raises(ValueError, match='VGG'):
        compute_distribution_metrics(real, gen, logits, config=config)


def test_clean_fid_statistics_match_published_implementation():
    official = pytest.importorskip('cleanfid.fid')
    from mms_eval.distribution import fid_from_features, polynomial_mmd_unbiased
    rng = np.random.RandomState(18)
    real, gen = rng.normal(size=(30, 5)), rng.normal(size=(30, 5)) + .4
    assert fid_from_features(real, gen) == pytest.approx(official.fid_from_feats(real, gen), rel=1e-10, abs=1e-10)
    assert polynomial_mmd_unbiased(real, gen) == pytest.approx(official.kernel_distance(real, gen, num_subsets=1, max_subset_size=30), rel=1e-10, abs=1e-10)
