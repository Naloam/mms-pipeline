import numpy as np
import pytest

from mms_eval.detection import binary_metrics, bootstrap_indices
from mms_eval.semantic import SCORE_NAMES


def test_detection_denominators_ranking_ties_and_undefined_cases():
    m = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0], [.8, .7, .3, .2])
    assert [m[k] for k in ('tp', 'fp', 'fn', 'tn')] == [1, 1, 1, 1]
    assert m['tpr'] == m['fpr'] == m['precision'] == .5
    assert m['auroc'] == .75 and m['ap'] == pytest.approx(5/6)
    tied = binary_metrics([1, 0, 1, 0], [0, 0, 0, 0], [.5]*4)
    assert tied['auroc'] == .5 and tied['ap'] == .5
    assert tied['precision'] is None and tied['precision_ci95_wilson'] is None
    empty_positive = binary_metrics([0, 0], [0, 1], [.1, .3])
    assert empty_positive['tpr'] is None and empty_positive['auroc'] is None and empty_positive['ap'] is None


def test_bootstrap_keeps_parent_groups_and_source_quotas():
    groups = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
    strata = ['x', 'y']*4
    first = list(bootstrap_indices(groups, strata, repetitions=20, seed=3))
    again = list(bootstrap_indices(groups, strata, repetitions=20, seed=3))
    for draw, repeated in zip(first, again):
        np.testing.assert_array_equal(draw, repeated)
        counts = np.bincount(draw, minlength=8)
        np.testing.assert_array_equal(counts[::2], counts[1::2])
        assert sum(strata[i] == 'x' for i in draw) == 4
        assert sum(strata[i] == 'y' for i in draw) == 4


def test_paired_detection_table_unknowns_and_identical_method_differences():
    from mms_eval.detection import detection_table
    rows = []
    for i in range(12):
        row = {'image_id': str(i), 'setting_id': 'model', 'parent_id': str(i//2)}
        row.update({s: i/12 for s in SCORE_NAMES})
        row.update({'flag_'+s: i > 5 for s in SCORE_NAMES})
        rows.append(row)
    labels = {str(i): ('unknown' if i > 9 else 'yes' if i%2 else 'no') for i in range(12)}
    result = detection_table(rows, labels, repetitions=30, seed=7)
    assert result['coverage'] == 10/12
    assert result['prevalence_identification_interval'] == [5/12, 7/12]
    assert len(result['methods']) == 10
    assert result['bootstrap']['group_count'] == 5
    for diff in result['differences']:
        if diff['metric'] in ('auroc', 'ap'):
            assert diff['difference'] == 0
            assert diff['ci95'] == [0, 0]
    for row in result['methods'][5:]:
        assert row['ap'] is None and row['auroc'] is None
    unknown = detection_table(rows, {str(i): 'unknown' for i in range(12)}, repetitions=10)
    assert unknown['coverage'] == 0
    assert all(row['auroc'] is None for row in unknown['methods'])


def test_ranking_metrics_agree_with_sklearn_for_many_ties():
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.RandomState(10)
    for _ in range(20):
        y, scores = rng.randint(0, 2, 40), rng.randint(0, 6, 40)
        result = binary_metrics(y, scores > 3, scores)
        assert result['auroc'] == pytest.approx(roc_auc_score(y, scores))
        assert result['ap'] == pytest.approx(average_precision_score(y, scores))


def test_cross_source_seed_collisions_are_not_implicit_pairs():
    from mms_eval.detection import _identities
    rows=[{'image_id':str(i),'setting_id':source,'seed':0} for i,source in enumerate(('gan','ddpm'))]
    groups,_=_identities(rows,'auto','setting_id')
    assert groups[0]!=groups[1]
    for row in rows: row['pair_id']='registered_shared_latent_0'
    groups,_=_identities(rows,'auto','setting_id')
    assert groups[0]==groups[1]
