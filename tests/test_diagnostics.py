import pytest

from mms_eval.diagnostics import review_budget_curves
from mms_eval.semantic import SCORE_NAMES


def test_review_budget_ties_follow_fixed_ids_and_unknowns_cost_reviews():
    rows = [{'image_id': str(i), **{s: 1. for s in SCORE_NAMES}} for i in range(10)]
    labels = {str(i): 1 if i < 2 else None if i < 5 else 0 for i in range(10)}
    result = review_budget_curves(rows, labels, bootstrap_repetitions=50)
    assert result == review_budget_curves(list(reversed(rows)), labels, bootstrap_repetitions=50)
    halfway = next(r for r in result if r['method']=='entropy' and r['reviewed_n']==5)
    assert halfway['confirmed_MM_found'] == 2
    assert halfway['uniform_review_empirical_mean'] == pytest.approx(1, abs=.1)
    assert halfway['unknown_reviewed_n'] == 3 and halfway['resolved_reviewed_n'] == 2
    assert halfway['unknown_n'] == 3 and halfway['cohort_n'] == 10
    assert halfway['recall_of_confirmed_MM'] == 1
