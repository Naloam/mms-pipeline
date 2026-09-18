import numpy as np
import pytest

from mms_eval.classifier_audit import classification_metrics


def test_uniform_balanced_classifier_has_known_calibration_and_loss():
    result=classification_metrics(np.zeros((4,2)),np.array([0,1,0,1]),bins=10)
    assert result['accuracy']==.5
    assert result['nll']==pytest.approx(np.log(2))
    assert result['multiclass_brier']==.5 and result['ece']==0
    assert result['confusion_rows_true_columns_predicted']==[[2,0],[2,0]]
    assert sum(r['n'] for r in result['reliability'])==4


def test_perfect_confidence_endpoint_belongs_to_last_bin():
    result=classification_metrics(np.array([[1000.,-1000.],[-1000.,1000.]]),np.array([0,1]),bins=10)
    assert result['accuracy']==1 and result['ece']==0
    assert result['reliability'][-1]['n']==2
