import numpy as np
import pytest

from mms_eval.prototypes import fit_prototypes,l2_normalize,prototype_logits,_without_labels
from mms_eval.semantic import posterior_scores


def test_prototypes_use_only_features_and_known_cosine_posterior():
    x=np.array([[1.,0,0],[2,0,0],[0,1,0],[0,3,0],[0,0,1],[0,0,4]])
    centers=fit_prototypes(x)
    assert np.allclose(np.sort(centers,axis=0),np.sort(np.eye(3),axis=0))
    logits=prototype_logits(np.eye(3),np.eye(3))
    assert np.array_equal(logits,10*np.eye(3))
    p=posterior_scores(logits)['probabilities']
    assert np.diag(p)==pytest.approx(np.repeat(np.exp(10)/(np.exp(10)+2),3))
    records=[{'image_id':'a','path':'x','sha256':'abc','label':2,'class_name':'wild','human_target_class':'wild'}]
    assert _without_labels(records)==[{'image_id':'a','path':'x','sha256':'abc'}]
    with pytest.raises(ValueError,match='nonzero'): l2_normalize([[0,0]])
    with pytest.raises(ValueError,match='temperature'): prototype_logits(x,centers,temperature=0)
