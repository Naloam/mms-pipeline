import numpy as np
import pytest

from mms_eval.domains import import_arrays, validate_domain_contract
from mms_eval.images import load_rgb
from mms_eval.splits import prepare_real_data
from mms_eval.utils import read_jsonl, write_jsonl


def test_array_export_and_manifest_real_split(tmp_path):
    data = np.random.RandomState(4).randint(0, 256, (30, 8, 8), dtype=np.uint8)
    np.save(tmp_path/'images.npy', data)
    np.save(tmp_path/'labels.npy', np.arange(30)%2)
    result = import_arrays(tmp_path/'images.npy', tmp_path/'imported', layout='NHW',
                           value_range='uint8', dataset_id='toy-v1', labels=tmp_path/'labels.npy',
                           classes=['zero', 'one'], source_directory='train')
    rows = read_jsonl(tmp_path/'imported'/'manifest.jsonl')
    assert result['n'] == 30 and all(load_rgb(r['path']).size == (8,8) for r in rows)
    assert np.array_equal(np.asarray(load_rgb(rows[0]['path']))[:,:,0], data[0])
    for i in range(26,30):
        rows[i]['source_directory'] = 'val'
    write_jsonl(tmp_path/'real.jsonl', rows)
    config = {'domain': 'toy', 'classes': ['zero', 'one'], 'validation_n': 6, 'calibration_pool_n': 6,
              'internal_audit_n': 6, 'calibration_n': 4, 'split_seed': 2}
    split = prepare_real_data(tmp_path/'real.jsonl', tmp_path/'splits', config=config)
    assert split['counts']['audit_external'] == 4
    assert sum(split['counts'].values()) == 30


def test_domain_contract_rejects_multilabel_and_imagenet_pretraining_overlap():
    base = {'domain_contract_version': 1, 'domain': 'ImageNet', 'classes': ['n1', 'n2'],
            'category_mode': 'exclusive', 'dataset_version': 'test-v1', 'semantic_definition': 'two synsets',
            'pretraining_audit': {'status': 'disjoint_declared', 'evidence': 'train images excluded',
                                  'calibration_source': 'imagenet1k_train'}}
    with pytest.raises(ValueError, match='ImageNet'):
        validate_domain_contract(base, evaluator_config={'weights': 'IMAGENET1K_V1'})
    base['pretraining_audit']['calibration_source'] = 'independent_new_holdout'
    assert validate_domain_contract(base, evaluator_config={'weights': 'IMAGENET1K_V1'})['mode'] == 'semantic'
    base.update(domain='CelebA', category_mode='multilabel')
    with pytest.raises(ValueError, match='exclusive'):
        validate_domain_contract(base)
    assert validate_domain_contract(base, mode='quality_only')['mode'] == 'quality_only'


def test_from_scratch_must_explicitly_disable_default_pretrained_weights():
    config={'domain':'MNIST','classes':[str(i) for i in range(10)],'category_mode':'exclusive',
            'dataset_version':'toy','semantic_definition':'toy','digit_polarity':'light_on_dark',
            'pretraining_audit':{'status':'from_scratch','evidence':'toy'}}
    with pytest.raises(ValueError,match='from_scratch'):
        validate_domain_contract(config,evaluator_config={'architecture':'resnet50'})
    assert validate_domain_contract(config,evaluator_config={'architecture':'resnet50','weights':None})


def test_array_import_requires_explicit_range_and_safe_ids(tmp_path):
    np.save(tmp_path/'x.npy', np.zeros((2,4,4), dtype=np.float32))
    with pytest.raises(ValueError, match='value_range'):
        import_arrays(tmp_path/'x.npy', tmp_path/'bad', layout='NHW', value_range=None, dataset_id='toy')
    with pytest.raises(ValueError, match='dataset_id'):
        import_arrays(tmp_path/'x.npy', tmp_path/'bad2', layout='NHW', value_range='0_1', dataset_id='')
