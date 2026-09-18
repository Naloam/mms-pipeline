import csv

import numpy as np
import pytest
from PIL import Image

from mms_eval.controls import build_degradation_controls, build_mixture_controls
from mms_eval.utils import read_jsonl, sha256_file, write_json


def pool(tmp_path):
    records = []
    for i in range(20):
        path = tmp_path/f'{i}.png'
        Image.fromarray(np.random.RandomState(i).randint(0, 256, (12, 12, 3), dtype=np.uint8)).save(path)
        records.append({'image_id': str(i), 'path': str(path), 'sha256':sha256_file(path), 'setting_id': 'model_A'})
    labels = tmp_path/'final_labels.csv'
    with labels.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'mm_label_final','sha256'])
        writer.writeheader()
        writer.writerows({'image_id': str(i), 'mm_label_final': 'no' if i < 10 else 'yes','sha256':records[i]['sha256']} for i in range(20))
    write_json(tmp_path/'annotation_finalization.json', {'status': 'complete', 'pack_role': 'development',
               'final_labels_sha256': sha256_file(labels)})
    return records, labels


def test_mixtures_are_nested_unique_and_match_known_prevalence(tmp_path):
    records, labels = pool(tmp_path)
    result = build_mixture_controls(records, labels, tmp_path/'mix', n=10,
                                    proportions=[0, .2, .6], repetitions=2)
    assert len(result['cohorts']) == 6
    for cohort in result['cohorts']:
        rows = read_jsonl(tmp_path/'mix'/cohort['manifest'])
        assert len({r['image_id'] for r in rows}) == 10
        assert sum(int(r['image_id']) >= 10 for r in rows) == cohort['n_positive']
    zero, low, high = [read_jsonl(tmp_path/'mix'/r['manifest']) for r in result['cohorts'][:3]]
    assert {r['image_id'] for r in low if int(r['image_id']) >= 10} <= {r['image_id'] for r in high}
    with pytest.raises(ValueError, match='Insufficient'):
        build_mixture_controls(records, labels, tmp_path/'bad', n=20, proportions=[0])
    with pytest.raises(FileExistsError):
        build_mixture_controls(records, labels, tmp_path/'mix', n=10)


def test_degradations_preserve_pairing_but_require_new_labels(tmp_path):
    records, labels = pool(tmp_path)
    for directory in ('a', 'b'):
        result = build_degradation_controls(records, labels, tmp_path/directory, base_n=3)
        rows = read_jsonl(tmp_path/directory/'manifest.jsonl')
        assert result['n'] == 30
        assert len({r['parent_id'] for r in rows}) == 3
        assert all(int(r['parent_id']) < 10 for r in rows)
        assert all('mm_label_final' not in r for r in rows)
        assert all(r['requires_reannotation'] for r in rows)
    first, second = (read_jsonl(tmp_path/d/'manifest.jsonl') for d in ('a', 'b'))
    assert [r['sha256'] for r in first] == [r['sha256'] for r in second]
    duplicate = {**records[0], 'image_id': 'different-id'}
    with pytest.raises(ValueError, match='Duplicate decoded'):
        build_degradation_controls(records+[duplicate], labels, tmp_path/'dup', base_n=3)


def test_pending_or_unsealed_labels_cannot_supply_confirmed_controls(tmp_path):
    records, labels = pool(tmp_path)
    write_json(tmp_path/'annotation_finalization.json', {'status': 'awaiting_adjudication_or_second_rater',
               'final_labels_sha256': sha256_file(labels)})
    with pytest.raises(ValueError, match='completed finalization'):
        build_degradation_controls(records, labels, tmp_path/'bad')


def test_duplicate_replacement_keeps_fixed_n_and_original_counts(tmp_path):
    from mms_eval.controls import build_duplicate_controls
    records,_ = pool(tmp_path)
    result = build_duplicate_controls(records,tmp_path/'duplicates',n=20)
    for cohort,expected in zip(result['cohorts'],(20,15,10)):
        rows = read_jsonl(tmp_path/'duplicates'/cohort['manifest'])
        assert len(rows)==20 and len({r['image_id'] for r in rows})==20
        assert len({r['original_image_id'] for r in rows})==expected
        assert sum(r['duplicate_copy'] for r in rows)==20-expected
        assert len({r['parent_id'] for r in rows})==expected


def test_human_class_composition_excludes_unconfirmed_and_unknown_classes(tmp_path):
    from mms_eval.controls import build_composition_controls
    records,labels = pool(tmp_path)
    with labels.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['image_id','mm_label_final','human_target_class','sha256'])
        w.writeheader()
        for i in range(20):
            w.writerow({'image_id':str(i),'mm_label_final':'no' if i<10 else 'yes',
                        'human_target_class':'cat' if i<5 else 'dog' if i<9 else 'unknown','sha256':records[i]['sha256']})
    write_json(tmp_path/'annotation_finalization.json',{'status':'complete','final_labels_sha256':sha256_file(labels)})
    spec={'cohorts':[{'cohort_id':'balanced','class_counts':{'cat':3,'dog':3}},
                     {'cohort_id':'more_cat','class_counts':{'cat':4,'dog':2}}]}
    result=build_composition_controls(records,labels,spec,tmp_path/'classes')
    assert result['pool_counts']=={'cat':5,'dog':4}
    first,second=[read_jsonl(tmp_path/'classes'/c['manifest']) for c in result['cohorts']]
    assert len(first)==len(second)==6
    assert {r['image_id'] for r in first if r['human_target_class']=='cat'} <= {r['image_id'] for r in second}
    spec['cohorts'][0]['class_counts']={'cat':6}
    with pytest.raises(ValueError,match='Insufficient'):
        build_composition_controls(records,labels,spec,tmp_path/'insufficient')


def test_curated_families_are_selection_strata_not_truth(tmp_path):
    from mms_eval.controls import build_curated_controls
    records,_=pool(tmp_path)
    spec={'sources':[{'family':family,'count':2,'input':records[i*5:(i+1)*5],
                      'curation_reason':'Toy fixture selection only'}
                     for i,family in enumerate(('ood','legal_multiple_subjects','mm_candidates'))]}
    result=build_curated_controls(spec,tmp_path/'curated')
    rows=read_jsonl(tmp_path/'curated/manifest.jsonl')
    assert result['n']==6 and all(r['requires_reannotation'] for r in rows)
    assert not any('mm_label_final' in r for r in rows)
    spec['sources'][1]['input']=records[:5]
    with pytest.raises(ValueError,match='Duplicate original'):
        build_curated_controls(spec,tmp_path/'bad_curated')


def test_degradation_analysis_distinguishes_paired_response_and_reconfirmed_truth(tmp_path,monkeypatch):
    from mms_eval import control_analysis
    from mms_eval.semantic import SCORE_NAMES
    records,labels=pool(tmp_path)
    scores,cohort=[],[]
    for p in range(3):
        for family,severity in (('original',0),('noise',.1)):
            image_id=str(p*2+int(family=='noise'))
            cohort.append({'image_id':image_id,'parent_id':str(p),'control_family':family,'severity':severity,
                           'cohort_role':'diagnostic_degradation','sha256':records[int(image_id)]['sha256']})
            scores.append({'image_id':image_id,**{s:float(family=='noise')*.2 for s in SCORE_NAMES},
                           **{'flag_'+s:family=='noise' for s in SCORE_NAMES}})
    monkeypatch.setattr(control_analysis,'load_evaluation',lambda _:({'scores_sha256':'fixture'},scores,cohort,{}))
    result=control_analysis.analyze_degradation('fixture',labels,tmp_path/'effects',repetitions=50)
    noise=[r for r in result['effects'] if r['family']=='noise']
    assert all(r['mean_paired_score_change']==pytest.approx(.2) for r in noise)
    assert all(r['score_change_ci95_low']==pytest.approx(.2) and r['score_change_ci95_high']==pytest.approx(.2) for r in noise)
    assert all(r['FPR_on_reconfirmed_non_MM']==1 for r in noise)
    assert all(r['flag_change_ci95_low']==r['flag_change_ci95_high']==1 for r in noise)
    cohort.pop();scores.pop()
    with pytest.raises(ValueError,match='same conditions'):
        control_analysis.analyze_degradation('fixture',labels,tmp_path/'incomplete',repetitions=50)


def test_label_identity_cannot_transfer_to_replaced_image_and_mixtures_reject_source_confounding(tmp_path):
    records,labels=pool(tmp_path)
    records[15]['setting_id']='model_B'
    with pytest.raises(ValueError,match='generation source'):
        build_mixture_controls(records,labels,tmp_path/'confounded',n=10)
    records[15]['setting_id']='model_A'
    Image.new('RGB',(12,12),'red').save(records[0]['path'])
    records[0]['sha256']=sha256_file(records[0]['path'])
    with pytest.raises(ValueError,match='content SHA-256'):
        build_mixture_controls(records,labels,tmp_path/'changed',n=10)


def test_control_source_subset_reuses_unchanged_multisource_sealed_labels(tmp_path):
    records,labels=pool(tmp_path)
    for i,row in enumerate(records): row['setting_id']='A' if i%2 else 'B'
    before=sha256_file(labels)
    selected=[r for r in records if r['setting_id']=='A']
    result=build_mixture_controls(selected,labels,tmp_path/'source_A',n=4,proportions=[0,.5],repetitions=1)
    assert result['label_provenance']['control_pool_label_subset']['full_label_n']==20
    assert result['label_provenance']['control_pool_label_subset']['matched_n']==10
    assert sha256_file(labels)==before
    assert result['pool_positive_n']==result['pool_negative_n']==5
