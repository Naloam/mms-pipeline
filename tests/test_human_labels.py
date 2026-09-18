import csv

import pytest

from mms_eval.utils import write_json, write_jsonl, sha256_file, read_jsonl


def test_final_labels_require_independent_judgments_and_preserve_unknowns(tmp_path):
    from mms_eval.human_labels import finalize_annotations
    key = [{'blind_id': str(i), 'image_id': f'image-{i}', 'sha256':f'content-{i}'} for i in range(4)]
    write_jsonl(tmp_path/'pack'/'private_key.jsonl', key)
    write_json(tmp_path/'pack'/'pack.json', {'n': 4, 'private_key_sha256': sha256_file(tmp_path/'pack'/'private_key.jsonl')})
    raw = tmp_path/'raw.csv'
    with raw.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['display_id', 'rater_id', 'mm_label', 'other_defects'])
        for row in [('0','A','yes','blur'), ('0','B','yes','blur'), ('1','A','no',''),
                    ('1','B','yes',''), ('2','A','unknown',''), ('2','B','unknown',''), ('3','A','no','')]:
            w.writerow(row)
    pending = finalize_annotations(tmp_path/'pack', [raw], tmp_path/'pending')
    assert pending['status'] == 'awaiting_adjudication_or_second_rater'
    rows = read_jsonl(tmp_path/'pending'/'final_labels.jsonl')
    assert [r['mm_label_final'] for r in rows] == ['yes', '', 'unknown', '']
    from mms_eval.analysis_reporting import load_final_labels
    with pytest.raises(ValueError,match='completed finalization'):
        load_final_labels(tmp_path/'pending/final_labels.csv')
    assert rows[0]['other_defects'] == 'blur'
    assert pending['agreement']['items_with_multiple_raters'] == 3
    assert pending['agreement']['missing_ratings'] == 1
    adjudications = tmp_path/'decisions.csv'
    adjudications.write_text('blind_id,mm_label_final,decision_method,reason\n1,no,third_rater,complete separate subjects\n3,no,single_rater_exception,second rater unavailable recorded exception\n')
    complete = finalize_annotations(tmp_path/'pack', [raw], tmp_path/'complete', adjudications=adjudications)
    assert complete['status'] == 'complete_with_recorded_exceptions'
    rows = read_jsonl(tmp_path/'complete'/'final_labels.jsonl')
    assert [r['mm_label_final'] for r in rows] == ['yes', 'no', 'unknown', 'no']
    assert raw.read_text().count('yes') == 3
    assert rows[0]['sha256']=='content-0'


def test_registered_annotation_guide_must_match_and_classes_need_independent_agreement(tmp_path):
    from mms_eval.human_labels import finalize_annotations
    key=[{'image_id':'image','blind_id':'1','sha256':'content'}]
    write_jsonl(tmp_path/'pack/private_key.jsonl',key)
    write_json(tmp_path/'pack/pack.json',{'private_key_sha256':sha256_file(tmp_path/'pack/private_key.jsonl'),
                                        'annotation_guide_version':'v2'})
    raw=tmp_path/'raw.csv'
    raw.write_text('blind_id,annotator_id,label,human_target_class,annotation_guide_version\n1,A,no,cat,v1\n1,B,no,dog,v2\n')
    with pytest.raises(ValueError,match='guide version'):
        finalize_annotations(tmp_path/'pack',raw,tmp_path/'bad')
    raw.write_text(raw.read_text().replace(',v1',',v2'))
    finalize_annotations(tmp_path/'pack',raw,tmp_path/'good')
    row=read_jsonl(tmp_path/'good/final_labels.jsonl')[0]
    assert row['mm_label_final']=='no' and row['human_target_class']=='unknown'


def test_nominal_krippendorff_treats_unknown_as_category_and_missing_as_absent():
    from mms_eval.human_labels import nominal_agreement
    perfect = nominal_agreement({'A': {'x': 1, 'y': 0, 'z': None}, 'B': {'x': 1, 'y': 0, 'z': None}}, ['x','y','z'])
    assert perfect['krippendorff_alpha_nominal'] == 1
    assert perfect['missing_ratings'] == 0
    partial = nominal_agreement({'A': {'x': 1, 'y': 0}, 'B': {'x': 0}}, ['x','y'])
    assert partial['missing_ratings'] == 1
    assert partial['items_with_multiple_raters'] == 1
    assert partial['krippendorff_alpha_nominal'] == 0
