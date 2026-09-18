import numpy as np
import pytest

from mms_eval.tail_comparison import select_pair,register_tail_pair,analyze_tail_pair
from mms_eval.utils import sha256_file,write_json,write_jsonl,read_json


def evaluation(root,name,n=1000,entropy=.1,candidates=100):
    records=[{'image_id':f'{name}:{i}','sha256':f'{name}-sha-{i}','cohort_role':'development'} for i in range(n)]
    rows=[{**r,'entropy':entropy,'flag_entropy':i<candidates} for i,r in enumerate(records)]
    write_jsonl(root/'input_manifest.jsonl',records);write_jsonl(root/'scores.jsonl',rows)
    write_json(root/'report.json',{'n':n,'reference_signature':'reference','mms':{'classes':['cat','dog','wild']},
                'manifest_sha256':sha256_file(root/'input_manifest.jsonl'),'scores_sha256':sha256_file(root/'scores.jsonl')})
    return {'setting_id':name,'evaluation':str(root)}


def test_development_selection_threshold_ties_and_no_eligible_pair():
    rows=[{'setting_id':name,'mean_entropy':.1,'MMS':mms} for name,mms in [('c',.3),('b',.3),('a',.1)]]
    result=select_pair(rows,3)
    assert result['selected']==['a','b'] and len(result['candidates'])==3
    rows[1]['mean_entropy']=1.;rows[2]['mean_entropy']=.5
    assert select_pair(rows,3)['selected'] is None


def test_frozen_pair_independent_final_known_differences_and_overlap_refusal(tmp_path):
    sources=[evaluation(tmp_path/'devA','A',candidates=100),evaluation(tmp_path/'devB','B',candidates=300)]
    meta=register_tail_pair({'role':'development','analysis_plan':'registered toy test','sources':sources},tmp_path/'registration')
    assert meta['selected']==['A','B']
    final=[evaluation(tmp_path/'finalA','freshA',n=20,entropy=.1,candidates=0),
           evaluation(tmp_path/'finalB','freshB',n=20,entropy=.2,candidates=20)]
    for item,name in zip(final,['A','B']): item['setting_id']=name
    result=analyze_tail_pair(tmp_path/'registration/tail_registration.json',{'sources':final},tmp_path/'analysis',repetitions=40)
    assert not result['final_mean_match']
    means,mms,human=result['differences']
    assert means['first_minus_second']==pytest.approx(-.1)
    assert mms['first_minus_second']==-1 and mms['ci95']==[-1,-1]
    assert human['ci95'] is None and human['first_minus_second'] is None
    assert result['semantic_evidence_status']=='awaiting_registered_final_human_labels'
    assert (tmp_path/'analysis/entropy_distributions.pdf').is_file()
    final[0]=sources[0]
    with pytest.raises(ValueError,match='independent'):
        analyze_tail_pair(tmp_path/'registration/tail_registration.json',{'sources':final},tmp_path/'bad',repetitions=40)
    meta['selected']=['B','A'];write_json(tmp_path/'registration/tail_registration.json',meta)
    with pytest.raises(ValueError,match='changed'):
        analyze_tail_pair(tmp_path/'registration/tail_registration.json',{'sources':final},tmp_path/'tamper')
