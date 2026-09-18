"""Independent rerun of saved statistics and batch-size stability on real outputs."""
import argparse
from pathlib import Path

import numpy as np

from mms_eval.distribution import FidelityExtractor, compute_distribution_metrics
from mms_eval.evaluator import predict_paths
from mms_eval.images import canonicalize_records, collect_images
from mms_eval.pipeline import load_reference
from mms_eval.semantic import aggregate_mms, posterior_scores
from mms_eval.utils import read_json, sha256_file, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    root=Path(args.evaluation).resolve(); report=read_json(root/'report.json')
    ref_path=Path(report['reference']); ref=load_reference(ref_path)
    assert sha256_file(root/'features.npz')==report['features_sha256']
    with np.load(root/'features.npz',allow_pickle=False) as d:
        generated={k:d[k] for k in d.files}
    with np.load(ref_path.parent/ref['arrays_file'],allow_pickle=False) as d:
        real=d['real_features']
    dist=compute_distribution_metrics(real,generated['features'],generated['logits'],config=ref['config']['metrics'])
    deltas={}
    for name in ('fid','kid','is','precision_recall'):
        assert dist[name]['status']==report['distribution'][name]['status']
        for field in ('value','mean','std','precision','recall'):
            if field in dist[name]:
                delta=abs(dist[name][field]-report['distribution'][name][field]); deltas[name+'_'+field]=delta
                assert delta < 1e-10, (name,field,delta)
    mms=aggregate_mms(posterior_scores(generated['semantic_logits'],ref['config']['temperature']),ref['calibrations'],ref['classes'])
    assert mms['candidate_count']==report['mms']['candidate_count']
    records=collect_images(root/'input_manifest.jsonl')[:8]
    canon=canonicalize_records(records,root/'cache'/'rgb',ref['image_policy'])
    paths=[r['path'] for r in canon]
    checkpoint=ref['evaluator_checkpoint']
    a=predict_paths(paths,checkpoint,device=args.device,batch_size=1)['logits']
    b=predict_paths(paths,checkpoint,device=args.device,batch_size=4)['logits']
    np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(b,generated['semantic_logits'][:len(paths)],rtol=1e-4,atol=1e-4)
    pa,pb=posterior_scores(a),posterior_scores(b)
    assert aggregate_mms(pa,ref['calibrations'],ref['classes'])['candidate_count']==aggregate_mms(pb,ref['calibrations'],ref['classes'])['candidate_count']
    extractor=FidelityExtractor(args.device,1)
    q1=extractor.extract_paths(paths)
    extractor.batch_size=4
    q4=extractor.extract_paths(paths)
    # Convolution reductions vary slightly by batch shape; bound actual differences.
    np.testing.assert_allclose(q1['features'],q4['features'],rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(q1['logits'],q4['logits'],rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(q4['features'],generated['features'][:len(paths)],rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(q4['logits'],generated['logits'][:len(paths)],rtol=2e-4,atol=2e-4)
    result={'status':'passed','n':report['n'],'report_sha256':sha256_file(root/'report.json'),
            'offline_metric_absolute_differences':deltas,'offline_MMS_candidate_count':mms['candidate_count'],
            'batch_check_n':len(paths),'batch_sizes':[1,4],
            'semantic_logits_max_absolute_difference':float(np.abs(a-b).max()),
            'entropy_max_absolute_difference':float(np.abs(pa['entropy']-pb['entropy']).max()),
            'inception_features_max_absolute_difference':float(np.abs(q1['features']-q4['features']).max()),
            'inception_logits_max_absolute_difference':float(np.abs(q1['logits']-q4['logits']).max()),
            'cached_batch32_vs_batch4_semantic_max_absolute_difference':float(np.abs(b-generated['semantic_logits'][:len(paths)]).max()),
            'cached_batch32_vs_batch4_inception_features_max_absolute_difference':float(np.abs(q4['features']-generated['features'][:len(paths)]).max()),
            'scope':'Offline full-cohort recomputation and first-eight-image batch stability; not cross-hardware bitwise equivalence'}
    write_json(root/'verification.json',result)
    print(result)


if __name__=='__main__': main()
