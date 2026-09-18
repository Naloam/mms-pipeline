"""Verify that the delivery source reuses extraction and preserves prior numerical results."""
import argparse
from pathlib import Path
import numpy as np

from mms_eval.pipeline import build_reference,evaluate
from mms_eval.utils import read_json,write_json,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--evaluator',choices=['A','B'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--device',required=True);args=p.parse_args()
    root,name=Path(args.root),args.evaluator
    cache=root/('shared_feature_cache_v2' if name=='A' else 'shared_feature_cache_B_v2')
    reference=root/f'reference_{name}_delivery_v5'
    ref=build_reference(root/'splits',root/f'evaluator_{name}/best.pt',reference,config=read_json(args.config),
                        device=args.device,cache_dir=cache)
    oldref=read_json(root/f'reference_{name}_clean_engineering_v2/reference.json')
    if ref['calibrations']!=oldref['calibrations']: raise AssertionError('Calibration drift in delivery source')
    report=evaluate(root/'scale_acceptance_clean_v2/input_5000.jsonl',reference/'reference.json',root/f'evaluation_{name}_delivery_5000_v5',
                    device=args.device,cache_dir=cache)
    oldroot=root/('scale_acceptance_clean_v2/n5000' if name=='A' else 'evaluation_B_clean_5000_v2')
    old=read_json(oldroot/'report.json')
    with np.load(oldroot/'features.npz',allow_pickle=False) as before,np.load(root/f'evaluation_{name}_delivery_5000_v5/features.npz',allow_pickle=False) as after:
        same=np.array_equal(before['semantic_logits'],after['semantic_logits'])
    if not same or report['mms']!=old['mms'] or report['distribution']!=old['distribution']:
        raise AssertionError('Numerical delivery result differs from the frozen acceptance result')
    proof={'evaluator':name,'n':report['n'],'calibrations_identical':True,'semantic_logits_identical':same,
           'mms_identical':True,'distribution_identical':True,'new_reference':str(reference/'reference.json'),
           'new_report_sha256':sha256_file(root/f'evaluation_{name}_delivery_5000_v5/report.json'),
           'original_report_sha256':sha256_file(oldroot/'report.json'),'feature_cache':report['feature_cache'],
           'status':'delivery_source_numerically_verified_human_validity_pending'}
    write_json(root/f'delivery_{name}_v5_proof.json',proof)
    print('DELIVERY_'+name+'_VERIFIED',flush=True)


if __name__=='__main__':main()
