"""Per-category real audit: diagnostics, without replacing the pooled MMS rule."""
import argparse
from pathlib import Path

import numpy as np

from mms_eval.pipeline import load_reference
from mms_eval.semantic import SCORE_NAMES, apply_threshold, posterior_scores, wilson_interval
from mms_eval.utils import read_jsonl, write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--reference',required=True); p.add_argument('--out',required=True)
    args=p.parse_args(); path=Path(args.reference); ref=load_reference(path); result={}
    with np.load(path.parent/ref['arrays_file'],allow_pickle=False) as arrays:
        for split in ('audit_internal','audit_external'):
            records=read_jsonl(path.parent/f'{split}.jsonl'); labels=np.array([r['label'] for r in records])
            scores=posterior_scores(arrays[split+'_logits'],ref['config']['temperature']); part=[]
            for i,name in enumerate(ref['classes']):
                mask=labels==i; n=int(mask.sum()); row={'class':name,'n':n,
                    'classification_accuracy':float((scores['class_order'][mask,0]==i).mean()) if n else None}
                for score_name in SCORE_NAMES:
                    flags=apply_threshold(scores[score_name],ref['calibrations'][score_name]); count=int(flags[mask].sum())
                    row[score_name]={'flag_count':count,'flag_rate':count/n if n else None,'ci95':wilson_interval(count,n)}
                part.append(row)
            result[split]=part
    write_json(args.out,{'reference_signature':ref['signature'],'by_class':result,
               'interpretation':'Pooled calibration gives a marginal rule, not a per-class guarantee. These class-stratified audits diagnose differences; they do not reweight or replace the main MMS.'})


if __name__=='__main__': main()
