"""Plot stored sensitivity records without refitting or selecting parameters."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mms_eval.semantic import SCORE_NAMES
from mms_eval.utils import read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    source,out=Path(args.input),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    fields=('generated_flag_rate','audit_internal_flag_rate','audit_external_flag_rate')
    for filename,xkey,name in [('temperature_rows.jsonl','temperature','temperature'),
                               ('alpha_rows.jsonl','alpha','alpha'),
                               ('adaptivity_rows.jsonl','n_reference','reference_size')]:
        rows=read_jsonl(source/filename)
        if name=='reference_size': rows=[r for r in rows if r['alpha']==.05]
        fig,axes=plt.subplots(3,5,figsize=(15,8),constrained_layout=True)
        for column,score in enumerate(SCORE_NAMES):
            selected=[r for r in rows if r['score']==score]
            conditions=sorted({r.get('condition','registered') for r in selected})
            for condition in conditions:
                subset=[r for r in selected if r.get('condition','registered')==condition]
                x=sorted({r[xkey] for r in subset})
                for row,field in enumerate(fields):
                    values=[np.array([r[field] for r in subset if r[xkey]==v],dtype=float) for v in x]
                    ax=axes[row,column]
                    ax.plot(x,[np.median(v) for v in values],'o-',label=condition)
                    if any(len(v)>1 for v in values):
                        ax.fill_between(x,[v.min() for v in values],[v.max() for v in values],alpha=.15)
                    ax.set(xlabel=xkey,ylabel=field if column==0 else '',ylim=(0,1));ax.grid(alpha=.2)
            axes[0,column].set_title(score.replace('_',' '),fontsize=8)
            axes[0,column].legend(fontsize=6)
        fig.suptitle(name+' — all registered scores; shaded reference ranges are descriptive, not confidence intervals',fontsize=10)
        fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.pdf');plt.close(fig)


if __name__=='__main__': main()
