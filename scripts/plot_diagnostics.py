"""Standalone diagnostic figures from saved scores; no model fitting or selection."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mms_eval.semantic import posterior_scores
from mms_eval.utils import read_json, read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',required=True); p.add_argument('--reference',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); run=Path(a.evaluation); ref_path=Path(a.reference); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    r,ref=read_json(run/'report.json'),read_json(ref_path)
    rows=read_jsonl(run/'scores.jsonl')
    with np.load(ref_path.parent/ref['arrays_file'],allow_pickle=False) as d:
        h=posterior_scores(d['calibration_logits'],ref['config']['temperature'])['entropy']
    g=np.array([x['entropy'] for x in rows]); threshold=ref['calibrations']['entropy']['threshold']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    bins=np.linspace(0,np.log(len(ref['classes'])),45)
    axes[0].hist(h,bins=bins,weights=np.ones(len(h))/len(h),histtype='step',linewidth=1.7,label=f'Real calibration (n={len(h)})',color='#47778a')
    axes[0].hist(g,bins=bins,weights=np.ones(len(g))/len(g),histtype='step',linewidth=1.7,label=f'Generated (n={len(g)})',color='#ca6b3c')
    if isinstance(threshold,(int,float)):
        axes[0].axvline(threshold,linestyle='--',color='#313946',label=f'Threshold {threshold:.4g}')
    axes[0].set_yscale('log'); axes[0].set_xlabel('Predictive entropy (natural log)'); axes[0].set_ylabel('Fraction per bin')
    axes[0].legend(fontsize=8); axes[0].set_title('Fixed real-reference threshold')
    audits=ref['audits']; values=[audits[k]['value'] for k in ('audit_internal','audit_external')]+[r['mms']['value']]
    cis=[audits[k]['ci95'] for k in ('audit_internal','audit_external')]+[r['mms']['ci95']]
    values=np.array(values)*100; cis=np.array(cis)*100
    axes[1].bar(['Internal real','External real','Generated'],values,color=['#47778a','#7ba7b4','#ca6b3c'],width=.6)
    axes[1].errorbar(range(3),values,yerr=np.array([values-cis[:,0],cis[:,1]-values]),fmt='none',ecolor='#26333d',capsize=4)
    axes[1].axhline(ref['config']['alpha']*100,linestyle='--',color='#66717c',linewidth=1)
    for i,v in enumerate(values): axes[1].text(i,cis[i,1]+.7,f'{v:.1f}%',ha='center',fontsize=9)
    axes[1].set_ylim(0,max(cis[:,1])+max(5,max(cis[:,1])*.2)); axes[1].set_ylabel('Candidate flag rate (%)')
    axes[1].set_title('Real audits and generated candidates')
    matrix=np.array(r['mms']['pair_counts_upper_triangle'])
    axes[2].imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
    axes[2].set_xticks(range(len(ref['classes'])),ref['classes']); axes[2].set_yticks(range(len(ref['classes'])),ref['classes'])
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            axes[2].text(j,i,str(matrix[i,j]) if j>i else '—',ha='center',va='center',color='white' if matrix[i,j]>.6*max(1,matrix.max()) else '#26333d')
    axes[2].set_title('Top-two pairs among candidates')
    fig.suptitle('AFHQ 500k · DDIM50 engineering run | Candidate flags are not human-confirmed mixing',fontsize=11)
    fig.savefig(out/'diagnostics.png',dpi=200); fig.savefig(out/'diagnostics.pdf'); plt.close(fig)
    sensitivity=run/'adaptivity'/'adaptivity.json'
    if sensitivity.exists():
        summaries=read_json(sensitivity)['summaries']
        fig,ax=plt.subplots(figsize=(6,3.8),layout='constrained')
        for alpha in sorted({x['alpha'] for x in summaries}):
            part=sorted((x for x in summaries if x['alpha']==alpha),key=lambda x:x['n_reference'])
            x=np.array([z['n_reference'] for z in part]); y=np.array([z['generated']['mean'] for z in part])*100
            low=np.array([z['generated']['min'] for z in part])*100; high=np.array([z['generated']['max'] for z in part])*100
            ax.plot(x,y,marker='o',label=f'alpha={alpha:g}'); ax.fill_between(x,low,high,alpha=.13)
        ax.set_xlabel('Real calibration sample count'); ax.set_ylabel('Generated candidate fraction (%)')
        ax.set_title('Reference sensitivity; bands show resampling min–max'); ax.legend()
        fig.savefig(out/'reference_sensitivity.png',dpi=200); fig.savefig(out/'reference_sensitivity.pdf'); plt.close(fig)
    print(out)


if __name__=='__main__': main()
