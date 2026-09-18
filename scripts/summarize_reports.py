"""Collect comparable evaluation reports in CSV without silently pooling domains."""
import argparse
import csv
from pathlib import Path

from mms_eval.utils import read_json, sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('reports',nargs='+',help='Paths to report.json files')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    rows=[]
    for name in args.reports:
        path=Path(name).resolve(); r=read_json(path); d=r['distribution']
        row={'report':str(path),'report_sha256':sha256_file(path),'protocol_id':r['protocol_id'],
             'reference_signature':r.get('reference_signature','quality_only_see_real_manifest'),
             'n':r['n'],'MMS':r['mms']['value'],'MMS_status':r['mms']['status'],
             'FID':d['fid'].get('value'),'KID_mean':d['kid'].get('mean'),'KID_std':d['kid'].get('std'),
             'IS_mean':d['is'].get('mean'),'IS_std':d['is'].get('std'),
             'PR_precision':d['precision_recall'].get('precision'),'PR_recall':d['precision_recall'].get('recall'),
             'semantic_status':r['semantic_status']}
        for key in ('fid','kid','is','precision_recall'):
            row[key+'_status']=d[key]['status']; row[key+'_reason']=d[key].get('reason')
        rows.append(row)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} reports to {out}. Compare only matched protocols, reference versions and sample counts.')


if __name__=='__main__': main()
