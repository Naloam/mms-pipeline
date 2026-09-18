"""Finish the registered 1,000-image engineering cohort after sampling completes."""
import subprocess
import sys
from pathlib import Path

from mms_eval.analysis import adaptivity_audit
from mms_eval.annotation import annotation_pack
from mms_eval.pipeline import evaluate
from mms_eval.utils import read_json

root=Path('/root/autodl-tmp/mms_20260911/artifacts/afhq')
sampling=read_json(root/'ddim50_engineering'/'sampling.json')
if sampling['status']!='complete' or sampling['n']!=1000:
    raise RuntimeError('The registered cohort must finish before final evaluation')
ref=root/'reference_A_fp32'/'reference.json'; run=root/'evaluations'/'ddim50_n1000_fp32'
report=evaluate(root/'ddim50_engineering'/'manifest.jsonl',ref,run,device='cuda:0',batch_size=32)
print('FULL_EVALUATION_COMPLETE',flush=True)
adaptivity_audit(ref,run,run/'adaptivity')
if not (root/'annotation_200_fp32'/'pack.json').exists():
    annotation_pack(run,root/'annotation_200_fp32',count=200)
print('ADAPTIVITY_AND_BLIND_PACK_COMPLETE',flush=True)
subprocess.run([sys.executable,'scripts/verify_real_run.py','--evaluation',str(run)],check=True)
subprocess.run([sys.executable,'scripts/stratify_real_audit.py','--reference',str(ref),'--out',str(root/'real_audit_by_class.json')],check=True)
subprocess.run([sys.executable,'scripts/summarize_reports.py',str(run/'report.json'),
                str(root/'evaluations'/'ddpm1000_n8_fp32'/'report.json'),str(root/'evaluations'/'single_image_fp32'/'report.json'),
                '--out',str(root/'summary.csv')],check=True)
print('AFHQ_1000_ACCEPTANCE_COMPLETE',flush=True)
