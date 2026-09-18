"""Prespecified 1k historical-image control using the existing frozen evaluation."""
from pathlib import Path
import numpy as np
from mms_eval.pipeline import evaluate
from mms_eval.images import collect_images
from mms_eval.utils import read_json, write_json, write_jsonl

source=Path('/root/autodl-tmp/DDPM/eval_runs/afhq256_uncond_500k_50kset/images_all')
root=Path('/root/autodl-tmp/mms_20260911/artifacts/afhq')
out=root/'historical_ddpm250_comparison'
files=sorted(p for p in source.glob('*.png') if p.is_file())
if len(files)!=50010: raise RuntimeError(f'Historical population is not intact: {len(files)}')
seed=2026091107
indices=np.random.RandomState(seed).choice(len(files),1000,replace=False)
records=collect_images([{'path':str(files[i]),'image_id':'historic50010:'+files[i].name,'setting_id':'historical_DDPM250'} for i in indices])
write_jsonl(out/'input.jsonl',records)
write_json(out/'selection.json',{'source_population':str(source),'population_n':len(files),'selected_n':1000,'selection_seed':seed,'indices':indices.tolist(),'selection':'uniform_without_replacement_before_inspecting_scores'})
r=evaluate(out/'input.jsonl',root/'reference_A_fp32'/'reference.json',out/'evaluation',device='cuda:0',batch_size=32)
print({k:r['distribution'][k].get('value',r['distribution'][k].get('mean')) for k in ['fid','kid','is']},flush=True)
print({'MMS':r['mms']['value'],'PR':{k:r['distribution']['precision_recall'][k] for k in ['precision','recall']}},flush=True)
print('HISTORICAL_COMPARISON_COMPLETE',flush=True)
