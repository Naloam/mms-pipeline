"""One concrete AFHQ acceptance run; paths are explicit and outputs remain separate."""
import json
from pathlib import Path

from mms_eval.pipeline import build_reference, evaluate
from mms_eval.utils import read_json, write_json

root=Path('/root/autodl-tmp/mms_20260911/artifacts/afhq')
ref=root/'reference_A_fp32'/'reference.json'
build_reference(root/'splits',root/'evaluator_A'/'best.pt',ref.parent,
                config=read_json('configs/afhq.json'),device='cuda:0',batch_size=32)
for label,source in [('ddpm1000_n8_fp32',root/'ddpm1000_engineering'/'manifest.jsonl'),
                     ('single_image_fp32',root/'ddim50_engineering'/'images'/'0000000000.png')]:
    report=evaluate(source,ref,root/'evaluations'/label,device='cuda:0',batch_size=32)
    print(json.dumps({'completed':label,'n':report['n'],'MMS':report['mms']['value']}),flush=True)
print('REFERENCE_AND_SMALL_RUNS_COMPLETE',flush=True)
