"""Nested historical-image scale acceptance with a real killed extraction process."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from mms_eval.images import collect_images
from mms_eval.pipeline import build_reference, evaluate
from mms_eval.utils import read_json, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root/'scale_acceptance_clean_v2'
    cache = root/'shared_feature_cache_v2'
    reference = root/'reference_A_clean_engineering_v2'/'reference.json'
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.images).glob('*.png'))
    if len(files) != 50010:
        raise ValueError(f'Expected the preserved 50,010-image population; found {len(files)}')
    seed = 2026091110
    order = np.random.RandomState(seed).permutation(len(files))
    records = collect_images([{'path': str(files[i]), 'image_id': 'historic50010:'+files[i].name,
                               'setting_id': 'AFHQ_DDPM250_500k', 'cohort_role': 'engineering_historical_scale'} for i in order])
    selection = {'population_n': len(files), 'selection_seed': seed, 'order': order.tolist(),
                 'sampling': 'one fixed permutation, nested prefixes; no score selection',
                 'role': 'engineering_historical_scale'}
    if (out/'selection.json').exists() and read_json(out/'selection.json') != selection:
        raise ValueError('Scale selection was already frozen differently')
    write_json(out/'selection.json', selection)
    write_jsonl(out/'population.jsonl', records)
    for n in (1000,5000,10000,50010):
        write_jsonl(out/f'input_{n}.jsonl', records[:n])
    config = read_json(args.config)
    build_reference(root/'splits', root/'evaluator_A'/'best.pt', reference.parent,
                    config=config, device=args.device, batch_size=32, cache_dir=cache)
    # The child has a separate CUDA context and is terminated with os._exit;
    # no Python cleanup or in-memory state survives the interruption.
    proof_path = out/'resume_proof.json'
    results = []
    for n in (1000,5000,10000,50010):
        target = out/f'n{n}'
        if n == 5000 and not proof_path.exists():
            code = '''import os,sys
from mms_eval import pipeline
original=pipeline.make_quality_extractor
def factory(*args,**kwargs):
    extractor=original(*args,**kwargs)
    extract=extractor.extract_paths
    calls=[0]
    def interrupted(paths):
        if calls[0] == 8:
            print('FORCED_PROCESS_EXIT_AFTER_COMMITTED_SHARDS',flush=True)
            os._exit(75)
        calls[0]+=1
        return extract(paths)
    extractor.extract_paths=interrupted
    return extractor
pipeline.make_quality_extractor=factory
pipeline.evaluate(sys.argv[1],sys.argv[2],sys.argv[3],device=sys.argv[4],cache_dir=sys.argv[5])
'''
            import torch
            torch.cuda.empty_cache()
            child = subprocess.run([sys.executable, '-u', '-c', code, str(out/f'input_{n}.jsonl'),
                                    str(reference), str(target), args.device, str(cache)])
            if child.returncode != 75:
                raise RuntimeError(f'Expected deliberate exit 75 for resume verification; received {child.returncode}')
            write_json(proof_path, {'forced_exit_code': child.returncode, 'complete_new_shards_before_exit': 8,
                                    'chunk_size': 256, 'resumed': False})
        started = time.perf_counter()
        report = evaluate(out/f'input_{n}.jsonl', reference, target, device=args.device, cache_dir=cache)
        if n == 5000:
            proof = read_json(proof_path)
            reused = report['feature_cache']['quality']['reused_rows']
            if reused < 2048:
                raise AssertionError(f'Interrupted extraction did not reuse committed rows: {reused}')
            proof.update(resumed=True, reused_quality_rows=reused, resulting_n=report['n'],
                         quality_cache=report['feature_cache']['quality'])
            write_json(proof_path, proof)
        result = {'n': n, 'elapsed_seconds_this_attempt': time.perf_counter()-started,
                  'MMS': report['mms']['value'], 'FID': report['distribution']['fid'].get('value'),
                  'KID': report['distribution']['kid'].get('mean'), 'IS': report['distribution']['is'].get('mean'),
                  'PR': report['distribution']['precision_recall'], 'runtime': report['runtime'],
                  'semantic_reused_rows': report['feature_cache']['semantic']['reused_rows'],
                  'quality_reused_rows': report['feature_cache']['quality']['reused_rows']}
        results.append(result)
        write_json(out/'scale_results.json', {'status': 'complete' if n == 50010 else 'running', 'results': results,
                                            'reference': str(reference), 'selection': selection})
        print({'stage_complete': n, 'MMS': result['MMS'], 'FID': result['FID']}, flush=True)
    print('SCALE_ACCEPTANCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
