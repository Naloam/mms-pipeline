"""Compare the existing DDIM50 and frozen DDPM250 1k cohorts under delivery v5."""
import argparse
from pathlib import Path
import subprocess

import numpy as np

from mms_eval.artifacts import load_evaluation
from mms_eval.comparison import compare_evaluations
from mms_eval.images import collect_images
from mms_eval.pipeline import code_signature, evaluate, freeze_request, load_reference
from mms_eval.utils import read_json, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Existing artifacts/afhq directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    ref_path = root/'reference_A_delivery_v5/reference.json'
    ref = load_reference(ref_path)
    assert ref['code_sha256'] == code_signature(), 'Use the frozen delivery modules'
    sources = {
        'DDPM250': root/'scale_acceptance_clean_v2/input_1000.jsonl',
        'DDIM50': root/'ddim50_engineering/manifest.jsonl',
    }
    records = {name: collect_images(path) for name, path in sources.items()}
    for name, rows in records.items():
        assert len(rows) == 1000, f'{name} must contain exactly the existing 1,000 images'
        assert len({row['sha256'] for row in rows}) == 1000, f'{name} contains duplicate files'
    assert not ({r['sha256'] for r in records['DDPM250']} &
                {r['sha256'] for r in records['DDIM50']}), 'Cohorts share image contents'
    sampling_path = root/'ddim50_engineering/sampling.json'
    sampling = read_json(sampling_path)
    assert sampling['status'] == 'complete' and sampling['n'] == 1000
    assert sampling['sampler'] == 'ddim' and sampling['steps'] == 50
    assert sampling['manifest_sha256'] == sha256_file(sources['DDIM50'])
    current_weight = Path(sampling['weight_path'])
    assert sha256_file(current_weight) == sampling['checkpoint_sha256']
    historical_metadata = []
    for folder in sorted({Path(row['path']).parent.parent for row in records['DDPM250']}):
        path = folder/'metadata.json'
        meta = read_json(path)
        assert Path(meta['model_path']).resolve() == current_weight.resolve()
        assert str(meta['timestep_respacing']) == '250' and meta['use_ddim'] is False
        assert meta['clip_denoised'] is True and meta['color_calibration'] is False
        historical_metadata.append({'path': str(path), 'sha256': sha256_file(path), 'metadata': meta})
    request = {
        'experiment': 'existing_DDPM250_vs_DDIM50_common1000_A_clean_v5',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': sha256_file(__file__), 'module_code_sha256': code_signature(),
        'reference_sha256': sha256_file(ref_path), 'reference_signature': ref['signature'],
        'sources': {name: {'path': str(path), 'sha256': sha256_file(path), 'n': 1000}
                    for name, path in sources.items()},
        'selection': 'All 1000 existing DDIM images; unchanged previously frozen DDPM 1k subset. No score-based reselection.',
        'pairing': 'Unpaired cohorts. Historical DDPM per-image noise seeds are not established; do not join by numeric seed.',
        'generator_checkpoint': {'path': str(current_weight), 'current_sha256': sampling['checkpoint_sha256'],
                                 'historical_limit': 'DDPM metadata names the same 500k path; historical weight bytes are not independently authenticated by path equality.'},
        'ddim_sampling': sampling, 'ddpm_metadata': historical_metadata,
        'scientific_scope': 'Descriptive same-N sampler-setting comparison, not independent model-family replication or human MM validation.',
    }
    freeze_request(out/'comparison_request.json', request)
    cache = root/'shared_feature_cache_v2'
    reports = {}
    for name, source in sources.items():
        print(f'EVALUATING {name}: 1000 existing images', flush=True)
        reports[name] = evaluate(source, ref_path, out/name, device=args.device, cache_dir=cache)
        load_evaluation(out/name, features=True)
        print(f'COMPLETE {name}: MMS={reports[name]["mms"]["value"]}', flush=True)
    old = read_json(root/'scale_acceptance_clean_v2/n1000/report.json')
    with np.load(root/'scale_acceptance_clean_v2/n1000/features.npz', allow_pickle=False) as before, \
            np.load(out/'DDPM250/features.npz', allow_pickle=False) as after:
        identical_logits = np.array_equal(before['semantic_logits'], after['semantic_logits'])
    assert identical_logits and reports['DDPM250']['mms'] == old['mms']
    assert reports['DDPM250']['distribution'] == old['distribution']
    comparison = compare_evaluations(
        [{'name': name, 'path': str(out/name),
          'generator_settings': {'sampler': name, 'cohort': 'existing_unpaired_engineering',
                                 'training_step_recorded': 500000}}
         for name in sources], out/'comparison', mode='models')
    proof = {
        'status': 'complete', 'request_sha256': sha256_file(out/'comparison_request.json'),
        'same_reference_and_quality_protocol': True, 'common_n': 1000,
        'DDPM_previous_logits_mms_distribution_exactly_reproduced': True,
        'results': comparison['rows'],
        'candidates': {name: {'count': r['mms']['candidate_count'], 'ci95': r['mms']['ci95']}
                       for name, r in reports.items()},
        'cache': {name: {kind: {k: r['feature_cache'][kind][k]
                                for k in ['computed_rows', 'reused_rows']}
                         for kind in ['semantic', 'quality']} for name, r in reports.items()},
        'limitations': [request['pairing'], request['scientific_scope'],
                        'KID/IS subset standard deviations are not confidence intervals or a test of sampler differences.',
                        'PR uses only 1000 real and 1000 generated images in this equal-N comparison.'],
    }
    write_json(out/'completion.json', proof)
    print('SAMPLER_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
