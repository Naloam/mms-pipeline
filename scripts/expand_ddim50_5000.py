"""Preserve the old DDIM 1k, sample disjoint additional seeds, then compare A/B at 5k."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from mms_eval.artifacts import load_evaluation
from mms_eval.comparison import compare_evaluations
from mms_eval.images import collect_images
from mms_eval.pipeline import code_signature, evaluate, freeze_request, load_reference
from mms_eval.sampling import ADM_CONFIG, AFHQ_500K_SHA256
from mms_eval.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


SHARDS = [('seeds_1000_2999', 1000, 2000), ('seeds_3000_4999', 3000, 2000)]
CONTRACT_FIELDS = ('sampler_version', 'checkpoint_sha256', 'sampler', 'steps', 'ddim_eta',
                   'model', 'role', 'setting_id', 'clip_denoised', 'quantization',
                   'initial_noise', 'code_sha256', 'torch_version')


def run_parallel(jobs, out, stage):
    """Each worker owns a separate output/cache; completed shards survive a restart."""
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        processes = []
        for name, command in jobs:
            log = out/f'{name}.log'
            stream = stack.enter_context(log.open('a', encoding='utf-8'))
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((name, process))
        write_json(out/f'{stage}_workers.json', {
            'stage': stage, 'workers': [{'name': name, 'pid': p.pid} for name, p in processes]})
        last = None
        while True:
            progress = {'stage': stage, 'elapsed_seconds': round(time.monotonic()-started, 1),
                        'workers': {name: p.poll() for name, p in processes}}
            if stage == 'sampling':
                progress['new_image_counts'] = {
                    name: len(read_jsonl(out/'shards'/name/'manifest.jsonl'))
                    if (out/'shards'/name/'manifest.jsonl').exists() else 0
                    for name, _, _ in SHARDS}
            write_json(out/'progress.json', progress)
            state = (progress['workers'], progress.get('new_image_counts'))
            if state != last:
                print(progress, flush=True)
                last = state
            if all(p.poll() is not None for _, p in processes):
                break
            time.sleep(30)
        failures = {name: p.returncode for name, p in processes if p.returncode != 0}
        if failures:
            raise RuntimeError(f'{stage} workers failed; retain logs and resume after repair: {failures}')


def preflight(root, out, devices):
    import torch

    old_manifest = root/'ddim50_engineering/manifest.jsonl'
    old_sampling = root/'ddim50_engineering/sampling.json'
    old = read_json(old_sampling)
    records = collect_images(old_manifest)
    assert old['status'] == 'complete' and old['n'] == len(records) == 1000
    assert [r['seed'] for r in records] == list(range(1000))
    assert old['sampler'] == 'ddim' and old['steps'] == 50 and old['batch_size'] == 4
    assert old['model'] == ADM_CONFIG and old['torch_version'] == str(torch.__version__)
    assert sha256_file(old_manifest) == old['manifest_sha256']
    assert sha256_file(old['weight_path']) == old['checkpoint_sha256'] == AFHQ_500K_SHA256
    for filename, digest in old['code_sha256'].items():
        assert sha256_file(Path(old['repository'])/'guided_diffusion'/filename) == digest
    assert sorted(seed for _, start, count in SHARDS for seed in range(start, start+count)) == list(range(1000, 5000))
    references, baselines = {}, {}
    for name in ['A', 'B']:
        refpath = root/f'reference_{name}_delivery_v5/reference.json'
        ref = load_reference(refpath)
        assert ref['code_sha256'] == code_signature()
        path = root/f'evaluation_{name}_delivery_5000_v5'
        report, _, _, _ = load_evaluation(path)
        assert report['n'] == 5000 and report['reference_signature'] == ref['signature']
        assert sha256_file(path/'report.json') == read_json(root/f'delivery_{name}_v5_proof.json')['new_report_sha256']
        references[name] = {'path': str(refpath), 'sha256': sha256_file(refpath)}
        baselines[name] = {'path': str(path), 'report_sha256': sha256_file(path/'report.json')}
    initial_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    request_path = out/'expansion_request.json'
    if request_path.exists():
        initial_commit = read_json(request_path)['initial_git_commit']
    request = {
        'experiment': 'DDIM50_existing1k_plus_fixed4k_vs_DDPM250_5k_AB',
        'initial_git_commit': initial_commit, 'script_sha256': sha256_file(__file__),
        'module_code_sha256': code_signature(), 'old_manifest': str(old_manifest),
        'old_manifest_sha256': sha256_file(old_manifest),
        'old_sampling_sha256': sha256_file(old_sampling), 'original_sampling': old,
        'new_shards': [{'name': n, 'seed_start': s, 'count': c, 'device': d, 'batch_size': 4}
                       for (n, s, c), d in zip(SHARDS, devices)],
        'references': references, 'DDPM250_baselines': baselines,
        'selection': 'Preserve old seeds 0..999; add all seeds 1000..4999, with no score-based selection.',
        'pairing': 'Sampler cohorts are unpaired; historical DDPM noise identities are not established.',
        'scope': 'Same fixed checkpoint path/config; descriptive sampler-setting comparison, human MM validity pending.',
    }
    freeze_request(request_path, request)
    return request


def merge_shards(root, out, request):
    old = request['original_sampling']
    assert sha256_file(request['old_manifest']) == request['old_manifest_sha256']
    assert sha256_file(root/'ddim50_engineering/sampling.json') == request['old_sampling_sha256']
    original = collect_images(request['old_manifest'])
    records = [{**r, 'sampling_source': 'original_1000'} for r in original]
    proof = []
    for name, start, count in SHARDS:
        folder = out/'shards'/name
        metadata = read_json(folder/'sampling.json')
        assert metadata['status'] == 'complete' and metadata['n'] == count
        assert metadata['seed_start'] == start and metadata['batch_size'] == 4
        assert {k: metadata[k] for k in CONTRACT_FIELDS} == {k: old[k] for k in CONTRACT_FIELDS}
        assert metadata['original_timestep_map'] == old['original_timestep_map']
        assert sha256_file(folder/'manifest.jsonl') == metadata['manifest_sha256']
        rows = collect_images(folder/'manifest.jsonl')
        assert [r['seed'] for r in rows] == list(range(start, start+count))
        records.extend({**r, 'sampling_source': name} for r in rows)
        proof.append({'name': name, 'n': len(rows), 'metadata': metadata,
                      'manifest_sha256': sha256_file(folder/'manifest.jsonl')})
    records.sort(key=lambda r: r['seed'])
    assert len(records) == 5000 and [r['seed'] for r in records] == list(range(5000))
    assert len({r['image_id'] for r in records}) == len({r['sha256'] for r in records}) == 5000
    assert [(r['image_id'], r['sha256']) for r in records[:1000]] == [(r['image_id'], r['sha256']) for r in original]
    ddpm = collect_images(root/'scale_acceptance_clean_v2/input_5000.jsonl')
    assert not ({r['sha256'] for r in ddpm} & {r['sha256'] for r in records})
    freeze_request(out/'merged_cohort_request.json', {'records': records, 'sampling_sources': proof})
    write_jsonl(out/'ddim50_5000.jsonl', records)
    write_json(out/'cohort_verification.json', {
        'n': 5000, 'original_1000_preserved': True, 'new_images': 4000,
        'seeds_exactly_0_to_4999': True, 'unique_file_contents': 5000,
        'no_file_content_overlap_with_DDPM250_5000': True,
        'scientific_sampling_contracts_match_excluding_shard_seed_start': True,
        'manifest_sha256': sha256_file(out/'ddim50_5000.jsonl')})


def evaluate_one(root, out, evaluator, device):
    cache = root/('shared_feature_cache_v2' if evaluator == 'A' else 'shared_feature_cache_B_v2')
    report = evaluate(out/'ddim50_5000.jsonl', root/f'reference_{evaluator}_delivery_v5/reference.json',
                      out/f'evaluation_DDIM50_{evaluator}', device=device, cache_dir=cache)
    load_evaluation(out/f'evaluation_DDIM50_{evaluator}', features=True)
    print(f'EVALUATOR_{evaluator}_COMPLETE MMS={report["mms"]["value"]}', flush=True)


def compare_all(root, out):
    results = {}
    for name in ['A', 'B']:
        results[name] = compare_evaluations([
            {'name': 'DDPM250', 'path': str(root/f'evaluation_{name}_delivery_5000_v5')},
            {'name': 'DDIM50', 'path': str(out/f'evaluation_DDIM50_{name}')},
        ], out/f'comparison_{name}', mode='models')
    ab = compare_evaluations([
        {'name': name, 'path': str(out/f'evaluation_DDIM50_{name}')}
        for name in ['A', 'B']], out/'comparison_DDIM50_AB', mode='evaluators')
    rows = [load_evaluation(out/f'evaluation_DDIM50_{name}')[1] for name in ['A', 'B']]
    a, b = [{r['image_id'] for r in table if r['flag_entropy']} for table in rows]
    write_json(out/'completion.json', {
        'status': 'complete', 'common_n': 5000,
        'request_sha256': sha256_file(out/'expansion_request.json'),
        'merged_manifest_sha256': sha256_file(out/'ddim50_5000.jsonl'),
        'same_N_reference_quality_contract_verified_per_evaluator': True,
        'comparisons': {name: value['rows'] for name, value in results.items()},
        'DDIM50_evaluators': ab['rows'],
        'DDIM50_candidate_overlap': {'both': len(a & b), 'A_only': len(a-b), 'B_only': len(b-a),
                                    'neither': 5000-len(a | b), 'jaccard': len(a & b)/len(a | b) if a | b else None},
        'human_evidence_status': 'pending; candidate overlap is not human accuracy',
    })
    print('DDIM50_5000_AB_COMPARISON_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device-a', default='cuda:0')
    parser.add_argument('--device-b', default='cuda:1')
    parser.add_argument('--evaluate-only', choices=['A', 'B'])
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    devices = [args.device_a, args.device_b]
    if args.evaluate_only:
        evaluate_one(root, out, args.evaluate_only, devices['AB'.index(args.evaluate_only)])
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = preflight(root, out, devices)
        sampling = request['original_sampling']
        jobs = [(name, [sys.executable, '-u', '-m', 'mms_eval', 'sample-afhq500k',
                        '--checkpoint', sampling['weight_path'], '--repository', sampling['repository'],
                        '--out', str(out/'shards'/name), '--count', str(count), '--seed-start', str(start),
                        '--sampler', 'ddim', '--steps', '50', '--batch-size', '4', '--role', sampling['role'],
                        '--device', device]) for (name, start, count), device in zip(SHARDS, devices)]
        run_parallel(jobs, out, 'sampling')
        merge_shards(root, out, request)
        jobs = [(f'evaluation_{name}', [sys.executable, '-u', str(Path(__file__).resolve()),
                                       '--root', str(root), '--out', str(out), '--evaluate-only', name,
                                       '--device-a', devices[0], '--device-b', devices[1]]) for name in ['A', 'B']]
        run_parallel(jobs, out, 'evaluation')
        compare_all(root, out)


if __name__ == '__main__':
    main()
