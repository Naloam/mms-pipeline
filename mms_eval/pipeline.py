"""Frozen references and one reusable entry point for images and collections."""
from __future__ import annotations

import html
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .distribution import FidelityExtractor, compute_distribution_metrics
from .evaluator import PredictionSession, describe_checkpoint, predict_paths
from .feature_cache import cached_features
from .images import as_rgb_image, canonicalize_records, collect_images, save_png_atomic, validate_policy
from .semantic import SCORE_NAMES, aggregate_mms, make_score_rows, posterior_scores, rank_calibrate
from .splits import load_split
from .utils import atomic_text, read_json, sha256_file, stable_hash, write_json, write_jsonl

VERSION = "frozen-reference-v1"


def code_signature():
    return stable_hash({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def extraction_code_signature(*names):
    return stable_hash({name: sha256_file(Path(__file__).with_name(name)) for name in names})


def freeze_request(path, request):
    if Path(path).exists() and read_json(path) != request:
        raise ValueError('Output contains a different evaluation/reference request; use a new directory')
    write_json(path, request)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _identity(records):
    return [{k: r[k] for k in ('image_id', 'sha256', 'canonical_sha256') if k in r} for r in records]


def semantic_features(records, checkpoint, cache, device, batch_size, metadata=None, *, chunk_size=256):
    metadata = metadata or describe_checkpoint(checkpoint)
    session = PredictionSession()
    return cached_features(Path(cache)/'semantic', records,
                   {'code': extraction_code_signature('evaluator.py', 'images.py'), 'checkpoint_sha256': metadata['checkpoint_sha256'],
                    'preprocessing': metadata['preprocessing'], 'image_policy': metadata['image_policy'],
                    'software': metadata['software'], 'device': device, 'batch_size': batch_size},
                   lambda rows: predict_paths([r['path'] for r in rows], checkpoint,
                                         device=device, batch_size=batch_size, session=session), chunk_size=chunk_size)


def quality_features(records, extractor, cache, *, chunk_size=256):
    return cached_features(Path(cache)/'inception', records,
                   {'code': extraction_code_signature('distribution.py', 'quality_backends.py'), **extractor.metadata},
                   lambda rows: extractor.extract_paths([r['path'] for r in rows]), chunk_size=chunk_size)


def make_quality_extractor(config, device, batch_size):
    if config.get('feature_backend', 'torch-fidelity') == 'torch-fidelity' and config.get('pr_feature_backend', 'inception') == 'inception':
        return FidelityExtractor(device, batch_size)
    from .quality_backends import ProtocolExtractor
    return ProtocolExtractor(config, device, batch_size)


def reject_seen(records, seen, label):
    ids = {r['image_id'] for r in seen}
    hashes = {r['sha256'] for r in seen}
    if any(r['image_id'] in ids or r['sha256'] in hashes for r in records):
        raise ValueError(f'{label} overlaps evaluator training/validation or another independent split')


def build_reference(split_dir, checkpoint, output_dir, *, config, device='cpu', batch_size=32,
                    cache_dir=None, cache_chunk_size=256):
    """Calibrate only on held-out real images, then independently audit the rule."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve() if cache_dir else out/'cache'
    metadata = describe_checkpoint(checkpoint)
    if config.get('domain_contract_version'):
        from .domains import validate_domain_contract
        validate_domain_contract(config,evaluator_config=metadata['config'])
    policy = validate_policy(config.get('image_policy'))
    if config.get('category_mode') != 'exclusive':
        raise ValueError('MMS needs explicitly validated mutually exclusive categories')
    if metadata['classes'] != config['classes'] or validate_policy(metadata['image_policy']) != policy:
        raise ValueError('Domain classes/image policy differ from the evaluator checkpoint')
    split_meta = read_json(Path(split_dir) / 'split.json')
    if split_meta['classes'] != config['classes'] or split_meta['domain'] != config['domain']:
        raise ValueError('Real split and domain profile disagree')
    contract = {'version': VERSION, 'config': config, 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'split_signature': split_meta['signature'], 'code_sha256': code_signature()}
    signature = stable_hash(contract)
    ref_path = out / 'reference.json'
    if ref_path.exists():
        ref = load_reference(ref_path)
        if ref['signature'] != signature:
            raise ValueError('Reference already frozen with a different contract; use a new output directory')
        return ref
    freeze_request(out/'reference_request.json', {'signature': signature, 'contract': contract})
    parts = {name: collect_images(load_split(split_dir, name)) for name in
             ('calibration', 'calibration_pool', 'audit_internal', 'audit_external', 'quality_reference')}
    if not metadata.get('seen_images'):
        raise ValueError('Evaluator lacks training identities needed for leakage auditing')
    for name in ('calibration_pool', 'audit_internal', 'audit_external'):
        reject_seen(parts[name], metadata['seen_images'], name)
    reject_seen(parts['audit_internal'], parts['calibration_pool'], 'internal audit')
    reject_seen(parts['audit_external'], parts['calibration_pool']+parts['audit_internal'], 'external audit')
    if not {r['sha256'] for r in parts['calibration']} <= {r['sha256'] for r in parts['calibration_pool']}:
        raise ValueError('Main calibration must be a subset of the frozen pool')
    calibration_scores, audits, arrays, cache_meta = {}, {}, {}, {}
    for name in ('calibration', 'calibration_pool', 'audit_internal', 'audit_external'):
        print(f'Extracting semantic reference: {name} ({len(parts[name])} images)', flush=True)
        canonical = canonicalize_records(parts[name], cache/'rgb', policy)
        values, cache_meta[name] = semantic_features(canonical, checkpoint, cache, device, batch_size, metadata,
                                                    chunk_size=cache_chunk_size)
        arrays[name+'_logits'] = values['logits']
        scores = posterior_scores(values['logits'], config.get('temperature', 1))
        if name == 'calibration':
            calibration_scores = {s: rank_calibrate(scores[s], config.get('alpha', .05)) for s in SCORE_NAMES}
        if name.startswith('audit_'):
            audits[name] = aggregate_mms(scores, calibration_scores, config['classes'])
            labels = np.array([r['label'] for r in parts[name]])
            audits[name]['classification_accuracy'] = float((scores['class_order'][:, 0] == labels).mean())
            audits[name]['interpretation'] = 'Real-image flag rate; external split is an empirical transport check'
    print(f"Extracting quality reference ({len(parts['quality_reference'])} images)", flush=True)
    extractor = make_quality_extractor(config.get('metrics', {}), device, batch_size)
    canonical = canonicalize_records(parts['quality_reference'], cache/'rgb', policy)
    quality, quality_meta = quality_features(canonical, extractor, cache, chunk_size=cache_chunk_size)
    arrays['real_features'] = quality['features']
    if 'pr_features' in quality:
        arrays['real_pr_features'] = quality['pr_features']
    save_arrays(out/'reference_arrays.npz', **arrays)
    for name, records in parts.items():
        write_jsonl(out/f'{name}.jsonl', records)
    ref = {**contract, 'signature': signature, 'protocol_id': config['protocol_id'],
           'evaluator_id': config.get('evaluator_id', metadata.get('architecture', 'A')),
           'classes': config['classes'], 'image_policy': policy,
           'evaluator_checkpoint': str(Path(checkpoint).resolve()),
           'evaluator_metadata': {k: v for k,v in metadata.items() if k != 'seen_images'},
           'semantic_inference': cache_meta['calibration']['metadata'].get('inference_contract'),
           'calibrations': calibration_scores, 'audits': audits,
           'inception_metadata': quality_meta['metadata'],
           'arrays_file': 'reference_arrays.npz', 'arrays_sha256': sha256_file(out/'reference_arrays.npz'),
           'manifests_sha256': {n: sha256_file(out/f'{n}.jsonl') for n in parts},
           'counts': {n: len(r) for n,r in parts.items()},
           'semantic_status': config.get('semantic_status', 'human_MM_validity_pending')}
    ref['integrity_sha256'] = stable_hash(ref)
    write_json(ref_path, ref)
    return ref


def load_reference(path):
    path = Path(path).resolve()
    ref = read_json(path)
    if stable_hash({k:v for k,v in ref.items() if k != 'integrity_sha256'}) != ref.get('integrity_sha256'):
        raise ValueError('Frozen reference content was modified')
    if stable_hash({k: ref[k] for k in ('version', 'config', 'checkpoint_sha256', 'split_signature', 'code_sha256')}) != ref['signature']:
        raise ValueError('Reference contract signature mismatch')
    if sha256_file(path.parent/ref['arrays_file']) != ref['arrays_sha256']:
        raise ValueError('Reference arrays were modified')
    for name, digest in ref['manifests_sha256'].items():
        if sha256_file(path.parent/f'{name}.jsonl') != digest:
            raise ValueError(f'Reference manifest changed: {name}')
    return ref


def _matching_extractor(a, b):
    def scientific(value):
        if isinstance(value, dict):
            return {k: scientific(v) for k,v in value.items()
                    if k not in ('device','batch_size','weights_file','n_images','input_size_counts')}
        if isinstance(value, list):
            return [scientific(v) for v in value]
        return value
    keys = ('backend', 'backend_version', 'weights_sha256', 'extractor_source_sha256',
            'feature_layer', 'logit_layer', 'preprocessing', 'torch_version', 'torchvision_version', 'pillow_version', 'components')
    if any(scientific(a.get(k)) != scientific(b.get(k)) for k in keys):
        raise ValueError('Generated and real Inception extraction protocols differ')


def write_report(path, report):
    write_json(path, report)
    dist = report['distribution']
    rows = [('MMS candidate fraction', report['mms']['value'], report['mms']['status'])]
    for name in ('fid', 'kid', 'is', 'precision_recall'):
        item = dist[name]
        value = {k: item[k] for k in ('value','mean','std','precision','recall') if k in item}
        rows.append((name, value, item.get('reason') or item['status']))
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)
    body = f'''<!doctype html><meta charset="utf-8"><title>MMS evaluation</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#17242d}}td,th{{border-bottom:1px solid #ddd;padding:12px;text-align:left}}table{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Image evaluation · {report['n']} images</h1><p>Protocol: {html.escape(report['protocol_id'])}</p>
<p>MMS measures calibrated candidate frequency. Human semantic validation is pending unless separately documented. Sample size alone does not establish validity; interpret results under the registered study protocol.</p>
<table><tr><th>Metric</th><th>Result</th><th>Status / reason</th></tr>{table}</table>
<h2>Full report</h2><pre>{html.escape(Path(path).read_text(encoding='utf-8'))}</pre>'''
    atomic_text(Path(path).with_suffix('.html'), body)


def evaluate(source, reference, output_dir, *, checkpoint=None, device='cpu', batch_size=32,
             cache_dir=None, cache_chunk_size=256):
    """Evaluate one file, folder, image list or CSV/JSONL manifest with a frozen reference."""
    started = time.perf_counter()
    timings = {}
    ref_path, out = Path(reference).resolve(), Path(output_dir).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else out/'cache'
    ref = load_reference(ref_path)
    if ref['code_sha256'] != code_signature():
        raise ValueError('Evaluation code differs from the frozen reference; version and rebuild the reference')
    checkpoint = checkpoint or ref['evaluator_checkpoint']
    metadata = describe_checkpoint(checkpoint)
    if metadata['checkpoint_sha256'] != ref['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the frozen calibration evaluator')
    records = collect_images(source)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        prior = read_json(out/'report.json')
        from .utils import read_jsonl
        if prior['reference_signature'] != ref['signature'] or stable_hash(read_jsonl(out/'input_manifest.jsonl')) != stable_hash(records):
            raise ValueError('Output already contains a different evaluation; use a new directory')
    freeze_request(out/'evaluation_request.json', {'reference_signature': ref['signature'], 'images': _identity(records),
                                                 'records_sha256': stable_hash(records)})
    write_jsonl(out/'input_manifest.jsonl', records)
    canonical = canonicalize_records(records, cache/'rgb', ref['image_policy'])
    timings['input_and_canonicalization'] = time.perf_counter()-started
    stage = time.perf_counter()
    logits, semantic_meta = semantic_features(canonical, checkpoint, cache, device, batch_size, metadata,
                                              chunk_size=cache_chunk_size)
    timings['semantic_features'] = time.perf_counter()-stage
    scores = posterior_scores(logits['logits'], ref['config'].get('temperature', 1))
    mms = aggregate_mms(scores, ref['calibrations'], ref['classes'])
    stage = time.perf_counter()
    extractor = make_quality_extractor(ref['config'].get('metrics', {}), device, batch_size)
    _matching_extractor(ref['inception_metadata'], extractor.metadata)
    generated, quality_meta = quality_features(canonical, extractor, cache, chunk_size=cache_chunk_size)
    timings['quality_features_including_model_loading'] = time.perf_counter()-stage
    stage = time.perf_counter()
    with np.load(ref_path.parent/ref['arrays_file'], allow_pickle=False) as data:
        real = data['real_features']
        real_pr = data['real_pr_features'] if 'real_pr_features' in data.files else None
    distribution = compute_distribution_metrics(real, generated['features'], generated['logits'],
                          real_pr_features=real_pr, generated_pr_features=generated.get('pr_features'),
                          config={**ref['config'].get('metrics', {}), 'feature_metadata': quality_meta['metadata']})
    timings['distribution_statistics'] = time.perf_counter()-stage
    stage = time.perf_counter()
    rows = make_score_rows(records, logits['logits'], ref['calibrations'], ref['classes'],
                           temperature=ref['config'].get('temperature',1), protocol_id=ref['protocol_id'],
                           evaluator_id=ref.get('evaluator_id', 'A'))
    for row, original in zip(rows, records):
        row.update(path=original['path'], sha256=original['sha256'])
    write_jsonl(out/'scores.jsonl', rows)
    write_jsonl(out/'input_manifest.jsonl', records)
    save_arrays(out/'features.npz', semantic_logits=logits['logits'], **generated)
    report = {'version': VERSION, 'protocol_id': ref['protocol_id'], 'n': len(records),
              'evaluator_id': ref.get('evaluator_id', 'A'),
              'reference_signature': ref['signature'], 'reference_sha256': sha256_file(ref_path),
              'split_signature': ref['split_signature'],
              'quality_reference_manifest_sha256': ref['manifests_sha256']['quality_reference'],
              'evaluator_checkpoint_sha256': ref['checkpoint_sha256'], 'code_sha256': code_signature(),
              'image_policy': ref['image_policy'], 'manifest_sha256': sha256_file(out/'input_manifest.jsonl'),
              'scores_sha256': sha256_file(out/'scores.jsonl'), 'features_sha256': sha256_file(out/'features.npz'),
              'mms': mms, 'distribution': distribution, 'real_audits': ref['audits'],
              'duplicate_file_count': len(records)-len({r['sha256'] for r in records}),
              'calibrations': ref['calibrations'], 'semantic_status': ref['semantic_status'],
              'semantic_inference': ref.get('semantic_inference'),
              'reference': str(ref_path), 'semantic_cache_key': semantic_meta['key'],
              'quality_cache_key': quality_meta['key'], 'per_image_scores': 'scores.jsonl', 'features': 'features.npz'}
    report['feature_cache'] = {'semantic': semantic_meta, 'quality': quality_meta}
    timings['score_and_feature_export'] = time.perf_counter()-stage
    timings['total_before_report_write'] = time.perf_counter()-started
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['runtime'] = {'seconds': timings, 'peak_process_rss_mib': peak/(1024**2 if sys.platform == 'darwin' else 1024),
                         'device': device, 'batch_size': batch_size, 'cache_chunk_size': cache_chunk_size}
    write_report(out/'report.json', report)
    return report


def evaluate_image(image, reference, output_dir, *, value_range=None, layout='HWC', **kwargs):
    """Python entry for PIL, uint8 arrays, and tensors with an explicit floating range."""
    if isinstance(image, (str, Path)):
        return evaluate(image, reference, output_dir, **kwargs)
    path = Path(output_dir).resolve()/'input'/'image.png'
    save_png_atomic(as_rgb_image(image, value_range=value_range, layout=layout), path)
    return evaluate(path, reference, output_dir, **kwargs)
