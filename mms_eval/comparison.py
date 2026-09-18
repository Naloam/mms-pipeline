"""Equal-budget batch evaluation and protocol-checked comparison artifacts."""
from pathlib import Path
import re

import numpy as np

from .annotation import _csv
from .artifacts import load_evaluation
from .images import collect_images
from .pipeline import evaluate, freeze_request, load_reference
from .utils import read_json, sha256_file, stable_hash, write_json, write_jsonl


def evaluate_batch(spec, output_dir, *, device='cpu', batch_size=32, cache_dir=None):
    out = Path(output_dir).resolve()
    n, seed = spec.get('common_n'), spec.get('seed')
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or not isinstance(seed, int):
        raise ValueError('A fixed common_n >= 2 and selection seed are required')
    mode=spec.get('mode','semantic')
    if mode not in ('semantic','quality_only'):
        raise ValueError('Batch mode must be semantic or quality_only')
    reference=Path(spec['reference']).resolve() if mode=='semantic' else None
    ref=load_reference(reference) if reference else None
    real=collect_images(spec['real']) if mode=='quality_only' else None
    sources, ids = [], set()
    for source in spec['sources']:
        name = source['source_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in ids:
            raise ValueError('Batch source IDs must be unique safe file names')
        ids.add(name)
        records = collect_images(source['input'])
        if len(records) < n:
            raise ValueError(f'Insufficient images for the fixed budget: {name}')
        # Independent generator seeds are not treated as paired merely because
        # they share an integer. Pair IDs must be supplied by the experiment.
        rng = np.random.RandomState(int(stable_hash({'seed': seed, 'source_id': name})[:8], 16))
        selected = [{**records[i], 'setting_id': records[i].get('setting_id', name),
                     'batch_source_id': name} for i in rng.permutation(len(records))[:n]]
        sources.append({'source_id': name, 'population_n': len(records), 'selected': selected})
    if not sources:
        raise ValueError('Supply at least one generation source')
    request = {'spec': spec, 'reference_signature': ref['signature'] if ref else None, 'real':real, 'sources': sources}
    freeze_request(out/'batch_request.json', request)
    evaluations = []
    for source in sources:
        folder = out/source['source_id']
        write_jsonl(folder/'selected_input.jsonl', source['selected'])
        if mode=='semantic':
            evaluate(folder/'selected_input.jsonl', reference, folder/'evaluation', device=device,
                     batch_size=batch_size, cache_dir=cache_dir or out/'shared_cache')
        else:
            from .quality_only import evaluate_quality
            evaluate_quality(folder/'selected_input.jsonl',real,folder/'evaluation',config=spec.get('config'),
                             device=device,batch_size=batch_size,cache_dir=cache_dir or out/'shared_cache')
        evaluations.append({'name': source['source_id'], 'path': str(folder/'evaluation')})
        write_json(out/'batch_progress.json', {'complete_sources': evaluations, 'total_sources': len(sources)})
    return compare_evaluations(evaluations, out/'comparison')


def _scientific(value):
    if isinstance(value, dict):
        return {k: _scientific(v) for k,v in value.items() if k not in
                ('device','batch_size','weights_file','n_images','input_size_counts')}
    if isinstance(value, list):
        return [_scientific(v) for v in value]
    return value


def _quality_contract(report):
    dist = report['distribution']
    real_hash = report.get('quality_reference_manifest_sha256', report.get('real_manifest_sha256'))
    if real_hash is None:
        ref = load_reference(report['reference'])
        if ref['signature'] != report['reference_signature']:
            raise ValueError('Quality reference signature disagrees with report')
        real_hash = ref['manifests_sha256']['quality_reference']
    return {'image_policy': report['image_policy'], 'real_manifest': real_hash,
            'features': _scientific(dist['metadata']['features']),
            'arithmetic_dtype': dist['metadata']['arithmetic_dtype'],
            'implementation': dist['metadata']['implementation'],
            'rng_seed': dist['metadata']['rng_seed'],
            'kid': {k: dist['kid'].get(k) for k in ('method','subsets','subset_size','seed')},
            'is': {k: dist['is'].get(k) for k in ('method','splits','seed','shuffle')},
            'pr': {k: dist['precision_recall'].get(k) for k in ('method','k','balanced_collections','max_samples_per_collection','selection_seed')}}


def compare_evaluations(evaluations, output_dir, *, mode='models', plots=True):
    """modes: models (same reference/N), evaluators (same image identities/N), scale.

    Different metric implementations or real-reference identities always fail.
    A/B MMS results remain separately calibrated; no pooled candidate proportion.
    """
    if mode not in ('models', 'evaluators', 'scale'):
        raise ValueError('Comparison mode must be models, evaluators, or scale')
    rows, reports, populations = [], [], []
    names = set()
    for item in evaluations:
        path, name = item['path'], item['name']
        if name in names:
            raise ValueError('Comparison names must be unique')
        names.add(name)
        root = Path(path).resolve()
        if root.is_file():
            root = root.parent
        semantic = read_json(root/'report.json').get('kind') != 'quality_only'
        report, _, records, _ = load_evaluation(root, semantic=semantic)
        if reports:
            if _quality_contract(report) != _quality_contract(reports[0]):
                raise ValueError('Metric protocol or real reference differs; do not rank these reports together')
            if mode != 'scale' and report['n'] != reports[0]['n']:
                raise ValueError('Model/evaluator comparisons require the same generated N')
            if mode != 'scale' and any(report['distribution']['precision_recall'].get(k) != reports[0]['distribution']['precision_recall'].get(k)
                                       for k in ('n_real','n_generated')):
                raise ValueError('PR effective sample budgets differ')
            if mode in ('models', 'scale') and report.get('reference_signature') != reports[0].get('reference_signature'):
                raise ValueError('Use the same semantic evaluator and frozen calibration for model comparisons')
        population = {r['image_id']: r['sha256'] for r in records}
        if mode == 'evaluators' and populations and population != populations[0]:
            raise ValueError('Evaluator A/B comparisons require exactly the same image identities/content')
        populations.append(population)
        dist, mms = report['distribution'], report['mms']
        rows.append({'name': name, 'n': report['n'], 'protocol_id': report['protocol_id'],
                     'evaluator_id': report.get('evaluator_id'), 'reference_signature': report.get('reference_signature'),
                     'FID': dist['fid'].get('value'), 'KID': dist['kid'].get('mean'), 'IS': dist['is'].get('mean'),
                     'precision': dist['precision_recall'].get('precision'), 'recall': dist['precision_recall'].get('recall'),
                     'PR_n_real': dist['precision_recall'].get('n_real'), 'PR_n_generated': dist['precision_recall'].get('n_generated'),
                     'MMS_candidate_fraction': mms.get('value'), 'report_sha256': sha256_file(root/'report.json')})
        rows[-1].update({'mean_entropy':mms.get('mean_entropy'),'predicted_class_counts':mms.get('predicted_class_counts'),
                         'runtime':report.get('runtime'), 'generator_settings':item.get('generator_settings'),
                         'human_MM_rate':None,'human_evidence_status':'join_registered_final_labels_separately'})
        reports.append(report)
    if not rows:
        raise ValueError('Supply evaluation reports to compare')
    if mode == 'scale':
        ordered = sorted(populations, key=len)
        if any(any(large.get(k) != v for k,v in small.items()) for small,large in zip(ordered, ordered[1:])):
            raise ValueError('Scale curves require nested samples of the same unchanged images')
    out = Path(output_dir)
    result = {'mode': mode, 'rows': rows, 'quality_contract': _quality_contract(reports[0]),
              'interpretation': 'MMS is a candidate proportion, not verified human MM prevalence. KID/IS subset standard deviations are not confidence intervals. Scale curves are descriptive nested-sample results.'}
    _csv(out/'comparison.csv', list(rows[0]), rows)
    write_json(out/'comparison.json', result)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,3, figsize=(12,6), constrained_layout=True)
        for ax, field in zip(axes.flat, ('FID','KID','IS','precision','recall','MMS_candidate_fraction')):
            values = [r[field] if r[field] is not None else np.nan for r in rows]
            if mode == 'scale':
                order = np.argsort([r['n'] for r in rows])
                ax.plot(np.array([r['n'] for r in rows])[order], np.array(values)[order], 'o-')
                ax.set_xscale('log'); ax.set_xlabel('Generated N')
            else:
                ax.plot(np.arange(len(rows)), values, 'o')
                ax.set_xticks(np.arange(len(rows)), [r['name'] for r in rows], rotation=30, ha='right')
            ax.set_title(field.replace('_',' ')); ax.grid(alpha=.2)
            if field in ('precision','recall','MMS_candidate_fraction'):
                ax.set_ylim(0,1)
        fig.savefig(out/'comparison.png', dpi=180)
        fig.savefig(out/'comparison.pdf')
        plt.close(fig)
    return result
