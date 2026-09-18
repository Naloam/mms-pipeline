"""Frozen-logit temperature, alpha, and reference-size calibration experiments."""
from pathlib import Path

import numpy as np

from .artifacts import load_evaluation
from .detection import binary_metrics
from .pipeline import load_reference
from .semantic import SCORE_NAMES, apply_threshold, posterior_scores, rank_calibrate
from .utils import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def _scores(arrays, temperature):
    result = {}
    for key, value in arrays.items():
        scores = posterior_scores(value, temperature)
        result[key] = {s: scores[s] for s in SCORE_NAMES}
        result[key]['predicted_class'] = scores['class_order'][:, 0]
    return result


def _measure(scores, score, calibration, true_classes, human):
    output = {}
    for name in ('audit_internal', 'audit_external', 'generated'):
        values = scores[name][score]
        flags = apply_threshold(values, calibration)
        output[name+'_n'] = len(flags)
        output[name+'_flag_count'] = int(flags.sum())
        output[name+'_flag_rate'] = float(flags.mean())
        labels = true_classes.get(name, scores[name]['predicted_class'])
        output[name+'_by_class'] = {
            str(c): {'n': int((labels==c).sum()), 'flag_count': int(flags[labels==c].sum()),
                     'flag_rate': float(flags[labels==c].mean())}
            for c in np.unique(labels)}
        if name == 'generated' and human is not None:
            indices, labels = human
            output['human'] = binary_metrics(labels, flags[indices], values[indices])
            output['human']['n_labeled_including_unknown'] = true_classes['human_labeled_n']
            output['human']['n_unknown'] = true_classes['human_labeled_n']-len(indices)
    return output


def _summaries(rows):
    result = []
    for score, n, alpha in sorted({(r['score'], r['n_reference'], r['alpha']) for r in rows}):
        selected = [r for r in rows if (r['score'],r['n_reference'],r['alpha']) == (score,n,alpha)]
        summary = {'score': score, 'n_reference': n, 'alpha': alpha, 'repetitions': len(selected)}
        for part in ('audit_internal','audit_external','generated'):
            values = np.array([r[part+'_flag_rate'] for r in selected])
            summary[part] = {'mean': float(values.mean()), 'std_across_references': float(values.std()),
                             'min': float(values.min()), 'max': float(values.max()),
                             'descriptive_quantiles_025_975': np.quantile(values,[.025,.975]).tolist()}
        result.append(summary)
    return result


def adaptivity_audit(reference, evaluation, output_dir, *, sizes=None,
                     alphas=(.01,.025,.05,.1), temperatures=(.5,1.,2.,4.),
                     repetitions=20, seed=2026091106, final_labels=None):
    ref_path, run, out = Path(reference).resolve(), Path(evaluation).resolve(), Path(output_dir).resolve()
    ref, report = load_reference(ref_path), load_evaluation(run)[0]
    ref_digest = sha256_file(ref_path)
    if report['reference_signature'] != ref['signature']:
        raise ValueError('Sensitivity run must use the same frozen reference and evaluator')
    if ref['config'].get('temperature', 1) != 1:
        raise ValueError('This registered temperature experiment requires a T=1 main reference')
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError('Reference repetitions must be a positive integer')
    if sha256_file(run/'features.npz') != report['features_sha256']:
        raise ValueError('Evaluation feature cache changed')
    with np.load(ref_path.parent/ref['arrays_file'], allow_pickle=False) as data:
        arrays = {k.removesuffix('_logits'): data[k] for k in data.files if k.endswith('_logits')}
    with np.load(run/'features.npz', allow_pickle=False) as data:
        arrays['generated'] = data['semantic_logits']
    real_records = {name: read_jsonl(ref_path.parent/f'{name}.jsonl') for name in
                    ('calibration','calibration_pool','audit_internal','audit_external')}
    true_classes = {name: np.array([r['label'] for r in real_records[name]]) for name in ('audit_internal','audit_external')}
    generated_records = read_jsonl(run/'input_manifest.jsonl')
    if len(arrays['generated']) != len(generated_records):
        raise ValueError('Generated identities and logit counts disagree')
    human = None
    if final_labels:
        from .analysis_reporting import load_final_labels
        labels, _ = load_final_labels(final_labels,records=generated_records)
        ids = {r['image_id'] for r in generated_records}
        if not set(labels) <= ids:
            raise ValueError('Human labels contain IDs outside this evaluation')
        selected = [(i, labels[r['image_id']]) for i,r in enumerate(generated_records)
                    if r['image_id'] in labels and labels[r['image_id']] is not None]
        human = (np.array([i for i,_ in selected], dtype=int), np.array([y for _,y in selected], dtype=int))
        true_classes['human_labeled_n'] = len(labels)
    baseline = _scores(arrays, 1.)
    main_ids = [r['image_id'] for r in real_records['calibration']]
    main_subset = stable_hash(main_ids)
    pool_n = len(arrays['calibration_pool'])
    if sizes is None:
        defaults = (250,500,1000,2000,2500) if ref['config']['domain'].lower().replace('-','') == 'cifar10' else (250,500,1000,1500)
        sizes = (*defaults, pool_n)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError('Reference sizes must be positive integers')
    sizes = sorted(set(n for n in sizes if n <= pool_n))
    if not sizes or not alphas or not temperatures:
        raise ValueError('Sensitivity grids must contain supported values')
    for alpha in alphas:
        rank_calibrate([0.], alpha)
    for temperature in temperatures:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperatures must be finite and positive')

    def row_for(scored, score, calibration, *, experiment, temperature, subset_id, condition, rep=0):
        identity = {'reference_signature': ref['signature'], 'score': score, 'temperature': float(temperature),
                    'subset_id': subset_id, 'condition': condition, 'alpha': calibration['alpha']}
        return {**identity, 'calibration_id': stable_hash(identity), 'experiment': experiment,
                'n_reference': calibration['n'], 'k': calibration['k'], 'threshold': calibration['threshold'],
                'repetition': rep, **_measure(scored, score, calibration, true_classes, human)}

    rows, subsets = [], []
    for n in sizes:
        for rep in range(1 if n == pool_n else repetitions):
            indices = np.arange(pool_n) if n == pool_n else np.random.RandomState(seed+rep).permutation(pool_n)[:n]
            ids = [real_records['calibration_pool'][i]['image_id'] for i in indices]
            subset_id = stable_hash(ids)
            subsets.append({'subset_id': subset_id, 'n': n, 'repetition': rep, 'seed': seed+rep,
                            'pool_indices': indices.tolist(), 'image_ids': ids, 'full_pool': n == pool_n})
            for alpha in alphas:
                for score in SCORE_NAMES:
                    calibration = rank_calibrate(baseline['calibration_pool'][score][indices], alpha)
                    row = row_for(baseline, score, calibration, experiment='reference_size', temperature=1.,
                                  subset_id=subset_id, condition='recalibrated', rep=rep)
                    row['seed'] = seed+rep
                    rows.append(row)
    temperature_rows = []
    for temperature in temperatures:
        scored = baseline if temperature == 1 else _scores(arrays, temperature)
        for score in SCORE_NAMES:
            for condition in ('fixed_T1_threshold','recalibrated'):
                calibration = (ref['calibrations'][score] if condition == 'fixed_T1_threshold' else
                               rank_calibrate(scored['calibration'][score], ref['config'].get('alpha', .05)))
                temperature_rows.append(row_for(scored, score, calibration, experiment='temperature', temperature=temperature,
                                                subset_id=main_subset, condition=condition))
    alpha_rows = []
    for alpha in alphas:
        for score in SCORE_NAMES:
            calibration = rank_calibrate(baseline['calibration'][score], alpha)
            alpha_rows.append(row_for(baseline, score, calibration, experiment='alpha', temperature=1.,
                                      subset_id=main_subset, condition='recalibrated'))
    summaries = _summaries(rows)
    output = {'reference_signature': ref['signature'], 'evaluation_sha256': sha256_file(run/'report.json'),
              'evaluator_id': report.get('evaluator_id', 'A'), 'seed': seed,
              'summaries': [r for r in summaries if r['score'] == 'entropy'], 'score_summaries': summaries,
              'temperatures': list(temperatures), 'alphas': list(alphas), 'sizes': sizes,
              'main_calibration_ids_sha256': main_subset,
              'generated_ids_sha256': stable_hash([r['image_id'] for r in generated_records]),
              'human_labels_sha256': sha256_file(final_labels) if final_labels else None,
              'main_reference_unchanged': sha256_file(ref_path) == ref_digest,
              'interpretation': 'Same real IDs and raw logits for every score and calibration strategy. Full pool is evaluated once. Overlapping subsets provide descriptive sensitivity, not independent trial standard errors. Main reference remains frozen. Human outcomes are pending unless labels are supplied.'}
    write_jsonl(out/'adaptivity_rows.jsonl', rows)
    write_jsonl(out/'reference_subsets.jsonl', subsets)
    write_jsonl(out/'temperature_rows.jsonl', temperature_rows)
    write_jsonl(out/'alpha_rows.jsonl', alpha_rows)
    write_json(out/'adaptivity.json', output)
    return output
