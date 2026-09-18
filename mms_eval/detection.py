"""Human detection statistics, with paired resampling at the independent source.

Unknown judgments remain visible. Two extreme unknown assignments are
sensitivity scenarios, not mathematical bounds on every detection metric.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .semantic import SCORE_NAMES, wilson_interval

METRICS = ('auroc', 'ap', 'tpr', 'fpr', 'precision')


def binary_metrics(labels, predictions, scores=None):
    y, p = np.asarray(labels), np.asarray(predictions)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('Labels and predictions must be aligned binary vectors')
    y, p = y.astype(bool), p.astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    result = {'n': len(y), 'positive': tp+fn, 'negative': fp+tn,
              'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'auroc': None, 'ap': None}
    for name, num, den in [('tpr', tp, tp+fn), ('fpr', fp, fp+tn), ('precision', tp, tp+fp)]:
        result[name] = num/den if den else None
        result[name+'_ci95_wilson'] = wilson_interval(num, den)
    if scores is not None:
        score = np.asarray(scores, dtype=float)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError('Continuous scores must be aligned and finite')
        if y.any() and (~y).any():
            # Group ties at a shared threshold. AP is the step-weighted
            # precision, and AUROC is trapezoidal ROC area including (0,0).
            order = np.argsort(-score, kind='stable')
            ranked, truth = score[order], y[order]
            ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked)-1]
            tps = np.cumsum(truth)[ends]
            fps = ends+1-tps
            tpr, fpr = np.r_[0, tps/y.sum()], np.r_[0, fps/(~y).sum()]
            result['auroc'] = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
            result['ap'] = float(np.sum(np.diff(np.r_[0, tps])/y.sum() * tps/(ends+1)))
    return result


def bootstrap_indices(groups, strata=None, *, repetitions=2000, seed=2026091108):
    """Resample whole groups, conditional on their source/count profile.

    Equal profiles are resampled together, preserving exact per-source image
    quotas even when a parent/seed spans sources. Singleton profiles cannot
    contribute uncertainty and are reported by the analysis caller.
    """
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError('Bootstrap repetitions must be a positive integer')
    groups = list(groups)
    strata = ['all']*len(groups) if strata is None else list(strata)
    if not groups or len(groups) != len(strata):
        raise ValueError('Bootstrap requires nonempty aligned group and source identities')
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[str(group)].append(i)
    profiles = defaultdict(list)
    for indices in members.values():
        profile = tuple(sorted(Counter(str(strata[i]) for i in indices).items()))
        profiles[profile].append(np.asarray(indices, dtype=int))
    rng = np.random.RandomState(seed)
    for _ in range(repetitions):
        yield np.concatenate([items[i] for items in profiles.values()
                              for i in rng.randint(0, len(items), size=len(items))])


def mm_label(value):
    """Normalize the manual's ternary outcome while retaining defects elsewhere."""
    if value in ('yes', '是', 'mixed_semantics', 1, True):
        return 1
    if value in ('no', '否', 'clear_single', 'clear_nonmixed', 0, False):
        return 0
    if value in ('unknown', '无法判断', 'uncertain', 'unrecognizable', None):
        return None
    raise ValueError(f'Unknown human MM label: {value!r}')


def _identities(rows, group_by, stratum_by):
    from .utils import stable_hash
    groups, strata = [], []
    for row in rows:
        field = group_by
        if field == 'auto':
            field = next((name for name in ('parent_id', 'pair_id', 'seed') if row.get(name) is not None), 'image_id')
        if field not in row or row[field] is None:
            raise ValueError(f'Missing independent-group field {field}')
        # Identical integer seeds from different generators are not shared
        # latent draws. Only explicit pair/parent identities cross sources.
        namespace=row.get('setting_id',row.get('source_id','unspecified')) if field=='seed' else None
        groups.append(stable_hash([field, namespace, row[field]]))
        strata.append(str(row.get(stratum_by, 'all')) if stratum_by else 'all')
    return groups, strata


def _design(groups, strata):
    members = defaultdict(list)
    for group, stratum in zip(groups, strata):
        members[group].append(stratum)
    profiles = Counter(tuple(sorted(Counter(items).items())) for items in members.values())
    return {'group_count': len(members), 'profile_count': len(profiles),
            'resampleable_groups': sum(n for n in profiles.values() if n > 1),
            'singleton_profile_groups': sum(n for n in profiles.values() if n == 1),
            'clustered': len(members) < len(groups)}


def _interval(values, requested, can_estimate, min_valid_fraction):
    finite = np.asarray([v for v in values if v is not None], dtype=float)
    valid = len(finite)
    estimate = can_estimate and valid >= 2 and valid >= min_valid_fraction*requested
    return {'ci95': np.quantile(finite, [.025, .975]).tolist() if estimate else None,
            'valid_resamples': valid, 'invalid_resamples': requested-valid,
            'interval_status': 'ok' if estimate else 'insufficient_valid_resamples_or_independent_groups'}


def detection_table(rows, labels, *, group_by='auto', stratum_by='setting_id',
                    repetitions=2000, seed=2026091108, min_valid_fraction=.8):
    """Compare all frozen scores and original MMR rules on the same labeled IDs.

    AP/AUROC use continuous scores only. Original binary MMR rules have no
    fabricated ranking metrics. All method differences use identical draws.
    """
    rows = list(rows)
    ids = [r['image_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(labels) != set(ids):
        raise ValueError('Detection rows and final labels must have identical unique image IDs')
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError('Bootstrap repetitions must be a positive integer')
    if not 0 < min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction must be in (0,1]')
    truth = [mm_label(labels[i]) for i in ids]
    resolved = np.array([y is not None for y in truth])
    y = np.array([v for v in truth if v is not None], dtype=int)
    grouped_rows = [row for row, keep in zip(rows, resolved) if keep]
    groups, strata = _identities(grouped_rows, group_by, stratum_by)
    design = _design(groups, strata)
    scores = {}
    for name in SCORE_NAMES:
        scores[name] = np.asarray([r[name] for r in rows], dtype=float)
        if not np.isfinite(scores[name]).all():
            raise ValueError('Detection scores must be finite')
        if any(not isinstance(r['flag_'+name], (bool, np.bool_)) for r in rows):
            raise ValueError('Detection flags must be booleans from frozen calibration')
    methods = [(name, np.array([r['flag_'+name] for r in rows]), scores[name]) for name in SCORE_NAMES]
    methods += [(f'mmr_original_p2_ge_{t:g}', scores['second_probability'] >= t, None)
                for t in (.1, .11, .13, .16, .2)]
    point = {name: binary_metrics(y, flags[resolved], None if score is None else score[resolved])
             for name, flags, score in methods}
    boot = {name: {metric: [] for metric in METRICS} for name, *_ in methods}
    differences = {(name, metric): [] for name in SCORE_NAMES[1:] for metric in METRICS}
    if len(y):
        for indices in bootstrap_indices(groups, strata, repetitions=repetitions, seed=seed):
            sampled = {}
            for name, flags, score in methods:
                metrics = binary_metrics(y[indices], flags[resolved][indices], None if score is None else score[resolved][indices])
                sampled[name] = metrics
                for metric in METRICS:
                    boot[name][metric].append(metrics[metric])
            for (name, metric), values in differences.items():
                a, b = sampled['entropy'][metric], sampled[name][metric]
                values.append(None if a is None or b is None else a-b)
    method_rows = []
    can_estimate = design['resampleable_groups'] >= 2
    for name, flags, score in methods:
        row = {'method': name, **point[name]}
        row['uncertainty'] = {metric: _interval(boot[name][metric], repetitions, can_estimate, min_valid_fraction)
                              for metric in METRICS}
        for metric in METRICS:
            row[metric+'_ci95'] = (row[metric+'_ci95_wilson'] if metric in ('tpr', 'fpr', 'precision') and not design['clustered']
                                  else row['uncertainty'][metric]['ci95'])
        row['unknown_sensitivity'] = {}
        for name_unknown, replacement in [('unknown_as_negative', 0), ('unknown_as_positive', 1)]:
            scenario = np.array([replacement if v is None else v for v in truth])
            row['unknown_sensitivity'][name_unknown] = binary_metrics(scenario, flags, score)
        method_rows.append(row)
    diff_rows = []
    for (name, metric), values in differences.items():
        a, b = point['entropy'][metric], point[name][metric]
        diff_rows.append({'method_a': 'entropy', 'method_b': name, 'metric': metric,
                          'difference': None if a is None or b is None else a-b,
                          **_interval(values, repetitions, can_estimate, min_valid_fraction)})
    positive, unknown, n = int(y.sum()), int((~resolved).sum()), len(rows)
    return {'n_sampled': n, 'n_resolved': len(y), 'n_positive': positive,
            'n_negative': len(y)-positive, 'n_unknown': unknown, 'coverage': len(y)/n,
            'prevalence_identification_interval': [positive/n, (positive+unknown)/n],
            'resolved_prevalence': positive/len(y) if len(y) else None,
            'resolved_prevalence_ci95_wilson': wilson_interval(positive, len(y)),
            'methods': method_rows, 'differences': diff_rows,
            'bootstrap': {**design, 'requested': repetitions, 'seed': seed, 'group_by': group_by,
                          'stratum_by': stratum_by, 'minimum_valid_fraction': min_valid_fraction,
                          'method': 'same-index paired percentile bootstrap of whole groups conditional on source-count profiles'},
            'interpretation': 'Detection metrics condition on resolvable human judgments; report coverage. Unknown assignments are sensitivity scenarios, not strict metric bounds. Wilson intervals assume independent images; clustered inference uses group bootstrap. No model-training uncertainty is estimated.'}
