"""Describe an untouched, single-rater development return without promoting it to a trial."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from mms_eval.artifacts import load_evaluation
from mms_eval.detection import binary_metrics, detection_table
from mms_eval.human_labels import finalize_annotations
from mms_eval.semantic import SCORE_NAMES, apply_threshold, wilson_interval
from mms_eval.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) >= 2 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--evaluation-a', required=True)
    parser.add_argument('--evaluation-b', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026091601)
    args = parser.parse_args()
    pack, labels, out = [Path(p).resolve() for p in (args.pack, args.labels, args.out)]
    if out.exists():
        raise FileExistsError('Preserve the existing analysis; choose a new output directory')
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap repetitions')
    meta = read_json(pack/'pack.json')
    if meta['role'] != 'development':
        raise ValueError('This script is only for exploratory development labels')
    assert sha256_file(pack/'private_key.jsonl') == meta['private_key_sha256']
    keys = read_jsonl(pack/'private_key.jsonl')
    by_blind = {r['blind_id']: r for r in keys}
    assert len(by_blind) == len(keys) == meta['n']
    with labels.open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == len(keys) and {r['blind_id'] for r in raw} == set(by_blind)
    assert len({r['blind_id'] for r in raw}) == len(raw)
    raters = {r['annotator_id'].strip() for r in raw}
    assert len(raters) == 1 and '' not in raters
    for row in raw:
        key = by_blind[row['blind_id']]
        assert row['pack_id'] == meta['pack_id']
        assert row['annotation_guide_version'] == meta['annotation_guide_version']
        assert row['label'] in ('yes', 'no', 'unknown')
        assert row['display_sha256'] == key['display_sha256']
        assert sha256_file(pack/'annotator/images'/f"{row['blind_id']}.png") == key['display_sha256']
        assert float(row['seconds']) >= 0 and np.isfinite(float(row['seconds']))
    out.mkdir(parents=True)
    (out/'raw').mkdir()
    shutil.copy2(labels, out/'raw'/labels.name)
    assert sha256_file(labels) == sha256_file(out/'raw'/labels.name)
    raw_by_image = {by_blind[r['blind_id']]['image_id']: r for r in raw}
    truth = {i: r['label'] for i, r in raw_by_image.items()}
    ordered_ids = [by_blind[r['blind_id']]['image_id'] for r in raw]
    finalize = finalize_annotations(pack, [labels], out/'pending_independent_labels')
    assert finalize['status'] == 'awaiting_adjudication_or_second_rater'
    assert finalize['pending'] == len(keys)
    panels, method_rows, difference_rows, joined_rows, audits, subtype_rows = {}, [], [], [], [], []
    for name, evaluation in [('A', args.evaluation_a), ('B', args.evaluation_b)]:
        report, rows, _, _ = load_evaluation(evaluation)
        scored = {r['image_id']: r for r in rows}
        selected = [scored[i] for i in ordered_ids]
        for row in selected:
            human = raw_by_image[row['image_id']]
            assert row['sha256'] == by_blind[human['blind_id']]['sha256']
            for score in SCORE_NAMES:
                flag = apply_threshold([row[score]], report['calibrations'][score])[0]
                assert bool(flag) == row['flag_'+score]
            joined_rows.append({'evaluator': name, 'blind_id': human['blind_id'],
                'image_id': row['image_id'], 'sha256': row['sha256'],
                'human_label_unmodified': human['label'], 'human_reason': human['reason'],
                'human_other_defects': human['other_defects'], 'human_target_class': human['human_target_class'],
                **{s: row[s] for s in SCORE_NAMES}, **{'flag_'+s: row['flag_'+s] for s in SCORE_NAMES}})
        result = detection_table(selected, truth, group_by='image_id', stratum_by='setting_id',
                                 repetitions=args.bootstrap, seed=args.seed)
        result.update({'role': 'single_rater_development_exploratory', 'evaluator': name,
                       'reference_signature': report['reference_signature'],
                       'report_sha256': sha256_file(Path(evaluation)/'report.json'),
                       'thresholds': report['calibrations'],
                       'unknown_entropy_flags': sum(r['flag_entropy'] for r in selected if truth[r['image_id']]=='unknown'),
                       'all_200_entropy_flags': sum(r['flag_entropy'] for r in selected),
                       'not_confirmatory': True})
        panels[name] = result
        for m in result['methods']:
            line = {'evaluator': name, 'method': m['method'],
                    **{k: m[k] for k in ['n', 'positive', 'negative', 'tp', 'fp', 'fn', 'tn']}}
            for metric in ['tpr', 'fpr', 'precision', 'auroc', 'ap']:
                interval = m[metric+'_ci95'] or [None, None]
                line.update({metric: m[metric], metric+'_ci95_low': interval[0], metric+'_ci95_high': interval[1]})
            method_rows.append(line)
        for d in result['differences']:
            difference_rows.append({'evaluator': name, **{k: v for k, v in d.items() if k != 'ci95'},
                                   'ci95_low': (d['ci95'] or [None,None])[0],
                                   'ci95_high': (d['ci95'] or [None,None])[1]})
        for audit_name, audit in report['real_audits'].items():
            audits.append({'evaluator': name, 'audit': audit_name, 'n': audit['n'],
                           'entropy_flag_rate': audit['value'],
                           'interpretation': 'real reference over-threshold rate, not generated semantic FPR'})
        defects = sorted(set(d for r in raw for d in r['other_defects'].split(';') if d))
        for subgroup in ['all_human_no', 'no_recorded_defects']+defects:
            group = [r for r in selected if truth[r['image_id']]=='no' and
                     (subgroup=='all_human_no' or
                      (subgroup=='no_recorded_defects' and not raw_by_image[r['image_id']]['other_defects']) or
                      subgroup in raw_by_image[r['image_id']]['other_defects'].split(';'))]
            count = sum(r['flag_entropy'] for r in group)
            interval = wilson_interval(count, len(group)) or [None,None]
            subtype_rows.append({'evaluator': name, 'subgroup': subgroup, 'n_human_no': len(group),
                                 'candidate_count': count, 'descriptive_fpr': count/len(group) if group else None,
                                 'ci95_low': interval[0], 'ci95_high': interval[1],
                                 'limitation': 'overlapping observed subgroups; one rater; not controlled causal degradation evidence'})
    # Independent library cross-checks the continuous point estimates.
    from sklearn.metrics import average_precision_score, roc_auc_score
    resolved_ids = [i for i in ordered_ids if truth[i]!='unknown']
    y = np.array([truth[i]=='yes' for i in resolved_ids])
    maps = {e: {r['image_id']: r for r in joined_rows if r['evaluator']==e} for e in ['A','B']}
    for e in ['A','B']:
        for score in SCORE_NAMES:
            values = [maps[e][i][score] for i in resolved_ids]
            m = next(m for m in panels[e]['methods'] if m['method']==score)
            assert abs(m['auroc']-roc_auc_score(y, values)) < 1e-12
            assert abs(m['ap']-average_precision_score(y, values)) < 1e-12
    # A/B differences use the same independently sampled image indices.
    rng = np.random.RandomState(args.seed)
    differences = {k: [] for k in ['tpr','fpr','precision','auroc','ap']}
    data = {e: ([maps[e][i]['flag_entropy'] for i in resolved_ids],
                [maps[e][i]['entropy'] for i in resolved_ids]) for e in ['A','B']}
    for _ in range(args.bootstrap):
        idx = rng.randint(0, len(y), len(y))
        metrics = {e: binary_metrics(y[idx], np.array(data[e][0])[idx], np.array(data[e][1])[idx]) for e in ['A','B']}
        for metric in differences:
            a, b = metrics['A'][metric], metrics['B'][metric]
            differences[metric].append(None if a is None or b is None else b-a)
    ab = []
    for metric, values in differences.items():
        point = {e: next(m for m in panels[e]['methods'] if m['method']=='entropy')[metric] for e in ['A','B']}
        interval = ci(values) or [None,None]
        ab.append({'comparison': 'B_minus_A_entropy', 'metric': metric,
                   'difference': point['B']-point['A'], 'ci95_low': interval[0], 'ci95_high': interval[1],
                   'valid_resamples': sum(v is not None for v in values), 'role': 'exploratory_no_multiple_comparison_adjustment'})
    times = np.array([float(r['seconds']) for r in raw])
    result = {'status': 'completed_exploratory_analysis', 'role': 'single_rater_development',
              'raters': sorted(raters), 'n': len(raw), 'label_counts': dict(Counter(r['label'] for r in raw)),
              'n_missing': 0, 'n_reasons': sum(bool(r['reason']) for r in raw),
              'source_labels_sha256': sha256_file(labels), 'pack_sha256': sha256_file(pack/'pack.json'),
              'all_label_values_preserved': True, 'n_automatic_relabels': 0,
              'bootstrap_repetitions': args.bootstrap, 'seed': args.seed,
              'formal_finalization_status': finalize['status'], 'formal_pending_count': finalize['pending'],
              'sklearn_metric_crosscheck': 'pass', 'panels': panels, 'evaluator_differences': ab,
              'time_seconds': {'sum': float(times.sum()), 'median': float(np.median(times)),
                               'quartiles': np.quantile(times,[.25,.75]).tolist(),
                               'over_300': [{'blind_id': r['blind_id'], 'seconds': float(r['seconds'])} for r in raw if float(r['seconds'])>300],
                               'interpretation': 'foreground dwell proxy includes long idle intervals; not actual labor time or a reliable total budget'},
              'scope_review': {'status': 'user_clarification_requested',
                               'examples': [{'blind_id': r['blind_id'], 'label': r['label'], 'reason': r['reason']} for r in raw if r['blind_id'] in ['0004','0023']],
                               'concern': 'Some positive reasons describe within-wild species distinctions outside the current three-class posterior.'},
              'interpretation': 'Observed agreement with one rater on an existing development sample. Not formal human accuracy, not general model performance, and no tuned thresholds.'}
    write_json(out/'development_analysis.json', result)
    table(out/'methods_development.csv', method_rows)
    table(out/'score_differences_exploratory.csv', difference_rows)
    table(out/'evaluator_differences_exploratory.csv', ab)
    table(out/'observed_negative_subgroups.csv', subtype_rows)
    table(out/'real_reference_audits.csv', audits)
    table(out/'human_reason_inventory.csv', [{k:r[k] for k in ['blind_id','label','human_target_class','other_defects','reason','seconds']} for r in raw])
    write_jsonl(out/'joined_development_scores.jsonl', joined_rows)
    plot(out, panels)
    print(json.dumps({'output': str(out), 'labels': result['label_counts'],
                      'role': result['role'], 'formal_status': finalize['status'],
                      'source_sha256': result['source_labels_sha256']}, ensure_ascii=False))


def plot(out, panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    colors = ['#2e6c7c','#9e642c']
    for ax, metric, title in zip(axes, ['tpr','fpr','precision'],
                                 ['Recall of human yes', 'Flag rate on human no', 'Human yes among flags']):
        values, lower, upper = [], [], []
        for e in ['A','B']:
            m = next(m for m in panels[e]['methods'] if m['method']=='entropy')
            values.append(m[metric]); interval=m[metric+'_ci95']
            lower.append(m[metric]-interval[0]); upper.append(interval[1]-m[metric])
        ax.bar(['A','B'], values, color=colors, width=.55)
        ax.errorbar(['A','B'], values, yerr=[lower,upper], fmt='none', color='#333',capsize=5)
        ax.set_ylim(0,1); ax.set_title(title); ax.set_ylabel('Fraction')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Development only: one rater, 28 yes / 157 no / 15 unknown', fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'development_entropy_comparison.png', dpi=180)
    fig.savefig(out/'development_entropy_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
