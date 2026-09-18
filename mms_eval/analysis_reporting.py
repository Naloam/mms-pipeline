"""Join sealed human labels to frozen scores and export manuscript table inputs."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .annotation import _csv
from .artifacts import load_evaluation
from .detection import METRICS, detection_table, mm_label
from .semantic import SCORE_NAMES, apply_threshold, posterior_scores
from .utils import read_json, read_jsonl, sha256_file, write_json


def load_final_labels(path, *, records=None):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = read_jsonl(path)
    else:
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    metadata_path = path.parent/'annotation_finalization.json'
    metadata = read_json(metadata_path) if metadata_path.exists() else {'status': 'external_final_labels', 'pack_role': 'unspecified'}
    if metadata.get('status') not in ('complete','complete_with_recorded_exceptions','external_final_labels'):
        raise ValueError('Scientific analysis requires completed finalization; pending ratings/adjudication are not semantic unknowns')
    record_map={r['image_id']:r for r in records} if records is not None else None
    labels = {}
    for row in rows:
        image_id = row.get('image_id')
        label = row.get('mm_label_final')
        if not image_id or image_id in labels or label is None or label == '':
            raise ValueError('Final labels require unique image IDs and explicit yes/no/unknown judgments')
        labels[image_id] = mm_label(label)
        if record_map is not None and image_id in record_map:
            if not row.get('sha256') or row['sha256']!=record_map[image_id].get('sha256'):
                raise ValueError('Human label content SHA-256 does not match the evaluated image; IDs alone are insufficient')
    if not labels:
        raise ValueError('Final label file is empty')
    field = 'final_labels_jsonl_sha256' if path.suffix == '.jsonl' else 'final_labels_sha256'
    if metadata_path.exists() and sha256_file(path) != metadata.get(field):
        raise ValueError('Sealed final labels changed; record a new revision before analysis')
    return labels, metadata


def _real_audits(report):
    result = defaultdict(dict)
    ref_path = Path(report.get('reference', ''))
    if ref_path.is_file():
        from .pipeline import load_reference
        ref = load_reference(ref_path)
        if ref['signature'] != report['reference_signature']:
            raise ValueError('Real audit reference and generated evaluation disagree')
        with np.load(ref_path.parent/ref['arrays_file'], allow_pickle=False) as data:
            for part in ('audit_internal', 'audit_external'):
                scores = posterior_scores(data[part+'_logits'], ref['config'].get('temperature', 1))
                flags = {name: apply_threshold(scores[name], ref['calibrations'][name]) for name in SCORE_NAMES}
                flags.update({f'mmr_original_p2_ge_{t:g}': scores['second_probability'] >= t for t in (.1,.11,.13,.16,.2)})
                for method, values in flags.items():
                    result[method].update({part+'_n': len(values), part+'_count': int(values.sum()), part+'_rate': float(values.mean())})
    else:
        # Downloaded reports can retain their verified aggregate audits while
        # the original reference path is on another host. No missing MMR audit
        # is inferred from the entropy audit.
        for part, audit in report.get('real_audits', {}).items():
            for method, values in audit.get('baselines', {}).items():
                result[method].update({part+'_n': values['n'], part+'_count': values['candidate_count'],
                                       part+'_rate': values['candidate_fraction']})
    return result


def analyze_detection(evaluations, final_labels, output_dir, *, repetitions=2000, seed=2026091108,
                      group_by='auto', stratum_by='setting_id'):
    """Keep domain/evaluator/source panels separate; never pool design quotas."""
    evaluations = [evaluations] if isinstance(evaluations, (str, Path)) else list(evaluations)
    labels, label_meta = load_final_labels(final_labels)
    used_ids, panels, table, differences, scenarios = set(), [], [], [], []
    for evaluation in evaluations:
        report, rows, records, _ = load_evaluation(evaluation)
        load_final_labels(final_labels,records=records)
        original = {r['image_id']: r for r in records}
        sources = defaultdict(list)
        for row in rows:
            if row['image_id'] in labels:
                joined = {**original[row['image_id']], **row}
                sources[str(joined.get('setting_id', Path(evaluation).parent.name if Path(evaluation).is_file() else Path(evaluation).name))].append(joined)
        audits = _real_audits(report)
        for source, selected in sources.items():
            chosen = {row['image_id']: labels[row['image_id']] for row in selected}
            used_ids.update(chosen)
            result = detection_table(selected, chosen, group_by=group_by, stratum_by=stratum_by,
                                     repetitions=repetitions, seed=seed)
            identity = {'protocol_id': report['protocol_id'], 'evaluator_id': report.get('evaluator_id', selected[0].get('evaluator_id', 'A')),
                        'source': source, 'reference_signature': report['reference_signature']}
            panels.append({**identity, 'evaluation': str(Path(evaluation).resolve()), **result})
            for method in result['methods']:
                row = {**identity, 'method': method['method'], 'n_sampled': result['n_sampled'],
                       'n_unknown': result['n_unknown'], 'coverage': result['coverage'],
                       'prevalence_lower': result['prevalence_identification_interval'][0],
                       'prevalence_upper': result['prevalence_identification_interval'][1],
                       **{k: method[k] for k in ('n','positive','negative','tp','fp','fn','tn',*METRICS)}}
                for metric in METRICS:
                    interval = method[metric+'_ci95']
                    row[metric+'_ci95_low'], row[metric+'_ci95_high'] = interval or [None, None]
                    row[metric+'_invalid_resamples'] = method['uncertainty'][metric]['invalid_resamples']
                for part in ('audit_internal', 'audit_external'):
                    for statistic in ('n','count','rate'):
                        key = part+'_'+statistic
                        row[key] = audits.get(method['method'], {}).get(key)
                table.append(row)
                for scenario, statistics in method['unknown_sensitivity'].items():
                    scenarios.append({**identity, 'method': method['method'], 'scenario': scenario,
                                      **{k: statistics[k] for k in ('n','positive','negative',*METRICS)}})
            for difference in result['differences']:
                interval = difference['ci95']
                differences.append({**identity, **{k: v for k, v in difference.items() if k != 'ci95'},
                                    'ci95_low': interval[0] if interval else None,
                                    'ci95_high': interval[1] if interval else None})
    if set(labels) != used_ids:
        raise ValueError(f'{len(set(labels)-used_ids)} final-label IDs have no matching supplied evaluation')
    role = label_meta.get('pack_role', 'unspecified')
    output = {'panels': panels, 'labels_sha256': sha256_file(final_labels), 'label_provenance': label_meta,
              'analysis_role': role, 'status': 'descriptive_detection_analysis',
              'note': 'Sources and backbones remain separate. Diagnostic samples do not estimate natural prevalence. Final confirmatory claims require the registered random-sampling protocol and sealed adjudication.'}
    out = Path(output_dir)
    stem = 'diagnostic' if role == 'diagnostic' else 'main'
    _csv(out/f'table_detection_{stem}.csv', list(table[0]), table)
    _csv(out/'table_detection_differences.csv', list(differences[0]), differences)
    _csv(out/'table_unknown_sensitivity.csv', list(scenarios[0]), scenarios)
    write_json(out/'detection.json', output)
    return output
