"""Seal independent raw labels, record adjudication, and retain unresolved cases."""
from collections import Counter, defaultdict
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from .annotation import _csv
from .detection import mm_label
from .utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def nominal_agreement(by_rater, image_ids):
    raters = sorted(by_rater)
    pooled, disagreement, total, multi = Counter(), 0., 0, 0
    pairs = []
    from sklearn.metrics import cohen_kappa_score
    # None is an explicit unknown judgment. Absence of a dict key is missing.
    category = lambda x: 2 if x is None else int(x)
    for image_id in image_ids:
        values = [category(by_rater[r][image_id]) for r in raters if image_id in by_rater[r]]
        m = len(values)
        if m < 2:
            continue
        counts = Counter(values)
        disagreement += (m*m-sum(n*n for n in counts.values()))/(m-1)
        total += m
        multi += 1
        pooled.update(values)
    observed = disagreement/total if total else None
    expected = (total*total-sum(n*n for n in pooled.values()))/(total*(total-1)) if total > 1 else None
    alpha = 1-observed/expected if expected else None
    for a, b in combinations(raters, 2):
        ids = [i for i in image_ids if i in by_rater[a] and i in by_rater[b]]
        x, y = [category(by_rater[a][i]) for i in ids], [category(by_rater[b][i]) for i in ids]
        kappa = float(cohen_kappa_score(x, y, labels=[0,1,2])) if ids and len(set(x+y)) > 1 else None
        pairs.append({'raters': [a,b], 'n': len(ids), 'agreement': float(np.mean(np.array(x)==y)) if ids else None,
                      'cohen_kappa': kappa if kappa is not None and np.isfinite(kappa) else None})
    return {'raters': raters, 'n_items': len(image_ids), 'items_with_multiple_raters': multi,
            'missing_ratings': len(image_ids)*len(raters)-sum(len(v) for v in by_rater.values()),
            'krippendorff_alpha_nominal': alpha, 'observed_disagreement': observed,
            'expected_disagreement': expected, 'pairwise': pairs,
            'categories': ['no', 'yes', 'unknown'],
            'note': 'Unknown is a nominal category; missing ratings are excluded. Alpha is undefined without expected category variation.'}


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [{k: (v or '').strip() for k, v in r.items()} for r in csv.DictReader(f)]


def finalize_annotations(pack, raw_files, output_dir, *, adjudications=None, minimum_raters=2):
    root, out = Path(pack).resolve(), Path(output_dir).resolve()
    if minimum_raters < 2:
        raise ValueError('At least two independent raters are required; record exceptions explicitly')
    if (out/'annotation_finalization.json').exists():
        raise FileExistsError('Final labels already sealed; use a new output directory for a revision')
    meta = read_json(root/'pack.json')
    if sha256_file(root/'private_key.jsonl') != meta['private_key_sha256']:
        raise ValueError('Annotation private mapping changed')
    keys = read_jsonl(root/'private_key.jsonl')
    ids = [r['blind_id'] for r in keys]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate blind IDs in annotation mapping')
    raw_files = [raw_files] if isinstance(raw_files, (str, Path)) else list(raw_files)
    by_rater, defects, raw_records = {}, defaultdict(set), []
    target_classes = defaultdict(list)
    for path in raw_files:
        for row in _read_csv(path):
            blind = row.get('blind_id') or row.get('display_id')
            rater = row.get('annotator_id') or row.get('rater_id')
            label = row.get('label') or row.get('mm_label')
            if blind not in ids:
                raise ValueError('Raw annotation refers to an unknown blind ID')
            if rater:
                by_rater.setdefault(rater, {})
            if not label:
                continue
            guide=meta.get('annotation_guide_version')
            if guide and row.get('annotation_guide_version')!=guide:
                raise ValueError('Completed annotation guide version must match the frozen pack; missing is not a match')
            if not rater or blind in by_rater[rater]:
                raise ValueError('Raw annotation has a missing rater or duplicate rater/image pair')
            by_rater[rater][blind] = mm_label(label)
            if row.get('other_defects'):
                defects[blind].add(row['other_defects'])
            raw_records.append(row)
            if row.get('human_target_class'):
                target_classes[blind].append(row['human_target_class'])
    if not raw_records:
        raise ValueError('No completed independent labels were supplied')
    decisions = {}
    for row in _read_csv(adjudications) if adjudications else []:
        blind = row.get('blind_id') or row.get('display_id')
        if blind not in ids or blind in decisions or not row.get('decision_method') or not row.get('reason'):
            raise ValueError('Adjudication requires a unique known blind ID, method, and reason')
        label = row.get('mm_label_final') or row.get('label')
        if not label:
            raise ValueError('Adjudication must explicitly record yes, no, or unknown; blank is missing')
        row['_label'] = mm_label(label)
        decisions[blind] = row
    final, needed, exceptions, disagreements = [], [], 0, 0
    for key in keys:
        blind = key['blind_id']
        values = [v[blind] for v in by_rater.values() if blind in v]
        agree = len(values) >= minimum_raters and len(set(values)) == 1
        conflict = len(set(values)) > 1
        disagreements += int(conflict)
        reason, method = '', 'independent_agreement'
        if agree:
            label = values[0]
            if blind in decisions:
                raise ValueError('Adjudication supplied for an already agreed item; preserve independent labels')
        elif blind in decisions:
            decision = decisions[blind]
            if len(values) < minimum_raters:
                if not values or decision['decision_method'] != 'single_rater_exception':
                    raise ValueError('Insufficient independent ratings require an explicitly recorded single_rater_exception')
                exceptions += 1
            label, method, reason = decision['_label'], decision['decision_method'], decision['reason']
        else:
            label, method = None, 'pending_adjudication' if conflict else 'pending_second_rater'
            reason = 'Raw judgments disagree' if conflict else 'Too few independent judgments'
            needed.append({'blind_id': blind, 'mm_label_final': '', 'decision_method': '', 'reason': ''})
        classes = target_classes[blind]
        class_agreement = len(classes) >= minimum_raters and len(set(classes)) == 1
        target_class = classes[0] if class_agreement else 'unknown'
        if not key.get('sha256'):
            raise ValueError('Annotation mapping must bind every image ID to original content SHA-256')
        final.append({'image_id': key['image_id'], 'blind_id': blind, 'sha256':key['sha256'],
                      'human_target_class': target_class,
                      'target_class_status': 'independent_agreement' if class_agreement else 'unresolved_or_uncollected',
                      'mm_label_final': '' if method.startswith('pending_') else 'unknown' if label is None else 'yes' if label else 'no',
                      'decision_method': method, 'reason': reason, 'n_raters': len(values),
                      'raw_disagreement': conflict, 'other_defects': '; '.join(sorted(defects[blind]))})
    write_jsonl(out/'final_labels.jsonl', final)
    _csv(out/'final_labels.csv', list(final[0]), final)
    _csv(out/'adjudication_needed.csv', ['blind_id','mm_label_final','decision_method','reason'], needed)
    result = {'status': 'awaiting_adjudication_or_second_rater' if needed else 'complete_with_recorded_exceptions' if exceptions else 'complete',
              'n': len(final), 'pending': len(needed), 'single_rater_exceptions': exceptions,
              'raw_disagreements': disagreements, 'final_counts': dict(Counter(r['mm_label_final'] for r in final)),
              'agreement': nominal_agreement(by_rater, ids),
              'pack_sha256': sha256_file(root/'pack.json'), 'pack_role': meta.get('role', 'development'),
              'study_registration_sha256': meta.get('registration_sha256'),
              'study_sources': meta.get('sources'), 'annotation_guide_version': meta.get('annotation_guide_version'),
              'raw_files': [{'path': str(Path(p).resolve()), 'sha256': sha256_file(p)} for p in raw_files],
              'adjudications_sha256': sha256_file(adjudications) if adjudications else None,
              'final_labels_sha256': sha256_file(out/'final_labels.csv'),
              'final_labels_jsonl_sha256': sha256_file(out/'final_labels.jsonl'),
              'note': 'Raw labels remain untouched; final labels do not establish semantic validity without the registered sampling and analysis protocol.'}
    write_json(out/'annotation_finalization.json', result)
    return result
