"""Develop and freeze a mean-entropy-matched pair before independent final analysis."""
from itertools import combinations
from pathlib import Path

import numpy as np

from .analysis_reporting import load_final_labels
from .annotation import _csv
from .artifacts import load_evaluation
from .detection import _design, _identities, _interval, bootstrap_indices
from .utils import read_json, sha256_file, stable_hash, write_json


def select_pair(summaries, classes):
    if classes < 2:
        raise ValueError('At least two classes required')
    tolerance = .02*np.log(classes)
    candidates = []
    for a,b in combinations(sorted(summaries,key=lambda r:r['setting_id']),2):
        difference = abs(a['mean_entropy']-b['mean_entropy'])
        candidates.append({'settings': [a['setting_id'],b['setting_id']],
                           'mean_entropy_gap': difference, 'MMS_gap': abs(a['MMS']-b['MMS']),
                           'eligible': bool(difference<=tolerance)})
    eligible = sorted((r for r in candidates if r['eligible']),key=lambda r:(-r['MMS_gap'],r['settings']))
    return {'tolerance': float(tolerance), 'candidates': candidates,
            'selected': eligible[0]['settings'] if eligible else None}


def register_tail_pair(spec, output_dir):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve frozen development selection; use a new directory')
    if spec.get('role') != 'development' or not spec.get('analysis_plan'):
        raise ValueError('Explicit development role and a prospective final analysis plan are required')
    summaries, identities, reference, classes = [], [], None, None
    seen_names, seen_ids, seen_hashes = set(), set(), set()
    for source in spec['sources']:
        name = source['setting_id']
        if not name or name in seen_names:
            raise ValueError('Distinct nonempty setting IDs are required')
        seen_names.add(name)
        report, rows, records, _ = load_evaluation(source['evaluation'])
        if report['n'] != 1000 or any(r.get('cohort_role','').startswith(('diagnostic','final')) for r in records):
            raise ValueError('Use 1,000 natural development images per setting; no final/diagnostic cohorts')
        if reference is not None and report['reference_signature'] != reference:
            raise ValueError('Development settings must share a frozen semantic reference')
        reference, classes = report['reference_signature'], report['mms']['classes']
        ids, hashes = {r['image_id'] for r in records}, {r['sha256'] for r in records}
        if seen_ids & ids or seen_hashes & hashes or len(hashes)!=len(records):
            raise ValueError('Development settings require independent original images')
        seen_ids |= ids; seen_hashes |= hashes
        summaries.append({'setting_id': name, 'mean_entropy': float(np.mean([r['entropy'] for r in rows])),
                          'MMS': float(np.mean([r['flag_entropy'] for r in rows])),
                          'n': len(rows), 'scores_sha256': report['scores_sha256']})
        identities.extend({'image_id': r['image_id'], 'sha256': r['sha256']} for r in records)
    if len(summaries)<2:
        raise ValueError('At least two development settings are required')
    selection = select_pair(summaries,len(classes))
    result = {'role': 'development', 'reference_signature': reference, 'classes': classes,
              'summaries': summaries, **selection, 'development_identities': identities,
              'analysis_plan': spec['analysis_plan'],
              'status': 'frozen_pair' if selection['selected'] else 'no_eligible_development_pair',
              'interpretation': 'Working similarity tolerance, not equivalence or matched quality. Selection uses development scores only; final human quotas must be registered separately.'}
    result['integrity_sha256'] = stable_hash(result)
    write_json(out/'tail_registration.json', result)
    return result


def analyze_tail_pair(registration, spec, output_dir, *, final_labels=None, repetitions=2000, seed=2026091208):
    meta = read_json(registration)
    if meta['integrity_sha256'] != stable_hash({k:v for k,v in meta.items() if k!='integrity_sha256'}):
        raise ValueError('Tail-pair registration changed')
    if meta['selected'] is None:
        raise ValueError('No eligible development pair was frozen; do not select using final data')
    if repetitions < 2:
        raise ValueError('At least two bootstrap repetitions required')
    sources = {r['setting_id']:r for r in spec['sources']}
    if len(sources)!=len(spec['sources']) or set(sources)!=set(meta['selected']):
        raise ValueError('Final comparison must use exactly the development-selected pair')
    labels, label_meta = load_final_labels(final_labels) if final_labels else ({}, {})
    if final_labels and (label_meta.get('pack_role')!='final_random' or label_meta.get('status') not in ('complete','complete_with_recorded_exceptions') or not label_meta.get('study_registration_sha256')):
        raise ValueError('Final human evidence requires complete labels from a registered final_random study')
    seen_ids = {r['image_id'] for r in meta['development_identities']}
    seen_hashes = {r['sha256'] for r in meta['development_identities']}
    summaries, all_rows, arrays = [], [], []
    for name in meta['selected']:
        report, scores, records, _ = load_evaluation(sources[name]['evaluation'])
        if summaries and report['n']!=summaries[0]['n']:
            raise ValueError('Final tail model comparisons require equal generated N')
        if final_labels: load_final_labels(final_labels,records=records)
        if report['reference_signature'] != meta['reference_signature']:
            raise ValueError('Final semantic reference differs from the frozen development protocol')
        if any(r.get('cohort_role','').startswith('diagnostic') for r in records):
            raise ValueError('Final tail analysis requires natural generated cohorts')
        ids, hashes = {r['image_id'] for r in records}, {r['sha256'] for r in records}
        if seen_ids & ids or seen_hashes & hashes or len(hashes)!=len(records):
            raise ValueError('Final images must be independent of all development and other final sources')
        seen_ids |= ids; seen_hashes |= hashes
        rows = [{**r, **s, 'setting_id': name} for r,s in zip(records,scores)]
        values = np.array([[r['entropy'],r['flag_entropy']] for r in rows],dtype=float)
        human = [labels[r['image_id']] for r in rows if r['image_id'] in labels]
        known = [v for v in human if v is not None]
        summaries.append({'setting_id': name, 'n':len(rows), 'mean_entropy':float(values[:,0].mean()),
                          'MMS':float(values[:,1].mean()), 'human_labeled_n':len(human),
                          'human_unknown_n':len(human)-len(known),
                          'human_MM_rate_resolved': float(np.mean(known)) if known else None,
                          'human_MM_rate_lower': sum(v==1 for v in human)/len(human) if human else None,
                          'human_MM_rate_upper': sum(v!=0 for v in human)/len(human) if human else None,
                          'scores_sha256':report['scores_sha256']})
        all_rows.extend(rows); arrays.append(values)
    # One common group/stratum draw pairs all statistics and respects declared parents.
    grouped_rows=[{**r,'bootstrap_stratum':r['setting_id']+(':labeled' if r['image_id'] in labels else ':unlabeled')}
                  for r in all_rows]
    groups,strata = _identities(grouped_rows,'auto','bootstrap_stratum')
    design = _design(groups,strata)
    values = np.concatenate(arrays)
    names = np.array([r['setting_id'] for r in all_rows])
    human = np.array([np.nan if labels.get(r['image_id']) is None else labels[r['image_id']] for r in all_rows])
    boot = {'mean_entropy':[], 'MMS':[], 'human_MM_rate_resolved':[]}
    for draw in bootstrap_indices(groups,strata,repetitions=repetitions,seed=seed):
        a,b = [draw[names[draw]==name] for name in meta['selected']]
        for j,key in enumerate(('mean_entropy','MMS')):
            boot[key].append(float(values[a,j].mean()-values[b,j].mean()))
        ha,hb = human[a],human[b]
        ha,hb = ha[np.isfinite(ha)],hb[np.isfinite(hb)]
        boot['human_MM_rate_resolved'].append(float(ha.mean()-hb.mean()) if len(ha) and len(hb) else None)
    differences = []
    for key,draws in boot.items():
        a,b = [r[key] for r in summaries]
        differences.append({'measure':key, 'first_minus_second': a-b if a is not None and b is not None else None,
                            **_interval(draws,repetitions,design['resampleable_groups']>=2,.8)})
    out = Path(output_dir)
    result = {'registration_sha256':sha256_file(registration),'selected':meta['selected'],'summaries':summaries,
              'differences':differences, 'tolerance':meta['tolerance'],
              'final_mean_match':bool(abs(summaries[0]['mean_entropy']-summaries[1]['mean_entropy'])<=meta['tolerance']),
              'label_provenance':label_meta,'bootstrap_design':design,'seed':seed,'repetitions':repetitions,
              'semantic_evidence_status':'available_for_interpretation' if all(r['human_labeled_n'] for r in summaries) else 'awaiting_registered_final_human_labels',
              'interpretation':'Report this same pair even if final matching fails. Agreement of directions alone does not establish added semantic information; inspect mean entropy, intervals, unknown sensitivity, human evidence, and parallel quality metrics.'}
    write_json(out/'tail_analysis.json',result)
    _csv(out/'tail_summaries.csv',list(summaries[0]),summaries)
    _csv(out/'tail_differences.csv',list(differences[0]),differences)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    bins = np.linspace(0,np.log(len(meta['classes'])),41)
    for name,v in zip(meta['selected'],arrays):
        entropy = np.sort(v[:,0])
        axes[0].step(entropy,np.arange(1,len(v)+1)/len(v),where='post',label=name)
        axes[1].hist(entropy,bins=bins,density=True,alpha=.4,label=name)
    axes[0].set(xlabel='Entropy (nats)',ylabel='Empirical CDF')
    axes[1].set(xlabel='Entropy (nats)',ylabel='Density')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'entropy_distributions.png',dpi=180); fig.savefig(out/'entropy_distributions.pdf'); plt.close(fig)
    return result
