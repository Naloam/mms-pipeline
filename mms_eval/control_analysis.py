"""Separate detector response to edits from human-confirmed semantic changes."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis_reporting import load_final_labels
from .annotation import _csv
from .artifacts import load_evaluation
from .semantic import SCORE_NAMES, wilson_interval
from .utils import sha256_file, write_json


def analyze_degradation(evaluation, final_labels, output_dir, *, repetitions=2000, seed=2026091205):
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError('At least two paired bootstrap repetitions are required')
    report, scores, records, _ = load_evaluation(evaluation)
    labels, meta = load_final_labels(final_labels,records=records)
    if meta.get('status') not in ('complete','complete_with_recorded_exceptions'):
        raise ValueError('Degradation validity requires completed independent reannotation and adjudication')
    by_parent = defaultdict(dict)
    for record, score in zip(records, scores):
        if record.get('cohort_role') != 'diagnostic_degradation' or record['image_id'] not in labels:
            raise ValueError('Supply the complete degradation cohort and its independently finalized labels')
        condition = (record['control_family'], record['severity'])
        parent = record['parent_id']
        if condition in by_parent[parent]:
            raise ValueError('Duplicate parent/condition in degradation evaluation')
        by_parent[parent][condition] = {**record, **score, 'human': labels[record['image_id']]}
    parents = sorted(by_parent)
    conditions = set(by_parent[parents[0]])
    if ('original',0) not in conditions or any(set(v) != conditions for v in by_parent.values()):
        raise ValueError('Each parent must have the same conditions, including its original')
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(parents), (repetitions, len(parents)))
    rows, transitions = [], []
    for condition in sorted(conditions):
        current = [by_parent[p][condition] for p in parents]
        original = [by_parent[p][('original',0)] for p in parents]
        for before in (0,1,None):
            for after in (0,1,None):
                transitions.append({'family': condition[0], 'severity': condition[1],
                                    'original_human': 'unknown' if before is None else 'yes' if before else 'no',
                                    'edited_human': 'unknown' if after is None else 'yes' if after else 'no',
                                    'n': sum(a['human']==before and b['human']==after for a,b in zip(original,current))})
        negatives = np.array([r['human']==0 for r in current])
        for method in SCORE_NAMES:
            values = np.array([r[method] for r in current])
            delta = values-np.array([r[method] for r in original])
            flags = np.array([r['flag_'+method] for r in current], dtype=float)
            delta_flags = flags-np.array([r['flag_'+method] for r in original], dtype=float)
            n_negative, false_positive = int(negatives.sum()), int(flags[negatives].sum())
            score_ci = np.quantile(delta[draws].mean(1), [.025,.975]) if len(parents)>1 else [None,None]
            flag_ci = np.quantile(delta_flags[draws].mean(1), [.025,.975]) if len(parents)>1 else [None,None]
            rows.append({'family': condition[0], 'severity': condition[1], 'method': method,
                         'parents_n': len(parents), 'candidate_rate': float(flags.mean()),
                         'confirmed_non_MM_n': n_negative, 'unknown_n': sum(r['human'] is None for r in current),
                         'confirmed_MM_n': sum(r['human']==1 for r in current),
                         'FPR_on_reconfirmed_non_MM': false_positive/n_negative if n_negative else None,
                         'FPR_wilson_ci95': wilson_interval(false_positive,n_negative),
                         'mean_paired_score_change': float(delta.mean()), 'score_change_ci95_low': score_ci[0],
                         'score_change_ci95_high': score_ci[1], 'mean_paired_flag_change': float(delta_flags.mean()),
                         'flag_change_ci95_low': flag_ci[0], 'flag_change_ci95_high': flag_ci[1]})
    out = Path(output_dir)
    _csv(out/'degradation_effects.csv', list(rows[0]), rows)
    _csv(out/'human_transitions.csv', list(transitions[0]), transitions)
    result = {'effects': rows, 'human_transitions': transitions, 'labels_sha256': sha256_file(final_labels),
              'label_provenance': meta, 'scores_sha256': report['scores_sha256'], 'seed': seed,
              'bootstrap_repetitions': repetitions, 'bootstrap_unit': 'whole independent parent with all conditions paired',
              'interpretation': 'Candidate-rate shifts are detector responses. FPR uses only independently reconfirmed non-MM outputs. Report human transitions and unknown counts; edited sources are not assumed negative.'}
    write_json(out/'degradation_analysis.json', result)
    _plot_degradation(rows,out)
    return result


def _plot_degradation(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=sorted({r['family'] for r in rows if r['family']!='original'})
    if not families: return
    fig,axes=plt.subplots(2,len(families),squeeze=False,figsize=(4*len(families),7),constrained_layout=True)
    for column,family in enumerate(families):
        for method in SCORE_NAMES:
            selected=sorted((r for r in rows if r['family']==family and r['method']==method),key=lambda r:r['severity'])
            for row,field in enumerate(('candidate_rate','FPR_on_reconfirmed_non_MM')):
                values=[r[field] if r[field] is not None else np.nan for r in selected]
                axes[row,column].plot([r['severity'] for r in selected],values,'o-',label=method)
                axes[row,column].set(xlabel=family+' parameter',ylabel=field,ylim=(0,1))
                axes[row,column].grid(alpha=.2)
        axes[0,column].legend(fontsize=6)
    fig.savefig(out/'degradation_response.png',dpi=180);fig.savefig(out/'degradation_response.pdf');plt.close(fig)


def summarize_controls(evaluations, output_dir):
    """Export composition/repetition response without treating copies as independent."""
    rows=[]
    reference=None
    for path in evaluations:
        report,scores,records,_=load_evaluation(path)
        if reference is not None and report['reference_signature']!=reference:
            raise ValueError('Control response summaries must share one frozen semantic reference')
        reference=report['reference_signature']
        if any(not r.get('cohort_role','').startswith('diagnostic') for r in records):
            raise ValueError('Supply explicitly registered diagnostic control cohorts')
        groups=defaultdict(list)
        for record,score in zip(records,scores):
            row={**record,**score}
            groups[('all','all')].append(row)
            if record.get('human_target_class'):
                groups[('human_target_class',record['human_target_class'])].append(row)
            if record.get('control_family'):
                groups[('control_family',str(record['control_family']))].append(row)
        for (stratum,value),selected in groups.items():
            unique=len({r.get('parent_id',r['image_id']) for r in selected})
            for method in SCORE_NAMES:
                flags=[r['flag_'+method] for r in selected]
                rows.append({'evaluation':str(Path(path).resolve()),'stratum':stratum,'value':value,
                             'method':method,'n':len(selected),'unique_parent_n':unique,
                             'candidate_count':sum(flags),'candidate_rate':float(np.mean(flags)),
                             'mean_score':float(np.mean([r[method] for r in selected])),
                             'scores_sha256':report['scores_sha256']})
    if not rows: raise ValueError('Supply at least one evaluated control cohort')
    out=Path(output_dir)
    _csv(out/'control_response.csv',list(rows[0]),rows)
    result={'reference_signature':reference,'rows':rows,
            'interpretation':'Descriptive fixed-pool response only. Repeated images/overlapping cohorts are not independent replicates. Human-class strata are preserved separately from classifier predictions. No MM labels are inferred from control-family membership.'}
    write_json(out/'control_response.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    strata=sorted({(r['stratum'],r['value']) for r in rows})
    for index,(stratum,value) in enumerate(strata):
        selected=[r for r in rows if (r['stratum'],r['value'])==(stratum,value)]
        paths=list(dict.fromkeys(r['evaluation'] for r in selected))
        fig,ax=plt.subplots(figsize=(max(6,min(14,len(paths)*.7)),4),constrained_layout=True)
        for method in SCORE_NAMES:
            rates={r['evaluation']:r['candidate_rate'] for r in selected if r['method']==method}
            ax.plot(range(len(paths)),[rates.get(path,np.nan) for path in paths],'o-',label=method)
        ax.set(xticks=range(len(paths)),xticklabels=[Path(p).name for p in paths],
               ylabel='Candidate fraction',ylim=(0,1),title=f'{stratum}: {value} (descriptive fixed cohorts)')
        ax.tick_params(axis='x',labelrotation=45);ax.legend(fontsize=6);ax.grid(alpha=.2)
        fig.savefig(out/f'control_response_{index}.png',dpi=180);fig.savefig(out/f'control_response_{index}.pdf');plt.close(fig)
    return result
