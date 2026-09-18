"""Human-grounded review-budget curves and analyst-only confusion galleries."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .analysis_reporting import load_final_labels
from .annotation import _csv
from .artifacts import load_evaluation
from .images import load_rgb, save_png_atomic
from .semantic import SCORE_NAMES
from .utils import sha256_file, write_json


def review_budget_curves(rows, labels, *, random_repetitions=1000, bootstrap_repetitions=2000,
                        seed=2026091204):
    """Rank a fixed labeled cohort, retaining unknowns in review costs.

    Ties use fixed image-ID order as registered in the manual. Random review
    order has a separate Monte Carlo distribution; paired whole-group bootstrap
    describes uncertainty in the finite labeled cohort. Unknowns are retained.
    """
    rows = sorted((r for r in rows if r['image_id'] in labels), key=lambda r:r['image_id'])
    if not rows:
        raise ValueError('No labeled images match the evaluation')
    n = len(rows)
    y = np.array([labels[r['image_id']] == 1 for r in rows], dtype=float)
    known = np.array([labels[r['image_id']] is not None for r in rows])
    total = int(y.sum())
    budgets = sorted(set([0, n]+[int(np.ceil(n*x)) for x in (.01,.02,.05,.1,.2,.3,.5,.75)]))
    result, rng = [], np.random.RandomState(seed)
    if random_repetitions < 2 or bootstrap_repetitions < 2:
        raise ValueError('Random-order and bootstrap repetitions must each be at least two')
    random_counts = np.array([np.r_[0,np.cumsum(y[rng.permutation(n)])][budgets]
                              for _ in range(random_repetitions)])
    from .detection import _design, _identities, bootstrap_indices
    groups, strata = _identities(rows, 'auto', 'setting_id')
    design = _design(groups,strata)
    draws = list(bootstrap_indices(groups,strata,repetitions=bootstrap_repetitions,seed=seed))
    boot_found, boot_recall = {}, {}
    for method in SCORE_NAMES:
        scores = np.array([r[method] for r in rows])
        order = np.argsort(-scores, kind='stable')
        found = np.r_[0,np.cumsum(y[order])]
        resolved = np.r_[0,np.cumsum(known[order])]
        boot_found[method], boot_recall[method] = [], []
        for indices in draws:
            ranked = indices[np.lexsort((indices,-scores[indices]))]
            values = np.r_[0,np.cumsum(y[ranked])][budgets]
            boot_found[method].append(values)
            boot_recall[method].append(values/y[indices].sum() if y[indices].sum() else np.full(len(budgets),np.nan))
        boot_found[method], boot_recall[method] = np.array(boot_found[method]), np.array(boot_recall[method])
        for j,k in enumerate(budgets):
            recall_samples = boot_recall[method][:,j]
            recall_samples = recall_samples[np.isfinite(recall_samples)]
            can_estimate = design['resampleable_groups']>=2
            result.append({'method': method, 'reviewed_n': k, 'review_fraction': k/n,
                           'confirmed_MM_found': int(found[k]), 'resolved_reviewed_n': int(resolved[k]),
                           'unknown_reviewed_n': int(k-resolved[k]),
                           'recall_of_confirmed_MM': float(found[k]/total) if total else None,
                           'precision_on_resolved_reviewed': float(found[k]/resolved[k]) if resolved[k] else None,
                           'confirmed_MM_per_review': float(found[k]/k) if k else None,
                           'found_paired_bootstrap_ci95': np.quantile(boot_found[method][:,j],[.025,.975]).tolist() if can_estimate else None,
                           'recall_paired_bootstrap_ci95': np.quantile(recall_samples,[.025,.975]).tolist() if can_estimate and len(recall_samples)>=.8*bootstrap_repetitions else None,
                           'recall_invalid_resamples': bootstrap_repetitions-len(recall_samples),
                           'uniform_review_expected_MM': k*total/n, 'cohort_n': n,
                           'uniform_review_empirical_mean': float(random_counts[:,j].mean()),
                           'uniform_review_order_quantiles_025_975': np.quantile(random_counts[:,j],[.025,.975]).tolist(),
                           'random_order_repetitions': random_repetitions, 'bootstrap_repetitions': bootstrap_repetitions,
                           'confirmed_MM_n': total, 'unknown_n': sum(labels[r['image_id']] is None for r in rows)})
    for row in result:
        j = budgets.index(row['reviewed_n'])
        delta = boot_found['entropy'][:,j]-boot_found[row['method']][:,j]
        row['entropy_minus_method_found_ci95'] = np.quantile(delta,[.025,.975]).tolist() if design['resampleable_groups']>=2 else None
    return result


def detection_diagnostics(evaluation, final_labels, output_dir, *, max_per_cell=8, seed=2026091204):
    if max_per_cell < 1:
        raise ValueError('max_per_cell must be positive')
    report, rows, records, _ = load_evaluation(evaluation)
    labels, label_meta = load_final_labels(final_labels,records=records)
    allow_curves=label_meta.get('pack_role')=='final_random' and bool(label_meta.get('study_registration_sha256'))
    from .human_labels import _read_csv
    from .utils import read_jsonl
    details = read_jsonl(final_labels) if Path(final_labels).suffix=='.jsonl' else _read_csv(final_labels)
    details = {r['image_id']:r for r in details}
    records_by_id = {r['image_id']: r for r in records}
    sources = defaultdict(list)
    for row in rows:
        if row['image_id'] in labels:
            sources[str(records_by_id[row['image_id']].get('setting_id', 'unspecified'))].append({**records_by_id[row['image_id']], **row})
    curves, galleries, rng = [], [], np.random.RandomState(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for source_index, (source, selected) in enumerate(sorted(sources.items())):
        if allow_curves:
            panel = review_budget_curves(selected, labels, seed=seed)
            curves.extend({'source': source, **r} for r in panel)
            fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
            for method in SCORE_NAMES:
                values = [r for r in panel if r['method'] == method]
                ax.plot([r['review_fraction'] for r in values],
                        [r['recall_of_confirmed_MM'] if r['recall_of_confirmed_MM'] is not None else np.nan for r in values],
                        marker='.', label=method)
            ax.plot([0,1],[0,1], '--', color='gray', label='uniform review expectation')
            ax.set(xlabel='Fraction of labeled cohort reviewed', ylabel='Fraction of confirmed MM found', xlim=(0,1), ylim=(0,1))
            ax.legend(fontsize=7); ax.grid(alpha=.2)
            fig.savefig(out/f'review_budget_{source_index}.png', dpi=180)
            fig.savefig(out/f'review_budget_{source_index}.pdf'); plt.close(fig)
        cells = defaultdict(list)
        for row in selected:
            label = labels[row['image_id']]
            cell = 'unknown' if label is None else ('TP' if row['flag_entropy'] else 'FN') if label else ('FP' if row['flag_entropy'] else 'TN')
            cells[cell].append(row)
        for cell in ('TP','FP','FN','TN','unknown'):
            pool = cells[cell]
            chosen = [pool[i] for i in rng.permutation(len(pool))[:max_per_cell]]
            sheet = Image.new('RGB', (4*240, max(1,(len(chosen)+3)//4)*270), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(chosen):
                record = records_by_id[row['image_id']]
                if sha256_file(record['path']) != record['sha256']:
                    raise ValueError('Gallery source image changed')
                im = load_rgb(record['path'], report['image_policy']); im.thumbnail((196,196))
                x,y = index%4*240, index//4*270
                tau = report['calibrations']['entropy']['threshold']
                posterior = ','.join(f'{p:.3f}' for p in row['probabilities'])
                sheet.paste(im,(x,y)); draw.text((x+2,y+198), f'{cell} H={row["entropy"]:.4f}\ntau={tau}\np=[{posterior}]', fill='black')
                galleries.append({'source': source, 'cell': cell, 'image_id': row['image_id'], 'position': index,
                                  'entropy':row['entropy'], 'threshold':tau, 'probabilities':row['probabilities'],
                                  'class_order':report['mms']['classes'], 'human_MM':details[row['image_id']]['mm_label_final'],
                                  'other_defects':details[row['image_id']].get('other_defects',''),
                                  'file': f'gallery_{source_index}_{cell}.png'})
            if not chosen:
                draw.text((10,10), f'{cell}: no observed cases', fill='black')
            save_png_atomic(sheet, out/f'gallery_{source_index}_{cell}.png')
    if not sources:
        raise ValueError('No final label IDs match this evaluation')
    if curves: _csv(out/'review_budget.csv', list(curves[0]), curves)
    _csv(out/'gallery_key.csv', list(galleries[0]) if galleries else ['source','cell','image_id','position','file'], galleries)
    result = {'sources': list(sorted(sources)), 'labels_sha256': sha256_file(final_labels),
              'label_provenance': label_meta, 'scores_sha256': report['scores_sha256'],
              'role': 'analyst_diagnostic_only', 'tie_handling': 'ascending image_id within tied scores',
              'review_budget_status':'registered_final_random' if allow_curves else 'unavailable_requires_registered_final_random_labels',
              'interpretation': 'Curves describe the labeled cohort only. Unknown judgments cost reviews and do not count as confirmed MM. Galleries are uniform samples within observed entropy confusion cells; do not reuse them as a blinded final cohort.'}
    write_json(out/'diagnostics.json', result)
    return result
