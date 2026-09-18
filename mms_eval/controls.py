"""Registered diagnostic cohorts; human judgments are never inferred from edits."""
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .analysis_reporting import load_final_labels
from .images import collect_images, load_rgb, pixel_hash, save_png_atomic, validate_policy
from .utils import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def build_composition_controls(source, labels_file, spec, output_dir, *, seed=2026091206):
    """Change human-confirmed single-class quotas within fixed conditional pools."""
    from .human_labels import _read_csv
    out = _new_output(output_dir)
    records, labels, meta = _confirmed_pool(source, labels_file)
    details = read_jsonl(labels_file) if Path(labels_file).suffix == '.jsonl' else _read_csv(labels_file)
    classes = {r['image_id']: r.get('human_target_class', '') for r in details}
    pools = {}
    for record in records:
        name = classes.get(record['image_id'], '')
        if labels.get(record['image_id']) == 0 and name and name not in ('unknown', '无法确定'):
            pools.setdefault(name, []).append(record)
    rng, cohorts, total = np.random.RandomState(seed), [], None
    permutations = {c: [pool[i] for i in rng.permutation(len(pool))] for c,pool in sorted(pools.items())}
    seen = set()
    for item in spec['cohorts']:
        name, quotas = item['cohort_id'], item['class_counts']
        if name in seen or not name or not quotas or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in quotas.values()):
            raise ValueError('Composition cohorts need unique IDs and nonnegative integer class quotas')
        seen.add(name)
        n = sum(quotas.values())
        if n < 1 or total is not None and n != total:
            raise ValueError('Composition cohorts must have the same positive total N')
        total = n
        rows = []
        for c,count in sorted(quotas.items()):
            if count > len(permutations.get(c, [])):
                raise ValueError('Insufficient independently confirmed human class pool: '+c)
            rows.extend({**r, 'human_target_class': c, 'control_cohort_id': name,
                         'cohort_role': 'diagnostic_human_class_composition'} for r in permutations.get(c,[])[:count])
        filename = f'composition_{len(cohorts):03d}.jsonl'
        write_jsonl(out/filename, rows)
        cohorts.append({'cohort_id': name, 'n': n, 'class_counts': quotas, 'manifest': filename,
                        'manifest_sha256': sha256_file(out/filename)})
    if not cohorts:
        raise ValueError('Supply at least one composition cohort')
    result = {'kind': 'human_class_composition', 'role': 'diagnostic', 'seed': seed,
              'pool_counts': {c:len(p) for c,p in pools.items()}, 'cohorts': cohorts,
              'labels_sha256': sha256_file(labels_file), 'label_provenance': meta,
              'interpretation': 'Fixed confirmed non-MM class pools; paired nested selections within class. Report both overall and human-class candidate rates; changing overall MMS is not evidence of changing MM.'}
    write_json(out/'controls.json', result)
    return result


def build_duplicate_controls(source, output_dir, *, n=1000, rates=(0,.25,.5), seed=2026091207):
    """Replace a registered fraction by copies, retaining parent identity and fixed N."""
    out, records = _new_output(output_dir), collect_images(source)
    if not isinstance(n,int) or isinstance(n,bool) or n < 2 or n > len(records):
        raise ValueError('A feasible fixed N >= 2 is required')
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(r) or not 0<=r<1 for r in rates):
        raise ValueError('Distinct duplicate rates in [0,1) are required')
    if len({pixel_hash(load_rgb(r['path'])) for r in records}) != len(records):
        raise ValueError('Duplicate-control source must contain unique decoded originals')
    rng = np.random.RandomState(seed)
    base = [records[i] for i in rng.permutation(len(records))[:n]]
    cohorts = []
    for index,rate in enumerate(rates):
        duplicate_n = int((Decimal(str(rate))*n).to_integral_value(rounding=ROUND_HALF_UP))
        unique_n = n-duplicate_n
        if unique_n < 1:
            raise ValueError('Rounded duplicate count leaves no originals')
        originals = base[:unique_n]
        selected = originals+[originals[i] for i in rng.randint(0,unique_n,duplicate_n)]
        rows = [{**r, 'image_id': f'dup{index}:{i}:{r["image_id"]}',
                 'parent_id': r.get('parent_id',r['image_id']), 'original_image_id': r['image_id'],
                 'duplicate_copy': i>=unique_n, 'cohort_role': 'diagnostic_duplicate_rate',
                 'control_cohort_id': f'duplicate_{index}'} for i,r in enumerate(selected)]
        filename = f'duplicate_{index}.jsonl'
        write_jsonl(out/filename, rows)
        cohorts.append({'requested_duplicate_rate': rate, 'actual_duplicate_rate': duplicate_n/n,
                        'n': n, 'unique_original_n': unique_n, 'duplicate_n': duplicate_n,
                        'manifest': filename, 'manifest_sha256': sha256_file(out/filename)})
    result = {'kind': 'duplicate_rate', 'role': 'diagnostic', 'seed': seed, 'cohorts': cohorts,
              'interpretation': 'Fixed total N with nested unique originals and uniform copy parents. Scores/composition and coverage can change; no invariance is presumed. Copies are not independent observations.'}
    write_json(out/'controls.json', result)
    return result


def _confirmed_pool(source, labels_file):
    records = collect_images(source)
    # Reject alternate encodings/identities of one original before selection.
    seen, parents = set(), set()
    for row in records:
        digest = pixel_hash(load_rgb(row['path']))
        parent = row.get('parent_id', row['image_id'])
        if digest in seen or parent in parents:
            raise ValueError('Duplicate decoded content or parent in independent control pool')
        seen.add(digest)
        parents.add(parent)
    labels, metadata = load_final_labels(labels_file,records=records)
    if metadata['status'] not in ('complete', 'complete_with_recorded_exceptions'):
        raise ValueError('Controls require completed sealed human labels, including any explicit exceptions')
    # A sealed multi-source study remains intact while a declared manifest
    # selects one source/class for controls. Never require rewriting its labels.
    record_ids={r['image_id'] for r in records}
    all_label_n=len(labels)
    labels={key:value for key,value in labels.items() if key in record_ids}
    if not labels: raise ValueError('No finalized labels match the selected control pool')
    metadata={**metadata,'control_pool_label_subset':{'full_label_n':all_label_n,'matched_n':len(labels),
              'selected_identity_sha256':stable_hash([{'image_id':r['image_id'],'sha256':r['sha256']}
                                                     for r in records if r['image_id'] in labels])}}
    return records, labels, metadata


def _new_output(output_dir):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve frozen controls; use a new output directory')
    return out


def build_mixture_controls(source, labels_file, output_dir, *, n=100,
                           proportions=(0, .05, .1, .2, .4, .6), repetitions=20,
                           seed=2026091201):
    """Mix confirmed positive/negative pools, without replacement within a cohort.

    One permutation per class and repetition gives paired nested positive pools
    and nested negative pools across proportions. Repetitions reuse a finite pool
    and therefore are Monte Carlo variability, not independent population draws.
    """
    out = _new_output(output_dir)
    if n < 1 or repetitions < 1 or not proportions or len(set(proportions)) != len(proportions):
        raise ValueError('Positive n/repetitions and distinct proportions are required')
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in proportions):
        raise ValueError('Proportions must be finite and between zero and one')
    records, labels, meta = _confirmed_pool(source, labels_file)
    pools = {value: [r for r in records if labels.get(r['image_id']) == value] for value in (True, False)}
    confirmed=pools[True]+pools[False]
    if len({r.get('setting_id','unspecified') for r in confirmed})>1:
        raise ValueError('Group controlled mixtures by generation source before mixing; pooled sources can confound the response')
    from .human_labels import _read_csv
    details=read_jsonl(labels_file) if Path(labels_file).suffix=='.jsonl' else _read_csv(labels_file)
    known_classes={r['human_target_class'] for r in details if r['image_id'] in labels and
                   r.get('human_target_class') not in (None,'','unknown','无法确定')}
    if len(known_classes)>1:
        raise ValueError('Group controlled mixtures by human semantic class before mixing')
    counts = [int((Decimal(str(p))*n).to_integral_value(rounding=ROUND_HALF_UP)) for p in proportions]
    if max(counts) > len(pools[True]) or n-min(counts) > len(pools[False]):
        raise ValueError('Insufficient independently confirmed positive/negative images; sampling with replacement is prohibited')
    rng, cohorts = np.random.RandomState(seed), []
    for repetition in range(repetitions):
        pos = [pools[True][i] for i in rng.permutation(len(pools[True]))]
        neg = [pools[False][i] for i in rng.permutation(len(pools[False]))]
        for index, (p, positive_n) in enumerate(zip(proportions, counts)):
            selected = pos[:positive_n]+neg[:n-positive_n]
            selected = [selected[i] for i in rng.permutation(n)]
            cohort_id = f'mix_r{repetition:03d}_p{index:02d}'
            rows = [{**r, 'cohort_role': 'diagnostic_controlled_mixture',
                     'control_cohort_id': cohort_id} for r in selected]
            manifest = cohort_id+'.jsonl'
            write_jsonl(out/manifest, rows)
            cohorts.append({'cohort_id': cohort_id, 'repetition': repetition,
                            'requested_proportion': p, 'actual_proportion': positive_n/n,
                            'n': n, 'n_positive': positive_n, 'manifest': manifest,
                            'manifest_sha256': sha256_file(out/manifest),
                            'source_counts': dict(Counter(str(r.get('setting_id', 'unspecified')) for r in rows))})
    result = {'kind': 'controlled_mixture', 'role': 'diagnostic', 'seed': seed,
              'pool_positive_n': len(pools[True]), 'pool_negative_n': len(pools[False]),
              'human_class_composition_status':'one_recorded_class' if known_classes else 'uncollected_or_unresolved',
              'label_provenance': meta, 'labels_sha256': sha256_file(labels_file), 'cohorts': cohorts,
              'sampling': 'paired class permutations; no replacement within each cohort',
              'interpretation': 'Designed prevalence, not natural prevalence. Report realized source composition. Repetitions reuse the confirmed finite pool.'}
    write_json(out/'controls.json', result)
    return result


def build_degradation_controls(source, labels_file, output_dir, *, base_n=30,
                               seed=2026091202, image_policy=None):
    """Transform confirmed negatives in canonical pixel coordinates, then reblind.

    Includes unmodified canonical parents and nine edits per parent. Standard
    deviations for Gaussian noise are relative to [0,1]. All outputs are PNGs,
    including the decoded JPEG round trip, to avoid further codec differences.
    """
    out = _new_output(output_dir)
    records, labels, meta = _confirmed_pool(source, labels_file)
    negatives = [r for r in records if labels.get(r['image_id']) == 0]
    if base_n < 1 or base_n > len(negatives):
        raise ValueError('Insufficient independent confirmed negatives for requested base_n')
    policy, rng = validate_policy(image_policy), np.random.RandomState(seed)
    selected = [negatives[i] for i in rng.choice(len(negatives), base_n, replace=False)]
    rows = []
    operations = [('original', 0)]+[('blur', s) for s in (1,2,4)]+[('noise', s) for s in (.02,.05,.1)]+[('jpeg', q) for q in (90,60,30)]
    for base in selected:
        original = load_rgb(base['path'], policy)
        # One noise field per parent makes severity comparisons paired.
        noise = rng.normal(size=(original.height, original.width, 3))
        for operation, value in operations:
            if operation == 'blur':
                transformed = original.filter(ImageFilter.GaussianBlur(value))
            elif operation == 'noise':
                arr = np.rint(np.clip(np.asarray(original)/255+value*noise, 0, 1)*255).astype(np.uint8)
                transformed = Image.fromarray(arr)
            elif operation == 'jpeg':
                stream = io.BytesIO()
                original.save(stream, format='JPEG', quality=value, subsampling=2)
                stream.seek(0)
                with Image.open(stream) as decoded:
                    transformed = decoded.convert('RGB').copy()
            else:
                transformed = original.copy()
            image_id = 'control:'+stable_hash({'parent': base['image_id'], 'sha256': base['sha256'],
                                              'operation': operation, 'severity': value,
                                              'policy': policy, 'seed': seed})[:24]
            path = out/'images'/f'{image_id.split(":")[1]}.png'
            save_png_atomic(transformed, path)
            rows.append({'image_id': image_id, 'path': str(path), 'sha256': sha256_file(path),
                         'parent_id': base['image_id'], 'parent_sha256': base['sha256'],
                         'setting_id': base.get('setting_id', 'unspecified'),
                         'control_family': operation, 'severity': value, 'requires_reannotation': True,
                         'cohort_role': 'diagnostic_degradation', 'base_label_provenance': 'sealed_human_no'})
    write_jsonl(out/'manifest.jsonl', rows)
    result = {'kind': 'degradation', 'role': 'diagnostic', 'n': len(rows), 'base_n': base_n,
              'seed': seed, 'image_policy': policy, 'manifest_sha256': sha256_file(out/'manifest.jsonl'),
              'labels_sha256': sha256_file(labels_file), 'label_provenance': meta,
              'status': 'awaiting_evaluation_and_independent_reannotation',
              'operations': operations, 'noise_pairing': 'shared standard normal field within each parent',
              'interpretation': 'Only the source parents were confirmed non-MM. Edited outputs have no assumed label. Use parent-grouped paired analysis after reannotation.'}
    write_json(out/'controls.json', result)
    return result


def build_curated_controls(spec, output_dir, *, seed=2026091203):
    """Register human-curated OOD/multiple-subject/candidate source cohorts.

    Family membership is a declared selection criterion, not a final MM label.
    Each source entry supplies family, input, count, and curation_reason.
    """
    out, rng = _new_output(output_dir), np.random.RandomState(seed)
    rows, source_meta, seen, families = [], [], set(), set()
    for source in spec['sources']:
        family, count = source['family'], source['count']
        if family not in ('ood','legal_multiple_subjects','mm_candidates') or family in families:
            raise ValueError('Declare each OOD/multiple-subject/candidate family at most once')
        families.add(family)
        records = collect_images(source['input'])
        if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > len(records) or not source.get('curation_reason'):
            raise ValueError('Curated controls require feasible positive quotas and a recorded selection reason')
        for record in records:
            decoded = pixel_hash(load_rgb(record['path']))
            if decoded in seen:
                raise ValueError('Duplicate original across curated control sources')
            seen.add(decoded)
        selected = [records[i] for i in rng.choice(len(records),count,replace=False)]
        rows.extend({**r, 'control_family': family, 'requires_reannotation': True,
                     'cohort_role': 'diagnostic_curated'} for r in selected)
        source_meta.append({'family': family, 'population_n': len(records), 'n': count,
                            'curation_reason': source['curation_reason']})
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Supply nonempty curated controls with unique identities')
    write_jsonl(out/'manifest.jsonl', rows)
    result = {'kind': 'curated_controls', 'role': 'diagnostic', 'seed': seed, 'n': len(rows),
              'sources': source_meta, 'manifest_sha256': sha256_file(out/'manifest.jsonl'),
              'status': 'awaiting_evaluation_and_independent_reannotation',
              'interpretation': 'OOD, multiple-subject, and candidate are selection strata. They are not automatic MM ground truth or natural prevalence samples.'}
    write_json(out/'controls.json', result)
    return result
