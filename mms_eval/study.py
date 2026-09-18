"""Freeze per-source human sample quotas and exclusions before annotation."""
from pathlib import Path

import numpy as np

from .annotation import _csv
from .artifacts import load_evaluation
from .images import load_rgb, pixel_hash, save_png_atomic
from .utils import atomic_text, read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def register_study(spec, output_dir):
    """Freeze one random sample per source; never silently reallocate a quota.

    ``sources`` specify source_id, evaluation, and count. ``exclusion_packs``
    must explicitly list all prior development/diagnostic packs, or []. Stored
    registration proves software choices were frozen, not an external timestamp.
    """
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Study registration is immutable; choose a new named revision')
    role = spec.get('role')
    if role not in ('development', 'final_random', 'diagnostic') or 'exclusion_packs' not in spec:
        raise ValueError('Declare a study role and all exclusion_packs explicitly')
    if not spec.get('annotation_guide_version') or not spec.get('annotation_guide') or not spec.get('analysis_plan'):
        raise ValueError('Freeze an annotation guide version and analysis plan before sampling')
    if not isinstance(spec.get('seed'), int) or not spec.get('sources'):
        raise ValueError('A fixed integer seed and source quotas are required')
    excluded_ids, excluded_hashes, excluded_pixels, exclusion_provenance = set(), set(), set(), []
    excluded_parents=set()
    for pack in spec['exclusion_packs']:
        root = Path(pack).resolve()
        meta = read_json(root/'pack.json')
        if sha256_file(root/'private_key.jsonl') != meta['private_key_sha256']:
            raise ValueError('An exclusion pack mapping changed')
        for row in read_jsonl(root/'private_key.jsonl'):
            excluded_ids.add(row['image_id'])
            excluded_parents.add(row.get('parent_id',row['image_id']))
            excluded_hashes.add(row['sha256'])
            if sha256_file(row['path']) != row['sha256']:
                raise ValueError('Exclusion source image changed')
            excluded_pixels.add(pixel_hash(load_rgb(row['path'])))
        exclusion_provenance.append({'path': str(root), 'pack_sha256': sha256_file(root/'pack.json')})
    rng, selections, sources, source_ids = np.random.RandomState(spec['seed']), [], [], set()
    # Global duplicate identities are rejected across sources, keeping natural
    # source prevalence interpretable and preventing repeated blinded originals.
    pool_ids, pool_pixels, pool_parents = set(), set(), set()
    eligible_pools={}
    for source in spec['sources']:
        source_id, count = source['source_id'], source['count']
        if role=='final_random' and not source.get('eligibility_declaration','').strip():
            raise ValueError('Final sources require a recorded eligibility declaration covering prior viewing/development use and exclusions')
        if not source_id or source_id in source_ids or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError('Study sources need unique IDs and positive integer quotas')
        source_ids.add(source_id)
        report, scores, records, _ = load_evaluation(source['evaluation'])
        eligible = []
        for record, score in zip(records, scores):
            if sha256_file(record['path']) != record['sha256']:
                raise ValueError('Source image changed after evaluation')
            decoded = pixel_hash(load_rgb(record['path']))
            image_id = record['image_id']
            parent=record.get('parent_id',image_id)
            if image_id in excluded_ids or record['sha256'] in excluded_hashes or decoded in excluded_pixels or parent in excluded_parents:
                continue
            if image_id in pool_ids or decoded in pool_pixels:
                raise ValueError('Duplicate original within/across study sources; provide independent cohorts')
            pool_ids.add(image_id); pool_pixels.add(decoded)
            if role=='final_random' and parent in pool_parents:
                raise ValueError('Final independent-original quotas cannot count repeated parent identities')
            pool_parents.add(parent)
            if role == 'final_random' and record.get('cohort_role', '').startswith(('diagnostic','development')):
                raise ValueError('Diagnostic/development cohorts cannot become final random prevalence samples')
            eligible.append({**record, **score, 'study_source_id': source_id,
                             'display_policy': report['image_policy'], 'display_classes': report['mms']['classes'],
                             'source_pixel_sha256': decoded})
        if count > len(eligible):
            raise ValueError(f'Insufficient eligible images for source {source_id}; do not transfer its quota')
        eligible_pools[source_id]=eligible
        sources.append({'source_id': source_id, 'n_population': len(records), 'n_eligible': len(eligible), 'n_sampled': count,
                        'evaluation': str(Path(source['evaluation']).resolve()),
                        'scores_sha256': report['scores_sha256'], 'manifest_sha256': report['manifest_sha256'],
                        'protocol_id': report['protocol_id'], 'reference_signature': report['reference_signature']})
        sources[-1]['eligibility_declaration']=source.get('eligibility_declaration')
    paired_groups=spec.get('paired_source_groups',[])
    source_meta={r['source_id']:r for r in sources}
    grouped=set()
    pairing=[]
    for members in paired_groups:
        if not isinstance(members,list) or len(members)<2 or len(set(members))!=len(members) or not set(members)<=source_ids or grouped & set(members):
            raise ValueError('Paired source groups need distinct known sources, each in at most one group')
        grouped.update(members)
        quotas={source_meta[name]['n_sampled'] for name in members}
        if len(quotas)!=1:
            raise ValueError('Paired sources must have equal quotas')
        by_pair={}
        for name in members:
            pool=eligible_pools[name]
            if any(r.get('pair_id') is None or r.get('pair_id')=='' for r in pool):
                raise ValueError('Every eligible paired-source image needs an explicit pair_id')
            pairs={str(r['pair_id']):r for r in pool}
            if len(pairs)!=len(pool): raise ValueError('Duplicate pair_id within a source')
            by_pair[name]=pairs
        common=sorted(set.intersection(*(set(rows) for rows in by_pair.values())))
        count=next(iter(quotas))
        if len(common)<count:
            raise ValueError('Insufficient shared eligible pairs; do not replace pairs with independent samples')
        chosen=[common[i] for i in rng.choice(len(common),count,replace=False)]
        for name in members:
            selections.extend(by_pair[name][p] for p in chosen)
            source_meta[name]['n_eligible_before_pair_intersection']=source_meta[name]['n_eligible']
            source_meta[name]['n_eligible']=len(common)
        pairing.append({'sources':members,'eligible_pair_n':len(common),'sampled_pair_ids':chosen})
    # Explicit pair identities appearing across sources must not be silently
    # discarded by independent sampling.
    pair_sources={}
    for name,pool in eligible_pools.items():
        for row in pool:
            if row.get('pair_id') is not None:
                pair_sources.setdefault(str(row['pair_id']),set()).add(name)
    for members in pair_sources.values():
        if len(members)>1 and not any(members<=set(g) for g in paired_groups):
            raise ValueError('Declare paired_source_groups for cross-source pair IDs, or namespace independent identities')
    for name,pool in eligible_pools.items():
        if name not in grouped:
            count=source_meta[name]['n_sampled']
            selections.extend(pool[i] for i in rng.choice(len(pool),count,replace=False))
    selections = [selections[i] for i in rng.permutation(len(selections))]
    write_jsonl(out/'selected_private.jsonl', selections)
    result = {'schema_version': 1, 'role': role, 'seed': spec['seed'], 'n': len(selections), 'sources': sources,
              'exclusions': exclusion_provenance, 'annotation_guide_version': spec['annotation_guide_version'],
              'annotation_guide': spec['annotation_guide'],
              'analysis_plan': spec['analysis_plan'], 'sampling': 'uniform_without_replacement_fixed_source_quotas_shared_ids_for_declared_pairs',
              'paired_source_groups':pairing,
              'selected_sha256': sha256_file(out/'selected_private.jsonl'),
              'interpretation': 'Estimate source-specific rates. Quota totals do not estimate a pooled natural prevalence. Registration is locally frozen, not externally timestamped.'}
    result['integrity_sha256'] = stable_hash(result)
    write_json(out/'registration.json', result)
    return result


def study_pack(registration, output_dir):
    root, out = Path(registration).resolve(), Path(output_dir).resolve()
    if root.is_file():
        root = root.parent
    meta = read_json(root/'registration.json')
    if stable_hash({k:v for k,v in meta.items() if k != 'integrity_sha256'}) != meta['integrity_sha256']:
        raise ValueError('Study registration changed')
    if sha256_file(root/'selected_private.jsonl') != meta['selected_sha256']:
        raise ValueError('Registered sample changed')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Annotation pack already exists; preserve original blind identities')
    keys, template = [], []
    for i, row in enumerate(read_jsonl(root/'selected_private.jsonl')):
        blind = f'{i+1:05d}'
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError('Registered image changed')
        path = out/'annotator'/'images'/f'{blind}.png'
        save_png_atomic(load_rgb(row['path'], row['display_policy']), path)
        keys.append({**row, 'blind_id': blind, 'display_sha256': sha256_file(path)})
        template.append({'blind_id': blind, 'annotator_id': '', 'label': '', 'other_defects': '', 'human_target_class': '',
                         'reason': '', 'seconds': '', 'annotation_guide_version': meta['annotation_guide_version']})
    _csv(out/'annotator'/'labels_template.csv', list(template[0]), template)
    atomic_text(out/'annotator'/'README.md',
                '# 独立人工标注\n\n仅将 annotator 文件夹交给标注者。每人独立填写一份 CSV，不查看模型分数或讨论后再填。\n\n'
                'yes：同一主体存在可指出的不合理语义融合；no：无上述融合，可同时存在其他缺陷或多个完整主体；'
                'unknown：无法可靠判断。空白表示尚未标注。其他缺陷另填 other_defects。\n\n'
                f'标注指南版本：{meta["annotation_guide_version"]}。先按该版本指南统一边界，再独立完成本批标注。\n\n'
                + meta['annotation_guide'] + '\n')
    write_jsonl(out/'private_key.jsonl', keys)
    result = {**meta, 'registration_sha256': sha256_file(root/'registration.json'),
              'private_key_sha256': sha256_file(out/'private_key.jsonl'),
              'human_validity': 'pending', 'allowed_labels': ['yes','no','unknown']}
    write_json(out/'pack.json', result)
    return result
