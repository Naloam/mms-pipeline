"""Conditional unlabeled-posterior audit with fixed granularity and all declared seeds."""
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import load_evaluation
from .feature_cache import cached_features
from .images import collect_images, load_rgb, save_png_atomic
from .pipeline import code_signature, freeze_request, load_reference, save_arrays, write_report
from .semantic import SCORE_NAMES, aggregate_mms, make_score_rows, posterior_scores, rank_calibrate
from .splits import load_split
from .utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def l2_normalize(features):
    values=np.asarray(features,dtype=np.float64)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A nonempty finite feature matrix is required')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or (norms<=0).any():
        raise ValueError('Prototype features/centers must have finite nonzero norm')
    return values/norms


def fit_prototypes(features, *, k=3, seed=0):
    """This boundary accepts a feature matrix only; target labels have no input."""
    from sklearn.cluster import KMeans
    values=l2_normalize(features)
    if not isinstance(k,int) or isinstance(k,bool) or k<2 or k>len(values):
        raise ValueError('Task-given K must be between two and the number of fit images')
    model=KMeans(n_clusters=k,random_state=seed,n_init=10,algorithm='lloyd').fit(values)
    if len(set(model.labels_))!=k:
        raise ValueError('Fewer distinct fitted clusters than the given K')
    return l2_normalize(model.cluster_centers_)


def prototype_logits(features, centers, *, temperature=.1):
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Prototype temperature must be positive and finite')
    values,centers=l2_normalize(features),l2_normalize(centers)
    if values.shape[1]!=centers.shape[1]:
        raise ValueError('Feature dimensions differ from prototype centers')
    return np.clip(values@centers.T,-1,1)/temperature


def _without_labels(records):
    return [{k:r[k] for k in ('image_id','path','sha256') if k in r} for r in records]


def _cluster_audit(records, logits, centers, embeddings, classes, output_dir, *, seed, image_policy=None):
    from PIL import Image,ImageDraw
    out=Path(output_dir)
    cluster=np.argmax(logits,axis=1)
    composition=[]
    gallery=[]
    rng=np.random.RandomState(seed)
    for c in range(len(centers)):
        indices=np.flatnonzero(cluster==c)
        counts=Counter(records[i]['class_name'] if 'class_name' in records[i] else classes[records[i]['label']] for i in indices)
        composition.append({'cluster_id':c,'n':len(indices),'true_class_counts':dict(counts)})
        # Nearest and uniform representatives are both visible, with selection recorded.
        similarities=l2_normalize(embeddings[indices])@centers[c] if len(indices) else np.array([])
        closest=indices[np.argsort(-similarities,kind='stable')[:4]]
        remainder=np.array([i for i in indices if i not in set(closest)],dtype=int)
        random=remainder[rng.permutation(len(remainder))[:4]]
        selected=[(int(i),'closest_to_center') for i in closest]+[(int(i),'uniform_remaining') for i in random]
        sheet=Image.new('RGB',(4*180,2*206),'white');draw=ImageDraw.Draw(sheet)
        for j,(i,rule) in enumerate(selected):
            im=load_rgb(records[i]['path'],image_policy);im.thumbnail((176,176))
            x,y=j%4*180,j//4*206;sheet.paste(im,(x,y))
            draw.text((x+2,y+178),f'cluster {c}\n{rule}',fill='black')
            gallery.append({'cluster_id':c,'image_id':records[i]['image_id'],'selection':rule,'position':j,
                            'true_class':classes[records[i]['label']]})
        save_png_atomic(sheet,out/f'cluster_{c}.png')
    write_jsonl(out/'representatives.jsonl',gallery)
    result={'composition':composition,'representative_seed':seed,'status':'class_composition_available_visual_semantic_review_pending',
            'interpretation':'Cluster IDs have no preassigned semantic meaning. Inspect backgrounds/colors/poses in the representatives; do not refit or choose seeds using these labels.'}
    write_json(out/'cluster_audit.json',result)
    return result


def run_prototype_audit(real_reference, split_dir, evaluation, output_dir, *, repository, weights,
                        device='cpu', batch_size=32, cache_dir=None, k=3, temperature=.1, seeds=(0,1,2,3,4)):
    """Build label-free prototypes, calibrate real references, and rescore unchanged images.

    Saved per-seed references/evaluations use the ordinary downstream detection
    interface. Existing quality metrics are linked by their original report hash.
    Human validity remains pending until the registered final labels are joined.
    """
    from .prototype_features import DinoExtractor
    ref_path,out=Path(real_reference).resolve(),Path(output_dir).resolve()
    evaluation=Path(evaluation).resolve()
    if evaluation.is_file(): evaluation=evaluation.parent
    base=load_reference(ref_path)
    base_report,_,generated,_=load_evaluation(evaluation)
    if base_report['reference_signature']!=base['signature']:
        raise ValueError('Base evaluation and original real reference disagree')
    if read_json(Path(split_dir)/'split.json')['signature']!=base['split_signature']:
        raise ValueError('Prototype fit split differs from the frozen real split')
    if list(seeds)!=[0,1,2,3,4] or temperature!=.1 or k!=3:
        raise ValueError('This registered AFHQ audit requires K=3, temperature=.1, and all seeds 0–4')
    if base['config']['domain'].lower()!='afhq' or len(base['classes'])!=3:
        raise ValueError('This registered conditional prototype audit is scoped to AFHQ with given K=3')
    parts={'fit':collect_images(load_split(split_dir,'fit'))}
    parts.update({name:collect_images(read_jsonl(ref_path.parent/f'{name}.jsonl'))
                  for name in ('calibration','calibration_pool','audit_internal','audit_external')})
    parts['generated']=collect_images(generated)
    real_ids={r['image_id'] for name,rows in parts.items() if name!='generated' for r in rows}
    real_hashes={r['sha256'] for name,rows in parts.items() if name!='generated' for r in rows}
    if any(r['image_id'] in real_ids or r['sha256'] in real_hashes for r in generated):
        raise ValueError('Generated images overlap the real prototype/reference data')
    extractor=DinoExtractor(repository,weights,device=device,batch_size=batch_size,image_policy=base['image_policy'])
    request={'kind':'conditional_unlabeled_AFHQ_audit','original_reference_sha256':sha256_file(ref_path),
             'original_evaluation_sha256':sha256_file(Path(evaluation)/'report.json'),
             'K':k,'prototype_temperature':temperature,'seeds':list(seeds),'primary_seed':0,'n_init':10,
             'features':extractor.metadata,'fit_receives_labels':False,'code_sha256':code_signature(),
             'part_identities':{name:_without_labels(rows) for name,rows in parts.items()},
             'human_semantic_validity':'pending_registered_final_random_annotation'}
    freeze_request(out/'prototype_protocol.json',request)
    cache=Path(cache_dir).resolve() if cache_dir else out/'cache'
    features,cache_meta={},{}
    for name,records in parts.items():
        print(f'Prototype features {name}: {len(records)}',flush=True)
        values,cache_meta[name]=cached_features(cache,_without_labels(records),extractor.metadata,extractor.extract)
        features[name]=values['embeddings']
    summaries=[]
    for seed in seeds:
        root=out/f'seed{seed}'
        centers=fit_prototypes(features['fit'],k=k,seed=seed)
        save_arrays(root/'centers.npz',centers=centers)
        classes=[f'cluster_{c}' for c in range(k)]
        arrays={name+'_logits':prototype_logits(values,centers,temperature=temperature)
                for name,values in features.items() if name not in ('fit','generated')}
        calibration=posterior_scores(arrays['calibration_logits'])
        thresholds={name:rank_calibrate(calibration[name],base['config'].get('alpha',.05)) for name in SCORE_NAMES}
        audits={name:aggregate_mms(posterior_scores(arrays[name+'_logits']),thresholds,classes)
                for name in ('audit_internal','audit_external')}
        config={**base['config'],'classes':classes,'temperature':1.,'prototype_temperature':temperature,
                'prototype_k':k,'prototype_seed':seed,'evaluator_kind':'unlabeled_prototype',
                'protocol_id':base['protocol_id']+f'-dinov2-clusters-seed{seed}'}
        contract={'version':'prototype-reference-v1','config':config,'checkpoint_sha256':sha256_file(root/'centers.npz'),
                  'split_signature':base['split_signature'],'code_sha256':request['code_sha256']}
        refroot=root/'reference'
        save_arrays(refroot/'reference_arrays.npz',**arrays)
        for name,records in parts.items():
            if name!='generated':
                write_jsonl(refroot/f'{name}.jsonl',_without_labels(records) if name=='fit' else records)
        ref={**contract,'signature':stable_hash(contract),'protocol_id':config['protocol_id'],'classes':classes,
             'evaluator_id':f'dinov2_prototypes_seed{seed}','image_policy':base['image_policy'],
             'calibrations':thresholds,'audits':audits,'arrays_file':'reference_arrays.npz',
             'arrays_sha256':sha256_file(refroot/'reference_arrays.npz'),
             'manifests_sha256':{name:sha256_file(refroot/f'{name}.jsonl') for name in parts if name!='generated'},
             'counts':{name:len(records) for name,records in parts.items() if name!='generated'},
             'source_features':extractor.metadata,'original_reference_sha256':sha256_file(ref_path),
             'centers_file':str(root/'centers.npz'),'semantic_status':'cluster_semantics_and_human_MM_validity_pending'}
        ref['integrity_sha256']=stable_hash(ref);write_json(refroot/'reference.json',ref)
        # Audit labels are consulted only after center fitting/calibration are frozen above.
        cluster_audits={}
        for name in ('audit_internal','audit_external'):
            cluster_audits[name]=_cluster_audit(parts[name],arrays[name+'_logits'],centers,features[name],base['classes'],root/name,seed=seed,image_policy=base['image_policy'])
        logits=prototype_logits(features['generated'],centers,temperature=temperature)
        scores=make_score_rows(generated,logits,thresholds,classes,temperature=1.,evaluator_id=ref['evaluator_id'],protocol_id=ref['protocol_id'])
        for row,record in zip(scores,generated): row['sha256']=record['sha256']
        evalroot=root/'evaluation'
        write_jsonl(evalroot/'input_manifest.jsonl',generated);write_jsonl(evalroot/'scores.jsonl',scores)
        save_arrays(evalroot/'features.npz',logits=logits,embeddings=features['generated'])
        mms=aggregate_mms(posterior_scores(logits),thresholds,classes)
        report={'kind':'prototype_semantic_with_linked_quality','n':len(generated),'mms':mms,
                'distribution':base_report['distribution'],'image_policy':base['image_policy'],
                'protocol_id':ref['protocol_id'],'evaluator_id':ref['evaluator_id'],
                'reference':str(refroot/'reference.json'),'reference_signature':ref['signature'],
                'reference_sha256':sha256_file(refroot/'reference.json'),'calibrations':thresholds,'real_audits':audits,
                'manifest_sha256':sha256_file(evalroot/'input_manifest.jsonl'),'scores_sha256':sha256_file(evalroot/'scores.jsonl'),
                'features_sha256':sha256_file(evalroot/'features.npz'),'quality_reference_manifest_sha256':base['manifests_sha256']['quality_reference'],
                'quality_original_report_sha256':request['original_evaluation_sha256'],
                'semantic_status':'cluster_semantics_and_human_MM_validity_pending',
                'interpretation':'Task-given cluster granularity; no target-domain labels used for features or centers. Linked quality scores use exactly the unchanged generated images. External pretraining is not absent. Human evidence, when added, reuses the registered labeled cohort and is not a new independent test.'}
        write_report(evalroot/'report.json',report)
        summaries.append({'seed':seed,'role':'primary' if seed==0 else 'stability_appendix','MMS':mms['value'],
                          'mean_entropy':mms['mean_entropy'],'calibrations':thresholds,'real_audits':audits,
                          'cluster_audits':cluster_audits,'evaluation':str(evalroot)})
        print(f'Prototype seed {seed} complete: MMS={mms["value"]}',flush=True)
    result={'protocol_sha256':sha256_file(out/'prototype_protocol.json'),'all_seeds':summaries,
            'feature_cache':cache_meta,'status':'numerical_audit_complete_visual_and_human_semantic_validation_pending',
            'original_reference_unchanged':sha256_file(ref_path)==request['original_reference_sha256']}
    write_json(out/'prototype_audit.json',result)
    return result
