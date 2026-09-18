import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mms_eval import pipeline
from mms_eval.annotation import annotation_pack, analyze_annotations
from mms_eval.analysis import adaptivity_audit
from mms_eval.images import as_rgb_image, canonicalize_records, collect_images, load_rgb, save_png_atomic, validate_policy
from mms_eval.semantic import posterior_scores, rank_calibrate, apply_threshold, aggregate_mms, SCORE_NAMES
from mms_eval.splits import prepare_real_data, load_split
from mms_eval.utils import read_json, sha256_file, write_json


def test_entropy_extreme_logits_and_uniform():
    scores = posterior_scores(np.array([[1e308,-1e308,0],[0,0,0]]),1e-300)
    assert np.isfinite(scores['entropy']).all()
    assert scores['entropy'][0] == 0
    assert scores['entropy'][1] == pytest.approx(np.log(3))
    assert scores['class_order'][1].tolist() == [0,1,2]


def test_exact_rank_and_strict_ties():
    cal = rank_calibrate(np.arange(19),.05)
    assert cal['k'] == 19 and cal['threshold'] == 18
    assert apply_threshold([18,18.01],cal).tolist() == [False,True]
    assert rank_calibrate([1,2],.05)['threshold'] == 'infinity'
    assert not apply_threshold([1000],rank_calibrate([1,2],.05))[0]


def test_pair_counts_singleton_and_nonfinite():
    scores = posterior_scores(np.zeros((4,3)))
    cal = {k:rank_calibrate(np.zeros(19),.05) for k in SCORE_NAMES}
    result = aggregate_mms(scores,cal,['cat','dog','wild'])
    assert result['candidate_count'] == 4
    assert np.array(result['pair_counts_upper_triangle']).sum() == 4
    one = aggregate_mms(posterior_scores(np.zeros((1,3))),cal,['cat','dog','wild'])
    assert one['value'] is None and one['single_image_candidate']
    with pytest.raises(ValueError):
        posterior_scores([[np.nan,0]])


def test_image_conversion_sizes_ranges_and_alpha(tmp_path):
    gray = np.zeros((7,9),dtype=np.uint8)
    assert as_rgb_image(gray).mode == 'RGB'
    rgba = np.zeros((7,9,4),dtype=np.uint8)
    assert np.array(as_rgb_image(rgba)).min() == 255
    chw = np.ones((3,7,9),dtype=np.float32)
    assert as_rgb_image(chw,layout='CHW',value_range='0_1').size == (9,7)
    with pytest.raises(ValueError): as_rgb_image(chw,layout='CHW')
    with pytest.raises(ValueError): as_rgb_image(np.full((3,7,9),2.),layout='CHW',value_range='0_1')
    im = as_rgb_image(gray,policy={'canonical_size':[12,13]})
    assert im.size == (12,13)
    im.save(tmp_path/'a.png')
    records = collect_images(tmp_path/'a.png')
    canonical = canonicalize_records(records,tmp_path/'cache')
    Image.new('RGB',(12,13),'red').save(canonical[0]['path'])
    with pytest.raises(ValueError,match='cache changed'): canonicalize_records(records,tmp_path/'cache')


def test_manifest_rejects_modified_and_duplicate_id(tmp_path):
    path = tmp_path/'a.png'; Image.new('RGB',(3,4)).save(path)
    row = collect_images(path)[0]
    with pytest.raises(ValueError,match='Duplicate'): collect_images([row,row])
    row['sha256'] = 'wrong'
    with pytest.raises(ValueError,match='hash mismatch'): collect_images([row])


def test_registered_study_excludes_prior_labels_and_freezes_quotas(ready):
    from mms_eval.study import register_study, study_pack
    from mms_eval.utils import read_jsonl
    root, ref, images, *_ = ready
    pipeline.evaluate(images, ref, root/'study_eval')
    annotation_pack(root/'study_eval', root/'development', count=2)
    spec = {'role': 'final_random', 'seed': 9, 'annotation_guide_version': 'test-v1',
            'annotation_guide': 'Toy fixture only. Classes cat/dog/wild; classify same-subject incompatible fusion.',
            'analysis_plan': 'Source-specific detection tables with paired bootstrap',
            'exclusion_packs': [str(root/'development')],
            'sources': [{'source_id': 'modelA', 'evaluation': str(root/'study_eval'), 'count': 6,
                         'eligibility_declaration':'Toy test: prior two-image pack is fully excluded; remaining originals unused.'}]}
    registration = register_study(spec, root/'registered')
    assert registration['sources'][0]['n_eligible'] == 6
    pack = study_pack(root/'registered', root/'final_pack')
    assert pack['role'] == 'final_random' and pack['registration_sha256']
    old = {r['image_id'] for r in read_jsonl(root/'development'/'private_key.jsonl')}
    new = {r['image_id'] for r in read_jsonl(root/'final_pack'/'private_key.jsonl')}
    assert not old & new
    spec['sources'][0]['count'] = 7
    with pytest.raises(ValueError, match='Insufficient'):
        register_study(spec, root/'invalid_registration')
    with pytest.raises(ValueError, match='registration'):
        annotation_pack(root/'study_eval', root/'invalid_pack', count=2, role='final_random')


def test_comparison_rejects_unequal_budgets_and_non_nested_scale(ready):
    from mms_eval.comparison import compare_evaluations, evaluate_batch
    root, ref, images, *_ = ready
    pipeline.evaluate(images, ref, root/'large')
    pipeline.evaluate(images[:4], ref, root/'small')
    entries = [{'name':'4', 'path':str(root/'small')}, {'name':'8', 'path':str(root/'large')}]
    # Use the same KID subset budget even on these tiny synthetic fixtures.
    with pytest.raises(ValueError, match='same generated N'):
        compare_evaluations(entries, root/'bad', plots=False)
    assert len(compare_evaluations(entries, root/'scale', mode='scale', plots=False)['rows']) == 2
    result = evaluate_batch({'reference': str(ref), 'common_n': 4, 'seed': 3,
                             'sources': [{'source_id': 'generator', 'input': images}]}, root/'batch')
    assert result['rows'][0]['n'] == 4


@pytest.fixture
def ready(tmp_path,monkeypatch):
    data = tmp_path/'data'
    rng = np.random.RandomState(7)
    for directory,n in [('train',40),('val',8)]:
        for category in ['cat','dog','wild']:
            for i in range(n):
                save_png_atomic(Image.fromarray(rng.randint(0,256,(7,9,3),dtype=np.uint8)),data/directory/category/f'{i}.png')
    cfg = {'domain':'test','protocol_id':'test_v1','category_mode':'exclusive','classes':['cat','dog','wild'],
           'validation_n':18,'calibration_pool_n':24,'internal_audit_n':18,'calibration_n':20,
           'image_policy':{},'temperature':1.,'alpha':.05,'metrics':{'kid_subset_size':4,'kid_subsets':3,'isc_splits':2,'pr_k':3},
           'semantic_status':'pending'}
    splits = tmp_path/'splits'
    prepare_real_data(data,splits,config=cfg)
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'test fixture')
    meta = {'classes':cfg['classes'],'checkpoint_sha256':sha256_file(checkpoint),'image_policy':{},
            'preprocessing':{},'software':{},'seen_images':load_split(splits,'fit')+load_split(splits,'validation')}
    monkeypatch.setattr(pipeline,'describe_checkpoint',lambda p:meta)
    def predict(paths,checkpoint,**kwargs):
        means = np.array([np.array(load_rgb(p)).mean((0,1)) for p in paths])/50
        return {'logits':means,'metadata':meta}
    monkeypatch.setattr(pipeline,'predict_paths',predict)
    class Extractor:
        def __init__(self,*args):
            self.metadata = {'backend':'test','weights_sha256':'testweights','preprocessing':{}}
        def extract_paths(self,paths):
            means = np.array([np.array(load_rgb(p)).mean((0,1)) for p in paths])/255
            return {'features':means,'logits':means,'metadata':self.metadata}
    monkeypatch.setattr(pipeline,'FidelityExtractor',Extractor)
    reference = tmp_path/'ref'/'reference.json'
    pipeline.build_reference(splits,checkpoint,reference.parent,config=cfg)
    # External images here are synthetic test inputs, not actual generated evidence.
    images = [r['path'] for r in load_split(splits,'audit_external')[:8]]
    return tmp_path,reference,images,splits,cfg,checkpoint


def test_full_reference_collection_single_and_cache(ready):
    root,ref,images,splits,cfg,checkpoint = ready
    result = pipeline.evaluate(images,ref,root/'run')
    assert result['n'] == 8
    assert all(result['distribution'][k]['status']=='ok' for k in ['fid','kid','is','precision_recall'])
    repeated = pipeline.evaluate(images,ref,root/'run')
    assert repeated['quality_cache_key'] == result['quality_cache_key']
    one = pipeline.evaluate_image(np.zeros((11,8),dtype=np.uint8),ref,root/'one')
    assert one['mms']['value'] is None
    assert all(one['distribution'][k]['status']=='unavailable' for k in ['fid','kid','is','precision_recall'])
    assert (root/'one'/'features.npz').exists()
    with pytest.raises(ValueError,match='different evaluation'): pipeline.evaluate(images[:4],ref,root/'run')
    frozen = pipeline.build_reference(splits,checkpoint,ref.parent,config=cfg)
    assert frozen['signature'] == result['reference_signature']


def test_reference_tampering_rejected(ready):
    root,ref,*_ = ready
    data = read_json(ref); data['calibrations']['entropy']['threshold']=999
    write_json(ref,data)
    with pytest.raises(ValueError,match='modified'): pipeline.load_reference(ref)


def test_evaluation_expansion_uses_shared_cache_and_keeps_scores_aligned(ready):
    root, ref, images, *_ = ready
    pipeline.evaluate(images[:4], ref, root/'small', cache_dir=root/'shared', cache_chunk_size=2)
    large = pipeline.evaluate(images[::-1], ref, root/'large', cache_dir=root/'shared', cache_chunk_size=2)
    for kind in ('semantic', 'quality'):
        assert large['feature_cache'][kind]['reused_rows'] == 4
        assert large['feature_cache'][kind]['computed_rows'] == 4
    fresh = pipeline.evaluate(images[::-1], ref, root/'fresh', cache_chunk_size=2)
    assert fresh['mms'] == large['mms']
    assert fresh['distribution'] == large['distribution']
    assert (root/'fresh'/'scores.jsonl').read_bytes() == (root/'large'/'scores.jsonl').read_bytes()


def test_split_tampering_and_leakage_rejected(ready):
    root,ref,images,splits,cfg,checkpoint = ready
    with (splits/'calibration.jsonl').open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='modified'): load_split(splits,'calibration')
    with pytest.raises(ValueError,match='overlaps'): pipeline.reject_seen([{'image_id':'b','sha256':'same'}],[{'image_id':'a','sha256':'same'}],'audit')


def test_human_pack_and_adaptivity(ready):
    root,ref,images,*_ = ready
    pipeline.evaluate(images,ref,root/'run')
    audit = adaptivity_audit(ref,root/'run',root/'sensitivity',sizes=(10,20),alphas=(.01,.05,.1),repetitions=2)
    assert len(audit['summaries'])==6
    pack = annotation_pack(root/'run',root/'pack',count=6)
    assert pack['human_validity']=='pending'
    with (root/'labels.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['blind_id','annotator_id','label'])
        writer.writeheader()
        for i in range(6):
            for r in ('A','B'):
                writer.writerow({'blind_id':f'{i+1:04d}','annotator_id':r,'label':'clear_single' if i%2 else 'mixed_semantics'})
    result = analyze_annotations(root/'pack',root/'labels.csv',root/'human_result')
    assert result['agreement'][0]['cohen_kappa']==1
    assert result['raters']['A']['missing']==0


def test_complete_adaptivity_shares_subsets_and_does_not_repeat_full_pool(ready):
    from mms_eval.utils import read_jsonl
    root, ref, images, *_ = ready
    pipeline.evaluate(images, ref, root/'run')
    output = adaptivity_audit(ref, root/'run', root/'adaptive', sizes=(10,24), repetitions=3)
    rows = read_jsonl(root/'adaptive'/'adaptivity_rows.jsonl')
    full = [r for r in rows if r['n_reference'] == 24]
    assert len(full) == 4*len(SCORE_NAMES)
    assert {r['repetition'] for r in full} == {0}
    partial = [r for r in rows if r['n_reference'] == 10 and r['repetition'] == 0]
    assert len({r['subset_id'] for r in partial}) == 1
    assert {r['score'] for r in partial} == set(SCORE_NAMES)
    temperature = read_jsonl(root/'adaptive'/'temperature_rows.jsonl')
    assert len(temperature) == 4*2*len(SCORE_NAMES)
    for name in SCORE_NAMES:
        base = [r for r in temperature if r['temperature'] == 1 and r['score'] == name]
        assert base[0]['threshold'] == base[1]['threshold']
        assert base[0]['generated_flag_rate'] == base[1]['generated_flag_rate']
    assert output['main_reference_unchanged']


def test_quality_only_does_not_invent_semantics(ready,monkeypatch):
    from mms_eval import quality_only
    root,ref,images,splits,cfg,_ = ready
    monkeypatch.setattr(quality_only,'FidelityExtractor',pipeline.FidelityExtractor)
    report = quality_only.evaluate_quality(images,load_split(splits,'quality_reference'),root/'quality',config=cfg)
    assert report['mms']['value'] is None
    assert report['distribution']['fid']['status']=='ok'


def test_conditional_prototype_end_to_end_keeps_all_seeds_and_original_reference(ready,monkeypatch):
    from mms_eval import prototype_features
    from mms_eval.prototypes import run_prototype_audit
    from mms_eval.artifacts import load_evaluation
    from mms_eval.utils import read_jsonl
    root,_,_,_,cfg,checkpoint=ready
    cfg={**cfg,'domain':'afhq'}
    split=root/'proto_splits';prepare_real_data(root/'data',split,config=cfg)
    ref=root/'proto_base/reference.json'
    pipeline.build_reference(split,checkpoint,ref.parent,config=cfg)
    before=sha256_file(ref)
    generated=[]
    for i in range(8):
        path=root/f'new_gen_{i}.png'
        save_png_atomic(Image.fromarray(np.random.RandomState(80+i).randint(0,256,(8,8,3),dtype=np.uint8)),path)
        generated.append(str(path))
    base=pipeline.evaluate(generated,ref,root/'proto_eval_base')
    class ToyExtractor:
        def __init__(self,*args,**kwargs): self.metadata={'backend':'toy-test-only'}
        def extract(self,records):
            assert not any('label' in r or 'class_name' in r for r in records)
            return {'embeddings':np.array([np.random.RandomState(int(r['sha256'][:8],16)).normal(size=12) for r in records]),
                    'metadata':self.metadata}
    monkeypatch.setattr(prototype_features,'DinoExtractor',ToyExtractor)
    result=run_prototype_audit(ref,split,root/'proto_eval_base',root/'prototypes',repository='fixture',weights='fixture')
    assert [r['seed'] for r in result['all_seeds']]==[0,1,2,3,4]
    assert sha256_file(ref)==before and result['original_reference_unchanged']
    for item in result['all_seeds']:
        report,rows,records,_=load_evaluation(item['evaluation'],features=True)
        assert len(rows)==8 and report['distribution']==base['distribution']
        frozen=pipeline.load_reference(report['reference'])
        assert frozen['classes']==['cluster_0','cluster_1','cluster_2']
        assert not any('label' in r for r in read_jsonl(Path(report['reference']).parent/'fit.jsonl'))
        assert all('probabilities' in r for r in rows)
    with pytest.raises(ValueError,match='all seeds'):
        run_prototype_audit(ref,split,root/'proto_eval_base',root/'selected_seed',repository='fixture',weights='fixture',seeds=(0,))


def test_quality_only_batch_uses_common_n_and_never_invents_MMS(ready,monkeypatch):
    from mms_eval import quality_only
    from mms_eval.comparison import evaluate_batch
    root,_,images,split,cfg,_=ready
    monkeypatch.setattr(quality_only,'FidelityExtractor',pipeline.FidelityExtractor)
    spec={'mode':'quality_only','common_n':4,'seed':7,'real':load_split(split,'quality_reference'),
          'config':cfg,'sources':[{'source_id':'one','input':images},{'source_id':'two','input':images}]}
    result=evaluate_batch(spec,root/'quality_batch')
    assert [r['n'] for r in result['rows']]==[4,4]
    assert all(r['MMS_candidate_fraction'] is None for r in result['rows'])


def test_quality_request_frozen_before_interrupted_extraction(ready,monkeypatch):
    from mms_eval import quality_only
    root,_,images,split,cfg,_=ready
    class BrokenExtractor:
        def __init__(self,*args): self.metadata={'backend':'toy'}
        def extract_paths(self,paths): raise RuntimeError('simulated interruption')
    monkeypatch.setattr(quality_only,'FidelityExtractor',BrokenExtractor)
    real=load_split(split,'quality_reference')
    with pytest.raises(RuntimeError,match='simulated interruption'):
        quality_only.evaluate_quality(images,real,root/'interrupted_quality',config=cfg)
    assert (root/'interrupted_quality/quality_request.json').exists()
    with pytest.raises(ValueError,match='different evaluation/reference request'):
        quality_only.evaluate_quality(images[:4],real,root/'interrupted_quality',config=cfg)


def test_domain_reference_checks_actual_checkpoint_initialization(ready,monkeypatch):
    root,_,_,split,cfg,checkpoint=ready
    actual={**pipeline.describe_checkpoint(checkpoint),'config':{'weights':'IMAGENET1K_V2'}}
    monkeypatch.setattr(pipeline,'describe_checkpoint',lambda _:actual)
    cfg={**cfg,'domain_contract_version':1,'domain':'MNIST','dataset_version':'toy-v1',
         'classes':[str(i) for i in range(10)],'digit_polarity':'light_on_dark','semantic_definition':'toy',
         'pretraining_audit':{'status':'from_scratch','evidence':'declared before training'}}
    with pytest.raises(ValueError,match='from_scratch'):
        pipeline.build_reference(split,checkpoint,root/'bad_checkpoint_reference',config=cfg)


def test_detection_gallery_keeps_threshold_posterior_and_independent_defects(ready):
    from mms_eval.human_labels import finalize_annotations
    from mms_eval.diagnostics import detection_diagnostics
    root,ref,images,*_=ready
    pipeline.evaluate(images,ref,root/'gallery_eval')
    annotation_pack(root/'gallery_eval',root/'gallery_pack',count=8)
    raw=root/'gallery_raw.csv'
    with raw.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['blind_id','annotator_id','label','other_defects','human_target_class'])
        w.writeheader()
        for i in range(8):
            for rater in ('A','B'):
                w.writerow({'blind_id':f'{i+1:04d}','annotator_id':rater,'label':'yes' if i<3 else 'unknown' if i==3 else 'no',
                            'other_defects':'blur','human_target_class':'cat' if i%2 else 'dog'})
    finalize_annotations(root/'gallery_pack',raw,root/'gallery_final')
    result=detection_diagnostics(root/'gallery_eval',root/'gallery_final/final_labels.csv',root/'galleries',max_per_cell=4)
    with (root/'galleries/gallery_key.csv').open() as stream: keys=list(csv.DictReader(stream))
    assert keys and all(r['other_defects']=='blur' and r['probabilities'] and r['threshold'] for r in keys)
    assert result['role']=='analyst_diagnostic_only'
    assert result['review_budget_status']=='unavailable_requires_registered_final_random_labels'
    assert not (root/'galleries/review_budget_0.pdf').exists()


def test_registered_paired_sources_sample_same_ids_and_enable_final_review_curves(ready):
    from mms_eval.study import register_study,study_pack
    from mms_eval.human_labels import finalize_annotations
    from mms_eval.diagnostics import detection_diagnostics
    from mms_eval.utils import read_jsonl
    root,ref,images,*_=ready
    sources=[]
    for source in ('A','B'):
        records=[]
        for i,path in enumerate(images):
            image=load_rgb(path)
            if source=='B': image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target=root/f'{source}_pair_{i}.png';save_png_atomic(image,target)
            records.append({'image_id':f'{source}-{i}','path':str(target),'pair_id':f'shared-{i}','setting_id':source})
        pipeline.evaluate(records,ref,root/f'paired_{source}')
        sources.append({'source_id':source,'evaluation':str(root/f'paired_{source}'),'count':4,
                        'eligibility_declaration':'Synthetic fixture: all listed originals unused.'})
    spec={'role':'final_random','seed':9,'annotation_guide_version':'toy-v1','annotation_guide':'Toy annotation definition',
          'analysis_plan':'Paired source protocol test','sources':sources,'exclusion_packs':[]}
    with pytest.raises(ValueError,match='paired_source_groups'):
        register_study(spec,root/'undeclared_pairs')
    spec['paired_source_groups']=[['A','B']]
    register_study(spec,root/'paired_registration');study_pack(root/'paired_registration',root/'paired_pack')
    key=read_jsonl(root/'paired_pack/private_key.jsonl')
    assert {r['pair_id'] for r in key if r['setting_id']=='A'}=={r['pair_id'] for r in key if r['setting_id']=='B'}
    raw=root/'paired_raw.csv'
    with raw.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['blind_id','annotator_id','label','annotation_guide_version']);w.writeheader()
        for i,row in enumerate(key):
            for rater in ('one','two'):
                w.writerow({'blind_id':row['blind_id'],'annotator_id':rater,'label':'yes' if i%2 else 'no','annotation_guide_version':'toy-v1'})
    finalize_annotations(root/'paired_pack',raw,root/'paired_final')
    result=detection_diagnostics(root/'paired_A',root/'paired_final/final_labels.csv',root/'paired_gallery')
    assert result['review_budget_status']=='registered_final_random'
    assert (root/'paired_gallery/review_budget_0.pdf').exists()


def test_final_registration_excludes_transformed_parents_and_rejects_duplicate_parents(ready):
    from mms_eval.study import register_study
    from mms_eval.utils import read_jsonl
    root,ref,images,*_=ready
    pipeline.evaluate(images,ref,root/'parent_base')
    annotation_pack(root/'parent_base',root/'parent_seen',count=1)
    seen=read_jsonl(root/'parent_seen/private_key.jsonl')[0]
    records=[]
    for i,path in enumerate(images):
        target=root/f'parent_variant_{i}.png'
        save_png_atomic(load_rgb(path).transpose(Image.Transpose.FLIP_TOP_BOTTOM),target)
        records.append({'image_id':f'variant-{i}','path':str(target),'parent_id':seen['image_id'] if i==0 else f'parent-{i}'})
    pipeline.evaluate(records,ref,root/'parent_variants')
    spec={'role':'final_random','seed':2,'annotation_guide_version':'v1','annotation_guide':'Toy guide','analysis_plan':'Toy test',
          'exclusion_packs':[str(root/'parent_seen')],
          'sources':[{'source_id':'variant','evaluation':str(root/'parent_variants'),'count':7,'eligibility_declaration':'Exclude seen parents'}]}
    meta=register_study(spec,root/'parent_registration')
    assert meta['sources'][0]['n_eligible']==7
    assert 'variant-0' not in {r['image_id'] for r in read_jsonl(root/'parent_registration/selected_private.jsonl')}
    records[2]['parent_id']=records[1]['parent_id']
    pipeline.evaluate(records,ref,root/'parent_duplicate')
    spec['sources'][0]['evaluation']=str(root/'parent_duplicate')
    with pytest.raises(ValueError,match='repeated parent'):
        register_study(spec,root/'duplicate_registration')


def test_human_finalization_and_detection_exports_join_frozen_ids(ready):
    from mms_eval.human_labels import finalize_annotations
    from mms_eval.analysis_reporting import analyze_detection
    root, ref, images, *_ = ready
    pipeline.evaluate(images, ref, root/'run')
    annotation_pack(root/'run', root/'pack', count=8)
    with (root/'raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['blind_id','annotator_id','label'])
        writer.writeheader()
        for i in range(8):
            for rater in ('A','B'):
                writer.writerow({'blind_id': f'{i+1:04d}', 'annotator_id': rater,
                                 'label': 'unknown' if i == 7 else 'yes' if i%2 else 'no'})
    sealed = finalize_annotations(root/'pack', root/'raw.csv', root/'final')
    assert sealed['status'] == 'complete'
    result = analyze_detection(root/'run', root/'final'/'final_labels.csv', root/'detection', repetitions=20)
    assert len(result['panels']) == 1
    assert result['panels'][0]['n_unknown'] == 1
    assert (root/'detection'/'table_detection_main.csv').is_file()
    assert (root/'detection'/'table_detection_differences.csv').is_file()
    with (root/'final'/'final_labels.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='Sealed final labels changed'):
        analyze_detection(root/'run', root/'final'/'final_labels.csv', root/'tampered')
