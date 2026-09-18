"""Image-set metrics for domains without an established MMS semantic interface."""
from pathlib import Path

from .distribution import FidelityExtractor, compute_distribution_metrics
from .images import canonicalize_records, collect_images, validate_policy
from .pipeline import code_signature, freeze_request, make_quality_extractor, quality_features, save_arrays, write_report
from .utils import read_json, sha256_file, stable_hash, write_jsonl


def evaluate_quality(source, real_source, output_dir, *, config=None, device='cpu', batch_size=32,
                     cache_dir=None, cache_chunk_size=256):
    config = config or {}
    if config.get('domain_contract_version'):
        from .domains import validate_domain_contract
        validate_domain_contract(config,mode='quality_only')
    out = Path(output_dir).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else out/'cache'
    real, generated = collect_images(real_source), collect_images(source)
    policy = validate_policy(config.get('image_policy'))
    contract = {'config':config,'image_policy':policy,'code_sha256':code_signature(),
                'real':[{k:r[k] for k in ('image_id','sha256')} for r in real],
                'generated':[{k:r[k] for k in ('image_id','sha256')} for r in generated]}
    signature = stable_hash(contract)
    freeze_request(out/'quality_request.json',{'signature':signature,'contract':contract,'real':real,'generated':generated})
    if (out/'report.json').exists() and read_json(out/'report.json').get('signature') != signature:
        raise ValueError('Output contains a different quality evaluation; use a new directory')
    metrics = config.get('metrics', {})
    extractor = (FidelityExtractor(device, batch_size)
                 if metrics.get('feature_backend', 'torch-fidelity') == 'torch-fidelity' and metrics.get('pr_feature_backend', 'inception') == 'inception'
                 else make_quality_extractor(metrics, device, batch_size))
    r,rm = quality_features(canonicalize_records(real,cache/'rgb',policy),extractor,cache,chunk_size=cache_chunk_size)
    g,gm = quality_features(canonicalize_records(generated,cache/'rgb',policy),extractor,cache,chunk_size=cache_chunk_size)
    distribution = compute_distribution_metrics(r['features'],g['features'],g['logits'],
                    real_pr_features=r.get('pr_features'), generated_pr_features=g.get('pr_features'),
                    config={**config.get('metrics',{}),'feature_metadata':gm['metadata']})
    write_jsonl(out/'real_manifest.jsonl',real); write_jsonl(out/'input_manifest.jsonl',generated)
    save_arrays(out/'features.npz',**g)
    report = {'kind':'quality_only','signature':signature,'protocol_id':config.get('protocol_id','quality_only_v1'),
              'n':len(generated),'image_policy':policy,'code_sha256':code_signature(),
              'real_manifest_sha256':sha256_file(out/'real_manifest.jsonl'),
              'manifest_sha256':sha256_file(out/'input_manifest.jsonl'),
              'features_sha256':sha256_file(out/'features.npz'),
              'real_cache_key':rm['key'],'quality_cache_key':gm['key'],'distribution':distribution,
              'mms':{'status':'unavailable','value':None,
                     'reason':'No frozen, validated mutually exclusive semantic evaluator and calibration were supplied'},
              'semantic_status':'MMS_unavailable_semantic_interface_required'}
    report['feature_cache'] = {'real': rm, 'generated': gm}
    write_report(out/'report.json',report)
    return report
