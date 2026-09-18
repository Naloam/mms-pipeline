"""Freeze independent evaluator B and run the shared historical 5k cohort."""
import argparse
from pathlib import Path

from mms_eval.analysis import adaptivity_audit
from mms_eval.pipeline import build_reference, evaluate
from mms_eval.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = read_json(args.config)
    reference = root/'reference_B_clean_engineering_v2'
    cache = root/'shared_feature_cache_B_v2'
    build_reference(root/'splits', root/'evaluator_B'/'best.pt', reference,
                    config=config, device=args.device, cache_dir=cache)
    print('REFERENCE_B_COMPLETE', flush=True)
    evaluation = root/'evaluation_B_clean_5000_v2'
    report = evaluate(root/'scale_acceptance_clean_v2'/'input_5000.jsonl', reference/'reference.json',
                      evaluation, device=args.device, cache_dir=cache)
    print({'EVALUATION_B_COMPLETE': report['n'], 'MMS': report['mms']['value']}, flush=True)
    result = adaptivity_audit(reference/'reference.json', evaluation, root/'adaptivity_B_clean_v2')
    write_json(root/'evaluator_B'/'acceptance.json', {
        'reference': str(reference), 'evaluation': str(evaluation),
        'adaptivity': str(root/'adaptivity_B_clean_v2'),
        'n': report['n'], 'main_reference_unchanged': result['main_reference_unchanged'],
        'status': 'engineering_complete_human_MM_validity_pending'})
    print('EVALUATOR_B_ACCEPTANCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
