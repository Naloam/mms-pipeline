"""Command line interface; every scientific choice is a saved JSON artifact."""
import argparse
import json
from pathlib import Path

from .utils import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mms-eval', description='Reusable MMS and image distribution evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Freeze real class-folder splits and remove duplicate sources')
    p.add_argument('--data', required=True); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('train', help='Fit an evaluator using real fit/validation splits only')
    p.add_argument('--splits', required=True); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('reference', help='Freeze calibration and quality references, audit held-out real images')
    p.add_argument('--splits', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cache-dir'); p.add_argument('--cache-chunk-size', type=int, default=256)
    p = sub.add_parser('evaluate', help='Evaluate one image, folder or JSONL/CSV manifest')
    p.add_argument('--input', required=True); p.add_argument('--reference', required=True); p.add_argument('--out', required=True)
    p.add_argument('--checkpoint'); p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cache-dir'); p.add_argument('--cache-chunk-size', type=int, default=256)
    p = sub.add_parser('quality', help='FID/KID/IS/PR without an MMS semantic evaluator')
    p.add_argument('--input', required=True); p.add_argument('--real', required=True); p.add_argument('--out', required=True)
    p.add_argument('--config'); p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cache-dir'); p.add_argument('--cache-chunk-size', type=int, default=256)
    p = sub.add_parser('sample-afhq500k', help='Sample the registered AFHQ DDPM checkpoint')
    p.add_argument('--checkpoint', required=True); p.add_argument('--repository', required=True); p.add_argument('--out', required=True)
    p.add_argument('--count', required=True, type=int); p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--sampler', choices=['ddpm','ddim'], default='ddim'); p.add_argument('--steps', type=int, default=50)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--role', default='engineering')
    p = sub.add_parser('adaptivity', help='Fixed real-reference calibration sensitivity audit')
    p.add_argument('--reference', required=True); p.add_argument('--evaluation', required=True); p.add_argument('--out', required=True)
    p.add_argument('--labels'); p.add_argument('--sizes', nargs='+', type=int)
    p.add_argument('--alphas', nargs='+', type=float, default=[.01,.025,.05,.1])
    p.add_argument('--temperatures', nargs='+', type=float, default=[.5,1.,2.,4.])
    p.add_argument('--repetitions', type=int, default=20); p.add_argument('--seed', type=int, default=2026091106)
    p = sub.add_parser('annotation-pack', help='Create a blind uniform human annotation sample')
    p.add_argument('--evaluation', required=True); p.add_argument('--out', required=True); p.add_argument('--count', type=int, default=200)
    p.add_argument('--seed', type=int, default=2026091105)
    p.add_argument('--role', choices=['development','final_random','diagnostic'], default='development')
    p = sub.add_parser('annotation-analyze', help='Compare independent human labels against candidate flags')
    p.add_argument('--pack', required=True); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('annotation-finalize', help='Seal independent raw labels and record adjudication')
    p.add_argument('--pack', required=True); p.add_argument('--labels', required=True, nargs='+'); p.add_argument('--out', required=True)
    p.add_argument('--adjudications')
    p = sub.add_parser('detection-analyze', help='Human detection tables, paired differences, and unknown sensitivity')
    p.add_argument('--evaluation', required=True, nargs='+'); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p.add_argument('--bootstrap', type=int, default=2000); p.add_argument('--seed', type=int, default=2026091108)
    p.add_argument('--group-by', default='auto'); p.add_argument('--stratum-by', default='setting_id')
    p = sub.add_parser('controls-mix', help='Build paired designed-prevalence cohorts from sealed human labels')
    p.add_argument('--input', required=True); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=100); p.add_argument('--repetitions', type=int, default=20)
    p.add_argument('--proportions', nargs='+', type=float, default=[0,.05,.1,.2,.4,.6])
    p.add_argument('--seed', type=int, default=2026091201)
    p = sub.add_parser('controls-degrade', help='Build paired blur/noise/JPEG controls for independent reannotation')
    p.add_argument('--input', required=True); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p.add_argument('--base-count', type=int, default=30); p.add_argument('--seed', type=int, default=2026091202)
    p.add_argument('--config', help='Use the same canonical image policy as the frozen reference')
    p = sub.add_parser('domain-check', help='Validate semantic scope and pretraining provenance before freezing a domain')
    p.add_argument('--config', required=True); p.add_argument('--evaluator-config')
    p.add_argument('--mode', choices=['semantic','quality_only'], default='semantic')
    p = sub.add_parser('import-arrays', help='Export a declared .npy batch to a provenance-preserving image manifest')
    p.add_argument('--images', required=True); p.add_argument('--out', required=True); p.add_argument('--dataset-id', required=True)
    p.add_argument('--layout', required=True, choices=['NHW','NHWC','NCHW'])
    p.add_argument('--value-range', required=True, choices=['uint8','0_1','-1_1'])
    p.add_argument('--labels'); p.add_argument('--classes', nargs='+'); p.add_argument('--source-directory')
    p.add_argument('--setting-id'); p.add_argument('--config')
    p = sub.add_parser('register-study', help='Freeze per-source annotation quotas, guide, and prior-pack exclusions')
    p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('study-pack', help='Render a previously frozen multi-source blind sample')
    p.add_argument('--registration', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('evaluate-batch', help='Evaluate generation sources with frozen common sample budgets')
    p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32); p.add_argument('--cache-dir')
    p = sub.add_parser('controls-summary', help='Descriptive overall and human-class response with unique-parent counts')
    p.add_argument('--evaluation',nargs='+',required=True);p.add_argument('--out',required=True)
    p = sub.add_parser('compare', help='Validate matching protocols and export model, A/B, or nested-scale comparisons')
    p.add_argument('--evaluation', nargs='+', required=True); p.add_argument('--names', nargs='+')
    p.add_argument('--out', required=True); p.add_argument('--mode', choices=['models','evaluators','scale'], default='models')
    p = sub.add_parser('detection-diagnostics', help='Human-grounded review curves and analyst-only TP/FP/FN/TN galleries')
    p.add_argument('--evaluation', required=True); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p.add_argument('--max-per-cell', type=int, default=8); p.add_argument('--seed', type=int, default=2026091204)
    p = sub.add_parser('controls-curated', help='Register OOD/multiple-subject/candidate controls for blind reannotation')
    p.add_argument('--spec', required=True); p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=2026091203)
    p = sub.add_parser('controls-analyze', help='Paired degradation response, reconfirmed-negative FPR, and human transitions')
    p.add_argument('--evaluation', required=True); p.add_argument('--labels', required=True); p.add_argument('--out', required=True)
    p.add_argument('--bootstrap', type=int, default=2000); p.add_argument('--seed', type=int, default=2026091205)
    p = sub.add_parser('classifier-audit', help='Untouched-real classification loss, confusion, reliability, and candidate strata')
    p.add_argument('--reference', required=True); p.add_argument('--out', required=True); p.add_argument('--bins', type=int, default=15)
    p = sub.add_parser('controls-composition', help='Fixed-N human-confirmed class composition controls')
    p.add_argument('--input', required=True); p.add_argument('--labels', required=True); p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=2026091206)
    p = sub.add_parser('controls-duplicates', help='Fixed-N duplicate replacement with original parent provenance')
    p.add_argument('--input', required=True); p.add_argument('--out', required=True); p.add_argument('--count', type=int, default=1000)
    p.add_argument('--rates', nargs='+', type=float, default=[0,.25,.5]); p.add_argument('--seed', type=int, default=2026091207)
    p = sub.add_parser('tail-register', help='Freeze a development-only mean-entropy-matched setting pair')
    p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('tail-analyze', help='Test the same frozen pair on independent final images')
    p.add_argument('--registration', required=True); p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p.add_argument('--labels'); p.add_argument('--bootstrap', type=int, default=2000); p.add_argument('--seed', type=int, default=2026091208)
    p = sub.add_parser('prototype-audit', help='Registered AFHQ DINOv2 cluster-posterior audit with all seeds 0–4')
    p.add_argument('--reference', required=True); p.add_argument('--splits', required=True); p.add_argument('--evaluation', required=True)
    p.add_argument('--repository', required=True); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32); p.add_argument('--cache-dir')
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if command == 'prepare':
        from .splits import prepare_real_data
        result = prepare_real_data(args['data'], args['out'], config=read_json(args['config']))
    elif command == 'train':
        from .splits import load_split
        from .evaluator import train_evaluator
        result = train_evaluator(load_split(args['splits'],'fit'), load_split(args['splits'],'validation'),
                                 args['out'], config=read_json(args['config']))
        result.pop('metadata', None)
    elif command == 'reference':
        from .pipeline import build_reference
        result = build_reference(args['splits'], args['checkpoint'], args['out'], config=read_json(args['config']),
                                 device=args['device'], batch_size=args['batch_size'],
                                 cache_dir=args['cache_dir'], cache_chunk_size=args['cache_chunk_size'])
        result = {k: result[k] for k in ('signature','counts','calibrations','audits')}
    elif command == 'evaluate':
        from .pipeline import evaluate
        result = evaluate(args['input'], args['reference'], args['out'], checkpoint=args['checkpoint'],
                          device=args['device'], batch_size=args['batch_size'],
                          cache_dir=args['cache_dir'], cache_chunk_size=args['cache_chunk_size'])
        result = {'n':result['n'], 'MMS':result['mms']['value'], 'report':str(Path(args['out']).resolve()/'report.html')}
    elif command == 'quality':
        from .quality_only import evaluate_quality
        result = evaluate_quality(args['input'], args['real'], args['out'],
                                  config=read_json(args['config']) if args['config'] else None,
                                  device=args['device'], batch_size=args['batch_size'],
                                  cache_dir=args['cache_dir'], cache_chunk_size=args['cache_chunk_size'])
        result = {'n':result['n'], 'report':str(Path(args['out']).resolve()/'report.html')}
    elif command == 'sample-afhq500k':
        from .sampling import sample_guided
        args['output_dir'] = args.pop('out')
        result = sample_guided(**args)
    elif command == 'adaptivity':
        from .analysis import adaptivity_audit
        result = adaptivity_audit(args['reference'],args['evaluation'],args['out'], sizes=args['sizes'],
                                 alphas=args['alphas'], temperatures=args['temperatures'], repetitions=args['repetitions'],
                                 seed=args['seed'], final_labels=args['labels'])
        result = {'evaluator_id': result['evaluator_id'], 'sizes': result['sizes'], 'temperatures': result['temperatures'],
                  'main_reference_unchanged': result['main_reference_unchanged'], 'out': str(Path(args['out']).resolve())}
    elif command == 'annotation-pack':
        from .annotation import annotation_pack
        result = annotation_pack(args['evaluation'],args['out'],count=args['count'],seed=args['seed'],role=args['role'])
    elif command == 'annotation-analyze':
        from .annotation import analyze_annotations
        result = analyze_annotations(args['pack'], args['labels'],args['out'])
    elif command == 'annotation-finalize':
        from .human_labels import finalize_annotations
        result = finalize_annotations(args['pack'], args['labels'], args['out'], adjudications=args['adjudications'])
    elif command == 'detection-analyze':
        from .analysis_reporting import analyze_detection
        result = analyze_detection(args['evaluation'], args['labels'], args['out'], repetitions=args['bootstrap'],
                                   seed=args['seed'], group_by=args['group_by'], stratum_by=args['stratum_by'])
        result = {'panels': len(result['panels']), 'out': str(Path(args['out']).resolve()), 'role': result['analysis_role']}
    elif command == 'controls-mix':
        from .controls import build_mixture_controls
        result = build_mixture_controls(args['input'], args['labels'], args['out'], n=args['count'],
                    proportions=args['proportions'], repetitions=args['repetitions'], seed=args['seed'])
    elif command == 'controls-degrade':
        from .controls import build_degradation_controls
        config = read_json(args['config']) if args['config'] else {}
        result = build_degradation_controls(args['input'], args['labels'], args['out'], base_n=args['base_count'],
                    seed=args['seed'], image_policy=config.get('image_policy'))
    elif command == 'domain-check':
        from .domains import validate_domain_contract
        result = validate_domain_contract(read_json(args['config']), mode=args['mode'],
                    evaluator_config=read_json(args['evaluator_config']) if args['evaluator_config'] else None)
    elif command == 'import-arrays':
        from .domains import import_arrays
        config = read_json(args['config']) if args['config'] else {}
        result = import_arrays(args['images'], args['out'], layout=args['layout'], value_range=args['value_range'],
                    dataset_id=args['dataset_id'], labels=args['labels'], classes=args['classes'],
                    source_directory=args['source_directory'], setting_id=args['setting_id'], image_policy=config.get('image_policy'))
    elif command == 'register-study':
        from .study import register_study
        result = register_study(read_json(args['spec']), args['out'])
    elif command == 'study-pack':
        from .study import study_pack
        result = study_pack(args['registration'], args['out'])
    elif command == 'evaluate-batch':
        from .comparison import evaluate_batch
        result = evaluate_batch(read_json(args['spec']), args['out'], device=args['device'],
                    batch_size=args['batch_size'], cache_dir=args['cache_dir'])
    elif command == 'compare':
        from .comparison import compare_evaluations
        names = args['names'] or [Path(p).name for p in args['evaluation']]
        if len(names) != len(args['evaluation']):
            parser.error('--names must match the number of --evaluation paths')
        result = compare_evaluations([{'name': name, 'path': path} for name,path in zip(names,args['evaluation'])],
                                    args['out'], mode=args['mode'])
    elif command == 'detection-diagnostics':
        from .diagnostics import detection_diagnostics
        result = detection_diagnostics(args['evaluation'], args['labels'], args['out'],
                                      max_per_cell=args['max_per_cell'], seed=args['seed'])
    elif command == 'controls-curated':
        from .controls import build_curated_controls
        result = build_curated_controls(read_json(args['spec']), args['out'], seed=args['seed'])
    elif command == 'controls-analyze':
        from .control_analysis import analyze_degradation
        result = analyze_degradation(args['evaluation'], args['labels'], args['out'],
                                     repetitions=args['bootstrap'], seed=args['seed'])
    elif command == 'classifier-audit':
        from .classifier_audit import audit_classifier
        result = audit_classifier(args['reference'], args['out'], bins=args['bins'])
    elif command == 'controls-composition':
        from .controls import build_composition_controls
        result = build_composition_controls(args['input'], args['labels'], read_json(args['spec']), args['out'], seed=args['seed'])
    elif command == 'controls-duplicates':
        from .controls import build_duplicate_controls
        result = build_duplicate_controls(args['input'], args['out'], n=args['count'], rates=args['rates'], seed=args['seed'])
    elif command == 'tail-register':
        from .tail_comparison import register_tail_pair
        result = register_tail_pair(read_json(args['spec']), args['out'])
    elif command == 'tail-analyze':
        from .tail_comparison import analyze_tail_pair
        result = analyze_tail_pair(args['registration'], read_json(args['spec']), args['out'], final_labels=args['labels'],
                                   repetitions=args['bootstrap'], seed=args['seed'])
    elif command == 'prototype-audit':
        from .prototypes import run_prototype_audit
        result = run_prototype_audit(args['reference'], args['splits'], args['evaluation'], args['out'],
                                     repository=args['repository'], weights=args['weights'], device=args['device'],
                                     batch_size=args['batch_size'], cache_dir=args['cache_dir'])
        result = {'status':result['status'],'seeds':[{'seed':r['seed'],'MMS':r['MMS']} for r in result['all_seeds']]}
    elif command == 'controls-summary':
        from .control_analysis import summarize_controls
        result=summarize_controls(args['evaluation'],args['out'])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
