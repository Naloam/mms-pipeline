"""CPU-only post-run audits of immutable A/B references and stored predictions."""
import argparse
from pathlib import Path

from mms_eval.analysis import adaptivity_audit
from mms_eval.classifier_audit import audit_classifier
from mms_eval.comparison import compare_evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    for name in ('A','B'):
        audit_classifier(root/f'reference_{name}_clean_engineering_v2/reference.json',
                         root/f'classifier_audit_{name}_clean_v3')
        print('CLASSIFIER_AUDIT_'+name+'_COMPLETE',flush=True)
    adaptivity_audit(root/'reference_A_clean_engineering_v2/reference.json',
                     root/'scale_acceptance_clean_v2/n5000', root/'adaptivity_A_clean_v3')
    print('ADAPTIVITY_A_COMPLETE',flush=True)
    compare_evaluations([{'name':'ResNet50_A','path':root/'scale_acceptance_clean_v2/n5000'},
                         {'name':'ViT_B16_B','path':root/'evaluation_B_clean_5000_v2'}],
                        root/'comparison_AB_clean_v3',mode='evaluators')
    compare_evaluations([{'name':str(n),'path':root/f'scale_acceptance_clean_v2/n{n}'}
                         for n in (1000,5000,10000,50010)],root/'comparison_scale_clean_v3',mode='scale')
    print('FINAL_AUDITS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
