"""Classification and probability-reliability audits on untouched real splits."""
from pathlib import Path

import numpy as np

from .annotation import _csv
from .pipeline import load_reference
from .semantic import SCORE_NAMES, apply_threshold, posterior_scores, wilson_interval
from .utils import read_jsonl, sha256_file, write_json


def classification_metrics(logits, labels, *, temperature=1., bins=15):
    scores = posterior_scores(logits, temperature)
    p = scores['probabilities']
    labels = np.asarray(labels)
    if (labels.shape != (len(p),) or not np.issubdtype(labels.dtype, np.integer) or
            labels.min() < 0 or labels.max() >= p.shape[1] or not isinstance(bins,int) or bins < 1):
        raise ValueError('Provide aligned valid class labels and a positive reliability bin count')
    pred = scores['class_order'][:,0]
    correct = pred == labels
    confidence = p.max(1)
    confusion = np.zeros((p.shape[1],p.shape[1]), dtype=int)
    np.add.at(confusion, (labels,pred), 1)
    bin_id = np.minimum((confidence*bins).astype(int),bins-1)
    reliability = []
    for b in range(bins):
        mask = bin_id==b
        n = int(mask.sum())
        reliability.append({'lower':b/bins,'upper':(b+1)/bins,'n':n,
                            'mean_confidence':float(confidence[mask].mean()) if n else None,
                            'accuracy':float(correct[mask].mean()) if n else None,
                            'accuracy_ci95':wilson_interval(int(correct[mask].sum()),n)})
    ece = sum(r['n']/len(p)*abs(r['accuracy']-r['mean_confidence']) for r in reliability if r['n'])
    onehot = np.eye(p.shape[1])[labels]
    return {'n':len(p),'accuracy':float(correct.mean()),'accuracy_ci95':wilson_interval(int(correct.sum()),len(p)),
            'nll':float(-scores['log_probabilities'][np.arange(len(p)),labels].mean()),
            'multiclass_brier':float(np.square(p-onehot).sum(1).mean()),'ece':float(ece),
            'ece_bins':bins,'ece_bin_rule':'equal-width [lo,hi), last includes 1',
            'confusion_rows_true_columns_predicted':confusion.tolist(),'reliability':reliability}


def audit_classifier(reference, output_dir, *, bins=15, plots=True):
    path, out = Path(reference).resolve(), Path(output_dir).resolve()
    ref = load_reference(path)
    if ref['config'].get('evaluator_kind')=='unlabeled_prototype':
        raise ValueError('Cluster IDs are not supervised class predictions; use the prototype cluster-composition audit')
    before = sha256_file(path)
    result, strata = {}, []
    with np.load(path.parent/ref['arrays_file'], allow_pickle=False) as arrays:
        for split in ('audit_internal','audit_external'):
            records = read_jsonl(path.parent/f'{split}.jsonl')
            labels = np.array([r['label'] for r in records])
            logits = arrays[split+'_logits']
            scores = posterior_scores(logits,ref['config'].get('temperature',1))
            result[split] = classification_metrics(logits,labels,temperature=ref['config'].get('temperature',1),bins=bins)
            error = scores['class_order'][:,0] != labels
            masks = [('true_class', name, labels==c) for c,name in enumerate(ref['classes'])]
            masks += [('classification_status','correct',~error),('classification_status','error',error)]
            for method in SCORE_NAMES:
                flags = apply_threshold(scores[method],ref['calibrations'][method])
                for kind,name,mask in masks:
                    n,count = int(mask.sum()),int(flags[mask].sum())
                    strata.append({'split':split,'stratum':kind,'value':name,'method':method,
                                   'n':n,'candidate_count':count,'candidate_rate':count/n if n else None,
                                   'candidate_rate_ci95':wilson_interval(count,n)})
    out.mkdir(parents=True,exist_ok=True)
    output = {'evaluator_id':ref['evaluator_id'],'classes':ref['classes'],'audits':result,'strata':strata,
              'reference_signature':ref['signature'],'reference_sha256':before,
              'main_reference_unchanged':before==sha256_file(path),
              'interpretation':'Real classification quality and probability calibration do not establish MM semantic validity. Per-class and error strata are descriptive; pooled calibration does not guarantee class-conditional rates.'}
    _csv(out/'classification_strata.csv',list(strata[0]),strata)
    write_json(out/'classifier_audit.json',output)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
        for column,(split,part) in enumerate(result.items()):
            reliability = [r for r in part['reliability'] if r['n']]
            ax=axes[0,column]
            ax.plot([0,1],[0,1],'--',color='gray')
            ax.scatter([r['mean_confidence'] for r in reliability],[r['accuracy'] for r in reliability],
                       s=[max(15,150*r['n']/part['n']) for r in reliability])
            ax.set(xlabel='Mean top-class probability',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title=split)
            confusion=np.array(part['confusion_rows_true_columns_predicted'])
            ax=axes[1,column]; ax.imshow(confusion,cmap='Blues')
            ax.set(xticks=range(len(ref['classes'])),yticks=range(len(ref['classes'])),
                   xticklabels=ref['classes'],yticklabels=ref['classes'],xlabel='Predicted class',ylabel='True class')
            if len(ref['classes']) <= 10:
                for (i,j),value in np.ndenumerate(confusion):
                    ax.text(j,i,str(value),ha='center',va='center',color='white' if value>confusion.max()/2 else 'black')
        fig.savefig(out/'classifier_audit.png',dpi=180);fig.savefig(out/'classifier_audit.pdf');plt.close(fig)
    return output
