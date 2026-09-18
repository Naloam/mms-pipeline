"""Blinded uniform audit packs and transparent human-label agreement analysis."""
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .images import load_rgb, save_png_atomic
from .semantic import wilson_interval
from .utils import atomic_text, read_json, read_jsonl, sha256_file, write_json, write_jsonl

LABELS = ('clear_single','mixed_semantics','unrecognizable','uncertain')


def _csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    atomic_text(path,stream.getvalue())


def annotation_pack(evaluation, output_dir, *, count=200, seed=2026091105, role='development'):
    run, out = Path(evaluation).resolve(), Path(output_dir).resolve()
    from .artifacts import load_evaluation
    report, rows, _, _ = load_evaluation(run)
    if role not in ('development', 'final_random', 'diagnostic'):
        raise ValueError('Annotation role must be development, final_random, or diagnostic')
    if role == 'final_random':
        raise ValueError('Final random packs require a frozen study registration; use register-study then study-pack')
    if count < 1 or count > len(rows):
        raise ValueError('Annotation count must be between one and the cohort size')
    if (out/'pack.json').exists():
        raise ValueError('Annotation pack already exists; preserve prior labels and choose a new directory')
    rng = np.random.RandomState(seed)
    sample = [rows[i] for i in rng.choice(len(rows),count,replace=False)]
    key, labels = [], []
    for i,row in enumerate(sample):
        blind = f'{i+1:04d}'
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError('Source image changed before blind display')
        save_png_atomic(load_rgb(row['path'], report['image_policy']),out/'annotator'/'images'/f'{blind}.png')
        key.append({'blind_id':blind,**row})
        labels.append({'blind_id':blind,'annotator_id':'','label':'','other_defects':'','human_target_class':'','reason':'','seconds':'','annotation_guide_version':''})
    _csv(out/'annotator'/'labels_template.csv',list(labels[0]),labels)
    instructions = '''# 人工标注说明
只将 annotator 文件夹交给标注者，private_key.jsonl 留给分析者。
请独立看每张图，填写 labels_template.csv（可复制成每人一份）。不查看模型分数、不讨论后再填写。
- yes：同一主体有可指出的不合理语义融合特征，不要求强行猜出两个类别。
- no：可判断为无上述混合；可以有多个完整主体、模糊或其他独立缺陷。
- unknown：无法可靠判断。缺陷另填 other_defects，可与 yes/no 同时存在；严重损坏不自动等于混合。
已定义类别见下方。模糊图像、两个完整主体同框、背景复杂等均不能仅凭这一点判为混合。
先用独立示例统一语义边界，再对正式样本独立标注。保留原始 CSV；将各人的文件交给 annotation-finalize，分歧另行裁决。
'''
    instructions += '\n本次类别：' + '、'.join(report['mms']['classes']) + '\n'
    atomic_text(out/'annotator'/'README.md',instructions)
    # Contact sheets carry only blind IDs; no score, class, flag, or model identity.
    for page,start in enumerate(range(0,count,100)):
        selected = key[start:start+100]
        sheet = Image.new('RGB',(10*144,((len(selected)+9)//10)*164),'white')
        draw = ImageDraw.Draw(sheet)
        for j,row in enumerate(selected):
            im = load_rgb(out/'annotator'/'images'/f"{row['blind_id']}.png")
            im.thumbnail((140,140))
            x,y=(j%10)*144,(j//10)*164
            sheet.paste(im,(x+(144-im.width)//2,y))
            draw.text((x+4,y+143),row['blind_id'],fill='black')
        save_png_atomic(sheet,out/'annotator'/f'contact_sheet_{page+1}.png')
    write_jsonl(out/'private_key.jsonl',key)
    meta = {'n':count,'cohort_n':len(rows),'seed':seed,'sampling':'uniform_without_replacement',
            'scores_sha256':report['scores_sha256'],'private_key_sha256':sha256_file(out/'private_key.jsonl'),
            'allowed_labels':('yes','no','unknown'),'human_validity':'pending', 'role': role,
            'display_policy': report['image_policy']}
    write_json(out/'pack.json',meta)
    return meta


def analyze_annotations(pack, labels, output_dir):
    root = Path(pack)
    meta = read_json(root/'pack.json')
    if sha256_file(root/'private_key.jsonl') != meta['private_key_sha256']:
        raise ValueError('Annotation mapping changed')
    key = {r['blind_id']:r for r in read_jsonl(root/'private_key.jsonl')}
    with Path(labels).open(encoding='utf-8-sig',newline='') as f:
        records = list(csv.DictReader(f))
    seen, by_rater = set(), {}
    for row in records:
        row['label'] = {'yes': 'mixed_semantics', 'no': 'clear_single', 'unknown': 'uncertain'}.get(row['label'], row['label'])
        pair = row['blind_id'],row['annotator_id'].strip()
        if pair in seen or pair[0] not in key or not pair[1] or row['label'] not in LABELS:
            raise ValueError('Invalid, duplicate, unknown, or incomplete annotation row')
        seen.add(pair); by_rater.setdefault(pair[1],{})[pair[0]] = row['label']
    if not records:
        raise ValueError('No annotations supplied')
    result = {}
    for rater, data in by_rater.items():
        # All four labels remain separate; unresolved and unrecognizable are not clear negatives.
        counts = {label:sum(x==label for x in data.values()) for label in LABELS}
        resolvable = {i:x for i,x in data.items() if x in ('clear_single','mixed_semantics')}
        tp = sum(x=='mixed_semantics' and key[i]['flag_entropy'] for i,x in resolvable.items())
        fp = sum(x=='clear_single' and key[i]['flag_entropy'] for i,x in resolvable.items())
        fn = sum(x=='mixed_semantics' and not key[i]['flag_entropy'] for i,x in resolvable.items())
        tn = sum(x=='clear_single' and not key[i]['flag_entropy'] for i,x in resolvable.items())
        result[rater] = {'n_labeled':len(data),'missing':meta['n']-len(data),'label_counts':counts,
            'resolved_n':len(resolvable),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision_on_resolved':tp/(tp+fp) if tp+fp else None,
            'recall_on_resolved':tp/(tp+fn) if tp+fn else None,
            'precision_ci95':wilson_interval(tp,tp+fp),'recall_ci95':wilson_interval(tp,tp+fn)}
    agreements = []
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    for a,b in combinations(sorted(by_rater),2):
        common = sorted(set(by_rater[a]) & set(by_rater[b]))
        if common:
            x,y=[by_rater[a][i] for i in common],[by_rater[b][i] for i in common]
            kappa = cohen_kappa_score(x,y,labels=list(LABELS)) if len(set(x+y))>1 else float('nan')
            agreements.append({'raters':[a,b],'n':len(common),'agreement':float(np.mean(np.array(x)==np.array(y))),
                               'cohen_kappa':float(kappa) if np.isfinite(kappa) else None})
    output = {'raters':result,'agreement':agreements,'pack_sha256':sha256_file(root/'pack.json'),
              'labels_sha256':sha256_file(labels),'status':'descriptive_human_audit',
              'note':'Precision/recall are conditional on clear or mixed judgments. Report unresolved counts and per-rater disagreement; no automatic adjudication or validity pass.'}
    write_json(Path(output_dir)/'human_audit.json',output)
    return output
