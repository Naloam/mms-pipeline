"""Prepare blank v2 development packs; never migrate or rewrite human judgments."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

from mms_eval.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl

GUIDE = 'mms-dev-major3-20260916-v2'
FIELDS = ['pack_id', 'blind_id', 'annotator_id', 'label', 'other_defects',
          'human_target_class', 'reason', 'seconds', 'annotation_guide_version', 'display_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pack', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory to preserve human work')
    repo = Path(__file__).resolve().parents[1]
    old = read_json(args.source_pack/'pack.json')
    assert sha256_file(args.source_pack/'private_key.jsonl') == old['private_key_sha256']
    private = read_jsonl(args.source_pack/'private_key.jsonl')
    with args.labels.open(encoding='utf-8-sig', newline='') as f:
        labels = list(csv.DictReader(f))
    by_id = {r['blind_id']: r for r in labels}
    assert len(labels) == len(by_id) == len(private) == 200
    for row in private:
        label = by_id[row['blind_id']]
        assert label['pack_id'] == old['pack_id']
        assert label['annotation_guide_version'] == old['annotation_guide_version']
        assert label['display_sha256'] == row['display_sha256']
        assert label['label'] in ('yes', 'no', 'unknown')
        assert sha256_file(args.source_pack/'annotator/images'/f"{row['blind_id']}.png") == row['display_sha256']
    guide = (repo/'docs/人工标注指南_三大类_v2.md').read_text()
    template = (repo/'MMS执行模板/annotation_ui_major3.template.html').read_text()
    assert GUIDE in guide and template.count('__PACK_JSON__') == 1
    summaries = []
    for name, selected in [('review43', [r for r in private if by_id[r['blind_id']]['label'] != 'no']),
                           ('independent200', private)]:
        pack_id = f'mms-dev-major3-{name}-20260916-v2'
        out = args.out/name
        public = out/'annotator'
        (public/'images').mkdir(parents=True)
        images, blank = [], []
        for row in selected:
            blind = row['blind_id']
            src = args.source_pack/'annotator/images'/f'{blind}.png'
            shutil.copy2(src, public/'images'/f'{blind}.png')
            images.append({'id': blind, 'file': f'images/{blind}.png', 'sha256': row['display_sha256']})
            blank.append(dict(zip(FIELDS, [pack_id, blind, '', '', '', 'unknown', '', '', GUIDE, row['display_sha256']])))
        # A second rater must not learn how the focused first-rater subset was selected.
        public_guide = guide.split('## 本轮版本与复核范围')[0]
        public_guide += ('\n## 本批用途\n\n这是原标注者的开发复核包。按新指南重新独立判断，旧版CSV不能导入。\n'
                        if name == 'review43' else
                        '\n## 本批用途\n\n这是完整开发样本的独立标注包。所有图片从空白开始；独立完成前不要查看他人的标签、理由或模型分析。旧版CSV不能导入。\n')
        (public/'README.md').write_text(public_guide)
        with (public/'labels_template.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(blank)
        payload = {'pack_id': pack_id, 'guide': GUIDE, 'images': images}
        (public/'index.html').write_text(template.replace('__PACK_JSON__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')))
        write_jsonl(out/'private_key.jsonl', selected)
        write_json(out/'pack.json', {
            'pack_id': pack_id, 'role': 'development', 'n': len(selected), 'human_validity': 'pending',
            'annotation_guide_version': GUIDE, 'annotation_guide_sha256': sha256_file(public/'README.md'),
            'private_key_sha256': sha256_file(out/'private_key.jsonl'),
            'source_pack_sha256': sha256_file(args.source_pack/'pack.json'),
            'source_labels_sha256': sha256_file(args.labels),
            'selection': 'prior_yes_or_unknown_development_only' if name == 'review43' else 'all_original_development_images',
            'exclude_from_final_random': True, 'no_auto_labels': True,
            'all_images_byte_identical': True, 'no_formal_prevalence_claim': True})
        (public/'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(public)}\n' for p in sorted(public.rglob('*')) if p.is_file()))
        archive = out/f'MMS_{name}_major3_v2.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(public.rglob('*')):
                if p.is_file(): z.write(p, str(p.relative_to(out)))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert all(p.startswith('annotator/') and 'private' not in p for p in z.namelist())
            assert len([p for p in z.namelist() if p.endswith('.png')]) == len(selected)
        summaries.append({'name': name, 'n': len(selected), 'zip_sha256': sha256_file(archive),
                          'zip_bytes': archive.stat().st_size, 'all_labels_blank': all(not r['label'] for r in blank)})
    write_json(args.out/'verification.json', {'packs': summaries, 'original_labels_sha256': sha256_file(args.labels),
                                           'original_labels_unchanged': True})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
