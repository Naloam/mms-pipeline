"""Repackage the existing development cohort without reselection or score exposure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from mms_eval.artifacts import load_evaluation
from mms_eval.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pack', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source, evaluation, out = map(lambda s: Path(s).resolve(),
                                  (args.source_pack, args.evaluation, args.out))
    if out.exists():
        raise FileExistsError('Use a new output directory; existing human work must remain untouched')
    old = read_json(source/'pack.json')
    assert sha256_file(source/'private_key.jsonl') == old['private_key_sha256']
    original = read_jsonl(source/'private_key.jsonl')
    report, scored, _, _ = load_evaluation(evaluation)
    by_hash = {r['sha256']: r for r in scored}
    assert len(by_hash) == len(scored)
    assert len(original) == 200 and len({r['blind_id'] for r in original}) == 200
    assert len({r['sha256'] for r in original}) == 200
    for row in original:
        assert row['sha256'] in by_hash
        assert sha256_file(source/'annotator/images'/f"{row['blind_id']}.png") == row['sha256']
    guide_version = 'mms-dev-20260914-v1'
    pack_id = 'mms-development-200-20260914-v1'
    public = out/'annotator'
    (public/'images').mkdir(parents=True)
    private, images, blank_rows = [], [], []
    for old_row in original:
        blind = old_row['blind_id']
        picture = public/'images'/f'{blind}.png'
        shutil.copy2(source/'annotator/images'/f'{blind}.png', picture)
        digest = sha256_file(picture)
        private.append({**by_hash[old_row['sha256']], 'blind_id': blind, 'display_sha256': digest})
        images.append({'id': blind, 'file': f'images/{blind}.png', 'sha256': digest})
        blank_rows.append({'pack_id': pack_id, 'blind_id': blind, 'annotator_id': '', 'label': '',
                           'other_defects': '', 'human_target_class': 'unknown', 'reason': '',
                           'seconds': '', 'annotation_guide_version': guide_version,
                           'display_sha256': digest})
    guide = (repo/'docs/人工标注指南_开发版.md').read_text()
    assert guide_version in guide
    (public/'README.md').write_text(guide)
    with (public/'labels_template.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(blank_rows[0]))
        writer.writeheader()
        writer.writerows(blank_rows)
    payload = {'pack_id': pack_id, 'guide': guide_version, 'images': images}
    template = repo/'MMS执行模板/annotation_ui.template.html'
    html = template.read_text()
    assert html.count('__PACK_JSON__') == 1
    (public/'index.html').write_text(html.replace('__PACK_JSON__',
        json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')))
    write_jsonl(out/'private_key.jsonl', private)
    meta = {'pack_id': pack_id, 'role': 'development', 'n': len(private),
            'cohort_n': report['n'], 'allowed_labels': ['yes', 'no', 'unknown'],
            'human_validity': 'pending', 'annotation_guide_version': guide_version,
            'annotation_guide_sha256': sha256_file(public/'README.md'),
            'private_key_sha256': sha256_file(out/'private_key.jsonl'),
            'scores_sha256': report['scores_sha256'], 'evaluation_report_sha256': sha256_file(evaluation/'report.json'),
            'display_policy': report['image_policy'],
            'source_pack': str(source), 'source_pack_sha256': sha256_file(source/'pack.json'),
            'source_private_key_sha256': old['private_key_sha256'],
            'source_sampling': old['sampling'], 'source_seed': old['seed'],
            'sampling': 'repackage_existing_development_sample_without_reselection',
            'original_blind_order_preserved': True, 'all_display_files_match_original_image_sha256': True,
            'exclude_from_final_random': True,
            'public_files': 'Only distribute annotator/ or the public ZIP; do not distribute private_key.jsonl.',
            'ui_template_sha256': sha256_file(template)}
    write_json(out/'pack.json', meta)
    files = sorted(p for p in public.rglob('*') if p.is_file())
    (public/'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(public)}\n' for p in files))
    archive = out/'MMS_development_200_20260914.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(p for p in public.rglob('*') if p.is_file()):
            z.write(path, arcname=str(path.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert name.startswith('annotator/') and 'private' not in name
            assert hashlib.sha256(z.read(name)).hexdigest() == sha256_file(out/name)
    write_json(out/'package_verification.json', {
        'n': 200, 'original_source_hashes_match': True,
        'final_evaluation_identity_match': True, 'role': 'development',
        'public_zip_sha256': sha256_file(archive), 'public_zip_bytes': archive.stat().st_size,
        'zip_entries': len(z.namelist()), 'zip_all_entry_hashes_verified': True,
        'no_auto_labels': True, 'research_goal_activated': False})
    print(f'Prepared {public}: 200 source-verified development images; no labels or research run created.')


if __name__ == '__main__':
    main()
