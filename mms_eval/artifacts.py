"""Validate saved evaluations before downstream statistics, display, or pooling."""
from pathlib import Path

import numpy as np

from .utils import read_json, read_jsonl, sha256_file


def load_evaluation(path, *, features=False, semantic=True):
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    report = read_json(root/'report.json')
    required = [('input_manifest.jsonl', 'manifest_sha256')]
    if semantic:
        required.append(('scores.jsonl', 'scores_sha256'))
    if features:
        required.append(('features.npz', 'features_sha256'))
    for filename, field in required:
        if report.get(field) != sha256_file(root/filename):
            raise ValueError(f'Evaluation artifact changed: {filename}')
    records = read_jsonl(root/'input_manifest.jsonl')
    ids = [r['image_id'] for r in records]
    if report['n'] != len(records) or len(set(ids)) != len(ids):
        raise ValueError('Evaluation count or image identities disagree')
    rows = read_jsonl(root/'scores.jsonl') if semantic else []
    if semantic and [r['image_id'] for r in rows] != ids:
        raise ValueError('Evaluation scores and input manifest are not aligned')
    for row, record in zip(rows, records):
        if row.get('sha256') != record['sha256']:
            raise ValueError('Evaluation score and input content hashes disagree')
    arrays = {}
    if features:
        with np.load(root/'features.npz', allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if any(v.ndim < 1 or len(v) != len(records) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Evaluation feature rows are invalid or not aligned')
    return report, rows, records, arrays
