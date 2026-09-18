"""Content-indexed feature shards with atomic commits and restart recovery.

Extraction is serialized per contract on a local POSIX filesystem. A failed
chunk is retried; committed chunks are reusable across cohorts and orderings.
The caller's contract must identify weights, preprocessing and implementation.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

import numpy as np

from .utils import read_json, sha256_file, stable_hash, write_json


def _content_id(record):
    digest = record.get('canonical_sha256') or record.get('sha256')
    if not digest:
        raise ValueError('Feature caching requires a verified image content hash')
    return digest


def _save(path, arrays):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.shard-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _read_index(path, contract):
    if not path.exists():
        return {'contract': contract, 'rows': {}, 'shards': {}, 'schema': None, 'metadata': {}}
    index = read_json(path)
    digest = index.pop('integrity_sha256', None)
    if digest != stable_hash(index) or index['contract'] != contract:
        raise ValueError('Modified feature cache index or contract')
    return index


def cached_features(root, records, contract, compute, *, chunk_size=256):
    """Return ordered feature arrays and provenance, extracting only missing content.

    ``compute`` receives a bounded list of image records and returns NumPy
    arrays with matching first dimensions, optionally plus JSON metadata.
    Corrupt committed shards are errors; they are never silently recomputed.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    records = list(records)
    if not records:
        raise ValueError('Feature extraction requires at least one image')
    ids = [_content_id(r) for r in records]
    contract = {'format': 'content-shards-v1', **contract}
    root = Path(root) / stable_hash(contract)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = root/'index.json'
        index = _read_index(index_path, contract)
        reused = sum(key in index['rows'] for key in ids)
        missing = {}
        for key, record in zip(ids, records):
            if key not in index['rows']:
                missing.setdefault(key, record)
        pending = list(missing.items())
        for lo in range(0, len(pending), chunk_size):
            batch = pending[lo:lo+chunk_size]
            result = compute([r for _, r in batch])
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            if not arrays or any(v.ndim < 1 or len(v) != len(batch) or
                                 not np.issubdtype(v.dtype, np.number) or
                                 np.iscomplexobj(v) or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Feature extraction returned invalid rows')
            schema = {k: {'shape': list(v.shape[1:]), 'dtype': v.dtype.str} for k, v in arrays.items()}
            if index['schema'] is not None and index['schema'] != schema:
                raise ValueError('Feature extraction schema changed between chunks')
            metadata = {k: v for k, v in result.get('metadata', {}).items()
                        if k not in ('seen_images', 'n_images', 'input_size_counts')}
            if index['schema'] is not None and stable_hash(index['metadata']) != stable_hash(metadata):
                raise ValueError('Feature extraction metadata changed between chunks')
            shard = stable_hash([key for key, _ in batch])
            path = root/f'{shard}.npz'
            _save(path, arrays)
            index['schema'], index['metadata'] = schema, metadata
            index['shards'][shard] = {'sha256': sha256_file(path), 'n': len(batch)}
            for pos, (key, _) in enumerate(batch):
                index['rows'][key] = [shard, pos]
            # Commit only after the complete shard is durable. Orphan files
            # left by a killed process are harmless and are overwritten on retry.
            write_json(index_path, {**index, 'integrity_sha256': stable_hash(index)})
            print(f'Feature cache: committed {min(lo+chunk_size, len(pending))}/{len(pending)} new images; {reused} existing rows', flush=True)
        output = {k: np.empty((len(ids), *s['shape']), dtype=s['dtype']) for k, s in index['schema'].items()}
        selections = {}
        for target, key in enumerate(ids):
            shard, position = index['rows'][key]
            selections.setdefault(shard, []).append((target, position))
        for shard, positions in selections.items():
            path = root/f'{shard}.npz'
            info = index['shards'][shard]
            if not path.is_file() or sha256_file(path) != info['sha256']:
                raise ValueError('Modified or missing feature cache shard')
            targets, sources = map(list, zip(*positions))
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(output):
                    raise ValueError('Modified feature cache schema')
                for key, dest in output.items():
                    value = arrays[key]
                    expected = index['schema'][key]
                    if value.shape != (info['n'], *expected['shape']) or value.dtype.str != expected['dtype'] or not np.isfinite(value).all():
                        raise ValueError('Modified feature cache array')
                    dest[targets] = value[sources]
        meta = {'key': stable_hash({'contract': contract, 'images': ids}),
                'contract': contract, 'path': str(root.resolve()),
                'metadata': {**index['metadata'], 'n_images': len(ids)},
                'reused_rows': reused, 'computed_rows': len(missing),
                'n_images': len(ids), 'unique_contents': len(set(ids)),
                'chunk_size': chunk_size, 'index_sha256': sha256_file(index_path)}
        return output, meta
