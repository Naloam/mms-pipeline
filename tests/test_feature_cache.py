import numpy as np
import pytest

from mms_eval.feature_cache import cached_features


def records(n):
    return [dict(image_id=str(i), canonical_sha256=f'content-{i}', path=str(i)) for i in range(n)]


def test_resume_then_expand_and_reorder_without_reextracting(tmp_path):
    seen = []
    interrupt = [True]

    def extract(rows):
        ids = [int(r['path']) for r in rows]
        if ids[0] == 2 and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError('simulated interruption')
        seen.extend(ids)
        return {'features': np.array(ids, dtype=np.float32)[:, None], 'metadata': {'backend': 'test'}}

    with pytest.raises(RuntimeError, match='interruption'):
        cached_features(tmp_path, records(5), {'extractor': 'v1'}, extract, chunk_size=2)
    values, meta = cached_features(tmp_path, records(5), {'extractor': 'v1'}, extract, chunk_size=2)
    assert seen == list(range(5))
    assert values['features'][:, 0].tolist() == list(range(5))
    assert meta['reused_rows'] == 2
    expanded = records(7)[::-1]
    values, meta = cached_features(tmp_path, expanded, {'extractor': 'v1'}, extract, chunk_size=2)
    assert values['features'][:, 0].tolist() == list(range(7))[::-1]
    assert sorted(seen) == list(range(7))
    assert meta['reused_rows'] == 5 and meta['computed_rows'] == 2


def test_content_dedup_contract_isolation_and_corruption(tmp_path):
    calls = []
    def extract(rows):
        calls.extend(r['path'] for r in rows)
        return {'features': np.arange(len(rows), dtype=float)[:, None]}
    rows = records(2)
    alias = {**rows[0], 'image_id': 'alias'}
    result, meta = cached_features(tmp_path, rows+[alias], {'weights': 'a'}, extract)
    assert calls == ['0', '1']
    np.testing.assert_array_equal(result['features'][0], result['features'][2])
    cached_features(tmp_path, rows, {'weights': 'b'}, extract)
    assert calls == ['0', '1', '0', '1']
    from pathlib import Path
    shard = next(Path(meta['path']).glob('*.npz'))
    shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='cache shard'):
        cached_features(tmp_path, rows, {'weights': 'a'}, extract)


def test_changed_content_and_failed_chunk_are_not_reused(tmp_path):
    calls = []
    def extract(rows):
        calls.extend(r['canonical_sha256'] for r in rows)
        return {'features': np.ones((len(rows), 2), dtype=np.float32)}
    cached_features(tmp_path, records(1), {}, extract)
    changed = [{**records(1)[0], 'canonical_sha256': 'changed'}]
    cached_features(tmp_path, changed, {}, extract)
    assert calls == ['content-0', 'changed']
    with pytest.raises(ValueError, match='invalid rows'):
        cached_features(tmp_path, records(2), {}, lambda r: {'features': np.full((len(r), 2), np.nan)})
    values, meta = cached_features(tmp_path, records(2), {}, extract)
    assert meta['reused_rows'] == 1
    assert values['features'].shape == (2, 2)
