"""Real-image extractor parity and independent PR-neighbour verification."""
import argparse
from pathlib import Path

import numpy as np

from mms_eval.distribution import precision_recall_from_features
from mms_eval.images import canonicalize_records, collect_images
from mms_eval.quality_backends import CleanFIDExtractor, VGGPRExtractor, precise_inference
from mms_eval.utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = Path(args.out)
    records = collect_images(sorted(Path(args.images).glob('*.png'))[:32])
    canonical = canonicalize_records(records, out/'rgb', {'canonical_size': [256, 256]})
    paths = [r['path'] for r in canonical]
    clean = CleanFIDExtractor(args.device, 8)
    actual = clean.extract_paths(paths)['features']
    from cleanfid.fid import get_files_features
    import torch
    with precise_inference(args.device):
        expected = get_files_features(paths, model=clean.model, num_workers=0, batch_size=8,
                                      device=torch.device(args.device), mode='clean', verbose=False)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    vgg = VGGPRExtractor(args.device, 8)
    features = vgg.extract_paths(paths)['features']
    real, gen = features[:16].astype(float), features[16:].astype(float)
    from scipy.spatial.distance import cdist
    rr, gg, rg = cdist(real, real), cdist(gen, gen), cdist(real, gen)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)
    rradius, gradius = np.sort(rr, axis=1)[:, 2], np.sort(gg, axis=1)[:, 2]
    expected_pr = {'precision': float((rg <= rradius[:, None]).any(axis=0).mean()),
                   'recall': float((rg <= gradius[None, :]).any(axis=1).mean())}
    actual_pr = precision_recall_from_features(real, gen, chunk_size=7)
    for name, value in expected_pr.items():
        assert actual_pr[name] == value, (name, actual_pr[name], value)
    result = {'status': 'passed', 'n': len(paths), 'clean_fid_max_abs_error': float(np.abs(actual-expected).max()),
              'clean_fid_metadata': clean.metadata, 'vgg_metadata': vgg.metadata,
              'pr_independent_cdist': expected_pr, 'pr_ours': actual_pr,
              'scope': 'Engineering agreement on real generated files, not human semantic validity'}
    write_json(out/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
