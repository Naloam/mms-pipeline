"""Explicit Clean-FID / standard IS / NVIDIA VGG-PR feature protocols.

Official model weights and source files are fingerprinted. No torchvision
Inception or VGG substitution is made when a published weight is unavailable.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np

from .distribution import FidelityExtractor, _positive_int
from .utils import sha256_file, stable_hash

CLEAN_FID_VERSION = '0.1.35'
VGG_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
STYLEGAN3_COMMIT = 'c233a919a6faee6e36a316ddd4eddababad1adf9'
STYLEGAN3_REPOSITORY = 'https://github.com/NVlabs/stylegan3.git'


def _stylegan3_runtime():
    """Load the pinned official persistence runtime used by its published PKL."""
    import torch
    root = Path(torch.hub.get_dir())/f'mms-stylegan3-{STYLEGAN3_COMMIT}'
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'torch_utils'/'persistence.py').is_file():
        subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(root), 'fetch', '--depth', '1', STYLEGAN3_REPOSITORY, STYLEGAN3_COMMIT], check=True, timeout=180)
        subprocess.run(['git', '-C', str(root), 'checkout', '--detach', STYLEGAN3_COMMIT], check=True)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != STYLEGAN3_COMMIT:
        raise ValueError('NVIDIA feature-detector runtime revision differs from its fixed protocol')
    for name in ('torch_utils', 'dnnlib'):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'{name} already loaded from a different repository; run quality evaluation in a fresh process')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sources = {str(p.relative_to(root)): sha256_file(p) for name in ('torch_utils', 'dnnlib') for p in (root/name).rglob('*.py')}
    return {'repository': STYLEGAN3_REPOSITORY, 'commit': revision, 'runtime_sources_sha256': stable_hash(sources)}


@contextmanager
def precise_inference(device):
    import torch
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False), torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _read(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        if image.mode != 'RGB' or 'transparency' in image.info:
            raise ValueError('Quality protocols require canonical RGB images')
        return np.asarray(ImageOps.exif_transpose(image), dtype=np.uint8).copy()


class CleanFIDExtractor:
    def __init__(self, device='cpu', batch_size=32, *, weights_dir=None):
        import torch
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import build_resizer
        version = importlib.metadata.version('clean-fid')
        if version != CLEAN_FID_VERSION:
            raise RuntimeError(f'Protocol requires clean-fid=={CLEAN_FID_VERSION}, found {version}')
        self.device, self.batch_size = torch.device(device), _positive_int(batch_size, 'batch_size')
        folder = Path(weights_dir) if weights_dir else Path(torch.hub.get_dir())/'checkpoints'/'clean-fid'
        folder.mkdir(parents=True, exist_ok=True)
        self.model = InceptionV3W(str(folder), download=True, resize_inside=False).eval().to(self.device)
        self.resize = build_resizer('clean')
        self.metadata = {
            'backend': 'clean-fid', 'backend_version': version, 'mode': 'clean',
            'feature_layer': '2048', 'feature_dimension': 2048,
            'weights_sha256': sha256_file(folder/'inception-2015-12-05.pt'),
            'extractor_source_sha256': sha256_file(inspect.getfile(InceptionV3W)),
            'resizer_source_sha256': sha256_file(inspect.getfile(build_resizer)),
            'torch_version': torch.__version__, 'pillow_version': importlib.metadata.version('Pillow'),
            'preprocessing': {'input': 'canonical RGB uint8', 'resize': 'Clean-FID clean PIL float-channel bicubic 299x299 without requantization',
                              'normalization': '(x-128)/128 in official InceptionV3W'},
            'device': str(self.device), 'batch_size': batch_size,
            'inference_dtype': 'float32; no autocast; TF32 disabled',
        }

    def extract_paths(self, paths):
        import torch
        paths = list(paths)
        features = np.empty((len(paths), 2048), dtype=np.float32)
        with precise_inference(self.device):
            for lo in range(0, len(paths), self.batch_size):
                values = [self.resize(_read(p)).transpose(2, 0, 1) for p in paths[lo:lo+self.batch_size]]
                tensor = torch.from_numpy(np.stack(values)).to(self.device)
                features[lo:lo+len(values)] = self.model(tensor).cpu().numpy()
        return {'features': features, 'metadata': self.metadata}


class VGGPRExtractor:
    def __init__(self, device='cpu', batch_size=32, *, weights_path=None):
        import torch
        self.device, self.batch_size = torch.device(device), _positive_int(batch_size, 'batch_size')
        path = Path(weights_path) if weights_path else Path(torch.hub.get_dir())/'checkpoints'/'mms-nvidia-vgg16.pt'
        if not path.is_file():
            if weights_path is not None:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.vgg-download-')
            os.close(fd)
            try:
                torch.hub.download_url_to_file(VGG_URL, temp)
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        with path.open('rb') as stream:
            format_header = stream.read(2)
        runtime = {}
        if format_header == b'PK':
            self.model = torch.jit.load(str(path), map_location=self.device).eval()
            model_format = 'torchscript'
        else:
            # Current official NGC weights use NVIDIA's persistent Python
            # pickle; its loader and revision are part of the frozen protocol.
            runtime = _stylegan3_runtime()
            with path.open('rb') as stream:
                self.model = pickle.load(stream).to(self.device).eval()
            model_format = 'nvidia_persistent_pickle'
        self.metadata = {
            'backend': 'nvidia-stylegan3-vgg16', 'weights_url': VGG_URL,
            'model_format': model_format, 'runtime': runtime,
            'weights_sha256': sha256_file(path), 'feature_dimension': 4096,
            'feature_layer': 'return_features=True', 'torch_version': torch.__version__,
            'preprocessing': {'input': 'canonical RGB uint8 NCHW', 'resize_normalization': 'inside published NVIDIA detector; no torchvision transforms'},
            'device': str(self.device), 'batch_size': batch_size,
            'inference_dtype': 'float32; no autocast; TF32 disabled',
            'distance_dtype': 'float64; differs from NVIDIA half-precision distance implementation',
        }

    def extract_paths(self, paths):
        import torch
        paths = list(paths)
        features = np.empty((len(paths), 4096), dtype=np.float32)
        with precise_inference(self.device):
            for lo in range(0, len(paths), self.batch_size):
                groups = {}
                for i in range(lo, min(len(paths), lo+self.batch_size)):
                    value = _read(paths[i])
                    groups.setdefault(value.shape, []).append((i, value))
                for entries in groups.values():
                    ids, values = zip(*entries)
                    tensor = torch.from_numpy(np.stack(values).transpose(0, 3, 1, 2).copy()).to(self.device)
                    result = self.model(tensor, return_features=True)
                    if tuple(result.shape) != (len(ids), 4096):
                        raise ValueError('Published VGG detector returned unexpected feature shape')
                    features[list(ids)] = result.cpu().numpy()
        return {'features': features, 'metadata': self.metadata}


class ProtocolExtractor:
    """Keep the three named measurement networks distinct behind one extractor."""
    def __init__(self, config, device='cpu', batch_size=32):
        backend = config.get('feature_backend', 'torch-fidelity')
        pr_backend = config.get('pr_feature_backend', 'inception')
        if backend not in ('torch-fidelity', 'clean-fid') or pr_backend not in ('inception', 'vgg16'):
            raise ValueError('Unsupported quality feature protocol')
        self.fidelity = FidelityExtractor(device, batch_size)
        self.clean = CleanFIDExtractor(device, batch_size) if backend == 'clean-fid' else None
        self.vgg = VGGPRExtractor(device, batch_size) if pr_backend == 'vgg16' else None
        components = {'fid_kid': (self.clean or self.fidelity).metadata,
                      'is': self.fidelity.metadata,
                      'pr': self.vgg.metadata if self.vgg else (self.clean or self.fidelity).metadata}
        self.metadata = {'backend': 'mms-quality-protocol', 'backend_version': '1',
                         'feature_backend': backend, 'pr_feature_backend': pr_backend,
                         'components': components, 'weights_sha256': stable_hash({k: v['weights_sha256'] for k,v in components.items()}),
                         'feature_dimension': 2048, 'logit_dimension': 1008,
                         'preprocessing': {k: v['preprocessing'] for k, v in components.items()},
                         'device': device, 'batch_size': batch_size}

    def extract_paths(self, paths):
        paths = list(paths)
        output = self.fidelity.extract_paths(paths)
        if self.clean:
            output['features'] = self.clean.extract_paths(paths)['features']
        if self.vgg:
            output['pr_features'] = self.vgg.extract_paths(paths)['features']
        return {**output, 'metadata': self.metadata}
