"""Collection metrics on cached Inception features; importing needs no PyTorch.

FID/KID/IS use the definitions in torch-fidelity 0.3.0. Improved precision
and recall use k-neighbour balls, with Inception features explicitly named.
These are collection statistics, never independent scores for one image.
See docs/metric_backend.md for provenance, interpretation, and limitations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


IMPLEMENTATION_VERSION = "mms-distribution-v1"
FIDELITY_VERSION = "0.3.0"
INCEPTION_NAME = "inception-v3-compat"
INCEPTION_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    "weights-inception-2015-12-05-6726825d.pth"
)


def _matrix(value: np.ndarray | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N, D] with D > 0")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numbers")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array.astype(np.float64, copy=False)


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def fid_from_features(real: np.ndarray, generated: np.ndarray) -> float:
    """Unbiased sample covariances and Gaussian 2-Wasserstein distance.

    The low-rank form is algebraically identical, and avoids decomposing a
    2048-square covariance during small engineering checks. This does not
    remove FID's finite-sample bias.
    """
    real = _matrix(real, "real_features")
    generated = _matrix(generated, "generated_features")
    if real is None or generated is None:
        raise ValueError("FID requires real and generated features")
    if real.shape[1] != generated.shape[1] or min(len(real), len(generated)) < 2:
        raise ValueError("FID requires matching dimensions and at least two samples in each set")
    from scipy import linalg

    mean_real, mean_generated = real.mean(0), generated.mean(0)
    a = (real - mean_real) / math.sqrt(len(real) - 1)
    b = (generated - mean_generated) / math.sqrt(len(generated) - 1)
    mean_distance = float(np.square(mean_real - mean_generated).sum())
    trace_real, trace_generated = float(np.square(a).sum()), float(np.square(b).sum())
    if max(len(real), len(generated)) < real.shape[1]:
        covariance_overlap = float(linalg.svdvals(a @ b.T).sum())
    else:
        covariance_real, covariance_generated = a.T @ a, b.T @ b
        # A symmetric PSD product avoids arbitrary complex parts of sqrtm(C1 C2).
        eigenvalues, eigenvectors = linalg.eigh(covariance_real)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        product = root @ covariance_generated @ root
        product = (product + product.T) * 0.5
        covariance_overlap = float(np.sqrt(np.maximum(linalg.eigvalsh(product), 0)).sum())
    result = mean_distance + trace_real + trace_generated - 2 * covariance_overlap
    # Roundoff at identity may make the result slightly negative. Do not hide
    # a substantially negative distance, which indicates a numerical failure.
    tolerance = 1e-7 * max(1.0, mean_distance + trace_real + trace_generated)
    if not np.isfinite(result) or result < -tolerance:
        raise FloatingPointError(f"Invalid FID numerical result: {result}")
    return float(max(0.0, result))


def polynomial_mmd_unbiased(real: np.ndarray, generated: np.ndarray) -> float:
    """Unbiased MMD² for k(x,y) = (xᵀy / D + 1)^3; may be negative."""
    real = _matrix(real, "real_features")
    generated = _matrix(generated, "generated_features")
    if real is None or generated is None or real.shape[1] != generated.shape[1]:
        raise ValueError("KID requires matching feature dimensions")
    nr, ng = len(real), len(generated)
    if min(nr, ng) < 2:
        raise ValueError("Unbiased KID requires at least two samples in each set")
    dimension = real.shape[1]
    rr = (real @ real.T / dimension + 1) ** 3
    gg = (generated @ generated.T / dimension + 1) ** 3
    rg = (real @ generated.T / dimension + 1) ** 3
    value = (
        (rr.sum() - np.trace(rr)) / (nr * (nr - 1))
        + (gg.sum() - np.trace(gg)) / (ng * (ng - 1))
        - 2 * rg.mean()
    )
    if not np.isfinite(value):
        raise FloatingPointError("KID kernel overflowed; inspect the feature cache")
    return float(value)


def kid_from_features(
    real: np.ndarray, generated: np.ndarray, *, subsets: int = 100,
    subset_size: int = 1000, seed: int = 2020,
) -> dict:
    real = _matrix(real, "real_features")
    generated = _matrix(generated, "generated_features")
    subsets = _positive_int(subsets, "kid_subsets")
    subset_size = _positive_int(subset_size, "kid_subset_size", 2)
    if real is None or generated is None or min(len(real), len(generated)) < subset_size:
        raise ValueError("Not enough samples for the configured KID subset_size")
    rng = np.random.RandomState(seed)
    values = [polynomial_mmd_unbiased(
        real[rng.choice(len(real), subset_size, replace=False)],
        generated[rng.choice(len(generated), subset_size, replace=False)],
    ) for _ in range(subsets)]
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
        "subsets": subsets, "subset_size": subset_size,
        "std_kind": "standard deviation across random subsets; not a confidence interval",
        "subset_values": values, "seed": int(seed),
    }


def inception_score_from_logits(
    logits: np.ndarray, *, splits: int = 10, seed: int = 2020, shuffle: bool = True,
) -> dict:
    """IS from official Inception logits, never the MMS domain classifier."""
    logits = _matrix(logits, "generated_logits")
    splits = _positive_int(splits, "isc_splits")
    if logits is None or len(logits) < 2 * splits:
        raise ValueError("IS requires at least two images per configured split")
    if logits.shape[1] < 2:
        raise ValueError("IS logits must have at least two classes")
    from scipy.special import logsumexp

    if shuffle:
        logits = logits[np.random.RandomState(seed).permutation(len(logits))]
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(log_probabilities)
    scores = []
    sizes = []
    for index in range(splits):
        lo, hi = index * len(logits) // splits, (index + 1) * len(logits) // splits
        log_p = log_probabilities[lo:hi]
        # logsumexp prevents 0 * log(0) for extremely unlikely classes.
        log_marginal = logsumexp(log_p, axis=0, keepdims=True) - math.log(hi - lo)
        kl = (probabilities[lo:hi] * (log_p - log_marginal)).sum(axis=1).mean()
        scores.append(float(np.exp(kl)))
        sizes.append(hi - lo)
    return {
        "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=0)),
        "splits": splits, "split_sizes": sizes, "split_values": scores,
        "std_kind": "standard deviation across splits; not a confidence interval",
        "shuffle": bool(shuffle), "seed": int(seed),
    }


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    distances = np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * (a @ b.T)
    if not np.isfinite(distances).all():
        raise FloatingPointError("PR distances overflowed; inspect the feature cache")
    return np.maximum(distances, 0.0)


def _knn_radii_squared(features: np.ndarray, k: int, chunk_size: int) -> np.ndarray:
    radii = np.empty(len(features), dtype=np.float64)
    for lo in range(0, len(features), chunk_size):
        hi = min(lo + chunk_size, len(features))
        nearest = np.full((hi - lo, k), np.inf)
        for col in range(0, len(features), chunk_size):
            end = min(col + chunk_size, len(features))
            distances = _squared_distances(features[lo:hi], features[col:end])
            # Exclude each point's own distance, but retain other identical points.
            overlap = np.arange(max(lo, col), min(hi, end))
            distances[overlap - lo, overlap - col] = np.inf
            candidates = np.concatenate((nearest, distances), axis=1)
            nearest = np.partition(candidates, k - 1, axis=1)[:, :k]
        radii[lo:hi] = nearest.max(axis=1)
    return radii


def _inside_manifold(
    probes: np.ndarray, centres: np.ndarray, radii: np.ndarray, chunk_size: int,
) -> np.ndarray:
    membership = np.zeros(len(probes), dtype=bool)
    for lo in range(0, len(probes), chunk_size):
        hi = min(lo + chunk_size, len(probes))
        inside = np.zeros(hi - lo, dtype=bool)
        for col in range(0, len(centres), chunk_size):
            end = min(col + chunk_size, len(centres))
            inside |= (_squared_distances(probes[lo:hi], centres[col:end]) <= radii[col:end]).any(1)
            if inside.all():
                break
        membership[lo:hi] = inside
    return membership


def precision_recall_from_features(
    real: np.ndarray, generated: np.ndarray, *, k: int = 3, chunk_size: int = 512,
) -> dict:
    """Improved PR using k-neighbour manifolds, memory O(chunk_size² + ND)."""
    real = _matrix(real, "real_features")
    generated = _matrix(generated, "generated_features")
    k = _positive_int(k, "pr_k")
    chunk_size = _positive_int(chunk_size, "pr_chunk_size")
    if real is None or generated is None or real.shape[1] != generated.shape[1]:
        raise ValueError("PR requires matching feature dimensions")
    if min(len(real), len(generated)) <= k:
        raise ValueError("PR requires more than k points in each set")
    real_radii = _knn_radii_squared(real, k, chunk_size)
    generated_radii = _knn_radii_squared(generated, k, chunk_size)
    precision_membership = _inside_manifold(generated, real, real_radii, chunk_size)
    recall_membership = _inside_manifold(real, generated, generated_radii, chunk_size)
    return {
        "precision": float(precision_membership.mean()),
        "recall": float(recall_membership.mean()),
        "precision_numerator": int(precision_membership.sum()),
        "recall_numerator": int(recall_membership.sum()),
        "n_real": len(real), "n_generated": len(generated),
        "k": k, "chunk_size": chunk_size,
    }


def _status(method: str, reason: str | None = None, **fields: Any) -> dict:
    return {
        "status": "unavailable" if reason else "ok", "method": method,
        **({"value": None, "reason": reason} if reason else {}), **fields,
    }


def compute_distribution_metrics(
    real_features: np.ndarray | None, generated_features: np.ndarray | None,
    generated_logits: np.ndarray | None, *, config: dict | None = None,
    real_pr_features: np.ndarray | None = None, generated_pr_features: np.ndarray | None = None,
) -> dict:
    """Compute each available metric, explaining unavailable metrics in JSON.

    Missing references are allowed (IS remains available). Malformed or
    mismatched caches are errors, rather than silently omitted metrics.
    KID subsets and IS splits are never automatically reduced.
    """
    config = dict(config or {})
    real = _matrix(real_features, "real_features")
    generated = _matrix(generated_features, "generated_features")
    logits = _matrix(generated_logits, "generated_logits")
    nr, ng = 0 if real is None else len(real), 0 if generated is None else len(generated)
    pr_backend = config.get('pr_feature_backend', 'inception')
    if pr_backend not in ('inception', 'vgg16'):
        raise ValueError('pr_feature_backend must be inception or vgg16')
    if pr_backend == 'vgg16':
        pr_real = _matrix(real_pr_features, 'real_pr_features')
        pr_gen = _matrix(generated_pr_features, 'generated_pr_features')
        if pr_real is None or pr_gen is None:
            raise ValueError('PR-VGG requires separately extracted real and generated VGG features')
        if len(pr_real) != nr or len(pr_gen) != ng or pr_real.shape[1] != pr_gen.shape[1]:
            raise ValueError('PR-VGG features must match their cohort rows and feature dimensions')
    else:
        if real_pr_features is not None or generated_pr_features is not None:
            raise ValueError('Separate PR features require the explicit vgg16 protocol')
        pr_real, pr_gen = real, generated
    if generated is not None and logits is not None and len(generated) != len(logits):
        raise ValueError("Generated feature and logit caches have different sample counts")
    if real is not None and generated is not None and real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated caches use different feature dimensions")
    seed = config.get("rng_seed", 2020)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("rng_seed must be an integer in [0, 2**32 - 1]")
    seed = int(seed)
    subset_size = _positive_int(config.get("kid_subset_size", 1000), "kid_subset_size", 2)
    subsets = _positive_int(config.get("kid_subsets", 100), "kid_subsets")
    splits = _positive_int(config.get("isc_splits", 10), "isc_splits")
    k = _positive_int(config.get("pr_k", 3), "pr_k")
    chunk = _positive_int(config.get("pr_chunk_size", 512), "pr_chunk_size")
    max_pr = config.get("pr_max_samples", 5000)
    if max_pr is not None:
        max_pr = _positive_int(max_pr, "pr_max_samples", k + 1)
    feature_metadata = config.get("feature_metadata", {})
    feature_backend = config.get('feature_backend', 'torch-fidelity')
    if feature_backend not in ('torch-fidelity', 'clean-fid'):
        raise ValueError('feature_backend must be torch-fidelity or clean-fid')
    feature_label = 'Clean-FID-0.1.35-clean' if feature_backend == 'clean-fid' else 'Inception'
    output = {"metadata": {
        "implementation": IMPLEMENTATION_VERSION,
        "reference_implementation": (f"clean-fid==0.1.35 clean (FID/KID features); torch-fidelity=={FIDELITY_VERSION} (IS)"
                                     if feature_backend == 'clean-fid' else f"torch-fidelity=={FIDELITY_VERSION} (FID/KID/IS)"),
        "features": feature_metadata,
        "n_real": nr, "n_generated": ng,
        "n_generated_logits": 0 if logits is None else len(logits),
        "feature_dimension": None if generated is None else generated.shape[1],
        "arithmetic_dtype": "float64", "rng_seed": seed,
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "small_sample_note": "Computable values below 5,000 samples are engineering/descriptive checks; not evidence of stable model ranking.",
    }}
    sizes = {"n_real": nr, "n_generated": ng}
    fid_method = f"FID-{feature_label} (sample covariance ddof=1; Gaussian Wasserstein distance)"
    if min(nr, ng) < 2:
        output["fid"] = _status(fid_method, "FID requires at least two real and two generated images; it has no single-image value.", **sizes)
    else:
        output["fid"] = _status(fid_method, value=fid_from_features(real, generated), **sizes,
                                small_sample_check=min(nr, ng) < 5000)
    kid_method = f"KID-{feature_label} (unbiased polynomial MMD²; degree=3, gamma=1/D, coef0=1)"
    if min(nr, ng) < subset_size:
        output["kid"] = _status(kid_method, f"Configured subset_size={subset_size} exceeds a collection size; no automatic reduction.", **sizes,
                                subsets=subsets, subset_size=subset_size)
    else:
        output["kid"] = _status(kid_method, **kid_from_features(real, generated, subsets=subsets, subset_size=subset_size, seed=seed),
                                **sizes, small_sample_check=min(nr, ng) < 5000)
    is_method = "Inception Score (official Inception logits_unbiased, not MMS classifier logits)"
    if logits is None or len(logits) < 2 * splits:
        output["is"] = _status(is_method, f"IS requires official logits and at least two images per configured split ({splits} splits); no single-image IS.", splits=splits)
    else:
        output["is"] = _status(is_method, **inception_score_from_logits(logits, splits=splits, seed=seed,
                                                                        shuffle=config.get("isc_shuffle", True)),
                               n_generated=len(logits), small_sample_check=len(logits) < 5000)
    pr_method = ("Improved PR-VGG (NVIDIA VGG16 features; k-neighbour balls; float64 distances; inclusive boundary)"
                 if pr_backend == 'vgg16' else
                 "Improved PR-Inception (k-neighbour balls; inclusive boundary; NOT PR-VGG)")
    if min(nr, ng) <= k:
        output["precision_recall"] = _status(pr_method, f"Both collections must contain more than k={k} images.", **sizes, k=k)
    else:
        rng = np.random.RandomState(seed)
        cap = min(nr, ng, max_pr or max(nr, ng)) if config.get('pr_balanced', False) else max_pr
        real_indices = np.arange(nr) if cap is None or nr <= cap else rng.choice(nr, cap, replace=False)
        gen_indices = np.arange(ng) if cap is None or ng <= cap else rng.choice(ng, cap, replace=False)
        output["precision_recall"] = _status(pr_method, **precision_recall_from_features(
            pr_real[real_indices], pr_gen[gen_indices], k=k, chunk_size=chunk),
            max_samples_per_collection=max_pr, selection_seed=seed,
            balanced_collections=config.get('pr_balanced', False),
            real_indices=real_indices.tolist(), generated_indices=gen_indices.tolist(),
            real_indices_sha256=hashlib.sha256(np.asarray(real_indices, dtype="<i8").tobytes()).hexdigest(),
            generated_indices_sha256=hashlib.sha256(np.asarray(gen_indices, dtype="<i8").tobytes()).hexdigest(),
            small_sample_check=min(len(real_indices), len(gen_indices)) < 5000)
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FidelityExtractor:
    """One official Inception pass per image yields 2048 features and IS logits.

    The caller should supply canonical RGB files produced by the shared image
    policy. Different native sizes are grouped within bounded loading batches,
    never padded, and resized only inside the TensorFlow-compatible extractor.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32, *,
                 weights_path: str | Path | None = None,
                 load_image: Callable | None = None):
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        version = importlib.metadata.version("torch-fidelity")
        if version != FIDELITY_VERSION:
            raise RuntimeError(f"This protocol requires torch-fidelity=={FIDELITY_VERSION}, found {version}")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.device = torch.device(device)
        self.load_image = load_image
        self.model = FeatureExtractorInceptionV3(
            INCEPTION_NAME, ["2048", "logits_unbiased"],
            feature_extractor_weights_path=None if weights_path is None else str(weights_path),
        ).eval().to(self.device)
        weights_file = Path(weights_path) if weights_path is not None else (
            Path(torch.hub.get_dir()) / "checkpoints" / INCEPTION_WEIGHTS_URL.rsplit("/", 1)[1]
        )
        if not weights_file.is_file():
            raise RuntimeError("Cannot locate the actual Inception weights file for provenance")
        import inspect
        self.metadata = {
            "backend": "torch-fidelity", "backend_version": version,
            "feature_extractor": INCEPTION_NAME, "feature_layer": "2048",
            "logit_layer": "logits_unbiased", "feature_dimension": 2048,
            "logit_dimension": 1008, "weights_url": INCEPTION_WEIGHTS_URL if weights_path is None else None,
            "weights_file": str(weights_file.resolve()), "weights_sha256": _file_sha256(weights_file),
            "extractor_source_sha256": _file_sha256(Path(inspect.getfile(FeatureExtractorInceptionV3))),
            "torch_version": torch.__version__, "torchvision_version": importlib.metadata.version("torchvision"),
            "pillow_version": importlib.metadata.version("Pillow"),
            "numpy_version": np.__version__, "device": str(self.device), "batch_size": self.batch_size,
            "inference_dtype": "float32; no autocast; TF32 disabled",
            "preprocessing": {
                "input": "canonical RGB uint8 files; EXIF transpose on reading",
                "resize": "torch-fidelity TensorFlow-1.x-compatible bilinear direct to 299x299",
                "normalization": "(x - 128) / 128",
                "external_resize": "none in this extractor; caller canonicalization must be recorded separately",
                "mixed_sizes": "group identical HxW inside bounded batches; preserve input order; no padding/crop",
                "alpha": "reject noncanonical alpha images; caller must first apply its recorded alpha policy",
            },
        }

    def extract_paths(self, paths: Sequence[str | Path]) -> dict:
        import torch
        from PIL import Image, ImageOps

        paths = list(paths)
        if not paths:
            raise ValueError("Feature extraction requires at least one image")
        features = np.empty((len(paths), 2048), dtype=np.float32)
        logits = np.empty((len(paths), 1008), dtype=np.float32)
        size_counts: dict[str, int] = {}
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode(), torch.backends.cudnn.flags(
                benchmark=False, deterministic=True, allow_tf32=False,
            ):
                for lo in range(0, len(paths), self.batch_size):
                    groups: dict[tuple, list] = {}
                    for index in range(lo, min(lo + self.batch_size, len(paths))):
                        if self.load_image is None:
                            with Image.open(paths[index]) as image:
                                if "A" in image.getbands() or "transparency" in image.info:
                                    raise ValueError(f"Noncanonical transparency in {paths[index]}; apply the shared image policy first")
                                if image.mode not in ("RGB", "L", "1", "P"):
                                    raise ValueError(f"Unsupported image mode {image.mode} in {paths[index]}; canonicalize to RGB uint8 first")
                                array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()
                        else:
                            array = np.asarray(self.load_image(paths[index]))
                        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                            raise ValueError(f"Image loader must return RGB uint8 HxWx3 for {paths[index]}")
                        key = (array.shape[0], array.shape[1])
                        groups.setdefault(key, []).append((index, array))
                        size_counts[f"{key[1]}x{key[0]}"] = size_counts.get(f"{key[1]}x{key[0]}", 0) + 1
                    for entries in groups.values():
                        indices, arrays = zip(*entries)
                        tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy()).to(self.device)
                        result = self.model(tensor)
                        features[list(indices)] = result[0].cpu().numpy()
                        logits[list(indices)] = result[1].cpu().numpy()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise FloatingPointError("Inception produced nonfinite outputs")
        return {"features": features, "logits": logits,
                "metadata": {**self.metadata, "n_images": len(paths), "input_size_counts": size_counts}}
