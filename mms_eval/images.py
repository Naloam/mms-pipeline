"""Explicit image conversion, including grayscale, transparency and tensor ranges."""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .utils import read_json, read_jsonl, resolve_path, sha256_file, stable_hash, write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_POLICY_VERSION = "rgb-exif-alpha-v1"


def validate_policy(policy: dict | None) -> dict:
    result = {"exif_transpose": True, "alpha_background": [255, 255, 255],
              "canonical_size": None, "resize": "bilinear", "high_bit_depth": "reject"}
    unknown = set(policy or {}) - set(result)
    if unknown:
        raise ValueError(f"Unknown image policy fields: {sorted(unknown)}")
    result.update(policy or {})
    bg = result["alpha_background"]
    if not isinstance(bg, (tuple, list)) or len(bg) != 3 or any(
        not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 255 for v in bg
    ):
        raise ValueError("alpha_background must contain three integers in [0,255]")
    size = result["canonical_size"]
    if size is not None:
        if not isinstance(size, (tuple, list)) or len(size) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size
        ):
            raise ValueError("canonical_size must be [width,height] or null")
        result["canonical_size"] = list(size)
    if result["resize"] != "bilinear" or result["high_bit_depth"] != "reject":
        raise ValueError("Supported image policy: bilinear resize; explicit rejection of high bit depth")
    if not isinstance(result["exif_transpose"], bool):
        raise ValueError("exif_transpose must be boolean")
    return result


def _convert_pil(image: Image.Image, policy: dict) -> Image.Image:
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Multi-frame images must be exported to individual still images first")
    if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
        raise ValueError("High-bit-depth image: explicitly convert and record its range before evaluation")
    image.load()
    if policy["exif_transpose"]:
        image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, tuple(policy["alpha_background"]) + (255,))
        image = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    size = policy["canonical_size"]
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.BILINEAR)
    return image.copy()


def load_rgb(path: str | Path, policy: dict | None = None) -> Image.Image:
    """Decode one still image using the same deterministic policy on real/fake data."""
    with Image.open(path) as image:
        return _convert_pil(image, validate_policy(policy))


def as_rgb_image(image: Any, *, value_range: str | None = None,
                 layout: str = "HWC", policy: dict | None = None) -> Image.Image:
    """Accept a path, PIL image, numpy array, or tensor. Float ranges are mandatory.

    Float arrays: value_range='0_1' or '-1_1'; integers: uint8 only.
    Single images only; batch tensors use import-arrays with explicit layout.
    """
    p = validate_policy(policy)
    if isinstance(image, (str, Path)):
        return load_rgb(image, p)
    if isinstance(image, Image.Image):
        return _convert_pil(image, p)
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if layout not in {"HWC", "CHW"}:
        raise ValueError("layout must be HWC or CHW")
    if arr.ndim == 3 and layout == "CHW":
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim not in {2, 3} or min(arr.shape[:2], default=0) <= 0:
        raise ValueError("Expected one nonempty HW, HWC, or CHW image")
    if arr.ndim == 3 and arr.shape[-1] not in {1, 3, 4}:
        raise ValueError("Expected 1, 3 or 4 channels")
    if np.issubdtype(arr.dtype, np.floating):
        if value_range not in {"0_1", "-1_1"}:
            raise ValueError("Float images require an explicit value_range='0_1' or '-1_1'")
        lo = 0 if value_range == "0_1" else -1
        if not np.isfinite(arr).all() or arr.min() < lo or arr.max() > 1:
            raise ValueError("Non-finite image values or values outside declared range")
        arr = (arr - lo) / (1 - lo)
        arr = np.rint(arr * 255).astype(np.uint8)
    elif arr.dtype != np.uint8 or value_range not in {None, "uint8"}:
        raise ValueError("Integer images must be uint8 with range [0,255]")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return _convert_pil(Image.fromarray(arr), p)


def pixel_hash(image: Image.Image) -> str:
    d = hashlib.sha256()
    d.update(f"RGB:{image.width}x{image.height}:".encode())
    d.update(image.tobytes())
    return d.hexdigest()


def save_png_atomic(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".image-", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def collect_images(source: str | Path | list[dict] | list[str]) -> list[dict]:
    """Resolve a file, directory, JSONL/CSV manifest, or explicit list deterministically."""
    if isinstance(source, list):
        raw = [dict(x) if isinstance(x, dict) else {"path": str(x)} for x in source]
        base = Path.cwd()
    else:
        source = Path(source).expanduser().resolve()
        base = source.parent
        if source.is_dir():
            base = source
            raw = [{"path": str(x)} for x in sorted(source.rglob("*"))
                   if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS and not x.name.startswith(".")]
        elif source.suffix.lower() == ".jsonl":
            raw = read_jsonl(source)
        elif source.suffix.lower() == ".csv":
            with source.open(encoding="utf-8-sig", newline="") as stream:
                raw = list(csv.DictReader(stream))
        elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            raw = [{"path": str(source)}]
        else:
            raise ValueError(f"Expected image, directory, CSV or JSONL manifest: {source}")
    if not raw:
        raise ValueError("Image collection is empty")
    result, ids = [], set()
    for row in raw:
        if row.get("status") in {"failed", "missing"}:
            raise ValueError("Input manifest contains failed/missing records; repair or explicitly version the cohort")
        value = row.get("path") or row.get("relative_path")
        if not value:
            raise ValueError("Each manifest row needs path or relative_path")
        path = resolve_path(value, base)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if row.get("sha256") and digest != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        image_id = str(row.get("image_id") or stable_hash({"path": str(value), "sha256": digest})[:24])
        if image_id in ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        ids.add(image_id)
        result.append({**row, "image_id": image_id, "path": str(path), "sha256": digest})
    return result


def canonicalize_records(records: list[dict], cache_dir: str | Path,
                         policy: dict | None = None) -> list[dict]:
    """Keep source identity; materialize explicit RGB inputs for all feature extractors."""
    p = validate_policy(policy)
    target = Path(cache_dir)
    result = []
    for row in records:
        key = stable_hash({"source_sha256": row["sha256"], "policy": p,
                           "policy_version": IMAGE_POLICY_VERSION})
        out = target / key[:2] / f"{key}.png"
        integrity = out.with_suffix('.json')
        if not out.exists():
            save_png_atomic(load_rgb(row["path"], p), out)
            write_json(integrity, {'sha256': sha256_file(out)})
        elif not integrity.exists():
            # Migrate older caches only after verifying decoded content against source.
            if pixel_hash(load_rgb(out)) != pixel_hash(load_rgb(row['path'], p)):
                raise ValueError(f'Canonical image cache changed: {out}')
            write_json(integrity, {'sha256': sha256_file(out)})
        elif sha256_file(out) != read_json(integrity)['sha256']:
            raise ValueError(f'Canonical image cache changed: {out}')
        # A corrupt cache never silently changes an input; decoding catches partial writes.
        with Image.open(out) as check:
            if check.mode != "RGB":
                raise ValueError(f"Non-RGB canonical cache: {out}")
            width, height = check.size
            check.verify()
        result.append({**row, "source_path": row["path"], "path": str(out.resolve()),
                       "canonical_key": key, "canonical_sha256": sha256_file(out),
                       "width": width, "height": height})
    return result
