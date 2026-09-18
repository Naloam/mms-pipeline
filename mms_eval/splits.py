"""A single auditable split of class-folder real data for each evaluation domain."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .images import IMAGE_EXTENSIONS, collect_images, load_rgb, pixel_hash
from .utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def _inspect(arguments):
    path, root, label, category, domain, directory = arguments
    image = load_rgb(path)
    decoded = pixel_hash(image)
    relative = path.relative_to(root).as_posix()
    return {"image_id": f"{domain}:{relative}", "path": str(path.resolve()),
            "relative_path": relative, "sha256": sha256_file(path), "pixel_sha256": decoded,
            "source_id": decoded, "label": label, "class_name": category, "domain": domain,
            "source_directory": directory, "width": image.width, "height": image.height}


def prepare_real_data(root: str | Path, output_dir: str | Path, *, config: dict) -> dict:
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    if config.get('domain_contract_version'):
        from .domains import validate_domain_contract
        validate_domain_contract(config, evaluator_config=config.get('evaluator_initialization'))
    classes = config["classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Need at least two distinct mutually exclusive categories")
    if config.get("category_mode", "exclusive") != "exclusive":
        raise ValueError("Multilabel attributes cannot be used as exclusive MMS categories")
    domain = config["domain"]
    train_dir, external_dir = config.get("train_subdir", "train"), config.get("external_subdir", "val")
    if train_dir == external_dir:
        raise ValueError("Training source and external audit source must be distinct")
    arguments = []
    records = []
    if root.is_file():
        for row in collect_images(root):
            category = row.get('class_name')
            directory = row.get('source_directory')
            if category not in classes or directory not in (train_dir, external_dir):
                raise ValueError('Real manifests require declared class_name and original source_directory')
            label = classes.index(category)
            if 'label' in row and int(row['label']) != label:
                raise ValueError('Real manifest numeric label disagrees with class order')
            image = load_rgb(row['path'])
            decoded = pixel_hash(image)
            records.append({**row, 'label': label, 'domain': domain, 'pixel_sha256': decoded,
                            'source_id': decoded, 'width': image.width, 'height': image.height})
    for directory in () if root.is_file() else (train_dir, external_dir):
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(base)
        for label, category in enumerate(classes):
            category_path = base / category
            if not category_path.is_dir():
                raise FileNotFoundError(category_path)
            for p in sorted(category_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
                    arguments.append((p, root, label, category, domain, directory))
    if not arguments and not records:
        raise ValueError("No real images found")
    with ThreadPoolExecutor(max_workers=int(config.get("workers", 8))) as pool:
        records.extend(pool.map(_inspect, arguments))
    # One decoded original is one independent source. Keep official external data
    # out of training; identical bytes with different encodings are detected too.
    seen, removed = {}, []
    order = sorted(records, key=lambda r: (r["source_directory"] != external_dir, r["image_id"]))
    for row in order:
        key = row["pixel_sha256"]
        if key in seen:
            if seen[key]["label"] != row["label"]:
                raise ValueError(f"Conflicting class labels for identical content: {row['image_id']}")
            removed.append({"removed": row["image_id"], "kept": seen[key]["image_id"], "pixel_sha256": key})
        else:
            seen[key] = row
    source = sorted((r for r in seen.values() if r["source_directory"] == train_dir), key=lambda r: r["image_id"])
    external = sorted((r for r in seen.values() if r["source_directory"] == external_dir), key=lambda r: r["image_id"])
    n_val = int(config["validation_n"])
    n_pool = int(config["calibration_pool_n"])
    n_audit = int(config["internal_audit_n"])
    n_cal = int(config["calibration_n"])
    if min(n_val, n_pool, n_audit, n_cal) <= 0 or n_cal > n_pool or len(source) <= n_val+n_pool+n_audit:
        raise ValueError("Insufficient real data for the frozen split counts")
    rng = np.random.RandomState(config.get("split_seed", 2026090601))
    shuffled = [source[i] for i in rng.permutation(len(source))]
    n_fit = len(source) - n_val - n_pool - n_audit
    fit, val, reserve = shuffled[:n_fit], shuffled[n_fit:n_fit+n_val], shuffled[n_fit+n_val:]
    rng2 = np.random.RandomState(config.get("reference_seed", 2026090602))
    reserve = [reserve[i] for i in rng2.permutation(len(reserve))]
    pool, audit = reserve[:n_pool], reserve[n_pool:]
    rng3 = np.random.RandomState(config.get("calibration_seed", 2026090603))
    calibration = [pool[i] for i in rng3.permutation(len(pool))[:n_cal]]
    parts = {"fit": fit, "validation": val, "calibration_pool": pool,
             "audit_internal": audit, "audit_external": external}
    for name, rows in parts.items():
        for row in rows:
            row["split"] = name
        if set(r["label"] for r in rows) != set(range(len(classes))):
            raise ValueError(f"Split {name} lacks declared categories; revise counts before freezing")
    inventory_signature = stable_hash([{k: r[k] for k in ("image_id", "sha256", "pixel_sha256", "label")}
                                       for r in sorted(records, key=lambda x:x["image_id"])])
    signature = stable_hash({"config": config, "inventory_signature": inventory_signature})
    protocol_path = output / "split.json"
    if protocol_path.exists():
        prior = read_json(protocol_path)
        if prior["signature"] != signature:
            raise ValueError("Existing split differs from requested source/config; use a new output directory")
        for name, digest in prior["manifest_sha256"].items():
            if sha256_file(output / f"{name}.jsonl") != digest:
                raise ValueError(f"Frozen split manifest was modified: {name}")
        return prior
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in {**parts, "calibration": calibration, "quality_reference": source}.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(output / "duplicates_removed.jsonl", removed)
    manifests = {p.stem: sha256_file(p) for p in sorted(output.glob("*.jsonl"))}
    report = {"schema_version": 1, "domain": domain, "classes": classes,
              "config": config, "signature": signature, "inventory_signature": inventory_signature,
              "source_root": str(root), "counts": {k: len(v) for k,v in parts.items()},
              "calibration_n": n_cal, "original_count": len(records), "removed_duplicate_count": len(removed),
              "class_counts": {k: dict(Counter(r["class_name"] for r in v)) for k,v in parts.items()},
              "manifest_sha256": manifests,
              "calibration_sampling": "uniform without replacement; no per-class quotas",
              "external_audit": "empirical transport check; not guaranteed exchangeable with internal pool"}
    write_json(protocol_path, report)
    return report


def load_split(split_dir: str | Path, name: str) -> list[dict]:
    root = Path(split_dir)
    protocol = read_json(root / "split.json")
    path = root / f"{name}.jsonl"
    if name not in protocol["manifest_sha256"] or sha256_file(path) != protocol["manifest_sha256"][name]:
        raise ValueError(f"Unknown or modified real split: {name}")
    return read_jsonl(path)
