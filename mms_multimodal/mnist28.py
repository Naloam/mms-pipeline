"""Train and apply a 10-class MNIST evaluator without touching AFHQ references."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import struct
from pathlib import Path

import numpy as np


CLASSES = [str(i) for i in range(10)]
ARCHITECTURE = "mnist28_cnn_v1"
NORMALIZATION = "uint8_div_255_light_on_dark"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_idx_gzip(path: str | Path, *, images: bool) -> np.ndarray:
    raw = gzip.decompress(Path(path).read_bytes())
    expected_magic = 2051 if images else 2049
    if len(raw) < (16 if images else 8):
        raise ValueError(f"Truncated IDX file: {path}")
    magic, count = struct.unpack_from(">II", raw)
    if magic != expected_magic or count <= 0:
        raise ValueError(f"Unexpected IDX magic/count in {path}")
    if images:
        height, width = struct.unpack_from(">II", raw, 8)
        if (height, width) != (28, 28) or len(raw) != 16 + count * 28 * 28:
            raise ValueError(f"Expected exactly {count} 28x28 images in {path}")
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()
    if len(raw) != 8 + count:
        raise ValueError(f"Expected exactly {count} labels in {path}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if labels.max() > 9:
        raise ValueError("MNIST labels must be in 0..9")
    return labels


def model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(256, 10),
    )


def _metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    if logits.shape != (len(labels), 10) or not np.isfinite(logits).all():
        raise ValueError("Expected finite 10-class logits aligned with labels")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, prediction), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    confidence = probabilities.max(axis=1)
    correct = prediction == labels
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        chosen = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if chosen.any():
            ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    return {
        "n": int(len(labels)), "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()), "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)).mean()),
        "ece_15_bins": float(ece), "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def _evaluate_torch(net, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, device: str, batch_size: int):
    import torch

    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = torch.from_numpy(images[indices[start:start + batch_size], None].copy()).to(device=device, dtype=torch.float32)
            outputs.append(net(batch.div_(255)).cpu().numpy())
    logits = np.concatenate(outputs)
    return logits, _metrics(logits, labels[indices])


def predict_logits(checkpoint: str | Path, images: np.ndarray, *, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Return raw [N,10] logits for exact 28x28 uint8 images."""
    import torch

    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8 or not len(images):
        raise ValueError("Expected a nonempty [N,28,28] uint8 batch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["architecture"] != ARCHITECTURE or saved["normalization"] != NORMALIZATION or saved["classes"] != CLASSES:
        raise ValueError("MNIST checkpoint contract mismatch")
    net = model().to(device)
    net.load_state_dict(saved["state_dict"])
    dummy_labels = np.zeros(len(images), dtype=np.uint8)
    logits, _ = _evaluate_torch(net, images, dummy_labels, np.arange(len(images)), device, batch_size)
    return logits


def train_classifier(data_dir: str | Path, output_dir: str | Path, *, epochs: int = 12,
                     batch_size: int = 256, seed: int = 2026091706, device: str = "cuda:0") -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source = Path(data_dir).resolve()
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserve the existing evaluator output: {out}")
    files = {name: source / name for name in (
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}
    train_images = read_idx_gzip(files["train-images-idx3-ubyte.gz"], images=True)
    train_labels = read_idx_gzip(files["train-labels-idx1-ubyte.gz"], images=False)
    test_images = read_idx_gzip(files["t10k-images-idx3-ubyte.gz"], images=True)
    test_labels = read_idx_gzip(files["t10k-labels-idx1-ubyte.gz"], images=False)
    if len(train_images) != 60000 or len(test_images) != 10000 or len(train_labels) != 60000 or len(test_labels) != 10000:
        raise ValueError("Expected official MNIST 60k train and 10k test files")
    train_pixels = {hashlib.sha256(row.tobytes()).digest() for row in train_images}
    test_pixels = {hashlib.sha256(row.tobytes()).digest() for row in test_images}
    if len(train_pixels) != 60000 or len(test_pixels) != 10000 or train_pixels & test_pixels:
        raise ValueError("Exact duplicate pixels within/across train and test must be resolved before training")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(60000)
    fit_indices, validation_indices = permutation[:55000], permutation[55000:]
    test_permutation = np.random.default_rng(seed + 1).permutation(10000)
    calibration_indices, audit_indices = test_permutation[:5000], test_permutation[5000:]
    for part, indices, labels in (("fit", fit_indices, train_labels), ("validation", validation_indices, train_labels),
                                  ("calibration", calibration_indices, test_labels), ("audit", audit_indices, test_labels)):
        if set(labels[indices].tolist()) != set(range(10)):
            raise ValueError(f"{part} split lacks digit classes")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net = model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = TensorDataset(torch.from_numpy(train_images[fit_indices, None].copy()),
                            torch.from_numpy(train_labels[fit_indices].astype(np.int64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.startswith("cuda"), generator=torch.Generator().manual_seed(seed))

    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256_file(path) for name, path in files.items()}
    split = {
        "seed": seed,
        "fit_train_indices": fit_indices.tolist(), "validation_train_indices": validation_indices.tolist(),
        "calibration_test_indices": calibration_indices.tolist(), "audit_test_indices": audit_indices.tolist(),
        "train_source_sha256": {k: v for k, v in source_hashes.items() if k.startswith("train")},
        "test_source_sha256": {k: v for k, v in source_hashes.items() if k.startswith("t10k")},
    }
    (out / "split.json").write_text(json.dumps(split, separators=(",", ":")) + "\n", encoding="utf-8")
    history = []
    best_validation_nll = float("inf")
    best_epoch = None
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        total_correct = 0
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255)
            labels = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        scheduler.step()
        _, validation = _evaluate_torch(net, train_images, train_labels, validation_indices, device, batch_size)
        record = {"epoch": epoch, "fit_loss": total_loss / len(dataset), "fit_accuracy": total_correct / len(dataset),
                  "validation_accuracy": validation["accuracy"], "validation_nll": validation["nll"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if validation["nll"] < best_validation_nll:
            best_validation_nll = validation["nll"]
            best_epoch = epoch
            checkpoint = {
                "architecture": ARCHITECTURE, "normalization": NORMALIZATION, "classes": CLASSES,
                "state_dict": net.state_dict(), "seed": seed, "epoch": epoch,
                "source_sha256": source_hashes,
                "split_sha256": sha256_file(out / "split.json"),
                "validation_metrics": validation,
            }
            temporary = out / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, out / "best.pt")

    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(selected["state_dict"])
    full_logits, full_test = _evaluate_torch(net, test_images, test_labels, np.arange(10000), device, batch_size)
    calibration = _metrics(full_logits[calibration_indices], test_labels[calibration_indices])
    audit = _metrics(full_logits[audit_indices], test_labels[audit_indices])
    np.save(out / "test_logits.npy", full_logits, allow_pickle=False)
    report = {
        "status": "trained_and_tested", "architecture": ARCHITECTURE, "classes": CLASSES,
        "normalization": NORMALIZATION, "epoch_budget": epochs, "best_epoch": best_epoch,
        "fit_n": 55000, "validation_n": 5000, "official_test_n": 10000,
        "held_out_test_partition": {"calibration_n": 5000, "audit_n": 5000,
                                    "status": "reserved; no MMS threshold fitted yet"},
        "test": full_test, "calibration_half": calibration, "audit_half": audit,
        "source_sha256": source_hashes, "checkpoint_sha256": sha256_file(out / "best.pt"),
        "split_sha256": sha256_file(out / "split.json"), "test_logits_sha256": sha256_file(out / "test_logits.npy"),
        "model_selection": "minimum validation NLL on 5k training-source holdout; official test not used for selection",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
