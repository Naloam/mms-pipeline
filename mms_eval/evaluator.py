"""Train and apply a frozen semantic evaluator for MMS.

Only real ``Rfit``/``Rval`` records belong in :func:`train_evaluator`. This
module intentionally has no access to calibration or generated-image labels.
Returned predictions are raw logits: temperature and MMS calibration live in
the scoring layer. PyTorch is imported lazily so manifest/scoring commands can
run without the model-training extra installed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence
import uuid


SCHEMA_VERSION = 1
PREPROCESSING = {
    "resize": [224, 224],
    "resize_mode": "whole_image_stretch",
    "interpolation": "PIL_BILINEAR",
    "crop": None,
    "channel_order": "RGB",
    "scale": "uint8_to_float32_divide_255",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
_DEFAULT_WEIGHTS = {"resnet50": "IMAGENET1K_V2", "vit_b_16": "IMAGENET1K_V1",
                    "efficientnet_b0": "IMAGENET1K_V1"}
_ALLOWED_WEIGHTS = {
    "resnet50": {None, "IMAGENET1K_V1", "IMAGENET1K_V2"},
    "vit_b_16": {None, "IMAGENET1K_V1"},
    "efficientnet_b0": {None, "IMAGENET1K_V1"},
}
_CONFIG_DEFAULTS = {
    "architecture": "resnet50",
    "epochs": 20,
    "micro_batch_size": 8,
    "effective_batch_size": 64,
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 0.01,
    "seed": 0,
    "horizontal_flip_probability": 0.0,
    "deterministic": True,
    "image_policy": {},
}
_OPERATIONAL_KEYS = {"device", "num_workers", "resume", "epochs_per_call"}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Evaluator training/inference requires PyTorch and torchvision.") from exc
    return torch


def _normalise_config(config: Mapping[str, Any]) -> tuple[dict, dict]:
    unknown = set(config) - set(_CONFIG_DEFAULTS) - _OPERATIONAL_KEYS - {"weights", "classes"}
    if unknown:
        raise ValueError(f"Unknown evaluator config keys: {sorted(unknown)}")
    protocol = {**_CONFIG_DEFAULTS, **{k: v for k, v in config.items() if k not in _OPERATIONAL_KEYS}}
    architecture = protocol["architecture"]
    if architecture not in _DEFAULT_WEIGHTS:
        raise ValueError("architecture must be resnet50, vit_b_16 or efficientnet_b0")
    protocol.setdefault("weights", _DEFAULT_WEIGHTS[architecture])
    if protocol["weights"] not in _ALLOWED_WEIGHTS[architecture]:
        raise ValueError(f"Use an explicit supported weight ID for {architecture}; DEFAULT is forbidden.")
    classes = protocol.get("classes")
    if not isinstance(classes, (tuple, list)) or len(classes) < 2:
        raise ValueError("classes must list at least two mutually exclusive class names in label order")
    if any(not isinstance(x, str) or not x.strip() for x in classes) or len(set(classes)) != len(classes):
        raise ValueError("classes must contain distinct nonempty strings")
    protocol["classes"] = list(classes)
    for name in ("epochs", "micro_batch_size", "effective_batch_size"):
        if not isinstance(protocol[name], int) or isinstance(protocol[name], bool) or protocol[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if protocol["micro_batch_size"] > protocol["effective_batch_size"]:
        raise ValueError("micro_batch_size cannot exceed effective_batch_size")
    for name in ("backbone_lr", "head_lr", "weight_decay"):
        if not isinstance(protocol[name], (float, int)) or not math.isfinite(protocol[name]):
            raise ValueError(f"{name} must be finite")
        if protocol[name] < 0 or (name != "weight_decay" and protocol[name] == 0):
            raise ValueError(f"Invalid {name}")
    if not isinstance(protocol["seed"], int) or not 0 <= protocol["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    flip = protocol["horizontal_flip_probability"]
    if not isinstance(flip, (float, int)) or not 0 <= flip <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    if not isinstance(protocol["deterministic"], bool):
        raise ValueError("deterministic must be a bool")
    if not isinstance(protocol["image_policy"], dict):
        raise ValueError("image_policy must be an object")
    # Copy nested values and reject nonserializable/NaN policy values before I/O.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    operational = {
        "device": config.get("device", "cpu"),
        "num_workers": config.get("num_workers", 0),
        "resume": config.get("resume", True),
        "epochs_per_call": config.get("epochs_per_call"),
    }
    if not isinstance(operational["num_workers"], int) or operational["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not isinstance(operational["resume"], bool):
        raise ValueError("resume must be a bool")
    cap = operational["epochs_per_call"]
    if cap is not None and (not isinstance(cap, int) or cap <= 0):
        raise ValueError("epochs_per_call must be a positive integer or null")
    return protocol, operational


def _validate_records(records: Sequence[Mapping[str, Any]], classes: list[str], split: str):
    if not records:
        raise ValueError(f"{split} records cannot be empty")
    normalised, seen_ids, seen_paths, seen_content = [], set(), set(), set()
    for record in records:
        if not all(key in record for key in ("path", "label", "image_id")):
            raise ValueError(f"{split} records require path, label, image_id")
        label, image_id = record["label"], str(record["image_id"])
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(classes):
            raise ValueError(f"Invalid class label in {split}: {label!r}")
        if not image_id:
            raise ValueError("image_id cannot be empty")
        path = Path(record["path"]).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Not an image file: {path}")
        content_hash = _file_hash(path)
        if image_id in seen_ids or str(path) in seen_paths or content_hash in seen_content:
            raise ValueError(f"Duplicate image ID, path, or file content within {split}: {image_id}")
        if record.get("sha256") and record["sha256"] != content_hash:
            raise ValueError(f"Content hash does not match manifest for {image_id}")
        seen_ids.add(image_id)
        seen_paths.add(str(path))
        seen_content.add(content_hash)
        normalised.append({"path": str(path), "label": label, "image_id": image_id, "sha256": content_hash})
    normalised.sort(key=lambda row: row["image_id"])
    if {row["label"] for row in normalised} != set(range(len(classes))):
        raise ValueError(f"{split} must contain all declared classes")
    # Absolute roots can change when an AutoDL image is cloned. Data identity does not.
    signature = _json_hash([{k: row[k] for k in ("image_id", "label", "sha256")} for row in normalised])
    return normalised, signature


def _prepare_data(train_records, val_records, classes):
    train, train_hash = _validate_records(train_records, classes, "Rfit")
    val, val_hash = _validate_records(val_records, classes, "Rval")
    for key in ("image_id", "path", "sha256"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"Rfit and Rval overlap by {key}; fix the split before training")
    return train, val, {"Rfit": train_hash, "Rval": val_hash}


def _resolve_device(name: str):
    torch = _torch()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {name}, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested mps, but Metal is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Evaluator device must be cpu, mps, or cuda[:index]")
    return device


def _build_model(architecture: str, weights: str | None, num_classes: int):
    torch = _torch()
    from torchvision import models

    if architecture == "resnet50":
        pretrained = None if weights is None else getattr(models.ResNet50_Weights, weights)
        model = models.resnet50(weights=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif architecture == "vit_b_16":
        pretrained = None if weights is None else getattr(models.ViT_B_16_Weights, weights)
        model = models.vit_b_16(weights=pretrained)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        head = model.heads.head
    elif architecture == "efficientnet_b0":
        pretrained = None if weights is None else getattr(models.EfficientNet_B0_Weights, weights)
        model = models.efficientnet_b0(weights=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier[1]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model, head


class _PathDataset:
    def __init__(self, records, image_policy, flip_probability=0.0):
        self.records = records
        self.image_policy = image_policy
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        from .images import load_rgb

        torch = _torch()
        record = self.records[index]
        image = load_rgb(record["path"], policy=self.image_policy)
        image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        if self.flip_probability and torch.rand(()).item() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor(PREPROCESSING["mean"]).view(3, 1, 1)
        std = tensor.new_tensor(PREPROCESSING["std"]).view(3, 1, 1)
        return (tensor - mean) / std, record.get("label", -1)


def _seed_worker(worker_id):
    import numpy as np

    seed = _torch().initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _seed_everything(seed, deterministic):
    import numpy as np

    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def _random_state():
    import numpy as np

    torch = _torch()
    np_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_random_state(state):
    import numpy as np

    torch = _torch()
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def _atomic_torch_save(path: Path, value: dict):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _torch().save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _directory_lock(path: Path):
    import fcntl

    with open(path / ".training.lock", "a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is training in {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load_checkpoint(path):
    # Our checkpoint contains primitive metadata and tensors only, including RNG states.
    checkpoint = _torch().load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluator checkpoint schema")
    if "model_state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("Incomplete evaluator checkpoint")
    metadata = checkpoint["metadata"]
    if metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Checkpoint preprocessing does not match this evaluator implementation")
    protocol, _ = _normalise_config(metadata["config"])
    if metadata.get("config_sha256") != _json_hash(protocol):
        raise ValueError("Checkpoint config hash is invalid")
    if metadata.get("classes") != protocol["classes"] or metadata.get("image_policy") != protocol["image_policy"]:
        raise ValueError("Checkpoint class/image metadata is inconsistent")
    return checkpoint


def describe_checkpoint(checkpoint_path: str | Path) -> dict:
    """Read metadata and the hash of the exact checkpoint bytes to identify caches."""
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = _load_checkpoint(path)
    return {
        **checkpoint["metadata"],
        "checkpoint_path": str(path),
        "checkpoint_sha256": _file_hash(path),
        "epoch": checkpoint["epoch"],
        "val_nll": checkpoint["val_nll"],
    }


def train_evaluator(train_records, val_records, output_dir, *, config: dict) -> dict:
    """Fine-tune on Rfit and choose an epoch using Rval mean NLL only.

    ``classes`` is required and its order is the label mapping. Classes must be
    mutually exclusive; CelebA's independent attribute vector is not a softmax
    class label. Operational keys ``device``, ``num_workers``, ``resume`` and
    ``epochs_per_call`` do not change the protocol fingerprint. The latter can
    pause cleanly at an epoch boundary. Interrupted partial epochs are rerun;
    best.pt is for frozen inference and last.pt is the resumable epoch state.
    All records are content-hashed, deduplicated and checked for split overlap.
    """
    protocol, operational = _normalise_config(config)
    train, val, data_hashes = _prepare_data(train_records, val_records, protocol["classes"])
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config": protocol,
        "config_sha256": _json_hash(protocol),
        "data_sha256": data_hashes,
    }
    # This must be set before CUDA BLAS is initialised by the process.
    if protocol["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with _directory_lock(output):
        contract_path = output / "training_config.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text(encoding="utf-8"))
            if previous != contract:
                raise ValueError("Existing evaluator has different config/data; use a new output directory")
            if not operational["resume"]:
                raise FileExistsError("Existing evaluator run; resume it or use a new output directory")
        elif any((output / name).exists() for name in ("best.pt", "last.pt", "metadata.json", "training_log.json")):
            raise ValueError("Existing untracked evaluator artifacts; use a new output directory")
        else:
            _atomic_json(contract_path, contract)

        torch = _torch()
        import torchvision
        import numpy as np
        import PIL

        device = _resolve_device(operational["device"])
        _seed_everything(protocol["seed"], protocol["deterministic"])
        last_path, best_path = output / "last.pt", output / "best.pt"
        previous_state = _load_checkpoint(last_path) if last_path.exists() else None
        if previous_state and (
            previous_state["metadata"]["config_sha256"] != contract["config_sha256"]
            or previous_state["metadata"]["data_sha256"] != data_hashes
        ):
            raise ValueError("Resume checkpoint does not match config/data contract")
        model, head = _build_model(
            protocol["architecture"], None if previous_state else protocol["weights"], len(protocol["classes"])
        )
        model.to(device)
        head_ids = {id(parameter) for parameter in head.parameters()}
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": protocol["backbone_lr"]},
                {"params": list(head.parameters()), "lr": protocol["head_lr"]},
            ],
            weight_decay=protocol["weight_decay"],
        )
        metadata = {
            **contract,
            "classes": protocol["classes"],
            "architecture": protocol["architecture"],
            "pretrained_weights": protocol["weights"],
            "pretrained_weight_id": (
                f"torchvision.models.{protocol['architecture']}.{protocol['weights']}"
                if protocol["weights"] else None
            ),
            "preprocessing": PREPROCESSING,
            "image_policy": protocol["image_policy"],
            "selection": "minimum_Rval_mean_negative_log_likelihood",
            "temperature_applied": False,
            "train_count": len(train),
            "val_count": len(val),
            "seen_images": [{"image_id": r["image_id"], "sha256": r["sha256"]}
                            for r in train + val],
            "software": {
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "numpy": str(np.__version__), "Pillow": str(PIL.__version__),
            },
            "training_device": str(device),
            "resumed_device_change": False,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if previous_state:
            metadata["pretrained_weights_sha256"] = previous_state["metadata"].get("pretrained_weights_sha256")
        elif protocol["weights"]:
            from torchvision import models
            enum_class = {"resnet50": models.ResNet50_Weights,
                          "vit_b_16": models.ViT_B_16_Weights,
                          "efficientnet_b0": models.EfficientNet_B0_Weights}[protocol["architecture"]]
            weight_name = getattr(enum_class, protocol["weights"]).url.rsplit("/", 1)[1]
            weight_file = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
            metadata["pretrained_weights_sha256"] = _file_hash(weight_file) if weight_file.exists() else None
        else:
            metadata["pretrained_weights_sha256"] = None
        history, start_epoch, best_nll, best_epoch = [], 0, math.inf, None
        if previous_state:
            old_metadata = previous_state["metadata"]
            if old_metadata["software"] != metadata["software"]:
                raise ValueError("Resume software versions changed; restore the recorded environment")
            metadata["resumed_device_change"] = (
                old_metadata.get("resumed_device_change", False)
                or old_metadata["training_device"] != str(device)
            )
            metadata["initial_training_device"] = old_metadata.get("initial_training_device", old_metadata["training_device"])
            model.load_state_dict(previous_state["model_state_dict"], strict=True)
            optimizer.load_state_dict(previous_state["optimizer_state_dict"])
            history = previous_state["history"]
            start_epoch = previous_state["epoch"]
            best_nll, best_epoch = previous_state["best_val_nll"], previous_state["best_epoch"]
            _restore_random_state(previous_state["random_state"])
            if not best_path.exists():
                raise ValueError("Resume checkpoint references a missing best.pt")
            best_state = _load_checkpoint(best_path)
            if best_state["epoch"] != best_epoch or best_state["val_nll"] != best_nll:
                raise ValueError("best.pt and last.pt disagree; restore a consistent saved run")
        train_dataset = _PathDataset(train, protocol["image_policy"], protocol["horizontal_flip_probability"])
        val_dataset = _PathDataset(val, protocol["image_policy"])
        micro = protocol["micro_batch_size"]
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=micro, shuffle=False, num_workers=operational["num_workers"],
            worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(protocol["seed"]),
        )
        end_epoch = protocol["epochs"]
        if operational["epochs_per_call"] is not None:
            end_epoch = min(end_epoch, start_epoch + operational["epochs_per_call"])
        for epoch in range(start_epoch, end_epoch):
            began = time.monotonic()
            # Whole effective batches stay on CPU; only micro batches enter GPU memory.
            # Loss sums are divided by the actual group length, including the final group.
            loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=protocol["effective_batch_size"], shuffle=True,
                num_workers=operational["num_workers"], worker_init_fn=_seed_worker,
                generator=torch.Generator().manual_seed(protocol["seed"] + epoch),
            )
            model.train()
            train_loss, train_correct, train_count = 0.0, 0, 0
            for images, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                group_length = len(labels)
                for begin in range(0, group_length, micro):
                    x, y = images[begin:begin + micro].to(device), labels[begin:begin + micro].to(device)
                    logits = model(x)
                    loss_sum = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
                    if not torch.isfinite(loss_sum):
                        raise FloatingPointError("Non-finite training loss; last completed epoch is preserved")
                    (loss_sum / group_length).backward()
                    train_loss += loss_sum.detach().item()
                    train_correct += (logits.detach().argmax(1) == y).sum().item()
                    train_count += len(y)
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite gradient; last completed epoch is preserved")
                optimizer.step()
            model.eval()
            val_loss, val_correct, val_count = 0.0, 0, 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    logits = model(images.to(device))
                    labels = labels.to(device)
                    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite validation output; do not use this epoch")
                    val_loss += loss.item()
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_count += len(labels)
            val_nll = val_loss / val_count
            row = {
                "epoch": epoch + 1, "train_nll": train_loss / train_count,
                "train_accuracy": train_correct / train_count,
                "val_nll": val_nll, "val_accuracy": val_correct / val_count,
                "train_count": train_count, "val_count": val_count,
                "elapsed_seconds": time.monotonic() - began,
            }
            history.append(row)
            is_best = val_nll < best_nll
            if is_best:
                best_nll, best_epoch = val_nll, epoch + 1
            checkpoint = {
                "schema_version": SCHEMA_VERSION, "metadata": metadata,
                "epoch": epoch + 1, "val_nll": val_nll,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            if is_best:
                _atomic_torch_save(best_path, checkpoint)
            _atomic_torch_save(last_path, {
                **checkpoint, "optimizer_state_dict": optimizer.state_dict(),
                "best_val_nll": best_nll, "best_epoch": best_epoch,
                "history": history, "random_state": _random_state(),
            })
            _atomic_json(output / "training_log.json", history)
            _atomic_json(output / "metadata.json", {
                **metadata, "best_epoch": best_epoch, "best_val_nll": best_nll,
                "best_checkpoint_sha256": _file_hash(best_path),
                "completed_epochs": epoch + 1,
            })
            print(json.dumps({"event": "evaluator_epoch", **row}), flush=True)
        completed = history[-1]["epoch"] if history else start_epoch
        if not best_path.exists():
            raise RuntimeError("No completed evaluator checkpoint was produced")
        return {
            "status": "complete" if completed >= protocol["epochs"] else "paused",
            "checkpoint_path": str(best_path), "last_checkpoint_path": str(last_path),
            "metadata_path": str(output / "metadata.json"),
            "log_path": str(output / "training_log.json"),
            "completed_epochs": completed, "best_epoch": best_epoch,
            "best_val_nll": best_nll,
            "metadata": describe_checkpoint(best_path),
        }


class PredictionSession:
    """Keep one frozen evaluator on its device across bounded feature chunks."""

    def __init__(self):
        self.identity = None
        self.metadata = None
        self.model = None

    def prepare(self, checkpoint_path, device):
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
        identity = (str(path), str(device))
        if self.identity is not None and self.identity != identity:
            raise ValueError('Prediction session belongs to a different checkpoint or device')
        if self.identity is None:
            state = _load_checkpoint(path)
            metadata = {
                **state['metadata'], 'checkpoint_path': str(path),
                'checkpoint_sha256': _file_hash(path), 'epoch': state['epoch'],
                'val_nll': state['val_nll'],
                'inference_contract': 'float32; no autocast; matmul and cuDNN TF32 disabled; deterministic cuDNN',
            }
            resolved = _resolve_device(device)
            model, _ = _build_model(metadata['architecture'], None, len(metadata['classes']))
            model.load_state_dict(state['model_state_dict'], strict=True)
            model.to(resolved).eval().requires_grad_(False)
            self.identity, self.metadata, self.model = identity, metadata, model
            self.device = resolved
        return self.model, self.metadata, self.device


def predict_paths(paths, checkpoint_path, *, device="cpu", batch_size=32, session=None) -> dict:
    """Return finite N×C raw logits in exactly the caller's path order.

    This reconstructs the checkpoint without fetching pretraining weights.
    Empty input returns an empty (0, C) float32 array. No threshold, temperature,
    softmax, image rejection or sample filtering is silently applied.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    else:
        paths = list(paths)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    import numpy as np

    session = session or PredictionSession()
    model, metadata, resolved_device = session.prepare(checkpoint_path, device)
    num_classes = len(metadata["classes"])
    if not paths:
        return {"logits": np.empty((0, num_classes), dtype=np.float32), "metadata": metadata}
    torch = _torch()
    dataset = _PathDataset([{"path": str(x)} for x in paths], metadata["image_policy"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.backends.cudnn.flags(
            benchmark=False, deterministic=True, allow_tf32=False,
        ), torch.autocast(device_type=resolved_device.type, enabled=False):
            for images, _ in loader:
                logits = model(images.to(resolved_device))
                if logits.ndim != 2 or logits.shape[1] != num_classes or not torch.isfinite(logits).all():
                    raise FloatingPointError("Evaluator returned invalid/non-finite logits")
                batches.append(logits.cpu().to(torch.float32).numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return {"logits": np.concatenate(batches, axis=0), "metadata": metadata}
