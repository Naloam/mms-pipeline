"""Reproducible, resumable sampling of the existing unconditional ADM checkpoint."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
import time

from .images import save_png_atomic
from .utils import read_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl

AFHQ_500K_SHA256 = "fcd8fdda3c25bd3cfdc047255d9c818939167a768f3a562760fbf01bb64efc0a"
ADM_CONFIG = {
    "image_size": 256, "class_cond": False, "diffusion_steps": 1000,
    "noise_schedule": "linear", "learn_sigma": True, "num_channels": 256,
    "num_res_blocks": 2, "num_head_channels": 64, "attention_resolutions": "32,16,8",
    "resblock_updown": True, "use_scale_shift_norm": True, "use_fp16": True,
}


@contextlib.contextmanager
def _lock(output: Path):
    import fcntl
    with (output / ".sampling.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sample_guided(checkpoint: str | Path, repository: str | Path, output_dir: str | Path, *,
                  count: int, seed_start: int = 10000, sampler: str = "ddim", steps: int = 50,
                  device: str = "cuda:1", batch_size: int = 4, role: str = "engineering",
                  setting_id: str | None = None, expected_sha256: str = AFHQ_500K_SHA256) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    if sampler not in {"ddpm", "ddim"} or count < 1 or batch_size < 1 or seed_start < 0:
        raise ValueError("Invalid sampler, count, batch size or starting seed")
    if steps < 1 or steps > 1000 or (sampler == "ddpm" and steps < 2):
        raise ValueError("Steps must be valid for the 1000-step training schedule")
    path, repo, out = Path(checkpoint).resolve(), Path(repository).resolve(), Path(output_dir).resolve()
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError("Generator checkpoint checksum differs from the explicitly registered weight")
    code_hashes = {f: sha256_file(repo / "guided_diffusion" / f) for f in
                   ("script_util.py", "unet.py", "gaussian_diffusion.py", "respace.py", "nn.py")}
    setting_id = setting_id or f"AF_500k_{sampler.upper()}{steps}"
    contract = {"sampler_version": "per-image-cpu-rng-v1", "checkpoint_sha256": actual_sha,
                "sampler": sampler, "steps": steps, "ddim_eta": 0.0 if sampler == "ddim" else None,
                "model": ADM_CONFIG, "seed_start": seed_start, "role": role,
                "setting_id": setting_id, "clip_denoised": True,
                "quantization": "floor(clamp((sample+1)*127.5,0,255)) to RGB uint8 PNG",
                "initial_noise": "separate CPU torch.Generator per image; seed is saved",
                "code_sha256": code_hashes, "torch_version": str(torch.__version__)}
    signature = stable_hash(contract)
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out):
        metadata_path = out / "sampling.json"
        manifest_path = out / "manifest.jsonl"
        if metadata_path.exists():
            prior = read_json(metadata_path)
            if prior["signature"] != signature:
                raise ValueError("Sampling directory belongs to a different generator/seed/config")
        elif manifest_path.exists():
            raise ValueError("Untracked existing sample manifest")
        existing = read_jsonl(manifest_path) if manifest_path.exists() else []
        by_seed = {}
        for row in existing:
            if row["seed"] in by_seed:
                raise ValueError("Duplicate seed in sample manifest")
            image_path = out / row["relative_path"]
            if not image_path.exists() or sha256_file(image_path) != row["sha256"]:
                raise ValueError(f"Existing sample is missing or changed: {image_path}")
            by_seed[row["seed"]] = row
        target = list(range(seed_start, seed_start + count))
        if set(by_seed) - set(target):
            raise ValueError("Cannot silently shrink an existing sample cohort")
        pending = [seed for seed in target if seed not in by_seed]
        if not pending:
            return {"status": "complete", "n": len(existing), "manifest": str(manifest_path),
                    "resumed": True, "signature": signature}
        sys.path.insert(0, str(repo))
        from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

        flags = model_and_diffusion_defaults()
        flags.update(ADM_CONFIG)
        flags["timestep_respacing"] = f"ddim{steps}" if sampler == "ddim" else str(steps)
        model, diffusion = create_model_and_diffusion(**flags)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device)
        model.convert_to_fp16()
        model.eval().requires_grad_(False)
        # Batch-dependent floating point rounding can remain; noise identities never change.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        metadata = {**contract, "signature": signature, "repository": str(repo),
                    "weight_path": str(path), "device": device, "batch_size": batch_size,
                    "requested_count": count, "original_timestep_map": list(diffusion.timestep_map),
                    "original_timestep_execution_order": list(reversed(diffusion.timestep_map)),
                    "status": "running"}
        write_json(metadata_path, metadata)
        start = time.monotonic()
        new_count = 0
        for lo in range(0, len(pending), batch_size):
            seeds = pending[lo:lo+batch_size]
            generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
            shape = (3, ADM_CONFIG["image_size"], ADM_CONFIG["image_size"])
            x = torch.stack([torch.randn(shape, generator=g) for g in generators]).to(device)
            batch_start = time.monotonic()
            with torch.inference_mode():
                for index in range(diffusion.num_timesteps - 1, -1, -1):
                    t = torch.full((len(seeds),), index, device=device, dtype=torch.long)
                    if sampler == "ddim":
                        x = diffusion.ddim_sample(model, x, t, clip_denoised=True, eta=0.)["sample"]
                    else:
                        result = diffusion.p_mean_variance(model, x, t, clip_denoised=True)
                        noise = torch.stack([torch.randn(shape, generator=g) for g in generators]).to(device)
                        x = result["mean"] + (index != 0) * torch.exp(.5 * result["log_variance"]) * noise
                if not torch.isfinite(x).all():
                    raise FloatingPointError("Nonfinite generated image; seeds retained for repair")
                array = ((x + 1) * 127.5).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
            elapsed = time.monotonic() - batch_start
            for seed, image_array in zip(seeds, array):
                image_path = out / "images" / f"{seed:010d}.png"
                save_png_atomic(Image.fromarray(image_array), image_path)
                row = {"image_id": f"{setting_id}:{role}:{seed}", "setting_id": setting_id,
                       "seed": seed, "role": role, "pair_id": f"AF500k:{role}:{seed}",
                       "relative_path": image_path.relative_to(out).as_posix(),
                       "sha256": sha256_file(image_path), "checkpoint_sha256": actual_sha,
                       "config_sha256": signature, "status": "complete", "batch_seconds": elapsed}
                by_seed[seed] = row
            write_jsonl(manifest_path, [by_seed[seed] for seed in sorted(by_seed)])
            new_count += len(seeds)
            print(f"sampled {len(by_seed)}/{count}; batch_seconds={elapsed:.2f}; "
                  f"new_images_per_second={new_count/(time.monotonic()-start):.4f}", flush=True)
        metadata.update(status="complete", n=len(by_seed), elapsed_seconds_this_call=time.monotonic()-start,
                        new_images_this_call=new_count, manifest_sha256=sha256_file(manifest_path))
        write_json(metadata_path, metadata)
        return {"status": "complete", "n": len(by_seed), "manifest": str(manifest_path),
                "signature": signature, "elapsed_seconds_this_call": metadata["elapsed_seconds_this_call"]}
