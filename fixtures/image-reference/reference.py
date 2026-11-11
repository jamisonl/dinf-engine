#!/usr/bin/env python3
"""Pinned same-input Image2.1 component producer; no text encoder or image-quality claim."""

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from diffusers import (
    QwenImage21Transformer2DModel,
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21KVCache,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_input(base, spec, device, dtype):
    name = spec["file"]
    assert Path(name).name == name
    raw = (base / name).read_bytes()
    assert sha(raw) == spec["sha256"]
    v = np.frombuffer(raw, dtype="<f4").copy().reshape(spec["shape"])
    assert np.isfinite(v).all()
    return torch.from_numpy(v).to(device=device, dtype=dtype)


def save(out, name, tensor):
    v = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    assert np.isfinite(v).all(), name
    raw = v.astype("<f4", copy=False).tobytes()
    file = name + ".f32"
    with (out / file).open("xb") as f:
        f.write(raw)
    return dict(file=file, shape=list(v.shape), sha256=sha(raw), finite=True)


def loaded(cls, path, dtype, device):
    model, info = cls.from_pretrained(
        path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    assert not any(
        info.get(k)
        for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ), info
    model = model.eval().to(device)
    assert all(
        p.device.type == device.type
        and (device.index is None or p.device.index == device.index)
        and p.dtype == dtype
        for p in model.parameters()
    )
    return model, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument(
        "mode",
        choices=[
            "schedule",
            "denoise_single",
            "denoise_cache",
            "denoise_uncached",
            "vae",
        ],
    )
    p.add_argument("backend", choices=["cpu", "mps"])
    p.add_argument("dtype", choices=["f32", "bf16"])
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert a.backend != "cpu" or a.dtype == "f32"
    assert torch.__version__.startswith("2.15.0")
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    device = torch.device(a.backend)
    spec = json.loads(a.fixture.read_text())
    assert spec["schema_version"] == 1 and spec["steps"] == 4
    a.output.mkdir()
    outputs = {}
    load_info = {}
    schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
        a.snapshot / "scheduler", local_files_only=True
    )
    slope = (schedule.config.max_shift - schedule.config.base_shift) / (
        schedule.config.max_image_seq_len - schedule.config.base_image_seq_len
    )
    mu = (
        1024 * slope
        + schedule.config.base_shift
        - slope * schedule.config.base_image_seq_len
    )
    schedule.set_timesteps(4, device=device, sigmas=np.linspace(1.0, 1 / 4, 4), mu=mu)
    outputs["sigmas"] = save(a.output, "sigmas", schedule.sigmas)
    outputs["timesteps"] = save(a.output, "timesteps", schedule.timesteps)
    if a.mode != "schedule":
        latent = load_input(a.fixture.parent, spec["latents"], device, dtype)
        assert list(latent.shape) == [1, 1024, 64]
        outputs["input_latents"] = save(a.output, "input_latents", latent)
        if a.mode == "vae":
            model, load_info = loaded(
                AutoencoderKLQwenImage21, a.snapshot / "vae", dtype, device
            )
            shape = latent.transpose(1, 2).reshape(1, 64, 1, 32, 32)
            mean = torch.tensor(
                model.config.latents_mean, device=device, dtype=dtype
            ).reshape(1, 64, 1, 1, 1)
            std = torch.tensor(
                model.config.latents_std, device=device, dtype=dtype
            ).reshape(1, 64, 1, 1, 1)
            raw = shape * std + mean
            outputs["denormalized_latents"] = save(
                a.output, "denormalized_latents", raw.squeeze(2)
            )
            decoded = model.decode(raw, return_dict=False)[0]
            assert list(decoded.shape) == [1, 4, 1, 512, 512]
            outputs["decoded"] = save(a.output, "decoded", decoded.squeeze(2))
        else:
            cond = load_input(a.fixture.parent, spec["conditioning"], device, dtype)
            assert list(cond.shape) == [1, 32, 4096]
            outputs["input_conditioning"] = save(a.output, "input_conditioning", cond)
            model, load_info = loaded(
                QwenImage21Transformer2DModel, a.snapshot / "transformer", dtype, device
            )
            model.set_attn_processor(QwenImage21AttnProcessor())
            cache = (
                QwenImage21KVCache(len(model.transformer_blocks))
                if a.mode == "denoise_cache"
                else None
            )
            mask = torch.tensor(
                [[False] * 32 + [True] * 256], device=device, dtype=torch.bool
            )
            schedule.set_begin_index(0)
            for i, t in enumerate(
                schedule.timesteps[:1]
                if a.mode == "denoise_single"
                else schedule.timesteps
            ):
                cm = ("extract" if i == 0 else "cached") if cache else None
                pred = model(
                    hidden_states=latent,
                    encoder_hidden_states=cond,
                    timestep=t.expand(1).to(dtype) / 1000,
                    img_shapes=[[(1, 32, 32)]],
                    img_mask=mask,
                    kv_cache=cache,
                    kv_cache_mode=cm,
                    return_dict=False,
                )[0][:, -1024:]
                outputs[f"prediction_{i}"] = save(a.output, f"prediction_{i}", pred)
                latent = schedule.step(pred, t, latent, return_dict=False)[0]
                outputs[f"latent_{i}"] = save(a.output, f"latent_{i}", latent)
    if a.backend == "mps":
        torch.mps.synchronize()
    result = dict(
        schema_version=1,
        runtime="diffusers-0.42.0/torch-2.15.0",
        backend=a.backend,
        dtype=a.dtype,
        mode=a.mode,
        fixture_sha256=sha(a.fixture.read_bytes()),
        outputs=outputs,
        loading_info=load_info,
        attention="QwenImage21AttnProcessor / native torch SDPA",
        quality_scope="synthetic conditioning, full component weights; no text encoder or semantic quality claim",
    )
    with (a.output / "result.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
