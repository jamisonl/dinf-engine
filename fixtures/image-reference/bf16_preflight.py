#!/usr/bin/env python3
"""Bounded native MPS BF16 operator preflight; no checkpoint files or model instances."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    assert torch.__version__.startswith("2.15.0")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    result = dict(
        passed=False,
        backend="mps",
        dtype="bf16",
        checkpoint_files_read=0,
        model_instances=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        operations={},
    )
    try:
        assert torch.backends.mps.is_available(), "MPS unavailable"
        x = (
            (torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) % 257 - 128)
            / 128
        ).to("mps", torch.bfloat16)
        w = x.transpose(0, 1).contiguous()
        q = x[:16].reshape(1, 4, 16, 16).contiguous()
        mask = torch.ones((16, 16), dtype=torch.bool, device="mps").tril()
        probes = {
            "matmul": lambda: x @ w,
            "rms_norm": lambda: F.rms_norm(x, (64,), eps=1e-6),
            "layer_norm": lambda: F.layer_norm(x, (64,), eps=1e-6),
            "gelu_tanh": lambda: F.gelu(x, approximate="tanh"),
            "gelu_erf": lambda: F.gelu(x, approximate="none"),
            "sdpa_masked": lambda: F.scaled_dot_product_attention(
                q, q, q, attn_mask=mask
            ),
            "sdpa_causal": lambda: F.scaled_dot_product_attention(
                q, q, q, is_causal=True
            ),
            "rope_product": lambda: x * x.float().cos().to(torch.bfloat16),
            "timestep_division": lambda: x / 1000,
        }
        for name, fn in probes.items():
            y = fn()
            assert y.device.type == "mps" and y.dtype == torch.bfloat16
            cpu = y.float().cpu().contiguous()
            assert torch.isfinite(cpu).all(), name
            result["operations"][name] = dict(
                shape=list(y.shape),
                dtype=str(y.dtype),
                device=str(y.device),
                sha256=hashlib.sha256(
                    cpu.numpy().astype("<f4", copy=False).tobytes()
                ).hexdigest(),
                finite=True,
            )
        torch.mps.synchronize()
        result.update(
            passed=True,
            allocated_bytes=torch.mps.current_allocated_memory(),
            driver_bytes=torch.mps.driver_allocated_memory(),
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
