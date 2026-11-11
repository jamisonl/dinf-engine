#!/usr/bin/env python3
"""Offline raw tensor comparison. All limits come from the pre-run gate file."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_output(base, item):
    p = base / item["file"]
    assert p.parent == base and p.name == item["file"]
    assert sha(p) == item["sha256"]
    v = np.fromfile(p, dtype="<f4")
    assert v.size == np.prod(item["shape"]) and np.isfinite(v).all()
    return v.astype(np.float64)


def compare(refdir, actdir, gates, cache=False):
    r = json.loads((refdir / "result.json").read_text())
    a = json.loads((actdir / "result.json").read_text())
    assert (
        r["schema_version"] == a["schema_version"] == 1
        and r["fixture_sha256"] == a["fixture_sha256"]
        and r["dtype"] == a["dtype"] == "bf16"
    )
    assert not cache, "BF16 cache comparison is not part of this one-step fixture"
    assert r["mode"] == "denoise_single"
    if cache:
        assert {r["mode"], a["mode"]} == {"denoise_cache", "denoise_uncached"}
    else:
        assert r["mode"] == a["mode"]
    mode = r["mode"]
    expected = {"sigmas", "timesteps"}
    if mode != "schedule":
        expected.add("input_latents")
    if mode == "vae":
        expected |= {"denormalized_latents", "decoded"}
    elif mode.startswith("denoise"):
        expected.add("input_conditioning")
        for i in range(1 if mode == "denoise_single" else 4):
            expected |= {f"prediction_{i}", f"latent_{i}"}
    assert set(r["outputs"]) == set(a["outputs"]) == expected
    limits = gates["same_runtime_cache" if cache else "component"]
    rows = {}
    passed = True
    if cache:
        assert (
            r["runtime"] == a["runtime"]
            and r["runtime"] in {"dinf-engine", "diffusers-0.42.0/torch-2.15.0"}
            and r["backend"] == a["backend"] == "cpu"
        )
    else:
        assert (
            r["runtime"] == "diffusers-0.42.0/torch-2.15.0"
            and a["runtime"] == "dinf-engine"
        )
        assert r["backend"] == "mps" and a["backend"] == "metal"
    for k in sorted(expected):
        x, y = r["outputs"][k], a["outputs"][k]
        shape = (
            [5]
            if k == "sigmas"
            else [4]
            if k == "timesteps"
            else [1, 4, 512, 512]
            if k == "decoded"
            else [1, 64, 32, 32]
            if k == "denormalized_latents"
            else [1, 32, 4096]
            if k == "input_conditioning"
            else [1, 1024, 64]
        )
        assert x["shape"] == y["shape"] == shape
        u, v = read_output(refdir, x), read_output(actdir, y)
        if k not in {"sigmas", "timesteps"}:
            assert np.all(
                (u.astype(np.float32).view(np.uint32) & 0xFFFF) == 0
            ) and np.all((v.astype(np.float32).view(np.uint32) & 0xFFFF) == 0), (
                "export does not preserve BF16 value bits"
            )
        delta = v - u
        normu = np.linalg.norm(u)
        normv = np.linalg.norm(v)
        rel = (
            float(np.linalg.norm(delta) / normu)
            if normu
            else (0.0 if not np.any(delta) else float("inf"))
        )
        cosine = (
            float(np.dot(u, v) / (normu * normv))
            if normu and normv
            else (1.0 if np.array_equal(u, v) else 0.0)
        )
        mx = float(np.max(np.abs(delta)))
        rms = float(np.sqrt(np.mean(delta * delta)))
        if k.startswith("input_"):
            ok = x["sha256"] == y["sha256"]
        elif k in {"sigmas", "timesteps"}:
            ok = mx <= gates["scheduler_max_abs"]
        else:
            ok = rel <= limits["relative_l2_max"] and cosine >= limits["cosine_min"]
        rows[k] = dict(
            shape=x["shape"],
            max_abs=mx,
            rms=rms,
            relative_l2=rel,
            cosine=cosine,
            passed=bool(ok),
        )
        passed &= ok
    return dict(
        valid=True,
        passed=bool(passed),
        reference=str(refdir),
        actual=str(actdir),
        cache_comparison=cache,
        rows=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("actual", type=Path)
    p.add_argument("gates", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--cache", action="store_true")
    a = p.parse_args()
    result = dict(valid=False, passed=False)
    try:
        result = compare(
            a.reference, a.actual, json.loads(a.gates.read_text()), a.cache
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    with a.report.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
