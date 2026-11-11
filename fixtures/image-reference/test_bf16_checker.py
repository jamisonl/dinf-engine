import copy, hashlib, importlib.util, json, tempfile, sys

sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checker", B / "check_bf16_results.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
gates = json.loads((B / "numeric-gates.json").read_text())
results = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    ref = root / "ref"
    act = root / "act"
    ref.mkdir()
    act.mkdir()
    outputs = {}
    for name, shape in {
        "sigmas": [5],
        "timesteps": [4],
        "input_latents": [1, 1024, 64],
        "input_conditioning": [1, 32, 4096],
        "prediction_0": [1, 1024, 64],
        "latent_0": [1, 1024, 64],
    }.items():
        raw = np.ones(shape, dtype="<f4").tobytes()
        file = name + ".f32"
        outputs[name] = dict(
            file=file, shape=shape, sha256=hashlib.sha256(raw).hexdigest(), finite=True
        )
        for p in [ref, act]:
            (p / file).write_bytes(raw)
    r = dict(
        schema_version=1,
        fixture_sha256="a" * 64,
        dtype="bf16",
        runtime="diffusers-0.42.0/torch-2.15.0",
        backend="mps",
        mode="denoise_single",
        outputs=outputs,
    )
    a = copy.deepcopy(r)
    a["runtime"] = "dinf-engine"
    a["backend"] = "metal"
    (ref / "result.json").write_text(json.dumps(r))
    (act / "result.json").write_text(json.dumps(a))
    assert mod.compare(ref, act, gates)["passed"]
    results.append("equal independent outputs pass")

    def rejected(changed):
        (act / "result.json").write_text(json.dumps(changed))
        try:
            mod.compare(ref, act, gates)
        except (AssertionError, ValueError):
            return
        raise AssertionError("invalid evidence accepted")

    bad = copy.deepcopy(a)
    bad["runtime"] = r["runtime"]
    rejected(bad)
    results.append("self-comparison rejected")
    # A cache comparison must use opposite modes; identical cached reports cannot qualify it.
    cached = copy.deepcopy(r)
    cached["mode"] = "denoise_cache"
    for d in [ref, act]:
        (d / "result.json").write_text(json.dumps(cached))
    try:
        mod.compare(ref, act, gates, cache=True)
    except AssertionError:
        results.append("same-cache-mode self-comparison rejected")
    else:
        raise AssertionError("same cache mode accepted")
    (ref / "result.json").write_text(json.dumps(r))
    (act / "result.json").write_text(json.dumps(a))

    bad = copy.deepcopy(a)
    bad["outputs"]["prediction_0"]["shape"] = [65536]
    rejected(bad)
    results.append("wrong full shape rejected")
    raw = (act / "prediction_0.f32").read_bytes()
    badraw = np.full([65536], float("nan"), dtype="<f4").tobytes()
    (act / "prediction_0.f32").write_bytes(badraw)
    bad = copy.deepcopy(a)
    bad["outputs"]["prediction_0"]["sha256"] = hashlib.sha256(badraw).hexdigest()
    rejected(bad)
    results.append("nonfinite matching-hash raw rejected")
    (act / "prediction_0.f32").write_bytes(raw)
    badraw = np.full([65536], 2, dtype="<f4").tobytes()
    (act / "prediction_0.f32").write_bytes(badraw)
    bad = copy.deepcopy(a)
    bad["outputs"]["prediction_0"]["sha256"] = hashlib.sha256(badraw).hexdigest()
    (act / "result.json").write_text(json.dumps(bad))
    v = mod.compare(ref, act, gates)
    assert v["valid"] and not v["passed"]
    results.append("finite numeric failure retained")
    (act / "prediction_0.f32").write_bytes(badraw[:-4])
    bad["outputs"]["prediction_0"]["sha256"] = hashlib.sha256(badraw[:-4]).hexdigest()
    rejected(bad)
    results.append("matching-hash short raw rejected")
print(json.dumps(dict(passed=True, tests=results, model_runs=0), indent=2))
