"""Bind the proved CPU F32-attention repair to each CPU oracle run.

This changes only the CPU oracle, not dinf-engine or the Metal oracle. Numerical
gates stay in verify_results.py. A correct component does not imply a model pass.
"""
import hashlib
import json
from pathlib import Path

POLICY = "ggml-cpu-flash-f32-v2"
REVISION = "15fa3c493bfcd040b5f4dcb29e1c998a0846de16"
ACTIVATIONS_POLICY = "ggml-cpu-f32-activations-v3"
QUERY_POLICY = "ggml-cpu-flash-f32-query-v4"
POLICIES = (POLICY, ACTIVATIONS_POLICY, QUERY_POLICY)
PATCH_SHA256 = "232d3a998ca7a69045a677b95cb25dbf94d729ed3286c31fdad72482e26704ac"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_pins(pins):
    require(isinstance(pins, dict) and pins, "missing CPU reference pins")
    for path, sha in pins.items():
        require(Path(path).is_absolute() and Path(path).is_file()
                and digest(path) == sha, f"CPU reference pin differs: {path}")


def check_file(record):
    path = Path(record["path"])
    require(path.is_absolute() and path.is_file(), "CPU runtime path is not an absolute file")
    require(path.stat().st_size == record["size_bytes"] and digest(path) == record["sha256"],
            f"CPU runtime bytes differ: {path}")
    return {"path": str(path.resolve()), "sha256": record["sha256"], "size_bytes": record["size_bytes"]}


def activation_notice():
    path = Path(__file__).with_name("cpu-reference") / "F32_ACTIVATIONS.json"
    value = json.loads(path.read_text())
    require(value["policy"] == ACTIVATIONS_POLICY and value["extends"] == POLICY
            and value["source_revision"] == REVISION, "unknown activation repair")
    return value


def check_activation_policy(selected):
    if selected["id"] == ACTIVATIONS_POLICY:
        require(selected.get("activation_policy") == activation_notice(),
                "activation repair policy differs")
    else:
        require("activation_policy" not in selected, "v2 must not declare activation repairs")


def query_notice():
    path = Path(__file__).with_name("cpu-reference") / "F32_QUERY.json"
    value = json.loads(path.read_text())
    require(value["policy"] == QUERY_POLICY and value["extends"] == POLICY
            and value["source_revision"] == REVISION, "unknown F32 query repair")
    return value


def check_query_policy(selected):
    if selected["id"] == QUERY_POLICY:
        require(selected.get("query_policy") == query_notice(), "F32 query repair policy differs")
    else:
        require("query_policy" not in selected, "v2/v3 must not declare F32 query repair")


def build_policy(receipt_path, revision):
    path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(path.read_text())
    require(receipt["passed"] is True and receipt["policy"] in POLICIES, "CPU repair build did not pass")
    require(receipt["source_revision"] == revision == REVISION, "CPU repair revision differs")
    require(receipt["platform"] == "darwin-arm64" and receipt["isa"] == "apple_m4",
            "CPU repair platform or ISA differs from the tested build")
    pins = {}
    for group in (receipt["source_pins"], receipt["runtime_pins"]):
        check_pins(group)
        for name, sha in group.items():
            require(name not in pins or pins[name] == sha, "CPU repair receipts disagree")
            pins[name] = sha
    patch_sha = receipt["source_patch_sha256"]
    require(patch_sha == PATCH_SHA256 and patch_sha in receipt["source_pins"].values(),
            "CPU patch is not the source-pinned precision fix")
    plugin = check_file(receipt["builds"]["candidate"]["plugin"])
    blas = check_file(receipt["blas_plugin"])
    require(Path(plugin["path"]).name == "libggml-cpu-apple_m4.so", "wrong CPU plugin")
    require(Path(blas["path"]).name == "libggml-blas.so", "wrong BLAS plugin")
    for record in (plugin, blas):
        pins[record["path"]] = record["sha256"]
    pins[str(path)] = digest(path)
    policy = {"schema_version": 1, "id": receipt["policy"], "source_revision": revision,
              "source_patch_sha256": patch_sha, "plugin": plugin, "blas_plugin": blas}
    if receipt["policy"] == ACTIVATIONS_POLICY:
        policy["activation_policy"] = receipt.get("activation_policy")
        check_activation_policy(policy)
        notice_path = Path(__file__).with_name("cpu-reference") / "F32_ACTIVATIONS.json"
        pins[str(notice_path.resolve())] = digest(notice_path)
        for repair in policy["activation_policy"]["patches"]:
            require(repair["sha256"] in receipt["source_pins"].values(),
                    "activation patch is not source-pinned")
    else:
        require("activation_policy" not in receipt, "v2 receipt has activation repairs")
    if receipt["policy"] == QUERY_POLICY:
        policy["query_policy"] = receipt.get("query_policy")
        check_query_policy(policy)
        notice_path = Path(__file__).with_name("cpu-reference") / "F32_QUERY.json"
        pins[str(notice_path.resolve())] = digest(notice_path)
        for repair in policy["query_policy"]["patches"]:
            require(repair["sha256"] in receipt["source_pins"].values(),
                    "F32 query patch is not source-pinned")
    else:
        require("query_policy" not in receipt, "v2/v3 receipt has F32 query repair")
    return policy, pins


def check_runtime(runtime, policy, pins):
    require(policy["schema_version"] == 1 and policy["id"] in POLICIES
            and policy["source_revision"] == REVISION
            and policy["source_patch_sha256"] == PATCH_SHA256,
            "CPU runtime does not use the proved precision repair")
    check_activation_policy(policy)
    check_query_policy(policy)
    for field in ("plugin", "blas_plugin"):
        require(check_file(policy[field]) == policy[field], "CPU runtime path must be canonical")
    require(runtime["policy"] == policy, "CPU runtime policy differs")
    require(runtime["cpu_plugin_image_count"] == runtime["blas_plugin_image_count"] == 1
            and runtime["registry_count"] == 2 and runtime["gpu_backends_loaded"] is False,
            "CPU runtime loaded unexpected backends")
    resolved = {}
    for name, sha in pins.items():
        path = str(Path(name).resolve())
        require(path not in resolved or resolved[path] == sha, "CPU runtime aliases disagree")
        resolved[path] = sha
    images = [Path(name).resolve() for name in runtime["runtime_images"]]
    for image in images:
        name = str(image)
        if name.startswith(("/System/Library/", "/usr/lib/")):
            continue  # OS identity covers dyld's shared system cache.
        require(name in resolved and digest(image) == resolved[name], f"unbound CPU runtime image: {image}")
    cpu = [p for p in images if p.name.startswith("libggml-cpu")]
    blas = [p for p in images if p.name == "libggml-blas.so"]
    require(cpu == [Path(policy["plugin"]["path"])] and blas == [Path(policy["blas_plugin"]["path"])],
            "CPU runtime selected a different plugin")
    require(not any(p.name == "libggml-metal.so" for p in images), "Metal plugin present in CPU run")


def check_suite(suite, backend, policy, pins):
    if backend == "metal":
        require("cpu_attention_policy" not in suite and "cpu_runtime_before" not in suite
                and "cpu_runtime_after" not in suite, "Metal oracle must keep its stock runtime")
        return
    require(backend == "cpu" and policy is not None and policy["id"] in POLICIES,
            "CPU oracle requires the pinned F32 attention repair")
    require(suite["cpu_attention_policy"] == policy, "CPU oracle policy differs")
    for field in ("cpu_runtime_before", "cpu_runtime_after"):
        check_runtime(suite[field], policy, pins)


def check_case(case, backend, policy):
    if backend == "cpu":
        require(policy is not None and case["cpu_attention_policy"] == policy,
                "CPU case lacks the selected repair policy")
    else:
        require("cpu_attention_policy" not in case, "Metal case unexpectedly declares a CPU repair")
