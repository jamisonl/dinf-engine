#!/usr/bin/env python3
"""Run one pinned artifact/backend, retaining oracle and engine evidence."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback

from build_engine import digest, write_new
from cpu_reference_policy import check_suite as check_cpu_suite, check_case as check_cpu_case

SMOKE_CASES = ["prose-512", "conversation-512", "json-2048", "rust-2048",
               "retrieval-4096", "multilingual-4096"]
INTERNAL_CASES = ["conversation-512", "json-2048", "retrieval-4096"]
SHORT_STATE_CASES = ["conversation-512"]
SMOKE_PROFILES = ("smoke-6", "smoke-6-short-state")


def state_cases(profile):
    return SHORT_STATE_CASES if profile == "smoke-6-short-state" else INTERNAL_CASES


def selected_cases(prepared, profile, pilot=False):
    cases = prepared["cases"]
    ids = [case["id"] for case in cases]
    require(len(ids) == len(set(ids)), "duplicate prepared case")
    if profile in SMOKE_PROFILES:
        require(not pilot and len(cases) in (6, 24), "smoke requires six cases or a preserved full reference")
        lookup = {case["id"]: case for case in cases}
        require(set(SMOKE_CASES) <= set(lookup), "missing smoke case")
        require(len(cases) != 6 or ids == SMOKE_CASES, "smoke case order differs")
        return [lookup[name] for name in SMOKE_CASES]
    require(1 <= len(cases) < 24 if pilot else len(cases) == 24,
            "use --pilot only for an explicit smaller pilot; extended coverage requires 24 cases")
    return cases


def check_smoke_identity(directory, model_id, cases):
    """Bind the same six numeric inputs across both smoke profiles and references.

    State coverage stays in each suite/report; this identity binds inputs only.
    """
    path = directory / f"{model_id}.smoke-6.json"
    value = {"model_id": model_id, "coverage_profile": "smoke-6", "cases": cases}
    extended = directory / f"{model_id}.json"
    if extended.is_file():
        old = {case["id"]: case for case in load(extended)["cases"]}
        require(all(old.get(case["id"]) == case for case in cases),
                "smoke inputs differ from the existing extended fixture")
    if path.exists():
        require(load(path) == value, "smoke inputs differ across tiers or backends")
    else:
        write_new(path, value)
    return {"path": str(path), "sha256": digest(path)}


def producer_argv(event):
    require(event["exit_code"] == 0 and event["timed_out"] is False, "reused producer did not finish")
    argv = event["command"]
    return argv[2:] if argv[:2] == ["/usr/bin/time", "-l"] else argv


def preserve_reference(args, reference, entry, record):
    """Copy checked reference bytes without changing their original metadata."""
    source, output = args.reuse_oracle, args.output
    require(args.profile in SMOKE_PROFILES and not args.pilot, "reference reuse requires a smoke profile")
    original_pins = load(source / "pins.json")
    commands = {name: load(source / f"{name}.command.json") for name in ("prepare", "oracle")}
    prepare, oracle = [producer_argv(commands[name]) for name in ("prepare", "oracle")]
    require(len(prepare) == len(oracle) == 5, "reused producer arguments differ")
    require(prepare[1] == "--prepare" and Path(prepare[2]).resolve() == entry
            and Path(prepare[3]).resolve() == args.corpus
            and Path(prepare[4]).resolve() == source / "prepared.json", "reused preparation differs")
    require(Path(oracle[0]).resolve() == Path(reference["binary"]["path"]).resolve()
            and Path(oracle[1]).resolve() == entry and oracle[2] == args.backend
            and Path(oracle[3]).resolve() == source / "prepared.json"
            and Path(oracle[4]).resolve() == source / "oracle", "reused reference inputs differ")
    require(any(Path(name).resolve() == Path(oracle[0]).resolve() and sha == reference["binary"]["sha256"]
                for name, sha in original_pins.items()), "reused reference binary is not pinned")
    corpus = load(args.corpus)
    native = load(source / "oracle/suite.json")
    prepared = load(source / "prepared.json")
    require(native["status"] == "complete" and native["expected_cases"] == native["completed_cases"] == len(prepared["cases"]),
            "reused reference is incomplete")
    require(native["prepared_sha256"] == digest(source / "prepared.json")
            and native["corpus_sha256"] == prepared["corpus_sha256"] == digest(args.corpus),
            "reused reference input hashes differ")
    require([c["id"] for c in native["cases"]] == [c["id"] for c in prepared["cases"]]
            == [c["id"] for c in corpus["cases"]], "reused reference cases differ")
    selected_cases(prepared, args.profile)
    sources = [(source / "prepared.json", "prepared.json"), (args.corpus, "corpus.json"),
               (source / "oracle/suite.json", "oracle/suite.json")]
    lookup = {case["id"]: case for case in native["cases"]}
    for name in SMOKE_CASES:
        case = load(source / "oracle" / f"{name}.json")
        require(digest(source / "oracle" / f"{name}.json") == lookup[name]["oracle_json_sha256"]
                and case["oracle_raw_sha256"] == lookup[name]["oracle_raw_sha256"], "reused oracle index differs")
        require(digest(source / "oracle" / f"{name}.f32") == case["oracle_raw_sha256"], "reused oracle raw differs")
        sources.extend((source / "oracle" / f"{name}.{suffix}", f"oracle/{name}.{suffix}") for suffix in ("json", "f32"))
    preserved = []
    for original, name in sources:
        expected = digest(original)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with original.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
        require(digest(destination) == expected == digest(original), "reference changed while preserving it")
        preserved.append({"source": {"path": str(original), "sha256": expected}, "destination": name})
    receipt = {"schema_version": 1, "source_job": str(source),
               "original_commands": {name: {"path": str(source / f"{name}.command.json"),
                                             "sha256": digest(source / f"{name}.command.json")} for name in commands},
               "original_pins": {"path": str(source / "pins.json"), "sha256": digest(source / "pins.json")},
               "reference_build": {"path": str(args.reference_build), "sha256": digest(args.reference_build)},
               "preserved_files": preserved}
    if args.reuse_admission:
        receipt["admission"] = {"path": str(args.reuse_admission), "sha256": digest(args.reuse_admission)}
    path = output / "oracle-reuse.json"
    write_new(path, receipt)
    record["oracle_reuse"] = {"path": str(path), "sha256": digest(path)}


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean_environment(source):
    removed = sorted(k for k in source if k.startswith(("DINF_", "CANDLE_", "GGML_", "GGUF_"))
                     or (k.startswith("DYLD_") and not k.startswith("DYLD_PRINT_"))
                     or k in {"LD_PRELOAD", "LD_LIBRARY_PATH"})
    return {k: v for k, v in source.items() if k not in removed}, removed


def check_pins(pins):
    for name, expected in pins.items():
        require(Path(name).is_file() and digest(name) == expected, f"pinned file differs: {name}")


def build_pins(path):
    value = load(path)
    pins = dict(value["source_pins"])
    pins.update(value["runtime_pins"])
    pins[value["binary"]["path"]] = value["binary"]["sha256"]
    check_pins(pins)
    return value, pins


def find_artifact(manifest, artifact_id):
    found = [(model, artifact) for model in manifest["models"] for artifact in model["artifacts"] if artifact["id"] == artifact_id]
    require(len(found) == 1, "artifact must occur exactly once")
    return found[0]


def artifact_files(artifact, entry):
    records = [artifact["preparation"]["expected_output"]] if artifact.get("requires_preparation") else artifact["files"]
    require(entry.name == Path(records[0]["filename"]).name, "entry name differs from manifest")
    files = []
    for record in records:
        path = entry.parent / Path(record["filename"]).name
        require(path.is_file() and path.stat().st_size == record["size_bytes"], f"missing or wrong-sized artifact: {path}")
        files.append({"name": path.name, "size_bytes": record["size_bytes"], "sha256": record["sha256"]})
    return files


def check_identity(directory, model_id, prepared):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_id}.json"
    value = {"model_id": model_id, "corpus_sha256": prepared["corpus_sha256"], "token_identity_sha256": prepared["token_identity_sha256"], "cases": prepared["cases"]}
    if path.exists():
        require(load(path) == value, "prompt/token identity differs across tiers or backends")
    else:
        write_new(path, value)
    return {"path": str(path), "sha256": digest(path)}


def available_memory():
    if platform.system() == "Darwin":
        output = subprocess.check_output(["vm_stat"], text=True)
        page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))
        pages = {key: int(value) for key, value in re.findall(r"^(Pages [^:]+):\s+(\d+)\.", output, re.MULTILINE)}
        # Inactive pages include reclaimable file cache. Purgeable pages can
        # overlap these groups and must not be added a second time.
        return page_size * sum(pages.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative")), "vm_stat free+inactive+speculative"
    if Path("/proc/meminfo").is_file():
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.MULTILINE)
        require(match is not None, "MemAvailable missing")
        return int(match.group(1)) * 1024, "Linux MemAvailable"
    raise RuntimeError("memory admission requires macOS vm_stat or Linux MemAvailable")


def admit_memory(model, artifact, files, name, record):
    full_gib = {"qwen25-coder-1.5b": 16, "mistral-7b-v03": 56, "phi3-mini-4k": 40,
                "qwen35-4b": 32, "qwen3-moe-30b-a3b": 80, "deepseek-v2-lite-chat": 96}
    needed = full_gib[model["id"]] * 1024**3 if artifact["tier"] == "FULL" else max(8 * 1024**3, 3 * sum(f["size_bytes"] for f in files) + 4 * 1024**3)
    available, method = available_memory()
    record.setdefault("memory_admission", []).append({"stage": name, "available_bytes": available, "required_bytes": needed, "method": method})
    require(available >= needed, f"{name} memory admission: need {needed / 1024**3:.1f} GiB available, have {available / 1024**3:.1f}")


def execute(command, name, output, env, record, timeout):
    wrapped = (["/usr/bin/time", "-l"] if platform.system() == "Darwin" else []) + list(map(str, command))
    started = time.time()
    timed_out = False
    with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(wrapped, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    event = {"name": name, "command": wrapped, "exit_code": code, "seconds": time.time() - started, "timeout_seconds": timeout, "timed_out": timed_out}
    record["commands"].append(event)
    write_new(output / f"{name}.command.json", event)
    print(json.dumps(event), flush=True)
    return code


def run(args, record):
    output = args.output
    manifest = load(args.manifest)
    model, artifact = find_artifact(manifest, args.artifact)
    paths = load(args.paths)
    if "artifacts" in paths:
        paths = {item["artifact_id"]: item["path"] for item in paths["artifacts"]}
    entry = Path(paths[args.artifact]).resolve()
    files = artifact_files(artifact, entry)
    engine, pins = build_pins(args.engine_build)
    reference, reference_pins = build_pins(args.reference_build)
    cpu_policy = reference.get("cpu_reference")
    require(args.backend != "cpu" or cpu_policy is not None, "CPU reference build lacks the proved attention repair")
    for key, value in reference_pins.items():
        require(key not in pins or pins[key] == value, f"build receipts disagree: {key}")
        pins[key] = value
    for path in [args.manifest, args.corpus, args.engine_build, args.reference_build, Path(__file__), Path(__file__).with_name("build_engine.py"), Path(__file__).with_name("cpu_reference_policy.py")]:
        pins[str(path.resolve())] = digest(path)
    env, removed = clean_environment(os.environ)
    # Preserve the exact inputs and driver even if a later fix changes the tree.
    snapshots = {}
    for source, name in [(args.corpus, "corpus.json"), (args.manifest, "models.json"),
                         (Path(__file__), "run.py"), (Path(__file__).with_name("build_engine.py"), "build_engine.py"),
                         (Path(__file__).with_name("cpu_reference_policy.py"), "cpu_reference_policy.py")]:
        saved = output / name
        with source.open("rb") as original, saved.open("xb") as copy:
            shutil.copyfileobj(original, copy)
        require(digest(saved) == pins[str(source.resolve())], f"input changed while preserving evidence: {source}")
        snapshots[name] = {"path": str(saved), "sha256": digest(saved)}
    record.update({"model_id": model["id"], "artifact": artifact, "backend": args.backend, "model_entry": str(entry), "source_pins": pins, "removed_environment_names": removed, "host": platform.platform()})
    record["archived_inputs"] = snapshots
    record["cpu_attention_policy"] = cpu_policy if args.backend == "cpu" else None
    write_new(output / "pins.json", pins)
    prepared_path = output / "prepared.json"
    if args.reuse_oracle:
        preserve_reference(args, reference, entry, record)
    else:
        require(execute([engine["binary"]["path"], "--prepare", entry, args.corpus, prepared_path], "prepare", output, env, record, args.timeout_seconds) == 0, "token preparation failed")
    prepared = load(prepared_path)
    require(prepared["model_files"] == files, "prepared artifact bytes differ from pinned model manifest")
    require(prepared["architecture"] == model["architecture"], "architecture differs")
    require(prepared["corpus_sha256"] == pins[str(args.corpus)], "corpus differs")
    selected = selected_cases(prepared, args.profile, args.pilot)
    record["pilot"] = args.pilot
    record["coverage_profile"] = args.profile
    identity_dir = args.identities / "smoke-6" if args.profile in SMOKE_PROFILES and len(prepared["cases"]) == 6 else args.identities
    record["identity"] = check_identity(identity_dir, model["id"], prepared)
    if args.profile in SMOKE_PROFILES:
        record["smoke_identity"] = check_smoke_identity(args.identities, model["id"], selected)
    # Retain both producers' selected F32 rows and leave 8 GiB for other work.
    required_disk = len(selected) * prepared["rows"] * prepared["vocab_size"] * (4 if args.reuse_oracle else 8) + 8 * 1024**3
    require(shutil.disk_usage(output).free >= required_disk, "insufficient space for complete raw evidence")
    check_pins(pins)
    if not args.reuse_oracle:
        admit_memory(model, artifact, files, "oracle", record)
        oracle_code = execute([reference["binary"]["path"], entry, args.backend, prepared_path, output / "oracle"], "oracle", output, env, record, args.timeout_seconds)
        check_pins(pins)
        require(oracle_code == 0, "reference suite failed; retained its partial evidence")
    check_cpu_suite(load(output / "oracle/suite.json"), args.backend, cpu_policy, pins)
    cases = []
    for case in selected:
        path = output / "oracle" / f"{case['id']}.json"
        oracle = load(path)
        check_cpu_case(oracle, args.backend, cpu_policy)
        cases.append({"id": case["id"], "reference": {"path": str(path), "sha256": digest(path)}, "logits_sha256": oracle["oracle_raw_sha256"]})
    suite = {"schema_version": 1, "suite_id": prepared["suite_id"], "backend": args.backend,
             "prepared": {"path": str(prepared_path), "sha256": digest(prepared_path)},
             "corpus": {"path": str(args.corpus), "sha256": digest(args.corpus)},
             "expected_binary_sha256": engine["binary"]["sha256"],
             "expected_token_identity_sha256": prepared["token_identity_sha256"],
             "coverage_profile": args.profile, "fail_fast": args.profile in SMOKE_PROFILES,
             "internal_cases": [case["id"] for case in prepared["cases"]] if args.pilot else state_cases(args.profile),
             "source_pins": [{"path": name, "sha256": value} for name, value in sorted(pins.items())],
             "cases": cases, "internal": {"decode_tokens": 8, "prefill_chunk": 128, "comparison_prefill_chunk": 32, "cache_max_bytes": 4 * 1024**3}}
    write_new(output / "suite.json", suite)
    admit_memory(model, artifact, files, "engine", record)
    engine_code = execute([engine["binary"]["path"], entry, args.backend, output / "suite.json", output / "engine"], "engine", output, env, record, args.timeout_seconds)
    check_pins(pins)
    report = load(output / "engine/report.json")
    record["post_run_pins_verified"] = True
    record["engine_valid"] = report["valid"]
    record["engine_passed"] = report["passed"]
    record["engine_exit_code"] = engine_code
    write_new(output / "execution.json", record)
    # Independent review does not trust either producer's acceptance decision.
    if args.validation_python:
        validator = Path(__file__).with_name("verify_results.py")
        require(validator.is_file(), "independent validator missing")
        record["validator"] = {"path": str(validator), "sha256": digest(validator)}
        review_options = ["--pilot"] if args.pilot else (["--profile", args.profile] if args.profile in SMOKE_PROFILES else [])
        code = execute([args.validation_python, validator, output, *review_options], "independent", output, env, record, args.timeout_seconds)
        record["independent_exit_code"] = code
        require(digest(validator) == record["validator"]["sha256"], "validator changed during review")
        record["independently_reviewed"] = (output / "independent-review.json").is_file()
        record["passed"] = engine_code == 0 and report["passed"] is True and code == 0
    else:
        record["passed"] = False
        record["review_pending"] = True
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--profile", choices=["extended-24", *SMOKE_PROFILES], default="extended-24")
    parser.add_argument("--reuse-oracle", type=Path, help="reuse a completed reference job without changing its metadata (smoke profiles only)")
    parser.add_argument("--reuse-admission", type=Path, help="optional saved provenance review for the reused reference")
    for name in ["paths", "engine-build", "reference-build", "output", "identities"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--validation-python", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    require(not args.reuse_admission or args.reuse_oracle, "reuse admission requires a reused reference")
    require(not (args.pilot and args.profile in SMOKE_PROFILES), "smoke profiles are fixed, not pilots")
    if args.reuse_oracle:
        require(args.profile in SMOKE_PROFILES, "reference reuse requires a smoke profile")
        original = producer_argv(load(args.reuse_oracle / "prepare.command.json"))
        require(args.corpus is None or args.corpus.resolve() == Path(original[3]).resolve(), "reused corpus path differs")
        args.corpus = Path(original[3])
    elif args.corpus is None:
        args.corpus = Path(__file__).with_name("corpus-smoke-6.json" if args.profile in SMOKE_PROFILES else "corpus.json")
    if args.profile in SMOKE_PROFILES:
        canonical = load(Path(__file__).with_name("corpus.json"))
        if not args.reuse_oracle:
            lookup = {case["id"]: case for case in canonical["cases"]}
            canonical["cases"] = [lookup[name] for name in SMOKE_CASES]
        require(load(args.corpus) == canonical, "smoke must retain the shared corpus prompts and context budgets")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving a venv's Python symlink loses the venv and its NumPy.
            setattr(args, name, value.absolute() if name == "validation_python" else value.resolve())
    args.output.mkdir()
    def interrupted(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError(f"runner received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    record = {"schema_version": 1, "passed": False, "commands": [], "started_unix": time.time()}
    code = 1
    try:
        args.identities.mkdir(parents=True, exist_ok=True)
        with (args.identities / ".model-process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = run(args, record)
    except Exception as error:
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
        print(record["traceback"], file=sys.stderr)
    finally:
        record["seconds"] = time.time() - record["started_unix"]
        write_new(args.output / "run.json", record)
    return code


if __name__ == "__main__":
    sys.exit(main())
