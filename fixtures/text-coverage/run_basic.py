#!/usr/bin/env python3
"""Run pinned text basics: --paths paths.json --binary qualify_text --output NEW_DIR.

Use --model ID repeatedly for a subset. --rss-caps accepts {ID: {backend: bytes}}
when a backend lacks a recorded cap. --checksum-cache reuses this runner's own
SHA256 results only while device/inode/size/mtime/ctime all remain unchanged.
These checks do not qualify numeric accuracy or the broader engine release.
The default full list exits zero only when all basic coverage gaps are closed.
Explicit --model or one --backend exits zero when that selected scope passes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

from run import available_memory, clean_environment, require

GIB = 1024 ** 3
RESERVE = 24 * GIB
SWAP_LIMIT = 128 * 1024 ** 2
CHECKS = ["chunked_prefill", "after_prefill_cancel", "interleaved_decode_cancel"]
PROFILE = {
    "tokens_per_target_run": 4, "target_runs": 3, "noise_tokens": 2,
    "prefill_chunk": 8,
    "prompt": "Write a Rust function to merge two sorted slices. Include tests.",
    "interleaved_prompt": "Name three oceans and explain how they differ.",
    "checks": ["finite logits", "exact greedy replay after chunked prefill",
               "partial-prefill cancellation", "interleaved decode cancellation",
               "cache positions", "empty cache after cleanup"],
}


def load(path):
    require(path.stat().st_size <= 8 * 1024 ** 2, f"JSON too large: {path}")
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def checksum(path, expected, cache):
    before = identity(path)
    require(path.is_file() and before[2] == expected["size_bytes"],
            f"missing or wrong-sized weight: {path}")
    prior = cache.get(str(path), {})
    cached = prior.get("identity") == before and prior.get("sha256") == expected["sha256"]
    if not cached:
        h = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            if platform.system() == "Darwin":
                import fcntl
                fcntl.fcntl(stream.fileno(), 48, 1)  # Darwin F_NOCACHE.
            while block := stream.read(4 * 1024 ** 2):
                h.update(block)
        require(h.hexdigest() == expected["sha256"], f"weight SHA256 differs: {path}")
    require(identity(path) == before, f"weight changed during verification: {path}")
    cache[str(path)] = {"identity": before, "sha256": expected["sha256"]}
    return {"path": str(path), **cache[str(path)], "checksum_cache_hit": cached}


def select(registry, models, backend):
    entries = registry["text_artifacts"]
    lookup = {m["id"]: m for m in entries}
    require(len(lookup) == len(entries) and entries, "empty/duplicate text registry")
    require(all(set(m["backends"]) == {"cpu", "metal"} for m in entries),
            "registry must declare both backends, including explicit unsupported entries")
    require(all(registry["profiles"]["basic-v1"].get(k) == v for k, v in PROFILE.items()),
            "unsupported basic profile")
    chosen = models or list(lookup)
    require(chosen and len(set(chosen)) == len(chosen) and set(chosen) <= set(lookup),
            "unknown, duplicate or empty model selection")
    backends = ["cpu", "metal"] if backend == "both" else [backend]
    required = {(m["id"], b) for m in entries for b in ("cpu", "metal")
                if m["backends"][b]["status"] != "unsupported"}
    return [(lookup[name], b) for name in chosen for b in backends], required


def validate_report(report, event, model, backend, entry):
    require(event["exit_code"] == 0 and not event["error"] and event["all_processes_closed"],
            "consumer process failed or did not close")
    require(report.get("passed") is True and report.get("profile") == "basic-v1",
            "missing successful basic-v1 report")
    require(report.get("backend") == backend and report.get("architecture") == model["architecture"]
            and Path(report.get("path", "")).resolve() == entry, "report identity differs")
    require(all(type(report.get(k)) is int for k in ("max_tokens", "prefill_chunk_tokens", "split_stages"))
            and report.get("max_tokens") == 4 and report.get("prefill_chunk_tokens") == 8
            and report.get("split_stages") == 1 and report.get("draft") is False
            and type(report.get("prompt_tokens")) is int and report["prompt_tokens"] > 8,
            "basic profile counts differ")
    runs = report.get("runs")
    require(isinstance(runs, list) and len(runs) == 1, "expected one greedy report")
    run = runs[0]
    require(run.get("mode") == "greedy" and run.get("checks") == CHECKS,
            "basic state checks differ")
    tokens = run.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == 4
            and all(type(t) is int and 0 <= t <= 0xffffffff for t in tokens),
            "expected four token IDs")


def group_processes(pgid):
    text = subprocess.check_output(["ps", "-axo", "pid=,pgid=,rss="], text=True)
    return [(int(pid), int(rss) * 1024) for line in text.splitlines()
            for pid, group, rss in [line.split()] if int(group) == pgid]


def swapout_bytes():
    if platform.system() == "Darwin":
        data = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        page = re.search(r"page size of (\d+) bytes", data)
        count = re.search(r"^Swapouts:\s+(\d+)\.", data, re.MULTILINE)
        require(page is not None and count is not None, "vm_stat swap count missing")
        return int(page.group(1)) * int(count.group(1))
    data = Path("/proc/vmstat").read_text()
    count = re.search(r"^pswpout\s+(\d+)$", data, re.MULTILINE)
    require(count is not None, "Linux swap count missing")
    return int(count.group(1)) * os.sysconf("SC_PAGE_SIZE")


def execute(argv, output, env, cap, timeout):
    start = time.monotonic()
    initial_swap = swapout_bytes()
    initial_available, _ = available_memory()
    event = {"argv": argv, "rss_cap_bytes": cap, "reserve_bytes": RESERVE,
             "timeout_seconds": timeout, "error": None, "peak_sampled_rss_bytes": 0,
             "swapout_limit_bytes": SWAP_LIMIT, "swapout_before_bytes": initial_swap,
             "maximum_swapout_growth_bytes": 0, "minimum_available_bytes": initial_available,
             "all_processes_closed": False}
    def check_memory():
        free, _ = available_memory()
        growth = max(0, swapout_bytes() - initial_swap)
        event["minimum_available_bytes"] = min(event["minimum_available_bytes"], free)
        event["maximum_swapout_growth_bytes"] = max(event["maximum_swapout_growth_bytes"], growth)
        require(free >= RESERVE and growth <= SWAP_LIMIT,
                "24 GiB memory reserve or 128 MiB swapout growth exceeded")

    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            while process.poll() is None:
                rss = sum(rss for _, rss in group_processes(process.pid))
                event["peak_sampled_rss_bytes"] = max(event["peak_sampled_rss_bytes"], rss)
                check_memory()
                if rss > cap:
                    raise RuntimeError("RSS cap exceeded")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("consumer timeout")
                if max(os.fstat(out.fileno()).st_size, os.fstat(err.fileno()).st_size) > 8 * 1024 ** 2:
                    raise RuntimeError("consumer log limit exceeded")
                time.sleep(0.25)
        except BaseException as error:
            event["error"] = repr(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if process.poll() is None or group_processes(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    event["error"] = event["error"] or f"process termination failed: {error}"
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                if group_processes(process.pid):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        event["error"] = event["error"] or f"process kill failed: {error}"
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                event["error"] = "process cleanup did not finish"
            event["exit_code"] = process.returncode
            event["all_processes_closed"] = not group_processes(process.pid)
            event["log_bytes"] = {"stdout": os.fstat(out.fileno()).st_size,
                                  "stderr": os.fstat(err.fileno()).st_size}
            try:
                check_memory()
                require(max(event["log_bytes"].values()) <= 8 * 1024 ** 2,
                        "consumer final log limit exceeded")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                event["error"] = event["error"] or repr(error)
    event["seconds"] = time.monotonic() - start
    return event


def aggregate(records, required, selected_required, open_gaps):
    selected = {(r["id"], r["backend"]) for r in records if r["status"] != "unsupported"}
    passed = {(r["id"], r["backend"]) for r in records if r["status"] == "passed"}
    selected_passed = bool(selected_required) and passed == selected == selected_required
    complete_text = selected_passed and selected_required == required
    gaps = [g for g in open_gaps if g.get("status") != "passed"]
    return {"selected_passed": selected_passed, "complete_text_registry": complete_text,
            "complete": complete_text and not gaps, "broader_basic_gaps": gaps,
            "scope": "Basic state coverage only; no numeric, quality or release qualification.",
            "records": records}


def run(args):
    registry = load(args.registry)
    jobs, required = select(registry, args.model, args.backend)
    paths = load(args.paths)
    caps = load(args.rss_caps) if args.rss_caps else {}
    ids = {m["id"] for m in registry["text_artifacts"]}
    require(set(paths) <= ids and set(caps) <= ids, "unknown ID in paths/caps")
    require(all(set(v) <= {"cpu", "metal"} and all(type(c) is int and c > 0 for c in v.values())
                for v in caps.values()), "invalid backend RSS caps")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
    binary = args.binary.resolve(strict=True)
    require(binary.is_file() and os.access(binary, os.X_OK), "binary must be executable")
    binary_identity = identity(binary)
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    cache = {"producer": "dinf-engine/run_basic.py", "version": 1, "files": {}}
    if args.checksum_cache and args.checksum_cache.exists():
        cache = load(args.checksum_cache)
        require(cache.get("producer") == "dinf-engine/run_basic.py" and cache.get("version") == 1
                and isinstance(cache.get("files"), dict), "not this runner's checksum cache")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "registry.json", registry)
    env, removed = clean_environment(os.environ)
    env.update(RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    selected_required = {(m["id"], b) for m, b in jobs if m["backends"][b]["status"] != "unsupported"}
    records, verified, abort = [], {}, False
    for index, (model, backend) in enumerate(jobs):
        record = {"id": model["id"], "backend": backend, "status": "not_run",
                  "binary_sha256": binary_hash, "removed_environment_keys": removed}
        records.append(record)
        directory = args.output / f"{index:03d}-{backend}"
        directory.mkdir()
        try:
            setting = model["backends"][backend]
            if setting["status"] == "unsupported":
                record.update(status="unsupported", reason=setting.get("reason"))
                continue
            if abort:
                record["reason"] = "earlier process cleanup failed"
                continue
            if model["id"] not in paths:
                record.update(status="missing_artifact", reason="no --paths entry")
                continue
            cap = caps.get(model["id"], {}).get(backend, setting.get("rss_cap_bytes"))
            if type(cap) is not int or cap <= 0:
                record.update(status="not_run_no_cap", reason="supply an explicit --rss-caps entry")
                continue
            free, method = available_memory()
            record["memory_admission"] = {"available_bytes": free, "rss_cap_bytes": cap,
                                          "reserve_bytes": RESERVE, "method": method}
            if free < cap + RESERVE:
                record.update(status="not_run_memory", reason="RSS cap plus reserve exceeds available RAM")
                continue
            entry = Path(paths[model["id"]]).expanduser().resolve(strict=True)
            require(entry.name == Path(model["entry"]).name, "entry filename differs")
            if model["id"] not in verified:
                names = [Path(f["filename"]).name for f in model["files"]]
                require(names and len(names) == len(set(names)) and entry.name in names,
                        "invalid artifact file set")
                verified[model["id"]] = [checksum(entry.parent / Path(f["filename"]).name, f, cache["files"])
                                          for f in model["files"]]
                if args.checksum_cache:
                    temp = args.checksum_cache.with_suffix(args.checksum_cache.suffix + ".new")
                    write(temp, cache)
                    os.replace(temp, args.checksum_cache)
            files = verified[model["id"]]
            record["files"] = files
            require(all(identity(Path(f["path"])) == f["identity"] for f in files)
                    and identity(binary) == binary_identity, "input changed before run")
            free, method = available_memory()
            record["memory_admission_after_checks"] = {"available_bytes": free, "method": method}
            if free < cap + RESERVE:
                record.update(status="not_run_memory", reason="memory changed during input verification")
                continue
            event = execute([str(binary), str(entry), backend, "--smoke"], directory, env, cap, args.timeout)
            record["process"] = event
            abort = not event["all_processes_closed"]
            require(all(identity(Path(f["path"])) == f["identity"] for f in files)
                    and identity(binary) == binary_identity, "input changed during run")
            report = load(directory / "stdout.log")
            validate_report(report, event, model, backend, entry)
            record.update(status="passed", report=report)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
            record.update(status="failed", error=str(error))
        finally:
            write(directory / "receipt.json", record)
            summary = aggregate(records, required, selected_required, registry.get("open_basic_coverage", []))
            summary.update(finished=len(records) == len(jobs), binary={"path": str(binary), "sha256": binary_hash, "identity": binary_identity},
                           required_text_jobs=len(required), selected_text_jobs=sum(m["backends"][b]["status"] != "unsupported" for m, b in jobs))
            write(args.output / "summary.json", summary)
    return summary


def exit_status(result, models, backend):
    full_scope = not models and backend == "both"
    return 0 if result["complete" if full_scope else "selected_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("basic-coverage.json"))
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal", "both"], default="both")
    parser.add_argument("--model", action="append")
    parser.add_argument("--rss-caps", type=Path)
    parser.add_argument("--checksum-cache", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    return exit_status(result, args.model, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
