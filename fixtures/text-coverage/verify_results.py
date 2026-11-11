#!/usr/bin/env python3
"""Independently check a text matrix job, one F32 row at a time.

Usage: python verify_results.py JOB_DIR
Writes a new independent-review.json. Exit 0 means complete, valid evidence;
the report's passed field can still be false. Exit 1 means invalid/incomplete
evidence. Requires NumPy; never loads a model or initializes a GPU.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from cpu_reference_policy import check_suite as check_cpu_suite, check_case as check_cpu_case

ROWS = 128
TOLERANCE = 1e-10
CONTEXTS = [512, 2048, 4096]
INTERNAL_CASES = ["conversation-512", "json-2048", "retrieval-4096"]
SMOKE_CASES = ["prose-512", "conversation-512", "json-2048", "rust-2048",
               "retrieval-4096", "multilingual-4096"]
CASE_KEYS = ["id", "domain", "context_tokens", "rendered_prompt_utf8", "continuation_utf8",
             "prompt_original_ids", "prompt_tokens", "continuation_original_ids",
             "teacher_forcing_tokens", "prompt_add_special", "continuation_add_special", "parse_special"]
SMOKE_INTERNALS = {"smoke-6": INTERNAL_CASES,
                   "smoke-6-short-state": ["conversation-512"]}
CHECKS = ["replay_after_other_request_and_chunked_prefill", "interleaved_decode_cancellation",
          "replay_after_partial_prefill_cancellation", "prompt_cache_publish",
          "prompt_cache_reuse", "prompt_cache_owner_isolation"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"invalid JSON number: {value}")


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load(path):
    require(path.stat().st_size <= 64 * 1024**2, f"JSON too large: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            and value not in (".", ".."), f"unsafe name: {value}")
    return value


def integer(value, lower, upper, label):
    require(type(value) is int and lower <= value <= upper, f"invalid {label}")
    return value


def tokens(value, vocab, label, count=None):
    require(isinstance(value, list) and (count is None or len(value) == count), f"invalid {label} length")
    for token in value:
        integer(token, 0, vocab - 1, label)
    return value


def stamp(path):
    s = path.stat()
    return {"bytes": s.st_size, "modified_ns": s.st_mtime_ns, "device": s.st_dev,
            "inode": s.st_ino, "changed_seconds": s.st_ctime_ns // 10**9,
            "changed_nanoseconds": s.st_ctime_ns % 10**9}


def resolve(base, value):
    return (base / value).resolve(strict=True)


def check_pin(pin, base, expected_path=None, full_hash=True):
    require(valid_hash(pin["sha256"]), "invalid pin SHA256")
    path = resolve(base, pin["path"])
    if expected_path is not None:
        require(path == expected_path.resolve(strict=True), f"pin path differs: {path}")
    before = stamp(path)
    if "identity" in pin:
        require(pin["identity"] == before, f"pin file identity differs: {path}")
    if full_hash:
        require(digest(path) == pin["sha256"], f"pin hash differs: {path}")
    require(stamp(path) == before, f"file changed during check: {path}")
    return path


def near(actual, expected, label):
    require(type(actual) in (float, int) and math.isfinite(actual)
            and abs(actual - expected) <= TOLERANCE,
            f"{label}: reported {actual!r}, recomputed {expected!r}")


def metric(reference, actual, teacher, row):
    require(reference.shape == actual.shape and reference.ndim == 1 and len(reference), "raw row shape differs")
    require(np.isfinite(reference).all() and np.isfinite(actual).all(), "nonfinite raw logits")
    # NumPy F64 reductions are independent from the engine's scalar Rust loop.
    r, a = reference.astype(np.float64), actual.astype(np.float64)
    rs, az = r - r.max(), a - a.max()
    rz, za = float(np.log(np.exp(rs).sum(dtype=np.float64))), float(np.log(np.exp(az).sum(dtype=np.float64)))
    log_r, log_a = rs - rz, az - za
    kl = max(0.0, float(np.sum(np.exp(log_r) * (log_r - log_a), dtype=np.float64)))
    delta = r - a
    rn, an = float(rz - rs[teacher]), float(za - az[teacher])
    return {"row": row, "reference_top": int(r.argmax()), "actual_top": int(a.argmax()),
            "teacher_token": teacher, "kl": kl, "max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.sum(delta * delta, dtype=np.float64) / len(r))),
            "reference_nll": rn, "actual_nll": an, "nll_delta": an - rn}


def numeric_pass(matches, mean_kl, max_kl):
    return matches >= 122 and mean_kl <= 0.02 and max_kl <= 0.1


def internal_checks(result, case, options, vocab):
    require(result["required"] is True and result["attempted"] is True and result["status"] == "complete"
            and result["valid"] is True and len(result["modes"]) == 2, "incomplete required internal modes")
    count = integer(options["decode_tokens"], 8, 16, "internal decode count")
    primary = min(options["prefill_chunk"], len(case["prompt_tokens"]) - 1)
    secondary = min(options["comparison_prefill_chunk"], len(case["prompt_tokens"]) - 1)
    all_pass = True
    for expected_mode, mode in zip(["greedy", "seeded_penalties"], result["modes"]):
        require(mode["mode"] == expected_mode, "internal mode order differs")
        baseline = tokens(mode["baseline_tokens"], vocab, "baseline", count)
        require([check["name"] for check in mode["checks"]] == CHECKS, "missing/reordered internal check")
        for check in mode["checks"]:
            actual = tokens(check["actual_tokens"], vocab, "internal output", count)
            first = next((i for i, (x, y) in enumerate(zip(baseline, actual)) if x != y), None)
            require(check["first_mismatch"] == first, "wrong internal mismatch index")
            passed = first is None
            detail, label = check["details"], check["name"]
            if label == CHECKS[0]:
                require(detail["prefill_chunk"] == primary, "replay chunk differs")
            elif label == CHECKS[1]:
                require(detail["cancel_after_tokens"] == 2, "decode cancellation count differs")
            elif label == CHECKS[2]:
                require(detail["prefill_chunk"] == secondary, "prefill cancellation chunk differs")
            elif label == CHECKS[3]:
                retained = integer(detail["retained_bytes"], 0, options["cache_max_bytes"], "retained prompt cache")
                reused = integer(detail["reused_prompt_tokens"], 0, len(case["prompt_tokens"]), "cold reuse")
                passed = passed and reused == 0 and retained > 0
            elif label == CHECKS[4]:
                passed = passed and detail["reused_prompt_tokens"] == len(case["prompt_tokens"])
            else:
                passed = passed and detail["reused_prompt_tokens"] == 0
            require(check["passed"] is passed, f"internal acceptance differs: {label}")
            all_pass = all_pass and passed
    require(result["passed"] is all_pass, "internal aggregate differs")
    return all_pass


def check_unrun_internal(result, selected):
    expected = {"required": selected, "attempted": False, "valid": None,
                "passed": None, "modes": [],
                "status": "not_run_after_numeric_failure" if selected else "not_selected"}
    require(result == expected and result["required"] is selected and result["attempted"] is False,
            "unrun state check has a verdict or wrong status")


def check_smoke_counts(report, reviews, stopped):
    attempted = [c for c in reviews if c["attempted"]]
    n = len(attempted)
    require(1 <= n <= 6 and [c["id"] for c in attempted] == SMOKE_CASES[:n],
            "smoke attempted cases are not a nonempty prefix")
    failed = [c for c in attempted if c["numeric_passed"] is False]
    if stopped:
        require(len(failed) == 1 and failed[0]["id"] == SMOKE_CASES[n - 1],
                "smoke did not stop at first numeric rejection")
        require(report.get("early_stop") == {"phase": "numeric", "case_id": failed[0]["id"],
                                              "reason": "numeric_failure"},
                "smoke early-stop reason differs")
    else:
        require(n == 6 and not failed and report.get("early_stop") is None,
                "state phase ran without six numeric passes")
    require(report["numeric_counts"] == {"planned": 6, "attempted": n,
            "passed": n - len(failed), "failed": len(failed), "unrun": 6 - n},
            "smoke numeric counts differ")
    planned_internal = len(SMOKE_INTERNALS[report["coverage_profile"]])
    checked = 0 if stopped else planned_internal
    internal_pass = sum(c["internal_passed"] is True for c in reviews)
    require(report["internal_counts"] == {"planned": planned_internal, "attempted": checked,
            "passed": internal_pass, "failed": checked - internal_pass, "unrun": planned_internal - checked},
            "smoke internal counts differ")
    require(report["counts"] == {"planned": 6, "attempted": n,
            "passed": sum(c["passed"] is True for c in reviews),
            "failed": sum(c["passed"] is False for c in reviews), "unrun": 6 - n,
            "incomplete": sum(c["attempted"] and c["passed"] is None for c in reviews)},
            "smoke case counts differ")


def check_oracle_reuse(job, execution, prepared, suite, inputs, pins):
    """Bind a preserved oracle without claiming a new prepare/oracle command."""
    receipt_path = check_pin(execution["oracle_reuse"], job, job / "oracle-reuse.json")
    receipt = load(receipt_path)
    require(receipt["schema_version"] == 1, "oracle reuse schema differs")
    if receipt.get("admission") is not None:
        check_pin(receipt["admission"], job)
    source_job = resolve(job, receipt["source_job"])
    require(source_job != job, "oracle reuse source is current job")
    old_pins_path = check_pin(receipt["original_pins"], job, source_job / "pins.json")
    old_pins = load(old_pins_path)
    reference_path = check_pin(receipt["reference_build"], job)
    require(pins.get(str(reference_path)) == digest(reference_path)
            or any(Path(p).resolve() == reference_path and h == digest(reference_path) for p, h in pins.items()),
            "reused reference build missing from current pins")
    reference = load(reference_path)
    reference_binary = check_pin(reference["binary"], job)
    expected = {"prepared.json", "corpus.json", "oracle/suite.json",
                *(f"oracle/{label}{suffix}" for label in SMOKE_CASES for suffix in (".json", ".f32"))}
    preserved = receipt["preserved_files"]
    require(len(preserved) == len(expected) and {p["destination"] for p in preserved} == expected,
            "oracle reuse preserved files differ")
    source_by_destination = {}
    for item in preserved:
        relative = item["destination"]
        destination = job / relative
        require(not destination.is_symlink() and job in destination.resolve().parents,
                "oracle reuse destination escapes job")
        source = check_pin(item["source"], job)
        require(source != destination.resolve() and (source_job in source.parents or relative == "corpus.json"),
                "oracle reuse source escapes original job")
        require(digest(destination) == item["source"]["sha256"], "oracle reuse copy differs")
        source_by_destination[relative] = source
    require(source_by_destination["prepared.json"] == source_job / "prepared.json"
            and source_by_destination["oracle/suite.json"] == source_job / "oracle/suite.json",
            "oracle reuse original layout differs")
    old_commands = receipt["original_commands"]
    require(set(old_commands) == {"prepare", "oracle"}, "oracle reuse original commands differ")
    for label in ("prepare", "oracle"):
        path = check_pin(old_commands[label], job, source_job / f"{label}.command.json")
        command = load(path)
        require(command["name"] == label and command["exit_code"] == 0
                and command.get("timed_out") is False, "original oracle producer did not close successfully")
        argv = command["command"]
        if argv[:2] == ["/usr/bin/time", "-l"]:
            argv = argv[2:]
        require(len(argv) == 5, "original oracle command shape differs")
        binary = resolve(source_job, argv[0])
        require(any(Path(p).resolve() == binary and valid_hash(h) for p, h in old_pins.items()),
                "original command executable unpinned")
        model_path = resolve(job, execution["model_entry"])
        if label == "prepare":
            require(argv[1] == "--prepare" and resolve(source_job, argv[2]) == model_path
                    and resolve(source_job, argv[3]) == source_by_destination["corpus.json"]
                    and resolve(source_job, argv[4]) == source_by_destination["prepared.json"],
                    "original prepare command inputs differ")
        else:
            require(binary == reference_binary and any(Path(p).resolve() == binary
                    and h == reference["binary"]["sha256"] for p, h in old_pins.items()),
                    "reused oracle executable differs from reference build")
            require(resolve(source_job, argv[1]) == model_path and argv[2] == suite["backend"]
                    and resolve(source_job, argv[3]) == source_by_destination["prepared.json"]
                    and resolve(source_job, argv[4]) == source_job / "oracle",
                    "original oracle command inputs differ")
    require(load(source_by_destination["prepared.json"]) == prepared,
            "oracle reuse prepared content differs")
    return execution["oracle_reuse"]


def review(job, pilot=False, profile="extended-24"):
    job = job.resolve(strict=True)
    evidence_paths = [job / p for p in ("prepared.json", "suite.json", "engine/report.json", "oracle/suite.json", "execution.json", "pins.json")]
    evidence_stamps = {path: stamp(path) for path in evidence_paths}
    prepared = load(job / "prepared.json")
    suite = load(job / "suite.json")
    report = load(job / "engine/report.json")
    oracle_suite = load(job / "oracle/suite.json")
    execution = load(job / "execution.json")
    pins = load(job / "pins.json")
    require(profile == "extended-24" or profile in SMOKE_INTERNALS, "unknown coverage profile")
    smoke = profile in SMOKE_INTERNALS
    require(not (smoke and pilot), "smoke is not a legacy pilot")
    require(suite.get("coverage_profile", "extended-24") == profile
            and report.get("coverage_profile", "extended-24") == profile,
            "coverage profile differs")
    if smoke:
        require(suite.get("fail_fast") is True and report.get("fail_fast") is True
                and report.get("execution_policy") == "numeric_then_state",
                "smoke execution policy differs")
        require(execution.get("coverage_profile") == profile
                and report.get("selected_case_ids") == SMOKE_CASES, "smoke selection receipt differs")
        require(report.get("state_phase_status") in ("complete", "not_run_after_numeric_failure"),
                "smoke state phase incomplete")
    else:
        require(suite.get("fail_fast", False) is False, "legacy profile cannot fail fast")
    require(execution["post_run_pins_verified"] is True, "missing runner post-run pin check")
    require(execution["source_pins"] == pins, "runner execution pins differ")
    require(report["schema_version"] == 1 and report["post_run_pins_verified"] is True
            and report["valid"] is True, "engine evidence incomplete")
    require(report["runtime_dispatch_overrides"] == [], "nondefault engine dispatch")
    require(type(report.get("oracle_micro_batch_tokens")) is int and report["oracle_micro_batch_tokens"] == 8,
            "engine oracle micro batch must be eight tokens")
    require(report["model_loads"] == 1, "engine model load count differs")
    require(suite["schema_version"] == prepared["schema_version"] == 1 and prepared["rows"] == ROWS, "schema/row count differs")
    require(suite["backend"] in ("cpu", "metal") and report["backend"] == suite["backend"], "backend differs")
    require(prepared["suite_id"] == suite["suite_id"] == report["suite_id"], "suite ID differs")
    require(report["architecture"] == prepared["architecture"], "architecture differs")
    require(report["limits"] == {"rows": ROWS, "min_argmax_fraction": 0.95, "mean_kl": 0.02,
                                  "max_kl": 0.1, "nll_gate": None, "kl_direction": "reference || actual"}, "numeric gates differ")
    for path, sha in pins.items():
        check_pin({"path": path, "sha256": sha}, job)
    require({p["path"]: p["sha256"] for p in suite["source_pins"]} == pins, "suite pins differ from runner")
    inputs = report["input_pins"]
    for key, path in [("suite", job / "suite.json"), ("prepared", job / "prepared.json")]:
        check_pin(inputs[key], job, path)
    check_pin(suite["prepared"], job, job / "prepared.json")
    corpus_path = check_pin(suite["corpus"], job)
    check_pin(inputs["corpus"], job, corpus_path)
    corpus = load(corpus_path)
    require(prepared["corpus_sha256"] == digest(corpus_path), "corpus hash differs")
    require(corpus["schema_version"] == 1 and corpus["rows"] == ROWS and corpus["suite_id"] == suite["suite_id"], "corpus scope differs")
    check_pin(inputs["binary"], job)
    require(inputs["binary"]["sha256"] == suite["expected_binary_sha256"], "engine binary differs")
    require(len(inputs["sources"]) == len(pins), "source pin count differs")
    for pin in inputs["sources"]:
        path = check_pin(pin, job)
        require(any(Path(p).resolve() == path and h == pin["sha256"] for p, h in pins.items()), "unexpected checked source pin")
    require(inputs["binary"]["sha256"] in pins.values(), "binary missing from runner pins")
    require(all(h in pins.values() for h in report["compiled_helper_sources"].values()), "compiled helper source missing from pins")
    prepared_cases = prepared["cases"]
    prepared_count = len(prepared_cases)
    require(prepared_count in (6, 24) if smoke else ((1 <= prepared_count < 24) if pilot else prepared_count == 24),
            "expected a smaller pilot" if pilot else "expected all24 domain/context cases")
    prepared_ids = [name(c["id"]) for c in prepared_cases]
    require(len(set(prepared_ids)) == len(prepared_ids), "duplicate case ID")
    ids = SMOKE_CASES if smoke else prepared_ids
    by_id = {c["id"]: c for c in prepared_cases}
    require(set(ids) <= set(prepared_ids), "missing selected case")
    if smoke and prepared_count == 6:
        require(prepared_ids == SMOKE_CASES, "smoke fixture order differs")
    cases = [by_id[label] for label in ids]
    case_count = len(cases)
    internal_ids = ids if pilot else SMOKE_INTERNALS[profile] if smoke else INTERNAL_CASES
    require(suite["internal_cases"] == internal_ids and report["internal_case_ids"] == internal_ids,
            "required internal case selection differs")
    require(set(internal_ids) <= set(ids), "missing required internal case")
    domains = sorted(set(c["domain"] for c in prepared_cases))
    if not pilot and prepared_count == 24:
        require(len(domains) == 8 and all(sorted(c["context_tokens"] for c in prepared_cases if c["domain"] == d) == CONTEXTS for d in domains), "domain/context grid differs")
    if smoke:
        require(all(c["id"] == f"{c['domain']}-{c['context_tokens']}" for c in cases)
                and sorted(c["context_tokens"] for c in cases) == [512, 512, 2048, 2048, 4096, 4096],
                "smoke domain/context selection differs")
    for collection in [suite["cases"], report["cases"]]:
        require([c["id"] for c in collection] == ids, "missing/reordered cases")
    for collection in [corpus["cases"], oracle_suite["cases"]]:
        require([c["id"] for c in collection] == prepared_ids, "fixture/oracle coverage differs")
    canonical = [{key: c[key] for key in CASE_KEYS} for c in prepared_cases]
    token_hash = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(token_hash == prepared["token_identity_sha256"] == suite["expected_token_identity_sha256"] == report["token_identity_sha256"], "token identity differs")
    shared_identity = check_pin(execution["identity"], job)
    require(load(shared_identity) == {"model_id": execution["model_id"], "corpus_sha256": prepared["corpus_sha256"],
                                     "token_identity_sha256": token_hash, "cases": prepared_cases},
            "shared tier/backend token fixture differs")
    if smoke:
        smoke_identity = check_pin(execution["smoke_identity"], job)
        require(load(smoke_identity) == {"model_id": execution["model_id"],
                "coverage_profile": "smoke-6", "cases": cases}, "shared smoke token fixture differs")
    vocab = integer(prepared["vocab_size"], 1, 10**7, "vocabulary")
    files = prepared["model_files"]
    require(files and report["model_files"] == files and oracle_suite["model_files"] == files, "model manifests differ")
    require(len(inputs["model_files"]) == len(files), "model pin count differs")
    require(prepared["model_sha256"] == files[0]["sha256"], "entry model hash differs")
    for file, pin in zip(files, inputs["model_files"]):
        name(file["name"])
        require(valid_hash(file["sha256"]) and pin["sha256"] == file["sha256"], "model hash receipt differs")
        path = check_pin(pin, job, full_hash=False)
        require(path.stat().st_size == file["size_bytes"], "model size differs")
    require(oracle_suite["status"] == "complete" and oracle_suite["completed_cases"] == oracle_suite["expected_cases"] == prepared_count,
            "oracle suite incomplete")
    require(oracle_suite["backend"] == suite["backend"] and oracle_suite["suite_id"] == suite["suite_id"], "oracle suite/backend differs")
    cpu_policy = oracle_suite.get("cpu_attention_policy")
    require(execution.get("cpu_attention_policy") == cpu_policy, "execution CPU policy differs")
    check_cpu_suite(oracle_suite, suite["backend"], cpu_policy, pins)
    require(oracle_suite["model_files_sha256_verified_once"] is True
            and oracle_suite["model_file_stats_before"] == oracle_suite["model_file_stats_after"], "oracle model identity changed")
    require(oracle_suite["kv_dtype"] == "F16" and oracle_suite["threads"] == 8
            and oracle_suite["prefill_chunk_tokens"] == 128 and oracle_suite["new_context_per_case"] is True, "oracle execution scope differs")
    require(type(oracle_suite.get("micro_batch_tokens")) is int and oracle_suite["micro_batch_tokens"] == 8,
            "oracle suite micro batch must be eight tokens")
    require(oracle_suite["prepared_sha256"] == digest(job / "prepared.json") and oracle_suite["corpus_sha256"] == prepared["corpus_sha256"], "oracle fixture differs")
    options = suite["internal"]
    require(options == report["internal_options"], "internal options differ")
    if smoke:
        require(options == {"decode_tokens": 8, "prefill_chunk": 128,
                "comparison_prefill_chunk": 32, "cache_max_bytes": 4 * 1024**3},
                "smoke state-check options differ")
        require(prepared_count == 6 or execution.get("oracle_reuse") is not None,
                "full24 smoke oracle requires explicit reuse")
    for key in ["prefill_chunk", "comparison_prefill_chunk"]:
        integer(options[key], 1, 4096, key)
    require(options["prefill_chunk"] != options["comparison_prefill_chunk"], "internal chunks identical")
    integer(options["cache_max_bytes"], 1, 8 * 1024**3, "prompt-cache bound")
    require(len(inputs["references"]) == case_count, "reference pin count differs")
    text_by_id = {c["id"]: c for c in corpus["cases"]}
    oracle_by_id = {c["id"]: c for c in oracle_suite["cases"]}
    reviews, raw_files = [], []
    stopped = smoke and report["state_phase_status"] == "not_run_after_numeric_failure"
    for index, (case, entry, saved) in enumerate(zip(cases, suite["cases"], report["cases"])):
        label = case["id"]
        text, oracle_entry = text_by_id[label], oracle_by_id[label]
        require(case["domain"] == text["domain"] and case["context_tokens"] == text["context_tokens"], f"{label}: domain/context differs")
        require(case["continuation_utf8"] == text["continuation_text"], f"{label}: natural continuation differs")
        require(case["prompt_add_special"] is True and case["continuation_add_special"] is False and case["parse_special"] is True, "tokenizer flags differ")
        prompt = tokens(case["prompt_tokens"], vocab, "prompt")
        require(len(prompt) >= 2 and prompt == case["prompt_original_ids"], "prompt changed/truncated")
        teacher = tokens(case["teacher_forcing_tokens"], vocab, "teacher", ROWS)
        original = tokens(case["continuation_original_ids"], vocab, "continuation")
        require(original[:ROWS] == teacher, "teacher is not original continuation prefix")
        require(len(prompt) + ROWS <= case["context_tokens"] <= prepared["model_context_length"], "context admission differs")
        oracle_path = job / "oracle" / f"{label}.json"
        check_pin(entry["reference"], job, oracle_path)
        oracle = load(oracle_path)
        check_cpu_case(oracle, suite["backend"], cpu_policy)
        require(oracle["source"].startswith("llama.cpp ") and oracle["producer_source_sha256"] in pins.values(), "unbound oracle source")
        require(oracle["producer_source_sha256"] == oracle_suite["producer_source_sha256"], "oracle sources differ")
        require(oracle["schema_version"] == 1 and oracle["case_id"] == label and oracle["context_tokens"] == case["context_tokens"], "oracle case differs")
        require(oracle["backend"] == suite["backend"] and oracle["vocab_size"] == vocab, "oracle backend/vocabulary differs")
        require(type(oracle.get("micro_batch_tokens")) is int and oracle["micro_batch_tokens"] == 8,
                "oracle case micro batch must be eight tokens")
        require(oracle["model_sha256"] == prepared["model_sha256"] and oracle["model_files"] == files, "oracle weights differ")
        require(oracle["prepared_sha256"] == digest(job / "prepared.json") and oracle["tokenization_verified"] is True, "oracle token proof differs")
        require(oracle["prompt_tokens"] == prompt and oracle["teacher_forcing_tokens"] == teacher, "oracle history differs")
        tops = tokens(oracle["continuation_tokens"], vocab, "oracle argmax", ROWS)
        require(oracle["logits_file"] == f"{label}.f32" and oracle_entry["oracle_json"] == f"{label}.json"
                and oracle_entry["oracle_raw"] == f"{label}.f32", "oracle output filenames differ")
        require(oracle_entry["oracle_json_sha256"] == digest(oracle_path), "oracle index JSON hash differs")
        raw_path = job / "oracle" / oracle["logits_file"]
        raw_sha = digest(raw_path)
        require(raw_sha == entry["logits_sha256"] == oracle["oracle_raw_sha256"] == oracle_entry["oracle_raw_sha256"], "oracle raw hash differs")
        expected_bytes = ROWS * vocab * 4
        require(raw_path.stat().st_size == oracle_entry["oracle_raw_bytes"] == expected_bytes,
                "oracle raw byte count differs")
        check_pin(inputs["references"][index]["json"], job, oracle_path)
        check_pin(inputs["references"][index]["raw"], job, raw_path)
        raw_files.append({"path": str(raw_path.relative_to(job)), "sha256": raw_sha, "size_bytes": expected_bytes})
        actual_path = job / "engine" / f"{label}.actual.f32"
        result = load(job / "engine" / f"{label}.result.json")
        require(result == saved, "case result differs from engine suite report")
        require(result["domain"] == case["domain"] and result["context_tokens"] == case["context_tokens"]
                and result["prompt_tokens"] == len(prompt) and result["rows"] == ROWS, "case scope differs")
        if smoke and result["attempted"] is False:
            require(stopped and result["valid"] is None and result["passed"] is None
                    and result.get("status") == "not_run_after_numeric_failure"
                    and result["reference"] == {"valid": None, "passed": None, "completed_rows": 0},
                    "unrun numeric case has a verdict or wrong status")
            require(not actual_path.exists() and not actual_path.is_symlink(), "unrun case has actual raw")
            check_unrun_internal(result["internal"], label in internal_ids)
            with raw_path.open("rb") as stream:
                for row in range(ROWS):
                    values = np.frombuffer(stream.read(vocab * 4), dtype="<f4")
                    require(np.isfinite(values).all() and int(values.argmax()) == tops[row],
                            "unrun reference row nonfinite or argmax differs")
            reviews.append({"id": label, "attempted": False, "valid_evidence": None,
                            "passed": None, "numeric_passed": None, "internal_required": label in internal_ids,
                            "internal_passed": None, "oracle_raw_sha256": raw_sha, "rows": []})
            continue
        require(result["attempted"] is True and result["valid"] is True, "case evidence incomplete")
        actual_identity = stamp(actual_path)
        ref = result["reference"]
        require(ref["valid"] is True and ref["completed_rows"] == ROWS and len(ref["comparisons"]) == ROWS, "incomplete raw rows")
        require(resolve(job, ref["actual_logits_file"]) == actual_path.resolve(), "actual raw path differs")
        actual_sha = digest(actual_path)
        require(actual_sha == ref["actual_logits_sha256"], "actual raw hash differs")
        require(raw_path.stat().st_size == actual_path.stat().st_size == ref["actual_logits_bytes"] == oracle_entry["oracle_raw_bytes"] == expected_bytes, "raw byte count differs")
        raw_files.append({"path": str(actual_path.relative_to(job)), "sha256": actual_sha, "size_bytes": expected_bytes})
        require(ref["prefill_chunk_tokens"] == options["prefill_chunk"], "engine reference chunk differs")
        check_pin(inputs["references"][index]["json"], job, oracle_path)
        check_pin(inputs["references"][index]["raw"], job, raw_path)
        rows = []
        with raw_path.open("rb") as rfile, actual_path.open("rb") as afile:
            for row, target in enumerate(teacher):
                r = np.frombuffer(rfile.read(vocab * 4), dtype="<f4")
                a = np.frombuffer(afile.read(vocab * 4), dtype="<f4")
                recomputed = metric(r, a, target, row)
                require(recomputed["reference_top"] == tops[row], "oracle top disagrees with raw row")
                claimed = ref["comparisons"][row]
                for key in ["row", "reference_top", "actual_top", "teacher_token"]:
                    require(type(claimed[key]) is int and claimed[key] == recomputed[key], f"{label}: {key} differs at row{row}")
                for key in ["kl", "max_abs", "rms", "reference_nll", "actual_nll", "nll_delta"]:
                    near(claimed[key], recomputed[key], f"{label} row{row} {key}")
                rows.append(recomputed)
        require(stamp(actual_path) == actual_identity, "actual raw changed during review")
        matches = sum(r["reference_top"] == r["actual_top"] for r in rows)
        mean_kl, max_kl = math.fsum(r["kl"] for r in rows) / ROWS, max(r["kl"] for r in rows)
        aggregate = {"matching_argmax": matches, "mean_kl": mean_kl, "max_kl": max_kl,
                     "reference_mean_nll": math.fsum(r["reference_nll"] for r in rows) / ROWS,
                     "actual_mean_nll": math.fsum(r["actual_nll"] for r in rows) / ROWS,
                     "mean_nll_delta": math.fsum(r["nll_delta"] for r in rows) / ROWS}
        require(ref["matching_argmax"] == matches, "argmax count differs")
        for key, value in aggregate.items():
            near(ref[key], value, f"{label} {key}")
        numeric = numeric_pass(matches, mean_kl, max_kl)
        require(ref["passed"] is numeric, "numeric acceptance differs from raw recomputation")
        selected = label in internal_ids
        if smoke and stopped:
            check_unrun_internal(result["internal"], selected)
            internal = None
        elif selected:
            if smoke:
                require(result["internal"].get("noise_case_id") == "prose-512",
                        "smoke state noise differs")
            internal = internal_checks(result["internal"], case, options, vocab)
        else:
            require(result["internal"] == {"required": False, "attempted": False, "status": "not_selected", "valid": None, "passed": None, "modes": []}
                    and result["internal"]["required"] is False and result["internal"]["attempted"] is False,
                    "unselected internal check must have no pass claim")
            internal = None
        case_pass = (None if numeric and selected else numeric) if stopped else numeric and (not selected or internal)
        require(result["passed"] is case_pass, "case acceptance differs")
        reviews.append({"id": label, "attempted": True, "valid_evidence": True, "passed": case_pass,
                        "numeric_passed": numeric, "internal_required": selected, "internal_passed": internal, **aggregate,
                        "oracle_raw_sha256": raw_sha, "actual_raw_sha256": actual_sha,
                        "rows": rows})
    passed = all(c["passed"] is True for c in reviews)
    count_pass = sum(c["passed"] is True for c in reviews)
    internal_count_pass = sum(c["internal_passed"] is True for c in reviews)
    if smoke:
        check_smoke_counts(report, reviews, stopped)
    else:
        require(report["internal_counts"] == {"planned": len(internal_ids), "attempted": len(internal_ids),
                                         "passed": internal_count_pass, "failed": len(internal_ids) - internal_count_pass},
            "internal case counts differ")
    require(report["passed"] is passed, "engine overall acceptance differs")
    if not smoke:
        require(report["counts"] == {"planned": case_count, "attempted": case_count, "passed": count_pass, "failed": case_count-count_pass}, "case counts differ")
    require(execution["engine_valid"] is True and execution["engine_passed"] is passed, "runner engine outcome differs")
    require(execution["engine_exit_code"] == (0 if passed else 1), "engine exit status differs")
    commands = execution["commands"]
    reuse_pin = None
    if execution.get("oracle_reuse") is not None:
        require(smoke, "oracle reuse requires smoke profile")
        reuse_pin = check_oracle_reuse(job, execution, prepared, suite, inputs, pins)
    require([c["name"] for c in commands] == (["engine"] if reuse_pin else ["prepare", "oracle", "engine"]),
            "missing/reordered commands")
    for command in commands:
        require(command == load(job / f"{command['name']}.command.json"), "command receipt differs")
        require(command["exit_code"] == (execution["engine_exit_code"] if command["name"] == "engine" else 0), "producer command failed")
        argv = command["command"]
        if argv[:2] == ["/usr/bin/time", "-l"]:
            argv = argv[2:]
        require(len(argv) == 5, "unexpected producer/consumer command arguments")
        binary_path = resolve(job, argv[0])
        require(any(Path(p).resolve() == binary_path for p in pins), "command executable not pinned")
        model_path = resolve(job, execution["model_entry"])
        require(model_path == resolve(job, inputs["model_files"][0]["path"]), "command model differs from checked entry")
        if command["name"] == "prepare":
            require(argv[1] == "--prepare" and resolve(job, argv[2]) == model_path
                    and resolve(job, argv[3]) == corpus_path and resolve(job, argv[4]) == job / "prepared.json",
                    "prepare command inputs differ")
            require(binary_path == resolve(job, inputs["binary"]["path"]), "prepare used another binary")
        else:
            require(resolve(job, argv[1]) == model_path and argv[2] == suite["backend"], "command model/backend differs")
            expected_input = job / ("prepared.json" if command["name"] == "oracle" else "suite.json")
            require(resolve(job, argv[3]) == expected_input and resolve(job, argv[4]) == job / ("oracle" if command["name"] == "oracle" else "engine"),
                    "command input/output paths differ")
            if command["name"] == "engine":
                require(binary_path == resolve(job, inputs["binary"]["path"]), "engine command binary differs")
    # Recheck inputs after raw processing. The source/model bytes need not be
    # read again: file identity catches mutation during this bounded review.
    for pin in [inputs["suite"], inputs["prepared"], inputs["corpus"], inputs["binary"], *inputs["sources"], *inputs["model_files"],
                *(pin for pair in inputs["references"] for pin in pair.values())]:
        check_pin(pin, job, full_hash=False)
    require(all(stamp(path) == identity for path, identity in evidence_stamps.items()), "job report changed during review")
    attempted_count = sum(c["attempted"] for c in reviews)
    internal_checked = 0 if stopped else len(internal_ids)
    if smoke:
        require({str(p.relative_to(job)) for p in job.rglob("*.f32")} == {r["path"] for r in raw_files},
                "unreviewed or missing smoke raw files")
    return {"schema_version": 1, "valid_evidence": True, "passed": passed,
            "classification": "accepted" if passed else "valid_numeric_or_internal_rejection",
            "backend": suite["backend"], "cases_checked": attempted_count, "rows_checked": ROWS * attempted_count,
            "coverage_profile": profile, "cases_planned": case_count, "selected_case_ids": ids,
            "unrun_case_ids": [c["id"] for c in reviews if not c["attempted"]],
            "oracle_cases_checked": case_count, "oracle_rows_checked": ROWS * case_count,
            "numeric_counts": report.get("numeric_counts"),
            "state_phase_status": report.get("state_phase_status", "complete"),
            "early_stop": report.get("early_stop"),
            "oracle_reuse": reuse_pin,
            "oracle_micro_batch_tokens": 8, "oracle_prefill_chunk_tokens": 128,
            "cpu_attention_policy": cpu_policy,
            "internal_case_ids": internal_ids,
            "internal_modes_checked": internal_checked * 2, "internal_checks_checked": internal_checked * 2 * len(CHECKS),
            "metric_absolute_tolerance": TOLERANCE, "nll_gate": None,
            "limits": report["limits"], "numpy_version": np.__version__,
            "validator_sha256": digest(Path(__file__)), "cases": reviews,
            "job_directory": str(job), "raw_files": raw_files, "pilot": pilot,
            "suite_id": suite["suite_id"], "prepared_sha256": digest(job / "prepared.json"),
            "model_payload_reads": 0,
            "scope": "same-backend raw logits, fixed teacher history, receipt identities, and declared internal checks; no independent semantic task score"}


def main():
    require(len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2] == "--pilot")
            or (len(sys.argv) == 4 and sys.argv[2] == "--profile" and sys.argv[3] in SMOKE_INTERNALS),
            "usage: verify_results.py JOB_DIR [--pilot | --profile smoke-6 | --profile smoke-6-short-state]")
    job = Path(sys.argv[1]).resolve(strict=True)
    output = job / "independent-review.json"
    require(not output.exists(), "refusing to replace independent review")
    try:
        result = review(job, pilot=len(sys.argv) == 3,
                        profile=sys.argv[3] if len(sys.argv) == 4 else "extended-24")
    except Exception as error:
        result = {"schema_version": 1, "valid_evidence": False, "passed": False,
                  "classification": "invalid_or_incomplete", "error": str(error),
                  "validator_sha256": digest(Path(__file__))}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, sort_keys=True))
    return 0 if result["valid_evidence"] else 1


if __name__ == "__main__":
    sys.exit(main())
