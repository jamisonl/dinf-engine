#!/usr/bin/env python3
"""Describe FULL -> Q8_0/Q4_K_M oracle differences on the same fixed history.

Usage: compare_tiers.py --full JOB --high JOB --low JOB --output NEW.json
Requires all three engine jobs to have complete, passing independent reviews.
Supports legacy extended-24 and fresh smoke jobs with all six numeric cases
and every declared state check passed (smoke-6:36; smoke-6-short-state:12). Full24-to-six oracle reuse stays unreported.
Reads F32LE rows or verified lossless zstd archives without loading a model.
KL and NLL differences describe this corpus; they have no acceptance threshold.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from cpu_reference_policy import POLICY, REVISION, PATCH_SHA256

ROWS = 128
INTERNAL = ["conversation-512", "json-2048", "retrieval-4096"]
SUITE = "text-core-three-tier-v1"
SMOKE_CASES = ["prose-512", "conversation-512", "json-2048", "rust-2048",
               "retrieval-4096", "multilingual-4096"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference_policy(backend, policy):
    """Compare runtime bytes across tiers; copied paths may differ."""
    if backend == "metal":
        require(policy is None, "Metal oracle unexpectedly declares a CPU policy")
        return {"id": "stock-metal"}
    require(isinstance(policy, dict) and policy.get("schema_version") == 1
            and policy.get("id") == POLICY and policy.get("source_revision") == REVISION
            and policy.get("source_patch_sha256") == PATCH_SHA256,
            "CPU tier comparison requires the proved reference repair")
    result = {key: policy[key] for key in ("id", "source_revision", "source_patch_sha256")}
    for key in ("plugin", "blas_plugin"):
        record = policy[key]
        require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64
                and set(record["sha256"]) <= set("0123456789abcdef")
                and type(record["size_bytes"]) is int and record["size_bytes"] > 0,
                "invalid CPU runtime identity")
        result[key] = {name: record[name] for name in ("sha256", "size_bytes")}
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    require(path.stat().st_size <= 64 * 1024**2, f"JSON exceeds bound: {path}")
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_constant=reject)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    st = path.stat()
    return (st.st_size, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns)


def contained(job, relative):
    require(isinstance(relative, str), "relative filename must be text")
    parts = PurePosixPath(relative).parts
    require(parts and not relative.startswith("/") and "\\" not in relative
            and all(p not in (".", "..") for p in parts)
            and PurePosixPath(relative).as_posix() == relative, "unsafe relative output path")
    path = job / relative
    require(job in path.resolve().parents, "output path escapes job")
    require(not path.is_symlink(), "output symlinks are not accepted")
    for parent in path.parents:
        if parent == job:
            break
        require(not parent.is_symlink(), "output parent symlinks are not accepted")
    return path


class Job:
    def __init__(self, directory, tier):
        self.path = directory.resolve(strict=True)
        self.json_pins = {}
        self.source_identity = {}
        self.prepared = self.json("prepared.json")
        self.review = self.json("independent-review.json")
        self.engine = self.json("engine/report.json")
        self.oracle_suite = self.json("oracle/suite.json")
        self.execution = self.json("execution.json")
        p, r, e, o, x = self.prepared, self.review, self.engine, self.oracle_suite, self.execution
        require(r["valid_evidence"] is True and r["passed"] is True and r["pilot"] is False,
                "tier comparison requires passing nonpilot engine evidence")
        self.profile = r.get("coverage_profile", "extended-24")
        require(self.profile in ("extended-24", "smoke-6", "smoke-6-short-state")
                and e.get("coverage_profile", "extended-24") == self.profile
                and x.get("coverage_profile", "extended-24") == self.profile, "coverage profile differs")
        smoke = self.profile in ("smoke-6", "smoke-6-short-state")
        internal = ["conversation-512"] if self.profile == "smoke-6-short-state" else INTERNAL
        state_count = len(internal)
        self.case_count = 6 if smoke else 24
        if smoke:
            require(x.get("oracle_reuse") is None,
                    "smoke tier comparison supports fresh six-case oracle jobs only; reuse stays unreported")
            require(e.get("fail_fast") is True and e.get("execution_policy") == "numeric_then_state"
                    and e.get("state_phase_status") == r.get("state_phase_status") == "complete"
                    and e.get("early_stop") is None and r.get("early_stop") is None,
                    "smoke tier comparison requires complete numeric and state phases")
            numeric = {"planned": 6, "attempted": 6, "passed": 6, "failed": 0, "unrun": 0}
            require(e.get("numeric_counts") == r.get("numeric_counts") == numeric
                    and e.get("selected_case_ids") == r.get("selected_case_ids") == SMOKE_CASES,
                    "smoke numeric coverage incomplete")
        require(r["suite_id"] == p["suite_id"] == e["suite_id"] == SUITE, "unexpected suite")
        require(r["job_directory"] == str(self.path) and r["prepared_sha256"] == self.json_pins["prepared.json"], "review input binding differs")
        require(r["cases_checked"] == self.case_count and r["rows_checked"] == self.case_count * ROWS
                and r["internal_modes_checked"] == 2 * state_count and r["internal_checks_checked"] == 12 * state_count
                and r["internal_case_ids"] == internal, "incomplete engine review scope")
        expected_counts = {"planned": self.case_count, "attempted": self.case_count,
                           "passed": self.case_count, "failed": 0}
        expected_internal = {"planned": state_count, "attempted": state_count, "passed": state_count, "failed": 0}
        if smoke:
            expected_counts.update(unrun=0, incomplete=0)
            expected_internal["unrun"] = 0
        require(e["valid"] is True and e["passed"] is True and e["post_run_pins_verified"] is True
                and e["counts"] == expected_counts
                and e["internal_case_ids"] == internal
                and e["internal_counts"] == expected_internal,
                "engine acceptance receipt differs")
        require(x["engine_valid"] is True and x["engine_passed"] is True and x["engine_exit_code"] == 0
                and x["post_run_pins_verified"] is True, "runner did not finish the engine check")
        require(x["artifact"]["tier"] == tier, "job has the wrong precision tier")
        if tier != "FULL":
            require(x["artifact"]["quantization"] == tier, "quantized artifact label differs")
        else:
            require(x["artifact"]["quantization"] in ("BF16", "F32", "F16"), "FULL must use unquantized stored weights")
        require(x["backend"] == e["backend"] == o["backend"] == r["backend"]
                and x["backend"] in ("cpu", "metal"), "job backend differs")
        self.backend, self.model_id = x["backend"], x["model_id"]
        self.cpu_policy = o.get("cpu_attention_policy")
        require(r.get("cpu_attention_policy") == x.get("cpu_attention_policy") == self.cpu_policy,
                "CPU policy differs across saved receipts")
        self.reference_policy = reference_policy(self.backend, self.cpu_policy)
        artifact = x["artifact"]
        artifact_files = [artifact["preparation"]["expected_output"]] if artifact.get("requires_preparation") else artifact["files"]
        expected_files = [{"name": Path(f["filename"]).name, "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in artifact_files]
        require(p["model_files"] == expected_files, "prepared model manifest differs from selected artifact")
        require(p["model_files"] == e["model_files"] == o["model_files"], "model manifests differ")
        require(e["token_identity_sha256"] == p["token_identity_sha256"], "engine token identity differs")
        require(p["model_sha256"] == p["model_files"][0]["sha256"], "entry hash differs")
        require(p["rows"] == ROWS and len(p["cases"]) == self.case_count, "prepared grid differs; smoke requires fresh six cases")
        self.ids = [c["id"] for c in p["cases"]]
        require(len(set(self.ids)) == self.case_count, "duplicate prepared case")
        domains = set(c["domain"] for c in p["cases"])
        if smoke:
            require(self.ids == SMOKE_CASES
                    and all((case["domain"], case["context_tokens"]) == (label.rsplit("-", 1)[0], int(label.rsplit("-", 1)[1]))
                            for case, label in zip(p["cases"], SMOKE_CASES)), "smoke domain/context grid differs")
        else:
            require(len(domains) == 8 and all(sorted(c["context_tokens"] for c in p["cases"] if c["domain"] == domain) == [512, 2048, 4096] for domain in domains), "incomplete domain/context grid")
        require(o["status"] == "complete" and o["expected_cases"] == o["completed_cases"] == self.case_count
                and o["prepared_sha256"] == self.json_pins["prepared.json"] and o["corpus_sha256"] == p["corpus_sha256"], "oracle suite incomplete")
        require(o["new_context_per_case"] is True and o["kv_dtype"] == "F16"
                and o["prefill_chunk_tokens"] == 128 and o["threads"] == 8, "oracle context/math scope differs")
        require(type(o.get("micro_batch_tokens")) is int and o["micro_batch_tokens"] == 8,
                "oracle suite micro batch must be eight tokens")
        require(type(e.get("oracle_micro_batch_tokens")) is int and e["oracle_micro_batch_tokens"] == 8
                and type(r.get("oracle_micro_batch_tokens")) is int and r["oracle_micro_batch_tokens"] == 8,
                "engine/review oracle micro batch must be eight tokens")
        storage = o["weight_storage"]["tensor_storage"]
        types = {item["type"] for item in storage}
        require(storage and all(type(item["tensor_count"]) is int and item["tensor_count"] > 0
                                and type(item["encoded_tensor_bytes"]) is int and item["encoded_tensor_bytes"] > 0 for item in storage),
                "invalid actual weight storage histogram")
        require(sum(item["tensor_count"] for item in storage) == o["weight_storage"]["total_tensor_count"], "storage tensor count differs")
        if tier == "FULL":
            require(types <= {"bf16", "f16", "f32"} and artifact["quantization"].lower() in types,
                    "FULL oracle contains quantized weights or lacks its stated floating dtype")
        else:
            require(("q8_0" if tier == "Q8_0" else "q4_K") in types,
                    "oracle storage lacks the stated quantization type")
        for cases in (r["cases"], e["cases"], o["cases"]):
            require([c["id"] for c in cases] == self.ids, "case order differs")
        require(all(c["valid_evidence"] is True and c["passed"] is True for c in r["cases"]), "per-case engine evidence failed")
        require(r["nll_gate"] is None, "NLL gate unexpectedly used")
        self.raw = {item["path"]: item for item in r["raw_files"]}
        expected = {f"{folder}/{case}{suffix}" for case in self.ids for folder, suffix in [("oracle", ".f32"), ("engine", ".actual.f32")]}
        require(len(r["raw_files"]) == 2 * self.case_count and set(self.raw) == expected, "review raw-file set differs")
        for item in self.raw.values():
            require(item["size_bytes"] == 128 * p["vocab_size"] * 4, "reviewed row byte count differs")
        self.archive = self.json("archive.json") if (self.path / "archive.json").is_file() else None
        self.archived = {}
        if self.archive is not None:
            a = self.archive
            require(a["suite_id"] == SUITE and a["prepared_sha256"] == self.json_pins["prepared.json"]
                    and a["independent_review"]["sha256"] == self.json_pins["independent-review.json"], "archive input binding differs")
            self.archived = {record["raw_path"]: record for record in a["files"]}
            require(len(a["files"]) == 2 * self.case_count and set(self.archived) == expected, "archive set differs from review")
            for relative, record in self.archived.items():
                item = self.raw[relative]
                require(record["raw_sha256"] == record["decompressed_sha256"] == item["sha256"]
                        and record["raw_bytes"] == record["decompressed_bytes"] == item["size_bytes"], "archive raw binding differs")
        self.oracles = []
        for case, indexed, checked in zip(p["cases"], o["cases"], r["cases"]):
            relative = f"oracle/{case['id']}.json"
            oracle = self.json(relative)
            require(oracle.get("cpu_attention_policy") == self.cpu_policy,
                    "per-case CPU reference policy differs")
            require(self.json_pins[relative] == indexed["oracle_json_sha256"], "oracle JSON hash differs")
            require(oracle["case_id"] == case["id"] and oracle["context_tokens"] == case["context_tokens"]
                    and oracle["backend"] == self.backend and oracle["vocab_size"] == p["vocab_size"], "oracle case scope differs")
            require(type(oracle.get("micro_batch_tokens")) is int and oracle["micro_batch_tokens"] == 8,
                    "oracle case micro batch must be eight tokens")
            require(oracle["prompt_tokens"] == case["prompt_tokens"] and oracle["teacher_forcing_tokens"] == case["teacher_forcing_tokens"]
                    and len(case["teacher_forcing_tokens"]) == 128 and oracle["tokenization_verified"] is True, "oracle fixed history differs")
            require(oracle["prepared_sha256"] == self.json_pins["prepared.json"] and oracle["model_files"] == p["model_files"]
                    and oracle["model_sha256"] == p["model_sha256"], "oracle model/input binding differs")
            require(oracle["source"] == o["source"] and oracle["producer_source_sha256"] == o["producer_source_sha256"], "oracle producer differs")
            raw_name = f"oracle/{case['id']}.f32"
            require(oracle["logits_file"] == f"{case['id']}.f32"
                    and self.raw[raw_name]["sha256"] == oracle["oracle_raw_sha256"] == indexed["oracle_raw_sha256"] == checked["oracle_raw_sha256"], "oracle raw binding differs")
            require(len(oracle["continuation_tokens"]) == 128, "oracle argmax count differs")
            self.oracles.append(oracle)

    def json(self, relative):
        path = contained(self.path, relative)
        before = identity(path)
        data = load(path)
        self.json_pins[relative] = digest(path)
        require(identity(path) == before, "JSON changed during read")
        self.source_identity[path] = before
        return data

    def recheck(self):
        require(all(identity(path) == value for path, value in self.source_identity.items()), "tier evidence changed during comparison")

    def evidence(self):
        return {"job_directory": str(self.path), "model_id": self.model_id, "backend": self.backend,
                "artifact_id": self.execution["artifact"]["id"], "tier": self.execution["artifact"]["tier"],
                "stored_precision": self.execution["artifact"]["quantization"],
                "reference_policy": self.reference_policy,
                "model_files": self.prepared["model_files"], "json_sha256": self.json_pins,
                "oracle_source": self.oracle_suite["source"], "weight_storage": self.oracle_suite.get("weight_storage"),
                "coverage_profile": self.profile,
                "engine_correctness": {"valid_evidence": True, "passed": True,
                                       "cases": self.case_count, "rows": self.case_count * ROWS}}


class HashedReader:
    def __init__(self, stream, expected):
        self.stream, self.expected = stream, expected
        self.hash, self.count = hashlib.sha256(), 0

    def read(self, size):
        require(self.count + size <= self.expected["size_bytes"], "read exceeds declared raw size")
        parts, left = [], size
        while left:
            data = self.stream.read(left)
            require(data, "raw/decompressed stream is short")
            parts.append(data)
            self.hash.update(data)
            self.count += len(data)
            left -= len(data)
        return b"".join(parts)

    def finish(self):
        require(self.count == self.expected["size_bytes"] and self.stream.read(1) == b"", "raw/decompressed byte count differs")
        require(self.hash.hexdigest() == self.expected["sha256"], "raw/decompressed SHA256 differs")


@contextmanager
def raw_reader(job, relative, zstd, evidence):
    expected = job.raw[relative]
    path = contained(job.path, relative)
    if path.is_file():
        before = identity(path)
        require(before[0] == expected["size_bytes"], "raw byte count differs")
        with path.open("rb") as stream:
            reader = HashedReader(stream, expected)
            yield reader
            reader.finish()
        require(identity(path) == before, "raw changed during comparison")
        job.source_identity[path] = before
        evidence.append({"job_directory": str(job.path), **expected, "storage": "raw"})
        return
    require(relative in job.archived, "raw missing without a bound archive")
    record = job.archived[relative]
    require(record["compressed_path"] == relative + ".zst", "archive filename differs")
    compressed = contained(job.path, record["compressed_path"])
    before = identity(compressed)
    require(before[0] == record["compressed_bytes"] and digest(compressed) == record["compressed_sha256"], "compressed bytes/hash differ")
    require(zstd is not None, "zstd is required for archived rows")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(zstd), "-q", "-d", "-c", "--", str(compressed)], stdout=subprocess.PIPE, stderr=errors)
        try:
            reader = HashedReader(process.stdout, expected)
            yield reader
            reader.finish()
            require(process.wait() == 0, "zstd decompression failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    require(identity(compressed) == before, "archive changed during comparison")
    job.source_identity[compressed] = before
    evidence.append({"job_directory": str(job.path), **expected, "storage": "verified_zstd",
                     "compressed_path": record["compressed_path"], "compressed_sha256": record["compressed_sha256"]})


def row_metrics(full, quant, teacher, row):
    require(full.shape == quant.shape and full.ndim == 1 and len(full) > 0, "logit row shapes differ")
    require(np.isfinite(full).all() and np.isfinite(quant).all(), "nonfinite oracle logit")
    require(type(teacher) is int and 0 <= teacher < len(full), "teacher outside vocabulary")
    full, quant = full.astype(np.float64), quant.astype(np.float64)
    f = full - full.max()
    q = quant - quant.max()
    f -= np.log(np.exp(f).sum(dtype=np.float64))
    q -= np.log(np.exp(q).sum(dtype=np.float64))
    return {"row": row, "teacher_token": teacher, "full_top": int(full.argmax()), "quant_top": int(quant.argmax()),
            "kl_full_to_quant": max(0.0, float(np.sum(np.exp(f) * (f-q), dtype=np.float64))),
            "full_nll": float(-f[teacher]), "quant_nll": float(-q[teacher]),
            "nll_delta_quant_minus_full": float(f[teacher]-q[teacher])}


def aggregate(rows):
    require(rows, "empty comparison")
    n = len(rows)
    matches = sum(row["full_top"] == row["quant_top"] for row in rows)
    return {"rows": n, "matching_argmax": matches, "argmax_agreement": matches/n,
            "mean_kl_full_to_quant": math.fsum(row["kl_full_to_quant"] for row in rows)/n,
            "max_row_kl_full_to_quant": max(row["kl_full_to_quant"] for row in rows),
            "mean_full_nll": math.fsum(row["full_nll"] for row in rows)/n,
            "mean_quant_nll": math.fsum(row["quant_nll"] for row in rows)/n,
            "mean_nll_delta_quant_minus_full": math.fsum(row["nll_delta_quant_minus_full"] for row in rows)/n}


def compare(full_path, high_path, low_path, zstd=None):
    jobs = [Job(full_path, "FULL"), Job(high_path, "Q8_0"), Job(low_path, "Q4_K_M")]
    require(len({job.path for job in jobs}) == 3, "tier jobs must be distinct")
    full = jobs[0]
    for quant in jobs[1:]:
        require(quant.profile == full.profile, "coverage profile differs across tiers")
        require(quant.engine["input_pins"]["binary"]["sha256"] == full.engine["input_pins"]["binary"]["sha256"]
                and quant.engine["compiled_helper_sources"] == full.engine["compiled_helper_sources"],
                "tested engine build differs across tiers")
        require(quant.model_id == full.model_id and quant.backend == full.backend, "checkpoint/backend differs across tiers")
        require(quant.reference_policy == full.reference_policy, "reference precision policy differs across tiers")
        require(quant.execution["source_pins"] == full.execution["source_pins"], "engine/reference source or runtime pins differ across tiers")
        for key in ("architecture", "vocab_size", "corpus_sha256", "token_identity_sha256", "cases"):
            require(quant.prepared[key] == full.prepared[key], f"tier prepared {key} differs")
        require(quant.oracle_suite["source"] == full.oracle_suite["source"]
                and quant.oracle_suite["producer_source_sha256"] == full.oracle_suite["producer_source_sha256"], "reference implementation differs across tiers")
        require(quant.prepared["model_files"] != full.prepared["model_files"], "FULL and quantized tiers point to identical artifacts")
    zstd = Path(zstd).resolve(strict=True) if zstd is not None else None
    zstd_pin = {"path": str(zstd), "sha256": digest(zstd)} if zstd else None
    row_bytes = full.prepared["vocab_size"] * 4
    inputs, cases = [], []
    for index, case in enumerate(full.prepared["cases"]):
        relative = f"oracle/{case['id']}.f32"
        pair_rows = [[], []]
        with raw_reader(jobs[0], relative, zstd, inputs) as f, raw_reader(jobs[1], relative, zstd, inputs) as h, raw_reader(jobs[2], relative, zstd, inputs) as l:
            for row, teacher in enumerate(case["teacher_forcing_tokens"]):
                vectors = [np.frombuffer(reader.read(row_bytes), dtype="<f4") for reader in (f, h, l)]
                for job, vector in zip(jobs, vectors):
                    require(np.isfinite(vector).all() and int(vector.argmax()) == job.oracles[index]["continuation_tokens"][row], "oracle raw argmax differs from JSON")
                for target, values in zip(pair_rows, vectors[1:]):
                    target.append(row_metrics(vectors[0], values, teacher, row))
        cases.append({"id": case["id"], "domain": case["domain"], "context_tokens": case["context_tokens"],
                      "prompt_tokens": len(case["prompt_tokens"]),
                      **{tier: {**aggregate(rows), "comparisons": rows} for tier, rows in zip(("Q8_0", "Q4_K_M"), pair_rows)}})
    summary = {}
    for tier in ("Q8_0", "Q4_K_M"):
        all_rows = [row for case in cases for row in case[tier]["comparisons"]]
        summary[tier] = {"all": aggregate(all_rows),
                         "domains": {d: aggregate([r for c in cases if c["domain"] == d for r in c[tier]["comparisons"]]) for d in sorted({c["domain"] for c in cases})},
                         "contexts": {str(n): aggregate([r for c in cases if c["context_tokens"] == n for r in c[tier]["comparisons"]]) for n in (512, 2048, 4096)}}
    for job in jobs:
        job.recheck()
    if zstd_pin is not None:
        require(digest(zstd) == zstd_pin["sha256"], "zstd changed during comparison")
    return {"schema_version": 1, "valid_evidence": True, "status": "complete", "suite_id": SUITE,
            "model_id": full.model_id, "backend": full.backend, "coverage_profile": full.profile,
            "cases_compared": full.case_count, "rows_per_pair": full.case_count * ROWS,
            "reference_policy": full.reference_policy,
            "quantization_acceptance_gate": None, "nll_acceptance_gate": None,
            "engine_correctness": {"required": True, "all_three_tiers_passed": True},
            "scope": "Descriptive same-backend reference-runtime precision differences on fixed teacher histories, including kernel and activation-quantization effects. FULL retains the published unquantized storage dtype. This is not isolated weight-rounding error, a task-quality score, or a cross-model ranking.",
            "token_identity_sha256": full.prepared["token_identity_sha256"], "corpus_sha256": full.prepared["corpus_sha256"],
            "sources": [job.evidence() for job in jobs], "raw_inputs": inputs,
            "runtime": {"numpy": np.__version__, "python": sys.version, "zstd": zstd_pin},
            "helper_sha256": digest(Path(__file__)), "summary": summary, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("full", "high", "low", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    try:
        result = compare(args.full, args.high, args.low, shutil.which("zstd"))
    except Exception as error:
        result = {"schema_version": 1, "valid_evidence": False, "status": "invalid_or_incomplete",
                  "error": str(error), "helper_sha256": digest(Path(__file__))}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "summary", "sources", "raw_inputs")}, sort_keys=True))
    return 0 if result["valid_evidence"] else 1


if __name__ == "__main__":
    sys.exit(main())
