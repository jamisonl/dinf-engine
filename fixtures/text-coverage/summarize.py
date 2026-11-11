#!/usr/bin/env python3
"""Summarize the 36 text-coverage cells from checked, retained evidence.

This tool does not run a model or recompute logits. It checks the independent
review, its row scores, receipt bindings and retained raw/archive hashes.
Every attempt remains listed. An older build, pilot, missing evidence or a
valid numeric rejection can never become a current-build pass.
"""

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import cpu_reference_policy as cpu_policy_check


ROWS = 128
SMOKE = ["prose-512", "conversation-512", "json-2048", "rust-2048",
         "retrieval-4096", "multilingual-4096"]
INTERNAL = ["conversation-512", "json-2048", "retrieval-4096"]
SHORT_INTERNAL = ["conversation-512"]
SMOKE_PROFILES = ("smoke-6", "smoke-6-short-state")
TIERS = ["FULL", "Q8_0", "Q4_K_M"]
HELPERS = ["qualify_matrix.rs", "qualify_matrix/common.rs",
           "qualify_matrix/metrics.rs", "qualify_matrix/prepare.rs",
           "qualify_matrix/session_checks.rs", "qualify_matrix/coverage.rs"]
LIMITS = {"rows": ROWS, "min_argmax_fraction": .95, "mean_kl": .02,
          "max_kl": .1, "nll_gate": None, "kl_direction": "reference || actual"}
CASE_KEYS = ["id", "domain", "context_tokens", "rendered_prompt_utf8",
             "continuation_utf8", "prompt_original_ids", "prompt_tokens",
             "continuation_original_ids", "teacher_forcing_tokens",
             "prompt_add_special", "continuation_add_special", "parse_special"]
METRICS = ["mean_kl", "max_kl", "reference_mean_nll", "actual_mean_nll",
           "mean_nll_delta"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON: {value}")


def identity(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def contained(root, name):
    require(isinstance(name, str), "relative path must be text")
    parts = PurePosixPath(name).parts
    require(parts and not name.startswith("/") and "\\" not in name
            and "\0" not in name and all(p not in (".", "..") for p in parts)
            and PurePosixPath(name).as_posix() == name, "unsafe relative path")
    path = root / name
    require(root in path.resolve().parents, "path escapes job")
    require(not path.is_symlink(), "output symlink is not accepted")
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), "output parent symlink is not accepted")
    return path


class Reader:
    """Hash each stable file once; recheck identities before returning."""

    def __init__(self):
        self.pins = {}

    def digest(self, path):
        path = Path(path).resolve(strict=True)
        stamp = identity(path)
        previous = self.pins.get(path)
        if previous is not None:
            require(previous[0] == stamp, f"file changed during summary: {path}")
            return previous[1]
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        require(identity(path) == stamp, f"file changed while hashing: {path}")
        self.pins[path] = (stamp, sha)
        return sha

    def load(self, path):
        path = Path(path)
        require(path.stat().st_size <= 128 * 1024**2, f"JSON exceeds bound: {path}")
        self.digest(path)
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
        self.digest(path)
        return value

    def check(self, path, sha):
        require(self.digest(path) == sha, f"SHA256 differs: {path}")

    def finish(self):
        for path, (stamp, _) in self.pins.items():
            require(identity(path) == stamp, f"evidence changed during summary: {path}")


def model_files(artifact):
    source = ([artifact["preparation"]["expected_output"]]
              if artifact.get("requires_preparation") else artifact["files"])
    result = []
    for item in source:
        name = item["filename"]
        parts = PurePosixPath(name).parts
        require(parts and not name.startswith("/") and "\\" not in name
                and "\0" not in name and all(p not in (".", "..") for p in parts)
                and PurePosixPath(name).as_posix() == name,
                "unsafe model repository path")
        require(type(item["size_bytes"]) is int and item["size_bytes"] > 0
                and isinstance(item["sha256"], str) and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]),
                "invalid model size/hash")
        # Hub filenames can include a repository subdirectory. The downloader
        # and runtime keep the canonical shard basenames beside the entry.
        result.append({"name": parts[-1], "size_bytes": item["size_bytes"],
                       "sha256": item["sha256"]})
    require(result and len({x["name"] for x in result}) == len(result),
            "empty or duplicate model manifest")
    return result


def near(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isfinite(expected)
            and abs(actual - expected) <= 1e-10, f"metric differs: {label}")


def scores(checked, result, case, vocab):
    """Recompute aggregate scores from all independently reviewed rows."""
    rows = checked["rows"]
    require(len(rows) == ROWS and len(result["reference"]["comparisons"]) == ROWS,
            "incomplete reviewed rows")
    for index, (row, actual) in enumerate(zip(rows, result["reference"]["comparisons"])):
        require(row["row"] == index and row["teacher_token"] == case["teacher_forcing_tokens"][index],
                "review row/history differs")
        for key in ("row", "teacher_token", "reference_top", "actual_top"):
            require(type(row[key]) is int and row[key] == actual[key],
                    "engine/review token rows differ")
        require(all(0 <= row[key] < vocab for key in ("teacher_token", "reference_top", "actual_top")),
                "reviewed token is outside vocabulary")
        for key in ("kl", "max_abs", "rms", "reference_nll", "actual_nll", "nll_delta"):
            near(actual[key], row[key], key)
        near(row["nll_delta"], row["actual_nll"] - row["reference_nll"], "row NLL delta")
    matches = sum(r["reference_top"] == r["actual_top"] for r in rows)
    values = {
        "matching_argmax": matches,
        "mean_kl": math.fsum(r["kl"] for r in rows) / ROWS,
        "max_kl": max(r["kl"] for r in rows),
        "reference_mean_nll": math.fsum(r["reference_nll"] for r in rows) / ROWS,
        "actual_mean_nll": math.fsum(r["actual_nll"] for r in rows) / ROWS,
        "mean_nll_delta": math.fsum(r["nll_delta"] for r in rows) / ROWS,
    }
    for value in (checked, result["reference"]):
        require(value["matching_argmax"] == matches, "argmax count differs")
        for key in METRICS:
            near(value[key], values[key], key)
    numeric = matches / ROWS >= .95 and values["mean_kl"] <= .02 and values["max_kl"] <= .1
    require(checked["numeric_passed"] is numeric and result["reference"]["passed"] is numeric,
            "numeric acceptance differs")
    return values, numeric


def retained_rows(reader, job, review, prepared_sha, vocab, case_ids, attempted_ids=None):
    actual_ids = case_ids if attempted_ids is None else attempted_ids
    expected = {f"oracle/{case}.f32" for case in case_ids} | {f"engine/{case}.actual.f32" for case in actual_ids}
    records = {item["path"]: item for item in review["raw_files"]}
    require(len(records) == len(review["raw_files"]) == len(expected) and set(records) == expected,
            "raw-file set differs")
    archived = {}
    archive_sha = None
    archive_path = contained(job, "archive.json")
    if archive_path.is_file():
        archive = reader.load(archive_path)
        archive_sha = reader.digest(archive_path)
        require(archive["state"] == "verified_archives_ready_for_raw_removal"
                and archive["suite_id"] == review["suite_id"]
                and archive["prepared_sha256"] == prepared_sha
                and archive["independent_review"]["sha256"] == reader.digest(job / "independent-review.json"),
                "archive is not bound to this review")
        archived = {item["raw_path"]: item for item in archive["files"]}
        require(len(archived) == len(archive["files"]) == len(expected) and set(archived) == expected,
                "archive file set differs")
    counts = Counter()
    for relative, item in records.items():
        path = contained(job, relative)
        require(item["size_bytes"] == ROWS * vocab * 4, "raw size differs from vocabulary")
        if path.is_file():
            require(path.stat().st_size == item["size_bytes"], "raw byte count differs")
            reader.check(path, item["sha256"])
            counts["raw"] += 1
        else:
            require(relative in archived, f"raw and verified archive both missing: {relative}")
            record = archived[relative]
            require(record["raw_sha256"] == record["decompressed_sha256"] == item["sha256"]
                    and record["raw_bytes"] == record["decompressed_bytes"] == item["size_bytes"],
                    "archive raw binding differs")
            compressed = contained(job, record["compressed_path"])
            require(record["compressed_path"] == relative + ".zst"
                    and compressed.stat().st_size == record["compressed_bytes"],
                    "compressed path/byte count differs")
            reader.check(compressed, record["compressed_sha256"])
            counts["verified_archive"] += 1
    return {"complete": True, "files": len(expected), "raw_files": counts["raw"],
            "archived_files": counts["verified_archive"], "archive_sha256": archive_sha,
            "archive_check": "compressed SHA256 and prior decompressed SHA256 receipt; no new decompression"}


def check_skipped_internal(value, selected):
    require(value == {"required": selected, "attempted": False, "valid": None,
                      "passed": None, "modes": [],
                      "status": "not_run_after_numeric_failure" if selected else "not_selected"}
            and value["required"] is selected and value["attempted"] is False,
            "skipped state check has a verdict")


def check_smoke_scope(report, review, cases, stopped, internal_ids=INTERNAL):
    attempted = [c for c in cases if c["attempted"]]
    n = len(attempted)
    failures = [c for c in attempted if c["numeric_passed"] is False]
    if stopped:
        require(len(failures) == 1 and failures[0]["id"] == SMOKE[n - 1]
                and all(c["numeric_passed"] is True for c in attempted[:-1]),
                "smoke did not stop at first numeric rejection")
        require(report["early_stop"] == {"phase": "numeric", "case_id": SMOKE[n - 1],
                                         "reason": "numeric_failure"}, "smoke stop reason differs")
    else:
        require(n == 6 and not failures and report["state_phase_status"] == "complete"
                and report.get("early_stop") is None, "smoke numeric/state work incomplete")
    require(report["numeric_counts"] == review["numeric_counts"] == {
            "planned": 6, "attempted": n, "passed": n - len(failures),
            "failed": len(failures), "unrun": 6 - n}, "smoke numeric counts differ")
    planned = len(internal_ids)
    checked = 0 if stopped else planned
    internal_passed = sum(c["internal_passed"] is True for c in cases)
    require(report["internal_counts"] == {"planned": planned, "attempted": checked,
            "passed": internal_passed, "failed": checked - internal_passed,
            "unrun": planned - checked}, "smoke state counts differ")
    require(report["counts"] == {"planned": 6, "attempted": n,
            "passed": sum(c["passed"] is True for c in cases),
            "failed": sum(c["passed"] is False for c in cases), "unrun": 6 - n,
            "incomplete": sum(c["attempted"] and c["passed"] is None for c in cases)},
            "smoke case counts differ")


def check_reuse(reader, job, reuse, prepared, run, pins, raw_bindings):
    """Read provenance, not original weight/logit files or changed old sources."""
    r = reader
    receipt_path = contained(job, "oracle-reuse.json")
    require(Path(reuse["path"]).resolve() == receipt_path, "reuse receipt is outside job")
    r.check(receipt_path, reuse["sha256"])
    receipt = r.load(receipt_path)
    require(receipt["schema_version"] == 1, "reuse schema differs")
    source = Path(receipt["source_job"]).resolve()
    require(source != job, "reuse source is current job")
    def document(pin, expected=None):
        path = Path(pin["path"]).resolve()
        require(expected is None or path == expected, "reuse provenance path differs")
        r.check(path, pin["sha256"])
        return r.load(path)
    old_pins = document(receipt["original_pins"], source / "pins.json")
    ref = document(receipt["reference_build"])
    require(any(Path(path).resolve() == Path(receipt["reference_build"]["path"]).resolve()
                and sha == receipt["reference_build"]["sha256"] for path, sha in pins.items()),
            "reference build missing from current pins")
    r.check(ref["binary"]["path"], ref["binary"]["sha256"])
    expected = {"prepared.json", "corpus.json", "oracle/suite.json",
                *(f"oracle/{case}{suffix}" for case in SMOKE for suffix in (".json", ".f32"))}
    files = receipt["preserved_files"]
    require(len(files) == len(expected) and {f["destination"] for f in files} == expected,
            "reuse preserved file set differs")
    original = {}
    for item in files:
        relative = item["destination"]; dest = contained(job, relative)
        src = Path(item["source"]["path"]).resolve()
        require(src != dest and (source in src.parents or relative == "corpus.json"), "reuse source path differs")
        expected_sha = item["source"]["sha256"]
        if relative.endswith(".f32"):
            require(raw_bindings.get(relative) == expected_sha, "reused raw hash differs from reviewed retained bytes")
        else:
            r.check(dest, expected_sha)
            r.check(src, expected_sha)
        original[relative] = src
    require(original["prepared.json"] == source / "prepared.json"
            and original["oracle/suite.json"] == source / "oracle/suite.json", "reuse original layout differs")
    require(set(receipt["original_commands"]) == {"prepare", "oracle"}, "reuse command set differs")
    for label, pin in receipt["original_commands"].items():
        event = document(pin, source / f"{label}.command.json")
        require(event["name"] == label and event["exit_code"] == 0 and event["timed_out"] is False,
                "reused producer did not close successfully")
        argv = event["command"]
        if argv[:2] == ["/usr/bin/time", "-l"]:
            argv = argv[2:]
        require(len(argv) == 5, "reused producer argv differs")
        binary = Path(argv[0]).resolve(); model = Path(run["model_entry"]).resolve()
        old_hashes = {sha for path, sha in old_pins.items() if Path(path).resolve() == binary}
        if label == "prepare":
            require(old_hashes == {prepared["preparation_binary_sha256"]}
                    and argv[1] == "--prepare" and Path(argv[2]).resolve() == model
                    and Path(argv[3]).resolve() == original["corpus.json"]
                    and Path(argv[4]).resolve() == original["prepared.json"], "old preparation provenance differs")
            require(all(sha in old_pins.values() for sha in prepared["preparation_source_sha256"].values()),
                    "old preparation source missing from original pins")
        else:
            require(binary == Path(ref["binary"]["path"]).resolve() and old_hashes == {ref["binary"]["sha256"]}
                    and Path(argv[1]).resolve() == model and argv[2] == run["backend"]
                    and Path(argv[3]).resolve() == original["prepared.json"]
                    and Path(argv[4]).resolve() == source / "oracle", "old oracle provenance differs")
    if receipt.get("admission") is not None:
        pin = receipt["admission"]; r.check(pin["path"], pin["sha256"])


class Matrix:
    def __init__(self, manifest_path, build_path, corpus_path, profile="extended-24"):
        require(profile in ("extended-24", *SMOKE_PROFILES), "unknown coverage profile")
        self.profile = profile
        self.smoke = profile in SMOKE_PROFILES
        self.internal_ids = SHORT_INTERNAL if profile == "smoke-6-short-state" else INTERNAL
        self.reader = Reader()
        r = self.reader
        self.manifest = r.load(manifest_path)
        self.build = r.load(build_path)
        self.corpus = r.load(corpus_path)
        self.input_pins = {str(Path(p).resolve()): r.digest(p)
                           for p in (manifest_path, build_path, corpus_path)}
        self.corpus_sha = r.digest(corpus_path)
        self.ids = [c["id"] for c in self.corpus["cases"]]
        require(self.corpus["schema_version"] == 1 and self.corpus["rows"] == ROWS
                and len(self.ids) == len(set(self.ids))
                and (len(self.ids) in (6, 24) if self.smoke else len(self.ids) == 24)
                and set(INTERNAL) <= set(self.ids), "corpus must contain all 24 cases")
        require(all(isinstance(name, str) and name and name[0].isascii() and name[0].isalnum()
                    and all(c.isascii() and (c.isalnum() or c in "-_") for c in name)
                    for name in self.ids), "unsafe case ID")
        if self.smoke:
            require(set(SMOKE) <= set(self.ids) and (len(self.ids) == 24 or self.ids == SMOKE), "smoke case selection differs")
        self.corpus_ids = self.ids
        self.ids = SMOKE if self.smoke else self.ids
        domains = {c["domain"] for c in self.corpus["cases"]}
        require((self.smoke and len(self.corpus_ids) == 6) or (len(domains) == 8 and all(sorted(c["context_tokens"] for c in self.corpus["cases"]
                if c["domain"] == domain) == [512, 2048, 4096] for domain in domains)),
                "corpus domain/context grid differs")
        self.required_pins = dict(self.build["source_pins"])
        self.required_pins.update(self.build["runtime_pins"])
        self.required_pins[self.build["binary"]["path"]] = self.build["binary"]["sha256"]
        require(self.build["source_pins"], "build has no source pins")
        for path, sha in self.required_pins.items():
            r.check(path, sha)
        self.helpers = {}
        for helper in HELPERS:
            found = [sha for path, sha in self.build["source_pins"].items()
                     if path.endswith("/examples/" + helper)]
            if helper == "qualify_matrix/coverage.rs" and not found and not self.smoke:
                continue  # Older explicitly selected extended builds predate the profile module.
            require(len(found) == 1, f"build does not pin helper: {helper}")
            self.helpers["examples/" + helper] = found[0]
        self.artifacts = {}
        require(len(self.manifest["models"]) == 6
                and len({m["id"] for m in self.manifest["models"]}) == 6,
                "expected six distinct model checkpoints")
        for model in self.manifest["models"]:
            require(sorted(a["tier"] for a in model["artifacts"]) == sorted(TIERS),
                    "each checkpoint must have FULL/Q8_0/Q4_K_M")
            for artifact in model["artifacts"]:
                require(artifact["id"] not in self.artifacts, "duplicate artifact ID")
                model_files(artifact)
                self.artifacts[artifact["id"]] = (model, artifact)

    def attempt(self, job):
        item = {"job_directory": str(job), "classification": "incomplete",
                "valid_evidence": False, "qualified": False}
        try:
            run_path = contained(job, "run.json")
            record_path = run_path if run_path.is_file() else contained(job, "execution.json")
            if not record_path.is_file():
                item["reason"] = "no completed runner receipt"
                return item
            record = self.reader.load(record_path)
            artifact_id, backend = record.get("artifact", {}).get("id"), record.get("backend")
            item.update(artifact_id=artifact_id, backend=backend,
                        recorded_error=record.get("error"),
                        recorded_passed=record.get("passed"),
                        recorded_engine_passed=record.get("engine_passed"),
                        recorded_cpu_attention_policy=record.get("cpu_attention_policy"),
                        run_receipt_sha256=self.reader.digest(record_path))
            require(artifact_id in self.artifacts and backend in ("cpu", "metal"),
                    "runner does not identify an expected artifact/backend")
            model, artifact = self.artifacts[artifact_id]
            item["model_id"] = model["id"]
            if record.get("pilot") is True:
                item.update(classification="pilot", reason="pilot cannot qualify a full cell")
                return item
            pins = record.get("source_pins", {})
            if any(pins.get(path) != sha for path, sha in self.required_pins.items()):
                item.update(classification="stale", reason="attempt does not use the selected current build/source pins")
                return item
            needed = ["prepared.json", "suite.json", "engine/report.json", "execution.json",
                      "oracle/suite.json", "independent-review.json", "pins.json", "run.json"]
            if any(not contained(job, name).is_file() for name in needed):
                item["reason"] = "full runner/oracle/engine/independent evidence is not complete"
                return item
            if backend == "cpu" and record.get("cpu_attention_policy") is None:
                other_policies = [self.reader.load(job / name).get("cpu_attention_policy")
                                  for name in ("execution.json", "oracle/suite.json", "independent-review.json")]
                require(all(policy is None for policy in other_policies),
                        "new CPU evidence is missing its runner reference policy")
                item.update(classification="stale",
                            reason="earlier or unrecorded CPU reference policy cannot qualify a repaired-reference cell")
                return item
            p = self.reader.load(job / "prepared.json")
            if record.get("coverage_profile", "extended-24") != self.profile:
                item.update(classification="stale", reason="attempt uses a different coverage profile")
                return item
            if not self.smoke and p["corpus_sha256"] != self.corpus_sha:
                item.update(classification="stale", reason="attempt uses a different corpus")
                return item
            item.update(self.validate(job, record, model, artifact, p))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            item.update(classification="invalid", valid_evidence=False, qualified=False,
                        reason=str(error))
        return item

    def validate(self, job, run, model, artifact, p):
        r = self.reader
        e, x, s, o, v = [r.load(contained(job, name)) for name in
                         ("engine/report.json", "execution.json", "suite.json",
                          "oracle/suite.json", "independent-review.json")]
        prep_sha = r.digest(job / "prepared.json")
        require(v["pilot"] is False and run["pilot"] is False
                and v["valid_evidence"] is True, "review is not valid full-suite evidence")
        require(v["prepared_sha256"] == prep_sha and v["job_directory"] == str(job),
                "review belongs to different prepared input/job")
        stopped = self.smoke and e.get("state_phase_status") == "not_run_after_numeric_failure"
        attempted_ids = [c["id"] for c in e["cases"] if c.get("attempted") is True]
        expected_internal = 0 if stopped else len(self.internal_ids)
        require(v["cases_checked"] == len(attempted_ids) and v["rows_checked"] == ROWS * len(attempted_ids)
                and v["internal_modes_checked"] == expected_internal * 2
                and v["internal_checks_checked"] == expected_internal * 12,
                "independent review scope is incomplete")
        if self.smoke:
            require(all(value.get("coverage_profile") == self.profile for value in (run, x, e, s, v)), "coverage profile differs")
            require(s["fail_fast"] is True and e["fail_fast"] is True
                    and e["execution_policy"] == "numeric_then_state", "smoke execution order differs")
            require(v["selected_case_ids"] == e["selected_case_ids"] == [c["id"] for c in s["cases"]] == SMOKE
                    and v["cases_planned"] == 6 and v["oracle_cases_checked"] == 6
                    and v["oracle_rows_checked"] == 768
                    and v["unrun_case_ids"] == SMOKE[len(attempted_ids):], "smoke review scope differs")
            require(1 <= len(attempted_ids) <= 6 and attempted_ids == SMOKE[:len(attempted_ids)], "smoke attempt prefix differs")
            require(v["state_phase_status"] == e["state_phase_status"]
                    and v["early_stop"] == e.get("early_stop"), "smoke phase receipt differs")
        else:
            require(len(attempted_ids) == 24, "extended review scope is incomplete")
        require(all(value["suite_id"] == self.corpus["suite_id"] for value in (p, e, s, o, v)),
                "suite identity differs")
        require(all(value["backend"] == run["backend"] for value in (e, x, s, o, v)),
                "backend differs")
        require(run["model_id"] == x["model_id"] == model["id"]
                and x["artifact"]["id"] == artifact["id"], "checkpoint differs")
        require(run["artifact"]["tier"] == x["artifact"]["tier"] == artifact["tier"]
                and run["artifact"]["quantization"] == x["artifact"]["quantization"] == artifact["quantization"],
                "actual precision labels differ")
        files = model_files(artifact)
        require(p["model_files"] == e["model_files"] == o["model_files"] == files
                and model_files(run["artifact"]) == model_files(x["artifact"]) == files
                and p["model_sha256"] == files[0]["sha256"], "model payload manifest differs")
        require(p["architecture"] == e["architecture"] == model["architecture"],
                "architecture differs")
        reuse = run.get("oracle_reuse")
        require(reuse == x.get("oracle_reuse") == v.get("oracle_reuse"), "oracle reuse receipt differs")
        if not reuse:
            require(p["preparation_binary_sha256"] == self.build["binary"]["sha256"]
                    and p["preparation_source_sha256"] == self.helpers,
                    "preparation used a different engine build")
        else:
            require(self.smoke, "reference reuse requires a six-case profile")
        prepared_ids = [c["id"] for c in p["cases"]]
        require(p["rows"] == ROWS and len(prepared_ids) == len(set(prepared_ids))
                and (prepared_ids == self.ids or (self.smoke and reuse and len(prepared_ids) == 24))
                and [c["id"] for c in o["cases"]] == prepared_ids
                and all([c["id"] for c in value["cases"]] == self.ids for value in (e, v)),
                "case set/order differs")
        corpus_path = Path(s["corpus"]["path"]).resolve()
        actual_corpus = r.load(corpus_path)
        actual_corpus_sha = r.digest(corpus_path)
        require(actual_corpus_sha == p["corpus_sha256"] and actual_corpus["suite_id"] == self.corpus["suite_id"]
                and [c["id"] for c in actual_corpus["cases"]] == prepared_ids, "prepared corpus differs")
        authored = {c["id"]: c for c in self.corpus["cases"]}
        actual_authored = {c["id"]: c for c in actual_corpus["cases"]}
        require(all(actual_authored[name] == authored[name] for name in self.ids), "selected corpus text differs")
        cases_by_id = {c["id"]: c for c in p["cases"]}
        selected_cases = [cases_by_id[name] for name in self.ids]
        for case, source in zip(p["cases"], actual_corpus["cases"]):
            require(case["domain"] == source["domain"]
                    and case["context_tokens"] == source["context_tokens"]
                    and case["continuation_utf8"] == source["continuation_text"]
                    and case["prompt_tokens"] == case["prompt_original_ids"]
                    and case["teacher_forcing_tokens"] == case["continuation_original_ids"][:ROWS]
                    and len(case["teacher_forcing_tokens"]) == ROWS
                    and len(case["prompt_tokens"]) + ROWS <= case["context_tokens"],
                    "case text/history/context differs")
        token_hash = hashlib.sha256(json.dumps(
            [{key: c[key] for key in CASE_KEYS} for c in p["cases"]],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        require(token_hash == p["token_identity_sha256"] == e["token_identity_sha256"]
                == s["expected_token_identity_sha256"], "token identity differs")
        pins = r.load(job / "pins.json")
        require(run["source_pins"] == x["source_pins"] == pins
                == {pin["path"]: pin["sha256"] for pin in s["source_pins"]},
                "runner/suite source pins differ")
        for path, sha in pins.items():
            r.check(path, sha)
        cpu_policy = o.get("cpu_attention_policy")
        require(run.get("cpu_attention_policy") == x.get("cpu_attention_policy")
                == v.get("cpu_attention_policy") == cpu_policy,
                "CPU reference policy differs across saved receipts")
        cpu_policy_check.check_suite(o, run["backend"], cpu_policy, pins)
        inputs = e["input_pins"]
        require(Counter((str(Path(pin["path"]).resolve()), pin["sha256"])
                        for pin in inputs["sources"])
                == Counter((str(Path(path).resolve()), sha) for path, sha in pins.items())
                and e["compiled_helper_sources"] == self.helpers, "compiled source pins differ")
        for key, expected_path in (("suite", job / "suite.json"), ("prepared", job / "prepared.json")):
            require(Path(inputs[key]["path"]).resolve() == expected_path
                    and inputs[key]["sha256"] == r.digest(expected_path), "engine input pin differs")
        require(Path(s["prepared"]["path"]).resolve() == job / "prepared.json"
                and s["prepared"]["sha256"] == prep_sha, "suite prepared binding differs")
        require(inputs["binary"]["sha256"] == s["expected_binary_sha256"]
                == self.build["binary"]["sha256"]
                and Path(inputs["binary"]["path"]).resolve() == Path(self.build["binary"]["path"]).resolve(),
                "engine binary differs")
        require(inputs["corpus"]["sha256"] == s["corpus"]["sha256"] == actual_corpus_sha,
                "engine corpus differs")
        require(v["validator_sha256"] == run["validator"]["sha256"],
                "independent validator binding differs")
        r.check(run["validator"]["path"], v["validator_sha256"])
        require(e["valid"] is True and e["post_run_pins_verified"] is True
                and x["engine_valid"] is True and x["post_run_pins_verified"] is True
                and run["engine_valid"] is True and run["post_run_pins_verified"] is True and e["model_loads"] == 1
                and e["runtime_dispatch_overrides"] == [], "engine/runner validity differs")
        require(e["limits"] == v["limits"] == LIMITS and v["nll_gate"] is None,
                "acceptance bounds differ")
        require(type(e["oracle_micro_batch_tokens"]) is int and e["oracle_micro_batch_tokens"] == 8,
                "engine did not admit the fixed oracle micro-batch policy")
        require(type(v["oracle_micro_batch_tokens"]) is int and v["oracle_micro_batch_tokens"] == 8
                and type(v["oracle_prefill_chunk_tokens"]) is int and v["oracle_prefill_chunk_tokens"] == 128,
                "independent review used a different oracle execution policy")
        require(s["internal_cases"] == e["internal_case_ids"] == v["internal_case_ids"] == self.internal_ids,
                "state-check case selection differs")
        require(o["status"] == "complete" and o["completed_cases"] == o["expected_cases"] == len(prepared_ids)
                and o["prepared_sha256"] == prep_sha and o["corpus_sha256"] == actual_corpus_sha,
                "oracle suite incomplete or stale")
        require(o["new_context_per_case"] is True and o["kv_dtype"] == "F16"
                and o["prefill_chunk_tokens"] == 128
                and type(o["micro_batch_tokens"]) is int and o["micro_batch_tokens"] == 8,
                "oracle context/micro-batch contract differs")
        storage = o["weight_storage"]["tensor_storage"]
        require(storage and all(type(row["tensor_count"]) is int and row["tensor_count"] > 0
                                and type(row["encoded_tensor_bytes"]) is int and row["encoded_tensor_bytes"] > 0
                                for row in storage)
                and sum(row["tensor_count"] for row in storage) == o["weight_storage"]["total_tensor_count"],
                "invalid observed weight storage")
        types = {row["type"] for row in storage}
        if artifact["tier"] == "FULL":
            require(types <= {"bf16", "f16", "f32"} and artifact["quantization"].lower() in types,
                    "FULL observed storage is not the stated floating precision")
        else:
            require(("q8_0" if artifact["tier"] == "Q8_0" else "q4_K") in types,
                    "observed storage lacks the stated quantization")
        raw_bindings = {row["path"]: row["sha256"] for row in v["raw_files"]}
        per_case = []
        oracle_by_id = {c["id"]: c for c in o["cases"]}
        for case, result, checked in zip(selected_cases, e["cases"], v["cases"]):
            oracle_index = oracle_by_id[case["id"]]
            label = case["id"]
            require(r.load(job / "engine" / f"{label}.result.json") == result,
                    "saved case result differs")
            require(result["rows"] == ROWS, "case row count differs")
            oracle_path = job / "oracle" / f"{label}.json"
            oracle = r.load(oracle_path)
            cpu_policy_check.check_case(oracle, run["backend"], cpu_policy)
            require(r.digest(oracle_path) == oracle_index["oracle_json_sha256"]
                    and oracle["prompt_tokens"] == case["prompt_tokens"]
                    and oracle["teacher_forcing_tokens"] == case["teacher_forcing_tokens"]
                    and oracle["tokenization_verified"] is True
                    and oracle["model_files"] == files and oracle["prepared_sha256"] == prep_sha
                    and oracle["model_sha256"] == p["model_sha256"]
                    and oracle["case_id"] == label and oracle["vocab_size"] == p["vocab_size"]
                    and type(oracle["micro_batch_tokens"]) is int and oracle["micro_batch_tokens"] == 8
                    and oracle["backend"] == run["backend"], "oracle input/history differs")
            require(oracle["oracle_raw_sha256"] == oracle_index["oracle_raw_sha256"]
                    == checked["oracle_raw_sha256"]
                    == raw_bindings[f"oracle/{label}.f32"],
                    "raw receipt bindings differ")
            if self.smoke and result["attempted"] is False:
                require(stopped and result["valid"] is None and result["passed"] is None
                        and result["status"] == "not_run_after_numeric_failure"
                        and result["reference"] == {"valid": None, "passed": None, "completed_rows": 0}, "unrun case has a verdict")
                require(checked["attempted"] is False and checked["valid_evidence"] is None
                        and checked["passed"] is None and checked["numeric_passed"] is None
                        and checked["rows"] == [] and checked["internal_passed"] is None
                        and checked["internal_required"] is (label in self.internal_ids), "unrun review has a verdict")
                check_skipped_internal(result["internal"], label in self.internal_ids)
                per_case.append({"id": label, "attempted": False, "passed": None, "numeric_passed": None,
                                 "internal_passed": None})
                continue
            require(result["valid"] is True and result["attempted"] is True
                    and (not self.smoke or checked["attempted"] is True)
                    and checked["valid_evidence"] is True and result["reference"]["completed_rows"] == ROWS,
                    "case rows incomplete")
            require(result["reference"]["actual_logits_sha256"] == checked["actual_raw_sha256"]
                    == raw_bindings[f"engine/{label}.actual.f32"], "actual raw receipt binding differs")
            values, numeric = scores(checked, result, case, p["vocab_size"])
            selected = label in self.internal_ids
            internal = result["internal"]
            if stopped:
                check_skipped_internal(internal, selected)
                require(checked["internal_passed"] is None, "skipped state check claims a verdict")
                internal_pass = None
            elif selected:
                require(internal["required"] is True and internal["attempted"] is True
                        and internal["status"] == "complete" and internal["valid"] is True
                        and len(internal["modes"]) == 2
                        and all(len(mode["checks"]) == 6 for mode in internal["modes"]),
                        "state checks incomplete")
                internal_pass = all(check["passed"] is True for mode in internal["modes"] for check in mode["checks"])
                require(internal["passed"] is internal_pass and checked["internal_passed"] is internal_pass,
                        "state-check outcome differs")
            else:
                require(internal == {"required": False, "attempted": False, "status": "not_selected",
                                     "valid": None, "passed": None, "modes": []}
                        and checked["internal_passed"] is None, "unselected check claims a pass")
                internal_pass = None
            passed = (None if numeric and selected and stopped else numeric and (not selected or internal_pass))
            require(result["passed"] is passed and checked["passed"] is passed
                    and checked["internal_required"] is selected, "case outcome differs")
            per_case.append({"id": label, **values, "attempted": True, "numeric_passed": numeric, "passed": passed,
                             "internal_passed": internal_pass})
        passed = all(c["passed"] is True for c in per_case)
        count = sum(c["passed"] is True for c in per_case)
        internal_count = sum(c["internal_passed"] is True for c in per_case)
        if self.smoke:
            check_smoke_scope(e, v, per_case, stopped, self.internal_ids)
        else:
            require(e["counts"] == {"planned": 24, "attempted": 24, "passed": count, "failed": 24-count}
                    and e["internal_counts"] == {"planned": 3, "attempted": 3, "passed": internal_count, "failed": 3-internal_count},
                    "aggregate counts differ")
        require(e["passed"] is passed and v["passed"] is passed
                and x["engine_passed"] is passed and run["engine_passed"] is passed
                and run["passed"] is passed and run["independent_exit_code"] == 0
                and x["engine_exit_code"] == run["engine_exit_code"] == (0 if passed else 1),
                "aggregate acceptance differs")
        retained = retained_rows(r, job, v, prep_sha, p["vocab_size"], self.ids, attempted_ids)
        if reuse:
            check_reuse(r, job, reuse, p, run, pins, raw_bindings)
        expected_commands = ["engine", "independent"] if reuse else ["prepare", "oracle", "engine", "independent"]
        require([command["name"] for command in run["commands"]] == expected_commands
                and x["commands"] == run["commands"][:-1], "command scope differs")
        for command in run["commands"]:
            require(r.load(job / f"{command['name']}.command.json") == command
                    and command["exit_code"] == (0 if command["name"] != "engine" or passed else 1),
                    "command completion receipt differs")
        seconds = {c["name"]: c["seconds"] for c in run["commands"]}
        require(all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in seconds.values()),
                "invalid timing")
        scored = [c for c in per_case if c["attempted"]]
        selected_token_hash = hashlib.sha256(json.dumps(
            [{key: c[key] for key in CASE_KEYS} for c in selected_cases],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return {"coverage_profile": self.profile, "cases_planned": len(self.ids),
                "cases_attempted": len(attempted_ids), "unrun_case_ids": [c["id"] for c in per_case if not c["attempted"]],
                "state_phase_status": e.get("state_phase_status", "complete"),
                "internal_checks_checked": v["internal_checks_checked"], "early_stop": e.get("early_stop"),
                "oracle_reuse": reuse, "preparation_binary_sha256": p["preparation_binary_sha256"],
                "prepared_case_count": len(prepared_ids), "prepared_token_identity_sha256": token_hash,
                "classification": "qualified" if passed else "failed", "qualified": passed,
                "valid_evidence": True, "passed": passed, "token_identity_sha256": selected_token_hash,
                "cpu_attention_policy": cpu_policy,
                "precision": artifact["quantization"], "cases": per_case,
                "numerical_summary": {
                    "min_matching_argmax_per_128": min(c["matching_argmax"] for c in scored),
                    "max_case_mean_kl": max(c["mean_kl"] for c in scored),
                    "max_row_kl": max(c["max_kl"] for c in scored),
                    "mean_reference_nll": math.fsum(c["reference_mean_nll"] for c in scored) / len(scored),
                    "mean_actual_nll": math.fsum(c["actual_mean_nll"] for c in scored) / len(scored),
                    "mean_nll_delta": math.fsum(c["mean_nll_delta"] for c in scored) / len(scored),
                    "nll_gate": None},
                "seconds": seconds, "retained_rows": retained,
                "observed_weight_storage": o["weight_storage"],
                "independent_review_sha256": r.digest(job / "independent-review.json"),
                "engine_report_sha256": r.digest(job / "engine/report.json")}

    def summarize(self, runs):
        runs = Path(runs).resolve(strict=True)
        jobs = sorted({path.parent for name in ("run.json", "execution.json", "prepared.json")
                       for path in runs.rglob(name) if not path.is_symlink()})
        attempts = [self.attempt(job) for job in jobs]
        # Identical authored inputs must produce the same frozen token IDs for
        # every precision/backend of a checkpoint, not just within one job.
        for model in self.manifest["models"]:
            valid = [a for a in attempts if a.get("model_id") == model["id"] and a["valid_evidence"]]
            if len({a["token_identity_sha256"] for a in valid}) > 1:
                for attempt in valid:
                    attempt.update(classification="invalid", qualified=False, valid_evidence=False,
                                   reason="token identity differs across current checkpoint attempts")
        cells = []
        for model in self.manifest["models"]:
            for tier in TIERS:
                artifact = next(a for a in model["artifacts"] if a["tier"] == tier)
                for backend in ("cpu", "metal"):
                    found = [a for a in attempts if a.get("artifact_id") == artifact["id"] and a.get("backend") == backend]
                    qualified = [a["job_directory"] for a in found if a["qualified"]]
                    classes = {a["classification"] for a in found}
                    status = ("qualified" if qualified else "failed" if "failed" in classes
                              else "stale" if classes and classes <= {"stale", "pilot"} and "stale" in classes
                              else "pending")
                    cells.append({"model_id": model["id"], "checkpoint": model["checkpoint"],
                                  "artifact_id": artifact["id"], "tier": tier,
                                  "actual_precision": artifact["quantization"], "backend": backend,
                                  "model_files": model_files(artifact), "status": status,
                                  "qualified_attempts": qualified,
                                  "multiple_current_passes": len(qualified) > 1, "attempts": found})
        summarizer_sha = self.reader.digest(Path(__file__))
        policy_helper_sha = self.reader.digest(Path(cpu_policy_check.__file__))
        self.reader.finish()
        counts = Counter(c["status"] for c in cells)
        return {"schema_version": 1, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "suite_id": self.corpus["suite_id"], "engine_binary": self.build["binary"],
                "input_pins": self.input_pins, "corpus_sha256": self.corpus_sha,
                "summarizer_sha256": summarizer_sha,
                "cpu_policy_helper_sha256": policy_helper_sha,
                "coverage_profile": self.profile, "expected_cells": 36, "cases_per_cell": len(self.ids), "rows_per_case": 128,
                "reference_policy": {"micro_batch_tokens": 8, "submitted_prefill_chunk_tokens": 128,
                                     "cpu": {"id": cpu_policy_check.POLICY,
                                             "source_revision": cpu_policy_check.REVISION,
                                             "source_patch_sha256": cpu_policy_check.PATCH_SHA256,
                                             "exact_runtime_records": "each accepted attempt retains its CPU/BLAS path, size and SHA"},
                                     "metal": {"id": "stock-metal"}},
                "internal_case_ids": self.internal_ids, "limits": LIMITS,
                "summary": {name: counts[name] for name in ("qualified", "failed", "pending", "stale")},
                "attempt_counts": dict(Counter(a["classification"] for a in attempts)),
                "all_qualified": counts["qualified"] == 36, "cells": cells,
                "unassigned_attempts": [a for a in attempts if a.get("artifact_id") not in self.artifacts
                                        or a.get("backend") not in ("cpu", "metal")],
                "scope": "Current-build fixed-history numeric and selected state checks. NLL is descriptive; no semantic task-quality claim.",
                "model_payload_reads": 0, "raw_metrics_recomputed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["extended-24", *SMOKE_PROFILES], default="extended-24")
    for name in ("runs", "engine-build", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to replace summary")
    value = Matrix(args.manifest, args.engine_build, args.corpus, args.profile).summarize(args.runs)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value["summary"]), flush=True)


if __name__ == "__main__":
    main()
