#!/usr/bin/env python3
"""Losslessly archive reviewed F32 rows from one new text-coverage job.

The review must bind every raw file by relative path, byte count, and SHA-256.
Compression is allowed for valid numerical failures as well as passes. The
helper never changes scores, JSON, logs, models, or files outside the given
new-results root. archive.json precedes unlink; archive-removal.json records
the completed removals. Both receipts use create-new semantics.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

from fetch_models import file_identity, hash_file, require


SUITE_ID = "text-core-three-tier-v1"
PILOT_SUITE_ID = "text-core-three-tier-pilot-v1"
CHUNK_BYTES = 1024 * 1024
SMOKE_CASES = ["prose-512", "conversation-512", "json-2048", "rust-2048",
               "retrieval-4096", "multilingual-4096"]

SMOKE_INTERNALS = {"smoke-6": ["conversation-512", "json-2048", "retrieval-4096"],
                   "smoke-6-short-state": ["conversation-512"]}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def relative_raw_path(job, value):
    require(isinstance(value, str), "raw path must be text")
    parts = PurePosixPath(value).parts
    require(parts and len(parts) >= 2 and parts[0] in ("oracle", "engine")
            and not value.startswith("/") and "\\" not in value
            and "\x00" not in value and PurePosixPath(value).as_posix() == value
            and all(part not in ("", ".", "..") for part in parts)
            and parts[-1].endswith(".f32"), "invalid raw-row path")
    path = job / value
    require(job in path.resolve().parents, "raw-row path escapes job")
    require(not path.is_symlink(), "raw-row symlink is not allowed")
    for parent in path.parents:
        if parent == job:
            break
        require(not parent.is_symlink(), "raw-row parent symlink is not allowed")
    return path


def validate_job(results_root, job):
    root = results_root.resolve(strict=True)
    job = job.resolve(strict=True)
    require(root != job and root in job.parents, "job must be inside new-results root")
    review_path, prepared_path = job / "independent-review.json", job / "prepared.json"
    require(review_path.is_file() and not review_path.is_symlink(), "independent review missing")
    require(prepared_path.is_file() and not prepared_path.is_symlink(), "prepared suite missing")
    review_sha, review_id = hash_file(review_path)
    prepared_sha, prepared_id = hash_file(prepared_path)
    review = json.loads(review_path.read_bytes())
    prepared = json.loads(prepared_path.read_bytes())
    require(review.get("valid_evidence") is True, "independent evidence is not valid")
    require(type(review.get("pilot")) is bool and type(review.get("passed")) is bool,
            "review must state pilot and numerical-pass scope")
    require(type(review.get("cases_checked")) is int and review["cases_checked"] > 0
            and type(review.get("rows_checked")) is int and review["rows_checked"] > 0,
            "review must state case and row counts")
    require(review.get("job_directory") == str(job), "review belongs to a different job")
    suite_id = prepared.get("suite_id")
    require(review.get("suite_id") == suite_id, "review and prepared suite differ")
    require(suite_id == SUITE_ID
            or (suite_id == PILOT_SUITE_ID and review["pilot"] is True),
            "not an authorized new text-coverage suite/pilot")
    require(review.get("prepared_sha256") == prepared_sha, "review does not bind prepared input")
    rows = review.get("raw_files")
    require(isinstance(rows, list) and rows, "review has no raw-file bindings")
    profile = review.get("coverage_profile", "extended-24")
    require(profile == "extended-24" or profile in SMOKE_INTERNALS, "unknown review coverage profile")
    if profile in SMOKE_INTERNALS:
        internal_ids = SMOKE_INTERNALS[profile]
        require(review.get("internal_case_ids") == internal_ids, "smoke state-case selection differs")
        n = review["cases_checked"]
        require(review["pilot"] is False and review.get("cases_planned") == 6
                and review.get("selected_case_ids") == SMOKE_CASES and 1 <= n <= 6
                and review["rows_checked"] == 128 * n
                and review.get("oracle_cases_checked") == 6
                and review.get("oracle_rows_checked") == 768
                and review.get("unrun_case_ids") == SMOKE_CASES[n:], "smoke review coverage differs")
        require([c["id"] for c in review["cases"]] == SMOKE_CASES
                and [c["id"] for c in review["cases"] if c["attempted"]] == SMOKE_CASES[:n],
                "smoke attempt order differs")
        expected_raw = {f"oracle/{label}.f32" for label in SMOKE_CASES}
        expected_raw.update(f"engine/{label}.actual.f32" for label in SMOKE_CASES[:n])
        require({r["path"] for r in rows} == expected_raw, "smoke raw coverage differs")
        if review.get("state_phase_status") == "not_run_after_numeric_failure":
            require(review["passed"] is False and review.get("internal_checks_checked") == 0
                    and review.get("internal_modes_checked") == 0
                    and review.get("early_stop") == {"phase": "numeric", "case_id": SMOKE_CASES[n - 1],
                                                       "reason": "numeric_failure"},
                    "smoke early rejection differs")
        else:
            require(n == 6 and review.get("state_phase_status") == "complete"
                    and review.get("internal_checks_checked") == 12 * len(internal_ids)
                    and review.get("internal_modes_checked") == 2 * len(internal_ids)
                    and review.get("early_stop") is None, "smoke state coverage differs")
    seen, files = set(), []
    for row in rows:
        path = relative_raw_path(job, row["path"])
        require(row["path"] not in seen, "duplicate raw-file binding")
        seen.add(row["path"])
        require(type(row["size_bytes"]) is int and row["size_bytes"] > 0
                and row["size_bytes"] % 4 == 0, "invalid F32 byte count")
        digest, identity = hash_file(path)
        require(identity["size_bytes"] == row["size_bytes"] and digest == row["sha256"],
                "reviewed raw file differs")
        files.append({"path": path, "relative_path": row["path"],
                      "bytes": row["size_bytes"], "sha256": digest, "identity": identity})
    actual = {path.relative_to(job).as_posix() for path in job.rglob("*.f32")}
    require(actual == seen, "unreviewed or missing raw F32 files")
    return job, review, files, {
        "review_sha256": review_sha, "review_identity": review_id,
        "prepared_sha256": prepared_sha, "prepared_identity": prepared_id,
    }


def decompress_digest(executable, path, expected_bytes):
    digest, total = hashlib.sha256(), 0
    with subprocess.Popen([str(executable), "-q", "-d", "-c", "--", str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                block = process.stdout.read(CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                require(total <= expected_bytes, "decompression exceeds raw byte count")
                digest.update(block)
            stderr = process.stderr.read()
            code = process.wait()
            require(code == 0, "zstd decompression failed: " + stderr.decode(errors="replace"))
        except Exception:
            process.kill()
            process.wait()
            raise
    require(total == expected_bytes, "decompression byte count differs")
    return digest.hexdigest()


def compress_one(executable, row):
    source = row["path"]
    compressed = source.with_name(source.name + ".zst")
    partial = compressed.with_name(compressed.name + ".part")
    require(not compressed.exists() and not compressed.is_symlink(), "archive already exists")
    require(file_identity(source) == row["identity"], "raw file changed before compression")
    with partial.open("xb") as stream:
        result = subprocess.run([str(executable), "-q", "-T1", "-3", "-c", "--", str(source)],
                                stdout=stream, stderr=subprocess.PIPE, check=False)
        require(result.returncode == 0,
                "zstd compression failed: " + result.stderr.decode(errors="replace"))
        stream.flush()
        os.fsync(stream.fileno())
    require(file_identity(source) == row["identity"], "raw file changed during compression")
    compressed_sha, compressed_id = hash_file(partial)
    require(decompress_digest(executable, partial, row["bytes"]) == row["sha256"],
            "decompressed SHA-256 differs from reviewed raw file")
    os.link(partial, compressed)
    partial.unlink()
    return {
        "raw_path": row["relative_path"], "raw_bytes": row["bytes"],
        "raw_sha256": row["sha256"], "raw_identity": row["identity"],
        "compressed_path": row["relative_path"] + ".zst",
        "compressed_bytes": compressed_id["size_bytes"],
        "compressed_sha256": compressed_sha, "compressed_identity": file_identity(compressed),
        "decompressed_bytes": row["bytes"], "decompressed_sha256": row["sha256"],
    }


def archive(results_root, job, executable):
    job, review, files, bindings = validate_job(results_root, job)
    output, removal_path = job / "archive.json", job / "archive-removal.json"
    require(not output.exists() and not removal_path.exists(), "archive receipt already exists")
    executable = executable.resolve(strict=True)
    executable_sha, executable_id = hash_file(executable)
    version = subprocess.run([str(executable), "--version"], capture_output=True,
                             text=True, check=True).stdout.strip()
    lock = job / ".archive.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        records = [compress_one(executable, row) for row in files]
        require(file_identity(executable) == executable_id, "zstd changed during archive")
        require(file_identity(job / "independent-review.json") == bindings["review_identity"],
                "independent review changed")
        require(file_identity(job / "prepared.json") == bindings["prepared_identity"],
                "prepared input changed")
        for row, record in zip(files, records):
            require(file_identity(row["path"]) == row["identity"], "raw changed before receipt")
            require(file_identity(job / record["compressed_path"]) == record["compressed_identity"],
                    "compressed file changed before receipt")
        receipt = {
            "schema_version": 1, "suite_id": review["suite_id"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "state": "verified_archives_ready_for_raw_removal",
            "raw_removal_receipt": "archive-removal.json",
            "independent_review": {"path": "independent-review.json",
                                   "sha256": bindings["review_sha256"],
                                   "valid_evidence": True, "passed": review["passed"],
                                   "pilot": review["pilot"],
                                   "cases_checked": review["cases_checked"],
                                   "rows_checked": review["rows_checked"],
                                   "classification": review.get("classification")},
            "coverage_profile": review.get("coverage_profile", "extended-24"),
            "selected_case_ids": review.get("selected_case_ids"),
            "unrun_case_ids": review.get("unrun_case_ids", []),
            "state_phase_status": review.get("state_phase_status", "complete"),
            "early_stop": review.get("early_stop"),
            "oracle_reuse": review.get("oracle_reuse"),
            "prepared_sha256": bindings["prepared_sha256"],
            "zstd": {"path": str(executable), "sha256": executable_sha, "version": version},
            "archive_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fetch_helper_sha256": hashlib.sha256(
                Path(__file__).with_name("fetch_models.py").read_bytes()).hexdigest(),
            "files": records,
        }
        write_new(output, receipt)
        removed = []
        for row, record in zip(files, records):
            require(file_identity(row["path"]) == row["identity"], "raw changed before removal")
            require(file_identity(job / record["compressed_path"]) == record["compressed_identity"],
                    "compressed file changed before removal")
            row["path"].unlink()
            removed.append(row["relative_path"])
        removal = {"schema_version": 1, "passed": True,
                   "archive_sha256": hash_file(output)[0], "removed_raw_paths": removed,
                   "removed_raw_bytes": sum(row["bytes"] for row in files),
                   "retained_compressed_bytes": sum(row["compressed_bytes"] for row in records)}
        write_new(removal_path, removal)
        return receipt, removal
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-results-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--zstd", type=Path,
                        default=Path(shutil.which("zstd")) if shutil.which("zstd") else None)
    args = parser.parse_args()
    require(args.zstd is not None, "zstd is not on PATH; supply --zstd")
    _, removal = archive(args.new_results_root, args.job_dir, args.zstd)
    print(json.dumps(removal))


if __name__ == "__main__":
    main()
