#!/usr/bin/env python3
"""Fetch pinned coverage artifacts using only the Python standard library.

Weights stay in --output-dir. A file becomes final only after its byte count
and full SHA-256 match the manifest. Interrupted .part files can resume; their
full prefix is hashed before resuming and the whole result is hashed again.
This script never loads a model or accepts F16 as a full-precision substitute.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.error
import urllib.parse
import urllib.request


CHUNK_BYTES = 8 * 1024 * 1024
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
SHA_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_identity(path):
    info = path.stat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def hash_file(path):
    require(not path.is_symlink(), f"refuse symlink: {path}")
    before = file_identity(path)
    require(stat.S_ISREG(path.stat().st_mode), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            (opened.st_dev, opened.st_ino, opened.st_size,
             opened.st_mtime_ns, opened.st_ctime_ns)
            == tuple(before[key] for key in (
                "device", "inode", "size_bytes", "mtime_ns", "ctime_ns")),
            f"file changed at open: {path}",
        )
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
        after_open = os.fstat(stream.fileno())
        require(
            (opened.st_dev, opened.st_ino, opened.st_size,
             opened.st_mtime_ns, opened.st_ctime_ns)
            == (after_open.st_dev, after_open.st_ino, after_open.st_size,
                after_open.st_mtime_ns, after_open.st_ctime_ns),
            f"file changed during hash: {path}",
        )
    require(before == file_identity(path), f"file changed after hash: {path}")
    return digest.hexdigest(), before


def validate_manifest(value):
    require(value.get("schema_version") == 1, "unsupported manifest version")
    artifacts = {}
    for model in value["models"]:
        require(ID_RE.fullmatch(model["id"]), "invalid model id")
        for artifact in model["artifacts"]:
            key = artifact["id"]
            require(ID_RE.fullmatch(key) and key not in artifacts,
                    "duplicate or invalid artifact id")
            require(artifact["quantization"] in ("Q4_K_M", "Q8_0", "BF16", "F32"),
                    "unexpected quantization tier")
            if "tier" in artifact:
                expected_tier = ("FULL" if artifact["quantization"] in ("BF16", "F32")
                                 else artifact["quantization"])
                require(artifact["tier"] == expected_tier, "tier and actual dtype differ")
            require(artifact["files"], "artifact has no files")
            names = set()
            for item in artifact["files"]:
                name = item["filename"]
                parts = PurePosixPath(name).parts
                require(
                    parts and not name.startswith("/")
                    and "\\" not in name and "\x00" not in name
                    and all(part not in ("", ".", "..") for part in parts)
                    and PurePosixPath(name).as_posix() == name
                    and name not in names,
                    "invalid or duplicate filename",
                )
                names.add(name)
                require(SHA_RE.fullmatch(item["sha256"]), "invalid SHA-256")
                require(REVISION_RE.fullmatch(item["revision"]), "unpinned revision")
                require(type(item["size_bytes"]) is int and item["size_bytes"] > 0,
                        "invalid file size")
                require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", item["repo"]),
                        "invalid HF repository")
            artifacts[key] = artifact
    require(artifacts, "empty manifest")
    return artifacts


def source_url(item):
    filename = urllib.parse.quote(item["filename"], safe="/")
    return f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{filename}"


def safe_destination(root, artifact_id, filename):
    path = root / artifact_id / filename
    require(root.resolve() in path.resolve().parents, "path escapes output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_space(directory, remaining, reserve_bytes):
    available = shutil.disk_usage(directory).free
    require(available - remaining >= reserve_bytes,
            f"insufficient disk: free={available}, remaining={remaining}, "
            f"reserve={reserve_bytes}")


def check_response(response, offset, size):
    status = response.status
    require(response.headers.get("Content-Encoding", "identity") == "identity",
            "encoded weight response is not supported")
    if offset or status == 206:
        require(status == 206, "server ignored Range; partial file retained")
        require(response.headers.get("Content-Range") == f"bytes {offset}-{size - 1}/{size}",
                "wrong Content-Range; partial file retained")
    else:
        require(status == 200, f"unexpected HTTP status: {status}")
    length = response.headers.get("Content-Length")
    if length is not None:
        require(int(length) == size - offset, "response length differs from pinned size")


def fetch_file(root, artifact_id, item, reserve_bytes, retries=5, opener=None):
    opener = opener or urllib.request.urlopen
    path = safe_destination(root, artifact_id, item["filename"])
    part = path.with_name(path.name + ".part")
    lock = path.with_name(path.name + ".download.lock")
    result = {"artifact_id": artifact_id, "filename": item["filename"],
              "source_url": source_url(item), "expected_sha256": item["sha256"],
              "expected_size_bytes": item["size_bytes"], "path": str(path),
              "started_at": utc_now(), "verified": False, "attempts": []}
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if path.exists() or path.is_symlink():
            digest, identity = hash_file(path)
            require(identity["size_bytes"] == item["size_bytes"]
                    and digest == item["sha256"], "existing final file does not match pin")
            result.update(verified=True, reused=True, sha256=digest, identity=identity)
            return result
        require(not part.is_symlink(), "refuse partial-file symlink")
        if not part.exists():
            with part.open("xb"):
                pass
        for attempt in range(retries + 1):
            prefix_digest, prefix_identity = hash_file(part)
            offset = prefix_identity["size_bytes"]
            require(offset <= item["size_bytes"], "partial file exceeds pinned size")
            check_space(root, item["size_bytes"] - offset, reserve_bytes)
            entry = {"number": attempt + 1, "resume_offset": offset,
                     "prefix_sha256": prefix_digest, "started_at": utc_now()}
            result["attempts"].append(entry)
            if offset == item["size_bytes"]:
                break
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            # A retry gets a fresh signed redirect if a proxy cached an expired one.
            url = source_url(item) + "?download=true"
            if attempt:
                url += f"&retry={time.time_ns()}"
            request = urllib.request.Request(url, headers=headers)
            try:
                with opener(request, timeout=60) as response:
                    check_response(response, offset, item["size_bytes"])
                    require(file_identity(part) == prefix_identity,
                            "partial file changed before append")
                    with part.open("ab") as stream:
                        opened = os.fstat(stream.fileno())
                        require(
                            (opened.st_dev, opened.st_ino, opened.st_size,
                             opened.st_mtime_ns, opened.st_ctime_ns)
                            == tuple(prefix_identity[key] for key in (
                                "device", "inode", "size_bytes", "mtime_ns", "ctime_ns")),
                            "partial file changed at append open",
                        )
                        written = offset
                        last_progress = time.monotonic()
                        while written < item["size_bytes"]:
                            block = response.read(min(CHUNK_BYTES, item["size_bytes"] - written))
                            if not block:
                                raise OSError("short response; partial file retained")
                            check_space(root, len(block), reserve_bytes)
                            stream.write(block)
                            written += len(block)
                            if time.monotonic() - last_progress >= 10:
                                print(json.dumps({"artifact_id": artifact_id,
                                                  "downloaded_bytes": written,
                                                  "size_bytes": item["size_bytes"]}), flush=True)
                                last_progress = time.monotonic()
                        require(not response.read(1), "response exceeds pinned size")
                        stream.flush()
                        os.fsync(stream.fileno())
                entry["completed_at"] = utc_now()
                break
            except (OSError, urllib.error.URLError) as error:
                entry["error"] = str(error)
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        digest, identity = hash_file(part)
        require(identity["size_bytes"] == item["size_bytes"], "incomplete download")
        require(digest == item["sha256"], "SHA-256 mismatch; partial file retained")
        # link() fails if another writer has created the final path; never overwrite.
        os.link(part, path)
        part.unlink()
        result.update(verified=True, reused=False, sha256=digest,
                      identity=file_identity(path), completed_at=utc_now())
        print(json.dumps({"artifact_id": artifact_id, "verified": True,
                          "filename": item["filename"], "sha256": digest}), flush=True)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("models.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", action="append", required=True,
                        help="Exact artifact id; repeat to select several")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    args = parser.parse_args()
    require(args.reserve_bytes >= 0, "negative disk reserve")
    require(not args.receipt.exists(), "receipt already exists")
    manifest_bytes = args.manifest.read_bytes()
    artifacts = validate_manifest(json.loads(manifest_bytes))
    require(len(set(args.artifact)) == len(args.artifact), "duplicate artifact selection")
    require(all(key in artifacts for key in args.artifact), "unknown artifact id")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(key, item) for key in args.artifact for item in artifacts[key]["files"]]
    remaining = 0
    for key, item in jobs:
        path = safe_destination(root, key, item["filename"])
        part = path.with_name(path.name + ".part")
        existing = path.stat().st_size if path.exists() else (part.stat().st_size if part.exists() else 0)
        remaining += max(0, item["size_bytes"] - existing)
    check_space(root, remaining, args.reserve_bytes)
    receipt = {"started_at": utc_now(), "passed": False,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "fetch_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "selected_artifacts": args.artifact, "reserve_bytes": args.reserve_bytes,
               "files": []}

    def run(job):
        key, item = job
        try:
            result = fetch_file(root, key, item, args.reserve_bytes)
            result["requires_preparation"] = artifacts[key].get("requires_preparation", False)
            result["ready_file"] = not result["requires_preparation"]
            return result
        except Exception as error:
            return {"artifact_id": key, "filename": item["filename"],
                    "verified": False, "error": str(error)}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipt["files"] = list(executor.map(run, jobs))
    receipt["completed_at"] = utc_now()
    receipt["passed"] = all(item["verified"] for item in receipt["files"])
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": str(args.receipt), "passed": receipt["passed"]}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
