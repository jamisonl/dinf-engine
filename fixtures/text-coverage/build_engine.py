#!/usr/bin/env python3
"""Build the matrix example and pin the source used by Cargo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


# Only build settings belong in the receipt. In particular, Cargo registry
# tokens, credential providers and authenticated proxy URLs do not.
BUILD_ENV_NAMES = {
    "CARGO_HOME", "CARGO_BUILD_JOBS", "CARGO_BUILD_INCREMENTAL",
    "CARGO_INCREMENTAL", "CARGO_CACHE_RUSTC_INFO", "CARGO_TARGET_DIR",
    "CARGO_BUILD_TARGET_DIR", "CARGO_BUILD_BUILD_DIR", "CARGO_BUILD_DEP_INFO_BASEDIR",
    "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTDOC", "CARGO_NET_OFFLINE",
    "CARGO_TERM_COLOR", "CARGO_TERM_VERBOSE", "CARGO_TERM_QUIET",
    "RUSTC", "RUSTDOC", "RUSTUP_TOOLCHAIN", "RUSTUP_HOME",
    "RUST_LOG", "RUST_BACKTRACE", "RUST_LIB_BACKTRACE",
    "MACOSX_DEPLOYMENT_TARGET", "SDKROOT",
}
BUILD_OVERRIDE_NAMES = {
    "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
}
PROFILE_ENV = re.compile(
    r"CARGO_PROFILE_[A-Z0-9_]+_(?:OPT_LEVEL|DEBUG|SPLIT_DEBUGINFO|STRIP|"
    r"DEBUG_ASSERTIONS|OVERFLOW_CHECKS|LTO|PANIC|INCREMENTAL|CODEGEN_UNITS|RPATH|RUSTFLAGS)"
)
TARGET_ENV = re.compile(r"CARGO_TARGET_[A-Z0-9_]+_(?:LINKER|RUNNER|RUSTFLAGS)")
SECRET_ENV = re.compile(
    r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIALS?|PRIVATE_KEY|ACCESS_KEY|API_KEY)(?:_|$)"
)


def validate_build_environment(source):
    for name, value in source.items():
        flags = name.endswith("_RUSTFLAGS") and (
            PROFILE_ENV.fullmatch(name) or TARGET_ENV.fullmatch(name)
        )
        if value and (name in BUILD_OVERRIDE_NAMES or flags):
            # Do not include the value: even a rejected flag may contain a key.
            raise RuntimeError(f"unset build override before qualification: {name}")


def recorded_build_environment(source):
    result = {}
    for name, value in source.items():
        if (name in BUILD_ENV_NAMES or name in BUILD_OVERRIDE_NAMES
                or PROFILE_ENV.fullmatch(name) or TARGET_ENV.fullmatch(name)):
            result[name] = "[redacted]" if SECRET_ENV.search(name) else value
    return result


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_files(root):
    # `root` is the Cargo workspace root: normally this crate's root, or a
    # workspace that vendors this crate as `dinf-engine/`. Vendored
    # crates live in `third_party/` next to the crate's Cargo.toml when present,
    # otherwise next to the workspace Cargo.toml.
    engine = root / "dinf-engine" if (root / "dinf-engine/Cargo.toml").is_file() else root
    vendored = engine / "third_party"
    if not vendored.is_dir():
        vendored = root / "third_party"
    files = {root / "Cargo.toml", root / "Cargo.lock", engine / "Cargo.toml"}
    for directory in [engine / "src", engine / "examples", vendored]:
        for path in directory.rglob("*"):
            if path.is_file() and not any(p in {"target", ".git", "__pycache__"} for p in path.parts):
                files.add(path)
    # Cargo config affects feature selection, compiler flags and patch paths.
    for base in [root / ".cargo", Path.home() / ".cargo"]:
        for name in ["config", "config.toml"]:
            if (base / name).is_file():
                files.add(base / name)
    return sorted(files)


def snapshot(root):
    return {str(path.resolve()): digest(path) for path in source_files(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    validate_build_environment(os.environ)
    output.mkdir()
    before = snapshot(root)
    command = ["cargo", "build", "--message-format=json-render-diagnostics", "--target-dir", str(root / "target"), "--release", "--locked", "--offline", "-p", "dinf-engine", "--example", "qualify_matrix"]
    started = time.time()
    with (output / "build.log").open("xb") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    if snapshot(root) != before:
        raise RuntimeError("source changed during Cargo build")
    artifacts = []
    for line in (output / "build.log").read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (event.get("reason") == "compiler-artifact" and event.get("target", {}).get("name") == "qualify_matrix" and event.get("executable")):
            artifacts.append(Path(event["executable"]).resolve())
    if len(artifacts) != 1:
        raise RuntimeError("Cargo did not report exactly one qualify_matrix executable")
    cargo_binary = artifacts[0]
    binary = output / "qualify_matrix"
    shutil.copy2(cargo_binary, binary)
    if digest(binary) != digest(cargo_binary):
        raise RuntimeError("binary changed while copying build evidence")
    archived_sources = {}
    for name, expected in before.items():
        original = Path(name)
        relative = original.relative_to(root) if original.is_relative_to(root) else Path("external") / hashlib.sha256(name.encode()).hexdigest() / original.name
        saved = output / "source" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, saved)
        if digest(saved) != expected:
            raise RuntimeError(f"source changed while archiving: {original}")
        archived_sources[name] = {"path": str(saved), "sha256": expected}
    if snapshot(root) != before:
        raise RuntimeError("source changed while preserving build evidence")
    runtime = {}
    linked = ""
    if platform.system() == "Darwin":
        linked = subprocess.check_output(["otool", "-L", binary], text=True)
        for line in linked.splitlines()[1:]:
            path = Path(line.strip().split(" (", 1)[0])
            if path.is_absolute() and path.is_file():
                runtime[str(path)] = digest(path)
    receipt = {
        "schema_version": 1, "command": command, "cwd": str(root),
        "seconds": time.time() - started, "host": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True).strip(),
        "build_environment": recorded_build_environment(os.environ),
        "binary": {"path": str(binary), "sha256": digest(binary)},
        "cargo_artifact": str(cargo_binary), "archived_sources": archived_sources,
        "source_pins": before, "runtime_pins": runtime, "linked_libraries": linked,
    }
    write_new(output / "build.json", receipt)
    print(output / "build.json", flush=True)


if __name__ == "__main__":
    main()
