#!/usr/bin/env python3
"""Build the pinned text-suite oracle and run checks that load no model.

Example (use matching llama.cpp/ggml headers and libraries):
  python3 build_reference.py --llama-source /path/to/llama.cpp \
    --include-dir /path/to/llama/include --include-dir /path/to/ggml/include \
    --library-dir /path/to/llama/lib --library-dir /path/to/ggml/lib \
    --revision FULL_COMMIT --out /new/path/llama_matrix_reference

The source tree supplies vendor/nlohmann/json.hpp. The caller must establish
that the installed libraries match --revision; this helper records their exact
bytes, rather than treating the supplied revision as proof of that match.
The reference submits 128-token prefill calls with fixed eight-token physical
batches. This avoids the pinned Metal matrix paths' lower-precision inputs;
CPU BF16 dot products still round inputs to BF16.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from cpu_reference_policy import build_policy, check_runtime


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(paths):
    return {str(Path(p).absolute()): digest(p) for p in sorted(set(paths))}


def check_pins(pins):
    for path, expected in pins.items():
        if digest(path) != expected:
            raise RuntimeError(f"build input changed: {path}")


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def runtime_inputs(binary, library_dirs):
    """Include dynamically loaded ggml backends, not just direct link inputs."""
    roots = []
    for directory in library_dirs:
        roots.extend(directory.glob("libllama*.dylib"))
        roots.extend(directory.glob("libllama*.so*"))
        roots.extend(directory.glob("libggml*.dylib"))
        roots.extend(directory.glob("libggml*.so*"))
        for backend_dir in (directory, directory.parent / "libexec"):
            roots.extend(backend_dir.glob("libggml*.so*"))
            roots.extend(backend_dir.glob("libggml*.dylib"))
    if not roots:
        raise RuntimeError("no llama/ggml runtime libraries found")
    queue = [Path(p).absolute() for p in roots]
    if binary is not None:
        queue.append(binary)
    seen, records, files = set(), [], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        resolved = path.resolve(strict=True)
        files.add(path)
        files.add(resolved)
        if sys.platform == "darwin":
            listing = command_output(["otool", "-L", str(path)])
            links = [line.strip().split(" (", 1)[0] for line in listing.splitlines()[1:]]
        elif sys.platform.startswith("linux"):
            listing = command_output(["ldd", str(path)])
            if "not found" in listing:
                raise RuntimeError(f"missing runtime dependency: {path}\n{listing}")
            links = []
            for line in listing.splitlines():
                line = line.strip()
                name = line.split("=>", 1)[-1].strip().split(" (", 1)[0]
                if name.startswith("/"):
                    links.append(name)
        else:
            raise RuntimeError("runtime pinning supports macOS and Linux")
        records.append({"path": str(path), "resolved_path": str(resolved),
                        "sha256": digest(path), "linked_libraries": links})
        for name in links:
            if name.startswith("@rpath/"):
                candidates = [d / name[7:] for d in [path.parent, *library_dirs]]
            elif name.startswith("@loader_path/"):
                candidates = [path.parent / name[len("@loader_path/"):]]
            elif name.startswith("@executable_path/") and binary is not None:
                candidates = [binary.parent / name[len("@executable_path/"):]]
            elif name.startswith("/"):
                candidates = [Path(name)]
            else:
                raise RuntimeError(f"unresolved runtime dependency {name} in {path}")
            existing = next((p for p in candidates if p.is_file()), None)
            if existing is None:
                # macOS ships these in the system dyld shared cache. Record the
                # link name and OS identity, and do not claim to hash that cache.
                if sys.platform == "darwin" and name.startswith(("/usr/lib/", "/System/Library/")):
                    continue
                raise RuntimeError(f"cannot locate runtime dependency {name} in {path}")
            queue.append(existing.absolute())
    return pin(files), sorted(records, key=lambda record: record["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path, action="append", required=True)
    parser.add_argument("--library-dir", type=Path, action="append", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-plugin-build", type=Path, help="proved CPU F32 flash repair build receipt; required for CPU inference")
    parser.add_argument("--legacy", action="store_true", help="Build the fixed-token producer with its original 4096-token batching")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be the full lowercase commit SHA")
    source = Path(__file__).resolve().with_name("llama_matrix_reference.cpp")
    if args.legacy:
        if args.cpu_plugin_build is None:
            parser.error("--legacy requires --cpu-plugin-build")
        source = source.parent.parent / "llama_reference.cpp"
    include_dirs = [p.absolute() for p in args.include_dir]
    library_dirs = [p.absolute() for p in args.library_dir]
    vendor = args.llama_source.absolute() / "vendor"
    json_header = vendor / "nlohmann/json.hpp"
    if not json_header.is_file():
        parser.error("llama source lacks vendor/nlohmann/json.hpp")
    binary = args.out.absolute()
    receipt = binary.with_name(binary.name + ".build.json")
    log = binary.with_name(binary.name + ".build.log")
    for path in (binary, receipt, log):
        if path.exists():
            parser.error(f"refusing to replace {path}")
    binary.parent.mkdir(parents=True, exist_ok=True)
    headers = [p for d in include_dirs for p in d.rglob("*.h")]
    source_pins = pin([source, Path(__file__).resolve(), Path(__file__).with_name("cpu_reference_policy.py").resolve(), json_header, *headers])
    cpu_policy, cpu_pins = (build_policy(args.cpu_plugin_build, args.revision) if args.cpu_plugin_build else (None, {}))
    if args.legacy:
        source_pins.update(pin([Path(__file__).resolve().parent / "cpu-reference/runtime.hpp"]))
    source_pins.update(cpu_pins)
    before_runtime, _ = runtime_inputs(None, library_dirs)
    cxx = shutil.which(args.cxx)
    if cxx is None:
        parser.error(f"compiler not found: {args.cxx}")
    command = [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-pthread",
               f'-DLLAMA_REFERENCE_REVISION="{args.revision}"',
               f'-DLLAMA_MATRIX_SOURCE_SHA256="{digest(source)}"',
               "-DLLAMA_CPU_REFERENCE_JSON=" + json.dumps(json.dumps(cpu_policy, separators=(",", ":"))), str(source)]
    for directory in [*include_dirs, vendor]:
        command += ["-I", str(directory)]
    for directory in library_dirs:
        command += ["-L", str(directory), f"-Wl,-rpath,{directory}"]
    command += ["-lllama", "-lggml", "-lggml-base", "-o", str(binary)]
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    check_pins(source_pins)
    check_pins(before_runtime)
    runtime_pins, runtime_records = runtime_inputs(binary, library_dirs)
    runtime_pins.update(cpu_pins)
    if args.legacy:
        self_test = {"models_loaded": 0, "legacy_batch_tokens": 4096,
                     "producer_source_sha256": digest(source)}
    else:
        self_test_output = command_output([str(binary), "--self-test"])
        self_test = json.loads(self_test_output)
        if (self_test.get("passed") is not True or self_test.get("models_loaded") != 0
                or self_test.get("backend_initialized") is not False
                or self_test.get("micro_batch_tokens") != 8
                or self_test.get("prefill_chunk_tokens") != 128
                or self_test.get("producer_source_sha256") != digest(source)):
            raise RuntimeError("no-model producer self-test failed")
    check_pins(source_pins)
    check_pins(runtime_pins)
    cpu_runtime_check = None
    cpu_runtime_check_stderr = None
    if cpu_policy is not None:
        cpu_check = subprocess.run([str(binary), "--check-cpu-runtime"], capture_output=True, text=True, check=True)
        cpu_runtime_check = json.loads(cpu_check.stdout)
        cpu_runtime_check_stderr = cpu_check.stderr
        if cpu_runtime_check.get("models_loaded") != 0 or cpu_runtime_check.get("backend_initialized") is not True:
            raise RuntimeError("CPU runtime check changed its no-model scope")
        check_runtime(cpu_runtime_check, cpu_policy, runtime_pins)
        check_pins(source_pins)
        check_pins(runtime_pins)
    result = {"schema_version": 1, "passed": True, "revision": args.revision,
              "micro_batch_tokens": 4096 if args.legacy else 8, "prefill_chunk_tokens": 4096 if args.legacy else 128,
              "cpu_reference": cpu_policy, "cpu_runtime_check": cpu_runtime_check,
              "cpu_runtime_check_stderr": cpu_runtime_check_stderr,
              "binary": {"path": str(binary), "sha256": digest(binary)},
              "source_pins": source_pins, "runtime_pins": runtime_pins,
              "runtime_libraries": runtime_records, "command": command,
              "compiler": {"path": cxx, "sha256": digest(cxx), "version": command_output([cxx, "--version"])},
              "platform": platform.platform(), "legacy": args.legacy, "self_test": self_test,
              "build_log": {"path": str(log), "sha256": digest(log)},
              "revision_provenance": "caller supplies source/library revision match; hashes pin exact build and runtime inputs"}
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(receipt)


if __name__ == "__main__":
    main()
