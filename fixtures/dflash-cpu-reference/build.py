#!/usr/bin/env python3
"""Build private control/repaired CPU fixtures; never open or run a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise ValueError("this recipe supports only the pinned macOS ARM64 build")
    upstream, base, output = (p.resolve() for p in
                              (args.upstream, args.base_build, args.output))
    if output.exists() or output.is_relative_to(upstream) or output.is_relative_to(base):
        raise ValueError("output must be new and outside both input trees")
    manifest = json.loads((ROOT / "inputs.json").read_text())
    inputs = [(upstream / p, h) for p, h in manifest["source_hashes"].items()]
    inputs += [(base / p, h) for p, h in manifest["base_build_hashes"].items()]
    inputs += [(ROOT / p, h) for p, h in manifest["files"].items()]
    for path, expected in inputs:
        if digest(path) != expected:
            raise ValueError(f"input differs: {path}")
    sdk = subprocess.check_output(
        ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
        text=True, timeout=10).strip()
    sdk_version = subprocess.check_output(
        ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"],
        text=True, timeout=10).strip()
    output.mkdir()
    receipt = {"passed": False, "steps": [], "model_calls": 0, "gpu_calls": 0,
               "sdk": sdk, "sdk_version": sdk_version,
               "manifest_sha256": digest(ROOT / "inputs.json"),
               "builder_sha256": digest(__file__)}
    deadline = time.monotonic() + 300
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C",
           "OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}

    def run(name, argv):
        seconds = min(180, deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError("300-second build/probe limit")
        start = time.monotonic()
        with (output / (name + ".stdout")).open("xb") as stdout, \
             (output / (name + ".stderr")).open("xb") as stderr:
            child = subprocess.Popen(argv, cwd=output, env=env, stdout=stdout,
                                     stderr=stderr, start_new_session=True)
            try:
                code = child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        receipt["steps"].append({"name": name, "argv": argv, "exit_code": code,
                                 "seconds": time.monotonic() - start})
        if code != 0:
            raise RuntimeError(f"{name} failed; preserved logs, no retry")
        for suffix in [".stdout", ".stderr"]:
            if (output / (name + suffix)).stat().st_size > 8 * 1024**2:
                raise ValueError("build log exceeds 8 MiB")

    def expand(value, source, obj):
        for token, path in [("@UPSTREAM@", upstream), ("@BASE_BUILD@", base),
                            ("@SDK@", sdk), ("@SOURCE@", source), ("@OBJECT@", obj)]:
            value = value.replace(token, str(path))
        return value

    try:
        private = output / "source"
        for rel in manifest["patched_source_hashes"]:
            target = private / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upstream / rel, target)
        run("apply-private-patch", ["/usr/bin/patch", "--batch", "--fuzz=0", "-p1",
                                   "-d", str(private), "-i", str(ROOT / "reference-repair.patch")])
        for rel, expected in manifest["patched_source_hashes"].items():
            if digest(private / rel) != expected:
                raise ValueError(f"patched source differs: {rel}")
        for mode in ["control", "repaired"]:
            stage = output / mode
            stage.mkdir()
            for name, rel in manifest["archive_paths"].items():
                shutil.copyfile(base / rel, stage / name)
            for command in manifest["compile_commands"]:
                source = (upstream if mode == "control" else private) / command["source"]
                obj = stage / (Path(command["source"]).name + ".o")
                run(mode + "-" + obj.name,
                    [expand(v, source, obj) for v in command["argv"]])
                run(mode + "-archive-" + command["archive"],
                    ["/usr/bin/ar", "r", str(stage / command["archive"]), str(obj)])
            libraries = [str(stage / name) for name in manifest["archive_order"]]
            includes = []
            for rel in ["include", "src", "ggml/include"]:
                includes += ["-isystem", str(upstream / rel)]
            flags = ["/usr/bin/clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                     "-isysroot", sdk, *includes]
            frameworks = ["-framework", "Accelerate", "-framework", "Foundation",
                          "-framework", "Metal", "-framework", "MetalKit"]
            producer = ROOT / ("control-producer.cpp" if mode == "control" else "producer.cpp")
            run(mode + "-link-producer", [*flags, str(producer), *libraries, *frameworks,
                                         "-o", str(stage / "producer")])
            run(mode + "-link-probe", [*flags, str(ROOT / "precision_probe.cpp"),
                                      *libraries, *frameworks, "-o", str(stage / "precision-probe")])
            argv = [str(stage / "precision-probe"), mode, str(stage / "proof")]
            if mode == "repaired":
                argv += [str(output / "control/proof.defaults.f32")]
            run(mode + "-synthetic-proof", argv)
        proof = json.loads((output / "repaired/proof.json").read_text())
        if not proof["passed"] or proof["default_control_bit_differences"] != 0:
            raise ValueError("synthetic precision or unchanged-default proof failed")
        for path, expected in inputs:
            if digest(path) != expected:
                raise ValueError(f"input changed during build: {path}")
        receipt.update(passed=True, synthetic_proof=proof,
                       artifacts={str(p.relative_to(output)): digest(p)
                                  for p in output.glob("*/producer")})
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        (output / "build-result.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
