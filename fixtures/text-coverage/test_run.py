"""Fail-closed admission checks; no model files or devices are needed."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from run import (SMOKE_CASES, artifact_files, available_memory, check_identity,
                 check_pins, check_smoke_identity, clean_environment, execute,
                 find_artifact, preserve_reference, selected_cases, state_cases)
from build_engine import digest, recorded_build_environment, validate_build_environment


class AdmissionTests(unittest.TestCase):
    def test_build_receipt_omits_credentials_and_keeps_supported_settings(self):
        supported = {
            "RUST_LOG": "warn", "CARGO_BUILD_JOBS": "4",
            "CARGO_INCREMENTAL": "0", "CARGO_PROFILE_RELEASE_LTO": "thin",
            "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_CODEGEN_UNITS": "1",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER": "/usr/bin/clang",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "",
            "MACOSX_DEPLOYMENT_TARGET": "14.0",
        }
        source = {
            **supported,
            "CARGO_REGISTRIES_PRIVATE_TOKEN": "fake-registry-credential",
            "CARGO_REGISTRY_TOKEN": "fake-default-credential",
            "CARGO_REGISTRIES_PRIVATE_CREDENTIAL_PROVIDER": "provider --key=fake-key",
            "CARGO_HTTP_PROXY": "https://user:fake-password@example.invalid",
            "RUST_PRIVATE_TOKEN": "fake-rust-credential",
            "MACOSX_DEPLOYMENT_TARGET_SECRET": "fake-prefix-credential",
            "CARGO_PROFILE_SECRET_LTO": "fake-profile-credential",
        }
        validate_build_environment(source)
        recorded = recorded_build_environment(source)
        self.assertEqual(recorded, {**supported, "CARGO_PROFILE_SECRET_LTO": "[redacted]"})
        self.assertNotIn("fake-", json.dumps(recorded))
        self.assertEqual(source["CARGO_REGISTRY_TOKEN"], "fake-default-credential")

    def test_unpinned_flags_and_wrapper_aliases_are_rejected_without_values(self):
        for name in (
            "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
            "CARGO_PROFILE_RELEASE_RUSTFLAGS",
        ):
            with self.assertRaises(RuntimeError) as caught:
                validate_build_environment({name: "fake-sensitive-value"})
            self.assertIn(name, str(caught.exception))
            self.assertNotIn("fake-sensitive-value", str(caught.exception))
            validate_build_environment({name: ""})

    def test_dispatch_overrides_are_removed_without_copying_secrets_to_receipt(self):
        clean, removed = clean_environment({"PATH": "/bin", "DINF_X": "secret", "CANDLE_FOO": "yes", "GGML_X": "1", "GGUF_X": "2", "DYLD_INSERT_LIBRARIES": "bad", "LD_PRELOAD": "bad", "DYLD_PRINT_LIBRARIES": "1"})
        self.assertEqual(clean, {"PATH": "/bin", "DYLD_PRINT_LIBRARIES": "1"})
        self.assertEqual(len(removed), 6)
        self.assertNotIn("secret", removed)

    def test_missing_or_duplicate_manifest_ids_fail(self):
        model = {"artifacts": [{"id": "a"}]}
        self.assertEqual(find_artifact({"models": [model]}, "a")[0], model)
        for manifest, name in [({"models": [model]}, "b"), ({"models": [model, model]}, "a")]:
            with self.assertRaises(ValueError):
                find_artifact(manifest, name)

    def test_runtime_library_search_overrides_are_removed(self):
        clean, removed = clean_environment({"DYLD_LIBRARY_PATH": "/bad", "DYLD_FALLBACK_LIBRARY_PATH": "/bad", "LD_LIBRARY_PATH": "/bad", "PATH": "/bin"})
        self.assertEqual(clean, {"PATH": "/bin"})
        self.assertEqual(len(removed), 3)

    def test_shards_require_all_expected_files_and_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [{"filename": f"sub/model-{i}.gguf", "size_bytes": 3, "sha256": "a" * 64} for i in (1, 2)]
            artifact = {"files": records}
            (root / "model-1.gguf").write_bytes(b"one")
            with self.assertRaises(ValueError):
                artifact_files(artifact, root / "model-1.gguf")
            (root / "model-2.gguf").write_bytes(b"two")
            self.assertEqual(len(artifact_files(artifact, root / "model-1.gguf")), 2)
            (root / "model-2.gguf").write_bytes(b"bad-size")
            with self.assertRaises(ValueError):
                artifact_files(artifact, root / "model-1.gguf")

    def test_prepared_artifact_uses_derived_hash_not_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "aligned.gguf"
            path.write_bytes(b"new")
            artifact = {"requires_preparation": True, "files": [], "preparation": {"expected_output": {"filename": path.name, "size_bytes": 3, "sha256": digest(path)}}}
            self.assertEqual(artifact_files(artifact, path)[0]["sha256"], digest(path))

    def test_token_identity_rejects_text_change_even_with_matching_hash_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = {"corpus_sha256": "c", "token_identity_sha256": "t", "cases": [{"id": "a", "prompt_tokens": [1, 2]}]}
            first = check_identity(root, "model", prepared)
            self.assertEqual(check_identity(root, "model", prepared), first)
            changed = json.loads(json.dumps(prepared))
            changed["cases"][0]["prompt_tokens"][1] = 3
            with self.assertRaises(ValueError):
                check_identity(root, "model", changed)
            self.assertEqual(check_identity(root, "model", prepared), first)

    def test_pins_reject_in_place_same_length_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source"
            path.write_bytes(b"old")
            pins = {str(path): digest(path)}
            check_pins(pins)
            path.write_bytes(b"new")
            with self.assertRaises(ValueError):
                check_pins(pins)

    def test_memory_count_does_not_count_purgeable_pages_twice(self):
        sample = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1.\nPages inactive: 2.\nPages speculative: 3.\nPages purgeable: 100.\nPages wired down: 50.\n"
        with patch("run.platform.system", return_value="Darwin"), patch("run.subprocess.check_output", return_value=sample):
            self.assertEqual(available_memory()[0], 6 * 16384)

    def test_timeout_stops_the_child_and_keeps_a_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"commands": []}
            with patch("run.platform.system", return_value="test"):
                code = execute([sys.executable, "-c", "import time; time.sleep(30)"], "hung", root, {}, record, 0.1)
            self.assertEqual(code, 124)
            self.assertTrue(record["commands"][0]["timed_out"])
            self.assertTrue((root / "hung.command.json").is_file())


class SmokeReferenceTests(unittest.TestCase):
    def test_short_state_profile_changes_only_state_selection(self):
        cases=self.cases()
        self.assertEqual(selected_cases({"cases":cases},"smoke-6-short-state"),
                         selected_cases({"cases":cases},"smoke-6"))
        self.assertEqual(state_cases("smoke-6-short-state"),["conversation-512"])
        self.assertEqual(state_cases("smoke-6"),
                         ["conversation-512","json-2048","retrieval-4096"])
        with self.assertRaises(ValueError):
            selected_cases({"cases":cases},"smoke-6-short-state",pilot=True)

    def cases(self):
        # Real domain/context ordering, with distinct text and token histories.
        return [{"id": f"{domain}-{context}", "context_tokens": context,
                 "rendered_prompt_utf8": f"{domain} prompt at {context}",
                 "prompt_tokens": [index + 1, context],
                 "teacher_forcing_tokens": [index + 2, index + 3]}
                for index, domain in enumerate(("prose", "rust", "math", "json",
                                                "conversation", "multilingual", "extraction", "retrieval"))
                for context in (512, 2048, 4096)]

    def fixture(self, root):
        root = root.resolve()
        source, output = root / "old", root / "new"
        source.mkdir(); output.mkdir(); (source / "oracle").mkdir()
        cases = self.cases()
        def save(path, value):
            path.write_text(json.dumps(value, indent=2) + "\n")
        corpus = source / "corpus.json"
        save(corpus, {"cases": cases})
        prepared = {"cases": cases, "corpus_sha256": digest(corpus)}
        save(source / "prepared.json", prepared)
        entry, old_engine, oracle_binary = [root / name for name in ("weight", "old-engine", "oracle")]
        for path in (entry, old_engine, oracle_binary):
            path.write_bytes(path.name.encode())
        reference = {"binary": {"path": str(oracle_binary), "sha256": digest(oracle_binary)}}
        reference_build = root / "reference.build.json"; save(reference_build, reference)
        save(source / "pins.json", {str(oracle_binary): digest(oracle_binary), str(old_engine): digest(old_engine)})
        commands = {"prepare": [str(old_engine), "--prepare", str(entry), str(corpus), str(source / "prepared.json")],
                    "oracle": [str(oracle_binary), str(entry), "cpu", str(source / "prepared.json"), str(source / "oracle")]}
        for name, command in commands.items():
            save(source / f"{name}.command.json", {"command": ["/usr/bin/time", "-l", *command],
                 "exit_code": 0, "timed_out": False})
        entries = []
        for case in cases:
            name = case["id"]
            raw = source / "oracle" / f"{name}.f32"; raw.write_bytes(name.encode())
            info = source / "oracle" / f"{name}.json"
            save(info, {"case_id": name, "oracle_raw_sha256": digest(raw),
                        "prepared_sha256": digest(source / "prepared.json")})
            entries.append({"id": name, "oracle_json_sha256": digest(info), "oracle_raw_sha256": digest(raw)})
        save(source / "oracle/suite.json", {"status": "complete", "expected_cases": 24,
             "completed_cases": 24, "prepared_sha256": digest(source / "prepared.json"),
             "corpus_sha256": digest(corpus), "cases": entries})
        args = SimpleNamespace(reuse_oracle=source, output=output, profile="smoke-6", pilot=False,
                               corpus=corpus, backend="cpu", reference_build=reference_build, reuse_admission=None)
        return args, reference, entry

    def test_six_selection_matches_full_reference_without_changing_either(self):
        full = {"cases": self.cases()}
        original = json.dumps(full)
        selected = selected_cases(full, "smoke-6")
        self.assertEqual([case["id"] for case in selected], SMOKE_CASES)
        self.assertEqual(selected_cases({"cases": selected}, "smoke-6"), selected)
        self.assertEqual(json.dumps(full), original)

    def test_selection_rejects_missing_duplicate_reordered_and_pilot_inputs(self):
        full = {"cases": self.cases()}
        six = selected_cases(full, "smoke-6")
        variants = [six[:-1], [*six[:-1], six[0]], list(reversed(six)),
                    [dict(case, id="unselected") if case["id"] == "json-2048" else case for case in full["cases"]]]
        for cases in variants:
            with self.subTest(ids=[case["id"] for case in cases]), self.assertRaises(ValueError):
                selected_cases({"cases": cases}, "smoke-6")
        with self.assertRaises(ValueError):
            selected_cases(full, "smoke-6", pilot=True)

    def test_full_and_fresh_six_share_identity_in_both_arrival_orders(self):
        for full_first in (False, True):
            with self.subTest(full_first=full_first), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                full = {"cases": self.cases(), "corpus_sha256": "full", "token_identity_sha256": "full-tokens"}
                six = selected_cases(full, "smoke-6")
                if full_first:
                    check_identity(root, "model", full)
                first = check_smoke_identity(root, "model", six)
                check_identity(root, "model", full)
                second = check_smoke_identity(root, "model", selected_cases(full, "smoke-6"))
                self.assertEqual(first, second)
                altered = json.loads(json.dumps(six)); altered[0]["prompt_tokens"][0] += 1
                with self.assertRaises(ValueError):
                    check_smoke_identity(root, "model", altered)
                self.assertEqual(check_smoke_identity(root, "model", six), first)

    def test_smoke_identity_rejects_changed_text_and_teacher_even_if_ids_match(self):
        for field in ("rendered_prompt_utf8", "teacher_forcing_tokens"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); full = {"cases": self.cases(), "corpus_sha256": "c", "token_identity_sha256": "t"}
                check_identity(root, "model", full)
                six = json.loads(json.dumps(selected_cases(full, "smoke-6")))
                six[0][field] = "different text" if field == "rendered_prompt_utf8" else [99, 100]
                with self.assertRaises(ValueError):
                    check_smoke_identity(root, "model", six)
                self.assertFalse((root / "model.smoke-6.json").exists())

    def test_preservation_keeps_full_metadata_and_only_six_raw_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            args, reference, entry = self.fixture(Path(temp)); record = {}
            before = {str(path): digest(path) for path in args.reuse_oracle.rglob("*") if path.is_file()}
            preserve_reference(args, reference, entry, record)
            receipt_path = Path(record["oracle_reuse"]["path"])
            self.assertEqual(digest(receipt_path), record["oracle_reuse"]["sha256"])
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["source_job"], str(args.reuse_oracle))
            self.assertEqual(len(receipt["preserved_files"]), 15)
            for item in receipt["preserved_files"]:
                self.assertEqual(Path(item["source"]["path"]).read_bytes(), (args.output / item["destination"]).read_bytes())
                self.assertEqual(digest(args.output / item["destination"]), item["source"]["sha256"])
            for key in ("original_pins", "reference_build"):
                self.assertEqual(digest(receipt[key]["path"]), receipt[key]["sha256"])
            for command in receipt["original_commands"].values():
                self.assertEqual(digest(command["path"]), command["sha256"])
            self.assertEqual(len(json.loads((args.output / "prepared.json").read_text())["cases"]), 24)
            self.assertEqual(len(json.loads((args.output / "oracle/suite.json").read_text())["cases"]), 24)
            self.assertEqual(sorted(path.stem for path in (args.output / "oracle").glob("*.f32")), sorted(SMOKE_CASES))
            self.assertEqual(before, {str(path): digest(path) for path in args.reuse_oracle.rglob("*") if path.is_file()})
            self.assertNotIn("commands", record)  # No invented prepare/oracle execution.

    def test_preservation_rejects_changed_raw_or_case_json(self):
        for suffix in ("f32", "json"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp:
                args, reference, entry = self.fixture(Path(temp)); record = {}
                path = args.reuse_oracle / "oracle" / f"prose-512.{suffix}"
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    preserve_reference(args, reference, entry, record)
                self.assertNotIn("oracle_reuse", record)
                self.assertFalse((args.output / "oracle-reuse.json").exists())

    def test_preservation_rejects_incomplete_or_failed_original_producer(self):
        for filename, field, value in (("oracle.command.json", "exit_code", 1),
                                        ("prepare.command.json", "timed_out", True),
                                        ("oracle/suite.json", "completed_cases", 23)):
            with self.subTest(filename=filename, field=field), tempfile.TemporaryDirectory() as temp:
                args, reference, entry = self.fixture(Path(temp)); path = args.reuse_oracle / filename
                doc = json.loads(path.read_text()); doc[field] = value; path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    preserve_reference(args, reference, entry, {})
                self.assertFalse((args.output / "oracle-reuse.json").exists())

    def test_preservation_rejects_wrong_backend_model_or_unpinned_binary(self):
        for change in ("backend", "entry", "binary-pin"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                args, reference, entry = self.fixture(Path(temp))
                if change == "backend":
                    args.backend = "metal"
                elif change == "entry":
                    entry = Path(temp) / "different-weight"
                else:
                    (args.reuse_oracle / "pins.json").write_text("{}")
                with self.assertRaises(ValueError):
                    preserve_reference(args, reference, entry, {})
                self.assertFalse((args.output / "oracle-reuse.json").exists())

    def test_preservation_rejects_existing_destination_with_different_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, reference, entry = self.fixture(Path(temp))
            destination = args.output / "prepared.json"; destination.write_bytes(b"unrelated")
            with self.assertRaises(ValueError):
                preserve_reference(args, reference, entry, {})
            self.assertEqual(destination.read_bytes(), b"unrelated")
            self.assertFalse((args.output / "oracle-reuse.json").exists())


if __name__ == "__main__":
    unittest.main()
