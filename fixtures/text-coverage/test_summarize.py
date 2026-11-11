"""Small stdlib-only receipt tests; no model, NumPy, Cargo or GPU calls."""

import hashlib
import json
from pathlib import Path
import tempfile
import copy
import shutil
import unittest

import summarize as s


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path), "sha256": sha(path)}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.runs = root / "runs"
        self.runs.mkdir()
        self.sources = {}
        for helper in s.HELPERS:
            path = root / "engine/examples" / helper
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(helper)
            self.sources[str(path)] = sha(path)
        binary = root / "engine-bin"
        binary.write_bytes(b"pinned test binary")
        self.binary = pin(binary)
        self.pins = {**self.sources, str(binary): sha(binary)}
        cpu_files = {}
        for key, name in (("plugin", "libggml-cpu-apple_m4.so"), ("blas_plugin", "libggml-blas.so")):
            path = root / name
            path.write_text(name)
            cpu_files[key] = {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}
            self.pins[str(path)] = sha(path)
        self.cpu_policy = {"schema_version": 1, "id": s.cpu_policy_check.POLICY,
                           "source_revision": s.cpu_policy_check.REVISION,
                           "source_patch_sha256": s.cpu_policy_check.PATCH_SHA256, **cpu_files}
        self.cpu_runtime = {"policy": self.cpu_policy, "cpu_plugin_image_count": 1,
                            "blas_plugin_image_count": 1, "registry_count": 2, "gpu_backends_loaded": False,
                            "runtime_images": [str(binary), *(value["path"] for value in cpu_files.values())]}
        self.helpers = {"examples/" + helper: self.sources[str(self.root / "engine/examples" / helper)]
                        for helper in s.HELPERS}
        self.build = root / "build.json"
        write(self.build, {"source_pins": self.sources, "runtime_pins": {},
                           "binary": self.binary})
        self.corpus = root / "corpus.json"
        self.cases = []
        for domain in ("conversation", "json", "retrieval", "prose",
                       "rust", "multilingual", "facts", "planning"):
            for context in (512, 2048, 4096):
                self.cases.append({"id": f"{domain}-{context}", "domain": domain,
                                   "context_tokens": context,
                                   "messages": [{"role": "user", "content": "shared text"}],
                                   "continuation_text": "fixed natural teacher"})
        write(self.corpus, {"schema_version": 1, "suite_id": "test-matrix",
                            "rows": 128, "cases": self.cases})
        self.models = []
        for index in range(6):
            model = {"id": f"model{index}", "checkpoint": f"publisher/model{index}",
                     "architecture": "llama", "artifacts": []}
            for tier in s.TIERS:
                name = f"model{index}-{tier}"
                model["artifacts"].append({
                    "id": name, "tier": tier, "quantization": "F32" if tier == "FULL" else tier,
                    "files": [{"filename": name + ".gguf", "size_bytes": 100,
                               "sha256": hashlib.sha256(name.encode()).hexdigest()}]})
            self.models.append(model)
        self.manifest = root / "models.json"
        write(self.manifest, {"schema_version": 1, "models": self.models})
        self.validator = root / "validator.py"
        self.validator.write_text("pinned independent validator fixture")

    def matrix(self):
        return s.Matrix(self.manifest, self.build, self.corpus)

    def job(self, name="job", numeric_failure=False, backend="cpu", token_text="chat"):
        job = self.runs / name
        (job / "engine").mkdir(parents=True)
        (job / "oracle").mkdir()
        artifact = self.models[0]["artifacts"][0]
        files = s.model_files(artifact)
        cases = []
        for source in self.cases:
            cases.append({key: value for key, value in {
                "id": source["id"], "domain": source["domain"],
                "context_tokens": source["context_tokens"],
                "rendered_prompt_utf8": token_text, "continuation_utf8": source["continuation_text"],
                "prompt_original_ids": [0, 1, 2], "prompt_tokens": [0, 1, 2],
                "continuation_original_ids": [1] * 129, "teacher_forcing_tokens": [1] * 128,
                "prompt_add_special": True, "continuation_add_special": False,
                "parse_special": True}.items()})
        token_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()
        prepared = {"schema_version": 1, "suite_id": "test-matrix", "rows": 128,
                    "corpus_sha256": sha(self.corpus), "model_files": files,
                    "model_sha256": files[0]["sha256"], "architecture": "llama",
                    "vocab_size": 3, "cases": cases, "token_identity_sha256": token_hash,
                    "preparation_binary_sha256": self.binary["sha256"],
                    "preparation_source_sha256": self.helpers}
        write(job / "prepared.json", prepared)
        prep_sha = sha(job / "prepared.json")
        suite = {"suite_id": "test-matrix", "backend": backend,
                 "source_pins": [{"path": p, "sha256": h} for p, h in self.pins.items()],
                 "expected_binary_sha256": self.binary["sha256"],
                 "expected_token_identity_sha256": token_hash,
                 "prepared": pin(job / "prepared.json"),
                 "internal_cases": s.INTERNAL, "corpus": pin(self.corpus)}
        write(job / "suite.json", suite)
        write(job / "pins.json", self.pins)
        results, reviews, raw_files, oracle_cases = [], [], [], []
        for index, case in enumerate(cases):
            label = case["id"]
            failed = numeric_failure and index == 0
            rows = [{"row": i, "teacher_token": 1, "reference_top": 1,
                     "actual_top": 0 if failed else 1, "kl": .2 if failed else 0.,
                     "max_abs": 1. if failed else 0., "rms": .5 if failed else 0.,
                     "reference_nll": 1., "actual_nll": 2. if failed else 1.,
                     "nll_delta": 1. if failed else 0.} for i in range(128)]
            values = {"matching_argmax": 0 if failed else 128,
                      "mean_kl": .2 if failed else 0., "max_kl": .2 if failed else 0.,
                      "reference_mean_nll": 1., "actual_mean_nll": 2. if failed else 1.,
                      "mean_nll_delta": 1. if failed else 0.}
            hashes = {}
            for folder, suffix in (("oracle", ".f32"), ("engine", ".actual.f32")):
                relative = f"{folder}/{label}{suffix}"
                path = job / relative
                path.write_bytes(b"\0" * (128 * 3 * 4))
                hashes[folder] = sha(path)
                raw_files.append({"path": relative, "size_bytes": path.stat().st_size,
                                  "sha256": sha(path)})
            oracle = {"case_id": label, "backend": backend, "vocab_size": 3, "micro_batch_tokens": 8,
                      "prompt_tokens": case["prompt_tokens"],
                      "teacher_forcing_tokens": case["teacher_forcing_tokens"],
                      "tokenization_verified": True, "model_files": files,
                      "model_sha256": files[0]["sha256"], "prepared_sha256": prep_sha,
                      "oracle_raw_sha256": hashes["oracle"]}
            if backend == "cpu":
                oracle["cpu_attention_policy"] = self.cpu_policy
            oracle_path = job / "oracle" / f"{label}.json"
            write(oracle_path, oracle)
            oracle_cases.append({"id": label, "oracle_json_sha256": sha(oracle_path),
                                 "oracle_raw_sha256": hashes["oracle"]})
            selected = label in s.INTERNAL
            internal = ({"required": True, "attempted": True, "status": "complete",
                         "valid": True, "passed": True,
                         "modes": [{"checks": [{"passed": True} for _ in range(6)]}
                                   for _ in range(2)]}
                        if selected else {"required": False, "attempted": False,
                                          "status": "not_selected", "valid": None,
                                          "passed": None, "modes": []})
            result = {"id": label, "valid": True, "attempted": True, "rows": 128,
                      "passed": not failed, "internal": internal,
                      "reference": {**values, "passed": not failed, "completed_rows": 128,
                                    "comparisons": rows, "actual_logits_sha256": hashes["engine"]}}
            write(job / "engine" / f"{label}.result.json", result)
            results.append(result)
            reviews.append({"id": label, "valid_evidence": True, "passed": not failed,
                            "numeric_passed": not failed, "internal_required": selected,
                            "internal_passed": True if selected else None,
                            "rows": rows, **values, "oracle_raw_sha256": hashes["oracle"],
                            "actual_raw_sha256": hashes["engine"]})
        passed = not numeric_failure
        report = {"suite_id": "test-matrix", "backend": backend, "architecture": "llama",
                  "valid": True, "passed": passed, "post_run_pins_verified": True,
                  "model_loads": 1, "runtime_dispatch_overrides": [], "limits": s.LIMITS,
                  "oracle_micro_batch_tokens": 8,
                  "model_files": files, "token_identity_sha256": token_hash,
                  "internal_case_ids": s.INTERNAL,
                  "internal_counts": {"planned": 3, "attempted": 3, "passed": 3, "failed": 0},
                  "counts": {"planned": 24, "attempted": 24,
                             "passed": 23 if numeric_failure else 24, "failed": int(numeric_failure)},
                  "compiled_helper_sources": self.helpers,
                  "input_pins": {"sources": [{"path": p, "sha256": h} for p, h in self.pins.items()],
                                 "suite": pin(job / "suite.json"), "prepared": pin(job / "prepared.json"),
                                 "corpus": pin(self.corpus), "binary": self.binary},
                  "cases": results}
        write(job / "engine/report.json", report)
        write(job / "oracle/suite.json", {
            "suite_id": "test-matrix", "backend": backend, "model_files": files,
            "status": "complete", "completed_cases": 24, "expected_cases": 24,
            "prepared_sha256": prep_sha, "corpus_sha256": sha(self.corpus),
            "kv_dtype": "F16", "new_context_per_case": True,
            "weight_storage": {"total_tensor_count": 1, "tensor_storage": [
                {"type": "f32", "tensor_count": 1, "encoded_tensor_bytes": 100}]},
            "prefill_chunk_tokens": 128, "micro_batch_tokens": 8, "cases": oracle_cases,
            **({"cpu_attention_policy": self.cpu_policy,
                "cpu_runtime_before": self.cpu_runtime, "cpu_runtime_after": self.cpu_runtime}
               if backend == "cpu" else {})})
        review = {"suite_id": "test-matrix", "valid_evidence": True, "passed": passed,
                  "oracle_micro_batch_tokens": 8, "oracle_prefill_chunk_tokens": 128,
                  "pilot": False, "prepared_sha256": prep_sha, "job_directory": str(job),
                  "cases_checked": 24, "rows_checked": 3072, "internal_modes_checked": 6,
                  "internal_checks_checked": 36, "backend": backend, "internal_case_ids": s.INTERNAL,
                  "validator_sha256": sha(self.validator), "limits": s.LIMITS, "nll_gate": None,
                  "cases": reviews, "raw_files": raw_files}
        review["cpu_attention_policy"] = self.cpu_policy if backend == "cpu" else None
        write(job / "independent-review.json", review)
        record = {"model_id": "model0", "artifact": artifact, "backend": backend,
                  "pilot": False, "source_pins": self.pins, "engine_valid": True,
                  "engine_passed": passed, "engine_exit_code": 0 if passed else 1,
                  "post_run_pins_verified": True, "passed": passed, "independent_exit_code": 0,
                  "validator": pin(self.validator),
                  "commands": [{"name": n, "seconds": 1., "exit_code": 0 if n != "engine" or passed else 1}
                               for n in ("prepare", "oracle", "engine", "independent")]}
        record["cpu_attention_policy"] = self.cpu_policy if backend == "cpu" else None
        for command in record["commands"]:
            write(job / f"{command['name']}.command.json", command)
        execution = {**record, "commands": record["commands"][:3]}
        write(job / "execution.json", execution)
        write(job / "run.json", record)
        return job

    def archive(self, job):
        review = read(job / "independent-review.json")
        records = []
        for item in review["raw_files"]:
            raw = job / item["path"]
            compressed = raw.with_name(raw.name + ".zst")
            # This tests receipt/hash admission, not a zstd implementation.
            compressed.write_bytes(b"opaque archive fixture")
            records.append({"raw_path": item["path"], "raw_bytes": item["size_bytes"],
                            "raw_sha256": item["sha256"], "decompressed_bytes": item["size_bytes"],
                            "decompressed_sha256": item["sha256"],
                            "compressed_path": item["path"] + ".zst",
                            "compressed_bytes": compressed.stat().st_size,
                            "compressed_sha256": sha(compressed)})
            raw.unlink()
        write(job / "archive.json", {
            "state": "verified_archives_ready_for_raw_removal", "suite_id": "test-matrix",
            "prepared_sha256": sha(job / "prepared.json"),
            "independent_review": {"sha256": sha(job / "independent-review.json")}, "files": records})


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name).resolve())

    def summary(self):
        return self.fixture.matrix().summarize(self.fixture.runs)

    def attempt(self):
        return self.summary()["cells"][0]["attempts"][0]

    def mutate(self, job, relative, change):
        path = job / relative
        value = read(path)
        change(value)
        write(path, value)

    def test_empty_matrix_has_all_36_pending_cells(self):
        result = self.summary()
        self.assertEqual(result["summary"], {"qualified": 0, "failed": 0, "pending": 36, "stale": 0})
        self.assertFalse(result["all_qualified"])
        self.assertEqual(len(result["cells"]), 36)

    def test_current_cpu_pass_retains_exact_reference_policy(self):
        self.fixture.job()
        value = self.attempt()
        self.assertEqual(value["classification"], "qualified", value)
        self.assertEqual(value["cpu_attention_policy"], self.fixture.cpu_policy)
        self.assertEqual(value["recorded_cpu_attention_policy"], self.fixture.cpu_policy)
        self.assertEqual(self.summary()["reference_policy"]["cpu"]["id"], s.cpu_policy_check.POLICY)

    def test_historical_stock_cpu_failure_is_retained_but_not_qualified(self):
        job = self.fixture.job(numeric_failure=True)
        for name in ("run.json", "execution.json", "independent-review.json", "oracle/suite.json"):
            self.mutate(job, name, lambda value: value.pop("cpu_attention_policy", None))
        value = self.attempt()
        self.assertEqual(value["classification"], "stale", value)
        self.assertFalse(value["qualified"])
        self.assertFalse(value["recorded_passed"])
        self.assertFalse(value["recorded_engine_passed"])
        self.assertIsNone(value["recorded_cpu_attention_policy"])
        self.assertIn("CPU reference policy", value["reason"])

    def test_new_cpu_attempt_missing_runner_policy_is_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "run.json", lambda value: value.pop("cpu_attention_policy"))
        value = self.attempt()
        self.assertEqual(value["classification"], "invalid")
        self.assertIn("missing its runner reference policy", value["reason"])

    def test_cpu_policy_mismatch_between_receipts_is_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "independent-review.json", lambda value:
                    value["cpu_attention_policy"].update(source_patch_sha256="0" * 64))
        self.assertIn("CPU reference policy differs", self.attempt()["reason"])

    def test_cpu_policy_mismatch_in_one_case_is_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "oracle/conversation-512.json", lambda value:
                    value["cpu_attention_policy"]["plugin"].update(sha256="0" * 64))
        self.assertIn("CPU case lacks", self.attempt()["reason"])

    def test_runtime_must_remain_the_selected_cpu_plugin(self):
        job = self.fixture.job()
        self.mutate(job, "oracle/suite.json", lambda value:
                    value["cpu_runtime_after"].update(cpu_plugin_image_count=2))
        self.assertIn("unexpected backends", self.attempt()["reason"])

    def test_metal_keeps_stock_reference_and_rejects_cpu_policy(self):
        job = self.fixture.job(backend="metal")
        value = self.fixture.matrix().attempt(job)
        self.assertEqual(value["classification"], "qualified", value)
        self.assertIsNone(value["cpu_attention_policy"])
        for name in ("run.json", "execution.json", "independent-review.json", "oracle/suite.json"):
            self.mutate(job, name, lambda value:
                        value.update(cpu_attention_policy=self.fixture.cpu_policy))
        value = self.fixture.matrix().attempt(job)
        self.assertEqual(value["classification"], "invalid", value)
        self.assertIn("Metal oracle", value["reason"])

    def test_repository_subdirectory_maps_to_local_shard_basename(self):
        artifact = {"files": [{"filename": "BF16/model-00001-of-00002.gguf",
                               "size_bytes": 12, "sha256": "a" * 64},
                              {"filename": "BF16/model-00002-of-00002.gguf",
                               "size_bytes": 20, "sha256": "b" * 64}]}
        self.assertEqual([x["name"] for x in s.model_files(artifact)],
                         ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"])
        artifact["files"][0]["filename"] = "../model.gguf"
        with self.assertRaises(ValueError):
            s.model_files(artifact)

    def test_current_pass_counts_one_cell_and_reports_actual_precision(self):
        self.fixture.job()
        result = self.summary()
        attempt = result["cells"][0]["attempts"][0]
        self.assertEqual(result["summary"]["qualified"], 1, attempt)
        self.assertEqual(result["cells"][0]["actual_precision"], "F32")
        self.assertEqual(attempt["numerical_summary"]["min_matching_argmax_per_128"], 128)
        self.assertEqual(attempt["retained_rows"]["raw_files"], 48)
        self.assertEqual(attempt["seconds"]["engine"], 1.)

    def test_valid_numeric_failure_remains_failed(self):
        self.fixture.job(numeric_failure=True)
        attempt = self.attempt()
        self.assertEqual(attempt["classification"], "failed", attempt)
        self.assertTrue(attempt["valid_evidence"])
        self.assertFalse(attempt["qualified"])
        self.assertEqual(self.summary()["summary"]["failed"], 1)

    def test_pilot_is_not_a_full_pass(self):
        job = self.fixture.job()
        self.mutate(job, "run.json", lambda v: v.update(pilot=True))
        self.assertEqual(self.attempt()["classification"], "pilot")
        self.assertEqual(self.summary()["summary"]["pending"], 36)

    def test_old_binary_cannot_qualify(self):
        job = self.fixture.job()
        self.mutate(job, "run.json", lambda v: v["source_pins"].update(
            {self.fixture.binary["path"]: "f" * 64}))
        self.assertEqual(self.attempt()["classification"], "stale")

    def test_old_corpus_cannot_qualify(self):
        job = self.fixture.job()
        self.mutate(job, "prepared.json", lambda v: v.update(corpus_sha256="f" * 64))
        self.assertEqual(self.attempt()["classification"], "stale")

    def test_old_or_missing_oracle_micro_batch_policy_cannot_qualify(self):
        for setting in (None, 128, True):
            job = self.fixture.job(f"old-policy-{setting}")
            def change(value):
                if setting is None:
                    value.pop("micro_batch_tokens")
                else:
                    value["micro_batch_tokens"] = setting
            self.mutate(job, "oracle/suite.json", change)
        result = self.summary()
        self.assertEqual(result["summary"]["qualified"], 0)
        self.assertEqual(result["attempt_counts"]["invalid"], 3)

    def test_each_oracle_case_must_use_the_fixed_micro_batch(self):
        job = self.fixture.job()
        path = job / "oracle/conversation-512.json"
        self.mutate(job, "oracle/conversation-512.json", lambda v: v.update(micro_batch_tokens=128))
        self.mutate(job, "oracle/suite.json", lambda v: v["cases"][0].update(oracle_json_sha256=sha(path)))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_engine_must_report_the_admitted_micro_batch(self):
        job = self.fixture.job()
        self.mutate(job, "engine/report.json", lambda v: v.pop("oracle_micro_batch_tokens"))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_independent_review_must_bind_the_oracle_policy(self):
        job = self.fixture.job()
        self.mutate(job, "independent-review.json", lambda v: v.pop("oracle_micro_batch_tokens"))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_partial_job_is_pending(self):
        job = self.fixture.job()
        (job / "independent-review.json").unlink()
        self.assertEqual(self.attempt()["classification"], "incomplete")

    def test_missing_or_changed_raw_invalidates_evidence(self):
        job = self.fixture.job()
        path = job / "engine/conversation-512.actual.f32"
        path.write_bytes(b"x" + path.read_bytes()[1:])
        self.assertEqual(self.attempt()["classification"], "invalid")
        path.unlink()
        self.assertIn("both missing", self.attempt()["reason"])

    def test_verified_archive_receipt_preserves_acceptance(self):
        job = self.fixture.job()
        self.fixture.archive(job)
        attempt = self.attempt()
        self.assertEqual(attempt["classification"], "qualified", attempt)
        self.assertEqual(attempt["retained_rows"]["archived_files"], 48)
        self.assertIsNotNone(attempt["retained_rows"]["archive_sha256"])

    def test_corrupt_archive_is_invalid(self):
        job = self.fixture.job()
        self.fixture.archive(job)
        path = job / "oracle/conversation-512.f32.zst"
        path.write_bytes(b"x" + path.read_bytes()[1:])
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_duplicate_passes_are_preserved_without_double_counting(self):
        self.fixture.job("first")
        self.fixture.job("second")
        result = self.summary()
        self.assertEqual(result["summary"]["qualified"], 1, result["cells"][0])
        self.assertEqual(len(result["cells"][0]["qualified_attempts"]), 2)
        self.assertTrue(result["cells"][0]["multiple_current_passes"])

    def test_later_pass_keeps_the_original_valid_failure(self):
        self.fixture.job("first-failure", numeric_failure=True)
        self.fixture.job("second-pass")
        result = self.summary()
        self.assertEqual(result["summary"]["qualified"], 1)
        self.assertEqual(result["attempt_counts"], {"failed": 1, "qualified": 1})
        self.assertEqual(len(result["cells"][0]["attempts"]), 2)

    def test_cross_backend_token_difference_invalidates_both(self):
        self.fixture.job("cpu")
        self.fixture.job("metal", backend="metal", token_text="different chat")
        result = self.summary()
        self.assertEqual(result["summary"]["qualified"], 0)
        self.assertEqual(result["attempt_counts"]["invalid"], 2)

    def test_changed_current_source_rejects_build(self):
        Path(next(iter(self.fixture.sources))).write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.fixture.matrix()

    def test_wrong_artifact_hash_is_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "run.json", lambda v: v["artifact"]["files"][0].update(sha256="f" * 64))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_compiled_helper_pin_must_match_current_build(self):
        job = self.fixture.job()
        self.mutate(job, "engine/report.json", lambda v: v["compiled_helper_sources"].update(
            {"examples/qualify_matrix.rs": "f" * 64}))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_source_aliases_match_canonical_paths_without_losing_counts(self):
        job = self.fixture.job()
        original = Path(next(iter(self.fixture.sources)))
        alias = self.fixture.root / "source-alias"
        alias.symlink_to(original)
        expected = sha(original)
        for name in ("run.json", "execution.json", "pins.json"):
            def change(value):
                target = value if name == "pins.json" else value["source_pins"]
                target[str(alias)] = expected
            self.mutate(job, name, change)
        self.mutate(job, "suite.json", lambda v: v["source_pins"].append(
            {"path": str(alias), "sha256": expected}))
        self.mutate(job, "engine/report.json", lambda v: v["input_pins"]["sources"].append(
            {"path": str(original), "sha256": expected}))
        # Bind the changed suite before checking canonical input source pins.
        self.mutate(job, "engine/report.json", lambda v: v["input_pins"].update(suite=pin(job / "suite.json")))
        self.assertEqual(self.attempt()["classification"], "qualified", self.attempt())
        self.mutate(job, "engine/report.json", lambda v: v["input_pins"]["sources"].pop())
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_full_must_have_observed_float_storage(self):
        job = self.fixture.job()
        self.mutate(job, "oracle/suite.json", lambda v: v["weight_storage"]["tensor_storage"][0].update(type="q4_K"))
        self.assertIn("FULL observed storage", self.attempt()["reason"])

    def test_archive_cannot_rebind_raw_hash(self):
        job = self.fixture.job()
        self.fixture.archive(job)
        self.mutate(job, "archive.json", lambda v: v["files"][0].update(decompressed_sha256="f" * 64))
        self.assertIn("archive raw binding", self.attempt()["reason"])

    def test_incomplete_rows_and_false_gate_claim_are_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "independent-review.json", lambda v: v["cases"][0]["rows"].pop())
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_unselected_internal_pass_is_invalid(self):
        job = self.fixture.job()
        self.mutate(job, "engine/report.json", lambda v: v["cases"][1]["internal"].update(passed=True))
        self.assertEqual(self.attempt()["classification"], "invalid")

    def test_duplicate_json_keys_and_path_escape_rejected(self):
        path = self.fixture.root / "duplicate.json"
        path.write_text('{"value":1,"value":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            s.Reader().load(path)
        for name in ("../escape", "/escape", "oracle/../escape", "oracle\\escape"):
            with self.assertRaises(ValueError):
                s.contained(self.fixture.runs, name)


class SmokeSummaryTests(unittest.TestCase):
    setUp = SummaryTests.setUp
    mutate = SummaryTests.mutate
    def smoke_job(self, name="smoke", stop=None, reuse=False, backend="cpu", profile="smoke-6"):
        internal_ids = s.SHORT_INTERNAL if profile == "smoke-6-short-state" else s.INTERNAL
        state_count = len(internal_ids)
        f = self.fixture
        job = f.job(name, backend=backend)
        prepared = read(job / "prepared.json")
        original_corpus = read(f.corpus)
        report = read(job / "engine/report.json")
        review = read(job / "independent-review.json")
        oracle_suite = read(job / "oracle/suite.json")
        suite = read(job / "suite.json")
        run = read(job / "run.json")
        selected = {c["id"]: c for c in prepared["cases"]}
        cases = [selected[c] for c in s.SMOKE]
        results = {c["id"]: c for c in report["cases"]}
        reviews = {c["id"]: c for c in review["cases"]}
        raw = review["raw_files"]
        n = 6 if stop is None else stop + 1
        model = f.root / "unused-model-path"
        run["model_entry"] = str(model)
        if reuse:
            old = f.root / (name + "-source")
            old.mkdir(); shutil.copytree(job / "oracle", old / "oracle")
            old_engine = f.root / (name + "-old-engine"); old_engine.write_text("original preparer")
            old_helper = f.root / (name + "-old-helper"); old_helper.write_text("original helper")
            prepared["preparation_binary_sha256"] = sha(old_engine)
            prepared["preparation_source_sha256"] = {"examples/qualify_matrix.rs": sha(old_helper)}
            write(job / "prepared.json", prepared)
            write(old / "prepared.json", prepared)
            shutil.copy2(f.corpus, old / "corpus.json")
            corpus_path = job / "corpus.json"; shutil.copy2(f.corpus, corpus_path)
        else:
            corpus_path = job / "corpus.json"
            chosen = {c["id"]: c for c in original_corpus["cases"]}
            write(corpus_path, {**original_corpus, "cases": [chosen[c] for c in s.SMOKE]})
            prepared["cases"] = cases
            prepared["corpus_sha256"] = sha(corpus_path)
            prepared["token_identity_sha256"] = hashlib.sha256(json.dumps(
                cases, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            write(job / "prepared.json", prepared)
        prep_sha = sha(job / "prepared.json")
        oracle_entries = {c["id"]: c for c in oracle_suite["cases"]}
        for label in (selected if reuse else s.SMOKE):
            path = job / "oracle" / f"{label}.json"
            doc = read(path); doc["prepared_sha256"] = prep_sha; write(path, doc)
            oracle_entries[label]["oracle_json_sha256"] = sha(path)
            if reuse:
                shutil.copy2(path, old / "oracle" / path.name)
        oracle_suite.update(cases=[oracle_entries[c["id"]] for c in prepared["cases"]],
                            completed_cases=len(prepared["cases"]), expected_cases=len(prepared["cases"]),
                            prepared_sha256=prep_sha, corpus_sha256=sha(corpus_path))
        write(job / "oracle/suite.json", oracle_suite)
        if reuse:
            write(old / "oracle/suite.json", oracle_suite)
            reference = f.root / (name + "-reference.build.json")
            write(reference, {"binary": f.binary})
            old_pins = {str(old_engine): sha(old_engine), str(old_helper): sha(old_helper), **f.pins}
            write(old / "pins.json", old_pins)
            for label, argv in {"prepare": [str(old_engine), "--prepare", str(model), str(old / "corpus.json"), str(old / "prepared.json")],
                                "oracle": [f.binary["path"], str(model), backend, str(old / "prepared.json"), str(old / "oracle")]}.items():
                write(old / f"{label}.command.json", {"name": label, "command": argv, "exit_code": 0, "timed_out": False})
            preserved = [{"source": pin(old / relative), "destination": relative}
                         for relative in ["prepared.json", "corpus.json", "oracle/suite.json",
                                          *(f"oracle/{case}{suffix}" for case in s.SMOKE for suffix in (".json", ".f32"))]]
            write(job / "oracle-reuse.json", {"schema_version": 1, "source_job": str(old),
                  "original_pins": pin(old / "pins.json"), "reference_build": pin(reference),
                  "original_commands": {label: pin(old / f"{label}.command.json") for label in ("prepare", "oracle")},
                  "preserved_files": preserved})
            run["oracle_reuse"] = pin(job / "oracle-reuse.json")
            run["source_pins"] = {**run["source_pins"], str(reference): sha(reference)}
        for index, label in enumerate(s.SMOKE):
            result, checked = results[label], reviews[label]
            result["status"] = "complete"
            if label not in internal_ids:
                result["internal"] = {"required": False, "attempted": False, "valid": None,
                                      "passed": None, "modes": [], "status": "not_selected"}
                checked.update(internal_required=False, internal_passed=None)
            checked["attempted"] = index < n
            if stop is not None:
                internal = {"required": label in internal_ids, "attempted": False, "valid": None,
                            "passed": None, "modes": [], "status": "not_run_after_numeric_failure" if label in internal_ids else "not_selected"}
                result["internal"] = internal; checked["internal_passed"] = None
                if index > stop:
                    result.update(attempted=False, valid=None, passed=None, status="not_run_after_numeric_failure",
                                  reference={"valid": None, "passed": None, "completed_rows": 0})
                    checked.update(valid_evidence=None, passed=None, numeric_passed=None, rows=[])
                    checked.pop("actual_raw_sha256")
                    (job / "engine" / f"{label}.actual.f32").unlink()
                elif index == stop:
                    for row in checked["rows"]:
                        row.update(actual_top=0, kl=.2, max_abs=1., rms=.5, actual_nll=2., nll_delta=1.)
                    result["reference"]["comparisons"] = copy.deepcopy(checked["rows"])
                    values = {"matching_argmax": 0, "mean_kl": .2, "max_kl": .2,
                              "reference_mean_nll": 1., "actual_mean_nll": 2., "mean_nll_delta": 1.}
                    checked.update(**values, numeric_passed=False, passed=False)
                    result["reference"].update(**values, passed=False); result.update(passed=False, status="numeric_failure")
                elif label in internal_ids:
                    result["passed"] = checked["passed"] = None
            write(job / "engine" / f"{label}.result.json", result)
        # Remove fixture-only rows outside the selected smoke scope.
        for directory in (job / "engine", job / "oracle"):
            for path in directory.iterdir():
                if path.name in ("report.json", "suite.json"):
                    continue
                if not any(path.name in (c + ".json", c + ".f32", c + ".actual.f32", c + ".result.json") for c in s.SMOKE):
                    path.unlink()
        final_results = [results[c] for c in s.SMOKE]
        final_reviews = [reviews[c] for c in s.SMOKE]
        passed = stop is None
        numeric_counts = {"planned": 6, "attempted": n, "passed": n - int(not passed), "failed": int(not passed), "unrun": 6 - n}
        state = "complete" if passed else "not_run_after_numeric_failure"
        early = None if passed else {"phase": "numeric", "case_id": s.SMOKE[stop], "reason": "numeric_failure"}
        suite.update(coverage_profile=profile, internal_cases=internal_ids, fail_fast=True, cases=[{"id": c} for c in s.SMOKE],
                     prepared=pin(job / "prepared.json"), corpus=pin(corpus_path),
                     expected_token_identity_sha256=prepared["token_identity_sha256"],
                     source_pins=[{"path": p, "sha256": h} for p,h in run["source_pins"].items()])
        write(job / "suite.json", suite); write(job / "pins.json", run["source_pins"])
        report.update(coverage_profile=profile, internal_case_ids=internal_ids, fail_fast=True, execution_policy="numeric_then_state",
                      selected_case_ids=s.SMOKE, cases=final_results, passed=passed,
                      token_identity_sha256=prepared["token_identity_sha256"], state_phase_status=state, early_stop=early,
                      numeric_counts=numeric_counts,
                      counts={"planned": 6, "attempted": n, "passed": sum(c["passed"] is True for c in final_results),
                              "failed": sum(c["passed"] is False for c in final_results), "unrun": 6-n,
                              "incomplete": sum(c["attempted"] and c["passed"] is None for c in final_results)},
                      internal_counts={"planned": state_count, "attempted": state_count if passed else 0,
                                       "passed": state_count if passed else 0, "failed": 0, "unrun": 0 if passed else state_count})
        report["input_pins"].update(suite=pin(job / "suite.json"), prepared=pin(job / "prepared.json"), corpus=pin(corpus_path),
                                    sources=suite["source_pins"])
        write(job / "engine/report.json", report)
        review.update(coverage_profile=profile, internal_case_ids=internal_ids, selected_case_ids=s.SMOKE, cases_planned=6, cases_checked=n,
                      rows_checked=128*n, unrun_case_ids=s.SMOKE[n:], oracle_cases_checked=6, oracle_rows_checked=768,
                      internal_modes_checked=2*state_count if passed else 0, internal_checks_checked=12*state_count if passed else 0,
                      cases=final_reviews, passed=passed, prepared_sha256=prep_sha,
                      numeric_counts=numeric_counts, state_phase_status=state, early_stop=early,
                      oracle_reuse=run.get("oracle_reuse"),
                      raw_files=[v for v in raw if v["path"] in {*(f"oracle/{c}.f32" for c in s.SMOKE),
                                                                 *(f"engine/{c}.actual.f32" for c in s.SMOKE[:n])}])
        write(job / "independent-review.json", review)
        run.update(coverage_profile=profile, passed=passed, engine_passed=passed, engine_exit_code=0 if passed else 1)
        run["commands"] = [c for c in run["commands"] if not reuse or c["name"] in ("engine", "independent")]
        for command in run["commands"]:
            command["exit_code"] = 1 if command["name"] == "engine" and not passed else 0
            write(job / f"{command['name']}.command.json", command)
        write(job / "run.json", run); write(job / "execution.json", {**run, "commands": run["commands"][:-1]})
        return job

    def smoke_matrix(self, profile="smoke-6"):
        f = self.fixture
        return s.Matrix(f.manifest, f.build, f.corpus, profile)

    def test_smoke_fresh_six_pass(self):
        job = self.smoke_job(); value = self.smoke_matrix().attempt(job)
        self.assertEqual(value["classification"], "qualified", value)
        self.assertEqual(value["cases_attempted"], 6)
        self.assertEqual(value["internal_checks_checked"], 36)
        self.assertEqual(value["retained_rows"]["files"], 12)

    def test_smoke_early_first_middle_and_last_rejection_are_valid_failures(self):
        for stop in (0, 2, 5):
            with self.subTest(stop=stop):
                job = self.smoke_job(f"stop-{stop}", stop=stop)
                value = self.smoke_matrix().attempt(job)
                self.assertEqual(value["classification"], "failed", value)
                self.assertTrue(value["valid_evidence"])
                self.assertFalse(value["qualified"])
                self.assertEqual(value["unrun_case_ids"], s.SMOKE[stop+1:])
                self.assertEqual(value["internal_checks_checked"], 0)
                self.assertEqual(value["retained_rows"]["files"], 7+stop)

    def test_smoke_reused_full_fixture_keeps_old_preparer_and_matches_fresh_identity(self):
        a = self.smoke_job("fresh")
        b = self.smoke_job("reuse", reuse=True, backend="metal")
        result = self.smoke_matrix().summarize(self.fixture.runs)
        attempts = [v for c in result["cells"] for v in c["attempts"]]
        self.assertTrue(all(v["qualified"] for v in attempts), attempts)
        self.assertEqual(len({v["token_identity_sha256"] for v in attempts}), 1)
        reused = self.smoke_matrix().attempt(b)
        self.assertEqual(reused["prepared_case_count"], 24)
        self.assertNotEqual(reused["preparation_binary_sha256"], self.fixture.binary["sha256"])
        self.assertIsNotNone(reused["oracle_reuse"])

    def test_smoke_archived_early_rejection_retains_all_oracles_and_attempted_actuals(self):
        job = self.smoke_job(stop=2); self.fixture.archive(job)
        value = self.smoke_matrix().attempt(job)
        self.assertEqual(value["classification"], "failed", value)
        self.assertEqual(value["retained_rows"]["archived_files"], 9)

    def test_smoke_unrun_pass_and_early_stop_lies_are_invalid(self):
        for change in ("unrun-pass", "early-stop", "states", "raw-count"):
            job = self.smoke_job(change, stop=2)
            if change == "unrun-pass":
                self.mutate(job, "engine/report.json", lambda v: v["cases"][3].update(passed=True))
            elif change == "early-stop":
                self.mutate(job, "engine/report.json", lambda v: v["early_stop"].update(case_id="prose-512"))
            elif change == "states":
                self.mutate(job, "independent-review.json", lambda v: v.update(internal_checks_checked=36))
            else:
                self.mutate(job, "independent-review.json", lambda v: v["raw_files"].pop())
            value = self.smoke_matrix().attempt(job)
            self.assertEqual(value["classification"], "invalid", (change, value))

    def test_smoke_old_current_source_or_other_profile_stays_stale(self):
        job = self.smoke_job()
        self.assertEqual(self.fixture.matrix().attempt(job)["classification"], "stale")
        self.mutate(job, "run.json", lambda v: v["source_pins"].update({self.fixture.binary["path"]: "f"*64}))
        self.assertEqual(self.smoke_matrix().attempt(job)["classification"], "stale")

    def test_smoke_state_rejection_retains_all_numeric_rows_and_36_checks(self):
        job = self.smoke_job()
        e = read(job / "engine/report.json"); v = read(job / "independent-review.json")
        e["cases"][1]["internal"]["modes"][0]["checks"][0]["passed"] = False
        e["cases"][1]["internal"]["passed"] = e["cases"][1]["passed"] = False
        v["cases"][1]["internal_passed"] = v["cases"][1]["passed"] = False
        e["internal_counts"].update(passed=2, failed=1); e["counts"].update(passed=5, failed=1)
        e["passed"] = v["passed"] = False
        write(job / "engine/conversation-512.result.json", e["cases"][1])
        write(job / "engine/report.json", e); write(job / "independent-review.json", v)
        for name in ("run.json", "execution.json"):
            doc = read(job / name); doc.update(passed=False, engine_passed=False, engine_exit_code=1)
            next(c for c in doc["commands"] if c["name"] == "engine")["exit_code"] = 1
            write(job / name, doc)
        cmd = read(job / "engine.command.json"); cmd["exit_code"] = 1; write(job / "engine.command.json", cmd)
        value = self.smoke_matrix().attempt(job)
        self.assertEqual(value["classification"], "failed", value)
        self.assertEqual(value["cases_attempted"], 6)
        self.assertEqual(value["internal_checks_checked"], 36)
        self.assertEqual(value["unrun_case_ids"], [])

    def test_smoke_optional_reuse_admission_is_pinned(self):
        job = self.smoke_job(reuse=True)
        admission = self.fixture.root / "pre-edit.json"; write(admission, {"passed": True})
        receipt_path = job / "oracle-reuse.json"; receipt = read(receipt_path)
        receipt["admission"] = pin(admission); write(receipt_path, receipt)
        for name in ("run.json", "execution.json", "independent-review.json"):
            doc = read(job / name); doc["oracle_reuse"] = pin(receipt_path); write(job / name, doc)
        self.assertEqual(self.smoke_matrix().attempt(job)["classification"], "qualified")
        write(admission, {"passed": False})
        self.assertEqual(self.smoke_matrix().attempt(job)["classification"], "invalid")

    def test_smoke_reuse_does_not_revalidate_changed_old_preparer_sources(self):
        job = self.smoke_job(reuse=True)
        (self.fixture.root / "smoke-old-helper").write_text("later harness edit")
        value = self.smoke_matrix().attempt(job)
        self.assertEqual(value["classification"], "qualified", value)

    def test_smoke_reuse_provenance_tamper_is_invalid(self):
        job = self.smoke_job(reuse=True)
        receipt = read(job / "oracle-reuse.json")
        source = Path(receipt["original_commands"]["oracle"]["path"])
        doc = read(source); doc["exit_code"] = 1; write(source, doc)
        value = self.smoke_matrix().attempt(job)
        self.assertEqual(value["classification"], "invalid", value)

    def test_short_state_fresh_pass_has_only_twelve_checks(self):
        profile = "smoke-6-short-state"
        job = self.smoke_job(profile=profile)
        value = self.smoke_matrix(profile).attempt(job)
        self.assertEqual(value["classification"], "qualified", value)
        self.assertEqual(value["coverage_profile"], profile)
        self.assertEqual(value["internal_checks_checked"], 12)
        self.assertEqual(value["cases_attempted"], 6)
        result = self.smoke_matrix(profile).summarize(self.fixture.runs)
        self.assertEqual(result["internal_case_ids"], ["conversation-512"])

    def test_short_state_valid_early_rejections_keep_unrun_null(self):
        profile = "smoke-6-short-state"
        for stop in (0, 2, 5):
            with self.subTest(stop=stop):
                job = self.smoke_job(f"short-stop-{stop}", stop=stop, profile=profile)
                value = self.smoke_matrix(profile).attempt(job)
                self.assertEqual(value["classification"], "failed", value)
                self.assertTrue(value["valid_evidence"])
                self.assertEqual(value["internal_checks_checked"], 0)
                self.assertEqual(value["unrun_case_ids"], s.SMOKE[stop+1:])
                self.assertTrue(all(c["passed"] is None for c in value["cases"][stop+1:]))

    def test_short_state_fresh_and_reused_full_share_numeric_identity(self):
        profile = "smoke-6-short-state"
        a = self.smoke_job("short-fresh", profile=profile)
        b = self.smoke_job("short-reused", reuse=True, backend="metal", profile=profile)
        first = self.smoke_matrix(profile).attempt(a)
        second = self.smoke_matrix(profile).attempt(b)
        self.assertEqual(first["classification"], "qualified", first)
        self.assertEqual(second["classification"], "qualified", second)
        self.assertEqual(first["token_identity_sha256"], second["token_identity_sha256"])
        self.assertEqual(second["prepared_case_count"], 24)
        self.assertNotEqual(second["preparation_binary_sha256"], self.fixture.binary["sha256"])

    def test_short_and_original_smoke_are_distinct_profiles(self):
        short = self.smoke_job("short", profile="smoke-6-short-state")
        old = self.smoke_job("old")
        self.assertEqual(self.smoke_matrix().attempt(short)["classification"], "stale")
        self.assertEqual(self.smoke_matrix("smoke-6-short-state").attempt(old)["classification"], "stale")
        short_value = self.smoke_matrix("smoke-6-short-state").attempt(short)
        old_value = self.smoke_matrix().attempt(old)
        self.assertEqual(short_value["token_identity_sha256"], old_value["token_identity_sha256"])

    def test_short_state_rejects_extra_states_counts_and_mixed_receipt_profile(self):
        profile = "smoke-6-short-state"
        for change in ("count", "selection", "skipped-pass", "profile"):
            job = self.smoke_job(change, profile=profile)
            if change == "count":
                self.mutate(job, "independent-review.json", lambda v: v.update(internal_checks_checked=36))
            elif change == "selection":
                self.mutate(job, "engine/report.json", lambda v: v.update(internal_case_ids=s.INTERNAL))
            elif change == "skipped-pass":
                self.mutate(job, "engine/report.json", lambda v: v["cases"][2]["internal"].update(passed=True))
            else:
                self.mutate(job, "independent-review.json", lambda v: v.update(coverage_profile="smoke-6"))
            with self.subTest(change=change):
                value = self.smoke_matrix(profile).attempt(job)
                self.assertEqual(value["classification"], "invalid", value)

    def test_short_state_rejection_keeps_all_numeric_rows_and_twelve_checks(self):
        profile = "smoke-6-short-state"
        job = self.smoke_job(profile=profile)
        e = read(job / "engine/report.json"); v = read(job / "independent-review.json")
        e["cases"][1]["internal"]["modes"][0]["checks"][0]["passed"] = False
        e["cases"][1]["internal"]["passed"] = e["cases"][1]["passed"] = False
        v["cases"][1]["internal_passed"] = v["cases"][1]["passed"] = False
        e["internal_counts"].update(passed=0, failed=1); e["counts"].update(passed=5, failed=1)
        e["passed"] = v["passed"] = False
        write(job / "engine/conversation-512.result.json", e["cases"][1])
        write(job / "engine/report.json", e); write(job / "independent-review.json", v)
        for name in ("run.json", "execution.json"):
            doc = read(job / name); doc.update(passed=False, engine_passed=False, engine_exit_code=1)
            next(c for c in doc["commands"] if c["name"] == "engine")["exit_code"] = 1
            write(job / name, doc)
        cmd = read(job / "engine.command.json"); cmd["exit_code"] = 1; write(job / "engine.command.json", cmd)
        value = self.smoke_matrix(profile).attempt(job)
        self.assertEqual(value["classification"], "failed", value)
        self.assertEqual(value["internal_checks_checked"], 12)
        self.assertEqual(value["cases_attempted"], 6)

    def test_short_state_archived_rejection_keeps_all_six_oracles(self):
        profile = "smoke-6-short-state"
        job = self.smoke_job(stop=2, profile=profile)
        self.fixture.archive(job)
        value = self.smoke_matrix(profile).attempt(job)
        self.assertEqual(value["classification"], "failed", value)
        self.assertTrue(value["valid_evidence"])
        self.assertEqual(value["internal_checks_checked"], 0)

    def test_short_state_does_not_promote_incomplete_numeric_only_job(self):
        profile = "smoke-6-short-state"
        job = self.smoke_job(profile=profile)
        (job / "engine/report.json").unlink()
        value = self.smoke_matrix(profile).attempt(job)
        self.assertFalse(value["valid_evidence"])
        self.assertNotEqual(value["classification"], "qualified")


if __name__ == "__main__":
    unittest.main()
