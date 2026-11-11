"""No-model tests for complete evidence, raw math, and false-pass rejection."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import verify_results as v
import cpu_reference_policy as cpu


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pin(path, identity=False):
    result = {"path": str(path), "sha256": v.digest(path)}
    if identity:
        result["identity"] = v.stamp(path)
    return result


def fake_job(root, numeric_failure=False, internal_failure=False, pilot=False,
             suite_micro_batch=8, case_micro_batch=8):
    """Small synthetic vocab, but full row/case counts and all receipt layers."""
    root = root.resolve()
    (root / "oracle").mkdir()
    (root / "engine").mkdir()
    model, binary, source = [root / n for n in ("model.gguf", "engine-bin", "producer.cpp")]
    for path in (model, binary, source):
        path.write_text(path.name)
    source_pins = {str(binary): v.digest(binary), str(source): v.digest(source)}
    cpu_files = {}
    for key, name in [("plugin", "libggml-cpu-apple_m4.so"), ("blas_plugin", "libggml-blas.so")]:
        path = root / name
        path.write_text(name)
        cpu_files[key] = {"path": str(path), "sha256": v.digest(path), "size_bytes": path.stat().st_size}
        source_pins[str(path)] = v.digest(path)
    cpu_policy = {"schema_version": 1, "id": cpu.POLICY, "source_revision": cpu.REVISION,
                  "source_patch_sha256": cpu.PATCH_SHA256, **cpu_files}
    cpu_runtime = {"policy": cpu_policy, "cpu_plugin_image_count": 1, "blas_plugin_image_count": 1,
                   "registry_count": 2, "gpu_backends_loaded": False,
                   "runtime_images": [str(binary), *(value["path"] for value in cpu_files.values())]}
    write(root / "pins.json", source_pins)
    files = [{"name": model.name, "size_bytes": model.stat().st_size, "sha256": v.digest(model)}]
    cases, texts = [], []
    for domain in ["conversation", "json", "retrieval", "prose", "rust", "arithmetic", "extraction", "multilingual"]:
        for context in v.CONTEXTS:
            case = {"id": f"{domain}-{context}", "domain": domain, "context_tokens": context,
                    "rendered_prompt_utf8": "complete chat\n", "continuation_utf8": "natural continuation é",
                    "prompt_original_ids": [0, 1, 2], "prompt_tokens": [0, 1, 2],
                    "continuation_original_ids": [1] * 129, "teacher_forcing_tokens": [1] * 128,
                    "prompt_add_special": True, "continuation_add_special": False, "parse_special": True}
            cases.append(case)
            texts.append({"id": case["id"], "domain": case["domain"], "context_tokens": context,
                          "messages": [{"role": "user", "content": "complete message"}],
                          "continuation_text": case["continuation_utf8"]})
    if pilot:
        cases, texts = cases[:2], texts[:2]
    internal_ids = [case["id"] for case in cases] if pilot else v.INTERNAL_CASES
    corpus = {"schema_version": 1, "suite_id": "test-matrix", "rows": 128, "cases": texts}
    write(root / "corpus.json", corpus)
    token_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    prepared = {"schema_version": 1, "suite_id": "test-matrix", "rows": 128,
                "corpus_sha256": v.digest(root / "corpus.json"), "model_sha256": v.digest(model),
                "model_files": files, "architecture": "llama", "vocab_size": 3,
                "model_context_length": 4096, "token_identity_sha256": token_hash, "cases": cases}
    write(root / "prepared.json", prepared)
    write(root / "shared-identity.json", {"model_id": "model", "corpus_sha256": prepared["corpus_sha256"],
                                         "token_identity_sha256": token_hash, "cases": cases})
    options = {"decode_tokens": 8, "prefill_chunk": 128, "comparison_prefill_chunk": 32, "cache_max_bytes": 10000}
    suite = {"schema_version": 1, "suite_id": "test-matrix", "backend": "cpu",
             "prepared": pin(root / "prepared.json"), "corpus": pin(root / "corpus.json"),
             "expected_binary_sha256": v.digest(binary), "expected_token_identity_sha256": token_hash,
             "source_pins": [{"path": p, "sha256": sha} for p, sha in source_pins.items()],
             "cases": [], "internal": options, "internal_cases": internal_ids}
    oracle_suite = {"status": "complete", "completed_cases": len(cases), "expected_cases": len(cases),
                    "backend": "cpu", "suite_id": "test-matrix", "model_files": files,
                    "model_files_sha256_verified_once": True, "model_file_stats_before": [], "model_file_stats_after": [],
                    "kv_dtype": "F16", "threads": 8, "prefill_chunk_tokens": 128, "new_context_per_case": True,
                    "micro_batch_tokens": suite_micro_batch,
                    "cpu_attention_policy": cpu_policy, "cpu_runtime_before": cpu_runtime, "cpu_runtime_after": cpu_runtime,
                    "prepared_sha256": v.digest(root / "prepared.json"), "corpus_sha256": prepared["corpus_sha256"],
                    "producer_source_sha256": v.digest(source), "cases": []}
    results, refs = [], []
    for index, case in enumerate(cases):
        label = case["id"]
        r = np.tile(np.array([1, 2, -1], dtype="<f4"), (128, 1))
        a = r.copy()
        if numeric_failure and index == 0:
            a[:] = [-1, 0, 3]
        raw_path, actual_path = root / "oracle" / f"{label}.f32", root / "engine" / f"{label}.actual.f32"
        raw_path.write_bytes(r.tobytes())
        actual_path.write_bytes(a.tobytes())
        oracle = {"source": "llama.cpp pinned", "producer_source_sha256": v.digest(source), "schema_version": 1,
                  "case_id": label, "context_tokens": case["context_tokens"], "backend": "cpu", "vocab_size": 3,
                  "micro_batch_tokens": case_micro_batch, "cpu_attention_policy": cpu_policy,
                  "model_sha256": prepared["model_sha256"], "model_files": files,
                  "prepared_sha256": v.digest(root / "prepared.json"), "tokenization_verified": True,
                  "prompt_tokens": case["prompt_tokens"], "teacher_forcing_tokens": case["teacher_forcing_tokens"],
                  "continuation_tokens": [1] * 128, "logits_file": raw_path.name, "oracle_raw_sha256": v.digest(raw_path)}
        oracle_path = root / "oracle" / f"{label}.json"
        write(oracle_path, oracle)
        suite["cases"].append({"id": label, "reference": pin(oracle_path), "logits_sha256": v.digest(raw_path)})
        oracle_suite["cases"].append({"id": label, "oracle_json": oracle_path.name, "oracle_raw": raw_path.name,
                                      "oracle_json_sha256": v.digest(oracle_path), "oracle_raw_sha256": v.digest(raw_path),
                                      "oracle_raw_bytes": raw_path.stat().st_size})
        refs.append({"json": pin(oracle_path, True), "raw": pin(raw_path, True)})
        rows = [v.metric(r[i], a[i], 1, i) for i in range(128)]
        matches = sum(row["reference_top"] == row["actual_top"] for row in rows)
        mean_kl, max_kl = sum(row["kl"] for row in rows) / 128, max(row["kl"] for row in rows)
        numeric = v.numeric_pass(matches, mean_kl, max_kl)
        ref = {"valid": True, "passed": numeric, "completed_rows": 128, "comparisons": rows,
               "actual_logits_file": str(actual_path), "actual_logits_sha256": v.digest(actual_path),
               "actual_logits_bytes": actual_path.stat().st_size, "prefill_chunk_tokens": 128,
               "matching_argmax": matches, "mean_kl": mean_kl, "max_kl": max_kl,
               "reference_mean_nll": sum(row["reference_nll"] for row in rows) / 128,
               "actual_mean_nll": sum(row["actual_nll"] for row in rows) / 128,
               "mean_nll_delta": sum(row["nll_delta"] for row in rows) / 128}
        modes = []
        internal_pass = not (internal_failure and index == 0)
        for mode in ["greedy", "seeded_penalties"]:
            checks = []
            for check_index, label_check in enumerate(v.CHECKS):
                details = [{"prefill_chunk": 2}, {"cancel_after_tokens": 2}, {"prefill_chunk": 2},
                           {"reused_prompt_tokens": 0, "retained_bytes": 100}, {"reused_prompt_tokens": 3},
                           {"reused_prompt_tokens": 0}][check_index]
                actual = [1] * 8
                if not internal_pass and check_index == 0:
                    actual[4] = 2
                check_pass = actual == [1] * 8
                checks.append({"name": label_check, "passed": check_pass, "first_mismatch": None if check_pass else 4,
                               "actual_tokens": actual, "details": details})
            modes.append({"mode": mode, "baseline_tokens": [1] * 8, "checks": checks})
        internal = ({"required": True, "attempted": True, "status": "complete", "valid": True, "passed": internal_pass, "modes": modes}
                    if label in internal_ids else {"required": False, "attempted": False, "status": "not_selected", "valid": None, "passed": None, "modes": []})
        result = {"id": label, "domain": case["domain"], "context_tokens": case["context_tokens"],
                  "prompt_tokens": 3, "rows": 128, "attempted": True, "valid": True, "passed": numeric and internal_pass,
                  "reference": ref, "internal": internal}
        write(root / "engine" / f"{label}.result.json", result)
        results.append(result)
    write(root / "suite.json", suite)
    write(root / "oracle/suite.json", oracle_suite)
    passed = all(result["passed"] for result in results)
    count_pass = sum(result["passed"] for result in results)
    internal_count_pass = sum(result["internal"]["passed"] is True for result in results)
    report = {"schema_version": 1, "post_run_pins_verified": True, "valid": True, "passed": passed,
              "runtime_dispatch_overrides": [], "model_loads": 1, "backend": "cpu", "suite_id": "test-matrix",
              "oracle_micro_batch_tokens": 8,
              "architecture": "llama", "model_files": files, "token_identity_sha256": token_hash,
              "limits": {"rows": 128, "min_argmax_fraction": .95, "mean_kl": .02, "max_kl": .1,
                         "nll_gate": None, "kl_direction": "reference || actual"},
              "compiled_helper_sources": {"dummy.rs": v.digest(binary)}, "internal_options": options,
              "internal_case_ids": internal_ids,
              "internal_counts": {"planned": len(internal_ids), "attempted": len(internal_ids),
                                  "passed": internal_count_pass, "failed": len(internal_ids) - internal_count_pass},
              "input_pins": {"suite": pin(root / "suite.json", True), "prepared": pin(root / "prepared.json", True),
                             "corpus": pin(root / "corpus.json", True), "binary": pin(binary, True),
                             "sources": [pin(Path(p), True) for p in source_pins], "model_files": [pin(model, True)],
                             "references": refs}, "cases": results,
              "counts": {"planned": len(cases), "attempted": len(cases), "passed": count_pass, "failed": len(cases)-count_pass}}
    write(root / "engine/report.json", report)
    commands = []
    for label in ["prepare", "oracle", "engine"]:
        argv = ([binary, "--prepare", model, root / "corpus.json", root / "prepared.json"] if label == "prepare"
                else [binary, model, "cpu", root / ("prepared.json" if label == "oracle" else "suite.json"), root / label])
        command = {"name": label, "exit_code": 0 if label != "engine" or passed else 1, "command": list(map(str, argv)), "seconds": 1.0}
        write(root / f"{label}.command.json", command)
        commands.append(command)
    write(root / "execution.json", {"post_run_pins_verified": True, "engine_valid": True, "engine_passed": passed,
                                    "engine_exit_code": 0 if passed else 1, "commands": commands,
                                    "source_pins": source_pins, "identity": pin(root / "shared-identity.json"),
                                    "model_id": "model", "model_entry": str(model), "cpu_attention_policy": cpu_policy})


class RawMathTests(unittest.TestCase):
    def test_hand_computed_asymmetric_kl_nll_and_first_tie(self):
        r, a = np.array([np.log(3), 0], dtype="<f4"), np.zeros(2, dtype="<f4")
        row = v.metric(r, a, 1, 7)
        self.assertAlmostEqual(row["kl"], .75*np.log(1.5)+.25*np.log(.5), places=7)
        self.assertAlmostEqual(row["reference_nll"], np.log(4), places=7)
        self.assertAlmostEqual(row["nll_delta"], -np.log(2), places=7)
        self.assertEqual(row["actual_top"], 0)
        reverse = v.metric(a, r, 1, 7)
        self.assertGreater(abs(reverse["kl"]-row["kl"]), .01)

    def test_fixed_gates_and_tolerance(self):
        self.assertTrue(v.numeric_pass(122, .02, .1))
        for args in [(121, 0, 0), (128, .020000000001, .1), (128, .02, .100000000001)]:
            self.assertFalse(v.numeric_pass(*args))
        v.near(0, 1e-10, "boundary")
        with self.assertRaises(ValueError):
            v.near(0, 1.0001e-10, "beyond boundary")
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                v.metric(np.array([0], dtype="<f4"), np.array([bad], dtype="<f4"), 0, 0)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_complete_matrix_passes_all_counts(self):
        fake_job(self.root)
        review = v.review(self.root)
        self.assertTrue(review["valid_evidence"] and review["passed"])
        self.assertEqual((review["rows_checked"], review["internal_checks_checked"], len(review["raw_files"])), (3072, 36, 48))
        self.assertEqual(review["internal_case_ids"], v.INTERNAL_CASES)
        self.assertIsNone(review["cases"][1]["internal_passed"])
        self.assertFalse(review["pilot"])

    def test_numeric_failure_stays_valid_and_failed(self):
        fake_job(self.root, numeric_failure=True)
        review = v.review(self.root)
        self.assertTrue(review["valid_evidence"])
        self.assertFalse(review["passed"])
        self.assertFalse(review["cases"][0]["numeric_passed"])

    def test_wrong_suite_micro_batch_rejected(self):
        for value in (None, 1, 128, 8.0):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                fake_job(path, suite_micro_batch=value)
                with self.assertRaisesRegex(ValueError, "suite micro batch"):
                    v.review(path)

    def test_wrong_case_micro_batch_rejected(self):
        for value in (None, 1, 128, 8.0):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                fake_job(path, case_micro_batch=value)
                with self.assertRaisesRegex(ValueError, "case micro batch"):
                    v.review(path)

    def test_missing_engine_micro_batch_rejected(self):
        fake_job(self.root)
        path = self.root / "engine/report.json"
        report = v.load(path)
        del report["oracle_micro_batch_tokens"]
        write(path, report)
        with self.assertRaisesRegex(ValueError, "engine oracle micro batch"):
            v.review(self.root)

    def test_internal_failure_stays_valid_and_failed(self):
        fake_job(self.root, internal_failure=True)
        review = v.review(self.root)
        self.assertTrue(review["valid_evidence"])
        self.assertFalse(review["passed"])
        self.assertTrue(review["cases"][0]["numeric_passed"])
        self.assertFalse(review["cases"][0]["internal_passed"])

    def test_pilot_needs_explicit_mode(self):
        fake_job(self.root, pilot=True)
        with self.assertRaisesRegex(ValueError, "all24"):
            v.review(self.root)
        result = v.review(self.root, pilot=True)
        self.assertTrue(result["pilot"])
        self.assertEqual(result["rows_checked"], 256)

    def tamper_case(self, mutate):
        fake_job(self.root)
        report = v.load(self.root / "engine/report.json")
        mutate(report["cases"][0])
        write(self.root / "engine/report.json", report)
        write(self.root / "engine/conversation-512.result.json", report["cases"][0])

    def test_changed_row_metrics_rejected(self):
        self.tamper_case(lambda c: c["reference"]["comparisons"][3].update(kl=.001))
        with self.assertRaisesRegex(ValueError, "row3 kl"):
            v.review(self.root)

    def test_wrong_actual_history_id_rejected(self):
        self.tamper_case(lambda c: c["reference"]["comparisons"][0].update(teacher_token=2))
        with self.assertRaisesRegex(ValueError, "teacher_token"):
            v.review(self.root)

    def test_missing_mode_rejected(self):
        self.tamper_case(lambda c: c["internal"]["modes"].pop())
        with self.assertRaisesRegex(ValueError, "internal modes"):
            v.review(self.root)

    def test_omitted_required_state_check_rejected(self):
        self.tamper_case(lambda c: c.update(internal={"required": False, "attempted": False, "status": "not_selected", "valid": None, "passed": None, "modes": []}))
        with self.assertRaisesRegex(ValueError, "required internal"):
            v.review(self.root)

    def test_fake_pass_for_unselected_state_check_rejected(self):
        fake_job(self.root)
        report = v.load(self.root / "engine/report.json")
        report["cases"][1]["internal"]["passed"] = True
        write(self.root / "engine/report.json", report)
        write(self.root / "engine/conversation-2048.result.json", report["cases"][1])
        with self.assertRaisesRegex(ValueError, "no pass claim"):
            v.review(self.root)

    def test_false_cache_reuse_pass_rejected(self):
        self.tamper_case(lambda c: c["internal"]["modes"][0]["checks"][4]["details"].update(reused_prompt_tokens=0))
        with self.assertRaisesRegex(ValueError, "internal acceptance"):
            v.review(self.root)

    def test_actual_raw_corruption_rejected(self):
        fake_job(self.root)
        path = self.root / "engine/conversation-512.actual.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "actual raw hash"):
            v.review(self.root)

    def test_missing_row_even_with_new_hash_rejected(self):
        fake_job(self.root)
        path = self.root / "engine/conversation-512.actual.f32"
        path.write_bytes(path.read_bytes()[:-12])
        report = v.load(self.root / "engine/report.json")
        report["cases"][0]["reference"]["actual_logits_sha256"] = v.digest(path)
        report["cases"][0]["reference"]["actual_logits_bytes"] = path.stat().st_size
        write(self.root / "engine/report.json", report)
        write(self.root / "engine/conversation-512.result.json", report["cases"][0])
        with self.assertRaisesRegex(ValueError, "raw byte count"):
            v.review(self.root)

    def test_backend_mismatch_rejected(self):
        fake_job(self.root)
        report = v.load(self.root / "engine/report.json")
        report["backend"] = "metal"
        write(self.root / "engine/report.json", report)
        with self.assertRaisesRegex(ValueError, "backend differs"):
            v.review(self.root)

    def test_duplicate_json_keys_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"passed":false,"passed":true}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            v.load(path)


if __name__ == "__main__":
    unittest.main()
