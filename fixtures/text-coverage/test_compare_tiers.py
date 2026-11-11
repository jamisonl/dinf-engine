"""Tiny no-model tests for descriptive precision comparisons and archives."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import compare_tiers as c
import test_verify_results as fixture
import verify_results as v


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def tier_job(root, tier, offset, profile="extended-24"):
    root.mkdir()
    if profile in ("smoke-6", "smoke-6-short-state"):
        from test_smoke_results import smoke_job
        smoke_job(root, profile=profile)
    else:
        fixture.fake_job(root)
    review = v.review(root, profile=profile)
    p = c.load(root / "prepared.json")
    p["suite_id"] = c.SUITE
    p["model_sha256"] = hashlib.sha256(tier.encode()).hexdigest()
    p["model_files"] = [{"name": f"{tier}.gguf", "size_bytes": 100, "sha256": p["model_sha256"]}]
    write(root / "prepared.json", p)
    prep_sha = c.digest(root / "prepared.json")
    e = c.load(root / "engine/report.json")
    e.update(suite_id=c.SUITE, model_files=p["model_files"])
    write(root / "engine/report.json", e)
    o = c.load(root / "oracle/suite.json")
    o.update(suite_id=c.SUITE, prepared_sha256=prep_sha, model_files=p["model_files"], source="llama.cpp pinned",
             weight_storage={"tensor_storage": [{"type": {"FULL": "bf16", "Q8_0": "q8_0", "Q4_K_M": "q4_K"}[tier],
                                                  "tensor_count": 1, "encoded_tensor_bytes": 100}], "total_tensor_count": 1})
    review.update(suite_id=c.SUITE, prepared_sha256=prep_sha)
    for index, case in enumerate(p["cases"]):
        raw_path = root / "oracle" / f"{case['id']}.f32"
        rows = np.tile(np.array([1+offset, 2-offset, -1], dtype="<f4"), (128, 1))
        raw_path.write_bytes(rows.tobytes())
        sha = c.digest(raw_path)
        path = root / "oracle" / f"{case['id']}.json"
        oracle = c.load(path)
        oracle.update(prepared_sha256=prep_sha, model_files=p["model_files"], model_sha256=p["model_sha256"],
                      oracle_raw_sha256=sha, continuation_tokens=[int(rows[0].argmax())]*128)
        write(path, oracle)
        o["cases"][index].update(oracle_json_sha256=c.digest(path), oracle_raw_sha256=sha)
        review["cases"][index]["oracle_raw_sha256"] = sha
        for raw in review["raw_files"]:
            if raw["path"] == f"oracle/{case['id']}.f32":
                raw["sha256"] = sha
    write(root / "oracle/suite.json", o)
    write(root / "independent-review.json", review)
    x = c.load(root / "execution.json")
    x.update(backend="cpu", source_pins={"fixed-engine-source": "0"*64},
             artifact={"id": tier, "tier": tier, "quantization": "BF16" if tier == "FULL" else tier,
                       "files": [{"filename": p["model_files"][0]["name"], "size_bytes": 100, "sha256": p["model_sha256"]}]})
    write(root / "execution.json", x)


def archive_job(job, zstd):
    review = c.load(job / "independent-review.json")
    records = []
    for raw in review["raw_files"]:
        path = job / raw["path"]
        output = path.with_name(path.name + ".zst")
        with output.open("xb") as stream:
            subprocess.run([zstd, "-q", "-c", str(path)], stdout=stream, check=True)
        records.append({"raw_path": raw["path"], "raw_sha256": raw["sha256"], "decompressed_sha256": raw["sha256"],
                        "raw_bytes": raw["size_bytes"], "decompressed_bytes": raw["size_bytes"],
                        "compressed_path": raw["path"] + ".zst", "compressed_bytes": output.stat().st_size,
                        "compressed_sha256": c.digest(output)})
        path.unlink()
    write(job / "archive.json", {"suite_id": c.SUITE, "prepared_sha256": c.digest(job / "prepared.json"),
                                 "independent_review": {"sha256": c.digest(job / "independent-review.json")}, "files": records})


class MetricsTests(unittest.TestCase):
    def test_direction_teacher_nll_and_argmax(self):
        f = np.array([np.log(3), 0], dtype="<f4")
        q = np.zeros(2, dtype="<f4")
        row = c.row_metrics(f, q, 1, 0)
        self.assertAlmostEqual(row["kl_full_to_quant"], .75*np.log(1.5)+.25*np.log(.5), places=7)
        self.assertAlmostEqual(row["nll_delta_quant_minus_full"], -np.log(2), places=7)
        self.assertEqual((row["full_top"], row["quant_top"]), (0, 0))
        self.assertNotAlmostEqual(row["kl_full_to_quant"], c.row_metrics(q, f, 1, 0)["kl_full_to_quant"], places=3)

    def test_reject_nonfinite_and_bad_teacher(self):
        for data, teacher in [([np.nan], 0), ([np.inf], 0), ([0], 1), ([0], -1), ([0], True)]:
            with self.assertRaises(ValueError):
                c.row_metrics(np.array([0], dtype="<f4"), np.array(data, dtype="<f4"), teacher, 0)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.jobs = [self.root / name for name in ("full", "high", "low")]
        for path, tier, offset in zip(self.jobs, ("FULL", "Q8_0", "Q4_K_M"), (0, .01, 1)):
            tier_job(path, tier, offset)

    def compare(self):
        return c.compare(*self.jobs, shutil.which("zstd"))

    def test_raw_complete_descriptive_comparison_has_no_gate(self):
        result = self.compare()
        self.assertTrue(result["valid_evidence"])
        self.assertEqual(result["rows_per_pair"], 3072)
        self.assertEqual(len(result["raw_inputs"]), 72)
        self.assertIsNone(result["quantization_acceptance_gate"])
        self.assertIsNone(result["nll_acceptance_gate"])
        self.assertNotIn("passed", result)
        self.assertEqual(result["summary"]["Q4_K_M"]["all"]["matching_argmax"], 0)
        self.assertGreater(result["summary"]["Q4_K_M"]["all"]["mean_kl_full_to_quant"], .1)

    def test_lossless_archive_matches_raw_metrics(self):
        zstd = shutil.which("zstd")
        if zstd is None:
            self.skipTest("zstd not installed")
        before = self.compare()
        archive_job(self.jobs[1], zstd)
        after = self.compare()
        self.assertEqual(before["summary"], after["summary"])
        self.assertEqual(sum(x["storage"] == "verified_zstd" for x in after["raw_inputs"]), 24)

    def test_corrupt_archive_rejected(self):
        zstd = shutil.which("zstd")
        if zstd is None:
            self.skipTest("zstd not installed")
        archive_job(self.jobs[1], zstd)
        path = self.jobs[1] / "oracle/conversation-512.f32.zst"
        path.write_bytes(path.read_bytes() + b"bad")
        with self.assertRaisesRegex(ValueError, "compressed bytes/hash"):
            self.compare()

    def test_same_size_raw_corruption_rejected(self):
        path = self.jobs[1] / "oracle/conversation-512.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.compare()

    def test_failed_engine_review_is_not_quantization_evidence(self):
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value["passed"] = False
        write(path, value)
        with self.assertRaisesRegex(ValueError, "passing nonpilot"):
            self.compare()

    def test_pilot_rejected(self):
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value["pilot"] = True
        write(path, value)
        with self.assertRaisesRegex(ValueError, "passing nonpilot"):
            self.compare()

    def test_changed_token_identity_rejected(self):
        path = self.jobs[1] / "prepared.json"
        value = c.load(path)
        value["token_identity_sha256"] = "f"*64
        write(path, value)
        with self.assertRaises(ValueError):
            self.compare()

    def test_backend_mismatch_rejected(self):
        path = self.jobs[1] / "execution.json"
        value = c.load(path)
        value["backend"] = "metal"
        write(path, value)
        with self.assertRaisesRegex(ValueError, "backend differs"):
            self.compare()

    def test_wrong_actual_storage_dtype_rejected(self):
        path = self.jobs[1] / "oracle/suite.json"
        value = c.load(path)
        value["weight_storage"]["tensor_storage"][0]["type"] = "q4_K"
        write(path, value)
        with self.assertRaisesRegex(ValueError, "storage lacks"):
            self.compare()

    def test_old_or_missing_suite_micro_batch_rejected(self):
        path = self.jobs[1] / "oracle/suite.json"
        value = c.load(path)
        for batch in (None, 1, 128, 8.0):
            with self.subTest(batch=batch):
                value["micro_batch_tokens"] = batch
                write(path, value)
                with self.assertRaisesRegex(ValueError, "suite micro batch"):
                    self.compare()

    def test_old_case_micro_batch_rejected(self):
        path = self.jobs[1] / "oracle/conversation-512.json"
        value = c.load(path)
        value["micro_batch_tokens"] = 128
        write(path, value)
        suite_path = self.jobs[1] / "oracle/suite.json"
        suite = c.load(suite_path)
        suite["cases"][0]["oracle_json_sha256"] = c.digest(path)
        write(suite_path, suite)
        with self.assertRaisesRegex(ValueError, "case micro batch"):
            self.compare()

    def test_runtime_pin_difference_rejected(self):
        path = self.jobs[1] / "execution.json"
        value = c.load(path)
        value["source_pins"]["fixed-engine-source"] = "1"*64
        write(path, value)
        with self.assertRaisesRegex(ValueError, "runtime pins differ"):
            self.compare()

    def test_stock_cpu_receipt_cannot_mix_with_repaired_tiers(self):
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value.pop("cpu_attention_policy")
        write(path, value)
        with self.assertRaisesRegex(ValueError, "CPU policy differs"):
            self.compare()

    def test_different_cpu_kernel_bytes_rejected(self):
        for name in ("independent-review.json", "execution.json", "oracle/suite.json"):
            path = self.jobs[1] / name
            value = c.load(path)
            value["cpu_attention_policy"]["plugin"]["sha256"] = "1" * 64
            write(path, value)
        # The case binding must reject this before any raw-row comparison.
        with self.assertRaisesRegex(ValueError, "per-case CPU reference policy"):
            self.compare()

    def test_missing_case_and_archive_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            c.contained(self.jobs[0], "../escape.f32")
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value["cases"].pop()
        write(path, value)
        with self.assertRaisesRegex(ValueError, "case order"):
            self.compare()


class SmokeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.jobs = [self.root / name for name in ("full", "high", "low")]
        for path, tier, offset in zip(self.jobs, ("FULL", "Q8_0", "Q4_K_M"), (0, .01, 1)):
            tier_job(path, tier, offset, profile="smoke-6")

    def compare(self):
        # Raw tiny fixtures need no decompressor or model process.
        return c.compare(*self.jobs)

    def test_fresh_smoke_compares_exact_six_cases_without_new_metric_gate(self):
        result = self.compare()
        self.assertTrue(result["valid_evidence"])
        self.assertEqual(result["coverage_profile"], "smoke-6")
        self.assertEqual((result["cases_compared"], result["rows_per_pair"]), (6, 768))
        self.assertEqual([case["id"] for case in result["cases"]], c.SMOKE_CASES)
        self.assertEqual(len(result["raw_inputs"]), 18)
        self.assertIsNone(result["quantization_acceptance_gate"])
        self.assertIsNone(result["nll_acceptance_gate"])
        self.assertNotIn("passed", result)

    def test_mixed_profile_is_rejected(self):
        extended = self.root / "extended-high"
        tier_job(extended, "Q8_0", .01)
        self.jobs[1] = extended
        with self.assertRaisesRegex(ValueError, "coverage profile differs across tiers"):
            self.compare()

    def test_full24_oracle_reuse_stays_unreported(self):
        path = self.jobs[1] / "execution.json"
        value = c.load(path)
        value["oracle_reuse"] = {"path": "original-reuse.json", "sha256": "1" * 64}
        write(path, value)
        with self.assertRaisesRegex(ValueError, "fresh six-case oracle jobs only"):
            self.compare()

    def test_missing_numeric_or_state_work_cannot_pass_tier_admission(self):
        path = self.jobs[1] / "engine/report.json"
        original = c.load(path)
        for field, replacement in [("numeric_counts", {"planned": 6, "attempted": 5, "passed": 5, "failed": 0, "unrun": 1}),
                                   ("internal_counts", {"planned": 3, "attempted": 2, "passed": 2, "failed": 0, "unrun": 1})]:
            with self.subTest(field=field):
                changed = dict(original)
                changed[field] = replacement
                write(path, changed)
                with self.assertRaises(ValueError):
                    self.compare()
        write(path, original)

    def test_different_binary_is_rejected_even_with_matching_source_labels(self):
        path = self.jobs[1] / "engine/report.json"
        value = c.load(path)
        value["input_pins"]["binary"]["sha256"] = "f" * 64
        write(path, value)
        with self.assertRaisesRegex(ValueError, "tested engine build differs"):
            self.compare()

    def test_smoke_still_rejects_runtime_pin_drift(self):
        path = self.jobs[1] / "execution.json"
        value = c.load(path)
        value["source_pins"]["fixed-engine-source"] = "f" * 64
        write(path, value)
        with self.assertRaisesRegex(ValueError, "runtime pins differ"):
            self.compare()

    def test_smoke_still_requires_all_three_engine_passes(self):
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value["passed"] = False
        write(path, value)
        with self.assertRaisesRegex(ValueError, "passing nonpilot"):
            self.compare()


class ShortStateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.jobs = [self.root / name for name in ("full", "high", "low")]
        for path, tier, offset in zip(self.jobs, ("FULL", "Q8_0", "Q4_K_M"), (0, .01, 1)):
            tier_job(path, tier, offset, profile="smoke-6-short-state")

    def test_fresh_short_state_keeps_six_numeric_rows_and_no_new_gate(self):
        result = c.compare(*self.jobs)
        self.assertTrue(result["valid_evidence"])
        self.assertEqual(result["coverage_profile"], "smoke-6-short-state")
        self.assertEqual((result["cases_compared"], result["rows_per_pair"]), (6, 768))
        self.assertIsNone(result["quantization_acceptance_gate"])
        self.assertNotIn("passed", result)

    def test_old_smoke_state_scope_cannot_mix_with_short_state(self):
        old = self.root / "old-high"
        tier_job(old, "Q8_0", .01, profile="smoke-6")
        with self.assertRaisesRegex(ValueError, "coverage profile differs across tiers"):
            c.compare(self.jobs[0], old, self.jobs[2])

    def test_missing_short_state_check_is_rejected(self):
        path = self.jobs[1] / "independent-review.json"
        value = c.load(path)
        value["internal_checks_checked"] = 11
        write(path, value)
        with self.assertRaisesRegex(ValueError, "incomplete engine review scope"):
            c.compare(*self.jobs)


if __name__ == "__main__":
    unittest.main()
