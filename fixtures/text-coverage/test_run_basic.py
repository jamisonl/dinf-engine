"""Basic-runner failure/selection checks; no model files, devices or inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_basic import (PROFILE, CHECKS, GIB, aggregate, checksum, execute, exit_status, run, select,
                       validate_report)


def registry():
    return {"profiles": {"basic-v1": copy.deepcopy(PROFILE)}, "text_artifacts": [
        {"id": "a", "backends": {"cpu": {"status": "passed"}, "metal": {"status": "not_run_memory"}}},
        {"id": "b", "backends": {"cpu": {"status": "unsupported"}, "metal": {"status": "passed"}}},
    ]}


class BasicRunnerTests(unittest.TestCase):
    def test_selection_does_not_drop_unrun_backends_or_accept_unknown_scope(self):
        jobs, required = select(registry(), None, "both")
        self.assertEqual(len(jobs), 4)
        self.assertEqual(required, {("a", "cpu"), ("a", "metal"), ("b", "metal")})
        for names in [["missing"], ["a", "a"]]:
            with self.assertRaises(ValueError):
                select(registry(), names, "both")
        bad = registry()
        del bad["text_artifacts"][0]["backends"]["metal"]
        with self.assertRaises(ValueError):
            select(bad, None, "both")
        bad = registry()
        bad["profiles"]["basic-v1"]["tokens_per_target_run"] = 1
        with self.assertRaises(ValueError):
            select(bad, None, "both")

    def test_partial_pending_missing_and_broader_gaps_never_become_complete(self):
        required = {("a", "cpu"), ("a", "metal")}
        record = {"id": "a", "backend": "cpu", "status": "passed"}
        partial = aggregate([record], required, {("a", "cpu")}, [])
        self.assertTrue(partial["selected_passed"])
        self.assertFalse(partial["complete"])
        self.assertFalse(aggregate([record], required, required, [])["selected_passed"])
        for status in ["missing_artifact", "not_run_memory", "not_run_no_cap", "failed"]:
            second = {"id": "a", "backend": "metal", "status": status}
            self.assertFalse(aggregate([record, second], required, required, [])["complete"])
        second = {"id": "a", "backend": "metal", "status": "passed"}
        whole = aggregate([record, second], required, required, [{"status": "pending", "model": "vision"}])
        self.assertTrue(whole["complete_text_registry"])
        self.assertFalse(whole["complete"])
        self.assertFalse(aggregate([], set(), set(), [])["selected_passed"])

    def test_full_list_exit_requires_closed_gaps_but_targeted_exit_uses_selected_scope(self):
        partial = {"selected_passed": True, "complete": False}
        self.assertEqual(exit_status(partial, None, "both"), 1)
        self.assertEqual(exit_status(partial, ["a"], "both"), 0)
        self.assertEqual(exit_status(partial, None, "cpu"), 0)
        self.assertEqual(exit_status(partial, None, "metal"), 0)
        self.assertEqual(exit_status({"selected_passed": False, "complete": False}, ["a"], "cpu"), 1)
        self.assertEqual(exit_status({"selected_passed": True, "complete": True}, None, "both"), 0)

    def test_success_text_cannot_hide_exit_failure_or_profile_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            entry = (Path(temp) / "model.gguf").resolve()
            report = {"passed": True, "profile": "basic-v1", "path": str(entry),
                      "architecture": "llama", "backend": "cpu", "max_tokens": 4,
                      "prefill_chunk_tokens": 8, "split_stages": 1, "draft": False,
                      "prompt_tokens": 16, "runs": [{"mode": "greedy", "tokens": [1, 2, 3, 4], "checks": CHECKS}]}
            event = {"exit_code": 0, "error": None, "all_processes_closed": True}
            model = {"architecture": "llama"}
            validate_report(report, event, model, "cpu", entry)
            for key, value in [("exit_code", 1), ("error", "timeout"), ("all_processes_closed", False)]:
                with self.assertRaises(ValueError):
                    validate_report(report, {**event, key: value}, model, "cpu", entry)
            for key, value in [("profile", "standard"), ("max_tokens", 3), ("prefill_chunk_tokens", 4),
                               ("backend", "metal"), ("draft", True), ("passed", 1)]:
                with self.assertRaises(ValueError):
                    validate_report({**report, key: value}, event, model, "cpu", entry)
            for change in [{"checks": CHECKS[:-1]}, {"tokens": [1, 2, 3]}, {"tokens": [1, 2, 3, True]}]:
                changed = copy.deepcopy(report)
                changed["runs"][0].update(change)
                with self.assertRaises(ValueError):
                    validate_report(changed, event, model, "cpu", entry)

    def test_checksum_cache_cannot_hide_changed_bytes_or_expected_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "weight"
            p.write_bytes(b"abc")
            expected = {"size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            cache = {}
            self.assertFalse(checksum(p, expected, cache)["checksum_cache_hit"])
            self.assertTrue(checksum(p, expected, cache)["checksum_cache_hit"])
            p.write_bytes(b"xyz")
            with self.assertRaises(ValueError):
                checksum(p, expected, cache)
            with self.assertRaises(ValueError):
                checksum(p, {**expected, "sha256": "0" * 64}, cache)

    def test_historical_pass_and_missing_cap_cannot_launch_or_count_as_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = registry()
            data["text_artifacts"] = data["text_artifacts"][:1]
            data["text_artifacts"][0]["backends"]["cpu"]["rss_cap_bytes"] = 128 * GIB
            manifest = root / "registry.json"
            manifest.write_text(json.dumps(data))
            paths = root / "paths.json"
            paths.write_text(json.dumps({"a": str(root / "a.gguf")}))
            args = SimpleNamespace(registry=manifest, paths=paths, model=None, backend="both",
                                   rss_caps=None, timeout=1, binary=Path(sys.executable),
                                   checksum_cache=None, output=root / "output")
            with patch("run_basic.available_memory", return_value=(100 * GIB, "test")), \
                    patch("run_basic.execute") as launch:
                summary = run(args)
            launch.assert_not_called()
            self.assertEqual([r["status"] for r in summary["records"]],
                             ["not_run_memory", "not_run_no_cap"])
            self.assertFalse(summary["selected_passed"])
            self.assertFalse(summary["complete"])

    def test_real_process_failure_and_timeout_keep_closed_failed_receipts(self):
        with tempfile.TemporaryDirectory() as temp, patch("run_basic.available_memory", return_value=(100 * GIB, "test")):
            root = Path(temp)
            failed = root / "failed"
            failed.mkdir()
            event = execute([sys.executable, "-c", "print('{}'); raise SystemExit(7)"], failed, {}, GIB, 2)
            self.assertEqual(event["exit_code"], 7)
            self.assertTrue(event["all_processes_closed"])
            timed = root / "timed"
            timed.mkdir()
            event = execute([sys.executable, "-c", "import time; time.sleep(10)"], timed, {}, GIB, 0.01)
            self.assertIn("timeout", event["error"])
            self.assertNotEqual(event["exit_code"], 0)
            self.assertTrue(event["all_processes_closed"])

    def test_swap_growth_and_final_stderr_cannot_pass_a_successful_exit(self):
        with tempfile.TemporaryDirectory() as temp, patch("run_basic.available_memory", return_value=(100 * GIB, "test")):
            root = Path(temp)
            swapped = root / "swapped"
            swapped.mkdir()
            with patch("run_basic.swapout_bytes", side_effect=[0] + [129 * 1024 ** 2] * 8):
                event = execute([sys.executable, "-c", "pass"], swapped, {}, GIB, 2)
            self.assertIn("swapout", event["error"])
            self.assertEqual(event["maximum_swapout_growth_bytes"], 129 * 1024 ** 2)
            self.assertTrue(event["all_processes_closed"])
            burst = root / "burst"
            burst.mkdir()
            event = execute([sys.executable, "-c", "import sys; sys.stderr.write('x' * (8*1024**2+1))"],
                            burst, {}, GIB, 2)
            self.assertIn("log limit", event["error"])
            self.assertTrue(event["all_processes_closed"])


if __name__ == "__main__":
    unittest.main()
