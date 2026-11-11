import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cpu_reference_policy as policy


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.files = {}
        for name in ["libggml-cpu-apple_m4.so", "libggml-blas.so", "producer", "libggml-base.dylib"]:
            path = self.root / name
            path.write_bytes(name.encode())
            self.files[name] = {"path": str(path), "sha256": policy.digest(path), "size_bytes": path.stat().st_size}
        self.pins = {x["path"]: x["sha256"] for x in self.files.values()}
        self.selected = {"schema_version": 1, "id": policy.POLICY, "source_revision": policy.REVISION,
                         "source_patch_sha256": policy.PATCH_SHA256,
                         "plugin": self.files["libggml-cpu-apple_m4.so"], "blas_plugin": self.files["libggml-blas.so"]}
        self.runtime = {"policy": self.selected, "cpu_plugin_image_count": 1, "blas_plugin_image_count": 1,
                        "registry_count": 2, "gpu_backends_loaded": False,
                        "runtime_images": list(self.pins) + ["/usr/lib/libSystem.B.dylib"]}

    def test_exact_runtime(self):
        policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_explicit_activation_runtime(self):
        self.selected["id"] = policy.ACTIVATIONS_POLICY
        self.selected["activation_policy"] = policy.activation_notice()
        policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_changed_activation_patch_rejected(self):
        self.selected["id"] = policy.ACTIVATIONS_POLICY
        self.selected["activation_policy"] = policy.activation_notice()
        self.selected["activation_policy"]["patches"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "activation repair policy differs"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_activation_policy_cannot_be_labeled_v2(self):
        self.selected["activation_policy"] = policy.activation_notice()
        with self.assertRaisesRegex(ValueError, "v2 must not declare"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_missing_activation_policy_rejected(self):
        self.selected["id"] = policy.ACTIVATIONS_POLICY
        with self.assertRaisesRegex(ValueError, "activation repair policy differs"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_explicit_query_runtime(self):
        self.selected["id"] = policy.QUERY_POLICY
        self.selected["query_policy"] = policy.query_notice()
        policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_query_profile_cannot_be_labeled_v2_or_v3(self):
        for name in [policy.POLICY, policy.ACTIVATIONS_POLICY]:
            selected = copy.deepcopy(self.selected)
            selected["id"] = name
            selected["query_policy"] = policy.query_notice()
            with self.assertRaisesRegex(ValueError, "v2/v3 must not declare"):
                policy.check_query_policy(selected)

    def test_changed_or_missing_query_policy_rejected(self):
        self.selected["id"] = policy.QUERY_POLICY
        with self.assertRaisesRegex(ValueError, "F32 query repair policy differs"):
            policy.check_runtime(self.runtime, self.selected, self.pins)
        self.selected["query_policy"] = policy.query_notice()
        self.selected["query_policy"]["patches"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "F32 query repair policy differs"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_missing_binary_pin_rejected(self):
        del self.pins[self.files["producer"]["path"]]
        with self.assertRaisesRegex(ValueError, "unbound CPU runtime image"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_changed_bytes_rejected(self):
        Path(self.files["producer"]["path"]).write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "unbound CPU runtime image"):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_stock_plus_candidate_rejected(self):
        self.runtime["cpu_plugin_image_count"] = 2
        with self.assertRaises(ValueError):
            policy.check_runtime(self.runtime, self.selected, self.pins)

    def test_policy_mismatch_rejected(self):
        runtime = copy.deepcopy(self.runtime)
        runtime["policy"]["source_patch_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "policy differs"):
            policy.check_runtime(runtime, self.selected, self.pins)

    def test_cpu_missing_repair_rejected(self):
        with self.assertRaises(ValueError):
            policy.check_suite({}, "cpu", None, self.pins)

    def test_metal_unchanged(self):
        policy.check_suite({}, "metal", self.selected, self.pins)
        policy.check_case({}, "metal", self.selected)
        with self.assertRaises(ValueError):
            policy.check_case({"cpu_attention_policy": self.selected}, "metal", self.selected)

    def test_complete_suite_and_case(self):
        suite = {"cpu_attention_policy": self.selected, "cpu_runtime_before": self.runtime,
                 "cpu_runtime_after": self.runtime}
        policy.check_suite(suite, "cpu", self.selected, self.pins)
        policy.check_case({"cpu_attention_policy": self.selected}, "cpu", self.selected)
        del suite["cpu_runtime_after"]
        with self.assertRaises(KeyError):
            policy.check_suite(suite, "cpu", self.selected, self.pins)

    def test_receipt_revision_and_platform(self):
        receipt = {"passed": True, "policy": policy.POLICY, "source_revision": "0" * 40,
                   "platform": "darwin-arm64", "isa": "apple_m4"}
        path = self.root / "receipt.json"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "revision differs"):
            policy.build_policy(path, policy.REVISION)
        receipt["source_revision"] = policy.REVISION
        receipt["platform"] = "linux-aarch64"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "platform or ISA"):
            policy.build_policy(path, policy.REVISION)


if __name__ == "__main__":
    unittest.main()
