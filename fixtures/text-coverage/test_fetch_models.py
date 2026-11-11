#!/usr/bin/env python3
"""Small offline tests; no network, model libraries, or large files."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_models as fetch


PAYLOAD = b"pinned model fixture bytes\x00\x01\x02"


def item():
    return {
        "filename": "model.gguf",
        "repo": "publisher/checkpoint-GGUF",
        "revision": "a" * 40,
        "size_bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }


def manifest():
    return {
        "schema_version": 1,
        "models": [{
            "id": "model",
            "artifacts": [{
                "id": "model-q8_0",
                "quantization": "Q8_0",
                "files": [item()],
            }],
        }],
    }


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(body))}


class ManifestTests(unittest.TestCase):
    def test_valid_manifest(self):
        self.assertEqual(list(fetch.validate_manifest(manifest())), ["model-q8_0"])

    def test_bad_file_fields(self):
        for field, value in [
            ("filename", "../outside.gguf"),
            ("filename", "/outside.gguf"),
            ("filename", "nested/../outside.gguf"),
            ("filename", "a//model.gguf"),
            ("filename", "a\\model.gguf"),
            ("revision", "main"),
            ("sha256", "b" * 63),
            ("size_bytes", True),
            ("size_bytes", -1),
            ("repo", "publisher/../../outside"),
        ]:
            with self.subTest(field=field, value=value):
                value_manifest = manifest()
                value_manifest["models"][0]["artifacts"][0]["files"][0][field] = value
                with self.assertRaises(ValueError):
                    fetch.validate_manifest(value_manifest)

    def test_f16_is_not_bf16(self):
        value = manifest()
        value["models"][0]["artifacts"][0]["quantization"] = "F16"
        with self.assertRaises(ValueError):
            fetch.validate_manifest(value)

    def test_full_tier_accepts_actual_f32(self):
        value = manifest()
        artifact = value["models"][0]["artifacts"][0]
        artifact.update(quantization="F32", tier="FULL")
        self.assertEqual(len(fetch.validate_manifest(value)), 1)

    def test_full_tier_cannot_hide_quantized_source(self):
        value = manifest()
        value["models"][0]["artifacts"][0]["tier"] = "FULL"
        with self.assertRaisesRegex(ValueError, "tier and actual dtype"):
            fetch.validate_manifest(value)

    def test_duplicate_artifact(self):
        value = manifest()
        value["models"].append(copy.deepcopy(value["models"][0]))
        with self.assertRaises(ValueError):
            fetch.validate_manifest(value)

    def test_duplicate_file(self):
        value = manifest()
        value["models"][0]["artifacts"][0]["files"].append(item())
        with self.assertRaises(ValueError):
            fetch.validate_manifest(value)

    def test_repository_manifest(self):
        value = json.loads(Path(__file__).with_name("models.json").read_text())
        self.assertGreaterEqual(len(fetch.validate_manifest(value)), 12)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = fetch.safe_destination(self.root, "model-q8_0", "model.gguf")
        self.part = self.path.with_name(self.path.name + ".part")

    def run_fetch(self, opener):
        return fetch.fetch_file(self.root, "model-q8_0", item(), 0, retries=0, opener=opener)

    def test_download(self):
        result = self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD))
        self.assertTrue(result["verified"])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())

    def test_resume_hashes_prefix_and_final(self):
        self.part.write_bytes(PAYLOAD[:7])

        def opener(request, **kwargs):
            self.assertEqual(request.headers["Range"], "bytes=7-")
            return Response(PAYLOAD[7:], 206, {
                "Content-Length": str(len(PAYLOAD) - 7),
                "Content-Range": f"bytes 7-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
            })

        result = self.run_fetch(opener)
        self.assertEqual(result["attempts"][0]["prefix_sha256"],
                         hashlib.sha256(PAYLOAD[:7]).hexdigest())
        self.assertEqual(result["sha256"], item()["sha256"])

    def test_ignored_range_retains_partial(self):
        self.part.write_bytes(PAYLOAD[:7])
        with self.assertRaisesRegex(ValueError, "ignored Range"):
            self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD))
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:7])
        self.assertFalse(self.path.exists())

    def test_bad_range_retains_partial(self):
        self.part.write_bytes(PAYLOAD[:7])
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD[7:], 206, {
                "Content-Range": "bytes 6-100/101",
            }))
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:7])

    def test_corrupt_complete_partial_cannot_be_accepted(self):
        self.part.write_bytes(b"x" * len(PAYLOAD))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_fetch(lambda *args, **kwargs: self.fail("no request needed"))
        self.assertTrue(self.part.exists())
        self.assertFalse(self.path.exists())

    def test_wrong_length_response(self):
        with self.assertRaisesRegex(ValueError, "response length"):
            self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD + b"x"))
        self.assertFalse(self.path.exists())

    def test_extra_unannounced_bytes(self):
        with self.assertRaisesRegex(ValueError, "exceeds pinned size"):
            self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD + b"x", headers={}))
        self.assertFalse(self.path.exists())

    def test_short_response_retains_prefix(self):
        with self.assertRaisesRegex(OSError, "short response"):
            self.run_fetch(lambda *args, **kwargs: Response(PAYLOAD[:7], headers={}))
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:7])
        self.assertFalse(self.path.exists())

    def test_existing_good_final_requires_full_hash(self):
        self.path.write_bytes(PAYLOAD)
        result = self.run_fetch(lambda *args, **kwargs: self.fail("must not download"))
        self.assertTrue(result["verified"])
        self.assertTrue(result["reused"])

    def test_existing_bad_final_is_never_overwritten(self):
        self.path.write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "existing final"):
            self.run_fetch(lambda *args, **kwargs: self.fail("must not download"))
        self.assertEqual(self.path.read_bytes(), b"old")

    def test_symlink_partial_is_refused(self):
        victim = self.root / "victim"
        victim.write_bytes(b"untouched")
        self.part.symlink_to(victim)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_fetch(lambda *args, **kwargs: self.fail("must not download"))
        self.assertEqual(victim.read_bytes(), b"untouched")

    def test_other_process_lock_is_preserved(self):
        lock = self.path.with_name(self.path.name + ".download.lock")
        lock.write_text("owned")
        with self.assertRaises(FileExistsError):
            self.run_fetch(lambda *args, **kwargs: self.fail("must not download"))
        self.assertEqual(lock.read_text(), "owned")

    def test_space_reserve(self):
        with patch.object(fetch.shutil, "disk_usage") as usage:
            usage.return_value.free = 100
            with self.assertRaisesRegex(ValueError, "insufficient disk"):
                fetch.check_space(self.root, 40, 61)
            fetch.check_space(self.root, 40, 60)


if __name__ == "__main__":
    unittest.main()
