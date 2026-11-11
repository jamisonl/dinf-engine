#!/usr/bin/env python3
"""Tiny synthetic GGUF copy tests; no inference or external files."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import prepare_bf16 as prepare


def typed(value):
    if isinstance(value, str):
        return struct.pack("<I", 8) + prepare.string_bytes(value)
    if isinstance(value, list):
        return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(
            prepare.string_bytes(item) for item in value)
    return struct.pack("<II", 4, value)


def make_gguf(path, source=True, changes=None, extra_tensor=False,
              source_dtype=30, include_output=None):
    metadata = {
        "general.architecture": "qwen2",
        "general.file_type": 32 if source else 7,
        "qwen2.embedding_length": 32,
        "qwen2.context_length": 4096,
        "tokenizer.ggml.tokens": ["one", "two", "three", "four"],
        "tokenizer.chat_template": "old" if source else "new template " * 20,
    }
    metadata.update(changes or {})
    tensors = [
        ("output_norm.weight", [32], 0, bytes(range(128))),
        ("token_embd.weight", [32, 4], source_dtype if source else 8,
         bytes(range(256)) if source else b"e" * 136),
    ]
    if include_output is None:
        include_output = not source
    if include_output:
        tensors.append(("output.weight", [32, 4], 8, b"h" * 136))
    if extra_tensor:
        tensors.append(("unexpected.weight", [32], 0, b"u" * 128))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        header.extend(prepare.string_bytes(key) + typed(value))
    payload = bytearray()
    for name, shape, dtype, data in tensors:
        payload.extend(b"\x00" * ((-len(payload)) % 32))
        offset = len(payload)
        header.extend(prepare.string_bytes(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", dtype, offset))
        payload.extend(data)
    header.extend(b"\x00" * ((-len(header)) % 32))
    path.write_bytes(header + payload)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.gguf"
        self.reference = self.root / "reference.gguf"
        self.config = self.root / "config.json"
        self.output = self.root / "aligned.gguf"
        make_gguf(self.source)
        make_gguf(self.reference, source=False)
        self.config.write_text(json.dumps({"model_type": "qwen2", "tie_word_embeddings": True}))

    def run_prepare(self, source_sha=None):
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        return prepare.prepare(self.source, source_sha or sha(self.source),
                               self.reference, sha(self.reference), self.config,
                               sha(self.config), self.output)

    def test_only_template_changes_and_all_payload_bytes_match(self):
        original = self.source.read_bytes()
        result = self.run_prepare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["changed_metadata_keys"], ["tokenizer.chat_template"])
        self.assertEqual(result["logical_tensor_aliases"], {"output.weight": "token_embd.weight"})
        self.assertFalse(result["physical_output_tensor_added"])
        self.assertEqual(self.source.read_bytes(), original)
        old = prepare.read_header(self.source)
        new = prepare.read_header(self.output)
        self.assertNotEqual(old["data_offset"], new["data_offset"])
        self.assertEqual(original[old["data_offset"]:],
                         self.output.read_bytes()[new["data_offset"]:])
        self.assertEqual(old["tensors"], new["tensors"])

    def test_wrong_source_hash(self):
        with self.assertRaisesRegex(ValueError, "source SHA"):
            self.run_prepare("0" * 64)
        self.assertFalse(self.output.exists())

    def test_f16_source_rejected(self):
        make_gguf(self.source, source_dtype=1)
        with self.assertRaisesRegex(ValueError, "BF16 and F32"):
            self.run_prepare()

    def test_quantized_source_rejected(self):
        make_gguf(self.source, source_dtype=8)
        with self.assertRaisesRegex(ValueError, "BF16 and F32"):
            self.run_prepare()

    def test_untied_config_rejected(self):
        self.config.write_text(json.dumps({"model_type": "qwen2", "tie_word_embeddings": False}))
        with self.assertRaisesRegex(ValueError, "not tied"):
            self.run_prepare()

    def test_other_architecture_rejected(self):
        self.config.write_text(json.dumps({"model_type": "llama", "tie_word_embeddings": True}))
        with self.assertRaisesRegex(ValueError, "source architecture"):
            self.run_prepare()

    def test_full_token_array_difference_rejected(self):
        make_gguf(self.source, changes={"tokenizer.ggml.tokens": ["one", "two", "THREE", "four"]})
        with self.assertRaisesRegex(ValueError, "tokenizer.ggml.tokens"):
            self.run_prepare()

    def test_context_difference_rejected(self):
        make_gguf(self.source, changes={"qwen2.context_length": 8192})
        with self.assertRaisesRegex(ValueError, "qwen2.context_length"):
            self.run_prepare()

    def test_additional_tokenizer_key_rejected(self):
        make_gguf(self.source, changes={"tokenizer.ggml.eos_token_id": 2})
        with self.assertRaisesRegex(ValueError, "key set"):
            self.run_prepare()

    def test_extra_tensor_rejected(self):
        make_gguf(self.source, extra_tensor=True)
        with self.assertRaisesRegex(ValueError, "tensor name difference"):
            self.run_prepare()

    def test_reference_without_physical_head_rejected(self):
        make_gguf(self.reference, source=False, include_output=False)
        with self.assertRaisesRegex(ValueError, "tensor name difference"):
            self.run_prepare()

    def test_existing_output_is_not_overwritten(self):
        self.output.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_prepare()
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_existing_partial_is_not_overwritten(self):
        partial = self.output.with_name(self.output.name + ".part")
        partial.write_bytes(b"partial")
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.assertEqual(partial.read_bytes(), b"partial")

    def test_truncated_header_rejected(self):
        self.source.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
