#!/usr/bin/env python3
"""Align a BF16 GGUF chat template without changing any tensor payload bytes.

This narrow helper supports the coverage matrix's Qwen2.5-Coder tied-head case.
It checks pinned source/reference/config hashes, exact token and model settings,
and the sole missing physical output.weight explained by the source's tied head.
It writes a new file and a receipt; it never edits the published source.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from fetch_models import check_space, file_identity, hash_file, require


CHUNK_BYTES = 8 * 1024 * 1024
HEADER_LIMIT = 64 * 1024 * 1024
DTYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 30: (1, 2)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class HeaderReader:
    def __init__(self, stream):
        self.stream = stream
        self.raw = bytearray()

    def read(self, size):
        require(0 <= size <= HEADER_LIMIT - len(self.raw), "GGUF header exceeds bound")
        value = self.stream.read(size)
        require(len(value) == size, "truncated GGUF header")
        self.raw.extend(value)
        return value

    def number(self, code):
        return struct.unpack("<" + code, self.read(struct.calcsize(code)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind, in_array=False):
        if kind in SCALARS:
            return self.number(SCALARS[kind])
        if kind == 8:
            return self.string()
        require(kind == 9 and not in_array, "unsupported metadata type")
        element, count = self.number("I"), self.number("Q")
        require(count <= 2_000_000, "metadata array exceeds bound")
        for _ in range(count):
            self.value(element, True)
        return {"array_type": element, "count": count}


def string_bytes(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def read_header(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        reader = HeaderReader(stream)
        require(reader.read(4) == b"GGUF" and reader.number("I") == 3,
                "only GGUF v3 is supported")
        tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
        require(0 < tensor_count <= 100_000 and metadata_count <= 100_000,
                "GGUF count exceeds bound")
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            require(key and "\x00" not in key and key not in metadata,
                    "duplicate or invalid metadata key")
            start = len(reader.raw)
            kind = reader.number("I")
            value = reader.value(kind)
            raw = bytes(reader.raw[start:])
            metadata[key] = {"value": value, "raw": raw,
                             "sha256": hashlib.sha256(raw).hexdigest()}
        tensor_table_start = len(reader.raw)
        tensors = {}
        for _ in range(tensor_count):
            name, rank = reader.string(), reader.number("I")
            require(name and "\x00" not in name and name not in tensors
                    and 1 <= rank <= 4, "invalid tensor name/rank")
            shape = [reader.number("Q") for _ in range(rank)]
            kind, offset = reader.number("I"), reader.number("Q")
            require(kind in DTYPES, "unsupported tensor dtype")
            block, block_bytes = DTYPES[kind]
            require(all(0 < value <= 2 ** 32 for value in shape)
                    and shape[0] % block == 0, "invalid tensor dimensions")
            tensors[name] = {"shape": shape, "dtype": kind, "offset": offset,
                             "size_bytes": math.prod(shape) // block * block_bytes}
        alignment = metadata.get("general.alignment", {"value": 32})["value"]
        require(type(alignment) is int and 1 <= alignment <= 65536
                and alignment & (alignment - 1) == 0, "invalid alignment")
        data_offset = (len(reader.raw) + alignment - 1) // alignment * alignment
        ranges = []
        for tensor in tensors.values():
            start, length = tensor["offset"], tensor["size_bytes"]
            require(start % alignment == 0 and data_offset + start + length <= size,
                    "tensor exceeds file or is unaligned")
            ranges.append((start, start + length))
        ranges.sort()
        require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
                "overlapping tensor ranges")
    return {"metadata": metadata, "tensors": tensors, "alignment": alignment,
            "data_offset": data_offset, "tensor_count": tensor_count,
            "tensor_table": bytes(reader.raw[tensor_table_start:]), "size_bytes": size}


def runtime_keys(header):
    return {key for key in header["metadata"]
            if not key.startswith(("general.", "quantize.", "split."))}


def check_contract(source, reference, config):
    require(config.get("tie_word_embeddings") is True, "source config is not tied")
    require(config.get("model_type") == "qwen2", "unexpected source architecture")
    for header in (source, reference):
        require(header["metadata"]["general.architecture"]["value"] == "qwen2",
                "unexpected GGUF architecture")
        require(not any(key.startswith("split.") for key in header["metadata"]),
                "this helper accepts unsplit GGUFs only")
    require(all(tensor["dtype"] in (0, 30) for tensor in source["tensors"].values()),
            "source must contain only BF16 and F32 tensors")
    require(any(tensor["dtype"] == 30 for tensor in source["tensors"].values()),
            "source has no BF16 weights")
    require(runtime_keys(source) == runtime_keys(reference), "model/tokenizer key set differs")
    for key in runtime_keys(source) - {"tokenizer.chat_template"}:
        require(source["metadata"][key]["raw"] == reference["metadata"][key]["raw"],
                f"model/tokenizer setting differs: {key}")
    require(isinstance(reference["metadata"]["tokenizer.chat_template"]["value"], str),
            "reference template is not a string")
    original, quantized = source["tensors"], reference["tensors"]
    require(set(quantized) - set(original) == {"output.weight"}
            and not set(original) - set(quantized), "unexpected tensor name difference")
    require("token_embd.weight" in original and "output.weight" not in original,
            "expected one implicit tied output")
    for name, tensor in original.items():
        require(tensor["shape"] == quantized[name]["shape"], f"shape differs: {name}")
    require(quantized["output.weight"]["shape"] == original["token_embd.weight"]["shape"],
            "tied output shape differs")


def encode_header(header, template):
    output = bytearray(b"GGUF" + struct.pack("<IQQ", 3, header["tensor_count"],
                                            len(header["metadata"])))
    for key, entry in header["metadata"].items():
        output.extend(string_bytes(key))
        output.extend(struct.pack("<I", 8) + string_bytes(template)
                      if key == "tokenizer.chat_template" else entry["raw"])
    output.extend(header["tensor_table"])
    padding = (-len(output)) % header["alignment"]
    output.extend(b"\x00" * padding)
    return bytes(output)


def range_digest(path, offset, size):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            data = stream.read(min(remaining, CHUNK_BYTES))
            require(data, "short payload read")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def prepare(source_path, source_sha, reference_path, reference_sha, config_path,
            config_sha, output, reserve_bytes=0):
    require(not output.exists() and not output.is_symlink(), "output already exists")
    for digest in (source_sha, reference_sha, config_sha):
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "invalid expected SHA-256")
    source_digest, source_id = hash_file(source_path)
    reference_digest, reference_id = hash_file(reference_path)
    config_digest, config_id = hash_file(config_path)
    require(source_digest == source_sha, "source SHA-256 mismatch")
    require(reference_digest == reference_sha, "reference SHA-256 mismatch")
    require(config_digest == config_sha, "config SHA-256 mismatch")
    source, reference = read_header(source_path), read_header(reference_path)
    config = json.loads(config_path.read_bytes())
    check_contract(source, reference, config)
    header = encode_header(source, reference["metadata"]["tokenizer.chat_template"]["value"])
    payload_bytes = source["size_bytes"] - source["data_offset"]
    output.parent.mkdir(parents=True, exist_ok=True)
    check_space(output.parent, len(header) + payload_bytes, reserve_bytes)
    partial = output.with_name(output.name + ".part")
    copied = hashlib.sha256()
    with partial.open("xb") as destination, source_path.open("rb") as original:
        require(file_identity(source_path) == source_id, "source changed before copy")
        opened = os.fstat(original.fileno())
        require(
            (opened.st_dev, opened.st_ino, opened.st_size,
             opened.st_mtime_ns, opened.st_ctime_ns)
            == tuple(source_id[key] for key in (
                "device", "inode", "size_bytes", "mtime_ns", "ctime_ns")),
            "source changed at copy open",
        )
        original.seek(source["data_offset"])
        destination.write(header)
        remaining = payload_bytes
        while remaining:
            data = original.read(min(remaining, CHUNK_BYTES))
            require(data, "short source payload")
            check_space(output.parent, len(data), reserve_bytes)
            destination.write(data)
            copied.update(data)
            remaining -= len(data)
        destination.flush()
        os.fsync(destination.fileno())
    require(file_identity(source_path) == source_id, "source changed during copy")
    require(file_identity(reference_path) == reference_id, "reference changed during copy")
    require(file_identity(config_path) == config_id, "config changed during copy")
    derived = read_header(partial)
    require(derived["tensors"] == source["tensors"], "tensor table changed")
    changed = [key for key in source["metadata"]
               if source["metadata"][key]["raw"] != derived["metadata"][key]["raw"]]
    require(changed in ([], ["tokenizer.chat_template"]), "unexpected metadata change")
    check_contract(derived, reference, config)
    require(derived["metadata"]["tokenizer.chat_template"]["raw"]
            == reference["metadata"]["tokenizer.chat_template"]["raw"],
            "template alignment failed")
    copied_sha = copied.hexdigest()
    require(range_digest(partial, derived["data_offset"], payload_bytes) == copied_sha,
            "copied payload differs")
    output_sha, _ = hash_file(partial)
    os.link(partial, output)
    partial.unlink()
    return {
        "passed": True, "operation": "replace_chat_template_only",
        "source": {"path": str(source_path), "sha256": source_sha, "identity": source_id},
        "reference": {"path": str(reference_path), "sha256": reference_sha,
                      "identity": reference_id},
        "source_config": {"path": str(config_path), "sha256": config_sha,
                          "identity": config_id, "tie_word_embeddings": True},
        "output": {"path": str(output), "sha256": output_sha, "identity": file_identity(output)},
        "changed_metadata_keys": changed, "tensor_count": source["tensor_count"],
        "tensor_table_unchanged": True, "tensor_payload_bytes": payload_bytes,
        "source_and_output_payload_sha256": copied_sha,
        "all_tensor_payload_bytes_unchanged": True,
        "logical_tensor_aliases": {"output.weight": "token_embd.weight"},
        "physical_output_tensor_added": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "config", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "reference-sha256", "config-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    args = parser.parse_args()
    require(args.reserve_bytes >= 0, "negative disk reserve")
    require(not args.receipt.exists(), "receipt already exists")
    receipt = {"passed": False, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "fetch_helper_sha256": hashlib.sha256(Path(__file__).with_name("fetch_models.py").read_bytes()).hexdigest()}
    try:
        receipt.update(prepare(args.source, args.source_sha256, args.reference,
                               args.reference_sha256, args.config, args.config_sha256,
                               args.output, args.reserve_bytes))
    except Exception as error:
        receipt.update(passed=False, error=str(error))
    receipt["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": receipt["passed"], "receipt": str(args.receipt)}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
