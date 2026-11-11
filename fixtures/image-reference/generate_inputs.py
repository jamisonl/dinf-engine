#!/usr/bin/env python3
"""Write the fixed Image2.1 component inputs; no model or numeric-library imports."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    result = {
        "schema_version": 1,
        "steps": 4,
        "scope": "deterministic synthetic conditioning and normalized latents; no prompt encoding claim",
        "image_shape": [1, 32, 32],
        "image_mask": {"text_slots": 32, "target_slots": 256},
        "generator": "integer sawtooth; values exactly representable in F32 and BF16",
    }
    for name, shape, multiplier, offset in [
        ("latents", [1, 1024, 64], 17, 23),
        ("conditioning", [1, 32, 4096], 29, 11),
    ]:
        count = 1
        for size in shape:
            count *= size
        raw = b"".join(
            struct.pack("<f", ((i * multiplier + offset) % 257 - 128) / 128)
            for i in range(count)
        )
        filename = name + ".f32"
        with (args.output / filename).open("xb") as stream:
            stream.write(raw)
        result[name] = {
            "file": filename,
            "shape": shape,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    with (args.output / "input.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
