#!/usr/bin/env python3
"""Fetch the non-expert tensors of ggml-org's Inkling 2 Mini BF16 GGUF.

Those tensors are the official checkpoint's BF16 weights (and the F32 ones
llama.cpp's converter keeps in F32), already in the merged `inkling2_loop`
layout. `inkling2_official_gguf.py` combines them with the official NVFP4
and MXFP8 expert banks. Only the header and the non-expert byte ranges are
downloaded (about 4 GB of the 52 GB file).

Usage: inkling2_fetch_dense.py OUTPUT_DIR

OUTPUT_DIR receives `header.gguf` (the GGUF header) and `dense.bin` (each
non-expert tensor's bytes, concatenated), with `index.json` giving each
tensor's offset and length in `dense.bin`.
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
    "db61c5b492a6a034d986328d28274af0a439aa6c/Inkling-2-Mini-BF16.gguf"
)
HEADER_FETCH = 16 << 20
# (elements per block, bytes per block) for the types this file uses.
LAYOUT = {0: (1, 4), 1: (1, 2), 30: (1, 2)}


def fetch(start: int, end: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    if len(data) != end - start:
        raise RuntimeError(f"short read {len(data)} of {end - start} at {start}")
    return data


def parse(header: bytes):
    pos = 4

    def read(fmt):
        nonlocal pos
        value = struct.unpack_from("<" + fmt, header, pos)
        pos += struct.calcsize("<" + fmt)
        return value[0]

    def string():
        nonlocal pos
        n = read("Q")
        s = header[pos : pos + n].decode()
        pos += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            element = read("I")
            return [value(element) for _ in range(read("Q"))]
        return read(scalar[kind])

    read("I")
    tensors, kvs = read("Q"), read("Q")
    alignment = 32
    for _ in range(kvs):
        key = string()
        v = value(read("I"))
        if key == "general.alignment":
            alignment = v
    infos = []
    for _ in range(tensors):
        name = string()
        dims = [read("Q") for _ in range(read("I"))]
        kind = read("I")
        offset = read("Q")
        infos.append((name, dims, kind, offset))
    data_start = (pos + alignment - 1) // alignment * alignment
    return infos, data_start


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    header = fetch(0, HEADER_FETCH)
    infos, data_start = parse(header)
    (out / "header.gguf").write_bytes(header[:data_start])
    wanted = []
    for name, dims, kind, offset in infos:
        if name.endswith("_exps.weight"):
            continue
        block, size = LAYOUT[kind]
        elements = 1
        for d in dims:
            elements *= d
        wanted.append((data_start + offset, elements // block * size, name, dims, kind))
    wanted.sort()
    # Merge neighbouring ranges so each request covers a run of small tensors.
    runs = []
    for start, length, *rest in wanted:
        if runs and start - runs[-1][1] < (1 << 20):
            runs[-1][1] = max(runs[-1][1], start + length)
            runs[-1][2].append((start, length, *rest))
        else:
            runs.append([start, start + length, [(start, length, *rest)]])
    index = {}
    written = 0
    with open(out / "dense.bin", "wb") as dense:
        for run_start, run_end, members in runs:
            data = fetch(run_start, run_end)
            for start, length, name, dims, kind in members:
                dense.write(data[start - run_start : start - run_start + length])
                index[name] = {"offset": written, "length": length, "dims": dims, "type": kind}
                written += length
            print(f"{written / 1e9:6.2f} GB", flush=True)
    (out / "index.json").write_text(json.dumps(index))
    print(f"{len(index)} tensors, {written} bytes")


if __name__ == "__main__":
    main()
