#!/usr/bin/env python3
"""Build an Inkling 2 Mini GGUF from the official checkpoint's expert banks.

The official checkpoint (`thinkingmachines/Inkling-2-Mini`) stores its
routed experts as ModelOpt NVFP4 (layers 0-19) and MXFP8 (layers 20-23), and
everything else in BF16 or F32. This writes the merged `inkling2_loop`
layout dinf-engine reads, with every weight exact:

- Metadata and non-expert tensors come from ggml-org's BF16 GGUF (fetched by
  `inkling2_fetch_dense.py`), which holds the official BF16 and F32 weights
  in this layout.
- NVFP4 banks become GGML NVFP4 (type 40) blocks: 64 weights in 36 bytes,
  four E4M3 scales then 32 bytes where byte j of a 16-weight group holds
  weight j (low nibble) and j + 8 (high). The scale bytes are copied and the
  nibbles reordered, so the values are unchanged. Each expert's outer scale,
  `amax / outer_scale_denominator`, becomes `ffn_*_exps.scale` (F32).
- MXFP8 banks keep their E4M3 bytes as `F8_E4M3` (type 43) tensors, with the
  UE8M0 scale of every 32 weights in `ffn_*_exps.mx_scale` (I8, type 24).
- The official gate/up bank interleaves gate and up rows; both are
  de-interleaved to `[gate; up]`, as llama.cpp's converter does.

Usage: inkling2_official_gguf.py OFFICIAL_DIR DENSE_DIR OUTPUT.gguf

OFFICIAL_DIR holds `model-00001-of-00001.safetensors` and
`hf_quant_config.json`; DENSE_DIR is the output of inkling2_fetch_dense.py.
Needs numpy.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
NVFP4, F8_E4M3, I8, F32 = 40, 43, 24, 0


def read_header(path: Path):
    """The GGUF header's raw key/value bytes, tensor infos and alignment."""
    data = path.read_bytes()
    pos = 4

    def read(fmt):
        nonlocal pos
        value = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return value[0]

    def string():
        nonlocal pos
        n = read("Q")
        s = data[pos : pos + n].decode()
        pos += n
        return s

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(kind):
        nonlocal pos
        if kind == 8:
            length = read("Q")  # read first: `pos += read(...)` would load the old pos
            pos += length
        elif kind == 9:
            element = read("I")
            for _ in range(read("Q")):
                skip(element)
        else:
            pos += sizes[kind]

    version = read("I")
    assert data[:4] == GGUF_MAGIC and version == 3, "expected a GGUF v3 header"
    tensors, kvs = read("Q"), read("Q")
    kv_start = pos
    alignment = 32
    for _ in range(kvs):
        key = string()
        kind = read("I")
        if key == "general.alignment":
            alignment = read("I")
        else:
            skip(kind)
    kv_bytes = data[kv_start:pos]
    infos = []
    for _ in range(tensors):
        name = string()
        dims = [read("Q") for _ in range(read("I"))]
        infos.append((name, dims, read("I")))
        read("Q")
    return kvs, kv_bytes, infos, alignment


class Safetensors:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name: str):
        info = self.header[name]
        start, end = info["data_offsets"]
        return self.map[self.base + start : self.base + end], info["shape"], info["dtype"]

    def array(self, name: str, dtype):
        raw, shape, _ = self.raw(name)
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)


def deinterleave(rows: np.ndarray) -> np.ndarray:
    """Interleaved gate/up rows `[g0, u0, g1, u1, ...]` as `[gate; up]`."""
    return np.concatenate([rows[0::2], rows[1::2]])


def nvfp4_blocks(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """One expert's ModelOpt NVFP4 rows as GGML NVFP4 blocks.

    `packed` is `[rows, cols / 2]` with weight 2i in the low nibble of byte i;
    `scales` is `[rows, cols / 16]` E4M3. Returns `[rows, cols / 64 * 36]`.
    """
    rows, half = packed.shape
    cols = half * 2
    bad = (scales & 0x7F) == 0x7F
    negative = (scales & 0x80) != 0
    if bad.any() or (negative & (scales != 0x80)).any():
        raise ValueError("NVFP4 scale bytes must be finite and nonnegative")
    values = np.empty((rows, cols), dtype=np.uint8)
    values[:, 0::2] = packed & 15
    values[:, 1::2] = packed >> 4
    groups = values.reshape(rows, cols // 64, 4, 16)
    quants = (groups[..., :8] | (groups[..., 8:] << 4)).reshape(rows, cols // 64, 32)
    blocks = np.concatenate([scales.reshape(rows, cols // 64, 4), quants], axis=2)
    return blocks.reshape(rows, cols // 64 * 36)


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    official, dense_dir, target = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if target.exists():
        sys.exit(f"{target} exists; choose a new output path")
    weights = Safetensors(official / "model-00001-of-00001.safetensors")
    quant = json.loads((official / "hf_quant_config.json").read_text())
    scheme = {}
    for group in quant["model"]["parameter_groups"]:
        for name in group["tensors"]:
            scheme[name] = group["weight_transform"]
    kvs, kv_bytes, infos, alignment = read_header(dense_dir / "header.gguf")
    index = json.loads((dense_dir / "index.json").read_text())
    dense = np.memmap(dense_dir / "dense.bin", dtype=np.uint8, mode="r")

    # (name, GGUF dims, GGUF type, producer of the bytes)
    outputs = []
    for name, dims, kind in infos:
        if not name.endswith("_exps.weight"):
            entry = index[name]
            outputs.append(
                (name, dims, kind, lambda e=entry: dense[e["offset"] : e["offset"] + e["length"]])
            )
            continue
        layer = int(name.split(".")[1])
        source = f"model.llm.layers.{layer}.mlp.experts." + (
            "w13_weight" if "gate_up" in name else "w2_weight"
        )
        gate_up = "gate_up" in name
        base = name.removesuffix(".weight")
        transform = scheme[source]
        experts = dims[2]
        if transform == "nvfp4_nearest":

            def nvfp4(source=source, gate_up=gate_up, experts=experts):
                packed = weights.array(source, np.uint8)
                scales = weights.array(source + "_scale", np.uint8)
                for expert in range(experts):
                    p, s = packed[expert], scales[expert]
                    if gate_up:
                        p, s = deinterleave(p), deinterleave(s)
                    yield nvfp4_blocks(p, s)

            outputs.append((name, dims, NVFP4, nvfp4))
            amax = weights.array(source + "_amax", np.float32)
            denominator = weights.array(source + "_outer_scale_denominator", np.float32)
            if not (np.isfinite(amax).all() and (amax >= 0).all()):
                raise ValueError(f"{source}_amax must be finite and nonnegative")
            if not (np.isfinite(denominator).all() and (denominator > 0).all()):
                raise ValueError(f"{source}_outer_scale_denominator must be positive")
            outer = (amax / denominator[0]).astype(np.float32)
            outputs.append((base + ".scale", [experts], F32, lambda o=outer: o))
        elif transform == "mxfp8_nearest":

            def fp8(source=source, gate_up=gate_up, experts=experts):
                values = weights.array(source, np.uint8)
                for expert in range(experts):
                    v = values[expert]
                    if ((v & 0x7F) == 0x7F).any():
                        raise ValueError(f"{source}: E4M3 NaN")
                    yield deinterleave(v) if gate_up else v

            def e8m0(source=source, gate_up=gate_up, experts=experts):
                scales = weights.array(source + "_scale", np.uint8)
                if (scales == 255).any():
                    raise ValueError(f"{source}_scale: UE8M0 NaN")
                for expert in range(experts):
                    s = scales[expert]
                    yield deinterleave(s) if gate_up else s

            outputs.append((name, dims, F8_E4M3, fp8))
            outputs.append((base + ".mx_scale", [dims[0] // 32, dims[1], experts], I8, e8m0))
        else:
            raise ValueError(f"{source}: unsupported scheme {transform}")

    sizes = {F32: 4, 1: 2, 30: 2, F8_E4M3: 1, I8: 1}

    def byte_len(dims, kind):
        elements = 1
        for d in dims:
            elements *= d
        return elements // 64 * 36 if kind == NVFP4 else elements * sizes[kind]

    with open(target, "wb") as out:
        out.write(GGUF_MAGIC + struct.pack("<IQQ", 3, len(outputs), kvs))
        out.write(kv_bytes)
        offset = 0
        for name, dims, kind, _ in outputs:
            encoded = name.encode()
            out.write(struct.pack("<Q", len(encoded)) + encoded)
            out.write(struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims))
            out.write(struct.pack("<IQ", kind, offset))
            offset += (byte_len(dims, kind) + alignment - 1) // alignment * alignment
        out.write(b"\0" * (-out.tell() % alignment))
        for name, dims, kind, produce in outputs:
            written = 0
            produced = produce()
            for part in produced if not isinstance(produced, np.ndarray) else [produced]:
                data = np.ascontiguousarray(part).tobytes()
                out.write(data)
                written += len(data)
            if written != byte_len(dims, kind):
                raise ValueError(f"{name}: wrote {written} bytes, expected {byte_len(dims, kind)}")
            out.write(b"\0" * (-written % alignment))
            print(f"{name} {dims} type {kind}", flush=True)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
