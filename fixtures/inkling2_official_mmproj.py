#!/usr/bin/env python3
"""Build the Inkling 2 Mini vision projector GGUF from the official checkpoint.

Reads the `model.visual.*` tensors (BF16) of `thinkingmachines/Inkling-2-Mini`
and its `config.json` / `processor_config.json`, and writes a `clip` GGUF with
projector type `inkling2v`, the layout the engine's vision loader reads:

- matrices stay BF16, byte for byte; 1-D tensors (norms, biases, layer
  scales, the class embedding) are widened to F32, which is exact;
- the linear patch embedding `[1024, 20 * 20 * 3]` (HWC patches) is laid out
  as a conv2d weight `[1024, 3, 20, 20]`;
- the two projector layers are `mm.1` and `mm.2`.

Usage: inkling2_official_mmproj.py OFFICIAL_DIR OUTPUT.gguf

OFFICIAL_DIR holds `model-00001-of-00001.safetensors`, `config.json` and
`processor_config.json`. Needs numpy. The output is deterministic.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

F32, BF16 = 0, 30
U32, F32V, BOOL, STRING, ARRAY = 4, 6, 7, 8, 9
ALIGNMENT = 32


class Safetensors:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def bf16(self, name: str) -> np.ndarray:
        """BF16 values as uint16 bit patterns of their shape."""
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name} is {info['dtype']}, expected BF16")
        start, end = info["data_offsets"]
        raw = self.map[self.base + start : self.base + end]
        return np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(info["shape"])


def widen(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def kv(key: str, kind: int, value) -> bytes:
    out = string(key) + struct.pack("<I", kind)
    if kind == U32:
        return out + struct.pack("<I", value)
    if kind == F32V:
        return out + struct.pack("<f", value)
    if kind == BOOL:
        return out + struct.pack("<?", value)
    if kind == STRING:
        return out + string(value)
    if kind == ARRAY:
        element, values = value
        out += struct.pack("<IQ", element, len(values))
        fmt = {F32V: "<f", U32: "<I"}[element]
        return out + b"".join(struct.pack(fmt, v) for v in values)
    raise ValueError(kind)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    official, target = Path(sys.argv[1]), Path(sys.argv[2])
    if target.exists():
        sys.exit(f"{target} exists; choose a new output path")
    config = json.loads((official / "config.json").read_text())
    vision = config["vision_config"]
    processor = json.loads((official / "processor_config.json").read_text())["image_processor"]
    weights = Safetensors(official / "model-00001-of-00001.safetensors")

    hidden = vision["hidden_size"]
    layers = vision["num_hidden_layers"]
    patch = vision["native_patch_size"]
    merge = vision["adapter_merge_size"]
    ffn = int(hidden * vision["mlp_ratio"])
    grid = vision["posemb_grid_size"]
    if vision.get("projector_bias"):
        raise ValueError("expected a projector without biases")

    # (GGUF name, row-major array, GGUF type); 1-D tensors as F32.
    tensors = []

    def add(gguf_name: str, source: str, transform=None):
        bits = weights.bf16(source)
        if transform is not None:
            bits = transform(bits)
        if bits.ndim == 1:
            tensors.append((gguf_name, widen(bits), F32))
        else:
            tensors.append((gguf_name, np.ascontiguousarray(bits), BF16))

    v = "model.visual.backbone."
    add("v.patch_embd.weight", v + "embeddings.patch_embedding.weight",
        lambda w: w.reshape(w.shape[0], patch, patch, 3).transpose(0, 3, 1, 2))
    add("v.position_embd.weight", v + "embeddings.position_embedding")
    add("v.class_embd", v + "embeddings.class_embedding")
    add("v.class_position_embd", v + "embeddings.cls_position_embedding")
    add("v.pre_ln.weight", v + "ln_pre.weight")
    add("v.pre_ln.bias", v + "ln_pre.bias")
    names = {
        "ln_1": "ln1", "ln_2": "ln2",
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_out",
        "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down",
    }
    for layer in range(layers):
        source = f"{v}layers.{layer}."
        for hf, gg in names.items():
            add(f"v.blk.{layer}.{gg}.weight", f"{source}{hf}.weight")
            add(f"v.blk.{layer}.{gg}.bias", f"{source}{hf}.bias")
        add(f"v.blk.{layer}.ls1.weight", f"{source}ls_1.gamma")
        add(f"v.blk.{layer}.ls2.weight", f"{source}ls_2.gamma")
    add("mm.1.weight", "model.visual.proj_in.weight")
    add("mm.2.weight", "model.visual.proj_out.weight")

    expected = {k for k in weights.header if k.startswith("model.visual.")}
    if len(tensors) != len(expected):
        raise ValueError(f"converted {len(tensors)} of {len(expected)} vision tensors")

    metadata = [
        kv("general.architecture", STRING, "clip"),
        kv("general.type", STRING, "mmproj"),
        kv("general.name", STRING, "Inkling 2 Mini vision projector"),
        kv("general.license", STRING, "apache-2.0"),
        kv("clip.has_vision_encoder", BOOL, True),
        kv("clip.vision.projector_type", STRING, "inkling2v"),
        kv("clip.vision.image_size", U32, grid * patch),
        kv("clip.vision.patch_size", U32, patch),
        kv("clip.vision.embedding_length", U32, hidden),
        kv("clip.vision.feed_forward_length", U32, ffn),
        kv("clip.vision.projection_dim", U32, vision["text_hidden_size"]),
        kv("clip.vision.block_count", U32, layers),
        kv("clip.vision.attention.head_count", U32, vision["num_attention_heads"]),
        kv("clip.vision.attention.layer_norm_epsilon", F32V, vision["layer_norm_eps"]),
        kv("clip.vision.spatial_merge_size", U32, merge),
        kv("clip.vision.image_mean", ARRAY, (F32V, processor["image_mean"])),
        kv("clip.vision.image_std", ARRAY, (F32V, processor["image_std"])),
    ]

    with open(target, "wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
        for entry in metadata:
            out.write(entry)
        offset = 0
        for name, array, kind in tensors:
            dims = list(reversed(array.shape))
            out.write(string(name) + struct.pack("<I", len(dims)))
            out.write(b"".join(struct.pack("<Q", d) for d in dims))
            out.write(struct.pack("<IQ", kind, offset))
            offset += (array.nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        out.write(b"\0" * (-out.tell() % ALIGNMENT))
        for name, array, kind in tensors:
            data = array.tobytes()
            out.write(data)
            out.write(b"\0" * (-len(data) % ALIGNMENT))
    print(f"wrote {target}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
