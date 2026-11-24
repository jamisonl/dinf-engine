#!/usr/bin/env python3
"""Build Inkling 2 Mini's DSpark drafter GGUF from the official checkpoint.

The official checkpoint (`thinkingmachines/Inkling-2-Mini`) ships its block
drafter as `mtp.safetensors` (all BF16) with `mtp_config` in `config.json`:
three dense layers over the target's features, block size 8, and a
recurrent (`rnn_hidden`) Markov head. This writes it as a
`general.architecture = dflash` GGUF in the tensor layout of the published
DSpark drafter GGUFs, so one file serves every runtime that reads them:

| checkpoint | GGUF |
|---|---|
| `layers.N.input_layernorm` / `post_attention_layernorm` | `blk.N.attn_norm` / `ffn_norm` |
| `layers.N.dflare_norm` | `blk.N.dflare_norm` |
| `layers.N.self_attn.{q,k,v,o}_proj` | `blk.N.attn_{q,k,v,output}` |
| `layers.N.self_attn.{k,v}_proj_ctx` | `blk.N.attn_{k,v}_ctx` |
| `layers.N.self_attn.{q,k,k_norm_ctx}` norms | `blk.N.attn_{q,k,k_ctx}_norm` |
| `layers.N.mlp.{gate,up,down}_proj` | `blk.N.ffn_{gate,up,down}` |
| `norm`, `mask_embed`, `fuse_weight` | `output_norm`, `mask_embd`, `dspark_fuse` |
| `markov_head.{markov_w1, rnn_proj, hidden_out, dflare_norm}` | `markov_w1`, `joint_proj`, `dspark_hidden_out`, `dspark_norm` |

Projections are Q8_0 (as ggml's `quantize_row_q8_0_ref`), or BF16 with
`--bf16`. The Markov head's matrices stay BF16; norms, biases, the mask
embedding and the raw (pre-softmax) fusion logits are F32 upcasts. The
token embedding and output head are not stored: the drafter borrows the
target's. Tokenizer keys are copied from the target GGUF.

Usage: inkling2_dspark_gguf.py [--bf16] OFFICIAL_DIR TARGET.gguf OUTPUT.gguf

OFFICIAL_DIR holds `mtp.safetensors` and `config.json`. Needs numpy.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

from inkling2_official_gguf import GGUF_MAGIC, Safetensors, q8_0_blocks

F32, BF16, Q8_0 = 0, 30, 8
STRING, BOOL, U32, F32_KV, ARRAY, I32 = 8, 7, 4, 6, 9, 5

LAYER = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "dflare_norm": "dflare_norm",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.k_proj_ctx": "attn_k_ctx",
    "self_attn.v_proj_ctx": "attn_v_ctx",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "self_attn.k_norm_ctx": "attn_k_ctx_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
GLOBAL = {
    "norm.weight": "output_norm.weight",
    "mask_embed": "mask_embd.weight",
    "fuse_weight": "dspark_fuse.weight",
    "markov_head.dflare_norm.weight": "dspark_norm.weight",
    "markov_head.hidden_out.weight": "dspark_hidden_out.weight",
    "markov_head.hidden_out.bias": "dspark_hidden_out.bias",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.rnn_proj.weight": "joint_proj.weight",
    "markov_head.rnn_proj.bias": "joint_proj.bias",
}
# Matrices that keep BF16 whatever the projection type.
MARKOV_MATRICES = {"dspark_hidden_out.weight", "markov_w1.weight", "joint_proj.weight"}


def tokenizer_kvs(path: Path) -> list[bytes]:
    """The raw key/value records of every `tokenizer.*` key in a GGUF v3."""
    data = path.read_bytes()
    pos = 4
    assert data[:4] == GGUF_MAGIC and struct.unpack_from("<I", data, pos)[0] == 3

    def read(fmt):
        nonlocal pos
        value = struct.unpack_from("<" + fmt, data, pos)[0]
        pos += struct.calcsize("<" + fmt)
        return value

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(kind):
        nonlocal pos
        if kind == STRING:
            length = read("Q")
            pos += length
        elif kind == ARRAY:
            element = read("I")
            for _ in range(read("Q")):
                skip(element)
        else:
            pos += sizes[kind]

    pos = 8
    read("Q")
    records = []
    for _ in range(read("Q")):
        start = pos
        key = data[pos + 8 : pos + 8 + struct.unpack_from("<Q", data, pos)[0]].decode()
        pos += 8 + len(key.encode())
        skip(read("I"))
        if key.startswith("tokenizer."):
            records.append(data[start:pos])
    return records


def kv_string(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<IQ", STRING, len(v)) + v


def kv_u32(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<II", U32, value)


def kv_f32(key: str, value: float) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<If", F32_KV, value)


def kv_bool(key: str, value: bool) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<IB", BOOL, int(value))


def kv_array(key: str, kind: int, values: list) -> bytes:
    k = key.encode()
    fmt = {I32: "i", BOOL: "B"}[kind]
    body = b"".join(struct.pack("<" + fmt, int(v)) for v in values)
    return struct.pack("<Q", len(k)) + k + struct.pack("<IIQ", ARRAY, kind, len(values)) + body


def main() -> None:
    args = sys.argv[1:]
    projection = Q8_0
    if args[:1] == ["--bf16"]:
        projection, args = BF16, args[1:]
    if len(args) != 3:
        sys.exit(__doc__)
    official, target_gguf, output = Path(args[0]), Path(args[1]), Path(args[2])
    if output.exists():
        sys.exit(f"{output} exists; choose a new output path")
    config = json.loads((official / "config.json").read_text())
    mtp, text = config["mtp_config"], config["text_config"]
    if mtp.get("markov_head_type") != "rnn_hidden" or mtp.get("q_bias") or mtp.get("o_bias"):
        sys.exit("unsupported mtp_config: expected an rnn_hidden Markov head without biases")
    weights = Safetensors(official / "mtp.safetensors")

    layers = mtp["num_hidden_layers"]
    local = set(mtp["local_layer_ids"])
    kvs = [
        kv_string("general.architecture", "dflash"),
        kv_string("general.type", "model"),
        kv_string("general.name", "Inkling-2-Mini DSpark"),
        kv_u32("dflash.block_count", layers),
        kv_u32("dflash.context_length", mtp["max_position_embeddings"]),
        kv_u32("dflash.embedding_length", mtp["hidden_size"]),
        kv_u32("dflash.feed_forward_length", mtp["intermediate_size"]),
        kv_u32("dflash.attention.head_count", mtp["num_attention_heads"]),
        kv_u32("dflash.attention.head_count_kv", mtp["num_key_value_heads"]),
        kv_f32("dflash.rope.freq_base", mtp["rope_theta"]),
        kv_f32("dflash.attention.layer_norm_rms_epsilon", mtp["rms_norm_eps"]),
        kv_u32("dflash.attention.key_length", mtp["head_dim"]),
        kv_u32("dflash.attention.value_length", mtp["head_dim"]),
        kv_u32("dflash.vocab_size", text["vocab_size"]),
        kv_u32("dflash.vocab_size_unpadded", text["unpadded_vocab_size"]),
        kv_u32("dflash.block_size", mtp["block_size"]),
        kv_array("dflash.target_layers", I32, mtp["target_layer_ids"]),
        kv_u32("dflash.dspark.markov_rank", mtp["markov_rank"]),
        kv_bool("dflash.sample_from_anchor", False),
        kv_bool("dflash.has_confidence_head", False),
        kv_bool("dflash.attention.causal", False),
        kv_u32("dflash.attention.sliding_window", mtp["sliding_window_size"]),
        kv_array("dflash.attention.sliding_window_pattern", BOOL, [i in local for i in range(layers)]),
        kv_u32("dflash.rope.dimension_count", mtp["head_dim"]),
        kv_f32("dflash.final_logit_softcapping", text["final_logit_softcapping"]),
        kv_u32("general.file_type", 7 if projection == Q8_0 else 32),
        kv_u32("general.quantization_version", 2),
        *tokenizer_kvs(target_gguf),
    ]

    def as_f32(name):
        raw, _, dtype = weights.raw(name)
        assert dtype == "BF16", f"{name}: expected BF16, got {dtype}"
        bits = np.frombuffer(raw.tobytes(), dtype=np.uint16).astype(np.uint32) << 16
        return lambda: bits.view(np.float32)

    def as_bf16(name):
        raw, _, _ = weights.raw(name)
        return lambda: np.frombuffer(raw.tobytes(), dtype=np.uint16)

    outputs = []

    def add(source, gguf_name):
        _, shape, dtype = weights.raw(source)
        assert dtype == "BF16", f"{source}: expected BF16, got {dtype}"
        dims = list(reversed(shape))
        if len(shape) == 1 or (len(shape) == 2 and shape == [4, 4]):
            outputs.append((gguf_name, dims, F32, as_f32(source)))
        elif gguf_name in MARKOV_MATRICES or projection == BF16:
            outputs.append((gguf_name, dims, BF16, as_bf16(source)))
        else:
            raw = weights.raw(source)[0]
            outputs.append((gguf_name, dims, Q8_0, lambda raw=raw, row=dims[0]: q8_0_blocks(raw, row)))

    for source, name in GLOBAL.items():
        add(source, name)
    for layer in range(layers):
        for source, name in LAYER.items():
            add(f"layers.{layer}.{source}.weight", f"blk.{layer}.{name}.weight")
    if len(outputs) != len(weights.header):
        sys.exit(f"mapped {len(outputs)} of {len(weights.header)} checkpoint tensors")

    def byte_len(dims, kind):
        elements = int(np.prod(dims))
        return elements // 32 * 34 if kind == Q8_0 else elements * (4 if kind == F32 else 2)

    alignment = 32
    with open(output, "wb") as out:
        out.write(GGUF_MAGIC + struct.pack("<IQQ", 3, len(outputs), len(kvs)))
        out.write(b"".join(kvs))
        offset = 0
        for name, dims, kind, _ in outputs:
            encoded = name.encode()
            out.write(struct.pack("<Q", len(encoded)) + encoded)
            out.write(struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims))
            out.write(struct.pack("<IQ", kind, offset))
            offset += (byte_len(dims, kind) + alignment - 1) // alignment * alignment
        out.write(b"\0" * (-out.tell() % alignment))
        for name, dims, kind, produce in outputs:
            data = np.ascontiguousarray(produce()).tobytes()
            if len(data) != byte_len(dims, kind):
                raise ValueError(f"{name}: wrote {len(data)} bytes, expected {byte_len(dims, kind)}")
            out.write(data)
            out.write(b"\0" * (-len(data) % alignment))
            print(f"{name} {dims} type {kind}", flush=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
