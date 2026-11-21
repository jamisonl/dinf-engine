#!/usr/bin/env python3
"""Repack an Inkling 2 Mini GGUF from llama.cpp's draft loop layout into the
merged `inkling2_loop` layout.

The draft layout (llama.cpp PR #32695, used by Unsloth's first Inkling 2 Mini
GGUFs) declares `general.architecture = inkling2` with `num_loops`. The merged
layout (llama.cpp #32694, ggml-org's files, and what dinf-engine reads) is
`inkling2_loop`. Every weight keeps its bytes and type; only names and
metadata change:

- `inkling2.*` keys become `inkling2_loop.*`. `block_count` and the per-layer
  arrays drop from the 48 layer visits to the physical layers. `num_loops`,
  `loop.core_start` and `loop.core_end` become `loop.count`, `loop.start` and
  `loop.end`, and `loop.varying = false` is added.
- `blk.N.loop_{attn,k,v,mlp}_{down,gain,forget}` become
  `blk.N.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`.
- The global `loop_res_query` (q) and `loop_kv_res_query` (kv), each
  `[2 * core, n_embd]`, become `blk.(start + i).loop_msg_query` with rows
  `[q[2i], kv[2i], kv[2i + 1], q[2i + 1]]`: the Q, K, V and MoE message
  queries, as llama.cpp's merged converter builds them from the same
  checkpoint tensors (conversion/inkling2.py, Inkling2LoopModel).

Usage: inkling2_draft_to_loop.py GGUF_PY_DIR INPUT.gguf OUTPUT.gguf

GGUF_PY_DIR is llama.cpp's gguf-py at or after 29b52d398b29, which knows
NVFP4. Needs numpy and tqdm.
"""

import re
import sys
from pathlib import Path

CONTROLLERS = {"attn": "q", "k": "k", "v": "v", "mlp": "ffn"}
LOOP_KEYS = {"num_loops": "loop.count", "loop.core_start": "loop.start", "loop.core_end": "loop.end"}
PER_LAYER_ARRAYS = {"attention.head_count", "attention.head_count_kv", "attention.sliding_window_pattern"}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.path.insert(0, sys.argv[1])
    import gguf
    import numpy as np
    from tqdm import tqdm

    source, target = Path(sys.argv[2]), Path(sys.argv[3])
    if target.exists():
        sys.exit(f"{target} exists; choose a new output path")
    reader = gguf.GGUFReader(source)
    fields = reader.fields
    arch = fields[gguf.Keys.General.ARCHITECTURE].contents()
    if arch != "inkling2" or "inkling2.num_loops" not in fields:
        sys.exit(f"{source} is not a draft-layout Inkling 2 loop GGUF (architecture {arch})")

    start = fields["inkling2.loop.core_start"].contents()
    end = fields["inkling2.loop.core_end"].contents()
    visits = fields["inkling2.block_count"].contents()
    layers = 1 + max(int(m.group(1)) for t in reader.tensors if (m := re.match(r"blk\.(\d+)\.", t.name)))
    passes = fields["inkling2.num_loops"].contents()
    if visits != layers + (passes - 1) * (end - start):
        sys.exit(f"{visits} visits do not match {layers} layers with core [{start}, {end}) x {passes}")

    writer = gguf.GGUFWriter(target, "inkling2_loop", endianess=reader.endianess)
    for field in fields.values():
        name = field.name
        if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if name.startswith("inkling2."):
            key = name.removeprefix("inkling2.")
            if key == "block_count":
                value = layers
            elif key in PER_LAYER_ARRAYS:
                # Visits past the physical layers repeat the core.
                extra = value[layers:]
                if extra != [value[start + i % (end - start)] for i in range(len(extra))]:
                    sys.exit(f"{name} does not repeat the loop core")
                value = value[:layers]
            name = "inkling2_loop." + LOOP_KEYS.get(key, key)
        writer.add_key_value(name, value, value_type, sub_type=sub_type)
    writer.add_key_value("inkling2_loop.loop.varying", False, gguf.GGUFValueType.BOOL)

    tensors = []
    queries = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name in ("loop_res_query.weight", "loop_kv_res_query.weight"):
            queries[name] = np.asarray(tensor.data, dtype=np.float32)
            continue
        m = re.fullmatch(r"blk\.(\d+)\.loop_(attn|k|v|mlp)_(down|gain|forget)\.weight", name)
        if m:
            name = f"blk.{m.group(1)}.loop_{CONTROLLERS[m.group(2)]}_ctrl_{m.group(3)}.weight"
        tensors.append((name, tensor.data, tensor.tensor_type))
    q, kv = queries["loop_res_query.weight"], queries["loop_kv_res_query.weight"]
    if q.shape != (2 * (end - start), q.shape[1]) or kv.shape != q.shape:
        sys.exit(f"cross-pass queries {q.shape} / {kv.shape} do not match the loop core")
    for i in range((end - start)):
        query = np.ascontiguousarray(np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]]))
        tensors.append((f"blk.{start + i}.loop_msg_query.weight", query, gguf.GGMLQuantizationType.F32))

    for name, data, kind in tensors:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    bar = tqdm(desc="Writing", total=sum(data.nbytes for _, data, _ in tensors), unit="byte", unit_scale=True)
    for _, data, _ in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        bar.update(data.nbytes)
    writer.close()


if __name__ == "__main__":
    main()
