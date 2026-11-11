#!/usr/bin/env python3
"""Exact-token Image2.1 text reference; prepare reads metadata only."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(snapshot, out, compute_mode):
    from tokenizers import Tokenizer

    prefix = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    )
    prompt = "A small red cube on a white table, soft daylight."
    rendered = (
        prefix + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
    tokenizer = Tokenizer.from_file(str(snapshot / "processor/tokenizer.json"))
    system = tokenizer.encode(prefix).ids
    ids = tokenizer.encode(rendered).ids
    assert ids[: len(system)] == system and len(system) < len(ids) <= 4096
    index = json.loads(
        (snapshot / "text_encoder/model.safetensors.index.json").read_text()
    )
    shards = sorted(set(index["weight_map"].values()))
    assert shards
    assert all(
        Path(name).name == name and name.endswith(".safetensors") for name in shards
    )
    # Full hashes bind every text shard before either runtime loads weights.
    pins = {
        "text_encoder/" + name: sha(snapshot / "text_encoder" / name) for name in shards
    }
    for name in [
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "processor/tokenizer.json",
    ]:
        pins[name] = sha(snapshot / name)
    dtype, backend = ("f32", "cpu") if compute_mode == "f32" else ("bf16", "mps")
    result = dict(
        schema_version=1,
        mode=compute_mode,
        backend=backend,
        dtype=dtype,
        compute_dtype=dtype,
        storage_dtype="f32le",
        prompt=prompt,
        token_ids=ids,
        drop_index=len(system),
        rendered_prompt=rendered,
        snapshot_files=[
            dict(path=path, sha256=digest) for path, digest in sorted(pins.items())
        ],
        scope="exact IDs and full pre-final-norm language hidden states; no vision or image quality claim",
    )
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            dict(
                mode="prepare",
                tokens=len(ids),
                drop_index=len(system),
                model_instances=0,
                output=str(out),
            )
        )
    )


def produce(snapshot, input_path, prefix):
    import torch
    import transformers
    from transformers import Qwen3VLForConditionalGeneration

    assert transformers.__version__ == "5.22.1" and torch.__version__.startswith(
        "2.15.0"
    )
    spec = json.loads(input_path.read_text())
    assert spec["schema_version"] == 1 and spec["storage_dtype"] == "f32le"
    mode = spec["mode"]
    assert mode in ("f32", "production_bf16")
    dtype, name, backend = (
        (torch.float32, "f32", "cpu")
        if mode == "f32"
        else (torch.bfloat16, "bf16", "mps")
    )
    assert spec["dtype"] == spec["compute_dtype"] == name and spec["backend"] == backend
    if backend == "mps":
        import os

        assert torch.backends.mps.is_available()
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    for row in spec["snapshot_files"]:
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert sha(snapshot / relative) == row["sha256"]
    assert 0 < spec["drop_index"] < len(spec["token_ids"]) <= 4096
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot / "text_encoder",
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not any(
        info.get(key)
        for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ), info
    model = model.to(backend).eval()
    assert all(
        parameter.device.type == backend
        and parameter.device.index in (None, 0)
        and parameter.dtype == dtype
        for parameter in model.parameters()
    )
    language = model.model.language_model
    # The pinned QwenImage21Pipeline uses this hook before reading the last
    # hidden state. The separate tiny fixture proves Transformers' capture rule.
    handle = language.norm.register_forward_hook(lambda module, args, output: args[0])
    ids = torch.tensor([spec["token_ids"]], dtype=torch.long, device=backend)
    try:
        hidden = language(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    finally:
        handle.remove()
    assert hidden.dtype == dtype and hidden.device.type == backend
    assert (
        list(hidden.shape) == [1, len(spec["token_ids"]), 4096]
        and torch.isfinite(hidden).all()
    )
    data = (
        hidden.detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes()
    )
    raw = prefix.with_suffix(".hidden.f32")
    with raw.open("xb") as stream:
        stream.write(data)
    spec.update(
        producer="Transformers Qwen3VL text-only forward",
        transformers=transformers.__version__,
        torch=torch.__version__,
        hidden_dtype=name,
        actual_device=str(hidden.device),
        parameter_devices=sorted(
            {str(parameter.device) for parameter in model.parameters()}
        ),
        parameter_dtypes=sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
        loading_info={
            key: sorted(info.get(key, []))
            for key in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        },
        attention="eager",
        hidden=dict(
            file=raw.name,
            shape=list(hidden.shape),
            sha256=hashlib.sha256(data).hexdigest(),
        ),
        input_sha256=sha(input_path),
        producer_sha256=sha(Path(__file__)),
    )
    encoded = json.dumps(spec, indent=2)
    with prefix.with_suffix(".json").open("x") as stream:
        stream.write(encoded)
    print(
        json.dumps(
            dict(
                mode="produce",
                shape=list(hidden.shape),
                compute_dtype=name,
                storage_dtype="f32le",
                backend=backend,
                output=str(prefix.with_suffix(".json")),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "produce"])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("input_or_output", type=Path)
    parser.add_argument("prefix", type=Path, nargs="?")
    parser.add_argument(
        "--compute-mode", choices=["f32", "production_bf16"], default="f32"
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        assert args.prefix is None
        prepare(args.snapshot, args.input_or_output, args.compute_mode)
    else:
        assert args.prefix is not None
        produce(args.snapshot, args.input_or_output, args.prefix)


if __name__ == "__main__":
    main()
