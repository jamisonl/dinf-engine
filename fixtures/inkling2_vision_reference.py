"""Check Inkling 2 image embeddings against the official vision weights.

Usage: inkling2_vision_reference.py OFFICIAL_DIR STEM [STEM...]

Each STEM names the `STEM.rgb`, `STEM.json` and `STEM.f32` files that
`examples/inkling2_vision.rs` writes. This recomputes the vision tower in
float32 numpy straight from the official `model.visual.*` safetensors (not
from our GGUF), on the same pixels, and compares the projector output.

The tower: 20-pixel HWC patches through a linear embedding, learned 32 x 32
positions resized bilinearly (half-pixel, clamped), a class token in front,
pre-LayerNorm, 23 pre-norm blocks with two-axis RoPE (the first half of each
head turns with the column, the second with the row, adjacent pairs; class at
(0, 0), patches at (x + 1, y + 1)) and layer scales, then the class token is
dropped, 2 x 2 patches are concatenated (TL, TR, BL, BR) and two bias-free
linears with exact GELU between them project to the decoder width.

Gates: relative L2 <= 0.01 and cosine >= 0.9999 per image.
Run with uv's numpy: `uv run --offline --no-project --with numpy python -I`.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

PATCH = 20
MERGE = 2
HEADS = 16
EPS = 1e-5
THETA = 10000.0


class Weights:
    def __init__(self, directory: Path):
        self.files = {}
        for path in sorted(directory.glob("*.safetensors")):
            with open(path, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)
            for name, info in header.items():
                if name.startswith("model.visual."):
                    self.files[name] = (data, info)

    def __call__(self, name: str) -> np.ndarray:
        data, info = self.files["model.visual." + name]
        start, end = info["data_offsets"]
        raw = np.asarray(data[start:end])
        if info["dtype"] == "BF16":
            values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F32":
            values = raw.view(np.float32)
        else:
            raise ValueError(f"{name}: unexpected dtype {info['dtype']}")
        return values.reshape(info["shape"]).astype(np.float32)


def layer_norm(x, weight, bias):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def gelu(x):
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * x * (1.0 + erf(x.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32)


def resize_positions(grid, cols, rows):
    side, hidden = int(round(math.sqrt(grid.shape[0]))), grid.shape[1]
    grid = grid.reshape(side, side, hidden)
    if (cols, rows) == (side, side):
        return grid.reshape(-1, hidden)

    def axis(out):
        x = (np.arange(out) + 0.5) * side / out - 0.5
        lo = np.clip(np.floor(x).astype(int), 0, side - 1)
        hi = np.clip(np.floor(x).astype(int) + 1, 0, side - 1)
        frac = np.clip(x - lo, 0.0, 1.0).astype(np.float32)
        return lo, hi, frac

    y0, y1, dy = axis(rows)
    x0, x1, dx = axis(cols)
    top = grid[y0][:, x0] * (1 - dx)[None, :, None] + grid[y0][:, x1] * dx[None, :, None]
    bottom = grid[y1][:, x0] * (1 - dx)[None, :, None] + grid[y1][:, x1] * dx[None, :, None]
    out = top * (1 - dy)[:, None, None] + bottom * dy[:, None, None]
    return out.reshape(-1, hidden)


def rope(x, pos_w, pos_h):
    """x: [tokens, heads, head]. Rotates adjacent pairs in each half."""
    head = x.shape[-1]
    half = head // 2
    freqs = THETA ** (-np.arange(0, half, 2, dtype=np.float64) / half)
    out = x.copy()
    for offset, pos in ((0, pos_w), (half, pos_h)):
        angle = np.outer(pos, freqs)  # [tokens, half / 2]
        cos = np.cos(angle).astype(np.float32)[:, None, :]
        sin = np.sin(angle).astype(np.float32)[:, None, :]
        even = x[..., offset : offset + half : 2]
        odd = x[..., offset + 1 : offset + half : 2]
        out[..., offset : offset + half : 2] = even * cos - odd * sin
        out[..., offset + 1 : offset + half : 2] = even * sin + odd * cos
    return out


def tower(w: Weights, rgb: np.ndarray, padded_width: int, padded_height: int) -> np.ndarray:
    height, width, _ = rgb.shape
    pixels = np.full((padded_height, padded_width, 3), -1.0, dtype=np.float32)
    pixels[:height, :width] = rgb
    pixels = (pixels - 127.5) / 127.5
    cols, rows = padded_width // PATCH, padded_height // PATCH
    patches = (
        pixels.reshape(rows, PATCH, cols, PATCH, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * cols, PATCH * PATCH * 3)
    )
    x = patches @ w("backbone.embeddings.patch_embedding.weight").T
    x = x + resize_positions(w("backbone.embeddings.position_embedding"), cols, rows)
    cls = w("backbone.embeddings.class_embedding") + w("backbone.embeddings.cls_position_embedding")
    x = np.concatenate([cls[None], x], 0)
    x = layer_norm(x, w("backbone.ln_pre.weight"), w("backbone.ln_pre.bias"))
    pos_w = np.concatenate([[0], np.tile(np.arange(cols) + 1, rows)]).astype(np.float64)
    pos_h = np.concatenate([[0], np.repeat(np.arange(rows) + 1, cols)]).astype(np.float64)
    tokens, hidden = x.shape
    head = hidden // HEADS
    layer = 0
    while f"model.visual.backbone.layers.{layer}.ln_1.weight" in w.files:
        p = f"backbone.layers.{layer}."
        h = layer_norm(x, w(p + "ln_1.weight"), w(p + "ln_1.bias"))
        q, k, v = (
            (h @ w(p + f"self_attn.{n}_proj.weight").T + w(p + f"self_attn.{n}_proj.bias")).reshape(
                tokens, HEADS, head
            )
            for n in "qkv"
        )
        q, k = rope(q, pos_w, pos_h), rope(k, pos_w, pos_h)
        scores = np.einsum("thd,shd->hts", q, k) / math.sqrt(head)
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        attn = np.einsum("hts,shd->thd", scores, v).reshape(tokens, hidden)
        attn = attn @ w(p + "self_attn.out_proj.weight").T + w(p + "self_attn.out_proj.bias")
        x = x + attn * w(p + "ls_1.gamma")
        h = layer_norm(x, w(p + "ln_2.weight"), w(p + "ln_2.bias"))
        h = gelu(h @ w(p + "mlp.c_fc.weight").T + w(p + "mlp.c_fc.bias"))
        h = h @ w(p + "mlp.c_proj.weight").T + w(p + "mlp.c_proj.bias")
        x = x + h * w(p + "ls_2.gamma")
        layer += 1
    x = x[1:].reshape(rows // MERGE, MERGE, cols // MERGE, MERGE, hidden)
    x = x.transpose(0, 2, 1, 3, 4).reshape(-1, MERGE * MERGE * hidden)
    x = gelu(x @ w("proj_in.weight").T)
    return x @ w("proj_out.weight").T, layer


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    weights = Weights(Path(sys.argv[1]))
    failed = False
    for stem in sys.argv[2:]:
        plan = json.loads(Path(stem + ".json").read_text())
        rgb = np.fromfile(stem + ".rgb", dtype=np.uint8).astype(np.float32)
        rgb = rgb.reshape(plan["height"], plan["width"], 3)
        expected, layers = tower(weights, rgb, plan["padded_width"], plan["padded_height"])
        actual = np.fromfile(stem + ".f32", dtype="<f4").reshape(expected.shape)
        diff = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        cosine = float(
            (actual * expected).sum() / (np.linalg.norm(actual) * np.linalg.norm(expected))
        )
        worst_row = min(
            float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e))) for a, e in zip(actual, expected)
        )
        ok = diff <= 0.01 and cosine >= 0.9999
        failed |= not ok
        print(
            json.dumps(
                {
                    "image": Path(stem).name,
                    "tokens": expected.shape[0],
                    "layers": layers,
                    "relative_l2": float(diff),
                    "cosine": cosine,
                    "worst_row_cosine": worst_row,
                    "passed": bool(ok),
                }
            )
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
