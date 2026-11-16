"""Synthetic-weight, batch-one Llama prefill layer for compiler integration tests.

The full configuration uses TinyLlama's 2048/5632 widths and 32/4 query/KV
heads. Parameters and rotary tables are explicit inputs, never compiler answers.
This covers one layer, including both residuals and returned current K/V; it
does not implement token embedding, historical KV-cache updates or decoding.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


class Layer(torch.nn.Module):
    def __init__(self, heads, kv_heads):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads

    def forward(self, x, norm1, norm2, wq, wk, wv, wo, wg, wu, wd, cos, sin, mask):
        b, s, h = x.shape
        d = h // self.heads

        def rms(value, weight):
            return (
                value
                * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-5)
                * weight
            )

        def linear(value, weight):
            return torch.mm(value.reshape(-1, value.shape[-1]), weight.t()).reshape(
                b, s, -1
            )

        def rotary(value):
            rotated = torch.cat((-value[..., d // 2 :], value[..., : d // 2]), dim=-1)
            return value * cos + rotated * sin

        a = rms(x, norm1)
        q = rotary(linear(a, wq).reshape(b, s, self.heads, d).transpose(1, 2))
        k = rotary(linear(a, wk).reshape(b, s, self.kv_heads, d).transpose(1, 2))
        v = linear(a, wv).reshape(b, s, self.kv_heads, d).transpose(1, 2)
        kr = (
            k[:, :, None, :, :]
            .expand(b, self.kv_heads, self.heads // self.kv_heads, s, d)
            .reshape(b, self.heads, s, d)
        )
        vr = (
            v[:, :, None, :, :]
            .expand(b, self.kv_heads, self.heads // self.kv_heads, s, d)
            .reshape(b, self.heads, s, d)
        )
        scores = torch.matmul(q, kr.transpose(-2, -1)) * (d**-0.5) + mask
        p = torch.softmax(scores, dim=-1)
        attention = torch.matmul(p, vr).transpose(1, 2).reshape(b, s, h)
        residual = x + linear(attention, wo)
        a = rms(residual, norm2)
        gate = linear(a, wg)
        out = residual + linear(gate * torch.sigmoid(gate) * linear(a, wu), wd)
        return out, k, v


def arguments(hidden=128, intermediate=256, heads=4, kv_heads=2, sequence=32, seed=7):
    rng = np.random.default_rng(seed)
    d, kv = hidden // heads, kv_heads * (hidden // heads)
    shapes = [
        (1, sequence, hidden),
        (hidden,),
        (hidden,),
        (hidden, hidden),
        (kv, hidden),
        (kv, hidden),
        (hidden, hidden),
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    ]
    arrays = [
        rng.uniform(-0.25, 0.25, size=shape).astype(np.float32) for shape in shapes
    ]
    arrays[1][:] = 1.0
    arrays[2][:] = 1.0
    positions = np.arange(sequence, dtype=np.float32)[:, None]
    frequency = 1.0 / (10000 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    angles = np.concatenate((positions * frequency, positions * frequency), axis=-1)[
        None, None
    ]
    arrays.extend(
        (
            np.cos(angles).astype(np.float32),
            np.sin(angles).astype(np.float32),
            np.where(np.triu(np.ones((sequence, sequence)), 1), -np.inf, 0).astype(
                np.float32
            )[None, None],
        )
    )
    return tuple(torch.from_numpy(a) for a in arrays)


def main():
    from torch_mlir import fx

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--full", action="store_true")
    geometry.add_argument(
        "--tail",
        action="store_true",
        help="120/270 widths and 30-wide heads exercise M/N/K tails",
    )
    parser.add_argument("--sequence", type=int, default=32)
    args = parser.parse_args()
    config = (
        (2048, 5632, 32, 4)
        if args.full
        else (120, 270, 4, 2)
        if args.tail
        else (128, 256, 4, 2)
    )
    values = arguments(*config, sequence=args.sequence)
    module = fx.export_and_import(
        Layer(*config[2:]), *values, output_type="linalg-on-tensors"
    )
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "layer.mlir").write_text(str(module))
    np.savez(
        args.destination / "inputs.npz",
        **{f"arg{i}": v.numpy() for i, v in enumerate(values)},
    )
    with torch.no_grad():
        expected = Layer(*config[2:])(*values)
    np.savez(
        args.destination / "torch-f32.npz",
        **{f"output{i}": v.numpy() for i, v in enumerate(expected)},
    )


if __name__ == "__main__":
    main()
