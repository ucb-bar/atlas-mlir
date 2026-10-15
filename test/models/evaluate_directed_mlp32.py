"""Evaluator-only original PyTorch outputs for the directed MLP fixture."""

import argparse
import json

import torch

from directed_mlp32 import get_model_and_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    model, (source,) = get_model_and_inputs()
    if args.variant:
        source = torch.roll(source, shifts=1, dims=1)
    with torch.no_grad():
        expected = model(source)
    print(json.dumps({"input": source.tolist(), "original_output": expected.tolist()}))


if __name__ == "__main__":
    main()
