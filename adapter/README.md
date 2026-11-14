# Atlas FP8 quantization adapter

This provides the `atlas_fp8` entry point for model2MLIR's quantization adapters. It reads the quantized model from [atlas-torchao](https://github.com/ucb-ee290c/atlas-torchao) as it is: FP8 codes, exponents, and ordinary parameters. It checks that they match the source and passes them to capture without calibration, conversion, or recoding.

The TinyLlama flow in https://github.com/ucb-bar/merlin/pull/63 uses it. From the atlas-mlir checkout, install this separate package with `python -m pip install ./adapter` next to model2MLIR, then follow the Merlin [decoder walkthrough](https://github.com/reed-nicolas/merlin/blob/feat/tinyllama/examples/atlas/whole-model/unchanged_fx/README.md).

This is a provisional home. It may move permanently next to the quantization code in atlas-torchao. [tests/test_adapter.py](tests/test_adapter.py) covers source and parameter identity, immutable code and exponent handling, capture integration, and fail-closed refusals for mutated inputs.
