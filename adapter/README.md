# Atlas FP8 quantization adapter

This provides the `atlas_fp8` entry point for model2MLIR's quantization adapters. It reads the quantized model from [atlas-torchao](https://github.com/ucb-ee290c/atlas-torchao) as it is: FP8 codes, exponents, and ordinary parameters. It checks that they match the source and passes them to capture without calibration, conversion, or recoding.

Before the native call census, it rewrites each biased `atlas::qlinear` into the bias-free contraction plus an ordinary addition, each `atlas::qconv2d` into a bias-free `atlas::qlinear` plus the reshape, permute, and bias addition of its definition, and each non-overlapping patch `aten.im2col` into a reshape and permute. The manifest lists every rewrite under `native_tail_rewrites`.

The TinyLlama flow in the Merlin PR {{PR:merlin}} uses it. From the atlas-mlir checkout, install this separate package with `python -m pip install ./adapter` next to model2MLIR, then follow the Merlin [decoder walkthrough](https://github.com/reed-nicolas/merlin/blob/49673bc57223f2da042bca01bee26d31d8ca8829/examples/atlas/whole-model/unchanged_fx/README.md#L1-L161).

This is a provisional home. It may move permanently next to the quantization code in atlas-torchao. The 40 tests in [tests/test_adapter.py](tests/test_adapter.py) pass. They cover source and parameter identity, immutable code and exponent handling, capture integration, these rewrites, and fail-closed refusals for mutated inputs.
