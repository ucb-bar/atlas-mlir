# Atlas FP8 quantization adapter

This provides the `atlas_fp8` entry point for model2MLIR's quantization adapters. It reads the quantized model from [atlas-torchao](https://github.com/ucb-ee290c/atlas-torchao) as it is: FP8 codes, exponents, and ordinary parameters. It checks that they match the source and passes them to capture without calibration, conversion, or recoding.

The atlas-mlir wheel ships this adapter and owns its entry point, so install it with Atlas's `capture` extra as the [capture setup](../docs/host-launch.md#install-model2mlir-for-capture) describes. Do not also install this directory as a separate package: `atlas_mlir.provenance.unique_plugins` refuses a second `atlas_fp8` entry point or one that another distribution owns. The TinyLlama flow in https://github.com/ucb-bar/merlin/pull/63 uses the adapter; follow the Merlin decoder walkthrough in that pull request.

This is a provisional home. It may move permanently next to the quantization code in atlas-torchao. [tests/test_adapter.py](tests/test_adapter.py) covers source and parameter identity, immutable code and exponent handling, capture integration, and fail-closed refusals for mutated inputs. The decoder attention test also needs `transformers` installed.
