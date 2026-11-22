# Host source families: GELU, layer norm, sign flip, and BF16 contraction

Merlin recovers four closed BF16 source operations that the frontend decomposes (GELU, layer norm, negation, and an unfolded convolution) as one declared node each. Atlas declares one generated family per node in `numerical_contracts.py` and `compiler_interface.py`. Each runs as checked host C, and none changes precision.

| Family | Source node | Arithmetic |
| --- | --- | --- |
| `atlas.host-gelu` | `bf16_gelu` (tanh or erf) | The source binary32 order with its literal bits, the pinned correctly rounded `tanhf` or `erff`, and one BF16 RNE. |
| `atlas.host-layer-norm` | `bf16_layer_norm` | The pinned torch wheel's CPU kernel: chunked Welford moments, 7 fused and 2 rounded multiply-add sites, one sqrt, one divide, and one BF16 RNE. |
| `atlas.host-sign-flip` | `bf16_neg` | Flip only the sign bit, for all 65,536 BF16 bit patterns. |
| `atlas.host-bf16-contraction` | `bf16_affine_contraction` | Exact BF16 widening, binary32 products, an ascending binary32 sum from +0, the bias, and one BF16 RNE. |

None of the four has an alternate route in `compiler_interface._ALTERNATE_ROUTES`: each runs only on its declared host route, or the region is refused.

## What EXACT means here

GELU and the sign flip are exact to the source order itself. For the contraction the source leaves the reduction order unresolved, and the family declares the ascending order. The generated C is bit-exact to that declared ascending order and within one BF16 unit in the last place of torch eager. Contraction outputs are therefore not expected to be bit-equal to exporter golden outputs that torch eager produced.

Layer norm is defined by its arithmetic, a fixed sequence of binary32 operations with 7 fused and 2 rounded multiply-add sites. It gives the same bits on any IEEE machine, so kits, expected outputs, and the emitted C do not depend on the CPU that builds them. The pinned torch 2.11.0+cpu wheel on an AVX2 or AVX512 CPU is the witness that this sequence is torch eager. On a CPU without AVX2, or with ATEN_CPU_CAPABILITY set, the same wheel runs a different kernel: the eager tests fail at their capability assertion, and eager outputs captured there can differ from Atlas by one BF16 unit. Run the eager tests and capture any eager comparison outputs on an AVX2 or AVX512 x86 host with ATEN_CPU_CAPABILITY unset. A torch version bump needs a new qualification and a new semantics id.

Merlin's evaluation references that set `FP_NUMERICS_ENV` force `ATEN_CPU_CAPABILITY=default`, which is the capability this family declares as not matched. A comparison of an Atlas layer norm against such a reference shows one-unit differences that come from the other torch kernel, not from Atlas.

## Conv2d: intake only

An unquantized BF16 Conv2d export site is admitted as a host site (`export_contract.py`), and its unfolded form selects `atlas.host-bf16-contraction`. This is intake only. No front-door kit holds a host BF16 Conv2d yet, and the independent reference has no aten convolution handler, so such a kit fails its exact check, and does not skip it, until that handler exists.

## Temporary parts and their replacements

- Merlin's source-composition pass is temporary. Its replacement is model2MLIR emitting each of these operations as one generic tagged with its numerical semantics, as it already does for BF16 sigmoid and batch norm.
- Merlin `affine_contraction.py` duplicates the BF16 linear policy of `bf16_linear.py`. Its replacement makes a BF16 linear the gather-free case of `bf16_affine_contraction`, and then deletes `bf16_linear.py` and Atlas `source_bf16_linear.py`.
