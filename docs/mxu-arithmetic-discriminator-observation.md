# MXU0 and MXU1 arithmetic discriminator

This is a bounded selected-standalone-`AtlasCore` observation for reset
matmul. It tests two sparse 32-product cells and checks the complete 32-by-32
BF16 output buffers. It does not qualify all FP8 encodings, accumulation
state, timing, or integrated SoC execution.

| Inspected role | Exact local revision |
| --- | --- |
| Selected Atlas RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Inspected npu_model implementation | `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Inspected Merlin software policy | local Merlin `96d26f24daa9b566ebdaf0776dfebce45df993e8` |
| MLIR/LLVM build | LLVM source `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |
| Hand OOT starting point | `atlas-mlir` `7fc21b2` |

The v2 plan inherited a different npu_model inspection pin,
`6c8601015a39b98e54ca1f9aeaac7536685ae43d`. These observations apply
to the local `5bb08624` source and do not silently qualify the earlier
revision. `npu_model/configs/isa_definition.py:94-107::_vmatmul` converts
FP8 operands to FP16, computes `activation_fp16 @ weight_fp16`, then converts
its FP16 result to BF16 for **both** units. The local Merlin
`examples/atlas/target/software-spec.yaml:6-15` declares ordered,
per-step BF16 round-to-nearest-even for the selected contraction policy.
Those are competing descriptions, not interchangeable references.

The selected RTL has distinct units. `mxu/sa/PE.scala` feeds MXU0's
`CustomFMA` through `E4M3FMA` at each systolic processing element.
`mxu/ipt/AnchorAccumulationTree.scala:48-118` instead forms E4M3 products,
aligns them to an exponent anchor, reduces signed integers, and converts
the MXU1 result to BF16. The second path may discard small aligned terms
outside this test's range; we do not label it mathematically exact for all
inputs.

The test's expected bits come from exact rational E4M3 decoding and an
independent integer quotient/remainder implementation of IEEE
round-to-nearest-even. Neither Torch, npu_model, the OOT emitter, nor the
selected core computes its expected values. All other product positions are
zero. Each row below lists the three nonzero weight and activation operands
identically, in K order.

| Witness (E4M3 bits) | Exact sum | Ordered BF16 steps | FP16 matmul result, then BF16 | Single final BF16 round | Selected MXU0 | Selected MXU1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `38,18,18` (`1,1/16,1/16`) | `1 + 2/256` | `3f80` | `3c08` → `3f81` | `3f81` | `3f80` | `3f81` |
| `38,18,08` (`1,1/16,1/64`) | `1 + 1/256 + 1/4096` | `3f80` | `3c04` → `3f80` | `3f81` | `3f80` | `3f81` |

For the first witness, each `1/256` product is exactly half a BF16 ULP
at 1. Repeated per-step ties remain at `0x3f80`, while the FP16 output
converts to `0x3f81`. For the second, the exact sum lies `1/4096` above
the BF16 midpoint. It rounds to the midpoint in FP16 and then ties down
to BF16 `0x3f80`, while MXU1's observed result is `0x3f81`.

A separate execution of the model's stated Torch expression in the
`npu_model-atlas` Python environment used 1-by-32 FP16 activations and
32-by-1 FP16 weights. It returned FP16 bits `0x3c08` and `0x3c04`, then
BF16 bits `0x3f81` and `0x3f80`, respectively. This checks the model
expression on the witnesses; it is not execution of a complete npu_model
instruction stream.

The test reuses the hand-authored 42-word
[`mxu0_pair.mlir`](../test/examples/mxu0_pair.mlir) and
[`mxu1_pair.mlir`](../test/examples/mxu1_pair.mlir) reset programs.
Both streams match their LLVM-lowered ELF32 RISC-V object words; the
matmul word matches the selected assembler for its unit. Four selected
standalone-core runs checked all 2,048 output bytes, both 1,024-byte FP8
input tiles, a 32-byte guard, halt, and 64 DMA read/64 DMA write beats.
The significant output is cell `(0,0)`; all other cells were checked zero.
The core matched ordered BF16 steps for MXU0 and a single final BF16 round
for MXU1 in these witnesses. It disagreed with the inspected npu_model
FP16-then-BF16 expression on the indicated witness for **each** unit.

These results are **MXU0 PASS** and **MXU1 PASS** for the two bounded
discriminators, with **model equivalence FAIL** for one discriminator per
unit. Full-domain MXU0 arithmetic, MXU1 anchor truncation/overflow,
exceptional values, seeded/continued accumulators, temporal availability,
and integrated execution remain **BLOCKED**. The 99-mode ledger remains
at 29 bounded mode-specific tests, 0 software-admitted modes, and 99
full-qualification blockers; this test deepens existing MXU reset evidence
without adding a new mode to the numerator.
