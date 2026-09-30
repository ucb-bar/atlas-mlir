# MXU0 FP8 signed-zero discriminator

This is a bounded hand-authored OOT observation on the selected standalone
`AtlasCore` ARC model. It compares FP8 E4M3 `+0` (`0x00`) and `-0` (`0x80`)
activation encodings while every weight is the finite normal `+1` (`0x38`)
or `-1` (`0xb8`). It is not a full MXU0 numerical or integrated SoC
qualification.

| Role | Exact identity |
| --- | --- |
| Selected RTL | `ucb-bar/atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Selected FMA dependency pin | `sp26-fp-units` `9a0cc09c41ab918a3548580185f39cca8d559e0d` |
| Inspected model source | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Standalone core shared library SHA-256 | `196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc` |
| ARC state JSON SHA-256 | `db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3` |
| MLIR/LLVM source | `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |
| ModeLIR driver | local `add52b0a7c96d72e0079b7938e07ca5080871e82` |

The selected source's `E4M3Mul.scala` retains the XOR sign for a zero
product. `E4M3ProdAddBF16.scala` returns the BF16 addend unchanged when the
product exponent is zero. A reset matmul supplies positive zero as its
initial addend. These source observations predict that changing only a
zero-product sign cannot change this program's output bits; the core run
checks that prediction. The inspected `npu_model` FP16 matmul expression is
not used as the expected-value oracle.

The committed [test](../test/test_mxu0_signed_zero_reference.py) uses the
42-word typed [`mxu0_pair.mlir`](../test/examples/mxu0_pair.mlir) stream. It
passes through `atlas-opt --convert-atlas-to-llvm`, `mlir-translate`, and
`llc` to an ELF32 RISC-V object; the test compares the extracted object words
with `atlas-emit`. Eight fresh core runs preload a 32-by-32 normal-weight
panel and one of the following activation panels:

| Weights | Activation panel | Expected first output row | Observed |
| --- | --- | --- | --- |
| all `+1` | all `+0` | all BF16 `+0` | match |
| all `+1` | all `-0` | all BF16 `+0` | match |
| all `-1` | all `+0` | all BF16 `+0` | match |
| all `-1` | all `-0` | all BF16 `+0` | match |
| all `+1` | first activation `+1`, rest `+0` | all BF16 `+1` | match |
| all `+1` | first activation `+1`, rest `-0` | all BF16 `+1` | match |
| all `+1` | first activations `-0,+1`, rest `-0` | all BF16 `+1` | match |
| all `+1` | first activations `+1,-1`, rest `-0` | all BF16 `+0` | match |

Every other output row is expected to contain BF16 `+0`. The test compares
all 2,048 output bytes across both BF16 register halves for each case, not
just one cell. Every run halted, performed 64 DMA reads and 64 DMA writes,
preserved both 1,024-byte input panels, and preserved a 32-byte guard.

**Bounded result: 8/8 selected-core cases passed.** In this reset program,
`0x80` activation storage behaves like `0x00` under the tested finite-normal
weights. This does not prove signed-zero equivalence for MXU1, FP8-to-BF16
conversion, FP8 accumulator seeding, exceptional operands, all temporal
states, or model-level precision policy. In particular, it does not justify
admitting every exponent-zero encoding or removing the explicit input-domain
guard from a compiler. The selected mode census remains 54/99 bounded modes
and 0/99 software-admitted modes; this deepens the existing MXU0 reset row.
