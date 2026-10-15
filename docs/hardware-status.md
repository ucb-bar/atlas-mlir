# Hardware status

This page lists which Atlas instructions have run on the RTL, what they
actually compute there, and where the RTL disagrees with `npu_model`.

All results are for the pinned RTL, `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2` (its `sp26-fp-units` submodule is
at `9a0cc09c41ab918a3548580185f39cca8d559e0d`). The software model it is
compared against is `npu_model-atlas`
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`. Later revisions of either may
behave differently.

"Tested on the RTL" means: the test builds a short Atlas program from the
dialect, checks that its words match both the RTL repo's own assembler
(`baremetal/assembler.py`) and the object file produced through LLVM, then
runs those words on an ARC (arcilator) build of the standalone `AtlasCore`.
ModeLIR's TileLink driver preloads DRAM, starts the program at reset and runs
it to `ECALL`. The test compares every output byte against an expected result
computed in Python, either with exact rational arithmetic or with a port of
the RTL's own lookup tables. It also checks that inputs and a guard region
were not overwritten. `npu_model` is never used to produce expected values.
A few behaviours are instead checked with Chisel/Verilator tests on a single
RTL block (MXU0 array, MXU1 lane, scalar decoder, scalar core).

Status values:

| Status | Meaning |
| --- | --- |
| Works | Ran on the RTL over full 32x32 panels with edge cases; behaviour is understood. Limits below still apply. |
| Limited | Ran on the RTL, but only for a narrow set of inputs, or it has a trap you must know about. |
| Untested | The compiler can emit the word, but it has never run on the RTL. |
| Not usable | Do not rely on it in compiled code. |

`docs/selected-variant-inventory.json` holds the per-mode record (bit
pattern, decoder controls, test name, known model differences) for all 99
decoder modes. Tooling reads it; do not edit it by hand.

## MXU

MXU0 is a systolic array; MXU1 is a set of inner-product trees. They do not
round the same way, so a compiler must pick one numerical rule per unit.

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| `VMATPUSH.W`, `VMATPUSH.A` (FP8), both units | Works | Used by every matmul test. Weights go in N-by-K orientation. | [test_mxu_reference.py](../test/test_mxu_reference.py) |
| `VMATMUL.MXU0` (reset) | Works | Adds each FP8 x FP8 product to a BF16 accumulator in K order, rounding to BF16 nearest-even after every step. Checked on every finite normal E4M3 code, mixed tiles, and `+0`/`-0` activations (`-0` behaves like `+0`). | [test_mxu_numerics.py](../test/test_mxu_numerics.py), [test_mxu0_signed_zero_reference.py](../test/test_mxu0_signed_zero_reference.py) |
| `VMATMUL.ACC.MXU0` | Works | Continues from the stored BF16 accumulator with the same per-step rounding. Two K tiles of 32 checked. | [test_mxu_k2_reference.py](../test/test_mxu_k2_reference.py) |
| MXU0 K head padding | Works | Padding a short K tile with leading FP8 `+0` does not change the result (K=31 and K=65 cases). | [HeadPadDiagnosticTest.scala](../test/rtl/HeadPadDiagnosticTest.scala) |
| `VMATMUL.MXU1` (reset) | Limited | Aligns all 32 products to one exponent anchor, sums them as integers, rounds once to BF16. Small products can be shifted out: 30 products of (1/64)^2 next to large cancelling terms give `0x0000`, not `0x3bf0`. The compiler only accepts E4M3 codes {0, ±1, ±2} for MXU1. | [test_mxu1_reference.py](../test/test_mxu1_reference.py), [AnchorDiagnosticTest.scala](../test/rtl/AnchorDiagnosticTest.scala), [test_mxu1_bounded_reference.py](../test/test_mxu1_bounded_reference.py) |
| `VMATMUL.ACC.MXU1` | Limited | Rounds to BF16 once per K tile, not once over the whole K. Checked with zero, ±1 and ±1/16 inputs. | [test_mxu1_continuation_reference.py](../test/test_mxu1_continuation_reference.py) |
| `VMATPOP.BF16`, both units | Works | Pops a BF16 pair (two 16-column halves). | (all matmul tests) |
| `VMATPUSH.ABF16`, `VMATPOP.FP8`, both units | Untested | | |

A half-ULP tie shows the difference: three products summing to `1 + 2/256`
give `0x3f80` on MXU0 and `0x3f81` on MXU1
([test_mxu_arithmetic_discriminator.py](../test/test_mxu_arithmetic_discriminator.py)).

## VPU arithmetic

All VPU BF16 operations work on register pairs. Source and destination pair
bases must be even; the verifier rejects odd bases.

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| `VADD.BF16` | Works | Adds in FP32, then keeps the top 16 bits (truncation, not BF16 rounding): `1 + 3/512` gives `0x3f80`. `+0 + x` returns `x` unchanged. | [test_vpu_add_reference.py](../test/test_vpu_add_reference.py) |
| `VSUB.BF16` | Works | Same FP32-then-truncate rule. `x - x` gives `+0`. | [test_vpu_sub_reference.py](../test/test_vpu_sub_reference.py) |
| `VMUL.BF16` | Works | One BF16 nearest-even rounding of the exact product. | [test_vpu_mul_reference.py](../test/test_vpu_mul_reference.py) |
| `VMIN.BF16`, `VMAX.BF16` | Works | Compares raw bit patterns, not IEEE values: `-0 < +0`, and NaNs are ordered by their bits (negative NaN is the smallest value, positive NaN the largest). Ties pick the second operand. | [test_vpu_min_reference.py](../test/test_vpu_min_reference.py), [test_vpu_max_reference.py](../test/test_vpu_max_reference.py) |
| `VMOV` | Limited | Moves a full register pair. Checked once, as a mutation of the ReLU test. | [test_vpu_relu_reference.py](../test/test_vpu_relu_reference.py) |
| `VRELU` | Works | Negative finite values become `+0`; non-negative values pass through bit for bit. `-0`, NaN, infinities and subnormals not checked. | [test_vpu_relu_reference.py](../test/test_vpu_relu_reference.py) |
| `VRECIP.BF16` | Works | Lookup-table approximation, not IEEE reciprocal. All 65,536 inputs match a port of the RTL table. Zero and subnormals give signed infinity; infinity and NaN give signed zero. The native compiler accepts positive normals in [1, 32]. | [test_vpu_recip_reference.py](../test/test_vpu_recip_reference.py), [characterize_vpu_recip.py](../test/diagnostics/characterize_vpu_recip.py) |
| `VSQRT` | Works | Lookup-table approximation. Ignores the sign, so `sqrt(-4) = 2`. Zero, subnormals and NaN give `+0`; infinity gives `+inf`. All 65,536 inputs match a port of the RTL table. | [test_vpu_sqrt_reference.py](../test/test_vpu_sqrt_reference.py), [characterize_vpu_sqrt.py](../test/diagnostics/characterize_vpu_sqrt.py) |
| `VEXP` | Limited | Fixed-point LUT approximation of `e^x`, not exact `exp`. The native compiler accepts normal inputs in [-80, 80] and `±0`; one generated panel ran on the RTL. | [test_vpu_exp_native_reference.py](../test/test_vpu_exp_native_reference.py) |
| `VEXP2` | Limited | Only integer inputs checked. Input 89 and above returns `+inf` even though 2^89 fits in BF16; 88 is finite. `±0` gives 1, `-inf` gives 0. | [test_vpu_exp2_reference.py](../test/test_vpu_exp2_reference.py) |
| `VSQUARE.BF16` | Limited | Only powers of two and special values checked. NaN, zero and subnormals give `+0`; overflow gives `+inf`. | [test_vpu_square_reference.py](../test/test_vpu_square_reference.py) |
| `VCUBE.BF16` | Limited | Only powers of two and special values checked. Keeps the sign: NaN, zero and subnormals give signed zero; overflow gives signed infinity. | [test_vpu_cube_reference.py](../test/test_vpu_cube_reference.py) |
| `VSIN`, `VCOS`, `VTANH`, `VLOG2` | Untested | | |

## VPU reductions

A BF16 register pair holds a 32x32 tile as two 32-row registers of 16 lanes
(left and right column halves).

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| `VREDSUM.ROW.BF16` | Works | Widens to FP32, adds adjacent lanes in a 5-level tree (FP32 nearest-even at each level), rounds once to BF16, writes the sum to every lane of the row. Order matters: `[2^25, 1, -2^25, 1, 2]` gives 2, a serial sum gives 3. | [test_vpu_row_sum_reference.py](../test/test_vpu_row_sum_reference.py) |
| `VREDMIN.ROW.BF16` | Works | Full 32-lane row minimum, broadcast. Raw-bit compare as for `VMIN`. Checked on finite values with a unique winner. | [test_vpu_row_min_reference.py](../test/test_vpu_row_min_reference.py) |
| `VREDMAX.ROW.BF16` | Works | Full 32-lane row maximum, broadcast. Raw-bit compare, checked including NaN and `±0`. | [test_vpu_row_max_reference.py](../test/test_vpu_row_max_reference.py) |
| `VREDSUM.BF16` (column) | Limited | 64x16 layout (see below). Adds rows serially in FP32 nearest-even, then truncates to the top 16 bits. | [test_vpu_col_sum_reference.py](../test/test_vpu_col_sum_reference.py) |
| `VREDMIN.BF16`, `VREDMAX.BF16` (column) | Limited | 64x16 layout (see below). Checked on finite normal values. | [test_vpu_col_min_reference.py](../test/test_vpu_col_min_reference.py), [test_vpu_col_max_reference.py](../test/test_vpu_col_max_reference.py) |

## Formats and data movement

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| `VLOAD`, `VSTORE` | Works | Raw byte copy between VMEM and a matrix register. The RTL needs a 1 KiB aligned base and a transfer inside one VMEM bank; the verifier does not check this, so the compiler must. | [test_vload_vstore_reference.py](../test/test_vload_vstore_reference.py) |
| `VLI.ALL` | Works | Writes the raw 16-bit immediate (no conversion) to every lane of both registers in the pair. | [test_vli_all_reference.py](../test/test_vli_all_reference.py) |
| `VLI.ROW`, `VLI.COL`, `VLI.ONE` | Works | Raw 16-bit immediate. ROW fills row 0 of both registers in the pair. COL fills lane 0 of every row and ONE fills row 0, lane 0, each in one register only; all other lanes of that register become zero. | [test_vli_selective_reference.py](../test/test_vli_selective_reference.py) |
| `VFP8PACK`, `VFP8UNPACK` | Limited | E8M0 scale code `c` means 2^(c-127), set with `SELI`. Pack joins two consecutive physical BF16 rows into one 32-byte FP8 row: packed row 0 is rows 0 and 1 of the first register. Only powers of two and zero checked. | [test_vpu_e8m0_pack_reference.py](../test/test_vpu_e8m0_pack_reference.py) |
| `VTRPOSE.XLU` | Works | 32x32 byte transpose. Reads the whole source before writing, so source and destination may be the same register. | [test_xlu_reference.py](../test/test_xlu_reference.py) |

## Scalar control

The PC is an instruction word index, not a byte address. Taken branches and
jumps have one delay slot.

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| 19 register and immediate ALU ops (`ADD` ... `SRAI`) | Works | Standard RV32 results. | [test_scalar_alu_reference.py](../test/test_scalar_alu_reference.py) |
| `LUI`, `AUIPC` | Works | `AUIPC` adds the word index: at word 1, upper immediate `0x90000` gives `0x90000001`. | [test_scalar_upper_reference.py](../test/test_scalar_upper_reference.py) |
| `LB`, `LBU`, `LH`, `LHU`, `LW`, `SB`, `SH`, `SW` | Limited | Results are correct, but a load writes its register several cycles later and nothing waits for it. Use a `DELAY` (8 was enough in tests) before reading the register. `ECALL` right after `DELAY` halts before the load lands; put another instruction in between. | [test_scalar_memory_reference.py](../test/test_scalar_memory_reference.py) |
| `BEQ`, `BLT` | Works | One delay slot. The encoded byte displacement is twice the word distance. Backward loops work. | [test_branch_reference.py](../test/test_branch_reference.py), [test_branch_two_positions_loop_reference.py](../test/test_branch_two_positions_loop_reference.py) |
| `BNE`, `BGE`, `BLTU`, `BGEU` | Untested | | |
| `JAL` | Works | Target = word PC + displacement/2; link = word index of the delay slot. LLVM lowering requires the target to be inside the same block. | [test_jal_reference.py](../test/test_jal_reference.py) |
| `JALR` | Limited | Target = `rs1 + imm` as a word index, low bit not cleared. LLVM lowering only accepts targets it can compute from `LUI`/`ADDI` constants in a straight-line prefix. Targets loaded from memory are rejected (not usable). | [test_jalr_reference.py](../test/test_jalr_reference.py), [test_jalr_load_delay_reference.py](../test/test_jalr_load_delay_reference.py) |
| `DELAY` | Works | Stalls fetch for the given count (unsigned 12 bits, 0..4095): `DELAY 4` holds the next instruction 4 cycles. Counts 0 and 4 checked. | [test_delay_reference.py](../test/test_delay_reference.py) |
| `FENCE` | Works | A no-op. It does not wait for scalar loads or DMA. | [test_fence_reference.py](../test/test_fence_reference.py) |
| `ECALL` | Works | Halts the core. Every test ends with it. | |
| CSR ops (`CSRRW` ... `CSRRCI`) | Untested | Used by tests to write debug CSRs `0xc10`/`0xc11`, but not tested on their own. Only `0xC00..0xC03` and `0xC10..0xC11` exist; any other address reads the cycle counter, so the verifier rejects it. | |
| `EBREAK`, `SELD`, `SELI` | Untested | `SELI` is used by the pack test to set a scale register. | |

## DMA

| Instruction | Status | What to know | Test |
| --- | --- | --- | --- |
| `DMA.LOAD`, `DMA.STORE` | Works | The DRAM address is read from the scalar register when the command issues; overwriting the register right after issue does not change the transfer. | [test_dma_reference.py](../test/test_dma_reference.py), [test_dma_pointer_capture.py](../test/test_dma_pointer_capture.py) |
| `DMA.WAIT` | Works | Blocks until the channel is idle. Reusing VMEM after the wait is safe. | [test_dma_pointer_lifetime.py](../test/test_dma_pointer_lifetime.py) |
| `DMA.CONFIG` | Limited | Encoded with funct7 0 (`0x0000007f` for channel 0). Sets the upper 32 bits of the DRAM address from a scalar register. Checked at the decoder and scalar core only, not through the DMA engine. | [DmaDecodeDiagnosticTest.scala](../test/rtl/DmaDecodeDiagnosticTest.scala), [DmaScalarCoreDiagnosticTest.scala](../test/rtl/DmaScalarCoreDiagnosticTest.scala) |

## RTL vs npu_model differences

Where they disagree, the compiler follows the RTL.

| Area | RTL | npu_model |
| --- | --- | --- |
| VR operand fields | `vd`, `vs1`, `vs2` are 6 bits each (bits 12:7, 18:13, 24:19) | Masks `vs1` to 7 bits and `vs2` to 5 bits, so `vs2 = m32` encodes as `m0` |
| `DMA.CONFIG` | funct7 0; funct7 1 is `DMA.WAIT` | funct7 1, which the RTL decodes as `DMA.WAIT` |
| `CSRRCI` | funct3 `111` | funct3 `100` |
| `VSQUARE`, `VCUBE` | funct7 `0x46`, `0x47` | funct7 `0x4e`, `0x4f` |
| Branch delay slots | One (`PcControl.scala`) | Architecture text says two |
| PC units | Word index (`JAL`, `JALR`, `AUIPC`, branches) | Byte address with pipeline-latency correction |
| `DELAY` | Stalls fetch | No-op |
| `VADD`, `VSUB` | FP32 result truncated to BF16 | Torch BF16 arithmetic |
| Row sum | Pairwise FP32 tree, one final round | Torch `sum` |
| Column sum/min/max | 64x16 physical layout | 32x32 logical tile |
| Min/max ops | Raw-bit ordering (NaN, `±0` as above) | Torch numeric `minimum`/`maximum` |
| `VRECIP`, `VSQRT`, `VEXP`, `VEXP2` | LUT approximations with the special-value rules above | Torch `reciprocal`, `sqrt`, `exp`, `exp2` |
| `VMOV` | Moves a register pair | Moves one register |
| `VLI.ROW/COL/ONE` | Writes the raw immediate bits | Converts the integer to BF16 first (`0x4000` becomes `0x4680`) |
| FP8 pack/unpack | Joins consecutive physical rows | Joins the two registers by logical row (a newer model revision reads physical rows but still has the 5-bit `vs2` encoding) |
| MXU arithmetic | MXU0: per-step BF16 rounding. MXU1: anchor-aligned integer sum, one rounding | FP16 matmul, then BF16, for both |
| Transpose | Separate XLU unit | Labelled as a vector-unit op (same encoding) |

## Column reductions

The RTL reads a register pair for `VREDSUM.BF16`, `VREDMIN.BF16` and
`VREDMAX.BF16` as 64 rows of 16 lanes: all 32 rows of the left-half register,
then all 32 rows of the right-half register (`VectorFSM.scala`). It reduces
each of the 16 lanes over those 64 rows and writes the same 16 results to
both destination registers. So lane `j` of the result combines logical columns
`j` and `j+16` of the 32x32 tile. The architecture text and `npu_model`
describe 32 separate column results instead.

The tests were built so the two readings give different answers, and the RTL
matched the 64x16 result every time. The compiler targets the 64x16 behaviour
for this RTL revision. If a later RTL changes it, that needs a separate target
configuration, not a silent change to this one.

## Running the RTL tests

Build and run the portable tests as described in the [top-level README](../README.md).
Tests that need the RTL skip when their paths are missing; a skip is not a pass.
Set these for the Python tests in `test/`:

| Variable | Points to |
| --- | --- |
| `ATLAS_OOT_BIN_DIR` | `bin/` of this repo's build (`atlas-opt`, `atlas-emit`) |
| `ATLAS_LLVM_BIN` | LLVM tools used to build object files (`mlir-translate`, `llc`) |
| `ATLAS_ASSEMBLER_ROOT` | `baremetal/` of the pinned `atlas-npu` checkout |
| `ATLAS_RTL_ROOT` | the pinned `atlas-npu` checkout |
| `ATLAS_MODEL_ROOT` | the pinned `npu_model-atlas` checkout (inventory check only) |
| `ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE` | ARC build of the standalone `AtlasCore` from the pinned RTL: shared library and state JSON |
| `ATLAS_MODELIR_ROOT` | ModeLIR checkout providing `mlc.backends.cosim_atlas` |

```sh
python -m unittest discover -s test -p test_vpu_add_reference.py -v
```

The two all-input sweeps run separately and take `--arc-model`, `--arc-state`,
`--modelir`, `--assembler`, `--rtl-root`, `--arithmetic-root` (the
`sp26-fp-units` checkout) and a fresh `--out` (the sqrt sweep also takes
`--model-root`); `--first-panel` and
`--panel-count` run a subset:

```sh
python test/diagnostics/characterize_vpu_recip.py --arc-model ... --out out/recip-all
```

The Scala tests in `test/rtl/` are not built here. Copy one into a clean
checkout of the pinned RTL under `src/test/scala/<package path>/` (the
`package` line says where; it must not already exist) and run it with mill and
a Verilator on `PATH`:

| Test | Copy to | Extra input |
| --- | --- | --- |
| `HeadPadDiagnosticTest.scala` | `atlas/sa/` | `ATLAS_MXU0_HEAD_PAD_FIXTURE` from `test/generate_mxu0_head_pad_fixture.py --out DIR` (optional; directed cases run without it) |
| `AnchorDiagnosticTest.scala` | `atlas/ipt/` | `ATLAS_MXU1_BROAD_FIXTURE` from `test/generate_mxu1_broad_fixture.py --out DIR`; `ATLAS_MXU1_TILE_FIXTURE` for a single 32x32 tile (`activation.bin`, `weight.bin`, `exact-bf16.bin`) |
| `DmaDecodeDiagnosticTest.scala` | `atlas/scalar/` | `ATLAS_DMA_DECODE_PROGRAM`, a compiled `program.bin` |
| `DmaScalarCoreDiagnosticTest.scala` | `atlas/scalar/` | none |

```sh
install -m 0644 test/rtl/HeadPadDiagnosticTest.scala \
  "$ATLAS_RTL_ROOT/src/test/scala/atlas/sa/HeadPadDiagnosticTest.scala"
cd "$ATLAS_RTL_ROOT"
ATLAS_MXU0_HEAD_PAD_FIXTURE="$WORK/mxu0-head-pad" \
  ./mill atlas.test.testOnly atlas.sa.HeadPadDiagnosticTest
```

`CompiledProgramAtlasCoreDiagnosticTest.scala` and
`CompiledProgramScalarCoreDiagnosticTest.scala` run whole compiled programs;
[native-merlin-binding.md](native-merlin-binding.md) covers them.

`tools/check_variant_inventory.py --rtl-root ... --model-root ...` checks the
inventory JSON against the pinned sources (bit patterns, decoder controls,
dialect op names, test names) and exits nonzero on drift. The CTest run does
the same when `ATLAS_RTL_ROOT` and `ATLAS_MODEL_ROOT` are set.

## Limits

- Every result above is a diagnostic run on the standalone `AtlasCore` or a
  single RTL block. Nothing has run on the full SoC or an FPGA.
- Most tests cover one register pair, one program layout and two input panels.
  Other registers, overlapping operations and in-place use are mostly untested.
- Test programs use fixed `DELAY` padding copied from public RTL examples
  (33 and 64 cycles after VPU ops, 256 for column reductions, 95 for MXU
  matmuls). That padding was enough; it is not a measured minimum latency.
- Unless a row says otherwise, subnormals, NaN, infinities and overflow were
  not checked for FP8 or BF16 inputs.
- No instruction mode is cleared for general compiler use yet: for all 99,
  the full input range, timing and integrated execution are still open.
