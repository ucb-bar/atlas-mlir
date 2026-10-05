# Selected Atlas mode census for the hand OOT reference

The frozen ledger is [selected-variant-inventory.json](selected-variant-inventory.json).
It is an evidence index for the selected RTL commit, not a second executable
semantic specification. Every row names one decoder mode, the hand dialect
operation and mode attributes, selected RTL BitPat and decoder controls,
inspected model class, architectural source files, parameter-domain candidates, observed
tests, source discrepancies, and unresolved obligations.

The selected source pins are `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2` and local
`npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`Instructions.scala` and `IDecode.scala` each have 99 selected rows. The
source-bound checker compares all 99 row names, exact 32-bit BitPat strings,
and all 17 decoder control fields per row. It also checks the source revisions,
tracked-file cleanliness, model class identities, dialect operation names,
and evidence test-method identities. The local operation fixtures must have
the same 99 names and declared mode attributes. Mutation tests show that a
changed pattern, decoder control, added or removed mode, or changed fixture
fails this check. None of these checks establishes runtime legality.

The 19 parameter domains record units, finite values or intervals, and selected
RTL source paths. Branch byte displacements, JAL byte displacements, and JALR
word offsets are distinct domains because their encodings and units differ.
The source-bound check requires the cited files at the pinned RTL revision.
All 19 remain `reviewed=false`: they are inputs to a legality review, not
admitted full-domain contracts. FP8 and BF16 register indices refer to one
physical MREG file. The E8M0 scale-register and code ranges do not establish
a model quantization policy.

## Denominators

These counts are at the **decoder semantic-mode** level, not at the level of
all address, register, scale-code, slot, or timing combinations. The checked
parameter domains are listed separately in the JSON ledger. The ledger's
`standalone_core_executed` flag points to a committed bounded test previously
run on the selected-source-linked standalone ARC model; a current source-only
run does not execute it.

| Family | Required modes | Represented and word-emitted | Bounded semantic and standalone-core test | Remaining full-qualification blockers |
| --- | ---: | ---: | ---: | ---: |
| Memory transfer | 2 | 2 | 2 | 2 |
| DMA | 4 | 4 | 3 | 4 |
| MXU0 | 7 | 7 | 2 | 7 |
| MXU1 | 7 | 7 | 2 | 7 |
| VPU arithmetic, reduction, pack | 25 | 25 | 20 | 25 |
| VLI | 4 | 4 | 4 | 4 |
| XLU | 1 | 1 | 1 | 1 |
| Scalar, control, CSR | 49 | 49 | 45 | 49 |
| **Total** | **99** | **99** | **79** | **99** |

All 99 modes are required for the selected source inventory. None has
`software_admitted=true` because a full-domain, reviewed semantic and temporal
contract has not been frozen. This does not mean the hardware lacks those
features. All 99 have typed representation, source-level word emission, and
at least one LLVM inline-assembly word route. JALR's LLVM route requires a
proven in-block word-index target from a restricted straight-line scalar
prefix; unresolved dynamic targets remain rejected. `blocked=99` means each mode still lacks some full D-gate evidence,
even when a bounded mode-specific test passed.

The executable compatibility direction for BF16 column sum, minimum, and maximum
is now the [selected RTL's 64-by-16 physical reduction](selected-rtl-column-reduction-contract.md).
This resolves which layout the compiler should target for this revision; it
does not change the software-admission or blocked counts above.

The independently checked bounded modes are VLOAD/VSTORE raw-byte transfer,
DMA load/store/wait, MXU0 reset and
continuation matmul, MXU1 reset and continuation matmul, VADD, VSUB, VMUL, VMIN, VMAX, VMOV,
VRELU, VRECIP, VEXP2, VSQRT, VSQUARE, VCUBE, column sum, minimum and maximum, row sum, row minimum and row maximum, all four VLI
modes, BF16/FP8 pack and unpack, XLU transpose, BEQ, BNE, BLT, BGE, BLTU,
BGEU, bounded JAL, bounded JALR, bounded DELAY, bounded FENCE, and bounded
LB/LBU/LH/LHU/LW/SB/SH/SW, plus all six CSR read/modify/write modes. The ledger links
each flag to a test method. These tests use restricted inputs, geometries,
programs, and the standalone core, so they cannot be promoted to complete
instruction semantics or integrated hardware coverage. Some supporting
transfer/CSR instructions appear in those programs but have no independent
mode-specific reference test; they remain at zero in the semantic numerator.
The four newly checked conditional modes use signed/unsigned discriminator
panels, taken and untaken paths, and a one-slot side effect. See the
[bounded branch observation](branch-mode-observation.md).
The [CSR mode observation](csr-mode-observation.md) checks old-value readback,
new debug-register contents, untouched sibling state, and the selected
CSRRCI encoding. It does not qualify untested addresses or timing.
The VLOAD/VSTORE test checks three full 1 KiB raw-byte panels copied through
one MREG into two separate VMEM windows, with both DRAM outputs, original
input, and a guard checked. See [the observation](vload-vstore-observation.md).
A [direct JAL observation](jal-direct-target-observation.md) checks a byte-
displacement target interpreted in instruction-word space, link, one delay
slot, two skipped words, and a changed-target discriminator. It does not
qualify general control flow or a callable ABI.
A [DELAY observation](delay-timing-observation.md) checks counter values
0 and 4 against selected-core cycle traces and CSR state. It does not
qualify interactions with asynchronous resources or the full count range.
A [FENCE observation](fence-no-wait-observation.md) checks one-step scalar
control flow and ties a typed FENCE word to the prior scalar-load no-wait
discriminator. It does not establish general memory ordering.
All 19 scalar register and immediate ALU modes now have a
[bounded selected-core check](../test/test_scalar_alu_reference.py): two directed
operand panels per mode, typed/selected-assembler/LLVM word agreement, and an
independent RV32 arithmetic result. This does not cover their full operand,
register, or timing domains.
LUI and AUIPC also have [bounded selected-core checks](scalar-upper-pc-observation.md).
AUIPC adds the instruction word index on the selected RTL, exposing a PC-unit
discrepancy with the inspected software model's byte-PC formulation.
The [scalar memory observation](scalar-memory-halt-observation.md) checks
signed/unsigned byte and halfword reads, word reads, and masked byte/halfword
stores against two directed panels. Its negative program exposes an ECALL halt
before a pending scalar load can write back when ECALL immediately follows DELAY.

## Missing qualifications and source discrepancies

The main gaps are full-domain FP8 pack/unpack conversion and MXU FP8 pop/seeded
accumulator variants, full-domain VLI qualification, remaining VPU unary and column-reduction
modes, scalar/CSR/control variants, complete reserved-field legality,
cross-family timing, and integrated execution. Model/RTL differences for
DMA config, CSRRCI, square/cube encodings, VMOV width, VADD rounding,
row-sum order, column-minimum/maximum layout, pack/unpack physical row order, and MXU arithmetic are recorded per mode. Selective VLI modes
now have a [bounded raw-bit standalone-core test](vli-selective-observation.md):
the selected RTL writes a raw immediate while the inspected model routine
numerically assigns that integer to a BF16 tensor before a bit view. See
[source discrepancies](source-discrepancies.md)
for context. No model class or BitPat is treated as an independent hardware
oracle.

Run the source-bound check with exact clean source checkouts:

```sh
python3 tools/check_variant_inventory.py \
  --rtl-root /path/to/selected/atlas-npu \
  --model-root /path/to/selected/npu_model-atlas
```

It prints machine-readable status and counts and exits nonzero on drift. The
CTest method performs the source-bound check when `ATLAS_RTL_ROOT` and
`ATLAS_MODEL_ROOT` are set; its portable mutation/fixture checks run without
those roots. No ARC job is needed for the census check.
