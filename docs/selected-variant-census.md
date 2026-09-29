# Selected Atlas mode census for the hand OOT reference

The frozen ledger is [selected-variant-inventory.json](selected-variant-inventory.json).
It is an evidence index for the selected RTL commit, not a second executable
semantic specification. Every row names one decoder mode, the hand dialect
operation and mode attributes, selected RTL BitPat and decoder controls,
inspected model class, architectural source files, parameter domains, observed
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

## Denominators

These counts are at the **decoder semantic-mode** level, not at the level of
all address, register, scale-code, slot, or timing combinations. The checked
parameter domains are listed separately in the JSON ledger. The ledger's
`standalone_core_executed` flag points to a committed bounded test previously
run on the selected-source-linked standalone ARC model; a current source-only
run does not execute it.

| Family | Required modes | Represented and word-emitted | Bounded semantic and standalone-core test | Remaining full-qualification blockers |
| --- | ---: | ---: | ---: | ---: |
| Memory transfer | 2 | 2 | 0 | 2 |
| DMA | 4 | 4 | 2 | 4 |
| MXU0 | 7 | 7 | 2 | 7 |
| MXU1 | 7 | 7 | 1 | 7 |
| VPU arithmetic, reduction, pack | 25 | 25 | 8 | 25 |
| VLI | 4 | 4 | 4 | 4 |
| XLU | 1 | 1 | 1 | 1 |
| Scalar, control, CSR | 49 | 49 | 1 | 49 |
| **Total** | **99** | **99** | **19** | **99** |

All 99 modes are required for the selected source inventory. None has
`software_admitted=true` because a full-domain, reviewed semantic and temporal
contract has not been frozen. This does not mean the hardware lacks those
features. All 99 have typed representation and source-level word emission;
98 also have an LLVM inline-assembly word route. JALR is deliberately refused
by the single-block LLVM pass because its dynamic target cannot be checked
there. `blocked=99` means each mode still lacks some full D-gate evidence,
even when a bounded mode-specific test passed.

The independently checked bounded modes are DMA load/store, MXU0 reset and
continuation matmul, MXU1 reset matmul, VADD, VSUB, VMUL, VMAX, VMOV, VRELU,
row sum and row minimum, all four VLI modes, XLU transpose, and BEQ. The ledger links
each flag to a test method. These tests use restricted inputs, geometries,
programs, and the standalone core, so they cannot be promoted to complete
instruction semantics or integrated hardware coverage. Some supporting
transfer/CSR instructions appear in those programs but have no independent
mode-specific reference test; they remain at zero in the semantic numerator.

## Missing qualifications and source discrepancies

The main gaps are FP8 pack/unpack scale interpretation, MXU FP8 pop/seeded
accumulator variants, full-domain VLI qualification, many VPU unary and column-reduction
modes, scalar/CSR/control variants, complete reserved-field legality,
cross-family timing, and integrated execution. Model/RTL differences for
DMA config, CSRRCI, square/cube encodings, VMOV width, VADD rounding,
row-sum order, and MXU arithmetic are recorded per mode. Selective VLI modes
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
