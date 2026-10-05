# Bounded selected-core checks for the six CSR modes

The hand OOT test `test/test_csr_mode_reference.py` exercises CSRRW, CSRRS,
CSRRC, CSRRWI, CSRRSI, and CSRRCI on the selected standalone AtlasCore. It
initializes writable debug CSR `0xC10`, then checks the selected operation's
old-value readback, new debug CSR value, and untouched debug CSR `0xC11`.
Two directed base/mask panels per mode distinguish replacement, bit set, and
bit clear. Typed emission agrees with the selected assembler and unmodified
LLVM RISC-V object words. The selected CSRRCI word has funct3 `111`, rather
than the inspected model class's `100`. Twelve invalid address/source cases
and four writes to a read-only CSR are rejected by the dialect verifier.

The selected RTL's `ScalarCore.scala`, `ScalarDecoder.scala`, and
`CSRFile.scala` jointly define these effects. The source-bound inventory now
includes the CSR file for every CSR mode. The observed run used RTL commit
`0079c0541111197741a231c002e3843fa6f545b2`, ModeLIR commit
`add52b0a7c96d72e0079b7938e07ca5080871e82`, and the previously built
selected-core `model.so` and `state.json` with SHA-256
`196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`
and `db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`.

```sh
ATLAS_OOT_BIN_DIR="$oot_build/bin" ATLAS_LLVM_BIN="$llvm_bin" \
ATLAS_ASSEMBLER_ROOT="$atlas_rtl/baremetal" \
ATLAS_ARC_MODEL="$selected_core/model.so" \
ATLAS_ARC_STATE="$selected_core/state.json" \
ATLAS_MODELIR_ROOT="$modelir" ATLAS_RTL_ROOT="$atlas_rtl" \
ATLAS_MODEL_ROOT="$inspected_model" \
ATLAS_CSR_RECEIPT=out/artifacts/csr-modes-r1/receipt.json \
PYTHONPATH=test OPENBLAS_NUM_THREADS=1 \
python3 -m unittest test_csr_mode_reference test_variant_inventory -v
```

On 2026-10-04 this combined run passed 7 test methods, with 12 selected-core
programs, 12 assembler/LLVM comparisons, and 16 invalid constructions.
The local invocation-owned `out/artifacts/csr-modes-r1/receipt.json` records
every observed old/new value, sibling register, cycle count, and source/tool
hash. The source-bound census reports 79/99 bounded modes and 0/99
software-admitted modes. This is restricted to one writable CSR address,
small masks, and serial execution on a standalone core; full CSR address,
read-only, timing, and integrated SoC behavior remain unqualified. Gate D
remains blocked.
