# Bounded selected-core checks for remaining conditional branches

The hand OOT test `test/test_branch_mode_reference.py` exercises BNE, BGE,
BLTU, and BGEU on the selected standalone AtlasCore. For each mode, three
operand pairs cover taken and untaken paths, including `-1` versus `1` to
distinguish signed and unsigned comparison, and equality for the relevant
boundary. Each program writes a marker in the first post-branch instruction,
another in the second, and a third at the target. The independent expected
result checks that the first marker is written in both cases, the second is
skipped only when taken, and the target marker is written. The test checks
typed emission against the selected independent assembler and unmodified LLVM
RISC-V object words before running the core. Eight invalid offset cases
(odd or outside the encoded range) are rejected. All 12 execution panels and
8 negative cases passed on 2026-10-04: 3 test methods, 0 failures, 0 skips.

Execution used RTL commit
`0079c0541111197741a231c002e3843fa6f545b2`, ModeLIR commit
`add52b0a7c96d72e0079b7938e07ca5080871e82`, and the previously built
selected-core `model.so` and `state.json` with SHA-256
`196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`
and `db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`.
The hand OOT emitter and optimizer binaries used SHA-256
`dab38e0f0186adf67e2b261ab0b1ad666e970c5eac81fe96cece635edf76a14c`
and `2f9a56a610f3e9f1864a0f03334f4b1f319b3c766f07941889cf9aad9834cbcb`.
The observed run used:

```sh
ATLAS_OOT_BIN_DIR="$oot_build/bin" ATLAS_LLVM_BIN="$llvm_bin" \
ATLAS_ASSEMBLER_ROOT="$atlas_rtl/baremetal" \
ATLAS_ARC_MODEL="$selected_core/model.so" \
ATLAS_ARC_STATE="$selected_core/state.json" \
ATLAS_MODELIR_ROOT="$modelir" ATLAS_RTL_ROOT="$atlas_rtl" \
ATLAS_BRANCH_RECEIPT=out/artifacts/branch-modes-r1/receipt.json \
PYTHONPATH=test OPENBLAS_NUM_THREADS=1 \
python3 -m unittest test_branch_mode_reference -v
```

The invocation-owned `out/artifacts/branch-modes-r1/receipt.json` records
every observed register triple, cycle count, test-source hash, model/state
hashes, and emitter hash. The seven-method combined branch/census run is
retained in `out/artifacts/branch-modes-r1/test.log` in the local worktree.

The source-bound inventory check passed with both selected source checkouts;
its result is 73/99 bounded modes and 0/99 software-admitted modes. This
single fixed branch displacement and the three operand panels do not qualify
the full branch displacement/register domain, arbitrary loops, temporal
interactions, or integrated SoC execution. Gate D remains blocked.
