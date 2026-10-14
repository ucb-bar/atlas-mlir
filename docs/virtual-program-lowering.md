# Virtual program lowering and handoff

`tools/lower_atlas_program.py` runs virtual verification, virtual scheduling,
register/VMEM/MXU/DMA placement, machine lowering, timing, and instruction
encoding. It accepts one bounded virtual function with the ABI attributes
described in the [dialect reference](dialect-reference.md).

From the repository root, after building:

```sh
python3 tools/lower_atlas_program.py \
  test/examples/virtual_attention_tile.mlir \
  --atlas-bin-dir build/bin \
  --output-dir out/attention
```

Use a new or empty output directory. `--timing schedule-atlas-stream` selects
physical scheduling instead of the default `insert-atlas-delays`.
`virtual_attention_tile_transposed_key.mlir` computes the key transpose on the
XLU; `virtual_attention_tile.mlir` expects the transposed key in DRAM.
`virtual_softmax_tile.mlir` exercises just the BF16 softmax path.

| Artifact | Contents |
| --- | --- |
| `1-virtual.mlir` | Verified virtual source |
| `2-scheduled.mlir` | Scheduled virtual source |
| `3-allocated.mlir` | Allocated, untimed machine IR and source contracts |
| `3-allocation.json` | Register pressure and placement, MXU slots, DMA channels, VMEM windows, untimed word count, and lowering diagnostics |
| `4-timed.mlir` | Checked machine IR with waits and delays |
| `5-words.hex` | One encoded instruction word per line |
| `5-map.json` | Instruction-to-operation map |
| `5-program.json` | Decoded fields, control flow, and final word count |
| `commands.txt` | Replayable commands, including working directories |
| `log.txt` | Command diagnostics and exit statuses |

The driver stops at the first failure and retains earlier artifacts and the
diagnostic. Lowering writes the allocation report after planning, including
failed allocations. Source verification failures occur before allocation and
are recorded in the log. Reports count BF16 registers in pairs, FP8 and scalar
registers individually, and VMEM in four-byte words. `used: null` means the
register coloring failed; `peak_live` includes the allocator's requirement
that results not overwrite operands. Fixed synchronous DMA channels and legacy
matmul slots are included alongside explicit resource handles.

The report's `fits` describes **untimed lowering**. Timing can still fail, and
the 2,048-word generated-program policy is checked again after delay insertion
and at emission. Use successful completion of the entire driver and the final
`5-program.json` word count for the emitted program's status.

## Combined attention + MLP status

The branch implements Jeremy's virtual-to-machine infrastructure, including
row sum/max/min, BF16 splats, FP8 transpose, and lowering for every admitted
unary and binary VPU kind. It includes standalone attention fixtures and the
artifact driver. PR #15 is already in the branch ancestry.

The upstream combined attention → BF16-to-FP8 conversion → MLP virtual program
is still a handoff dependency. These fixtures do not establish a PyTorch-derived
combined program, its resource fit, or its numerical correctness. Once that
program arrives, run the driver on it and retain the output directory and test
log for Nicolas. The upstream integration and Shuttle host ELF belong to the
other owner in the work split; model/RTL execution and hardware validation
belong to Nicolas.

The Python virtual evaluator supports `mov`, `relu`, `add`, `sub`, `mul`,
`exp2`, `recip`, BF16 row sum/min/max, raw BF16 splats and raw FP8 transpose.
Softmax and both attention fixtures have bounded numerical, guarded-memory
and scheduling regressions, including an asymmetric key-orientation check.
Compilation and contract tests do not replace numerical execution.
Two-input comparisons, surrounding-memory checks, and repeated kernel
invocation remain integration acceptance criteria.

## Validation

```sh
cmake --build build --parallel 2
PYTHONPATH=/path/to/npu_model ctest --test-dir build --output-on-failure
```

The CMake-selected Python interpreter needs the virtual evaluator dependencies,
including PyTorch and xDSL; the `npu_model` source root must be importable.
External LLVM, assembler, and selected-core tests require the paths listed in
the [README](../README.md#build-and-test). Skipped tests are not execution
evidence. Driver regression tests cover resource failures, final instruction
capacity, path quoting, command replay, and missing tools. Attention operation
tests cover accepted kinds, rejected fields, exact encoded fields, and contract
rejection after machine-field corruption.
