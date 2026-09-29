# Scalar load to JALR: fixed delay, no completion wait

This is a bounded diagnostic on the selected standalone `AtlasCore`. It does
not authorize runtime-computed JALR targets in the LLVM conversion pass.

| Source | Exact revision |
| --- | --- |
| Selected Atlas RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Inspected software model | `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| ModeLIR standalone driver | `add52b0a7c96d72e0079b7938e07ca5080871e82` |
| Hand OOT starting point | `atlas-mlir` `55f0131a18a2ec9758d6251e9334bc2eeea5d56d` |

`ScalarCore.scala` documents a fixed three-cycle scalar load and says only
`DELAY` and `DMA.WAIT` stall the frontend. The actual stall expression is
`delay_stall || dma_wait_stall`; `DMA.WAIT` observes DMA channel busy, not
scalar LSU busy. `LSU.scala` exposes `scalarResp.valid` and `scalarBusy`
internally. `ScalarCore` captures the response through `memLoadPending`, then
`memLoadRespValid`, and writes the scalar register on a later cycle. There is
no decoded scalar-load completion wait. `FENCE` is encoded but does not appear
in the stall expression; the inspected software model also implements it as
a no-op. The model's synchronous `LW.exec` does not establish RTL timing.

The [14-word typed program](../test/examples/jalr_load_delay.mlir) writes
word-index target 11 to VMEM address zero, waits four cycles for that store,
sets `x1=8`, loads VMEM[0] into `x1`, then executes JALR at word 6. Word 7 is
its delay slot. Target 8 writes marker 2; target 11 writes marker 3. Both
paths write the marker to CSR `dbg0` and halt. The selected assembler and
`atlas-emit` agree on all 14 words. Parser/printer round-trip passes.

| Word 5 | JALR-fire `x1` | JALR-fire load state | Link `x10` | Final `dbg0` | Halt word |
| --- | ---: | --- | ---: | ---: | ---: |
| `DELAY 8` | 11 | pending=0, LSU response=0 | 7 | 3 | 13 |
| `DELAY 0` | 8 | pending=1, LSU response=1 | 7 | 2 | 10 |
| `FENCE` | 8 | pending=1, LSU response=1 | 7 | 2 | 10 |

All three selected-core executions halted with zero DMA traffic and preserved
two disjoint DRAM guards. The cycle trace placed the LSU response at observed
cycle 12, `memLoadRespValid` at 13, and `x1=11` at 14; the un-delayed JALR
fired at cycle 12 with stale `x1=8`. These cycle numbers are observations of
this standalone elaboration and program. `DELAY 8` provides enough time in
this closed test, but it is a fixed delay rather than a hardware completion
event. Bank contention, concurrent agents, integrated stalls, and other
execution modes are not covered.

The current LLVM pass rejects all three variants because it cannot prove the
memory-derived register target from its typed scalar prefix. The test checks
that `FENCE` and zero delay do not bypass this rejection. Safe general
lowering would need a qualified scalar-load availability rule and a checked
finite in-block target set or a runtime target guard. Neither exists in this
reference. The JALR inventory row remains unadmitted and the D gate remains
blocked.

Focused reproducer, with the existing CMake build and selected source paths
provided by the caller:

```sh
ATLAS_OOT_BIN_DIR="$build_dir/bin" \
ATLAS_ASSEMBLER_ROOT="$atlas_rtl/baremetal" \
ATLAS_RTL_ROOT="$atlas_rtl" \
ATLAS_ARC_MODEL="$arc_model" ATLAS_ARC_STATE="$arc_state" \
ATLAS_MODELIR_ROOT="$modelir_root" \
python -m unittest -v test.test_jalr_load_delay_reference
```

The focused selected-source-linked run passed 3/3 methods with no skips.
