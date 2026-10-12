# RTL timing for Atlas

Per-operation engine timing is computed from the Atlas RTL rather than typed into the compiler. A C++ exporter turns CIRCT HW IR into JSON, a Python simulator executes each engine's control logic cycle by cycle, and the resulting facts (`merlin.op_timing.v1`) become the `rtl-op-timing-v1` timing provider that drives scheduling, delay insertion and final verification in the compiler. The compiler supplies instruction structure; the facts supply the numbers.

![Chisel and SystemVerilog converge in CIRCT core dialects](figures/chisel_and_sv_circt_convergence.svg)

## Pipeline

1. **Retain.** Lower the selected FIRRTL to HW IR with `firtool`.
2. **Extract.** `tools/extract-rtl-timing.py` exports the engine modules and evaluates the per-engine specs in `tools/rtl_extract/targets/atlas/`. Ports come from bundle naming, memory response latency is measured on the memory module, and command codes come from ISA words through the RTL decoder. See the [extractor reference](../../tools/rtl_extract/README.md).
3. **Select.** `--select-atlas-rtl-evidence` loads the facts file by SHA-256 and records the selection as `atlas.rtl_evidence` on the module. The selection makes `rtl-op-timing-v1` the module's timing provider: scheduling, delay insertion, `--verify-atlas-timing`, `atlas-emit` and the LLVM handoff take footprints, dependences, DMA rules and reservations from it (see [the provider](#the-provider)).
4. **Check.** The extractor tests, a compiler cross-check and end-to-end simulations validate the facts; see [verification](verification.md).

Without a selection the compiler's built-in rules (`npu-model-rtl-match-v1`) apply and behavior is unchanged.

## Reproduce

Lower the selected FIRRTL with the options of `tools/retain-ee290-hw-ir.py`, adding `-O=debug --preserve-values=named` and dropping `--repl-seq-mem` and `--repl-seq-mem-file`. The default lowering removes dead ports and wire names and turns on-chip SRAMs into external blackboxes; the extractor would report such records as unresolved. With inline memories, VMEM and MREG appear as `seq.firmem 1, 1, undefined`.

Build the exporter and extract (details in the [extractor reference](../../tools/rtl_extract/README.md#build-and-run)):

```sh
python3 tools/extract-rtl-timing.py --hw-ir ee290.debug.hw.mlir \
  --exporter build/rtl-extract/exporter/hw_ir_export --target atlas --output facts.json
```

Select the facts in the compiler, then schedule and verify:

```sh
atlas-opt prog.mlir \
  --select-atlas-rtl-evidence="op-timing=facts.json op-timing-sha256=$(sha256sum facts.json | cut -d' ' -f1) dma=wait" \
  --schedule-atlas-stream --verify-atlas-timing -o final.mlir
atlas-emit --rtl-timing-json final.mlir > resolved.json
```

`--insert-atlas-delays` replaces `--schedule-atlas-stream` for delay insertion only. `dma=wait` is optional and enables DMA (see below). `atlas-emit --rtl-timing-json` is a mode of its own: it fails without output if the stream is untimed or unsafe, uses an unsupported operation, or lacks a valid selection; it reports the resolved footprints of the final instruction stream. A virtual program is verified, scheduled and lowered untimed first (`--verify-atlas-virtual-stream --schedule-atlas-virtual --lower-atlas-virtual-to-machine`), then selected and timed the same way; selecting before lowering also works, since the lowering carries the selection.

## The provider

`rtl-op-timing-v1` (`lib/AtlasRTLSelection.cpp`) is a complete [timing provider](../dialect-reference.md#generated-artifact-checks) registered in the compiler library beside the model. The timing passes choose a provider by `provider=<id>`, else the retained `atlas.timing_provider`, else `rtl-op-timing-v1` when the module carries `atlas.rtl_evidence`, else the model; the model is refused on a module with selected evidence. The provider resolves every footprint from the facts (`lib/AtlasRTLEvidence.cpp`), orders work by the data dependences the footprints imply, queues DMA transfers under the rules below, reserves every unit exclusively, and bounds the program at 32,768 words and 1,000,000 issue cycles. Its scope admits transfers pending one per channel in at most eight command slots, each waited on its own channel within its block, blocks that start idle and reach ECALL, and `ADDI`/`LUI` delay slots; a DELAY immediately before a branch, jump or halt is rejected, so the timing passes end such stalls on a NOP. The common verifier (`--verify-atlas-timing`, also run by `atlas-emit` and the LLVM passes) checks the final stream under it; a stream that carries a selection but no timing metadata is checked at those boundaries too, while `atlas-emit --allow-untimed` inspects an untimed lowering without a timing claim. `--schedule-atlas-virtual` costs virtual operations from the same footprints and reservations on canonical operands when a selection is present or `provider=rtl-op-timing-v1` is given; DMA transfers keep the model's transfer estimate, since completion is not a fact.

## Scope and assumptions

- Each record describes one operation issued to an idle engine. Overlap between operations is admitted only where assertion-checked pair sweeps show it (see below); control flow and DMA lifetimes follow the compiler's conservative rules.
- Engine admission comes from facts-derived unit holds. VLOAD and VSTORE each hold only their own path, so they overlap; same-bank order comes from the VMEM bank windows (load→store 30, store→load 34), and MREG port and release rules are unchanged. A VPU admits the next VPU command at `next_issue_age` and the XLU at `first_free_age`; the VPU and XLU exclude each other and the MXU, but overlap VLS subject to MREG port (`reg % 32`) and release rules. The pair sweeps behind each rule run on AtlasCore with all 177 RTL assertions checked (all 841 ordered VPU pairs stay at or below the first operation's `next_issue_age`), and `test/test_rtl_gap_coverage.py` requires every compiler gap to be at least the swept minimum.
- On-chip VMEM and MREG reads take one cycle: the IR declares them `seq.firmem 1, 1`, and the LSU is always granted ahead of DMA and TileLink. Records report this as `scratchpad_read_latency`.
- External memory and DMA completion are not modeled; they are variable and governed by `dma.wait`.
- Registers without a reset value are accepted only when listed in a spec's `unreset`; every control output must still be known at every age.
- Ages are absolute from the first cycle after reset and flush. Hold ends are inclusive: a last occupied age of 34 means the next VLS issues at age 35.
- The compiler supports VLS, XLU, all VPU operations, MXU commands on both units (weight and accumulator push, matmul with or without accumulation, FP8/BF16 pop), scalar `lw`/`seld`/`sw`, `addi`/`lui`/`delay`/`ecall`/marker, and DMA under `dma=wait`.
- Each MXU command holds both MXUs and the VPU, XLU and VLS paths through its last event, so MXU work is not overlapped. A matmul keeps its weight slot until its last accumulator write, since weight reads are not extracted.
- Branches, loops and joins are accepted when every engine is idle and no DMA is pending at each block boundary; each block is verified from an idle state with register values merged over all paths. Branches read their registers at issue and a taken branch executes only its delay slot (ADDI or LUI). Addresses that change between loop iterations are unknown at the loop header and rejected. `--rtl-timing-json` gives each instruction's block and block-relative issue cycle, and each block's successors and earliest successor issue.
- Scalar ALU operations (including the `xor`, `slt`, `sltu`, `xori` and `sltiu` the virtual lowering emits), `seli`, `lw`/`seld`/`sw` with known addresses, `csrrw x0, 0xC10` and `ecall` are structural: they read and write at issue with no facts numbers. A VPU operation's register operands must lie in distinct physical banks (register mod 32); the allocator may place two BF16 pairs sixteen colors apart in one bank, which the provider rejects by name rather than repairing.

### DMA

A DMA launch captures its scalar operands and the global DRAM base, so scalar registers and the base (set synchronously by `dma_config`) may be reused immediately. The transfer's VMEM and DRAM effects stay in flight until a `dma.wait` on the same channel, and a completion marker, halt or block end requires every transfer to have been waited; empty, wrong-channel and stale waits are rejected, and a delay cannot replace a wait. Under `dma=wait` transfers queue: the DMA engine serves launches in launch order from eight command slots indexed by launch count, so launch *n* of a block requires launch *n*−8 waited, and a channel's busy bit clears when its slot retires, so a channel may not be relaunched before its wait (the RTL protects neither). Pending VMEM windows stay disjoint from each other and from every VLS tile; DRAM spans may be shared only by transfers that both read; a VLS, VPU or XLU command beside a pending transfer keeps its isolated timing (the LSU has priority and only delays the transfer's VMEM accesses), while an MXU command waits for every transfer, since that combination was not swept. These rules come from the Verilator sweeps recorded in [verification](verification.md). A virtual schedule that puts MXU work beside a pending transfer stays on the model provider (`test/test_rtl_virtual_timing.py`). Issue coordinates across a wait are lower bounds; the resolved export reports wait-relative issue epochs and marks completion ages unknown.

Evidence for concurrent transfers (`system/core_dma.yaml`, `pairs/core_dma.yaml`, all 177 assertions, plus hand-scheduled Verilator streams): a launch on another channel is queued, never dropped, and the engine issues requests in launch order; each channel completes and releases its wait independently; VLS, VPU and XLU work keeps its isolated timing while a transfer is pending, and a VLS on the same bank only delays the transfer's VMEM accesses. The RTL checks neither DMA/VLS address overlap nor reuse of a busy channel (its busy flag clears when the first transfer's slot retires), and its 8 command slots are indexed by launch count, so those stay compiler rules. The compiler on this branch still allows one pending transfer.

## Results

Computed from the debug lowering of the selected FIRRTL; all equal the compiler's built-in rules. With the relaxed admission rules, a pipelined load-load-add-store tile drops from 1120 to 916 cycles with delay insertion (1112 to 704 scheduled), a four-tile copy from 284 to 182 (282 to 178), and load-transpose-store from 480 to 344 (546 to 337).

| Operation | Reads | Writes | Next issue / free |
| --- | --- | --- | --- |
| `vload` / `vstore` | 1-32 | 3-34 | 35 |
| `vtrpose.xlu` | 1-32 | 34-65 | 66 |
| VPU elementwise (incl. `vmul.bf16`) | 0-63 | 2-65 | 65 |
| VPU row sum / row max, min | 0-31 | 7-38 / 2-33 | 38 / 33 |
| VPU column reductions | 0-127 | 66-129 | 129 |
| VPU pack / unpack | 0-63 / 0-31 | 3-65 (step 2) / 3-66 | 65 / 66 |
| VPU immediate pair / single | - | 1-64 / 1-32 | 64 / 32 |
| MXU0 / MXU1 matmul | 0-31 | accumulator 63-94 / 3-34 | - |
| Scalar load | 1 | 3 | 3 |

The extractor also reports MXU push/pop streams and busy windows, response-latency and operand variants, and an MXU1 accumulate boundary case.

## Not covered

- Same-MXU pipelining. The RTL allows, for example, MXU1 matmul→matmul 32 cycles apart, but some gaps (such as MXU0 matmul→weight push at 63) are not single-operation facts. Overlapping MXU commands needs pair results as a selected input and holds that may overlap within one MXU.
- Loop-varying addresses (per-state register enumeration is the next step) and engine work crossing block boundaries.
- VPU two-slot overlap: 402 of the 841 VPU pairs (a single-input first operation followed by a single-input operation sharing no lane logic) are safe at gap 1, but the compiler still waits for `next_issue_age`; cross-engine overlap with the MXU, and DMA overlap with other engines or more than one pending transfer.
- The virtual scheduler's unit model stays its own: the provider supplies each operation's latency and same-unit interval, not cross-engine exclusion or DMA completion.
- A VLOAD writing an MREG bank another engine is still reading is not asserted by the RTL; the compiler's release rules order it.
- Merlin consumption: a draft that merges `op_timing` blocks into Merlin's RTL facts is unreviewed, and no Phase 1/2 consumer reads them yet.
