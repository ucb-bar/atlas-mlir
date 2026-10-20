# Compiler pipeline

The compiler takes one fixed-shape virtual Atlas function (SSA tiles, no
registers, no timing) and produces checked Atlas machine IR, encoded
instruction words, and an LLVM module that `llc` and `ld.lld` turn into an ELF.
Op and pass details are in the [dialect reference](dialect-reference.md).
Hardware facts refer to the pinned RTL (`ucb-bar/atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`).

## Stages

`atlas-opt --atlas-compile` runs steps 1 to 5 as one registered pipeline, with
every verifier; each step is also registered on its own
(`--atlas-compile-verify`, `-schedule`, `-spill`, `-lower`, `-time`). The native binding's optimized pipeline and the captured Linalg
build both use it, and `--atlas-compile --dump-pass-pipeline` prints its exact
passes. It takes one function with explicit `atlas.input_dram_base` and
`atlas.output_dram_base`.

| Step | Result | Passes in `--atlas-compile` |
| --- | --- | --- |
| 1. Verify | virtual IR, live MXU handles saved as tiles | `--spill-atlas-mxu-state`, `--verify-atlas-virtual-stream` |
| 2. Schedule | boundary tiles as explicit DMA, each block reordered | `--lower-atlas-virtual-boundary-to-dma`, `--schedule-atlas-virtual`, `--verify-atlas-virtual-stream` |
| 3. Spill | register pressure recovered through VMEM | `--spill-atlas-virtual`, `--verify-atlas-virtual-stream` |
| 4. Lower | untimed machine IR | `--lower-atlas-virtual-to-machine` |
| 5. Time | adds `atlas.dma_wait`, reorders each block, adds `atlas.delay` | `--insert-atlas-dma-waits`, `--verify-atlas-machine-stream`, `--verify-atlas-generated-schedule`, `--schedule-atlas-stream`, then `--verify-atlas-machine-stream`, `--verify-atlas-timing`, `--verify-atlas-generated-schedule` |
| 6. Emit | words, word map, program JSON | `atlas-emit`, `--map-json`, `--program-json` |
| 7. LLVM | one inline-asm word block | `--convert-atlas-to-llvm-calls`, `--finalize-atlas-llvm-calls`, `mlir-translate`, `llc`, `ld.lld` |

### Exposing boundary DMA

`--lower-atlas-virtual-boundary-to-dma` replaces boundary `virtual_input_fp8`,
`virtual_input_bf16` and `virtual_output_bf16` operations with explicit virtual
DMA loads and stores before `--schedule-atlas-virtual`. It resolves each address
as `atlas.input_dram_base + index * 2048` or `atlas.output_dram_base + index *
2048`, moves 1024 bytes for FP8 or 2048 for BF16, and preserves the physical
tile layout and state chain. It checks the original input/output DRAM windows
before removing the slot indices. The scheduler may then reorder independent
transfers and overlap them with PACK, including the next tile's loads with this
one's work; allocation and wait insertion assign staging windows, channels, and
waits.
The captured Linalg shell build enables this pass by default.

### Reusing immutable weights

Boundary conversion, virtual scheduling and virtual-to-machine lowering automatically reuse repeated FP8 weight inputs in single-block functions. The `--reuse-atlas-immutable-weights` pass exposes the same transformation for inspecting virtual IR. Eligible values have only matmul weight-operand or typed MXU weight-load uses; activation inputs and persistent MXU handles are unchanged.

Reuse requires the complete original boundary DRAM spans to be valid and every boundary or explicit DMA store in the function to be provably disjoint from the weight's 1,024-byte range. Explicit FP8 DMA loads additionally require direct constant addresses and sizes. Boundary slot indices and explicit DMA addresses have separate reuse identities, preserving the original ABI/index checks. The state chain bypasses each removed load, while its tensor users share the first immutable value. Only newly dead address/size constants belonging to removed DMA loads are erased.

The compiler first applies reuse to a clone and runs the actual allocator and allocation verifiers, accepting it only when the extended register lifetimes fit. Uncertain aliasing, CFG functions and allocation failures retain the source unchanged; inputs without reusable weights retain their emitted words. This optimization reduces DRAM reads across token tiles; MXU weight pushes remain under the existing scheduler. Tests in `test_immutable_weight_reuse.py` cover automatic pipelines, alias refusals, unchanged cases, typed handles, state rewiring and bounded-allocation fallback.

### Driver

[`tools/lower_atlas_program.py`](../tools/lower_atlas_program.py) runs steps 1
to 6 and keeps every intermediate file: it runs `--atlas-compile` as its five
registered stages (`--atlas-compile-verify`, `-schedule`, `-spill`, `-lower`,
`-time`), one command per artifact. From the repository root:

```sh
python3 tools/lower_atlas_program.py test/examples/virtual_attention_tile.mlir \
  --atlas-bin-dir build/bin --output-dir out/attention
```

The output directory must be new or empty.

| File | Contents |
| --- | --- |
| `1-virtual.mlir` | Source after MXU state spilling and verification |
| `2-scheduled.mlir`, `2-spilled.mlir` | After boundary DMA and scheduling, then after spilling |
| `3-allocated.mlir` | Untimed machine IR with resource contracts, before DMA waits |
| `3-allocation.json` | Register pressure and placement, MXU slots, DMA channels, VMEM windows, untimed word count, errors |
| `4-timed.mlir` | Machine IR with DMA waits, stream-scheduled, with delays |
| `5-words.hex`, `5-map.json`, `5-program.json` | Words, word map, program JSON |
| `commands.txt`, `log.txt` | Replayable commands and their diagnostics |

The driver stops at the first failure and keeps what it has. The allocation
report is written even on failure (`used: null` means coloring failed); it
counts BF16 registers in pairs and VMEM in 4-byte words. Its `fits` and
`untimed_words` are measured before DMA waits and delays, so use the final
count in `5-program.json` (limit 2,048 words). Other inputs:
`virtual_attention_tile_transposed_key.mlir` (Kᵀ on the XLU),
`virtual_softmax_tile.mlir`, and the
[captured attention and MLP example](../examples/handoff/captured_attention_mlp/README.md).

Build as in the [README](../README.md#build). Tests need
[`requirements-virtual-evaluator.txt`](../tools/requirements-virtual-evaluator.txt)
and `npu_model` on `PYTHONPATH`; driver tests are in
[`test_lower_atlas_program.py`](../test/test_lower_atlas_program.py).

## Virtual IR and the SSA interpreter

Virtual IR is plain MLIR SSA in `func.func` with `cf.br`/`cf.cond_br`. `%t1`
is a value, not register 1. Branches pass values to the target block's
arguments, which act as phi nodes:

```mlir
^entry:
  %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %v = "atlas.virtual_input_bf16"(%s) {index = 0 : i32}
      : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  cf.br ^merge(%s1, %v : !atlas.virtual_state, !atlas.virtual_bf16)
^merge(%state: !atlas.virtual_state, %tile: !atlas.virtual_bf16):
```

[`virtual_bf16_cfg.mlir`](../test/examples/virtual_bf16_cfg.mlir) has complete
diamond and loop examples.

| Value | Type | Meaning |
| --- | --- | --- |
| Tile | `!atlas.virtual_bf16`, `!atlas.virtual_fp8` | Immutable logical 32×32 tile |
| Scalar | `i1`, `i32` | From `arith.constant/addi/cmpi` or a runtime control argument |
| State token | `!atlas.virtual_state` | Orders I/O, DMA, and MXU effects; carries no data |
| MXU handles | `!atlas.virtual_mxu_weight<u>`, `!atlas.virtual_mxu_acc<u>` | A resident weight, or one accumulator version, on unit `u` |
| Scale | `!atlas.virtual_scale` | Raw E8M0 code `0..255` from `virtual_scale_constant` |

An interpreter needs an environment keyed by MLIR `Value` identity (not the
`%t1` spelling), external memory for inputs, outputs, and DRAM, and a program
counter `(block, op index)` with a step budget. Treat the state token as an
event counter and reject an op that gets a stale one. At a branch, evaluate
all outgoing operands first, then bind the target's arguments all at once;
binding one by one breaks a backedge that swaps two values. For
`cf.cond_br`, run only the taken edge.

MXU handles are identities, not slots. A reset starts a contraction,
accumulate and readout consume the current accumulator version, readout keeps
the weight. Explicit handle block arguments require `--spill-atlas-mxu-state`
before scheduling or allocation: completed accumulators become BF16 tile
edges and weights become FP8 tile edges. Lowered handles stay block-local.
Keep virtual and machine IR apart: never read a register number from a virtual value, never
use the production pack or allocator to compute expected results, and report
"unsupported" for a mode whose numerics are unknown.

### Reference evaluator

[`tools/atlas_virtual_evaluator.py`](../tools/atlas_virtual_evaluator.py)
implements this: `parse_program` (reparse after editing the IR),
`validate_inputs` (exactly the declared inputs, untaken paths included),
`evaluate(program, inputs, max_steps=10000)` (checks the whole program, then
runs the taken path; several returns are allowed), and `compare_results`
(output indices, raw BF16 bits, mapped bytes; unmapped is not zero).

A `Tile` holds 1,024 raw BF16 words or FP8 E4M3 bytes, a `Scalar` holds
unsigned bits (`-1` is `Scalar(32, 0xffffffff)`), and a `MemoryRegion` is a
non-overlapping byte snapshot of 32-bit DRAM. FP8 tiles are 1,024 row-major
bytes. A BF16 tile in DRAM puts element `(row, col)` at byte
`(col // 16) * 1024 + (row * 16 + col % 16) * 2`. Weights are `W[N,K]`, so a
matmul computes `A[M,K] @ W[N,K].T`.

| Op (`atlas.virtual_*`) | Evaluator support and numerics |
| --- | --- |
| `vpu_unary` | `mov`, `relu`, `exp`, `exp2`, `recip`; EXP2 and RECIP use the RTL's raw-bit tables |
| `vpu_binary` | `add`, `sub`, `mul`; ADD and SUB compute in FP32 and chop to BF16, MUL rounds to nearest even |
| `vpu_reduce` | `row_sum` (balanced FP32 tree, one BF16 rounding), `row_min`, `row_max`; each result fills its row |
| `vpu_splat` | Copies raw `bits` into all 1,024 elements |
| `pack_fp8` | Scale codes `122..128` |
| `xlu_transpose` | Swaps rows and columns of the raw FP8 bytes |
| `mxu_*` | MXU0 and MXU1 have different arithmetic; the unit is part of the result |
| `mxu_readout_fp8` | MXU converter, including special scale codes. VPU pack divides by the scale and MXU readout multiplies, so one cannot stand in for the other |
| `dma_*`, `input_*`, `output_bf16` | Sequential memory effects (next section) |

Lowering accepts every VPU kind the dialect defines; the evaluator runs only
the kinds above. Tests: `test_virtual_evaluator_arithmetic.py`,
`test_virtual_evaluator_attention.py`, and `test_virtual_evaluator_core.py`
(compares with the RTL core model; setup in [hardware status](hardware-status.md)).

## DMA in virtual IR

Virtual DMA is atomic: a load returns its tile and a store is done when it
returns. There are no channels, handles, or waits at this level.

```mlir
%io1, %q = "atlas.virtual_dma_load_fp8"(%io0, %addr, %size)
    : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
%io2 = "atlas.virtual_dma_store_bf16"(%io1, %y, %out, %bytes)
    : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
```

A load reads DRAM as of its place in the state chain and a store writes at
its place, like `virtual_input_*` and `virtual_output_bf16`; run them in
order, overlaps included. Transfers are whole tiles (1,024 bytes FP8, 2,048
BF16). Address and length must be built from `arith.constant` and
`arith.addi` without overflow flags; runtime addresses are not supported. The
address must be at least `0x80000000`, 32-byte aligned, end at or below 2³²,
and miss the control mailbox. Stores count as outputs and go in the single
return block. Loads must read initialized bytes, and the host must leave the
ranges alone while the program runs. See
[`virtual_dma_tiles.mlir`](../test/examples/virtual_dma_tiles.mlir).

The virtual scheduler costs a load as a one-cycle launch whose tile is ready
after the transfer and its VLOADs, so it fills the gap before the first
consumer with independent work. It models one serial DMA engine and at most 8
transfers in flight, and orders two DMA ops only when their DRAM ranges
overlap and one writes. Virtual DMA stays in order with boundary tiles, but
not with PACK. Registers do not limit the order: a value beyond them costs a
VMEM save and a reload per use, as `--spill-atlas-virtual` inserts them, and
an order is kept only when its estimated time, spills included, is lowest. A
loaded tile counts toward them from its first use.
Run the spill pass after scheduling. `--schedule-atlas-virtual=random-seed=N`
picks a random legal order for soundness tests.

Lowering gives each transfer one of 8 channels and one of 8 2 KiB staging
windows (lowest free first; if none is free, the oldest transfer completes
first), emits the launch at the op, and moves a load's staging VLOADs down to
its first use, a later launch given its window, or the block end. Until then
the tile is in VMEM, so its registers are allocated from there, not from the
launch: a load issued early holds a staging window but no register. A store
copies into staging, then launches.
`--insert-atlas-dma-waits` then adds `atlas.dma_wait` right before the first
instruction that needs the transfer done: a staging-window access or reuse, a
launch on the same channel, an overlapping DRAM access where either side
writes, a helper DMA, scalar memory or a branch or jump outside an analyzed
scalar loop, an analyzed loop that touches its staging window, a CSR access,
the halt, or entry to any other branch target. Transfers stay in flight through
PACK ([DMA across loops](#dma-across-loops)).

## Physical program output

`atlas-emit prog.mlir` verifies a flat machine stream (state chain, encoding,
delay slots, in-block targets) and prints one hex word per line.
`--program-json` writes `atlas.physical_program.v1` for a functional model.
Its root has `selected_rtl_revision`, `entry_word_index = 0`,
`pc_unit = "instruction_word"`, `word_endianness = "little"`, `word_count`,
`timing_scope`, `timing_state`/`timing_provider` when timed, and `instructions`.

Each instruction row has `word_index`, `word_u32`, `word_hex`, `operation`,
typed `fields`, and `control`. The word after a branch or jump also has
`delay_slot_for_word_index`; an `atlas.delay` row has `explicit_delay_cycles`.
The model executes `word_u32`; compare those with the loaded binary before
trusting the rest.

| `control.kind` | Extra fields |
| --- | --- |
| `sequential` | none |
| `conditional_direct` | `target_word_index`, `fallthrough_word_index`, `delay_slot_word_index` |
| `jump_direct` | `target_word_index`, `delay_slot_word_index` |
| `jump_register` | `base_register`, `offset_words`, `delay_slot_word_index` |
| `trap` | none (ECALL/EBREAK) |

The PC counts instruction words, not bytes. A direct branch or JAL targets
`pc + offset_bytes / 2`; JALR targets a register value plus a word offset.
Every branch and jump runs one delay-slot word before taking the redirect or
fall-through.

`--map-json` writes `atlas.machine_word_map.v1`: per word, its byte offset,
operation, attributes, stall cycles for a delay, direct branch target,
delay-slot owner, and `availability: "unknown"`.

## LLVM handoff

`--convert-atlas-to-llvm-calls` turns each checked instruction into a marker
call, `llvm.call @atlas_emit_<op>()`, with `atlas.fields`, `atlas.source_op`,
`atlas.word`, and `atlas.word_index` attributes. The markers are compiler IR,
not runtime functions. `--finalize-atlas-llvm-calls` rebuilds the Atlas stream from them, rechecks
every field, word, index, and control target, and replaces them with
`llvm.func @atlas_program()` holding one side-effecting
`llvm.inline_asm ".word 0x...\0A.word 0x..."` block and `llvm.return`.

```sh
build/bin/atlas-opt --verify-atlas-machine-stream --convert-atlas-to-llvm-calls \
  test/examples/handoff_mlp_tile.mlir > out/mlp.structured-llvm.mlir
build/bin/atlas-opt --finalize-atlas-llvm-calls \
  out/mlp.structured-llvm.mlir > out/mlp.encoded-llvm.mlir
mlir-translate --mlir-to-llvmir out/mlp.encoded-llvm.mlir > out/mlp.ll
```

A single block keeps LLVM from putting code between a branch and its delay
slot or moving branch targets; stock LLVM has no Atlas instructions and treats
the words as opaque. The result is a reset-entry program body, not a
C-callable function, and it does not run on a generic RISC-V CPU.

[`tools/export_llvm_handoff.py`](../tools/export_llvm_handoff.py) builds the
full bundle for the examples in the [handoff guide](../examples/handoff/README.md):
Atlas MLIR, both LLVM MLIR stages, LLVM IR, assembly, object, ELF (entry
`atlas_program` at 0), disassembly, word map, program JSON, and a manifest with
hashes. It checks that the object and linked `.text` begin with the
`atlas-emit` words; LLVM's trailing return is not an Atlas instruction.
[`test_handoff_examples.py`](../test/test_handoff_examples.py) regenerates and
checks these stages.

Do scheduling and annotation work on Atlas machine IR or the marker stage. A
pass that edits the marker stream must keep fields, words, and indexes
consistent (the finalizer rejects mismatches) and regenerate the word map.
Editing words inside the final inline-assembly string skips every check.

## Limits

- One fixed-shape function per module, 32×32 tiles, no general tiling.
  `--spill-atlas-virtual` handles FP8/BF16 and scalar values, block arguments and
  entry controls through shared VMEM temporaries. Mutable slots remain reserved
  for the invocation. Completed MXU state can be saved through raw tiles before
  slot reuse and across explicit CFG edges; normalized handles remain local. See
  [register spilling](register-spilling.md).
- Pack scale codes `122..128` only; the FP8 readout scale must be a constant.
- Virtual IR picks the MXU unit. After spilling, it must fit two weights and
  two accumulators per unit within a block.
- Outside analyzed scalar loops, a DMA transfer is always completed before a
  branch, a branch target, or the halt, even when overlap would be safe.
- Delays come from npu-model's `rtl-match` timing rules, not RTL
  measurements; a delay alone does not prove an MXU result or DMA is ready.
  The hand-written machine examples (`handoff_*_tile.mlir`) use hand-picked
  delays instead.
- The evaluator runs a subset of VPU kinds and is tested on finite,
  representable inputs, not every FP8/BF16 encoding.
- Results on the RTL standalone core are diagnostic. Nothing here has run on
  an integrated SoC or FPGA.

### DMA across loops

The shared [scalar-loop analysis](loop-dma-analysis.md) lets explicit DMA stay
in flight through single-block scalar loops with provable independent memory
accesses. Discovery uses control flow and instruction effects rather than PACK
helper annotations. Unknown or unsupported loops require conservative waits.
PACK retains its separate provenance and exact-layout contract checks.
