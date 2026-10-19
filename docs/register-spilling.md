# Register spilling and VMEM temporaries

`--spill-atlas-virtual` handles MXU weights and accumulators, FP8/BF16 tile
values and `i1`/`i32` controls. Tensor and scalar values may pass through
branches and loops. Explicit MXU handle block arguments are converted to raw
tile edges before allocation. Run MXU lifetime
splitting before virtual scheduling, then tensor/scalar spilling before
physical allocation. It uses the existing register allocator, wait insertion
and physical timing passes.

```sh
mkdir -p out
build/bin/atlas-opt input.mlir \
  --spill-atlas-mxu-state --schedule-atlas-virtual --spill-atlas-virtual \
  --lower-atlas-virtual-to-machine --insert-atlas-dma-waits \
  --insert-atlas-delays --verify-atlas-generated-schedule \
  --verify-atlas-timing -o out/machine.mlir
build/bin/atlas-emit out/machine.mlir > out/words.hex
```

The native Merlin binding enables spilling in both compilation pipelines.
The stage-preserving driver enables it with `--spill` and retains the rewritten
virtual IR in `2-spilled.mlir`:

```sh
python tools/lower_atlas_program.py input.mlir --spill --output-dir out/spilled
```

## Scheduling integration

For programs that need spilling, use this order:

1. `--spill-atlas-mxu-state`: turn live MXU handles into saved tiles before
   scheduling or strict virtual-stream verification.
2. `--lower-atlas-virtual-boundary-to-dma`: expose boundary transfers when the
   scheduling pipeline uses explicit DMA.
3. `--schedule-atlas-virtual`: schedule the normalized virtual program.
4. `--spill-atlas-virtual`: recover tensor/scalar pressure and place VMEM slots.
5. `--lower-atlas-virtual-to-machine`: assign physical resources and lower.
6. Insert DMA waits and delays, then run the generated-schedule, machine-stream
   and timing verifiers before emission.

Spilling can add operations after virtual scheduling. Final waits and timing
checks must therefore run on the rewritten machine program. A pre-spill
schedule is not sufficient evidence that the final program is safe.

The scheduler does not keep blocks within the registers. It estimates each
value beyond them as one save plus a reload per remaining use of the live
value used last in source order, which stands in for the longest-lived value
this pass picks, and keeps the order whose estimated time, spills included, is
lowest. A scheduled order may therefore need spills the source order did not,
so step 4 must follow step 3.

The scheduler recognizes spill traffic conservatively through the helpers in
`include/Atlas/AtlasSpilling.h`. Slot declarations emit no instructions; saves,
restores and copies have memory effects and follow the state chain. Scheduling
changes should preserve those dependencies, account for added loads/stores,
and keep the source-to-machine records used by the verifiers. The current
spill costs are estimates, not hardware latency guarantees.

## Storage operations

| Operation | Meaning |
| --- | --- |
| `atlas.virtual_slot` | Declare invocation-local storage; allocation does not initialize it. |
| `atlas.virtual_save` | Store a typed scalar or tile in a slot. |
| `atlas.virtual_restore` | Read an initialized slot into a new SSA value. |
| `atlas.virtual_slot_copy` | Copy between distinct slots of the same format and extent. |
| `atlas.virtual_control` | Read an indexed entry control from the input buffer. Used by the memory control ABI below. |
| `atlas.virtual_spill_bf16`, `atlas.virtual_spill_fp8` | Save a tile and return the identity of that particular writer. |
| `atlas.virtual_reload_bf16`, `atlas.virtual_reload_fp8` | Reload the matching writer without changing its bits. |

Slot declarations have these attributes:

| Attribute | Values and units |
| --- | --- |
| `kind` | `bf16`, `fp8`, `i1`, `i32`, or `raw` |
| `words` | Four-byte words: 512 for BF16, 256 for FP8, one for a scalar; raw extents are positive multiples of 256. |
| `purpose` | `spill`, `edge`, or `scratch` |
| `slot_word` | VMEM word address; `-1` requests placement. |
| `pinned` | Optional boolean, default false. Keep an explicitly supplied address. |

A slot uses `!atlas.virtual_spill`, as do the immutable saved-writer identities.
Access verifiers distinguish them by their defining operation. A slot must be
declared in the entry block. Saves and restores must match its scalar or tile
type; raw slots support reservation and copies. Slot references cannot be passed
as block arguments.

Declarations have an allocation effect. Saves, restores and copies participate
in the virtual state chain and act as scheduling barriers. A restore requires
initialization on every reaching path. Immutable reloads instead require their
particular matching writer to dominate them.

## Shared VMEM placement

Spills, edge-copy temporaries and declared scratch slots use one placement
routine. Pinned allocations are placed first; the remaining allocations use the
first aligned available range. Overlap and exhaustion are errors.

The selected layout provides VMEM words `[135168, 196608)`, or bytes
`[540672, 786432)`: 240 KiB after the eight DMA staging windows in bank 2.
The existing boundary buffers, PACK scratch and DMA windows retain their own
reserved locations. This allocator does not move those buffers or assume that
unused DMA windows are available as scratch.

Scalar slots occupy four bytes. Tile and raw slots are aligned to 256 words
(1 KiB). Mutable slots are reserved for the invocation, including across
backedges. Immutable single-block saves may reuse storage after their last
reload. CFG lifetime coalescing is not implemented.

This covers contiguous device temporaries in the selected layout. It is not a
DRAM allocator or an allocator for arbitrary strided/aliased buffers. Merlin's
application buffer planning remains separate from device register and VMEM
placement.

## MXU state

At a branch or loop edge, `--spill-atlas-mxu-state` accepts explicitly passed
weight and accumulator handles. Each live accumulator must appear exactly once
on each successor edge. The two arms of a conditional branch may forward the
same current handle, since only one arm executes. Stale versions, duplicate
accumulators on one edge, dropped live accumulators and implicit cross-block
captures are errors.

The pass reads out each outgoing completed accumulator once as BF16, sends
that tile through the edge, and loads a new local handle at the destination.
Weights travel as their original FP8 tiles. The existing tensor spill and
allocation passes then place those values, including parallel copies on loop
backedges. This is conservative boundary materialization; it does not keep
physical MXU slots resident across blocks or snapshot an in-flight operation.
Direct virtual verification and lowering require this normalization first.

Each selected MXU has two weight slots and two accumulator slots. When a new
handle would exceed that capacity, the pass evicts the resident value whose
next use is farthest away. Ties follow source order. It never evicts an operand
needed by the current instruction.

For an accumulator, the rewrite inserts BF16 readout, a raw VMEM save, and a
reload followed by `virtual_mxu_load_acc_bf16` before its next use. A continuing
matmul consumes the restored handle. For a weight, it saves the original FP8
tile and later reloads it with `virtual_mxu_load_weight`; weights are immutable,
so subsequent evictions can reuse the saved copy. These temporaries use the
same VMEM placement as ordinary spills and scratch.

The selected RTL's `MxuParams` declares BF16 accumulator-buffer elements on
both units. `SystolicArraySequencer` and `InnerProductTreesSequencer` copy the
low and high BF16 halves directly in their `PushAccBF16` and `PopAccBF16` paths.
This is a completed-instruction save, not a snapshot of an active pipeline.
FP8 readout is never used to save accumulator state. Existing timing and
ownership checks enforce compute completion before readout and storage
completion before reuse.

`--spill-atlas-mxu-state` exposes the rewrite before scheduling. The combined
`--spill-atlas-virtual` pass also performs it for unscheduled input. Its
preflight defers only physical slot-capacity checks; stale versions, nonlinear
state chains, wrong units and live handles at block exit still fail. The usual
strict verifier runs again after rewriting and placement. The native optimized
pipeline and the stage-preserving driver invoke this step before scheduling.

`atlas.spill_report` adds `mxu_acc_saves` and `mxu_restores`. The first counts
BF16 accumulator snapshots; the second counts weight and accumulator reloads
into MXU slots. Tile saves and reloads remain in the existing totals. These are
logical operation counts, not cycles or encoded instruction counts.

After normalization, every MXU handle is local to its block. Source handles
must cross edges through explicit block arguments so that the pass can insert
readout and restore operations; implicit cross-block captures are rejected.
The pass does not snapshot an active matmul pipeline or change arithmetic or
reduction order.

## Pressure recovery and branch edges

On an allocation failure, the pass splits long-lived operation results with
saves and per-consumer reloads, places the slots, and retries allocation.
Reload results are not recursively spilled. Proven wrapping-i32 constants are
rematerialized at their uses; this retains the static address/size proofs
required by the existing DMA lowering.

For block arguments under pressure, incoming edges write slots and consumers
reload them. Edge copies are ordered so a destination cannot overwrite an unread
source. A cycle such as `(a, b) -> (b, a)` uses one temporary slot per format.
Copies use the reserved tensor register 62 or scalar register `x27`; ordinary
values keep the existing allocation pools. FP8 block arguments always use this
memory transport because the selected lowering has no FP8 register edge move.
Run the spill pass before lowering programs with FP8 block arguments.

State changes stay within the corresponding block. Edge splitting preserves the
branch condition and the incoming value order. A loop executes its saves and
copies on each visit; spilling does not change its iteration count. Blocks are
laid out in reverse postorder when needed for source-to-instruction records.

### Entry controls

When entry controls exceed the scalar register capacity, the pass switches that
function to `atlas.memory_controls`. The original signature and input order stay
intact. `atlas.virtual_control` reads each used argument by index and saves it
before tensor input DMA can reuse the control buffer. Subsequent uses reload
its slot. Boolean control loads mask the low bit.

The ABI still reads a 1 KiB buffer at `atlas.control_dram_base`, with up to 256
four-byte controls in signature order. Pad the buffer to 1 KiB. In emitted IR,
`atlas.scalar_arg_regs` uses `-1` for memory-resident arguments and
`atlas.control_argument_types` records their original bit widths. Small
functions retain the existing register control ABI.

## Bounds and checks

`max-spills` bounds rewrites (default 256). `vmem-words` bounds the shared pool;
zero selects the full selected pool. For example:

```sh
build/bin/atlas-opt --spill-atlas-virtual='max-spills=32 vmem-words=4096' input.mlir
```

`atlas.spill_report` records logical saves, reloads, slot copies, rematerialized
constants and the pool high-water mark. The legacy `frame_words_used` and
`frame_words_available` field names are retained. A logical BF16 save emits two
physical stores; these counts are not instruction counts. Use the final encoded
image for instruction size, including waits and delays.

Allocation is checked again after rewriting. Source and physical checks cover
slot formats, initialization, overlaps, scalar byte addresses, vector word
addresses, both BF16 halves, loaded copy bytes and control indices. Arbitrary
raw readbacks remain invalid. A timed image must still satisfy the existing
2,048-word compiler policy limit; spilling does not increase that limit.

The heuristic can fail for a feasible program. Work-limit, storage and program
size failures remain explicit compilation failures. MXU spilling preserves
completed BF16 buffer state on the selected RTL; it does not preserve
implementation state inside an active matmul.

## Tests

```sh
ATLAS_OOT_BIN_DIR=build/bin PYTHONPATH=.:test python -m unittest -v \
  test_virtual_spilling test_spill_resources test_mxu_spilling
ctest --test-dir build --output-on-failure
```

The focused tests cover tensor/scalar pressure, loop rotations, both branch
outcomes, entry controls, pinned scratch, rematerialization, repeated invocation,
input/output guards and size limits. The small byte/control executor treats DMA
as atomic and is not a numerical or timing simulator. The existing Python
virtual evaluator consumes pre-spill IR; it is not a production fallback.

The optional model test uses Python 3.14, Torch, and `npu_model` revision
`337e9f249f938a1dff007ddf8a763e5cd4b3aa2b`. With a Python environment that already
has the model's dependencies, install this test-only package separately:

```sh
uv pip install --python /path/to/python3.14 --no-deps --target out/model-deps \
  https://github.com/ucb-ee194-tapeout/npu_model/archive/337e9f249f938a1dff007ddf8a763e5cd4b3aa2b.zip
ATLAS_RUN_NPU_MODEL=1 ATLAS_OOT_BIN_DIR=build/bin \
  PYTHONPATH=out/model-deps:.:test /path/to/python3.14 -m unittest -v \
  test_spill_model_execution
```

It independently decodes emitted words using the selected RTL inventory and
requires each model instruction object to re-encode identically. It then runs
seven programs with VMEM noise and checks outputs, inputs, guards, halt and runtime
errors. The model executes instruction objects, not a binary decoder. These are
standalone model checks, not RTL, full-SoC, FPGA or whole-model qualification.
The compiler does not import or require `npu_model`.

The optional model suite also runs ten MXU pressure cases: raw BF16
save/restore, seeded continuation, reset/spill/continuation, and one- and
three-iteration handle-carrying loops on each unit.
References use exact small integers and independent pair-layout indexing.
Four additional input cases run one shared program with both MXUs under
pressure. A runtime boolean selects accumulation or a per-unit permutation of
the completed seeds, then both paths join with six outputs. Control words
0/2 and 1/3 check the existing low-bit boolean ABI. Both owners' weight and
accumulator slots, edge transport and output ordering are checked together.

To generate the corresponding RTL fixtures, including every BF16 bit pattern
through the raw path on each MXU:

```sh
ATLAS_OOT_BIN_DIR=build/bin PYTHONPATH=.:test python \
  test/rtl/make_mxu_spill_fixtures.py --out out/mxu-spill-fixtures
```

Pass each generated directory to `test/rtl/run_compiled_program_atlas_core.py`
with `--fixture`. There are 140 fixtures: 64 raw transport cases, two
arithmetic continuation cases and two loop cases per unit, plus four input
cases for the shared two-unit branch program. Every pattern appears in the oldest
accumulator, which this source order forces through VMEM spilling. The raw
cases cover all 65,536 BF16 patterns,
including signed zeros, subnormals, infinities and NaN payloads, without doing
arithmetic. This is not exhaustive arithmetic or whole-system qualification.
The runner records source/tool identities and actual execution results; fixture
generation alone does not establish a hardware pass.
