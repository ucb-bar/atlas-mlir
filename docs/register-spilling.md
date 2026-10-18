# Register spilling and VMEM temporaries

`--spill-atlas-virtual` handles FP8/BF16 tile values and `i1`/`i32` controls,
including values passed through branches and loops. Run it after virtual
scheduling and before physical allocation. It uses the existing register
allocator, wait insertion and physical timing passes.

```sh
mkdir -p out
build/bin/atlas-opt input.mlir \
  --schedule-atlas-virtual --spill-atlas-virtual \
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
size failures remain explicit compilation failures. Live MXU weights and
accumulators are not spillable without an exact save/restore route; BF16 readout
is not assumed to preserve internal state.

## Tests

```sh
ATLAS_OOT_BIN_DIR=build/bin PYTHONPATH=.:test python -m unittest -v \
  test_virtual_spilling test_spill_resources
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
