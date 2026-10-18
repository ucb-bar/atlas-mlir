# Register spilling

`--spill-atlas-virtual` recovers FP8 and BF16 tile register pressure through
VMEM stores and reloads. Run it after virtual scheduling and before physical
allocation. The first implementation supports single-block functions. CFG and
scalar pressure produce explicit errors. Live MXU weight and accumulator
handles are not spillable: BF16 readout is not assumed to preserve internal state.

```sh
mkdir -p out
build/bin/atlas-opt input.mlir \
  --schedule-atlas-virtual --spill-atlas-virtual \
  --lower-atlas-virtual-to-machine --insert-atlas-dma-waits \
  --insert-atlas-delays --verify-atlas-generated-schedule \
  --verify-atlas-timing -o out/machine.mlir
build/bin/atlas-emit out/machine.mlir > out/words.hex
```

The stage-preserving driver enables the pass with `--spill` and retains
`2-spilled.mlir` before allocation:

```sh
python tools/lower_atlas_program.py input.mlir --spill --output-dir out/spilled
```

The native Merlin binding runs it in both mandatory and optimized pipelines.
Existing scheduling, coloring, waits and timing checks remain in use.

## Operations and storage

| Operation | Inputs | Results |
| --- | --- | --- |
| `atlas.virtual_spill_bf16` | state, BF16 tile, `slot_word` | next state, saved identity |
| `atlas.virtual_spill_fp8` | state, FP8 tile, `slot_word` | next state, saved identity |
| `atlas.virtual_reload_bf16` | state, saved identity | next state, BF16 tile |
| `atlas.virtual_reload_fp8` | state, saved identity | next state, FP8 tile |

The saved identity has type `!atlas.virtual_spill`. A reload names its exact
writer and must agree with its format. All four ops have memory effects and
participate in the state chain. The virtual scheduler treats them as barriers.

The selected layout reserves VMEM bank 2 after its eight DMA staging windows:
four-byte words `[135168, 196608)`, or bytes `[540672, 786432)` (240 KiB).
Boundary I/O, PACK scratch and DMA staging are outside this frame. FP8 uses
256 words (1 KiB); BF16 uses 512 words (2 KiB), retaining both physical halves.
Slots are aligned to 256 words and remain occupied through their last reload.
Disjoint lifetimes may reuse storage.

Lowering emits raw `VSTORE`/`VLOAD` operations using the existing reserved
address register. It performs no conversion, requantization or accumulator
readout. Physical timing still has to protect read completion before reuse.

## Allocation and checks

The pass queries the existing allocator. On tensor pressure it chooses an
unspilled value with a long lifetime, inserts one save and a reload for each
consuming operation, places slots, and retries allocation. Reload results are
not spilled again. Source order breaks ties. This is a bounded heuristic;
failure does not prove that no spill strategy exists.

`max-spills` bounds rewrites (default 256). `vmem-words` limits the frame;
zero selects the full reserved frame. For example:

```sh
build/bin/atlas-opt --spill-atlas-virtual='max-spills=32 vmem-words=4096' input.mlir
```

Frame exhaustion, an insufficient register working set and the rewrite limit
are errors. `atlas.spill_report` records stores, reloads, and frame words used
and available. The physical allocation report also lists the slots.

Virtual verification checks bounds, writer/format identity and overlap through
the last reload. Register checks run again after rewriting. Generated tile and
buffer checks retain every save/reload half and check its physical address,
reaching writer and bytes. Arbitrary raw readbacks remain invalid.

## Tests and remaining work

```sh
ATLAS_OOT_BIN_DIR=build/bin python -m unittest discover \
  -s test -p test_virtual_spilling.py -v
ctest --test-dir build --output-on-failure
```

Focused tests force FP8/BF16 pressure, check transported bytes and preserved
inputs, exercise reuse and budgets, and reject changed addresses, formats and
overlaps. Their sequential copy executor is not an RTL or timing simulator;
emitted programs also pass the regular stream and timing checks.
The Python virtual evaluator consumes pre-spill IR; it does not interpret the
new saved identities or serve as a production fallback.

CFG spilling needs edge-aware ownership and loop-carried values. Scalar
spilling needs legal reload temporaries and memory operations. MXU state needs
a separately established exact save/restore route. These remain unsupported;
no case silently changes numerical representation or moves to host execution.
