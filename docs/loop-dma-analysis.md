# DMA overlap through scalar loops

DMA wait insertion, machine scheduling, and verification share the loop
analysis in `AtlasLoop.h` / `AtlasLoop.cpp`. PACK is one client, rather than
the condition that enables overlap.

## Discovery and proof

Discovery inspects the flat machine stream's redirects and instruction effects.
A supported loop is one block closed by a backward conditional branch and,
once timing fills it, its delay slot. Its body, the slot included, consists of
scalar loads, scalar stores, ALU operations, and delays. No other redirect may enter the body. DMA launches,
waits, configuration, nested/interior redirects, and unsupported effects are
not admitted. An indirect jump disables discovery because its destination
cannot be proved. No helper tags, source-operation IDs, particular registers,
copy order, or 32-row trip count are required.

Shape and proof are separate. The decoded stream records the loop's entry
register facts from its fall-through predecessor, then interprets scalar
instructions and branch conditions. Scalar load results become unknown; the
analysis does not guess memory contents. The same scalar-access rules used by
timing determine the bytes touched in every iteration, including address
alignment and the hardware's address slicing. The result conservatively bounds
all reads and all writes. Unresolved addresses, unresolved termination, or an
analysis budget of more than 256 iterations produce an unknown footprint.
Exceeding the budget does not establish that the loop is infinite.

Wait insertion compares each pending DMA staging span with that footprint:

- A DMA load writes staging, so loop reads and writes can conflict.
- A DMA store reads staging, so only loop writes can conflict.
- Unknown footprints require completion before loop entry.

Pending DMA may cross the entry and exit of a supported loop. Other CFG
boundaries remain conservative. Timing checks replay the loop's accesses
against the pending transfers over all iterations; generated-memory checks
use the shared footprint. The loop itself contains no DMA lifecycle changes.

This does not implement general multi-block/nested-loop analysis or
cross-iteration DMA pipelining. The discovery, bounded interpreter, memory
summary, and DMA consumers are separated so broader effect summaries and
control-flow regions can be added without reintroducing producer-specific
recognition.

## PACK contracts

`AtlasPackLoop.h` now contains only an additional provenance check used by the
PACK buffer contract. That check still requires PACK helper annotations and
consistent source ownership, and the buffer contract still proves the exact
32-row interleave. Removing PACK-specific recognition from DMA overlap does
not remove PACK's numerical or layout obligations.

## Performance comparison

Baseline: prototype `ccca5b4`, with the metadata-ID validation expressed using
an explicit integer bound so it builds with the workspace's LLVM. The loop
refactor uses the same compiler pipeline and inputs. Fresh one-output and
two-output builds through `build_fixture.sh` emit byte-identical `program.bin`
files before and after the change.

| Captured AttentionMLP | PACK-specific cycles | General-loop cycles | DMA busy, both |
|---|---:|---:|---:|
| One output | 9,774 | 9,774 | 52.76% |
| Two outputs | 10,680 | 10,680 | 57.95% |

Replay uses `bringup2026/npu_model` revision `2170649`, DefaultHardwareConfig,
remapped 4-MiB DRAM, staged checked-in weights, and a uniform FP8 0.5 activation.
The adapter corrects DMA VMEM operands to RTL word addressing for both bank
footprints and functional execution; runtime errors are fatal. Output bytes
match between implementations for every output. These are model diagnostics,
not RTL qualification.

Seven additional examples emit identical words: attention, transposed-key
attention, two-layer FP8 MLP, softmax, FP8 and BF16 explicit-DMA MXU examples,
and the virtual-scheduler overlap example. Comparison artifacts and replay
logs are in `out/loop-dma-comparison/`.

## Regression coverage

`test_loop_dma_overlap.py` exercises untagged non-PACK loops with varied trip
counts and positive, negative, and zero strides through both timing passes.
It checks later-iteration conflicts, DMA read/write direction, unknown
addresses and bounds, analysis-budget exhaustion, additional entries, and
DMA lifecycle operations inside loops. Unsupported loops remain legal when
DMA completes before entry. Existing PACK provenance, layout, wait insertion,
DMA memory, and timing tests remain in place.

Validation: all 24 compiled CTest suites pass. The full dependency-complete
Python run executes 716 tests with 307 skips and four failing subcases in
three numerical-evaluator tests. All four reproduce with the saved baseline
compiler and concern NaN encodings (`0xffff` versus `0x7fc0`) in the installed
model. The targeted DMA, PACK, metadata, buffer-contract, and generic-loop
regression tests pass.

## Non-PACK execution experiment

A separate untagged scalar loop runs 64 iterations. Iteration `i` loads a word
at VMEM `0x1000 + 12*i` and stores it at `0x2000 + 20*i`, both in bank 0.
A concurrent 1024-byte DRAM load writes VMEM `0x80000..0x803ff`, in bank 2.
There is no PACK instruction, helper annotation, or virtual source identity.
The serialized control differs only by moving DMA.WAIT from after the loop
to immediately after the DMA launch. DMA configuration precedes operand setup
so it completes before the first channel-zero launch.

| Timing pass | Wait before loop | Wait after loop | DMA-busy cycles during loop |
|---|---:|---:|---:|
| insert-atlas-delays | 1,170 cycles | 654 cycles | 0 vs 515 |
| schedule-atlas-stream | 1,036 cycles | 530 cycles | 0 vs 515 |

For stream scheduling, end-to-end DMA busy rises from 49.90% to 97.55%; runtime
falls 48.84%. Both variants execute identical useful DMA traffic. The model
reports 517 DMA-busy cycles total, including configuration. Loop intervals
are attributed from the first body issue until the first issue after its
back-edge delay slot; block-boundary scheduling can change those intervals
slightly between variants.

Every copied word, untouched destination byte, and incoming DMA payload is
checked using nonzero patterns. The two variants' copied-memory and DMA-payload
hashes match. Runtime errors are fatal. The same model revision and addressing
adapter described above are used. A variant whose loop writes the DMA staging
span is rejected. The saved PACK-only compiler rejects the untagged overlap
at its block boundary, confirming the generic analysis enables this case.

This experiment measures generic-loop admission, timing, and execution with
explicit before/after wait placements; it does not claim end-to-end automatic
wait placement from a new virtual CFG workload. Build, replay scripts, source,
timed IR, assembly, compiler diagnostics, and `results.json` are retained in
`out/nonpack-dma-overlap/`.
