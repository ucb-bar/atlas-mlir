# Shared construction of static schedules

`atlas_mlir_native.shared_static_schedule` uses Merlin to schedule a fixed-only,
straight-line physical instruction stream. It is an initial conservative consumer
of the shared scheduling design, not a replacement for the existing whole-model
scheduler. No compiler-generation loop is involved.

The steps are:

1. `atlas-emit --allow-untimed --fixed-effects-json` exports complete occupancy
   bounds from the selected timing provider, before scheduling. It rejects DMA,
   event-relative accesses and branches/jumps. The provider checks each instruction
   can issue alone. Its bound includes accesses, holds, register release windows,
   VPU lifetime and natural issue spacing.
2. Merlin's `schedule_fixed_block` constructs issue/completion annotations and
   `schedule.drain`. This adapter deliberately serializes whole-machine occupancy;
   it does not approximate Atlas's alternative ports or detailed overlap rules.
3. Merlin's `materialize_timing_plan` inserts actual target delay calls and checks
   exact source-call correspondence, natural advances and tail completion.
4. `atlas-opt --verify-atlas-timing --verify-atlas-machine-stream` validates the
   result. No Atlas scheduling pass runs here. Final encoding is compared with
   every preserved source word and emitted delay.

The existing model-derived timing provider remains diagnostic evidence. Neither
its provider name nor the shared plan is independent RTL qualification. Target
instruction facts stay in this repository; no Atlas latencies, ports, registers
or encodings were added to Merlin.

## Reproduce

Select the companion Merlin branch and an LLVM build, then rebuild the OOT tools.
The same command works with the isolated Merlin compiler-library export as
`MERLIN_LIBRARY`; it does not require importing the complete Merlin checkout.

```sh
cmake --build out/build/generalized-layer/compiler --target atlas-opt atlas-emit -j 4
PYTHONPATH="$MERLIN_LIBRARY:$PWD/python" "$COMPILER_PY" \
  -m atlas_mlir_native.shared_static_schedule test/examples/shared_static_matmul.mlir \
  --destination out/build/static-matmul \
  --atlas-bin-dir out/build/generalized-layer/compiler/bin
```

Use fresh destinations. `shared_static_vpu.mlir` exercises VLI(-1), ReLU and addition
of VLI(1). `shared_static_matmul.mlir` builds E4M3 ones, pushes weights, performs
32×32 matmul and reads BF16 output. Their independently expected tiles are all
ones and all 32s, respectively. The test independently decodes final words,
executes them in the pinned simulator and checks every output element exactly.
Removing generated delays fails the timing verifier. A DMA-containing input is
explicitly rejected instead of receiving a fixed completion estimate.

```sh
PYTHONPATH="$MERLIN_ROOT/src:$PWD/python:$PWD:$MODEL_ROOT" \
  ATLAS_OOT_BIN_DIR="$PWD/out/build/generalized-layer/compiler/bin" \
  "$MODEL_PY" -m pytest -q test/test_shared_static_schedule.py
```

Selected model revision: `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`.
Selected RTL identity: `0079c0541111197741a231c002e3843fa6f545b2`.
The source, provider export, plan, emitted calls, final machine IR, instruction
words and tool hashes are retained in each compilation directory. Numerical
validation is test-only and never an input to scheduling.

## Boundary scope

Merlin also exports `completion_schedule`: explicit completion decisions become
wait calls on existing `Call.produces`/`Call.awaits`, preserving counted loops.
Its queue test compiles and executes generated C and requires an actual token
speller. That mechanism is not yet connected to this Atlas fixed-only adapter.
An Atlas full-model scheduling replacement must additionally integrate DMA and
CFG boundaries, including branch timing/delay slots and loop-entry state.
Cross-iteration pipelining and optimized Atlas resource overlap remain separate
work. The existing full-model pipeline continues to supply those functions.

The companion design is `docs/design/static_scheduling.md`; the exported pure
model-partitioning API is `merlin.llvmlower.floating_contractions`. The installed
host compiler driver and independent execution/oracle machinery are not exported.

Validation on the separate generalized worktree passed 50 shared integration
checks (one optional replay skipped), all 158 CTest entries, and three tests
using only the exported Merlin library. Optional CTest fixtures retain their
explicit skips. The fixed VPU and matmul programs executed in 266 and 295 model
cycles, with 11 and 13 instruction words. These are device simulator results for
these tiny programs, not model throughput or a hardware performance claim.
