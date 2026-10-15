# Source-bound f32 matrix handoff

This example builds a diagnostic Shuttle ELF from **all** verified tiles of
one captured f32 matmul region. It accepts row-major `A[M,K]`, `RHS[K,N]`
or one leading batch dimension, `A[B,M,K]`, `RHS[B,K,N]`. The host converts
each matrix to the selected FP8 inputs, launches the same verified Atlas
program over every batch and M/N output tile, and stores row-major f32 values
decoded from BF16 readback. It handles partial output tiles and ordered K
windows. The batched runtime stages the complete result before publishing it.

The tile directory has `tile-0`, `tile-1`, ... and matching
`tile-0-verification.json`, `tile-1-verification.json`, ... receipts. Every
tile must have the same program, input/output addresses, source graph,
precision policy, and selected target. Batched tiles are ordered by batch,
then M/N output window. The exporter rejects missing tiles, changed artifacts,
changed batch or output windows, and mismatched frozen weights. A frozen RHS
requires `--compile-inputs` pointing to the compiler-only staged source. The
exporter replays each tile's declared weight, checks its packed payload against
the verified artifact, and embeds the source f32 bits in the host bundle. The
two diagnostic invocations then vary only the runtime LHS. Model graph
composition still requires separate support.

The bundle also includes `atlas_source_mn_memref.h/.c`. Its rank-two and
rank-three entrypoints accept MLIR f32 memref descriptors, copy supported
strided inputs into dense staging, and return a newly allocated dense result
only after all tiles succeed. The caller owns that result allocation. When
the RHS is frozen, the adapter checks the incoming source bits before any
Atlas launch. The standalone ELF above uses flat arrays and therefore does
not link these entrypoints or provide their allocator.

From the Atlas MLIR checkout, with Merlin's Python package installed:

```sh
python tools/atlas_source_mn_host_data.py \
  --tiles-root /path/to/verified/source-region-tiles \
  --host-map runtime/selected-host-map.json \
  --packer-header /path/to/merlin/merlin/runtime/c/row_major_k_window.h \
  --boundary-header /path/to/merlin/merlin/runtime/c/finite_float_boundary.h \
  --compile-inputs /path/to/staged-compile-inputs \
  --out "$TMPDIR/source-mn-host-data"
bash examples/handoff/source_mn/build.sh \
  "$TMPDIR/source-mn-host-data" "$TMPDIR/source-mn-elf"
```

For a rank-two or one-batch-dimension matmul selected as a complete tiled
region by Merlin's model probe, generate its exact outlined MLIR C-interface
callback:

```sh
python tools/atlas_source_mn_model_bridge.py \
  --probe /path/to/model-probe \
  --region-id matmul_0 \
  --host-bundle "$TMPDIR/source-mn-host-data" \
  --out "$TMPDIR/source-mn-model-wrapper"
```

Use the selected `batch_matmul#0` region ID for a batched source. The wrapper
checks the probe's source and target identity, each tile's native manifest,
the verified host bundle, and the outlined two-input function type. Rank-three
calls use `atlas_source_mn_run_memref3`; rank-two calls use the rank-two entry.
It emits `atlas_source_mn_model_wrapper.c/.h` with a host binder, a checked
status-returning call, and the `_mlir_ciface_*` callback. Compile the wrapper
with the host bundle's headers and runtime sources. The callback uses
`atlas_model_tile_fail` on failure; a platform may supply that non-returning
hook. `atlas_model_link.py` accepts this wrapper with `--source-bundle` beside
the checked Merlin model probe, host object, and host-object check. It checks
every tile's encoded reset/accumulate sequence before producing a relocatable
RV64 object:

```sh
python tools/atlas_model_link.py \
  --probe "$MODEL_PROBE" --host-check "$HOST_CHECK" \
  --host-object "$HOST_OBJECT" \
  --wrapper "$TMPDIR/source-mn-model-wrapper" \
  --source-bundle "$TMPDIR/source-mn-host-data" \
  --merlin-runtime-include "$MERLIN_ROOT/merlin/runtime/c" \
  --clang "$LLVM_BIN/clang" --nm "$LLVM_BIN/llvm-nm" \
  --out "$TMPDIR/source-mn-model-link"
```

The K=72 frozen-RHS test passes the callback check and links against a small
RV64 host stub. A separate Merlin model probe with both tiles and an audited
RV64 host object also linked with the same grid. That link leaves the platform
C allocator and memory functions unresolved. The
[`model-source-grid-host-test.c`](../../../test/model-source-grid-host-test.c)
driver calls the corresponding lowered x86 host entry twice with different
LHS values. Its fake device checks all instruction words and packed input
windows, then returns a deterministic BF16 pattern to check both output
windows and the result descriptor. Run it from a fresh output directory:

```sh
test/run_model_source_grid_host.sh \
  "$HOST_LOWERING/model.ll" "$TMPDIR/source-mn-host-data" \
  "$TMPDIR/source-mn-model-wrapper" "$TMPDIR/source-mn-model-link" \
  "$TMPDIR/source-mn-model-host-check" "$LLVM_BIN/clang"
```

This validates the model call path; it does not execute Atlas arithmetic or
the selected SoC.

For the public `M=3, K=72, N=35` case, that host test saves the FP8 windows
it actually staged under `source-mn-model-host-check/capture/`. The following
creates four selected-core fixtures from those bytes. The expected BF16 cells
come from an exact ordered-reduction reference over the captured FP8 values:

```sh
for trial in 0 1; do
  for tile in 0 1; do
    name="real_host_trial${trial}_n$((tile * 32))"
    tile_dir=$(printf '%s/regions/named-0000-tile-%04d' "$MODEL_PROBE" "$tile")
    python test/rtl/make_source_grid_capture_fixture.py \
      --capture "$TMPDIR/source-mn-model-host-check/capture" \
      --program "$tile_dir/program.bin" \
      --execution-plan "$tile_dir/execution_plan.json" \
      --trial "$trial" --tile-index "$tile" --m 3 --k 72 --n 35 \
      --name "$name" --out "$TMPDIR/source-mn-rtl-fixtures/$name"
  done
done
```

Pass those four directories as `--fixture` arguments to
[`run_compiled_program_atlas_core.py`](../../../test/rtl/run_compiled_program_atlas_core.py)
with the pinned RTL and toolchain arguments in the
[selected-core guide](../../../docs/native-program-full-core-observation.md).
The runner writes the physical output bytes to `$RTL_RUN/observed/` only
after checking every BF16 cell, DMA transaction, input, guard, and halt.
Replay those observed bytes through the same compiled host entry:

```sh
ATLAS_SOURCE_MN_RTL_OUTPUT_DIR="$RTL_RUN/observed" \
test/run_model_source_grid_host.sh \
  "$HOST_LOWERING/model.ll" "$TMPDIR/source-mn-host-data" \
  "$TMPDIR/source-mn-model-wrapper" "$TMPDIR/source-mn-model-link" \
  "$TMPDIR/source-mn-model-rtl-replay" "$LLVM_BIN/clang"
```

The replay checks the RTL result report, program identity, and output hashes
before running. It then checks the assembled `3×35` f32 result against the
public source values for both invocations. This is an offline host/AtlasCore
handoff check; the Shuttle CPU and AtlasCore are still separate processes.
On selected RTL `0079c0541111197741a231c002e3843fa6f545b2`, the exact
72-word image linked to the real model host object passed four captured-input
launches after the runner's 19 base launches. Each captured launch checked
1,024 physical BF16 cells and 256 DMA beats. Replaying the four observed
tiles through the compiled host entry produced the expected full `3×35`
tensor on both invocations. Changed output bytes and a changed program hash
were each refused by the replay check.

`build.sh` checks every exported source hash before compiling and writes
`atlas-source-mn.elf` and `build-manifest.json` only after a successful RV64
link. The ELF contains two runtime-generated public LHS patterns, each covering
every selected batch. Without `--compile-inputs`, RHS is generated at runtime
too; with a frozen RHS, both invocations use the staged parameter. Inspect
`atlas_source_matrix_status_word`, `atlas_source_matrix_device_word`, and
`atlas_source_matrix_outputs[2]` after execution. Status 1 means both
invocations completed; the outputs still need an independent arithmetic
check on the selected execution platform.

The selected host map has no FPGA bitstream identity and assumes coherent
memory. This ELF is a diagnostic handoff, not a qualified model or FPGA run.
Do not execute it on ordinary RISC-V hardware without the selected Atlas
MMIO/DRAM map and cache behavior.
