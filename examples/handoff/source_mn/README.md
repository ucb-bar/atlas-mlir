# Source f32 matmul host

This example builds a Shuttle host ELF from all verified tiles of one captured
f32 matmul region: row-major `A[M,K] × RHS[K,N]`, or with one leading batch
dimension `A[B,M,K] × RHS[B,K,N]`. The host converts inputs to FP8, launches the
same Atlas program for every batch and M/N tile (partial tiles and ordered K
windows included), and returns row-major f32 decoded from BF16.

The tiles directory holds `tile-0`, `tile-1`, ... and matching
`tile-N-verification.json` receipts, ordered by batch, then M/N window. Every
tile must share the program, DMA addresses, source graph, precision policy, and
target. A frozen RHS needs `--compile-inputs` pointing at the staged compiler
inputs; the exporter replays each tile's weight, checks it against the
artifact, and embeds the f32 bits. Then only the LHS varies at run time.

## Build the ELF

From this repository, with Merlin's Python package installed:

```sh
python3 tools/atlas_source_mn_host_data.py \
  --tiles-root /path/to/verified/source-region-tiles \
  --host-map runtime/selected-host-map.json \
  --packer-header "$MERLIN_ROOT/merlin/runtime/c/row_major_k_window.h" \
  --boundary-header "$MERLIN_ROOT/merlin/runtime/c/finite_float_boundary.h" \
  --compile-inputs /path/to/staged-compile-inputs \
  --out "$TMPDIR/source-mn-host-data"
bash examples/handoff/source_mn/build.sh \
  "$TMPDIR/source-mn-host-data" "$TMPDIR/source-mn-elf"
```

`build.sh` checks every exported source hash, then writes `atlas-source-mn.elf`
and `build-manifest.json`. The ELF runs two fixed LHS patterns. After it runs,
read `atlas_source_matrix_status_word` (1 = both runs done),
`atlas_source_matrix_device_word`, and `atlas_source_matrix_outputs[2]`.

The bundle also has `atlas_source_mn_memref.h/.c`, rank-two and rank-three
entry points that take MLIR f32 memrefs, copy strided inputs to dense staging,
and return a newly allocated dense result (owned by the caller) after all tiles
succeed. The standalone ELF does not use them.

## Call from a compiled model

For a rank-two or batched matmul region found by Merlin's model probe, generate
the outlined MLIR C-interface callback, then link it with the host object
(use region ID `batch_matmul#0` for a batched source):

```sh
python3 tools/atlas_source_mn_model_bridge.py \
  --probe "$MODEL_PROBE" --region-id matmul_0 \
  --host-bundle "$TMPDIR/source-mn-host-data" \
  --out "$TMPDIR/source-mn-model-wrapper"
python3 tools/atlas_model_link.py \
  --probe "$MODEL_PROBE" --host-check "$HOST_CHECK" \
  --host-object "$HOST_OBJECT" \
  --wrapper "$TMPDIR/source-mn-model-wrapper" \
  --source-bundle "$TMPDIR/source-mn-host-data" \
  --merlin-runtime-include "$MERLIN_ROOT/merlin/runtime/c" \
  --clang "$LLVM_BIN/clang" --nm "$LLVM_BIN/llvm-nm" \
  --out "$TMPDIR/source-mn-model-link"
```

The wrapper (`atlas_source_mn_model_wrapper.c/.h`) has a binder, a
status-returning call, and the `_mlir_ciface_*` callback. On failure the
callback calls `atlas_model_tile_fail`, which a platform may define. The link
step checks each tile's reset/accumulate sequence and writes a relocatable RV64
object; the platform still supplies the C allocator and memory functions.

## Test the call path

[`test/run_model_source_grid_host.sh`](../../../test/run_model_source_grid_host.sh)
builds [`test/model-source-grid-host-test.c`](../../../test/model-source-grid-host-test.c)
against the lowered x86 host entry and calls it twice with different LHS
values. A fake device checks instruction words and packed inputs and returns a
fixed BF16 pattern. Use a fresh output directory:

```sh
test/run_model_source_grid_host.sh \
  "$HOST_LOWERING/model.ll" "$TMPDIR/source-mn-host-data" \
  "$TMPDIR/source-mn-model-wrapper" "$TMPDIR/source-mn-model-link" \
  "$TMPDIR/source-mn-model-host-check" "$LLVM_BIN/clang"
```

To check real Atlas results for the `M=3, K=72, N=35` case, turn the FP8 windows
the test staged into RTL fixtures, run them on the core, and replay the output:

```sh
for trial in 0 1; do
  for tile in 0 1; do
    name="real_host_trial${trial}_n$((tile * 32))"
    tile_dir=$(printf '%s/regions/named-0000-tile-%04d' "$MODEL_PROBE" "$tile")
    python3 test/rtl/make_source_grid_capture_fixture.py \
      --capture "$TMPDIR/source-mn-model-host-check/capture" \
      --program "$tile_dir/program.bin" \
      --execution-plan "$tile_dir/execution_plan.json" \
      --trial "$trial" --tile-index "$tile" --m 3 --k 72 --n 35 \
      --name "$name" --out "$TMPDIR/source-mn-rtl-fixtures/$name"
  done
done
# Run the four fixtures with test/rtl/run_compiled_program_atlas_core.py
# (see docs/hardware-status.md); it writes output bytes to $RTL_RUN/observed/.
ATLAS_SOURCE_MN_RTL_OUTPUT_DIR="$RTL_RUN/observed" \
test/run_model_source_grid_host.sh \
  "$HOST_LOWERING/model.ll" "$TMPDIR/source-mn-host-data" \
  "$TMPDIR/source-mn-model-wrapper" "$TMPDIR/source-mn-model-link" \
  "$TMPDIR/source-mn-model-rtl-replay" "$LLVM_BIN/clang"
```

The replay checks the RTL report, program hash, and output hashes, then
compares the assembled `3×35` f32 result with the source values for both
calls. On the pinned RTL this case matches.

## Limits

The CPU and the Atlas core run as separate processes in the replay; this is not
an integrated run. The host map has no bitstream and assumes coherent memory.
Do not run the ELF on ordinary RISC-V hardware without the real Atlas MMIO/DRAM
map and cache handling.
