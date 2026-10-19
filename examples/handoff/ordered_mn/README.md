# Ordered M/N matmul host

The [64×96×64 Linalg source](../../../test/examples/native_mxu0_m64_n96_k64_linalg.mlir)
splits into six 32×32 output tiles. All six tiles compile to the same request,
so one verified 54-word Atlas program is launched six times with new activation
and weight data each time.

Compile with `native-compile --ordered-m-tile 32 --ordered-n-tile 32
--outer-tile-index 4 --ordered-k-tile 32` and verify with `native-verify` (see
[native Merlin binding](../../../docs/native-merlin-binding.md)). Then bundle the
host data and build the RV64 bare-metal ELF:

```sh
python3 tools/atlas_ordered_mn_host_data.py \
  --compiled COMPILED --verification VERIFY_JSON \
  --host-map runtime/selected-host-map.json \
  --packer-header "$MERLIN_ROOT/merlin/runtime/c/row_major_k_window.h" \
  --boundary-header "$MERLIN_ROOT/merlin/runtime/c/finite_float_boundary.h" \
  --out HOST_DATA
bash examples/handoff/ordered_mn/build.sh HOST_DATA FRESH_OUTPUT_DIR
```

The exporter refuses tiles whose request digests differ. The host uses two
fixed raw-FP8 input patterns and writes two 64×96 BF16 results to
`atlas_matrix_outputs`; `atlas_matrix_status_word == 1` means done. It checks
shapes, input codes, and buffer overlap before launching, and publishes the
result only after all six tiles succeed.

## Output layout

Atlas stores a BF16 tile as two 16-column halves. The execution plan records
`output_physical_layouts` per output, and the C runtime unpacks the halves while
assembling the matrix. In Python, call
`atlas_mlir_native.layout.unpack_native_bf16_output(plan, tile_bytes)` on each
tile before Merlin's `assemble_row_major_mnk_outputs`, which expects row-major
tiles. A `null` layout is refused; recompile older artifacts that lack the field.

For partial M/N shapes, the generated header selects
`atlas_ordered_mn_run_tails` and the host provides two full 32×K staging
buffers padded with `+0.0`.

## Tests

Build [`test/ordered-mn-host-test.c`](../../../test/ordered-mn-host-test.c) with
`-DATLAS_USE_GENERATED_MN_DATA=1 -DATLAS_TEST_MN_FROM_HEADER=1` against an
exported host-data directory. It checks every launch, input staging, result placement,
repeated calls, and refusals for up to four K tiles. Its fake MMIO returns
synthetic tiles, so it does not run the Atlas program.

## Limits

The host map has no bitstream, and the ELF assumes coherent memory and direct
MMIO. The program has been checked on the functional model only; it has not run
on the Shuttle + Atlas SoC or an FPGA.
