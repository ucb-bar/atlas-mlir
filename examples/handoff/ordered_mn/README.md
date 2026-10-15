# Exact M/N host handoff

The annotated [64×96×64 Linalg source](../../../test/examples/native_mxu0_m64_n96_k64_linalg.mlir)
partitions into six 32×32 output tiles. All six requests have the same typed
tile digest, so one verified 54-word Atlas program can be reused with new
activation and weight data for each output tile.

`tools/atlas_ordered_mn_host_data.py` accepts one `native-compile` artifact
made with `--ordered-m-tile 32 --ordered-n-tile 32 --outer-tile-index 4
--ordered-k-tile 32` and its matching `native-verify` receipt. It checks that
all outer tiles have the same request digest, then bundles the program, source
geometry, Merlin K-window packer, and Atlas host runtime sources. Build the
RV64 bare-metal ELF with:

```sh
bash examples/handoff/ordered_mn/build.sh VERIFIED_HOST_DATA_DIR FRESH_OUTPUT_DIR
```

The host uses two public raw-FP8 input patterns and records two 64×96 BF16
outputs in `atlas_matrix_outputs`. `atlas_matrix_status_word == 1` denotes
completion. It loads and launches the same program six times per invocation.
The runtime checks source shapes, input codes, and buffer separation before
launching, and publishes the logical output only after all six tiles succeed.
Atlas stores a BF16 tile as two physical 16-column halves. The native execution
plan records `output_physical_layouts` for every output. The C host runtime
converts those halves while assembling the logical matrix. Python checks must
call `atlas_mlir_native.layout.unpack_native_bf16_output(plan, tile_bytes)` on
each physical tile before passing it to Merlin's
`assemble_row_major_mnk_outputs`; that assembler expects row-major tile bytes.
An unqualified output layout is recorded as `null` and this decoder refuses it.
Older native artifacts without this field must be recompiled before host export.

For partial M/N shapes, the generated header selects
`atlas_ordered_mn_run_tails` and the host supplies two full 32×K staging
buffers. The exporter accepts verified v2 tiling records with positive-zero
padding. The default example is exact 64×96; rebuilding against a verified
tail-shape header selects the tail-aware host path.

`test/ordered-mn-host-test.c` also accepts
`ATLAS_USE_GENERATED_MN_DATA=1` and `ATLAS_TEST_MN_FROM_HEADER=1`. Compile it
against an exported host-data directory to check all declared launches,
input staging, result placement, repeated invocation, and refusal paths for
that source shape. Its fake MMIO returns synthetic output tiles; it does not
execute the Atlas binary. This bounded harness supports up to four K tiles.

The selected host map still has no bitstream identity. This ELF assumes
coherent memory and direct Atlas MMIO. The fake-MMIO test checks staging and
assembly; separate task-local `npu_model` execution checks the emitted
instructions. Neither check establishes selected SoC or FPGA execution.
