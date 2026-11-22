# Native compile presets

The selected Atlas support provider owns `--recipe dispatch-reduction` and `--board sophialake`. Merlin registers provider arguments through optional `configure_codegen_parser` and `expand_codegen_arguments` hooks. Atlas recipe options live in `atlas_mlir_native.options.CodegenOptions`; Merlin's options remain target neutral.

`dispatch-reduction` expands the qualified adjacent fusion selection, batch-panel operand reuse, exponent-page cache, VMEM input staging, optimized pipeline, static quantizer sizing, exact binary routes, lane grouping, binary and cast vector preflight, exact float unrolling, 128,000,000 executed instructions per launch, a 786,432-byte I/O window, and the device KV quantizer and value transpose. Every choice remains an individual option in the identity and application manifest. The recipe name does not enter identity. Explicit flags override expansion regardless of order, including Boolean overrides such as `--no-device-kv-quantizer` and `--no-exact-float-unroll`.

`sophialake` describes DDR at `0x80000000` with 1 GiB of physical capacity, and the Atlas DMA aperture at `0x80000000` with 2 GiB of addressable capacity. Physical capacity constrains allocations independently of the larger DMA aperture. It requires a measured host prefix at DDR base through `--host-memory-reserved-bytes` and measured staging demand through `--io-memory-bytes`. Both must be positive and tile aligned, and must leave capacity for residency. The host prefix covers the measured host footprint; it has no fixed default. The board enables residency and wide-cache operand reuse. Other codegen choices retain their defaults unless selected explicitly.

Counted quantized frames exclude resident operands from staging and host scratch. Complete measured Merlin placement and the final linker check are authoritative; the board prefix bridge alone cannot certify physical fit.

For measured prefix `H`, usable capacity is `[0x80000000 + H, 0xc0000000)`, and staging starts at `0x80000000 + H`. Explicit flags override board expansion. Explicit capacity overrides must stay outside the reserved host range and within physical DDR. Board identity equals the explicit geometry, residency, and wide-cache equivalent; a board name is not an identity component.

Activate an install that sets `MERLIN_NATIVE_SNAPSHOT` to its checked native snapshot, or pass `--snapshot` explicitly. Snapshot/profile validation still runs. Native compile defaults to `--abi auto`, application compilation defaults to diagnostic mode, invocation-local exact materialization reuse is enabled, and region reuse stays disabled because residency embeds addresses. `--no-materialization-cache` disables reuse. Region compilation defaults to strict-native mode; specify another mode for diagnostic requests.

`--recipe dispatch-reduction --board sophialake` is unsupported because of the known `spill rewrite work limit reached` failure observed with a 16 MiB staging window. It has not been requalified with measured board geometry. Argument expansion refuses the combination before loading the snapshot or compile inputs, including when individual flags override its defaults. No compiler limit is changed. Select the board alone for default codegen with wide-cache and residency, or the recipe alone at its 768 KiB staging window.

For a prepared layer capture, the board compile with default codegen and wide-cache is:

```sh
merlin-targetgen native-compile --engine merlin_native \
  --support atlas \
  --target-source "$ATLAS_ROOT" --compile-inputs "$CAPTURE" \
  --application --application-memory-bytes 536870912 \
  --already-quantized --application-parameters external \
  --application-name capsule \
  --board sophialake --host-memory-reserved-bytes "$MEASURED_HOST_BYTES" \
  --io-memory-bytes "$MEASURED_STAGING_BYTES" --out "$OUTPUT"
```

With measured host prefix `H` and staging demand `S`, the board expands to `io_memory_base = io_memory_capacity_base = 0x80000000 + H`, `io_memory_capacity_bytes = (1 << 30) - H`, and `io_memory_bytes = S`, with `--application-residency --quantized-operand-reuse wide-cache`. For the recipe at its 768 KiB window, replace `--board sophialake` with `--recipe dispatch-reduction`; this leaves residency disabled unless selected with an explicit physical capacity.

The application memory budget stays explicit because the install does not determine it. Captures with lifted mutable buffers additionally require a source-bound state declaration. The measured host prefix must cover the actual host footprint before deployment.
