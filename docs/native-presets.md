# Native compile presets

The selected Atlas support provider owns `--recipe dispatch-reduction` and `--board sophialake`. Merlin registers provider arguments through optional `configure_codegen_parser` and `expand_codegen_arguments` hooks. Atlas recipe options live in `atlas_mlir_native.options.CodegenOptions`; Merlin's options remain target neutral.

`dispatch-reduction` expands the qualified adjacent fusion selection, batch-panel operand reuse, exponent-page cache, VMEM input staging, optimized pipeline, static quantizer sizing, exact binary routes, lane grouping, binary and cast vector preflight, exact float unrolling, 128,000,000 executed instructions per launch, a 786,432-byte I/O window, and the device KV quantizer and value transpose. Every choice remains an individual option in the identity and application manifest. The recipe name does not enter identity. Explicit flags override expansion regardless of order, including Boolean overrides such as `--no-device-kv-quantizer` and `--no-exact-float-unroll`.

`sophialake` describes DDR at `0x80000000` with 1 GiB of physical capacity, and the Atlas DMA aperture at `0x80000000` with 2 GiB of addressable capacity. Physical capacity constrains allocations independently of the larger DMA aperture. It reserves a host range at DDR base, selects an 8 MiB staging window, and enables residency and wide-cache operand reuse. Other codegen choices retain their defaults unless selected explicitly. `host_reserved_bytes` is an explicit description field with a conservative 64 MiB default. This is an assumption until the board team confirms the ELF, stack, and heap footprint. Override it with `--host-memory-reserved-bytes`; the range must be positive, tile aligned, and leave capacity for staging and residency.

Counted quantized frames exclude resident operands from staging and host scratch. With the default 64 MiB host reservation assumption, the measured 22-layer TinyLlama decode capture requires 1,002,688,512 of 1,006,632,960 bytes at 8 MiB, leaving 3,944,448 bytes; an explicit 16 MiB window exceeds that capacity by 4,444,160 bytes. Execution was checked against Python npu_model only.

The default usable capacity is consequently `[0x84000000, 0xc0000000)`, and staging starts at `0x84000000`. Explicit flags override board defaults. Explicit capacity overrides must stay outside the reserved host range and within physical DDR. Board identity equals the explicit geometry, residency, and wide-cache equivalent; a board name is not an identity component.

Activate an install that sets `MERLIN_NATIVE_SNAPSHOT` to its checked native snapshot, or pass `--snapshot` explicitly. Snapshot/profile validation still runs. Native compile defaults to `--abi auto`, application compilation defaults to diagnostic mode, invocation-local exact materialization reuse is enabled, and region reuse stays disabled because residency embeds addresses. `--no-materialization-cache` disables reuse. Reuse rebinds single-tile bodies only; a staged resident-graph body, such as a masked softmax row loop, keeps its ordinary compile. This decline is temporary until staged-body rebinding (`bind_tile` per stage) replaces it. Region compilation defaults to strict-native mode; specify another mode for diagnostic requests.

`--recipe dispatch-reduction --board sophialake` is unsupported because of the known `spill rewrite work limit reached` failure observed with a 16 MiB staging window. It has not been requalified at the board's 8 MiB default. Argument expansion refuses the combination before loading the snapshot or compile inputs, including when individual flags override its defaults. No compiler limit is changed. Select the board alone for default codegen with wide-cache and residency, or the recipe alone at its 768 KiB staging window.

For a prepared layer capture, the board compile with default codegen and wide-cache is:

```sh
merlin-targetgen native-compile --engine merlin_native \
  --support atlas \
  --target-source "$ATLAS_ROOT" --compile-inputs "$CAPTURE" \
  --application --application-memory-bytes 536870912 \
  --already-quantized --application-parameters external \
  --application-name capsule \
  --board sophialake --out "$OUTPUT"
```

The board flag expands to `--io-memory-base 0x84000000 --io-memory-capacity-base 0x84000000 --io-memory-capacity-bytes 1006632960 --io-memory-bytes 8388608 --application-residency --quantized-operand-reuse wide-cache` under the default host reservation. For the recipe at its 768 KiB window, replace `--board sophialake` with `--recipe dispatch-reduction`; this leaves residency disabled unless selected with an explicit physical capacity.

The application memory budget stays explicit because the install does not determine it. Captures with lifted mutable buffers additionally require a source-bound state declaration. The board reservation assumption must match the actual host allocation before deployment.
