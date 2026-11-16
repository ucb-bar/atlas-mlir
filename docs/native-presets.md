# Native compile presets

The selected Atlas support provider owns `--recipe dispatch-reduction` and `--board sophialake`. Merlin registers provider arguments through optional `configure_codegen_parser` and `expand_codegen_arguments` hooks. Atlas recipe options live in `atlas_mlir_native.options.CodegenOptions`; Merlin's options remain target neutral.

`dispatch-reduction` expands the qualified adjacent fusion selection, batch-panel operand reuse, exponent-page cache, VMEM input staging, optimized pipeline, static quantizer sizing, exact binary routes, lane grouping, binary and cast vector preflight, exact float unrolling, 128,000,000 executed instructions per launch, a 786,432-byte I/O window, and the device KV quantizer and value transpose. Every choice remains an individual option in the identity and application manifest. The recipe name does not enter identity. Explicit flags override expansion regardless of order, including Boolean overrides such as `--no-device-kv-quantizer` and `--no-exact-float-unroll`.

`sophialake` describes DDR at `0x80000000` with 1 GiB of physical capacity, and the Atlas DMA aperture at `0x80000000` with 2 GiB of addressable capacity. Physical capacity constrains allocations independently of the larger DMA aperture. It reserves a host range at DDR base, selects a 16 MiB staging window, and enables residency. `host_reserved_bytes` is an explicit description field with a conservative 64 MiB default. This is an assumption until the board team confirms the ELF, stack, and heap footprint. Override it with `--host-memory-reserved-bytes`; the range must be positive, tile aligned, and leave capacity for staging and residency.

The default usable capacity is consequently `[0x84000000, 0xc0000000)`, and staging starts at `0x84000000`. The board preset overrides the recipe's smaller staging default; explicit `--io-memory-bytes` overrides both. Explicit capacity overrides must stay outside the reserved host range and within physical DDR. Board identity equals the explicit geometry and residency equivalent; a board name is not an identity component.

Activate an install that sets `MERLIN_NATIVE_SNAPSHOT` to its checked native snapshot, or pass `--snapshot` explicitly. Snapshot/profile validation still runs. Native compile defaults to `--abi auto`, application compilation defaults to diagnostic mode, invocation-local exact materialization reuse is enabled, and region reuse stays disabled because residency embeds addresses. `--no-materialization-cache` disables reuse. Region compilation defaults to strict-native mode; specify another mode for diagnostic requests.

The decode fixture with both presets at the 16 MiB staging default currently refuses with `spill rewrite work limit reached`. Resident recipe qualification is blocked pending compiler cache sizing or output/reduction partitioning. The command documents the intended interface.

For the qualified layer capture, a resident DDR recipe compile is:

```sh
merlin-targetgen native-compile --engine merlin_native \
  --support atlas \
  --target-source "$ATLAS_ROOT" --compile-inputs "$CAPTURE" \
  --application --application-memory-bytes 536870912 \
  --already-quantized --application-parameters external \
  --application-name capsule \
  --recipe dispatch-reduction --board sophialake --out "$OUTPUT"
```

The command has 12 flags. Its equivalent with the fourteen recipe fields and five board fields (with the shared I/O size counted once), explicit snapshot, ABI, mode, and materialization-cache selection has 32 flags. This count treats each option name as one flag and excludes the executable, subcommand, and values. The application memory budget stays explicit because the install does not determine it. Captures with lifted mutable buffers additionally require a source-bound state declaration. This command requires the board reservation assumption to match the actual host allocation before deployment.
