# Captured attention + MLP handoff

This example compiles one static 32×32 attention and MLP block: Q/K/V
projections, scaled dot-product attention with row softmax, and a two-layer
ReLU MLP. The checked-in Linalg program, argument manifest, and fixed weights
are under `test/examples/captured_attention_mlp32_*`. This branch starts from
that frozen source fixture; it does not ship the original PyTorch loader.

The importer applies the explicit policy in
`test/examples/atlas_attention_mlp_numeric_policy.json`: f32 source
arithmetic becomes FP8 operands, BF16 intermediates, and a BF16 VPU
exp/reciprocal/multiply softmax. This is a diagnostic precision transform;
it is not an equality-preserving lowering of arbitrary f32 attention.
The importer accepts positive-zero `linalg.fill` and `tensor.splat` matmul
initializers. It also recognizes the closed integer-zero → f32 →
ordered-less-than → select scalar ReLU pattern emitted by current model2MLIR.
Atlas ReLU clears the sign bit of negative zero, while that source select
preserves it; signed-zero and exceptional-value behavior need separate
numerical qualification.

The device path is `Linalg → Atlas virtual SSA → physical Atlas MLIR →
atlas-emit → program.bin`. It does not send Atlas instructions through LLVM.
The separate Shuttle host path compiles `host.c` and `runtime/atlas_host.c`
with an ordinary RV64 LLVM toolchain. Generated binaries and headers belong
under `out/`, not in this directory.

The Linalg importer preserves up to 16 ordered BF16 tile results in virtual
SSA. Packaging checks that every result fits the declared output DRAM region.
The generated host header declares every output address and size. This
example's `host.c` accepts the one- and two-result fixtures, passes every
declared output buffer to `atlas_host_start`, and checks each result after
completion. Other output counts need their own result checker.

## Inspect and generate the combined virtual program

[`00-atlas-virtual-ssa.mlir`](00-atlas-virtual-ssa.mlir) is an inspectable
two-output virtual SSA example. It contains Q/K/V projections, scaled
attention and row softmax, a BF16-to-FP8 pack before the MLP, and two MLP
matmuls with ReLU. It is a fixed example, not the input to the build command.
The virtual pack operation carries the E8M0 scale code; its exact or rounded
numerical policy is recorded in the typed request and `precision-plan.json`.
The [rounded diagnostic policy](../../../test/examples/attention_mlp_graph_rounded_pack.json)
selects RTL rounding, flush, and clamp behavior for varied BF16 producers.
To regenerate the program from the checked-in capture, weights, and numerical
policy, run from this repository's root with the compiler Python dependencies
installed:

```sh
PYTHON_BIN=python3 examples/handoff/captured_attention_mlp/build_fixture.sh \
  build/bin out/captured-attention-mlp/fixture-1
```

Use a fresh output directory for each run. The output
`out/captured-attention-mlp/fixture-1/atlas-virtual.mlir` contains Q/K/V,
scaled attention and row softmax, `atlas.virtual_pack_fp8` on the attention
result, and two MLP matmuls with ReLU. The same invocation emits
`atlas-machine.mlir`, `physical-program.json`, `program.words.txt`, and
`program.bin`. The virtual IR is a compiler input for placement; the physical
stream and binary are the inputs for functional-model and RTL checks.

For a two-result diagnostic, the checked-in
`test/examples/captured_attention_mlp32_two_outputs_linalg.mlir` returns the
ReLU MLP intermediate followed by the final tile. It differs from the
captured one-result source only at the function result and return statement.
Build it with the same weights and precision policy:

```sh
PYTHON_BIN=python3 examples/handoff/captured_attention_mlp/build_fixture.sh \
  build/bin out/captured-attention-mlp/two-results --two-output
python3 test/rtl/make_captured_attention_mlp_fixture.py \
  --compiled out/captured-attention-mlp/two-results \
  --out out/captured-attention-mlp/two-results-half \
  --name attention_mlp_two_half --panel half --two-output
```

The resulting RTL fixture has two 2,048-byte BF16 outputs in order. Generate
additional fixtures with `--panel one` and `--panel striped`, using a fresh
output directory and unique `--name` for each. The striped panel checks both
the ReLU intermediate and final tile with alternating input columns.
`two-output-layout.json` reserves both output tiles and a
guard for capsule packaging. The selected-core runner accepts these fixtures
through `--fixture`; [the selected-core result](../../../docs/captured-attention-mlp-core-observation.md#two-ordered-outputs)
records the passing cases. To build the two-output host ELF, use
`two-output-layout.json` for capsule packaging, generate
`atlas_captured_data.h` and `atlas_program.h` from that capsule, and compile
the same `host.c` and `runtime/atlas_host.c` shown below. The two-output host
checks two uniform inputs and both result tiles on each launch.

This fixture is one static 32×32 block with directed weights and an explicit
precision policy. The output files are local artifacts, not a qualified full
model or integrated SoC execution result.

The [selected-core diagnostic](../../../docs/captured-attention-mlp-core-observation.md)
shows how to turn this output into three mixed FP8/BF16 RTL fixtures. Two
uniform panels and one alternating-column panel have run on the selected
standalone `AtlasCore`; the
linked Shuttle host ELF has not run on the selected EE290 SoC or FPGA.

To package the one-result fixture with a diagnostic Shuttle host ELF, run
from the Atlas OOT repository root with fresh output directories:

```sh
PYTHON_BIN=/path/to/python-with-compiler-dependencies
ATLAS_BIN=/path/to/atlas-build/bin
LLVM_BIN=/path/to/llvm/bin
RUN=out/captured-attention-mlp
mkdir -p "$RUN"

PYTHON_BIN="$PYTHON_BIN" \
  examples/handoff/captured_attention_mlp/build_fixture.sh \
  "$ATLAS_BIN" "$RUN/compiled"

"$PYTHON_BIN" tools/atlas_boot_pack.py \
  --source "$RUN/compiled/atlas-machine.mlir" \
  --atlas-emit "$ATLAS_BIN/atlas-emit" \
  --layout examples/handoff/captured_attention_mlp/layout.json \
  --import-manifest "$RUN/compiled/import-manifest.json" \
  --out "$RUN/capsule"

mkdir -p "$RUN/host" "$RUN/.tmp"
"$PYTHON_BIN" tools/atlas_captured_host_data.py \
  --compiled "$RUN/compiled" --capsule "$RUN/capsule" \
  --out "$RUN/host/atlas_captured_data.h"
"$PYTHON_BIN" tools/atlas_host_embed.py \
  --capsule "$RUN/capsule" --host-map runtime/selected-host-map.json \
  --out "$RUN/host/atlas_program.h"

TMPDIR="$RUN/.tmp" "$LLVM_BIN/clang" --target=riscv64-unknown-elf -march=rv64imac -mabi=lp64 \
  -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-stack-protector \
  -fno-pic -nostdlib -fuse-ld=lld \
  -DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1 \
  -DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1 -DATLAS_MAX_POLLS=10000000 \
  -DATLAS_STANDALONE_MEMOPS=1 \
  -I runtime -I "$RUN/host" \
  examples/handoff/captured_attention_mlp/start.S \
  examples/handoff/captured_attention_mlp/host.c runtime/atlas_host.c \
  -Wl,-T,examples/handoff/captured_attention_mlp/link.ld -Wl,--no-relax \
  -o "$RUN/host/shuttle-host.elf"
```

`compiled/atlas-virtual.mlir` and `compiled/atlas-machine.mlir` show the two
Atlas levels. `compiled/physical-program.json` shows each emitted word and
its instruction fields; `compiled/program.words.txt` is the inspectable word
list. `capsule/program.bin` is the device image. Use
`$LLVM_BIN/llvm-objdump -d "$RUN/host/shuttle-host.elf"` for the host
assembly. The host ELF copies the compiled constants and launches the same
device image three times: FP8 inputs 0.5 and 1.0 in uniform tiles, then
0.5/1.0 alternating by column. It checks all 1,024 BF16 output cells and a
32-byte guard after each launch, then leaves a status word for a debugger.
The uniform outputs are exactly 0.0625 and 0.125; the alternating output
has 0.125 in even columns and 0.0625 in odd columns under the chosen weights.
With the two-output source and layout, the same host source checks the ReLU
intermediate and final tile on the two uniform inputs. A diagnostic RV64 Spike
run using a model-backed MMIO bridge passed both launches and matched the
selected-core fixtures. This did not execute the integrated Shuttle+Atlas SoC.

The bare-metal startup enables floating-point and vector state when the host
is built with those ISA extensions. It expects machine-mode entry. The
`rv64imac` build above has neither extension and emits no state setup.

Check the device image against the selected RTL decoder before handoff. Set
`ATLAS_RTL` to the checkout at the revision in the [repository README](../../../README.md):

```sh
"$PYTHON_BIN" tools/check_program_decode.py \
  --program "$RUN/capsule/physical-program.json" \
  --binary "$RUN/capsule/program.bin" \
  --decoded-out "$RUN/capsule/binary-decoded.json" \
  --rtl-instructions "$ATLAS_RTL/src/main/scala/atlas/scalar/Instructions.scala" \
  --rtl-decode "$ATLAS_RTL/src/main/scala/atlas/scalar/IDecode.scala"
```

This checks decoder modes and encoded operands against the structured stream.
`binary-decoded.json` contains modes and operands recovered from the binary
without using the stream's fields; it omits source and scheduling annotations.
The check does not execute the binary on RTL.

The host build command produces a **source-bound diagnostic ELF**. Its two
`ATLAS_DIAGNOSTIC_*` defines bypass the missing bitstream identity and assume
coherent DRAM. For an FPGA run, use the actual `EE290SimConfig` bitstream
manifest and generated Chipyard `*.memmap.json`. Record that file's SHA-256 as
`soc_memmap_sha256` in the reviewed host map. The graph mailbox build accepts
those files, a platform cache-maintenance callback, a linker script for the
selected Shuttle memory, and a measured poll bound. Set `MERLIN_ROOT` to the
Merlin checkout and `PYTHON` to its compiler environment. `SOURCE_LINALG` and
`COMPILED_NATIVE` must name the exact source and its native compilation:

```sh
bash examples/handoff/captured_attention_mlp/build_graph_mailbox.sh \
  "$MERLIN_ROOT" "$PYTHON" "$LLVM_BIN" "$SOURCE_LINALG" "$COMPILED_NATIVE" \
  "$FRESH_OUT" --platform "$HOST_MAP" "$SOC_MEMMAP" \
  "$CACHE_SYNC_C" "$LINKER_SCRIPT" "$MAX_POLLS"
```

This mode rejects an unbound map, a changed source/program identity, unmapped
host-visible IMEM/CSR or physical graph buffers, and unmapped final ELF load
segments. It writes `host/platform-preflight.json`, `host/platform-elf-check.json`,
and `host/handoff-status.json`. The host ELF then calls the supplied
`atlas_platform_cache_sync` before device reads and after device writes.
The callback must implement the selected Shuttle cache policy; merely linking
one does not qualify it. Host-visible address coverage also does not prove
Atlas DMA reachability. Confirm both with the bitstream build and runner before
executing. The final status remains
`platform_map_checked_execution_unqualified` until an actual platform run.
A regular Spike run cannot execute Atlas MMIO/DMA without a device adapter.

`runtime/selected-host-map.json` names `bringup-chipyard` revision
`245f2a9d` and its
[`EE290SimConfig`](https://github.com/ucb-ee194-tapeout/bringup-chipyard/blob/245f2a9d203be685176cc01e26e28029c5593202/generators/chipyard/src/main/scala/config/AtlasConfigs.scala)
in `config/AtlasConfigs.scala`. That revision pins the selected Atlas RTL
submodule. Its source-level tile composition does not
identify an FPGA bitstream or establish its decoded MMIO addresses; bind the
map from the actual build before using the host ELF on that platform.

## Baremetal IDE handoff

The [sp25 Baremetal IDE fork](https://github.com/ucb-ee290c/sp25-Baremetal-IDE/tree/963c68fd25983b9989939e3cb3aeb08310f19edc)
builds C applications with a selected chip platform and linker script. Its
published tree has no Atlas or EE290 platform entry. For an IDE build, create
an Atlas platform from the selected bitstream's memory map, replace the
default `app/src/main.c` in its `app` target with `host.c`, and add
`runtime/atlas_host.c` as a source. Add `runtime/` and the generated-header
directory to the target's include paths.
Use the IDE platform's startup, linker script, and C library; do not add
this example's `start.S` or `link.ld`, and leave
`ATLAS_STANDALONE_MEMOPS` undefined. Supply the qualified cache callback and
bitstream-bound map before enabling an FPGA run. The fork's cache-thrashing
benchmark helper is not a cache-maintenance callback for Atlas DMA.

The standalone ELF is built for `rv64imac/lp64`; the IDE's CMake flow uses
`lp64d` and defaults to RVV. For a scalar Shuttle host build, select
`-DENABLE_RVV=OFF` (or `-DRVV_TYPE=0`) when configuring the IDE. Rebuild the
C source under the selected platform instead of linking
objects from the two builds together. Alternatively, load the standalone ELF
only after checking its DRAM and MMIO map against the actual bitstream.

## Call the selected graph from compiled host MLIR

The native graph artifact can also be called through Merlin's typed host
wrapper. This path takes six rank-two f32 memrefs in source order, stages
their FP8 encodings and the generated BF16 constant, launches the 426-word
Atlas image, and returns the ReLU intermediate and final MLP result as two
ordered f32 memrefs. It is a diagnostic precision path; the f32 model's
quality and the integrated SoC execution are not qualified by this wrapper.

With a fresh rounded-policy native compilation in `$COMPILED`, create the
source-bound host declaration and Atlas callback in separate output roots:

```sh
merlin-targetgen native-bridge-graph \
  --linalg test/examples/captured_attention_mlp32_two_outputs_linalg.mlir \
  --entry forward --compiled "$COMPILED" \
  --device-symbol merlin_native_graph --out "$RUN/host-bridge"

PYTHONPATH="$PWD/python:$MERLIN_ROOT/src" python3 \
  tools/atlas_graph_model_bridge.py \
  --linalg test/examples/captured_attention_mlp32_two_outputs_linalg.mlir \
  --host-bridge "$RUN/host-bridge" --compiled "$COMPILED" \
  --out "$RUN/graph-wrapper"
```

`host-bridge.mlir` declares the graph call with MLIR's C interface and keeps
both outputs. Lower it with Merlin's host object path, then link the resulting
host object with the generated `atlas_graph_host_entry.c`,
`atlas_graph_model_wrapper.c`, `runtime/atlas_graph_memref.c`, and
`runtime/atlas_host.c`. The generated host entry forwards every declared
input and output memref to the lowered MLIR function. The wrapper needs the public
`merlin/runtime/c/finite_float_boundary.h` include path and
`ATLAS_DIAGNOSTIC_GRAPH_PRECISION=1`. Call the generated
`ATLAS_GRAPH_BINDING_FUNCTION` once with the platform's MMIO and DRAM hooks
before invoking the host entry. The generated manifest records the exact
source, native program, and host bridge hashes; changing any of these needs a
new wrapper output directory.

The linked host ABI test uses the lowered MLIR host entry, two successive
invocations, a fake device, and both returned results. It checks argument
ordering, FP8 staging, constant bytes, instruction loading, launch/wait, and
BF16 readback under sanitizers. The fake device supplies the results, so the
RTL suite above remains the execution check for the selected instruction
stream. The graph callback currently supports this fixed 32×32 BF16 pair-half
layout and one synchronous caller; it does not bind the EE290 bitstream map or
cache-maintenance method.

For a diagnostic RV64 ELF that calls this compiled host entry, use a fresh
output directory and the same source/native artifact. Here `COMPILED` is the
native compilation directory and `MERLIN_ROOT` is the Merlin checkout:

```sh
examples/handoff/captured_attention_mlp/build_graph_mailbox.sh \
  "$MERLIN_ROOT" "$PYTHON_BIN" "$LLVM_BIN" \
  test/examples/captured_attention_mlp32_two_outputs_linalg.mlir \
  "$COMPILED" "$RUN/graph-mailbox"
```

`graph-mailbox/host/shuttle-graph-host.elf` has an RV64 entry, the lowered
MLIR host object, generated Atlas callback, 426-word image, and a static
mailbox. `graph-mailbox/host/handoff-status.json` records the ELF hash,
mailbox address, byte offsets, and output count. The runner writes six 32×32 f32 input
tiles (five weights followed by invocation tokens), then a new nonzero
`request_id`. It leaves those buffers intact until the matching
`response_id` appears. A zero `status` means the host callback completed;
the ordered 32×32 f32 outputs then occupy the declared output window. The
compiler stages all six arguments on every invocation, so a new token tile
or new weights require no compiler edit.

The ELF records the mailbox base address in `handoff-status.json`. Relative
to that address, `request_id`, `response_id`, and `status` are 32-bit words at
offsets 0, 4, and 8. Six input tiles begin at offset 20; output tiles
begin at offset 24,596. Each tile is 4,096 bytes in row-major f32 order.
For `N` outputs, the trailing 32-bit guard is at offset `24596 + 4096*N`
and the mailbox occupies `24600 + 4096*N` bytes. Both guards must remain
`0x76a31c5b`. The two-output ELF therefore has a trailing guard at 32,788
and a 32,792-byte mailbox. The three-output residual-add example has a
trailing guard at 36,884 and a 36,888-byte mailbox.
`status=0xffffffff` means the request is active; other nonzero status values
mean a host, launch, or guard failure. The mailbox remains live through
repeated requests. Startup clears the mailbox, so wait until `guard_before`
reads `0x76a31c5b` before writing inputs. Use a new request ID for every
request: the host ignores an ID equal to the previous `response_id`, and a
reused ID would leave the earlier outputs in place.

The ELF uses source-derived IMEM/CSR addresses, assumes coherent DRAM, and
has a diagnostic poll bound. Its `diagnostic_unbound` status is intentional:
the selected EE290 bitstream map, Atlas DMA reachability, and Shuttle cache
maintenance must be supplied and checked before a real FPGA run. The linked
x86 host-call test exercises two invocations and buffer handling; this RV64
ELF has been linked and inspected but has not run on a Shuttle+Atlas SoC.

For an external run, save the complete mailbox after each response. The
`tools/check_graph_mailbox_dump.py` command checks an ELF and device-program
hash, request/response identity, status, all source input bytes, both guards,
and every output byte against a public physical BF16 fixture. Pass
`graph-mailbox/host/handoff-status.json`, `$COMPILED/execution_plan.json`,
the ELF, `$COMPILED/program.bin`, the mailbox dump, six concatenated source
f32 tiles, and the expected physical output written by
`test/rtl/make_graph_target_fixture.py` for the same inputs. The checker
returns a nonzero exit status for a mismatched request ID, status, guard,
input, or output, and labels matching dumps `reported_match_unqualified`.
Its report records the hashes of the dump, inputs, expected output, and
execution plan. It cannot detect a reused request ID from one dump; the
external run must separately record the bitstream identity and platform
execution trace.
