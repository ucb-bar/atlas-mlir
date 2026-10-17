# Captured attention + MLP

One static 32×32 block: Q/K/V projections, attention with row softmax, a
BF16-to-FP8 pack, and a two-layer ReLU MLP, from the Linalg capture
`test/examples/captured_attention_mlp32_linalg.mlir` (or `..._two_outputs_...`,
which also returns the MLP intermediate) and its checked-in weights. The f32
math becomes FP8 operands and BF16 intermediates under
`test/examples/atlas_attention_mlp_numeric_policy.json`, so results differ from
f32. Atlas ReLU also turns `-0.0` into `+0.0`.

| File | Use |
| --- | --- |
| [`00-atlas-virtual-ssa.mlir`](00-atlas-virtual-ssa.mlir) | Two-output virtual program to read. Not a build input. |
| [`build_fixture.sh`](build_fixture.sh) | Linalg → virtual SSA → machine MLIR → `program.bin`. |
| [`layout.json`](layout.json), [`two-output-layout.json`](two-output-layout.json) | Capsule DRAM layouts for one or two outputs. |
| [`host.c`](host.c) | Directed host test: fixed inputs, three launches, checks every output cell. |
| [`host_graph_mailbox.c`](host_graph_mailbox.c), [`build_graph_mailbox.sh`](build_graph_mailbox.sh) | Mailbox host ELF that takes runtime inputs. |
| [`toolchain.md`](toolchain.md) | Pinned LLVM source and host build setup. |
| [`start.S`](start.S), [`link.ld`](link.ld) | Bare-metal startup (machine mode) and linker script at `0x80000000`. |

Run from the repository root, with a fresh output directory each time:

```sh
export PYTHON_BIN=python3              # with the compiler Python dependencies
ATLAS_BIN=build/bin             # atlas-opt, atlas-emit
LLVM_BIN=/path/to/llvm23-install/bin  # clang, llvm-nm; see toolchain.md
RUN=out/captured-attention-mlp
```

## Build the device program

```sh
examples/handoff/captured_attention_mlp/build_fixture.sh "$ATLAS_BIN" "$RUN/compiled"
examples/handoff/captured_attention_mlp/build_fixture.sh "$ATLAS_BIN" "$RUN/two-results" --two-output
```

The compiler converts boundary inputs and outputs with `--lower-atlas-virtual-boundary-to-dma`,
then verifies and schedules virtual SSA before allocation, inserts DMA
waits, then runs `--schedule-atlas-stream` and verifies timing and the generated
machine schedule before emission.

Each output has `atlas-virtual.mlir`, `atlas-machine.mlir`,
`physical-program.json`, `program.words.txt`, and `program.bin`. To make an RTL
fixture (`--panel` is `one`, `half`, or `striped`; use a unique `--name`):

```sh
$PYTHON_BIN test/rtl/make_captured_attention_mlp_fixture.py \
  --compiled "$RUN/two-results" --out "$RUN/fixture-half" \
  --name attention_mlp_two_half --panel half --two-output
```

Run fixtures on the standalone core as described in
[hardware status](../../../docs/hardware-status.md). To check the binary against
the RTL decoder (`ATLAS_RTL` is an atlas-npu checkout at the pinned revision):

```sh
$PYTHON_BIN tools/check_program_decode.py \
  --program "$RUN/compiled/physical-program.json" --binary "$RUN/compiled/program.bin" \
  --decoded-out "$RUN/compiled/binary-decoded.json" \
  --rtl-instructions "$ATLAS_RTL/src/main/scala/atlas/scalar/Instructions.scala" \
  --rtl-decode "$ATLAS_RTL/src/main/scala/atlas/scalar/IDecode.scala"
```

## Directed host ELF

```sh
$PYTHON_BIN tools/atlas_boot_pack.py --source "$RUN/compiled/atlas-machine.mlir" \
  --atlas-emit "$ATLAS_BIN/atlas-emit" \
  --layout examples/handoff/captured_attention_mlp/layout.json \
  --import-manifest "$RUN/compiled/import-manifest.json" --out "$RUN/capsule"
mkdir -p "$RUN/host" "$RUN/.tmp"
$PYTHON_BIN tools/atlas_captured_host_data.py --compiled "$RUN/compiled" \
  --capsule "$RUN/capsule" --out "$RUN/host/atlas_captured_data.h"
$PYTHON_BIN tools/atlas_host_embed.py --capsule "$RUN/capsule" \
  --host-map runtime/selected-host-map.json --out "$RUN/host/atlas_program.h"
TMPDIR="$RUN/.tmp" "$LLVM_BIN/clang" --target=riscv64-unknown-elf \
  -march=rv64imac -mabi=lp64 -mcmodel=medany -O2 -ffreestanding -fno-builtin \
  -fno-stack-protector -fno-pic -nostdlib -fuse-ld=lld \
  -DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1 -DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1 \
  -DATLAS_MAX_POLLS=10000000 -DATLAS_STANDALONE_MEMOPS=1 -I runtime -I "$RUN/host" \
  examples/handoff/captured_attention_mlp/start.S \
  examples/handoff/captured_attention_mlp/host.c runtime/atlas_host.c \
  -Wl,-T,examples/handoff/captured_attention_mlp/link.ld -Wl,--no-relax \
  -o "$RUN/host/shuttle-host.elf"
```

The ELF launches the program three times (FP8 inputs 0.5, 1.0, then 0.5/1.0
alternating by column; expected outputs 0.0625, 0.125, then 0.125/0.0625), checks
every cell and a guard, and leaves `atlas_handoff_status` (0 = pass). For two
outputs, use `two-output-layout.json` and the `--two-output` build.

## Mailbox host ELF

This ELF takes six runtime f32 tiles per request through a mailbox. It needs a
Merlin-native compilation of the two-output source (see
[native Merlin binding](../../../docs/native-merlin-binding.md)), for example
with `--diagnostic-graph-policy test/examples/attention_mlp_graph_rounded_pack.json`,
in `$COMPILED`. `MERLIN_ROOT` is the Merlin checkout.

```sh
# usage: build_graph_mailbox.sh MERLIN_ROOT PYTHON LLVM_BIN SOURCE_LINALG COMPILED_NATIVE FRESH_OUT
#          [--platform HOST_MAP SOC_MEMMAP CACHE_SYNC_C LINKER_SCRIPT MAX_POLLS]
examples/handoff/captured_attention_mlp/build_graph_mailbox.sh \
  "$MERLIN_ROOT" "$PYTHON_BIN" "$LLVM_BIN" \
  test/examples/captured_attention_mlp32_two_outputs_linalg.mlir \
  "$COMPILED" "$RUN/graph-mailbox"
```

Without `--platform`, the build uses the source-derived map, assumes coherent
memory, and polls up to 10,000,000 times (`handoff-status.json` says
`diagnostic_unbound`). With `--platform`, it checks the map, SoC memmap, and
final ELF; see [building a board ELF](../../../docs/host-launch.md#building-a-board-elf).
The build writes `host/shuttle-graph-host.elf` and `host/handoff-status.json`
(ELF hash, mailbox address and offsets, output count). The mailbox protocol and
status codes are in [host launch](../../../docs/host-launch.md#graph-mailbox).

## Check a returned mailbox dump

After a run, save the whole mailbox for each request. Make the expected output
for the same inputs with `test/rtl/make_graph_target_fixture.py --compiled
"$COMPILED" --case CASE --out DIR` (cases are in
`test/examples/attention_mlp_graph_rtl_suite.json`), then:

```sh
$PYTHON_BIN tools/check_graph_mailbox_dump.py \
  --handoff-status "$RUN/graph-mailbox/host/handoff-status.json" \
  --execution-plan "$COMPILED/execution_plan.json" \
  --elf "$RUN/graph-mailbox/host/shuttle-graph-host.elf" \
  --program "$COMPILED/program.bin" \
  --dump mailbox.bin --inputs inputs.bin --expected DIR/expected.bin \
  --request-id 1 --out "$RUN/mailbox-check.json"
```

`--inputs` is the six f32 tiles the runner wrote, concatenated (24,576 bytes).
The checker compares request/response IDs, status, inputs, both guards, and
every output byte. It exits 0 with `reported_match_unqualified` on a match and
2 otherwise.

## Baremetal IDE

For the [sp25 Baremetal IDE](https://github.com/ucb-ee290c/sp25-Baremetal-IDE/tree/963c68fd25983b9989939e3cb3aeb08310f19edc),
add an Atlas platform from the board's memory map, build `host.c` and
`runtime/atlas_host.c` with the IDE's startup, linker script, and libc (drop
`start.S`, `link.ld`, and `ATLAS_STANDALONE_MEMOPS`), and pass `-DENABLE_RVV=OFF`
for a scalar build.

## Limits

The device program has run on the standalone Atlas core with directed panels.
The host ELFs have been linked and inspected but have not run on the Shuttle +
Atlas SoC or an FPGA. `runtime/selected-host-map.json` has no bitstream; bind
it before any board run. A plain Spike run cannot execute Atlas MMIO or DMA.
