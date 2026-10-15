# Native attention and MLP host diagnostic

This example loads the Merlin-native 32×32 attention → BF16-to-FP8 →
two-layer MLP image through `runtime/atlas_host.c`. The virtual IR and device
binary come from the [typed request](../../../test/examples/native_attention_mlp_request.json)
and the checked ABI and search limits beside it. The Q/K prefix came from
parsed Linalg; the remaining attention and MLP nodes are typed additions.
This is a directed tile, not a complete captured-model compilation.

The host source writes six FP8 input tiles, loads the emitted Atlas words,
and checks both BF16 output tiles and a guard over two launches. Its fixed
inputs and expected values make it a diagnostic test program. The production
host route must supply model inputs and weights through its invocation ABI.
When a checked native program also declares a BF16 constant, the host copies
the generated payload to its selected DRAM address before each launch.

Build from the repository root with the Atlas support package and Merlin CLI
installed. Set `ATLAS_OOT_BIN_DIR` to the directory containing `atlas-opt`
and `atlas-emit`, and use a fresh output directory:

```sh
RUN=out/native-attention-mlp-host/example
mkdir -p "$RUN/.tmp"
merlin-targetgen native-build --engine merlin_native --support atlas \
  --cargo-target-dir "$RUN/cargo" --source-revision "$(git rev-parse HEAD)" \
  --out "$RUN/snapshot"
merlin-targetgen native-compile --engine merlin_native --support atlas \
  --snapshot "$RUN/snapshot" \
  --request test/examples/native_attention_mlp_request.json \
  --abi test/examples/native_attention_mlp_abi.json \
  --search-limits test/examples/native_attention_mlp_search_limits.json \
  --target-source . --mode diagnostic --out "$RUN/compiled"
merlin-targetgen native-verify --engine merlin_native \
  --snapshot "$RUN/snapshot" \
  --request test/examples/native_attention_mlp_request.json \
  --mode diagnostic --artifact "$RUN/compiled" --out "$RUN/verify.json"
python3 tools/atlas_boot_pack.py \
  --source "$RUN/compiled/atlas-machine.mlir" \
  --atlas-emit "$ATLAS_OOT_BIN_DIR/atlas-emit" \
  --layout examples/handoff/native_attention_mlp/layout.json \
  --out "$RUN/capsule"
python3 tools/atlas_captured_host_data.py \
  --compiled "$RUN/compiled" --capsule "$RUN/capsule" \
  --native-verification "$RUN/verify.json" \
  --out "$RUN/atlas_captured_data.h"
python3 tools/atlas_host_embed.py \
  --capsule "$RUN/capsule" --host-map runtime/selected-host-map.json \
  --out "$RUN/atlas_program.h"
TMPDIR="$RUN/.tmp" clang --target=riscv64-unknown-elf \
  -march=rv64imac -mabi=lp64 -mcmodel=medany -O2 -ffreestanding \
  -fno-builtin -fno-stack-protector -fno-pic -nostdlib -fuse-ld=lld \
  -DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1 \
  -DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1 -DATLAS_MAX_POLLS=10000000 \
  -DATLAS_STANDALONE_MEMOPS=1 -I runtime -I "$RUN" \
  examples/handoff/captured_attention_mlp/start.S \
  examples/handoff/native_attention_mlp/host.c runtime/atlas_host.c \
  -Wl,-T,examples/handoff/captured_attention_mlp/link.ld -Wl,--no-relax \
  -o "$RUN/shuttle-host.elf"
```

The capsule must contain the same `program.bin` as `compiled/`, and
`atlas_program.h` embeds those exact words. The current source-only host map
has no bitstream identity. The diagnostic defines above bypass that missing
identity and assume coherent DRAM; this ELF is not qualified for an FPGA run.
For that handoff, use the selected SoC's generated address map, measured poll
bound, and cache-maintenance callback as described in
[host launch](../../../docs/host-launch.md).

The directed program has run on the selected standalone `AtlasCore` with both
uniform and alternating-feature panels. The built RV64 ELF still needs an
integrated host/device execution test. The native request's finite-domain
premises are checked for these panels, not all runtime inputs.
