# Native attention + MLP host test

This example launches the Merlin-native 32×32 attention → BF16-to-FP8 →
two-layer MLP image through `runtime/atlas_host.c`. The program comes from the
typed request [`native_attention_mlp_request.json`](../../../test/examples/native_attention_mlp_request.json)
and the ABI and search limits next to it. The Q/K part came from parsed Linalg;
the rest was added by hand in the request. It is a directed tile, not a full
model compile.

[`host.c`](host.c) writes six FP8 input tiles and the generated BF16 constant,
loads the program, launches twice, and checks both BF16 output tiles and a
guard. Inputs and expected values are fixed in the source.

Build from the repository root with the Atlas support package and the Merlin
CLI installed. `ATLAS_OOT_BIN_DIR` holds `atlas-opt` and `atlas-emit`. Use a
fresh output directory:

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

The capsule's `program.bin` must match `compiled/program.bin`, and
`atlas_program.h` embeds those words. The two `ATLAS_DIAGNOSTIC_*` defines skip
the missing bitstream and assume coherent DRAM. For a board build, see
[host launch](../../../docs/host-launch.md#building-a-board-elf).

## Limits

The program has run on the standalone Atlas core with uniform and
alternating-feature panels. The host ELF has not run on the Shuttle + Atlas SoC.
The request's finite-input assumptions are checked for those panels only.
