# Atlas examples

Each example keeps only its source files here. Generated files go under `out/`
and are rebuilt by the commands in each README.

| Example | Source | What it shows |
| --- | --- | --- |
| [MLP tile](mlp_tile/01-atlas-machine.mlir) | Atlas machine MLIR | Hand-written 32×32 instruction stream through LLVM to ELF. |
| [Attention tile](attention_tile/01-atlas-machine.mlir) | Atlas machine MLIR | Same, for an attention tile. |
| [Virtual MLP](virtual_mlp/00-atlas-virtual-ssa.mlir) | Atlas virtual SSA | SSA values before physical placement. |
| [Captured MLP](captured_mlp/00-linalg.mlir) | Linalg | Linalg import with an FP8/BF16 precision policy. |
| [Captured attention + MLP](captured_attention_mlp/README.md) | Linalg | Device program, directed host ELF, mailbox host ELF. |
| [Native attention + MLP](native_attention_mlp/README.md) | Typed Merlin request | Merlin-native image launched from a host ELF. |
| [Ordered M/N matmul](ordered_mn/README.md) | Linalg 64×96×64 | One tile program launched six times to build a larger output. |
| [Source f32 matmul](source_mn/README.md) | Verified native tiles | f32 host bundle, MLIR callback, and model link. |

Smaller single-operation inputs are in [`test/examples/`](../../test/examples).

## Atlas, LLVM, assembly, and ELF stages

Build the tools first (see the [root README](../../README.md)), then run from
the repository root with a fresh output directory:

```sh
python3 tools/export_llvm_handoff.py \
  --atlas-bin-dir build/bin \
  --llvm-bin-dir /path/to/llvm-install/bin \
  --linker /path/to/ld.lld \
  --output-dir out/handoff
/path/to/llvm-install/bin/llvm-objdump -d out/handoff/handoff_mlp_tile.elf
```

For the MLP tile, attention tile, and virtual MLP, the exporter writes
`.atlas.mlir`, `.llvm-structured.mlir` (one call marker per Atlas instruction),
`.llvm.mlir` and `.ll` (one inline-assembly word block), `.s`, `.o`, `.elf`,
`.disasm.txt`, `.word-map.json` (instruction to source operation),
`.physical-program.json`, and a manifest. The virtual example also writes
`.virtual.mlir`. The exporter checks that the object and ELF text match the
emitted words. The [compiler pipeline](../../docs/compiler-pipeline.md) covers
the captured MLP stages.

These ELFs are for inspection. A RISC-V CPU cannot run the Atlas words; the
Atlas core needs its own load-and-start sequence (see
[host launch](../../docs/host-launch.md)).
