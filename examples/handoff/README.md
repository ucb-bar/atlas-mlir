# Atlas → LLVM → RISC-V inspection examples

These fixed 32×32 programs target the selected Atlas RTL. They show a
*machine-code handoff* through the unmodified LLVM/MLIR tools used by this
build. `mlp_tile` and `attention_tile` start from hand-authored Atlas machine
streams. Their authored inputs live under `test/examples/handoff_*_tile.mlir`;
stage 01 is the checked-in source copy.
The [virtual MLP](virtual_mlp/00-atlas-virtual-ssa.mlir) adds a stage 00: SSA
tensor values precede generated physical register and scratch assignments.
Its stage 01 is produced by `--lower-atlas-virtual-to-machine`, then follows
the same LLVM path. Its source is
[`virtual_fp8_two_layer_mlp.mlir`](../../test/examples/virtual_fp8_two_layer_mlp.mlir).
The [captured MLP bundle](captured_mlp/00-linalg.mlir) starts from a sanitized
Model2MLIR Linalg graph and public frozen parameters. Its
[compiler manifest](captured_mlp/compiler-manifest.json) maps source arguments
to physical input/constant slots and records the LLVM object and ELF hashes.
It does not contain sample inputs or goldens.

Open a directory in numeric order:

| Stage | File | What to inspect |
| --- | --- | --- |
| 00 | [Virtual MLP SSA](virtual_mlp/00-atlas-virtual-ssa.mlir) | `%x`, `%h0`, `%h1`, `%h2`, and `%y` are logical tensor values; no physical register numbers. |
| 00→01 | [Captured MLP Linalg](captured_mlp/00-linalg.mlir) → [virtual SSA](captured_mlp/01-atlas-virtual-ssa.mlir) | Parsed transpose, zero-fill, matmul, broadcast bias, and ReLU become virtual operations under the proposed precision policy. |
| 01 | [MLP Atlas](mlp_tile/01-atlas-machine.mlir) · [attention Atlas](attention_tile/01-atlas-machine.mlir) · [virtual MLP Atlas](virtual_mlp/01-atlas-machine.mlir) | Typed `atlas.*` operations and ordered `!atlas.state` chain. |
| 02 | [MLP structured LLVM dialect](mlp_tile/02-llvm-structured.mlir) · [attention structured LLVM dialect](attention_tile/02-llvm-structured.mlir) | One `llvm.call @atlas_emit_*` per instruction, with physical fields, checked word index, conservative effects, and unknown availability. These calls are compiler markers; run the finalizer before LLVM IR translation. |
| 03 | [MLP encoded LLVM dialect](mlp_tile/03-llvm-encoded.mlir) · [attention encoded LLVM dialect](attention_tile/03-llvm-encoded.mlir) | Finalizer rechecks the stream and emits one side-effecting inline-assembly block. |
| 04 | [MLP LLVM IR](mlp_tile/04-llvm-ir.ll) · [attention LLVM IR](attention_tile/04-llvm-ir.ll) | LLVM IR translated by `mlir-translate`. |
| 05 | [MLP RV32 assembly](mlp_tile/05-riscv-words.s) · [attention RV32 assembly](attention_tile/05-riscv-words.s) | `llc` output. Atlas words appear as `.word` directives, followed by LLVM's unreachable `ret`. |
| 06 | [MLP relocatable object](mlp_tile/06-riscv-relocatable.o) · [attention relocatable object](attention_tile/06-riscv-relocatable.o) | ELF32 RISC-V `ET_REL` emitted by LLVM. |
| 07 | [MLP linked ELF](mlp_tile/07-riscv-linked.elf) · [attention linked ELF](attention_tile/07-riscv-linked.elf) | ELF32 RISC-V `ET_EXEC` linked with `ld.lld`; `atlas_program` entry is at address zero. |
| 08 | [MLP disassembly](mlp_tile/08-riscv-disassembly.txt) · [attention disassembly](attention_tile/08-riscv-disassembly.txt) | `llvm-objdump -d` view of the linked `.text`. Generic RISC-V decoding does not explain Atlas custom operations. |

The [MLP word map](mlp_tile/atlas-word-map.json), [attention word map](attention_tile/atlas-word-map.json),
and [virtual MLP word map](virtual_mlp/atlas-word-map.json)
connect each emitted word and byte offset back to its Atlas operation and
attributes. The paired `atlas-words.txt` files contain the independent
`atlas-emit` stream. These sidecars retain structure that LLVM's inline
assembly loses; they are the starting point for source-linked delay analysis.

For example, the MLP Atlas source contains
`"atlas.mxu_matmul"(...){unit = 0, ...}`. The structured LLVM dialect has a
corresponding `llvm.call @atlas_emit_mxu_matmul` with the same physical fields.
The final LLVM dialect stage contains one side-effecting `llvm.inline_asm`;
the LLVM IR represents that as `call void asm sideeffect`. `llc` prints the
same words in [stage 05](mlp_tile/05-riscv-words.s), and
[stage 08](mlp_tile/08-riscv-disassembly.txt) shows their addresses and bytes
inside the linked ELF. The [word map](mlp_tile/atlas-word-map.json) identifies
which Atlas operation produced each address. LLVM does not retain the names of
the MXU, VPU, or DMA operations in the assembly itself.

The MLP fixture computes one quantized two-layer tile:
`MXU0(X,W0) → BF16 ReLU → E8M0 pack → physical row relayout → MXU1(H,W1)`.
It has no bias, batches, tails, or generic shapes. The attention fixture uses
already supplied `Q`, transposed `K`, and `V` tiles:
`QKᵀ → row max/subtract → approximate scale → exp2 → row sum/reciprocal →
multiply → pack/relayout → PV`. It has no Q/K/V projections, mask, causal
state, batching, or qualified full-domain softmax policy. Both contain authored
diagnostic delays; those are not proved availability bounds.
The virtual MLP expresses the same operation sequence through SSA. The OOT
lowerer generates 109 selected words and a VMEM relayout for packed FP8 rows.
The same LLVM object words executed on the selected standalone AtlasCore with
two different runtime weight sets, checking all 1,024 output cells each time.
The test covers a small exactly representable FP8 set and does not prove the
full numerical or timing domain.
The captured MLP includes two vector biases, packed as BF16 tiles. Its
[numbered stages](captured_mlp/00-linalg.mlir) continue through machine Atlas,
both LLVM MLIR forms, LLVM IR, assembly, object, and linked ELF. A fresh
Model2MLIR capture and compile generated the same executable text as the
sanitized public fixture. The selected standalone core matched all 1,024
cells of the original PyTorch output for two runtime inputs. See
[scope and reproduction](../../docs/captured-mlp-compiler.md).

## Reproduce and inspect

This snapshot used unmodified LLVM/MLIR `23.0.0git` at
`a47bddccec30255619bb8c37fa59700e661d4e66` for MLIR translation, code
generation, and disassembly. The available linker was Ubuntu `ld.lld 18.1.3`;
it only linked the already emitted ELF object. The exporter records the exact
tool versions and hashes in its invocation-owned `manifest.json`.

```sh
python tools/export_llvm_handoff.py \
  --atlas-bin-dir build/bin \
  --llvm-bin-dir /path/to/llvm-install/bin \
  --linker /path/to/ld.lld \
  --output-dir out/handoff-fresh

/path/to/llvm-install/bin/llvm-readelf -h out/handoff-fresh/handoff_mlp_tile.elf
/path/to/llvm-install/bin/llvm-objdump -d out/handoff-fresh/handoff_mlp_tile.elf
```

The exporter checks the `ET_EXEC` machine and entry, compares linked `.text`
byte-for-byte with the LLVM object's `.text`, and checks the leading words
against `atlas-emit`. The final LLVM `ret` follows Atlas `ecall`; it is not
executed in the selected-core diagnostic runs. The test also extracts `.text`
from each committed ELF and runs those words on the selected standalone core
for one directed input panel per program. This ELF is inspectable, but
it is **not** a generic RISC-V application or a complete Atlas host/SoC launch
artifact. The selected Atlas PC and custom instructions require the selected
core and its separate preload/start protocol. The
[handoff notes](../../docs/llvm-handoff-examples.md) give the execution evidence
and limits.
