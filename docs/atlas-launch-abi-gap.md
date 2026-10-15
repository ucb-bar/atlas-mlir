# Atlas reset-entry capsule and launch limits

The hand OOT dialect can lower a validated flat Atlas stream to an LLVM
function containing the exact selected instruction words. The
[complete-function test](llvm-boot-entry-observation.md) executes a 37-word
VSQUARE function body from IMEM word zero on the selected standalone
`AtlasCore`. `atlas-boot-pack` makes that narrow reset-entry contract explicit
and reproducible. A separate [mailbox-call observation](mailbox-call-observation.md)
executed the same LLVM-produced program twice on one standalone core
with changed runtime pointers and data. This is a bounded custom launch ABI;
it does not establish a C-callable function ABI or integrated Atlas runtime.

## Bounded reset-entry capsule

`build/bin/atlas-boot-pack` takes a relocatable RISC-V ELF object, its typed
Atlas MLIR source, and an authored DRAM layout declaration. It calls the
OOT `atlas-emit` checker on that source, then requires the complete object
`.text` to equal the checked words plus LLVM's final RET. It refuses a moved
entry symbol, a `.text` relocation, compressed/misaligned words, a missing
ECALL-before-RET ending, an oversized IMEM program, an overlapping DRAM
region, a stale output destination, or a source/object mismatch. It copies
only `.text` into `program.bin` and writes `manifest.json` with input/object/
packer hashes, target revision declaration, fixed DRAM regions, IMEM base,
entry word, start CSR, and ECALL completion word. Layout v2 additionally
declares one mailbox and disjoint input/output address pools, with u32 pointer
field offsets, fixed tensor byte length, and alignment. It validates pointer
arguments before the host writes a descriptor. The manifest marks that the
packer does not prove arbitrary program-to-mailbox dataflow. Runtime tensor
values and expected outputs are absent from either capsule.

For the [VSQUARE layout](../test/examples/vpu_square_boot_layout.json),
the standalone driver loads all 37 words at `imemTL` byte address `0x20000`,
preloads input DRAM `0x90000000` (2,048 bytes), starts through `csrTL`
address `0x18`, and checks output DRAM `0x90000800` (2,048 bytes) plus a
guard at `0x90001000`. The packaged program halted by ECALL at instruction
word 35; LLVM's RET at word 36 was not issued. One 1,024-cell square panel
matched the independent reference, preserved input and guard memory, and
observed 64 DMA reads and 64 writes on the selected standalone core.
`test/test_boot_capsule.py` checks both the package and actual execution.

Example invocation after `atlas-opt`, `mlir-translate`, and `llc` produce
`square.o`:

```sh
build/bin/atlas-boot-pack \
  --object square.o \
  --source test/examples/vpu_square_pair.mlir \
  --atlas-emit build/bin/atlas-emit \
  --layout test/examples/vpu_square_boot_layout.json \
  --llvm-bin /path/to/llvm-install/bin \
  --out out/square-boot-capsule
```

Earlier local diagnostic runs wrote to
`out/qualifications/oot-boot-capsule-r1/` and
`out/qualifications/oot-mailbox-call-r1/`. These ignored outputs are not
included in a fresh checkout; each invocation must regenerate its artifacts.

## Remaining obligations

| Boundary | Evidence now | Still required for a reusable ABI |
| --- | --- | --- |
| Scalar registers and stack | The tested mailbox stream initializes the registers it uses and halts; the packer does not prove register initialization for arbitrary source. | Define/reset initial registers, argument registers, clobbers, callee saves, stack, and live scalar DMA pointers. The emitted stream can overwrite ordinary RISC-V return and saved registers. |
| Arguments and results | A declared 32-byte DRAM mailbox supplies one runtime input pointer and one output pointer. Two same-core invocations with changed addresses/data passed the bounded test. | Bind general shapes, constants, state, multiple outputs, and errors to a stable invocation interface. Verify source-to-mailbox dataflow for arbitrary programs and check every read has an initialized source. |
| IMEM loading | The capsule checks one symbol spanning `.text`, zero text relocations, 32-bit words, and the selected 128-KiB IMEM capacity. ModeLIR loads its words over `imemTL`. | Implement a qualified loader/linker for multiple functions/sections, placement, relocation, capacity, and physical SoC launch. Non-text ELF sections are discarded here. |
| Completion and memory visibility | The standalone driver starts twice via CSR `0x18`, observes two ECALL halts, and sees the first output retained after the second run. The stream issues its own DMA waits. | Establish accelerator/system drain and memory visibility guarantees for all operations, exceptions, timeouts, and general repeated invocations. |
| Branch and jump relocation | `atlas-emit` validates in-block direct targets and delay-slot adjacency; the capsule binds the exact checked words. | Qualify Atlas instruction-word PC conventions under linking, program placement, cross-block branches, and control-flow effects. The OOT converter refuses JALR. |
| Numerical and platform scope | The capsule test executes one restricted VSQUARE panel on standalone `AtlasCore` with ModeLIR's TileLink driver. | Qualify full admitted numerics, all required modes, integrated SoC execution, model composition, and host/accelerator boundaries. |

The selected layout JSON declares a target revision, but the packer does not
prove that a later simulator or chip actually has that revision. The tests
bind it to a separately pinned ARC model. The capsule is not a C function
call, generic ELF loader, runtime scheduler, or full D/F/M/N result.
