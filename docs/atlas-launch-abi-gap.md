# Atlas launch ABI gap after the LLVM boot-entry check

The hand OOT dialect can lower a validated flat Atlas stream to an LLVM
function containing the exact selected instruction words. The
[complete-function test](llvm-boot-entry-observation.md) executes a 37-word
VSQUARE function body from IMEM word zero on the selected standalone
`AtlasCore`. A new `atlas-boot-pack` tool makes that narrow reset-entry
contract explicit and reproducible. Neither result establishes a callable
function ABI or an integrated Atlas runtime.

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
entry word, start CSR, and ECALL completion word. Runtime tensor values and
expected outputs are absent from the capsule.

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

The current diagnostic artifact is under
`out/qualifications/oot-boot-capsule-r1/`; output paths belong to the
invocation, not the source tree.

## Remaining obligations

| Boundary | Evidence now | Still required for a reusable ABI |
| --- | --- | --- |
| Scalar registers and stack | This stream writes fixed Atlas registers and halts. The manifest marks register initialization unproved. | Define/reset initial registers, argument registers, clobbers, callee saves, stack, and live scalar DMA pointers. The emitted stream can overwrite ordinary RISC-V return and saved registers. |
| Arguments and results | The reset entry declares no call arguments or returns. Inputs and outputs use fixed DRAM addresses. | Bind dynamic pointers, shapes, constants, state, multiple outputs, and errors to a stable invocation interface. Check that every read has an initialized source. |
| IMEM loading | The capsule checks one symbol spanning `.text`, zero text relocations, 32-bit words, and the selected 128-KiB IMEM capacity. ModeLIR loads its words over `imemTL`. | Implement a qualified loader/linker for multiple functions/sections, placement, relocation, capacity, and physical SoC launch. Non-text ELF sections are discarded here. |
| Completion and memory visibility | The standalone driver starts via CSR `0x18` and observes halt after ECALL. This program issues its own DMA waits. | Establish accelerator/system drain and memory visibility guarantees for all operations, exceptions, timeouts, and repeated invocations. |
| Branch and jump relocation | `atlas-emit` validates in-block direct targets and delay-slot adjacency; the capsule binds the exact checked words. | Qualify Atlas instruction-word PC conventions under linking, program placement, cross-block branches, and control-flow effects. The OOT converter refuses JALR. |
| Numerical and platform scope | The capsule test executes one restricted VSQUARE panel on standalone `AtlasCore` with ModeLIR's TileLink driver. | Qualify full admitted numerics, all required modes, integrated SoC execution, model composition, and host/accelerator boundaries. |

The selected layout JSON declares a target revision, but the packer does not
prove that a later simulator or chip actually has that revision. The test
binds it to its separately pinned ARC model. The capsule is not a C function
call, generic ELF loader, runtime scheduler, or full D/F/M/N result.
