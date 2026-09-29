# LLVM-produced Atlas function as a bounded standalone boot entry

This diagnostic extends the [VSQUARE observation](vpu-square-observation.md)
from LLVM-object word comparison to actual standalone `AtlasCore` execution
of the complete LLVM-produced function body. It uses the ACT-independent hand
OOT `--convert-atlas-to-llvm` pass, `mlir-translate`, and pinned RISC-V LLVM
tools. It does not create a general Atlas launch ABI or run the function as a
C call.

The diagnostic boot-entry contract is fixed and explicit:

- Reset begins at `atlas_program` symbol offset zero, mapped to Atlas IMEM
  word zero through the standalone driver's `imemTL` base `0x20000`.
- The function has no arguments or return value. The stream itself sets its
  scalar registers; the driver preloads BF16 input halves at DRAM
  `0x90000000` and `0x90000400` and checks output halves at `0x90000800` and
  `0x90000c00`. A guard is preloaded at `0x90001000`.
- The driver starts the core through execution-control CSR `0x18` and waits
  for the stream's ECALL halt. There is no caller or return continuation.

`test/test_elf_entry_reference.py` builds an ELF32 little-endian RISC-V
relocatable object from the typed VSQUARE stream, then checks its structured
section/symbol/relocation metadata before loading code. The object has one
global `atlas_program` function occupying all 148 bytes of `.text` from
offset zero: 36 Atlas instruction words and the LLVM-added 32-bit RET. There
are no `.text` relocations. The object has an `.eh_frame` relocation, which
the diagnostic loader does not use. The test rejects a moved symbol,
truncated section, a text relocation, or compressed/misaligned words.

The driver loads **all 37 words** extracted from that function's `.text` into
IMEM. One 1,024-cell VSQUARE panel matched all 2,048 expected output bytes;
the complete input and 32-byte guard were preserved, and the core halted
after 64 DMA reads and 64 writes. The selected core's stage-1 PC probe uses
instruction-word indices: it observed ECALL at index 35 and never observed
the appended RET at index 36. The first diagnostic run failed only a test
assertion that incorrectly treated that probe as byte addresses; its log is
retained next to the passing corrected run under
`out/qualifications/oot-vpu-square-elf-entry-r1/`.

This loads code bytes from an ELF object after checking its symbol and
relocations; it is not an ELF linker or loader. It does not establish safe
standard RISC-V call/return register use, dynamic arguments, relocations,
stack setup, a reusable Atlas ABI, integrated SoC execution, or a complete
compiler. The RET is unreachable only because this specific stream halts by
ECALL first. The selected standalone core and ModeLIR TileLink driver are
the execution tier. Full D/F/M/N gates remain unqualified.
