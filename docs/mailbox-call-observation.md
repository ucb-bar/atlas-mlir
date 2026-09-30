# Bounded mailbox call on one selected standalone AtlasCore

The hand-authored OOT [typed program](../test/examples/vpu_square_mailbox.mlir)
uses a reset-entry mailbox to obtain runtime input and output **DRAM byte
addresses**. Its 39 Atlas instructions lower through the registered
Atlas-to-LLVM pass to one ordered LLVM inline-assembly block; LLVM appends an
unreached RET. The OOT emitter, selected Atlas assembler transcription, and
ELF32 RISC-V `.text` agree on all 39 words. `atlas-boot-pack` checks the source
against the complete object `.text` and publishes a 40-word `program.bin` and
a manifest using the [authored layout](../test/examples/vpu_square_mailbox_layout.json).
Neither the program nor capsule contains runtime input samples or
expected output tensors.

| Bound item | Contract exercised |
| --- | --- |
| Selected RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Standalone driver | `ModeLIR-atlas` `add52b0a7c96d72e0079b7938e07ca5080871e82` |
| Model observed separately | `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`; it is not the runtime oracle here |
| Mailbox | 32 bytes at `0x90000000`; little-endian u32 input pointer at byte 0, output pointer at byte 4 |
| Buffers | Each points to one 2,048-byte BF16 tensor, 32-byte aligned, inside disjoint declared input/output pools |
| Completion | The stream waits after its DMA transfers, writes the debug CSR, then halts by ECALL at PC word 38; LLVM RET at 39 is not issued |

The program DMA-loads the mailbox to VMEM, uses two scalar `LW` instructions
to read the pointers, and uses the loaded registers for input and output DMA.
The selected `LSU.scala` treats scalar addresses as bytes. The selected
`ScalarCore.scala` writes the loaded register only after the asynchronous
internal response. The program places `DELAY 8` after each `LW`, following the
separate [scalar-load timing discriminator](jalr-load-delay-observation.md).
This establishes only the tested standalone schedule; Atlas has no qualified
general scalar-load completion wait here.

The [test](../test/test_mailbox_call_reference.py) instantiated the selected
standalone core once, loaded IMEM once, then wrote the start CSR twice without
resetting the core or replacing code. Between starts the host changed both
mailbox pointers and the second input tensor. The two 1,024-cell input panels
came from the test's finite raw-bit generator; an independent exact-power
oracle supplied expected square results. Run 1 used input `0x90002000` and
output `0x90006000`; run 2 used input `0x90003000` and output `0x90007000`.
Each run halted after 390 observed cycles, 65 DMA read beats (one mailbox and
64 tensor), and 64 DMA write beats. Both 2,048-byte results matched their
independent expectations. Run 1 left run 2's output sentinel untouched; run 2
preserved run 1's result. Both inputs and a separate 32-byte guard remained
unchanged. The evaluator-owned runtime receipt at
`out/qualifications/oot-mailbox-call-r1/repeated-call-receipt.json` records
source/program/model hashes, addresses, counts, and output hashes.

The optional layout v2 validator checks pointer field placement, DRAM pool
overlap, alignment, buffer size, and each supplied pointer's membership in
its declared pool. Deliberate overlapping fields/pools and invalid runtime
pointers are rejected. The packer reports
`program_mailbox_binding_proved_by_packer: false`: source-to-interface binding
is supported by this source-linked execution test, not proved for arbitrary
typed programs by the packer.

The new focused suite passed 4/4 tests. A fresh source-linked CTest run passed
145/145 Python methods, with zero skips or failures. Logs and the packaged
object/program are under `out/qualifications/oot-mailbox-call-r1/` in this
worktree. The selected ARC model SHA-256 is
`196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`;
its state JSON SHA-256 is
`db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`.

This is a bounded **reset-entry launch ABI** for one fixed-size tensor program
on the standalone core. It is not a C/RISC-V callable function ABI: there is
no caller return, register-save convention, stack, dynamic shape, general
argument set, error result, linked multi-function image, or integrated SoC
launch qualification. The two starts demonstrate repeated invocation in the
same selected standalone core instance, not concurrency or full-system reuse.
Gate D and the complete Merlin D/F/M/N plan remain open.
