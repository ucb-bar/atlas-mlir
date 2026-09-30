# Bounded VLOAD/VSTORE observation

`test/examples/vload_vstore.mlir` is a hand-authored typed Atlas stream,
transcribed separately in `test/examples/vload_vstore.S`. It loads 1 KiB of
DRAM into VMEM, moves that raw byte tile into MREG 4 with VLOAD, then writes
the same MREG into two distinct VMEM windows with VSTORE. DMA copies both
windows back to separate DRAM outputs. This makes the reference relation a
byte-for-byte identity, including every byte encoding, with no floating-point
assumptions.

`test_vload_vstore_reference.py` compares all 29 source-emitted words with the
selected assembler and the actual LLVM object, checks parser/printer round
trip and rejected register/offset fields, and executes three 1 KiB panels on
the selected-source-linked standalone `AtlasCore` ARC model. Every output
byte, the input, and a separate DRAM guard are checked. The selected core
completed the stream with 32 DRAM reads and 64 DRAM writes per panel.

This establishes bounded raw transfer behavior for the exercised register,
addresses, format, and conservative diagnostic delays. It does not establish
all legal address/offset combinations, arbitrary issue timing, memory-bank
overlap behavior, or integrated SoC execution. The selected RTL's `LSU.scala`
requires 1 KiB aligned transfer bases and a transfer contained in one VMEM
bank; the present MLIR verifier only checks the encoded signed 32-byte offset
and register fields. Runtime address legality therefore remains an explicit
compiler obligation.
