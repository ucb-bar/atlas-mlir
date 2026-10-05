# Bounded DMA.CONFIG observation

The selected `atlas-npu` RTL revision
`0079c0541111197741a231c002e3843fa6f545b2` decodes
`DMA_CONFIG_ANY` with `funct7=0`. At issue, `ScalarCore.scala` copies the
selected scalar `rs1` value into one 32-bit `dmaBaseReg`. The following DMA
launch concatenates this register as the high half of its address. The
configuration's three-bit channel field does not select separate base state
in this RTL. The inspected `npu_model` defines per-channel classes with
`funct7=1`, so its encoder is not the selected executable encoding.

The [executable test](../test/test_dma_config_reference.py) checks four
programs spanning source registers x5/x6 and channel fields 0/7. Every typed
word matches the selected assembler and an object produced through
unmodified LLVM. The selected standalone AtlasCore captures each nonzero
source value in `scalar/dmaBaseReg`; the program then overwrites the source
scalar register with zero, and the base retains its original value. It also
checks the trap and that the post-trap instruction did not execute. Invalid
channel and register fields are rejected by the dialect verifier.

This is a bounded register-state observation. These programs do not launch a
DMA transfer with a nonzero high address, so concatenated address behavior,
asynchronous lifetime, all legal register/channel combinations, memory
visibility, and integrated EE290 execution remain unqualified. The test pins
the RTL checkout but does not independently re-attest the prebuilt ARC model.
Software admission and the D gate remain blocked.
