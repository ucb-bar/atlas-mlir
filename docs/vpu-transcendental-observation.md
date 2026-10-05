# Bounded VPU transcendental observation

The selected RTL revision `0079c0541111197741a231c002e3843fa6f545b2`
implements VEXP, VSIN, VCOS, VTANH, and VLOG2 with distinct LUT paths. The
inspected `npu_model` classes instead call Torch transcendental functions.
These implementations are not interchangeable numerical specifications.

The [executable test](../test/test_vpu_transcendental_reference.py) replaces
only the unary mode in the existing two-half BF16 program. For each of the
five modes, the hand OOT emits 36 words matching both the selected assembler
and an object produced through unmodified LLVM. Parser/printer and illegal
odd BF16 pair tests run for every mode. Two 1,024-element panels per mode
execute on the selected standalone AtlasCore with DMA round trips. The test
checks all 10,240 output elements, the unchanged 2 KiB input, and a 32-byte
guard.

The independent Python check rounds ordinary mathematical results to BF16
for eight finite input codes in `[0, 5]`. VLOG2 also checks the selected
zero input result of negative infinity. Exact anchors include exp(0)=1,
sin(0)=0, cos(0)=1, tanh(0)=0, and log2 of 1/2, 1, 2, and 4. Every output in
these panels was within one adjacent BF16 code of the independently rounded
mathematical value. This **one-code bound is a diagnostic observation**, not
a released error contract. The selected RTL differs by one code from rounded
mathematics for some non-anchor VSIN, VCOS, and VLOG2 inputs; for example,
sin(2) produced `0x3f68` instead of rounded `0x3f69`.

The test pins the RTL checkout but does not independently re-attest the
prebuilt ARC model to that revision. It does not cover negative trigonometric
inputs, all BF16 codes, exceptional values, LUT interpolation and rounding
across the full domain, port conflicts, or integrated EE290 execution. These
five modes remain software-unadmitted and blocked for the D gate.
