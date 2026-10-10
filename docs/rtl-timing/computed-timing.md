# Computed engine timing

The [extractor](../../tools/rtl_extract/README.md) derives per-operation engine timing deterministically from retained CIRCT HW IR. It exports the engine modules to JSON, then executes their control logic cycle by cycle: it issues one command at age 0, feeds on-chip SRAM responses back, and records when each request, write and busy signal changes. No timing value is read from a performance model or typed in by hand; target knowledge is limited to declarative per-engine specs (module, ports, opcodes, stimulus) under `tools/rtl_extract/targets/atlas/`.

## Results

All values below are computed from the debug lowering of the selected FIRRTL and equal the compiler's built-in rules in `lib/AtlasTiming.cpp`.

| Operation | Reads | Writes | Next issue / free |
| --- | --- | --- | --- |
| `vload` / `vstore` | 1–32 | 3–34 | 35 |
| `vtrpose.xlu` | 1–32 | 34–65 | 66 |
| VPU elementwise (incl. `vmul.bf16`) | 0–63 | 2–65 | 65 |
| VPU row sum / row max, min | 0–31 | 7–38 / 2–33 | 38 / 33 |
| VPU column reductions | 0–127 | 66–129 | 129 |
| VPU pack / unpack | 0–63 / 0–31 | 3–65 (step 2) / 3–66 | 65 / 66 |
| VPU immediate pair / single | — | 1–64 / 1–32 | 64 / 32 |
| MXU0 / MXU1 matmul | 0–31 | accumulator 63–94 / 3–34 | — |
| Scalar load | 1 | 3 | 3 |

The extractor also reports MXU push/pop streams and busy windows, response-latency and operand variants, and an MXU1 accumulate boundary case.

## Scope and assumptions

- On-chip VMEM and MREG reads take one cycle: the IR declares them `seq.firmem 1, 1`, and the LSU is always granted ahead of DMA and TileLink. Records report this as `scratchpad_read_latency`.
- External memory and DMA completion are not modeled. They are variable and governed by `dma.wait`.
- Registers without a reset value are accepted only when listed in a spec's `unreset`, each paired in Chisel with a reset valid bit; every control output must still be known at every age.
- Each record describes one operation issued from an idle engine. Overlap between operations, control flow and DMA lifetimes remain the compiler's scheduling rules.

## Checks

- Synthetic tests cover the interpreter, the coupled two-circuit recipe and spec validation without hardware IR.
- Hardware tests reproduce every value above, and mutations (an extra register stage, a payload-dependent control path) move or reject the result.
- The [compiler cross-check](../../tools/rtl_extract/README.md#compiler-cross-check) compares every mapped record with the compiler's footprints and fails on any difference.

## Next steps

- Load the computed facts in the compiler's selected-timing path instead of fixed trace values.
- Derive more spec inputs from the IR: memory latency from `seq.firmem`, opcode encodings from the decoder and command ports from their valid/ready naming.
- Add a pair recipe for overlap checks.
- Agree the `atlas.op_timing.v0-proposal` record shape with the Merlin Phase 0 extractor.
