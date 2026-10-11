# Atlas MLIR documentation

Start with the [repository README](../README.md) for the build and the
[example guide](../examples/handoff/README.md) for reproducible Atlas, LLVM,
assembly, and ELF artifacts. Generated files belong under `out/`.

## Compiler interfaces

| Document | Use it for |
| --- | --- |
| [Dialect and pass reference](dialect-reference.md) | Virtual and machine operations, verifiers, allocation policy, pass order, and how to add a pass. |
| [Virtual SSA interpreter interface](virtual-ssa-interpreter-contract.md) | SSA values, block arguments, state tokens, DMA handles, MXU state, reference execution, and admitted forms. |
| [Physical program interface](functional-stream-contract.md) | Encoded words and control metadata for an instruction-level functional model. |
| [LLVM handoff examples](llvm-handoff-examples.md) | Structured LLVM markers, final word block, maps, and timing-pass handoff. |
| [Captured MLP diagnostic](captured-mlp-compiler.md) | The bounded Linalg-to-Atlas path and its explicit precision policy. |
| [Reset-entry launch limits](atlas-launch-abi-gap.md) | Capsule layout, mailbox example, and remaining host/SoC ABI work. |

The implemented physical timing and scheduling passes operate on machine IR
before LLVM conversion. Structured LLVM calls expose checked instruction
fields for analysis. After finalization, LLVM sees one inline-assembly block;
it cannot schedule individual Atlas operations. The
[pass inventory](dialect-reference.md#pass-and-tool-inventory) identifies the
implemented passes and their limits. [Adding a pass](dialect-reference.md#adding-a-pass)
shows where to wire one into the OOT build.

## Selected source and qualification

- [Mode census](selected-variant-census.md) and its
  [machine-readable inventory](selected-variant-inventory.json): required modes,
  represented modes, bounded execution tests, and remaining blockers.
- [RTL/model discrepancies](source-discrepancies.md): encoding, arithmetic,
  layout, and control differences for the pinned source revisions.
- [Column-reduction compatibility decision](selected-rtl-column-reduction-contract.md):
  the selected RTL's 64×16 physical result versus the architectural 32×32
  intent.

The observations below report bounded tests on a selected-source-linked
standalone `AtlasCore` where stated. They are not full-domain semantics,
integrated SoC tests, or a current-run certificate. The
[mode census](selected-variant-census.md) is the status ledger; an observation
does not independently admit an instruction family for software use.

## Numerical and data-movement observations

**MXU:** [MXU0 reset and continuation](mxu0-k-continuation-observation.md),
[MXU0 signed zero](mxu0-signed-zero-observation.md),
[MXU1 anchor arithmetic](mxu1-anchor-observation.md),
[MXU1 continuation](mxu1-continuation-observation.md), and
[MXU0/MXU1/model discriminator](mxu-arithmetic-discriminator-observation.md).

**VPU arithmetic:** [addition](vpu-add-observation.md),
[subtraction](vpu-sub-observation.md),
[multiplication](vpu-mul-observation.md),
[minimum](vpu-min-observation.md),
[maximum](vpu-max-observation.md),
[ReLU](vpu-relu-observation.md),
[square](vpu-square-observation.md),
[cube](vpu-cube-observation.md),
[square root](vpu-sqrt-observation.md),
[reciprocal](vpu-recip-observation.md), and
[base-two exponential](vpu-exp2-observation.md).

**VPU reductions:** [row sum](vpu-row-sum-observation.md),
[row minimum](vpu-row-min-observation.md),
[row maximum](vpu-row-max-observation.md),
[column sum](vpu-col-sum-observation.md),
[column minimum](vpu-col-min-observation.md), and
[column maximum](vpu-col-max-observation.md).

**Formats and movement:** [E8M0 pack/unpack](vpu-e8m0-pack-observation.md),
[VLI all](vli-all-observation.md),
[VLI row/column/one](vli-selective-observation.md),
[XLU transpose](xlu-transpose-observation.md), and
[VLOAD/VSTORE](vload-vstore-observation.md).

## Control, DMA, and execution observations

- [Branch delay positions and backward loop](branch-two-position-loop-observation.md),
  [direct JAL](jal-direct-target-observation.md), and
  [JALR word target](jalr-word-target-observation.md).
- [Scalar load to JALR timing](jalr-load-delay-observation.md),
  [scalar memory and halt](scalar-memory-halt-observation.md), and
  [LUI/AUIPC PC units](scalar-upper-pc-observation.md).
- [DELAY timing](delay-timing-observation.md) and
  [FENCE no-wait behavior](fence-no-wait-observation.md).
- [DMA pointer capture](dma-pointer-capture-observation.md) and
  [DMA pointer lifetime and reuse](dma-pointer-lifetime-observation.md).
- [LLVM boot entry](llvm-boot-entry-observation.md) and
  [repeated mailbox call](mailbox-call-observation.md).

Read each note's tested scope and limits before using its result. For
implementation changes, rerun the applicable portable and selected-core tests
before updating a qualification claim.
