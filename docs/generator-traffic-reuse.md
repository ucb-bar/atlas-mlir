# Python generator traffic-reuse prototype

This page records the accepted three-shape prototype at `7493e39`. The current API admits 57 exact shapes through five helpers; see [current coverage, target limits and qualification](generator-reuse-breadth.md). The tables below are historical measurements, not the current allowlist.

## Initial captured pairs

All cycles and DMA.WAIT counts are **NPU Model curve**, revision `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, RTL numerics, profile `e5bc2419675e91138045da25f54cf3c4b071b99ab39471fe1d75219d94ab70c9`. Cycles run from first issue through drained ECALL, inclusive. Inputs and guarded output addresses remain at the original 4-GiB DRAM base. These tables contain no standalone RTL or SoC timing.

| Captured shape / ordinal | Original cycles | Reuse cycles | Fewer cycles | DMA.WAIT before → after |
|---|---:|---:|---:|---:|
| Tiny 128/2048/2048 / 0 | 17,847,881 | 7,191,210 | 59.708% | 13,258,268 → 3,734,108 |
| Smol 128/3072/768 / 48 | 17,007,240 | 4,162,791 | 75.523% | 14,420,668 → 2,218,708 |
| Smol 128/768/768 / 0 | 2,657,162 | 1,153,869 | 56.575% | 1,995,868 → 656,308 |

| Physical M/K/N | DMA loads before → after | Stores, both | Raw words |
|---|---:|---:|---:|
| 128/2048/2048 | 17,039,360 → 4,456,448 | 524,288 | 2,023 |
| 128/3072/768 | 18,874,368 → 2,752,512 | 196,608 | 3,346 |
| 128/768/768 | 2,457,600 → 688,128 | 196,608 | 458 |

Accepted native DMA beats multiplied by the pinned 32-byte beat size agree with emitted requests. Reuse reaches compulsory physical A+B+C traffic for these three shapes. Recipes retain operands, transpose cached B once and share weight pushes across accumulator rows. Spacing comes from the established generator; no timing constants were fitted or schedule search introduced.

## Numerical and memory contract

Each pair matches the original complete expected DRAM image, including operands and guards. Reduction order remains increasing K, with first overwrite, ordered accumulation, one BF16 pop and the original power-of-two rescale. There is no partial accumulator spill or reassociation. Exactly one static `vli.all m4` is patchable; changing the exponent changes that word only.

Only VMEM banks 0-1 are used. Each tile stays within one bank; VMEM addresses count four-byte words and DRAM addresses bytes. Output scatters to the original packed layout and guards. Safety requires exclusive raw ownership and completed DMA. Compiled mailbox/tensor allocations can overlap these banks, so recipes do not establish mixed-client safety.

## API and retained evidence

Use `atlas_program.assembly(..., traffic_reuse=True)`, `dma_load_bytes(..., traffic_reuse=True)` and `code_sources(traffic_reuse=True)`. Admission checks exact shape, original output offset and finite integer exponent. The option defaults off; the default generator/source set remains unchanged. Legacy residency/pretranspose flags cannot be combined with reuse. Current shared tests cover all admitted shapes; program limits are in the coverage page.

Each original pair records its source, program, input, expected-output and profile hashes, per-case commands and cap receipts. Supplementary captures, standalone RTL, later coverage, host-check improvements and SoC results are on the current coverage page. These three representatives alone do not qualify every numeric variant or a full-model execution.
