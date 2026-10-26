# Python generator traffic-reuse prototypes

All three requested captured shapes exceed the ≥3% curve-cycle reduction bar and pass byte-exact original outputs, inputs and guards. These are opt-in Python generator recipes; G0–G7 stay shelved. Launch ABI and host protocol are unchanged. The measurements qualify one exact captured launch per shape, not all occurrences or a full model.

## Measured comparison

Every cycle below is **NPU Model curve**, current model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, RTL numerics, profile SHA256 `e5bc2419675e91138045da25f54cf3c4b071b99ab39471fe1d75219d94ab70c9`. C counts first-issued instruction through drained ECALL inclusive. Both operands and all staging offsets remain at the original 4-GiB DRAM base; no experimental relocation is used. There are no FIFO43, standalone RTL or SoC timing claims.

| Captured model / physical M/K/N | Original C | Reuse C | Fewer cycles | Original DMA.WAIT | Reuse DMA.WAIT |
|---|---:|---:|---:|---:|---:|
| Tiny 128/2048/2048, ordinal 0 | 17,847,881 | 7,191,210 | 59.708% | 13,258,268 | 3,734,108 |
| Smol 128/3072/768, ordinal 48 | 17,007,240 | 4,162,791 | 75.523% | 14,420,668 | 2,218,708 |
| Smol 128/768/768, ordinal 0 | 2,657,162 | 1,153,869 | 56.575% | 1,995,868 | 656,308 |

| Physical M/K/N | Original DMA loads | Reuse DMA loads | Stores, both | Original total | Reuse total = compulsory A+B+C | Reuse words |
|---|---:|---:|---:|---:|---:|---:|
| 128/2048/2048 | 17,039,360 | 4,456,448 | 524,288 | 17,563,648 | 4,980,736 | 2,023 |
| 128/3072/768 | 18,874,368 | 2,752,512 | 196,608 | 19,070,976 | 2,949,120 | 3,346 |
| 128/768/768 | 2,457,600 | 688,128 | 196,608 | 2,654,208 | 884,736 | 458 |

DMA byte counts are accepted native beats multiplied by the pinned 32-byte beat size and independently agree with the emitted requests. The compulsory footprint is physical FP8 A + FP8 B + BF16 C, including physical padding. Every original input tile is loaded from DRAM once; every output tile is stored once. The recipes also share weight pushes across two rows and transpose each cached B tile once. All delay spacing comes from the established generator, with no fitted constants or scheduler search.

Tiny uses a fresh original captured launch on the same current model/profile, rather than its historical 17,847,882 reset-to-halt receipt. Its current first-issue baseline is 17,847,881. Smol uses the previously retained current-profile original-generator receipts for ordinals 0 and 48. Exact source/binary/input/expected/profile identities were checked before reuse and independently reviewed.

## VMEM and numerical recipe

Only the two physical MXU0 accumulator slots are used. For each B column panel, process two row pairs; one weight push serves both rows in a pair, and the cached B panel serves both pairs. Every output preserves increasing K order, first overwrite then ordered accumulate, one BF16 pop, and the original BF16 power-of-two rescale. There is no partial-accumulator pop/restore, FP32 replacement, or host splitting.

All following ranges are bytes, half-open, within banks 0 and 1, each 262,144 bytes:

| Shape | A residency | Cached transposed B column | Output scratch |
|---|---|---|---|
| Tiny 128/2048/2048 | Bank 1 [262144,524288) | Bank 0 [0,65536) | Bank 0 [65536,67584) |
| Smol 128/3072/768 | Bank 1 [262144,524288), bank 0 [0,131072) | Bank 0 [131072,229376) | Bank 0 [229376,231424) |
| Smol 128/768/768 | Bank 1 [262144,360448) | Bank 1 [360448,385024) | Bank 0 [10240,12288) |

Each 1,024-byte physical tile stays within one bank. The larger Smol A mapping crosses banks between packed tile 255 and 256 (row tile 2, K tiles 63/64), without changing the reduction chain. The shared B cache is transposed in place in VMEM; original DRAM operands remain untouched. Scalar addresses use the existing four-byte VMEM word units and DRAM byte units.

Column-first traversal scatters completed BF16 output tiles to the original row-major packed output offsets. Every host-visible output address, 256-byte guard and final ECALL protocol is preserved. Each program has exactly one static `vli.all m4` rescale patch site with encoding `((exponent+127)<<23)|0x25f`; changing only the exponent changes exactly that word. No host patcher or mailbox code changed. Program sizes obey the existing generator's 8,192-word software admission; this work performs no RTL/SoC IMEM or runtime qualification.

## Opt-in generator API

The existing `tools/atlas_program.py` dispatches only the three qualified physical shapes when `traffic_reuse=True`. The default generator and default source-set names remain unchanged; six captured defaults and the older frozen-assembly gates confirm byte identity. Legacy residency/pretranspose flags cannot be combined with this opt-in. The guarded original output offset and a normal finite integer BF16 rescale are required.

```python
from atlas_program import assembly, code_sources, dma_load_bytes

m, k, n = 128, 3072, 768
before = code_sources(traffic_reuse=True)
source = assembly(m, k, n, m*k + k*n + 256, -15, traffic_reuse=True)
loads = dma_load_bytes(m, k, n, traffic_reuse=True)
assert code_sources(traffic_reuse=True) == before
```

The opt-in source bookend includes all three new helper modules. No CLI or full-policy export enables the new recipes by default. Existing packing, arithmetic, host protocol and compiler pipeline are unchanged.

## Gates and scope

Root integrated unit/semantic gates: 57 run, 56 passed, one optional selected-RTL codec/full-capture fixture test skipped because its explicit fixtures were unset. Tests include complete operand origin/permutation, bank limits, ordered K chains, two slots, output halves/addresses, DMA drainage, rescale patch identity, negative controls, strict admission and default/source-closure compatibility. All three full captured native runs pass the original expected DRAM image; Tiny also has a fresh exact original baseline. Root independently replayed the full smaller Smol case through the integrated `traffic_reuse=True` API, reproducing cycles, bytes, stalls, profile and final image exactly. The replay uses the reviewed harness, not a separate numerical implementation.

The independent nonauthor GPT-6 Astra/high reviewer reproduced integrated source and binary identities, original baseline binaries, all source/facts/profile joins, full captured DRAM reconstructions and process caps for every case. No blocking finding remains. Root read all diffs and results, ran the integrated gates, and verified source bookends after execution.

Each simulation had an owned-process-group wall cap at most 3,490 seconds plus at most five seconds kill grace; short probes/root replay used 300 seconds. The longest new full run was the Tiny baseline, about 21.7 minutes. All processes exited normally, and no simulations remain. CPU affinity stayed within 0–15 with one numerical thread per process. No full Tiny live, full-policy Smol live, 3,555-kernel batch, VCS, Chipyard build or push occurred.

Only the measured kernel cycle gains are claimed. Full-policy timing, other captured operands/rescale variants and SoC speedups remain unqualified; host output-copy costs are unchanged. Stop for owner and separate-reviewer review. No default promotion or further implementation is authorized by this report.

## Evidence and delegation

The normalized table and all input/program/receipt/profile/source hashes are under `/tmp/reednicolas-atlas/sota-fast/generator-reuse/results.json`. Per-case reproducible commands, original assets, full DRAM results and cap receipts are in `tiny/`, `smol-stream/` and `smol-resident/` under that same output root. Root gate evidence is `root-unit-final.log` and `root-check/{fresh/0/receipt.json,replay-verification.json}`. The independent review is `review/REVIEW.txt`; its audit JSONs cover both Smol cases, Tiny paired results and assembler/ABI identity. Original model assets remain in the oct22 capture directories recorded by the receipts.

Delegation: reuse_tiny (GPT-6 Astra/high, CPU0–3) authored Tiny; reuse_smol_stream (GPT-6 Astra/high, CPU4–7) authored the split-bank recipe; reused payoff_bounds (GPT-6 Astra/high, CPU8–11) authored resident reuse and opt-in glue. Each used an isolated worktree/output directory. reuse_review (GPT-6 Astra/high, CPU4–7 after stream finished) independently reviewed all authors. reuse_receipt_audit (GPT-6 Luna/low, CPU8–11 after resident finished) collected links, logs and counts without numerical-oracle claims. Root used CPU12–15 for integration and gates. No speed mode was enabled.
