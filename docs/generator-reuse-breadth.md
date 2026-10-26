# Generator reuse: breadth, target safety and standalone RTL

All five extensions clear the ≥3% NPU Model curve bar and match their original captured full DRAM images byte for byte. Six additional real captures qualify different operands and rescale exponents for the three accepted recipes. The five new shapes also pass standalone AtlasCore RTL with no assertion failures. All recipes remain opt-in; there is no PR, push, default promotion, compiler, ABI, host or RTL source change. G0–G7 and Parts 3–4 remain shelved. Stop for owner and separate-reviewer review.

## New measured pairs

All cycle and DMA.WAIT counts below are **NPU Model curve**, current model 47b8599d8ffbe5ddd96059ffe2b5be2427a410f5, RTL numerics, pinned profile e5bc2419675e91138045da25f54cf3c4b071b99ab39471fe1d75219d94ab70c9. Cycle boundary: first issued instruction through drained ECALL inclusive. Every new original baseline is freshly measured on this same profile. Original 4-GiB DRAM base, packed inputs, output offsets and 256-byte guards are unchanged. No timing constants were fitted.

| Model · physical M/K/N | Capture ordinal | Original cycles | Reuse cycles | Fewer cycles |
|---|---|---|---|---|
| Tiny 128/5632/1024 | 10 | 41,395,016 | 9,404,320 | 77.282% |
| Tiny 32/2048/2048 | 280 | 4,462,007 | 3,335,959 | 25.236% |
| ResNet 224/2304/256 | 25 | 7,464,106 | 2,268,476 | 69.608% |
| ResNet 64/4608/512 | 44 | 8,476,924 | 3,203,152 | 62.213% |
| ResNet 800/1152/128 | 12 | 3,839,056 | 1,759,374 | 54.172% |

| Physical M/K/N | DMA loads before → after | Stores, both | DMA.WAIT before → after | Raw words |
|---|---|---|---|---|
| 128/5632/1024 | 46,137,344 → 12,255,232 | 262,144 | 35,099,548 → 4,623,116 | 141 |
| 32/2048/2048 | 4,259,840 → 4,259,840 | 131,072 | 3,314,588 → 2,147,436 | 107 |
| 224/2304/256 | 8,257,536 → 1,695,744 | 114,688 | 6,328,868 → 1,362,468 | 386 |
| 64/4608/512 | 9,437,184 → 2,654,208 | 65,536 | 7,187,708 → 2,054,028 | 160 |
| 800/1152/128 | 3,833,856 → 1,216,512 | 204,800 | 3,043,128 → 1,061,768 | 1044 |

DMA bytes are accepted native beats ×32 and match exact emitted requests. Tiny32 already had compulsory input traffic; its gain comes from four-tile B prefetch overlapping compute. Tiny128 keeps an A row pair resident and loads B twice across two row pairs. ResNet caches A blocks across banks 0/1, with one transposed B column per block and weight sharing between two accumulator rows; 224 and 800 use two A blocks and therefore repeat B once, while 64 reaches compulsory traffic. All preserve increasing K, first overwrite then ordered accumulate, one final BF16 pop/rescale, exactly one patchable static VLI.all m4, and original output scatter addresses. New recipes use compact scalar loops; no partial accumulator spill or reduction reassociation.

## Captured numeric breadth

| Accepted physical shape | Captured ordinals (rescale exponent) | Reuse cycles, each | Full image exact |
|---|---|---|---|
| Tiny 128/2048/2048 | 0 (-14), 3 (-21), 267 (-12) | 7,191,210 | PASS |
| Smol 128/3072/768 | 48 (-15), 144 (-17), 576 (-11) | 4,162,791 | PASS |
| Smol 128/768/768 | 0 (-19), 8 (-15), 56 (-14) | 1,153,869 | PASS |

Each supplementary capture has a different original initial-image hash and exponent from its representative. Expected outputs come from the original captured assets, not from the new recipe. All original operands and guards are checked along with C. Cycles, accepted DMA bytes and DMA.WAIT stalls reproduce the representative recipe exactly. This qualifies three captures per accepted shape, not every numeric variant. Each extension has one paired captured launch. The complete receipt/program/input/expected/source joins are in results.json under the output root.

## Launch-weighted whole-model estimate

The estimated saving in kernel cycles is Δ=Σ covered_shapes N_s(C_original,s−C_reuse,s), applying the measured delta to each exact matching full-model launch and leaving all other kernel and host costs unchanged. All 3609 full-launch records are joined to checked source/binary identities; the baseline instruction skeleton matches within each counted shape after masking only the verified output-rescale immediate. No first-block/selected-layer overlap is counted twice.

| Full model | Covered launches / total | Covered shapes / total | Estimated saved kernel cycles | Covered subtotal reduction |
|---|---|---|---|---|
| Tiny | 485 / 840 | 3 / 10 | 3,305,179,373 | 57.413% |
| Smol | 976 / 2,715 | 2 / 24 | 3,644,715,920 | 68.143% |
| ResNet | 13 / 54 | 3 / 21 | 55,313,824 | 64.640% |

These are absolute full-launch-ledger saving estimates under representative-cycle transfer. The percentage is for the covered subtotal. A complete current-profile baseline for the remaining launches is unavailable, so a whole-model execution-time percentage is not measured or asserted. Host copies, noncontraction operators and other launch costs are unchanged. The old9e ResNet full total is not mixed into a current point estimate.

For an explicit bound, let R be the unchanged unmeasured contraction cycles. Then C_before=covered_original+R, C_after=covered_reuse+R. The current-profile B1 resource lower bound supplies R_floor≤R. Therefore Δ/(covered_original+R_floor) is only an optimistic ceiling on the full contraction-cycle fraction, conditional on the same representative transfer assumption:

| Full model | Unmeasured launches | Weighted remainder resource floor | Optimistic full contraction ceiling |
|---|---|---|---|
| Tiny | 355 | 1,069,401,402 | 48.419% |
| Smol | 1,739 | 3,822,127,017 | 39.743% |
| ResNet | 41 | 79,036,453 | 33.603% |

resource-bounds.json records the named **NPU Model curve static resource lower-bound model**, its 59 original source programs, current 47b facts/profile and imported source hashes. resource_bounds.py reuses the B1 dynamic scalar-flow/roofline accounting with current hardware, converter and backend; it never invokes the historical setup or a cycle simulation. Floors describe ideal resource occupancy, not predicted or achieved runtime. No unmeasured remainder was filled with a fitted constant.

## EE290 target memory safety and IMEM

The actual default EE290 Atlas configuration has 1.5 MiB VMEM: six contiguous 256 KiB banks. DMA/VLOAD operands count four-byte local VMEM words. Banks0/1 local byte range [0,0x80000) maps to CPU physical [0x20000000,0x20080000). CPU IMEM [0x20000,0x40000), CSR page[0x40000,0x41000) and raw DRAM staging at 0x100000000 are disjoint. Local VMEM byte 0x40000 is CPU physical 0x20040000; it is not the CSR page at CPU physical 0x40000.

The selected raw host path has no VMEM mailbox: atlas_host_load_raw appends its existing DBG1→DBG0 CSR completion epilogue, net+5 instruction words. The host stages DRAM before start, polls CSRs while active, and reads output after successful wait. Raw kernels require known quiescence and exclusive ownership of their VMEM ranges.

The separate safety/full-host-audit.json and .txt bind the actual retained full-Tiny target/build/map: tiny_full_target.c calls raw load/start/wait, stages at the 4-GiB DRAM base, and places its 128-byte host mailbox at CPU 0xd237ef00, workspace at 0xcfd3ec80 and saved-input buffer at 0xd1d0ec80, all outside VMEM. The selected Smol-layer adapter also uses raw load/start/wait and 4-GiB staging; its host mailbox at CPU 0x80c87380 is outside VMEM. The full-Smol corpus is explicitly build-only (full_host_execution=false); its representative per-kernel ELF uses block_kernel_launch.c and the same raw runtime. There is no qualified full-policy host execution or full-policy ownership guarantee. These are static host-source/build/map bindings, not new host execution qualification. The current Merlin smol_host_ops.c differs from its retained selected-layer version; no current-Merlin-to-retained-ELF numerical equivalence is asserted.

**A blanket guarantee for mixed clients is false.** The compiler virtual ABI reserves local VMEM [0,1024) for a control mailbox plus input/output/PACK allocations in banks 0/1; its DMA staging begins in bank2. Several raw recipes intentionally occupy those mailbox/boundary locations. Concurrent host writes and live compiled allocations are outside this raw contract; arbitration is not ownership protection. Thus safety is confirmed for the audited exclusive raw per-kernel paths and full-Tiny/selected-layer allocation maps, with no universal mailbox-preservation claim and no recipe-only fix needed within that contract.

Source anchors: [AtlasMemMap](/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/src/main/scala/atlas/scalar/ScalarCore.scala:39), EE290Configs.scala:42, AtlasParams/VmemParams, AtlasTile/VMEM translation, [raw host loader](/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/compiler/sessions/oct25/sota-fast/generator-breadth/atlas-mlir/runtime/atlas_host.c:63), [raw ownership contract](/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/compiler/sessions/oct25/sota-fast/generator-breadth/atlas-mlir/runtime/atlas_host.h:75), kernel_launch.c:31–40 and AtlasVirtualAllocation.cpp:56–69,766. safety/safety.json retains exact source paths, line excerpts, SHA256s and 122 hash-verified retained RTL sources.

Actual EE290 and retained hardware IMEM are 128 KiB /32,768 words. The real raw loader requires raw_words+5≤32,768 (raw limit32,763). The prebuilt standalone driver imposes its own 8,192-word loaded-program cap, including its test wrapper; no 2,048-word limit applies.

| Physical M/K/N | Raw words | Real raw-host loaded words | Fits 32768 |
|---|---|---|---|
| 128/2048/2048 | 2023 | 2028 | PASS |
| 128/3072/768 | 3346 | 3351 | PASS |
| 128/768/768 | 458 | 463 | PASS |
| 224/2304/256 | 386 | 391 | PASS |
| 64/4608/512 | 160 | 165 | PASS |
| 800/1152/128 | 1044 | 1049 | PASS |
| 128/5632/1024 | 141 | 146 | PASS |
| 32/2048/2048 | 107 | 112 | PASS |

Every vector/DMA tile is aligned and lies within one 256 KiB bank. Native symbolic address/order proofs and standalone unallocated-VMEM poison checks support the ranges. The interactive review shows one maximum live block per shape; exact per-block maps are in each numerical receipt/allocation artifact and safety/target-layouts.json.

## Standalone RTL qualification

Pinned prebuilt AtlasCore binary d594f72d67c0f8076faab1feae0068c96fcd93b7c31323b5b005a0bb3702119c, built with assertions, uses a synthetic FIFO43 response memory. No RTL rebuild or VCS. This is **standalone RTL**, not NPU Model curve, NPU Model FIFO43 or a SoC timing result. No first-issue kernel-cycle comparison is claimed: the retained driver observer counts include IMEM load, VMEM setup, CSR polling and guard checks.

| Physical M/K/N | Captured kernel launches | Full DRAM bytes checked, each | Unused VMEM bytes checked | Process wall seconds | Result |
|---|---|---|---|---|---|
| 128/5632/1024 | 1 | 6,750,720 | 1,202,176 | 1381.662 | Exact; no assertion failures |
| 32/2048/2048 | 2 | 4,391,424 | 1,497,088 | 749.640 | Exact; no assertion failures |
| 224/2304/256 | 2 | 1,221,120 | 1,054,720 | 673.225 | Exact; no assertion failures |
| 64/4608/512 | 2 | 2,720,256 | 1,128,448 | 851.679 | Exact; no assertion failures |
| 800/1152/128 | 2 | 1,274,368 | 1,054,720 | 555.357 | Exact; no assertion failures |

The four smaller recipes execute twice in one retained-core session, restaging captured DRAM but not VMEM. Both calls check all outputs, inputs, guards, accepted DMA beats, pending-response drainage, status 5 / marker 1 / illegal 0, and every byte outside the allocation union. The larger Tiny recipe executes one complete captured kernel; its second stage contains no DMA or kernel body and only terminates the driver session. That stage is not a second kernel or a reentrancy claim.

The prebuilt driver requires two session images and does not initialize DBG1. Test-only prefix instructions initialize it, followed by the byte-identical shifted raw body and the exact existing raw-host suffix. The largest-case conditional prefix skips the body on stage2; its branch targets and unchanged relative control flow were independently decoded. This wrapper qualifies core outputs/drain, not actual host invocation-sequence lifecycle. The real host code and protocol remain untouched. The initial 100k-stage-cycle probe is deliberately censored, excluded from completed-launch counts and performance gains.

Every simulation owns a process group capped at 3490 seconds plus ≤5 seconds termination grace; probes use 300 seconds. All completed jobs exited normally before their caps. The wall admission margin in the RTL driver is only a timeout margin, not a timing-model constant. No full Tiny live, full Smol policy live or 3555-kernel batch occurred.

## Integration, review and reproducibility

Public API: atlas_program.assembly(...,traffic_reuse=True) and dma_load_bytes(...,traffic_reuse=True), now eight exact shapes. The default source-file set and emitted assembly/program bytes are unchanged; legacy residency flags cannot be combined with the new opt-in. code_sources(traffic_reuse=True) adds all five helper modules with duplicate module names deduplicated. No CLI/full-policy export opts in automatically.

Root integrated 66 tests:65 PASS, one optional selected-codec/full-capture fixture SKIP. Complete semantic tests cover operand provenance, transpose, bank limits, increasing K, two accumulator slots, pending DMA ownership, exact output halves/scatter, one rescale and negative controls. Root repeated full ResNet25 and Tiny280 through the integrated public API, reproducing programs, cycles, DMA, stalls, profile and entire expected image. Independent nonauthor review reassembled all eight recipes at extreme scales, verified the sole patch word, default compatibility, full original captured expectations, memory sources and wrapper CFG. No blockers remain; final hashes bind the review checkpoint.

Delegation: breadth_tiny and breadth_resnet authored isolated helpers/tests at GPT-6 Astra/high on CPUs 0–3 and 4–7. Reused reuse_smol_stream (GPT-6 Astra/high, CPUs 8–11) authored the separate read-only safety audit and output-only RTL harness. breadth_review (GPT-6 Astra/high, CPUs 4–7 after ResNet completed) independently reviewed all source/numerics/safety/RTL evidence and authored no implementation. breadth_receipts (GPT-6 Luna/low, CPUs 4–7 during idle review windows) audited small receipt links, caps, logs and line counts without numerical-oracle claims. Root CPUs 12–15 integrated source, read every diff/result, ran gates and two full API replays, reconstructed receipts, computed launch-weighted estimates and checked report/UI. Numerical threads 1; no Fast/Ultrafast mode.

Reproduce/inspect under /tmp/reednicolas-atlas/sota-fast/generator-breadth/: collect.py/results.json; resource_bounds.py/resource-bounds.json; tiny/ and resnet/ captured commands/cap receipts; breadth/ six variants; root-unit-final.log and root-check/; safety/; rtl/; review/; receipt-audit/. The final BREADTH-RECEIPT.json binds source, report, progress, tests and receipts. The standalone review.html follows the prior session review styling, using the DaisyUI/Tailwind fallback because the compiler project has no product UI design system. Light/dark/narrow browser gates and root visual inspection are recorded. Lavish Editor returned HTTP500 when opening this final artifact; the verified standalone HTML remains available. No publication.

Remaining qualification: other captures/rescales, untouched physical shapes, and real SoC behavior remain unmeasured. The owner will perform SoC spot checks. Host output-copy cost is unchanged. Stop for owner and separate-reviewer review; no PR or default promotion.
