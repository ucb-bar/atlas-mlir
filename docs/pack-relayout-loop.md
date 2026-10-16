# Plan: a faster PACK relayout loop

Branch: `perf/attention-mlp-bottlenecks`, after
[explicit DMA in flight across PACK](pack-dma-overlap.md). Status: plan, not
implemented.

## Goal

A PACK converts a BF16 tile to FP8 with `vpu_pack` (about 65 cycles), then
repairs the row layout through VMEM: a VSTORE of the raw result, a 32-row
scalar copy loop, and a VLOAD of the relayout. The loop is nearly all of a
PACK's 1,490 cycles. With DMA now overlapping PACK, PACK running with the DMA
engine idle is the largest remaining share of the combined attention + MLP
tile (npu_model main):

| Program | Cycles | DMA busy | PACK with DMA idle |
|---|---|---|---|
| 4 independent tiles | 27,182 | 76.5% | 21% of cycles |
| 4 tiles, shared weights | 23,690 | 49.8% | 45% of cycles |

## Why the loop is slow

Each row copies eight words, two halves of four (`lowerPack`):

```
lw x16, 0(x10)   sw x16, 0(x12)    lw x17, 0(x11)   sw x17, 16(x12)
lw x16, 4(x10)   sw x16, 4(x12)    ...              sw x17, 28(x12)
addi x10, 16     addi x11, 16      addi x12, 32     addi x13, 1
blt x13, x14     nop
```

In the timing model (`AtlasTiming.cpp`, the npu_model rtl-match port) a
scalar load holds the load path for three cycles, and its result can be stored
four cycles after it issues. Every store sits right behind its own load, so
each word waits for its load: eight RAW delays per row, about 43 cycles a row.

The floor for a word-at-a-time copy is the load path: 256 loads at three
cycles each, 768 cycles per PACK.

## Change

Software-pipeline the body by one copy, inside the iteration:

```
lw(0)
lw(1)  sw(0)
lw(2)  sw(1)
...
lw(7)  sw(6)
sw(7)
```

Each store now issues right after the next load, in a cycle the load path
leaves free, four cycles after its own load. Two temporaries still suffice:
copy `i` uses `x16` for even `i` and `x17` for odd `i`, and `lw(i + 1)`
overwrites the temporary that `sw(i - 1)` has just read. The loop's registers,
trip count, pointers, branch and instruction count are unchanged; nothing is
unrolled or duplicated. The stream scheduler keeps scalar loads and stores in
program order (it cannot prove the scratch halves disjoint) and fills the free
third cycle of each load with the loop's ALU work.

Expected: about 26 to 28 cycles a row, under 1,000 cycles per PACK, against
1,490 now. (An earlier experiment that kept loads ahead of stores with eight
temporaries reached 1,202, because its eight stores still issued after all
eight loads.)

The virtual scheduler's PACK cost (`costOf` in `AtlasVirtualScheduling.cpp`)
models the old serial copy: each word's load latency plus its store latency.
It becomes one load interval per word, plus the last word's load and store
latencies.

## What does not change

- The buffer contract runs the loop from its entry registers and checks every
  relayout word against the row interleave, so it proves the reordered copy
  without change.
- The PACK loop recognizer (`AtlasPackLoop.h`) accepts any order of the PACK's
  scalar loads, stores and ALU work, so DMA still overlaps the loop.
- The register-allocation checker's PACK effects name the same registers.

## Tests

- A lowering test: in the timed relayout loop, no store waits on its load (no
  "RAW on x16/x17 after atlas.scalar_load" delay), each store follows the next
  copy's load, and the timed loop body is at most 28 cycles a row.
- Existing buffer-contract mutations (31 or 33 rows, a shifted source word, an
  interior back-edge, reordered copy pairs) keep their diagnostics.

## Validation

1. Full ctest, 104 skips as on main.
2. npu_model, pinned and main: every example and the combined attention + MLP
   and MLP programs, against this branch before the change. Output bytes must
   match, and no program may get slower.
3. Report per-PACK cycles and the combined programs' cycles and DMA busy.

## Commits

1. This plan.
2. The pipelined loop body, the virtual scheduler's PACK cost and the tests.
3. Results in this file.

## Not in this change

- Fewer loop registers (one source pointer with offsets, the destination as
  the counter). It saves ALU work that already hides in the load path's free
  cycles.
- Removing the relayout altogether, which needs a PACK or XLU layout the MXU
  accepts: a hardware question.
