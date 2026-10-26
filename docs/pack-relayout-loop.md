# A faster PACK relayout loop

Branch: `perf/attention-mlp-bottlenecks`, after
[explicit DMA in flight across PACK](pack-dma-overlap.md). Status:
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

## Original serial loop

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

## Pipelined copy and two-row iterations

The original improvement software-pipelined one row by one copy:

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
overwrites the temporary that `sw(i - 1)` has just read. Lowering also places
the loop's ALU work in the load path's free cycles: the row counter right
after the first load, each source pointer once its half's last word is loaded,
and the destination pointer after the last store. The stream scheduler keeps
scalar loads and stores in program order (it cannot prove the scratch halves
disjoint), so both timing passes keep this order. The loop's registers, trip
count, branch and instruction count were unchanged in that first improvement.

That one-row loop took 29 cycles under `--insert-atlas-delays`, and 28 under
`--schedule-atlas-stream`, which moves the destination step into the
back-edge's delay slot, against about 43. (An earlier experiment that kept
loads ahead of stores with eight temporaries reached about 36, because its
stores still issued after all eight loads.)

The virtual scheduler's PACK cost (`costOf` in `AtlasVirtualScheduling.cpp`)
modelled the old serial copy, each word's load latency plus its store latency.
It models an iteration as one fewer load intervals than copied words, then
the last word's load and store latencies and two control instructions (the
back-edge and its delay slot holding the destination step).
`kPackRelayoutRowWords` (`AtlasVirtualAllocation.h`) is the row size both use.

Two rows now share an iteration (`kPackRelayoutUnrollFactor`): 16 copies, one
back-edge and one load/store tail per pair, 16 iterations. The virtual cost
uses the same factor. Copy `i` loads from row `i / 8`, word `(i % 8) / 2`,
half `i % 2`; source halves advance 32 bytes and the destination advances 64
bytes per iteration. The mandatory native stream has 51 instruction words
from the first load to the branch, so its B immediate is -102; the binding
checks that stream and its encoded branch using the same rows-per-iteration
value. These counts include the load-path delays.

## What does not change

- The buffer contract runs the loop from its entry registers and checks every
  relayout word against the row interleave, so it proves the reordered copy
  without change.
- The PACK loop recognizer (`AtlasPackLoop.h`) accepts any order of the PACK's
  scalar loads, stores and ALU work, so DMA still overlaps the loop.
- The register-allocation checker's PACK effects name the same registers.

## Tests

- `test_buffer_contract_verification.py::test_relayout_stores_follow_the_next_copy_load`:
  the lowered copy order is `lw, (lw, sw) x 15, sw`, and after either timing
  pass an iteration has at most one store waiting on its load (the last) and
  takes at most 53 cycles.
- The buffer-contract mutations (30 or 33 rows, a shifted source word, an
  interior back-edge) keep their diagnostics. The reordering case now moves
  every store back up to its own load, the old serial order, and still proves
  the interleave.

## Earlier one-row results

Full ctest passes (104 skips, as on main). Each program was lowered through
`tools/lower_atlas_program.py` (every verifier, `--timing
schedule-atlas-stream`) with main (`aea28c5`), with this branch before the
change, and after it, then simulated on the pinned npu_model (`0c4a1f9`) and
npu_model main (`9e6313b`). Output bytes are identical across all three builds
for every program on both models.

A PACK drops from about 1,430 to 1,550 cycles to about 980 to 1,110.

Combined attention + MLP, npu_model main:

| Program | Main | Before | After | vs before | vs main | DMA busy after |
|---|---|---|---|---|---|---|
| 1 tile | 10,136 | 8,610 | 7,671 | 1.12x | 1.32x | 68.1% |
| 2 independent tiles | 19,772 | 15,119 | 12,925 | 1.17x | 1.53x | 80.4% |
| 4 independent tiles | 39,405 | 27,182 | 21,577 | 1.26x | 1.83x | 95.8% |
| 2 tiles, shared weights | 16,964 | 13,745 | 11,489 | 1.20x | 1.48x | 64.5% |
| 4 tiles, shared weights | 30,654 | 23,690 | 19,176 | 1.24x | 1.60x | 61.3% |

On the pinned npu_model the same programs gain 1.22x to 1.34x over the
previous branch, 1.36x to 1.87x over main. Four independent tiles are now
DMA-bound: PACK runs almost entirely while transfers are in flight.

Elsewhere: `virtual_attention_tile` 1.09x, `virtual_fp8_two_layer_mlp` 1.09x,
the generated MLP tiles 1.03x to 1.12x where a PACK is exposed; programs
without a PACK are unchanged. One exception: `mlp_independent_4`, already 99%
DMA busy, is 35 cycles (0.2%) slower on both models. Its virtual schedule
differs only in the order of the last MXU readouts and stores, a tie the
cheaper PACK cost tips the other way.

## Next

- Shared-weight programs are still PACK-bound (61% DMA busy at 4 tiles), and
  the current two-row loop amortizes its tail and control across 16 copies.
  Going further needs fewer, wider moves, or no relayout at all.

## Not in this change

- Fewer loop registers (one source pointer with offsets, the destination as
  the counter). It saves ALU work that already hides in the load path's free
  cycles.
- Removing the relayout altogether, which needs a PACK or XLU layout the MXU
  accepts: a hardware question.

## Guarded MXU conversion

`atlas-opt --atlas-compile=guarded-pack=true` or
`tools/lower_atlas_program.py --guarded-pack` opts in to MXU conversion of
proven-finite BF16 sources, bit-exact with PACK for every finite encoding and
scale. The registered pipeline runs the conversion before boundary conversion,
scheduling and allocation, and always recovers register pressure through spilling.
For pass-level use, run `--lower-atlas-guarded-pack` before these stages.

Finiteness propagates from finite splats and raw FP8 reset contractions through
mov/ReLU, min/max of finite operands and BF16 accumulator copies. Unknown
sources, overflow-capable arithmetic and continued accumulations retain PACK.
It uses MXU0 (`unit=1` selects MXU1) in single-block functions with a free
accumulator slot. The ordinary spill stage handles register pressure introduced
by conversion. Native fallback still admits only scales 122..128. Exactness
does not guarantee a speedup under MXU contention or register pressure.
