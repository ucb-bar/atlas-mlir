# PACK layout repair

VPU PACK converts all 32 physical BF16 rows of the left-half register, then all 32 rows of the right-half register. Its FP8 result contains two 512-byte halves. An MXU operand instead needs each row's left and right 16-byte halves together. The compiler repairs this layout in registers when a free accumulator is available, retaining the scalar VMEM interleave as a fallback.

## Register-only repair

The scheduled compile pipeline runs `--lower-atlas-pack-relayout` in the spill stage, after virtual scheduling and the first pressure recovery; the unscheduled pipeline runs the same spill stage in source order. Both rerun spilling afterward to allocate the new BF16 value. The pass preserves the existing order, finds the current state token and a free accumulator on either MXU, and replaces an admitted `virtual_pack_fp8` with:

1. `virtual_pack_roundtrip_bf16`, which applies the original VPU PACK quantization, then unit-scale UNPACK into a BF16 tile.
2. A BF16 accumulator push followed by a unit-scale FP8 readout, which produces the canonical FP8 layout.

The roundtrip operation lowers to PACK using reserved scratch register m62, then UNPACK using an allocated BF16 destination pair. The source and destination pairs exclude m62. UNPACK reverses PACK's physical-row arrangement; the accumulator push reads the restored logical BF16 tile, and the FP8 pop writes ordinary row-major bytes. The conversion needs no VMEM scratch or DRAM transfer. Scale registers are rematerialized for each conversion, and the CFG contract checks the source, temporary, result pair and both scales.

The original PACK handles every exceptional BF16 input before UNPACK executes. Its output has exactly 239 possible codes: `0x00`, `0x08..0x7e`, and `0x88..0xfe`. It emits no negative zero, subnormal or `0x7f`/`0xff` code. Unit-scale UNPACK maps these codes to exact finite BF16 values; the MXU unit-scale converter maps them back byte-identically. In particular, the largest magnitude is 448, so the MXU converter's different treatment of rounded 480 cannot occur. Unlike direct guarded MXU conversion, this route needs no source-finiteness proof or clamping.

Read-only analysis exhaustively checked all 65,536 BF16 encodings at each admitted source scale `122..128`, with zero converter mismatches. Direct functional ISA checks also covered both MXUs and all these encodings in asymmetric tiles: 896 chains and 917,504 compared output bytes, all identical. These checks establish the selected model's conversion and layout behavior; they do not exercise encoded decode or timed execution.

If both MXUs' accumulator slots are occupied, the current block has no state token, or the scale is outside `122..128`, the pass leaves PACK unchanged. Direct virtual-to-machine lowering also retains the scalar fallback. The late placement avoids changing virtual scheduling priorities for tiled programs; the existing spilling and allocation machinery owns the new BF16 value and accumulator lifetime. The second virtual scheduling pass after the repair uses the same timing provider and search options as the first.

## Scalar fallback

The fallback applies `vpu_pack`, a raw-result VSTORE, the scalar interleave below, and a VLOAD. The body copies one row per iteration: eight words, 32 iterations, 16-byte source-half steps and a 32-byte destination step. The mandatory scalar stream branches back 27 instruction words, with a B immediate of -54. [Explicit DMA may remain in flight across the loop](pack-dma-overlap.md).

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

## Pipelined one-row copy

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
count, branch and instruction count are unchanged; nothing is unrolled or
duplicated.

A row now takes 29 cycles under `--insert-atlas-delays`, and 28 under
`--schedule-atlas-stream`, which moves the destination step into the
back-edge's delay slot, against about 43. (An earlier experiment that kept
loads ahead of stores with eight temporaries reached about 36, because its
stores still issued after all eight loads.)

The virtual scheduler's PACK cost (`costOf` in `AtlasVirtualScheduling.cpp`)
modelled the old serial copy, each word's load latency plus its store latency.
It now models a row as seven load intervals, then the last word's load and
store latencies and the row's two control instructions (the back-edge, and
its delay slot holding the destination step), which gives the measured 28
cycles.
`kPackRelayoutRowWords` (`AtlasVirtualAllocation.h`) is the row size both use.

## Scalar fallback: what does not change

- The buffer contract runs the loop from its entry registers and checks every
  relayout word against the row interleave, so it proves the reordered copy
  without change.
- The PACK loop recognizer (`AtlasPackLoop.h`) accepts any order of the PACK's
  scalar loads, stores and ALU work, so DMA still overlaps the loop.
- The register-allocation checker's PACK effects name the same registers.

## Scalar fallback: tests

- `test_buffer_contract_verification.py::test_relayout_stores_follow_the_next_copy_load`:
  the lowered copy order is `lw, (lw, sw) x 7, sw`, and after either timing
  pass a row has at most one store waiting on its load (the last) and takes at
  most 29 cycles.
- The buffer-contract mutations (31 or 33 rows, a shifted source word, an
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
