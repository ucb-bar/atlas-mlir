# Interpreting Atlas virtual SSA

The virtual Atlas stage uses ordinary MLIR SSA. A `%name` identifies a value
produced by an operation or received as a block argument. It is not a physical
tensor register. `cf.br` and `cf.cond_br` provide values to the destination
block's arguments; those arguments serve the role often called *phi nodes*.
This is a small control-flow convention for an interpreter, not a requirement
to reconstruct physical registers before interpreting the virtual program.

For example:

```mlir
^entry:
  %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %v = "atlas.virtual_input_bf16"(%s) {index = 0 : i32}
      : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  cf.br ^merge(%s1, %v : !atlas.virtual_state, !atlas.virtual_bf16)
^merge(%state: !atlas.virtual_state, %tile: !atlas.virtual_bf16):
  // %tile is the value passed by the edge, not a physical register read.
```

This fragment omits a return terminator; the checked fixtures under
`test/examples/virtual_bf16_cfg.mlir` have executable syntax and complete
control-flow examples.

## Evaluator interface

The evaluator provides a
[parser and typed runtime records](../tools/atlas_virtual_evaluator.py).
Parsing checks supported signatures, types, attributes, and modes; it does not
check state flow, dominance, handle lifetimes, DMA ranges, or hardware legality.
[Admitted forms and numerical sources](#admitted-forms-and-numerical-sources)
lists the operations and remaining questions for
[issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9).

Logical meaning comes from this contract, selected RTL and matching CIRCT
artifacts establish target behavior, and audited `npu-model` components supply
numerics; model timing assumptions do not establish RTL guarantees. The
interface of [PR #13](https://github.com/ucb-bar/atlas-mlir/pull/13) is the
compatibility target.

`parse_program` accepts flat IR or a selected function and retains xDSL SSA/block
identities; reparse after mutating its IR. `validate_inputs` requires exactly
the declared indices/formats, including untaken paths, and controls in entry
argument order.

`evaluate(program, inputs, max_steps=10000)` executes BF16/FP8 boundary inputs,
BF16 outputs, MOV/ReLU/ADD, scale-127 FP8 pack, i1/i32 controls, wrapping i32
addition, the ten integer comparisons, branches, returns, both MXU units and
explicit DMA with owned snapshots. It checks SSA dominance, edge types/arity and
state flow before execution, binds branch arguments simultaneously, and counts
each executed operation as one step; exhaustion raises `VirtualInterfaceError`
with the block and operation position. Unsupported modes are rejected even on
untaken paths, effects and numerical checks run only on the chosen path, and
only executed outputs appear in the result. ADD uses FP32 nearest-even addition
followed by BF16 truncation and canonical NaN. Multiple returns and
path-dependent outputs are accepted although lowering excludes them.

`compare_results(expected, actual)` checks identical output indices, raw BF16
bits, and mapped memory bytes, ignoring output/region ordering and adjacent
region partitioning but distinguishing unmapped bytes from zero. Failures report
the first differing tile coordinate or byte address. `evaluate_tile_operation(op,
operands)` checks a parsed pure VPU/pack operation on immutable tiles; pack uses
`RtlNumerics.to_fp8` with scale 127 and preserves logical row-major order.

```python
from tools.atlas_virtual_evaluator import RuntimeInputs, Scalar, Tile, parse_program

program = parse_program(source, function="choose_tile")
inputs = RuntimeInputs({0: Tile("bf16", (0x3f80,) * 1024)}, (Scalar(1, 1),))
program.validate_inputs(inputs)
```

Runtime values own immutable copies: `Tile(format, bits)` holds 1,024 raw BF16
words or FP8 E4M3 bytes; `Scalar(width, bits)` holds unsigned i1/i32 bits
(-1 is `Scalar(32, 0xffffffff)`); `MemoryRegion(address, data)` is a nonempty
byte snapshot within 32-bit DRAM, and supplied regions cannot overlap.
`RuntimeInputs(tiles, controls, memory)` and `EvaluationResult(outputs, memory)`
carry the boundary data and final region snapshots.

Logical tiles are independent of transport layout: BF16 pair-halves transport
places a word at byte offset `(col // 16) * 1024 + (row * 16 + col % 16) * 2`,
and FP8 payloads are 1,024 row-major bytes. Weights are `W[N,K]`, so contraction
is `A[M,K] @ W[N,K].T`; do not derive expected values from compiler relayout.

Install `tools/requirements-virtual-evaluator.txt` (xDSL 0.65.0) in an
environment compatible with the pinned model (Python 3.14, Torch 2.11.0, NumPy
2.4.4 in this session) with the model source root on `PYTHONPATH`. Compiler-backed
checks also need `ATLAS_OOT_BIN_DIR` and `ATLAS_LLVM_BIN`; scheduled comparisons
run unconditionally because the virtual scheduler is part of this compiler.

## Recommended interpreter state

Keep three distinct things:

1. An environment mapping MLIR `Value` identity to immutable runtime values.
   Virtual BF16/FP8 tiles are logical 32×32 arrays with explicit encoding.
   Scalar i1/i32 controls are ordinary scalar values. Use the MLIR `Value`
   object or a stable parsed identifier, never the textual spelling `%t1`.
2. External state for declared input/output buffers and visible effects.
   `!atlas.virtual_state` orders these effects; it is not the tensor data.
   The interpreter may represent the token as a monotonically increasing
   event identity. It must reject a boundary op given the wrong current token.
3. A program counter `(block, operation index)` and a finite step budget.
   A loop revisits the same static operations with new runtime bindings.

At a branch, evaluate all outgoing operands in the *current* environment,
then bind the destination block arguments simultaneously and continue at its
first operation. Simultaneous binding matters for a backedge that swaps two
values: sequential assignment would overwrite one source. For a conditional
branch, evaluate only the chosen successor. Keep an execution trace with the
source block, chosen edge, bound argument identities, boundary events, and
operation location. This makes loop and state errors diagnosable.

```text
values = evaluate(chosen_edge.operands, current_environment)
next_environment = bind_simultaneously(chosen_block.arguments, values)
block, operation_index = chosen_block, 0
```

The environment can be per dynamic block visit. A global map is also possible
if block arguments and operation results are rebound on every visit and no
stale binding can be read. The evaluator checks SSA dominance and edge
arity/types independently of xDSL parsing, along with static current-state
identities on both branch edges and fresh runtime tokens on the chosen path. The
compiler verifier additionally confines outputs to one return block, a lowering
restriction.

## Explicit DMA completion and scalar values

Channel-free DMA tile operations take ordinary i32 SSA values for DRAM byte addresses and byte lengths. The current admission proves these operands from constants and unflagged wrapping additions, requires complete FP8/BF16 tiles (1,024/2,048 bytes), and checks 32-byte DRAM alignment and the selected 32-bit address range. No scalar register is encoded by the SSA name. The contract fixes the upper DRAM half at zero and does not yet admit runtime DMA addresses or lengths.

An interpreter must distinguish a pending transfer from a ready tensor. `virtual_dma_load_fp8/bf16` creates a pending-load identity; its matching `virtual_dma_await_fp8/bf16` produces the usable tile. `virtual_dma_store_fp8/bf16` captures an immutable source tile into transfer-owned staging, and `virtual_dma_wait` establishes completion of the external write. Treat the staging as a private logical buffer owned through completion, not as an assigned VMEM window. An untimed interpreter may perform the copy eagerly internally, but must preserve these visibility and handle-lifetime rules.

At most two transfers may be pending, awaited in either order. Completion must
occur in the defining block, repeated or mismatched completions are rejected,
implicit boundary I/O and VPU pack require no pending transfers, and a completed
store counts as output (CFG stores and waits occur in the unique return block).

Awaiting B exposes B, not A. Logical completion remains separate from physical
progress; keep identities independent of channels, staging windows, and scalar
helpers. Scalar capture at launch does not release source memory; physical
ownership and release belong to
[issue #10](https://github.com/ucb-bar/atlas-mlir/issues/10).

The environment must keep external load sources stable and exclude conflicting accesses to transfer ranges until completion. The IR checks do not prove host-side synchronization.

Execution checks complete-tile lengths and addresses from i32 constants and
wrapping additions, 32-byte alignment, addresses at or above `0x80000000`, and a
widened end within `2^32`. Every executed span must be covered by supplied
regions or ABI payload mappings (holes not allowed), and loads require
initialized bytes. Executed transfers may share read ranges; overlap involving a
pending write is rejected.

ABI boundaries and explicit DMA share execution-owned bytes: returned mapped
outputs reflect final host-visible bytes, and unwritten output bytes and FP8 slot
padding are undefined unless supplied. No channels, VMEM windows, or cycles are
simulated.

## Explicit MXU handle extension

The dialect represents weight loading, reset contractions, accumulation, and
BF16/FP8 readout. `!atlas.virtual_mxu_weight<unit>` identifies resident weights;
`!atlas.virtual_mxu_acc<unit>` identifies a consumable accumulator version.
Neither names a physical slot. Each unit admits two weights and two independent
accumulator chains, and the interpreter must retain those distinct identities.

Each explicit MXU operation advances state. A weight load creates an identity;
reset starts a contraction, not merely a zero accumulator. FP8/BF16 accumulator
loads seed a chain and preserve weights; FP8 seeds decode without a scale.
Accumulation consumes the current version and produces its successor. Either
readout consumes that version while preserving weights. All handles must agree
on their unit, remain in their defining block, and every accumulator must be
read out before exit. Legacy `virtual_mxu_matmul` cannot overlap live explicit
weights or accumulators on its unit.

FP8 readout takes an immutable `!atlas.virtual_scale` produced by the pure `virtual_scale_constant` operation. Keep its raw code (`0..255`) in the value environment rather than treating it as a mutable physical scale register. The current slice admits constant scale definitions with ordinary dominance, but no scale block arguments or runtime scale inputs. A numerical interpreter must use the selected MXU converter, including its special-code behavior; VPU packing is not a substitute for MXU FP8 readout. FP8 readout produces the logical row-major tile layout used by virtual FP8 inputs.

The compiler's structural and lifetime checks do not qualify numerical
execution; slot and scratch-scale assignments stay outside the logical
reference. Reuse audited `RtlNumerics` components with MXU0's per-MAC BF16
rounding and MXU1's anchor accumulation, not a generic matmul. MXU FP8 readout
multiplies by the scale and must not use VPU packing.

Preflight counts remaining weight uses and unconsumed accumulator versions per
block, applies the two-handle admission, rejects same-unit legacy overlap and
cross-block handles, and requires accumulator readout before exit, including
untaken blocks. Runtime handles bind immutable tiles afresh on each block visit.

## Semantic boundaries

Interpret the virtual SSA stage and the physical machine stage separately.
Virtual `atlas.virtual_mxu_matmul` and `atlas.virtual_pack_fp8` describe
selected fixed-tile operations without physical register numbers. Their
reference semantics must name the selected MXU unit, arithmetic/rounding
policy, pack scale, and logical layout. The machine stage has concrete
registers, VMEM/DRAM addresses, launch/completion events, and delay slots; its
interpreter needs state transitions and timing. Reading a physical register
number from `%t1` would mix the two stages and hide allocator bugs.

An independent interpreter should compare the virtual result with the checked
machine execution on the same runtime inputs, including outputs, preserved
inputs, guards, and state. It should not call the production pack/allocator or
use its chosen physical layout as the expected answer. If a numerical policy
or timing fact is unknown, return an explicit unsupported/unknown result rather
than inventing a computation. The current two-layer MLP fixture is checked on
finite, exactly representable inputs; it is not a full-domain FP8/BF16 oracle.

## Initial tests

- A diamond whose two edges pass different tiles to one block argument.
- A loop-carried tile and a backedge swap, verifying simultaneous binding.
- A conditional branch that does not execute effects on the untaken edge.
- Multiple outputs and one shared producer, preserving output order.
- A wrong virtual-state edge, missing value, bad arity, and a step-budget exit.
- The same compiled program with different runtime tensors and controls.
- Virtual versus selected-core machine results on directed numerical and
  memory-guard cases, with the comparison's exact precision domain recorded.

The parser/input layer, scalar/CFG, admitted VPU/pack, both MXU units, and DMA
are implemented in this repository's Python verification tooling; selected-core
comparisons have not executed. These checks do not complete
[issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9).

## Admitted forms and numerical sources

The evaluator executes the 23 virtual forms below (names follow `atlas.virtual_`) plus the scalar/CFG subset (`arith.constant/addi/cmpi`, `cf.br/cond_br`, `func.return`). `S/B/F/E` denote state/BF16/FP8/scale; `L8/L16/D` pending FP8-load/BF16-load/store handles; `W<u>/A<u>` weight/accumulator handles for unit `u = 0..1`. Tiles are logical 32×32 raw encodings; DMA address/length are i32 SSA operands. Letters in the last column refer to the audit table.

| Suffix | Operands → results | Admission / meaning |
| --- | --- | --- |
| `start` | `() → S` | Unique first entry operation |
| `input_bf16` | `S → S, B` | Nonnegative `index`; snapshot input |
| `input_fp8` | `S → S, F` | Nonnegative `index`; snapshot input |
| `output_bf16` | `S, B → S` | Unique `index`; publish snapshot |
| `vpu_unary` | `B → B` | `kind`: `mov`, `relu`; V1 |
| `vpu_binary` | `B, B → B` | `kind`: `add`; V2 |
| `pack_fp8` | `B → F` | `scale_code = 127`; P |
| `scale_constant` | `() → E` | Raw `code = 0..255` |
| `mxu_matmul` | `F, F → B` | `unit = 0..1`; reset contraction; M |
| `dma_load_fp8` / `dma_load_bf16` | `S, i32, i32 → S, L8/L16` | Length 1,024 / 2,048; D |
| `dma_await_fp8` / `dma_await_bf16` | `S, L8/L16 → S, F/B` | Consume handle; expose tile; D |
| `dma_store_fp8` / `dma_store_bf16` | `S, F/B, i32, i32 → S, D` | Length 1,024 / 2,048; capture source; D |
| `dma_wait` | `S, D → S` | Consume handle; complete external output; D |
| `mxu_load_weight` | `S, F → S, W<u>` | Resident identity; M |
| `mxu_load_acc_fp8` / `mxu_load_acc_bf16` | `S, F/B → S, A<u>` | Unscaled / BF16 seed; M |
| `mxu_reset` | `S, F, W<u> → S, A<u>` | Initial contraction; M |
| `mxu_accumulate` | `S, F, W<u>, A<u> → S, A<u>` | Consume version; M |
| `mxu_readout_bf16` | `S, A<u> → S, B` | Consume version, retain weights; M |
| `mxu_readout_fp8` | `S, A<u>, E → S, F` | Constant scale; selected converter; R |

Other VPU modes and pack scales other than 127 are unsupported by evaluator admission and physical lowering.

| ID | Existing evidence | Remaining audit |
| --- | --- | --- |
| V1 | MOV raw copy and ReLU sign-bit rule over all 65,536 encodings; [ReLU](vpu-relu-observation.md) | Selected-core comparison |
| V2 | FP32 RNE/BF16 chop path with directed signed-zero, subnormal, overflow, infinity and NaN cases; [addition](vpu-add-observation.md) | Selected-core comparison; `1 + 3/512` separates chop `0x3f80` from nearest-even `0x3f81` |
| P | Scale-127 converter: ties, carry, saturation, underflow, specials, logical order; [E8M0 pack](vpu-e8m0-pack-observation.md) | Physical converter/transport and PACK→MXU comparisons |
| M | Per-MAC MXU0 and anchor MXU1 adapters: orientation, rounding, reset/continuation, seeds, handles; [discriminator](mxu-arithmetic-discriminator-observation.md) | Selected-core comparison, including exceptional BF16 accumulators |
| R | MXU converter, special scale codes, reserved rounded ±480; [MXU1 continuation](mxu1-continuation-observation.md) | Selected-core comparison; VPU pack divides by scale while MXU pop multiplies |
| D | Owned snapshots, matching completion, raw serialization, mapped spans, guards, two-handle cases; [pointer lifetime](dma-pointer-lifetime-observation.md) | Selected-core comparison; pending-write conflicts and mixed boundary/memory write aliases unqualified |

| Pin | Revision |
| --- | --- |
| Selected RTL | `0079c0541111197741a231c002e3843fa6f545b2` |
| Current `npu-model` | `0c4a1f9ee508c9e81fc9f21354229fa3a51c86e6` |

Core execution is pending: no comparison of these forms against a selected-core artifact has run, so the evidence above is reference-side only.

### Core comparison prerequisites (not reproduced)

- Build the ARC model with arcilator `--inline=false`; the default inlining produced a function too large for native code generation to finish.
- CIRCT firtool-1.75.0 needs the full arcilator LLVM pipeline; `--hw-convert-bitcasts` and `--arc-lower-arrays` are absent from it.
- The comparison drives the model through ModeLIR `mlc.backends.cosim_atlas.run_program` with explicit `halt_signal="scalar/halt_now"`; a local ModeLIR checkout exists at `compiler/ModeLIR`.
- Environment: `ATLAS_REQUIRE_VIRTUAL_CORE=1`, `ATLAS_OOT_BIN_DIR`, `ATLAS_LLVM_BIN`, `ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`; required mode fails on missing prerequisites and ordinary discovery skips these checks.
