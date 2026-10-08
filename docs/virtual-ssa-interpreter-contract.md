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

Contract version `atlas.virtual-evaluator.v1` provides a
[parser and typed runtime records](../tools/atlas_virtual_evaluator.py).
Parsing checks supported signatures, types, attributes, and modes;
it does not prove state flow, dominance, handle
lifetimes, DMA ranges, or hardware legality. The
[coverage inventory](virtual-evaluator-coverage.md) lists the admitted operations,
source revisions, and remaining numerical questions for
[issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9).

The sources have distinct roles:

- The pinned [PR #13](https://github.com/ucb-bar/atlas-mlir/pull/13) interface is
  the compatibility target; the default branch supplies existing regressions.
  Scheduler and allocator decisions are subjects of verification.
- This contract and reconciled specifications define logical meaning. Selected
  RTL and matching CIRCT artifacts establish behavior for the target configuration.
- Audited `npu-model` components supply numerical implementations. Independent
  expectations and selected-core comparisons check their use; model timing
  assumptions do not establish RTL guarantees.

`parse_program` accepts flat IR or a selected function, requiring a name for
multi-function modules. It retains xDSL SSA/block identities and boundary/control
declarations. Reparse after mutating its IR. `validate_inputs` requires exactly
the declared indices/formats, including untaken paths, and controls in entry
argument order.

`evaluate(program, inputs, max_steps=10000)` executes BF16/FP8 boundary inputs,
BF16 outputs, MOV/ReLU/ADD, scale-127 FP8 pack, i1/i32 controls/constants,
wrapping i32 addition, all ten integer comparisons, branches, and returns.
It checks SSA dominance, edge types/arity, and state flow before execution;
branch arguments bind simultaneously. Each executed
operation, including branches and returns, consumes one step. Exhaustion raises
`VirtualInterfaceError` with the block and operation position.

Unsupported operation modes are rejected even on untaken paths. Effects and
numerical checks run only on the chosen path. MOV preserves every raw encoding;
ReLU returns positive zero for sign-set inputs and otherwise preserves bits.
ADD uses FP32 nearest-even addition followed by BF16 truncation and canonical
NaN, rejecting host arithmetic that flushes subnormals or changes rounding.
Only executed outputs appear in the result. Memory snapshots remain unchanged
until explicit DMA execution is implemented. The evaluator can interpret
multiple returns and path-dependent outputs; the compiler's narrower lowering
restrictions still apply separately.

`evaluate_tile_operation(op, operands)` checks a parsed pure VPU/pack operation
and its tuple of immutable tiles. Pack uses `RtlNumerics.to_fp8` with scale 127:
nearest-even rounding, signed saturation to 448, and positive zero for NaNs,
subnormals, and underflow. It preserves logical row-major order. This helper
exposes FP8 results directly; full-program FP8 observation awaits MXU/DMA
consumers because the dialect has no FP8 boundary-output operation.

```python
from tools.atlas_virtual_evaluator import RuntimeInputs, Scalar, Tile, parse_program

program = parse_program(source, function="choose_tile")
inputs = RuntimeInputs({0: Tile("bf16", (0x3f80,) * 1024)}, (Scalar(1, 1),))
program.validate_inputs(inputs)
```

Runtime values own immutable copies:

- `Tile(format, bits)`: 1,024 row-major raw BF16 words or FP8 E4M3 bytes,
  preserving signed zeros and exceptional encodings without quantization.
- `Scalar(width, bits)`: unsigned i1/i32 bits; encode -1 as
  `Scalar(32, 0xffffffff)`.
- `MemoryRegion(address, data)`: a nonempty byte snapshot within 32-bit DRAM.
  Supplied regions, including guards, cannot overlap.
- `RuntimeInputs(tiles, controls, memory)`: boundary tiles, scalar controls, and
  initial memory. `EvaluationResult(outputs, memory)` describes BF16 outputs and
  final snapshots of the same regions. FP8 stores are memory effects.

Logical tiles are independent of transport layout. BF16 pair-halves transport
uses little-endian words at byte offset
`(col // 16) * 1024 + (row * 16 + col % 16) * 2`.
FP8 payloads are 1,024 row-major bytes; implicit input slots reserve 2,048 bytes.
Weights are `W[N,K]`, one row per output column, so contraction is
`A[M,K] @ W[N,K].T`. Adapt that orientation explicitly for numerical helpers;
do not derive expected values from compiler relayout.

Install and test the interface in a dedicated environment:

```sh
python -m pip install -r tools/requirements-virtual-evaluator.txt
python -m unittest discover -s test -p test_virtual_evaluator_interface.py -v
```

The parser pins xDSL 0.65.0. Execution tests use the selected model's Python 3.14
environment with Torch and the model source root on `PYTHONPATH`:

```sh
python -m unittest discover -s test -p 'test_virtual_evaluator_*.py' -v
ATLAS_REQUIRE_VIRTUAL_CORE=1 python -m unittest discover -s test -p 'test_virtual_evaluator*core.py' -v
```

The core comparison additionally needs `ATLAS_OOT_BIN_DIR`, `ATLAS_LLVM_BIN`,
`ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE`, and `ATLAS_MODELIR_ROOT` as described in
the [README](../README.md). Required mode fails on missing core prerequisites;
ordinary discovery skips those checks. The shared-input, scalar/CFG, VPU,
and physical pack comparisons have not executed against a selected-core
artifact. The physical pack probe checks converter/transport behavior;
virtual pack lowering with an observable consumer remains unqualified. See the
[coverage inventory](virtual-evaluator-coverage.md) for evidence and limits.

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
stale binding can be read. xDSL parsing does not establish SSA dominance or
edge arity/types, so the evaluator checks those independently. It also checks
static current-state identities on both branch edges and fresh runtime tokens
on the chosen path. The existing compiler verifier additionally confines
outputs to one return block; this is a lowering restriction.

## Explicit DMA completion and scalar values

Channel-free DMA tile operations take ordinary i32 SSA values for DRAM byte addresses and byte lengths. The current admission proves these operands from constants and unflagged wrapping additions, requires complete FP8/BF16 tiles (1,024/2,048 bytes), and checks 32-byte DRAM alignment and the selected 32-bit address range. No scalar register is encoded by the SSA name. The contract fixes the upper DRAM half at zero and does not yet admit runtime DMA addresses or lengths.

An interpreter must distinguish a pending transfer from a ready tensor. `virtual_dma_load_fp8/bf16` creates a pending-load identity; its matching `virtual_dma_await_fp8/bf16` produces the usable tile. `virtual_dma_store_fp8/bf16` captures an immutable source tile into transfer-owned staging, and `virtual_dma_wait` establishes completion of the external write. Treat the staging as a private logical buffer owned through completion, not as an assigned VMEM window. An untimed interpreter may perform the copy eagerly internally, but must preserve these visibility and handle-lifetime rules.

The local baseline permits one pending DMA transfer; the pinned
[PR #13](https://github.com/ucb-bar/atlas-mlir/pull/13) target
permits two independent handles, awaited in either order. Both require completion
in the defining block and reject repeated or mismatched completions. Implicit
boundary I/O and VPU pack require no pending transfers. A completed store counts
as output; CFG stores and waits occur in the unique return block.

Awaiting B exposes B, not A. Logical completion remains separate from physical
progress: a transfer may finish before its handle is consumed. Keep identities
independent of channels, staging windows, and scalar helpers. Scalar capture at
launch does not release source memory; physical ownership and release belong to
[issue #10](https://github.com/ucb-bar/atlas-mlir/issues/10). Model FIFO progression
does not establish RTL completion order.

The environment must keep external load sources stable and exclude conflicting accesses to transfer ranges until completion. The IR checks do not prove host-side synchronization.

## Explicit MXU handle extension

The dialect represents weight loading, reset contractions, accumulation, and
BF16/FP8 readout. `!atlas.virtual_mxu_weight<unit>` identifies resident weights;
`!atlas.virtual_mxu_acc<unit>` identifies a consumable accumulator version.
Neither names a physical slot. The local baseline tracks one weight/accumulator
per unit; the pinned target permits two weights and two independent accumulator
chains per unit. The interpreter must retain those distinct identities.

Each explicit MXU operation advances state. A weight load creates an identity;
reset starts a contraction, not merely a zero accumulator. FP8/BF16 accumulator
loads seed a chain and preserve weights; FP8 seeds decode without a scale.
Accumulation consumes the current version and produces its successor. Either
readout consumes that version while preserving weights. All handles must agree
on their unit, remain in their defining block, and every accumulator must be
read out before exit. In the pinned target, legacy `virtual_mxu_matmul` cannot
overlap live explicit weights or accumulators on its unit.

FP8 readout takes an immutable `!atlas.virtual_scale` produced by the pure `virtual_scale_constant` operation. Keep its raw code (`0..255`) in the value environment rather than treating it as a mutable physical scale register. The current slice admits constant scale definitions with ordinary dominance, but no scale block arguments or runtime scale inputs. A numerical interpreter must use the selected MXU converter, including its special-code behavior; VPU packing is not a substitute for MXU FP8 readout. FP8 readout produces the logical row-major tile layout used by virtual FP8 inputs.

The compiler's structural/lifetime checks do not qualify numerical execution.
Keep slot and scratch-scale assignments outside the logical reference. Reuse
audited `RtlNumerics` components while preserving MXU0's per-MAC BF16 arithmetic
and MXU1's distinct anchor accumulator/readout; a generic matmul or shared
rounding shortcut is insufficient.

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

The parser/input layer, scalar/CFG execution, and admitted VPU/pack execution
are implemented in this repository's Python verification tooling. Their
selected-core comparisons are implemented but unexecuted; MXU and DMA
execution remain future work. These partial checks do not complete
[issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9).
