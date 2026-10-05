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
stale binding can be read. MLIR's verifier supplies static dominance and type
checks; an interpreter still needs dynamic arity, type, current-token, and
step-limit checks. The existing virtual stream verifier additionally requires
the current state token on each CFG edge and one return block for outputs.

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

This contract enables an interpreter in a separate package. It does not add
one to this OOT dialect or assert that all virtual operation semantics are
qualified.
