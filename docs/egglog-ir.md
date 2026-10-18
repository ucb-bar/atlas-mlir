# Atlas virtual IR representation

`tools/atlas_egglog_ir.py` encodes the immutable dataflow of a flat virtual module or one argument-free, straight-line virtual function. It covers VPU elementwise operations, row/column reductions and raw-bit splats, FP8 transpose, PACK, pure selected-engine matmul, scale constants, boundary inputs/outputs, atomic DMA loads/stores, MXU weight/accumulator loads, reset/accumulate and BF16/FP8 readouts. Scalar i32 constants and plain `arith.addi` stay fixed. CFGs, loops, function arguments, nonvirtual IR and additional scalar operations are explicitly unsupported.

The base dialect has no virtual UNPACK operation. Physical `atlas.vpu_pack` with `direction = "fp8_to_bf16"` belongs to the machine seam and is unsupported by this virtual bridge. Existing virtual FP8 accumulator-load/BF16 readout sequences are represented as fixed operations, without claiming they implement the physical UNPACK identity.

`Program.operations` retains every operation's SSA results/operands, attributes, input/output types and source line. State tokens, MXU handles, scalar controls, effect ordering and MXU engine selection stay outside eclasses. Tile results from fixed operations become unique `Node("anchor", (), (("id", "%ssa"),), type)` leaves. Thus independent loads and stateful MXU results remain distinct even when they share types and source operands. Their computational dependencies remain accessible through the producing operation for external analyses.

`Node(op, operands, attrs, type)` is immutable. Operation names retain their `virtual_` prefix; attribute values retain canonical raw MLIR strings, for example `("kind", '"relu"')`, `("bits", "16256 : i32")` and `("finite_src", "unit")`. `node.attribute(name)` decodes those values to Python strings, integers, booleans or `None`. Both generic attributes and MLIR `<{properties}>` syntax parse. Namespaced `atlas.*` diagnostic attributes are preserved as opaque data. The bridge never treats them or `finite_src` as a proven fact or accepted caller contract.

The optional `atlas_egglog_schema` module defines `Value.anchor(id, type)` and `Value.apply(op, attrs_json, type, children)`, with `Inputs.empty()` / `Inputs.cons(head, tail)`. `attrs_json` canonically serializes a sorted list of raw attribute pairs. `to_egglog` / `from_egglog` (also `encode_node` / `decode_node`) translate between these constructors and `Node`. No arithmetic rewrite is enabled by this representation.

```python
from atlas_egglog_ir import load_program, egglog_roundtrip

program = load_program("test/examples/virtual_attention_tile.mlir")
decoded_text, extracted_roots = egglog_roundtrip(program)
```

`pure_roots(program)` returns one root per original pure SSA result. `decode_program(program, extracted_roots)` requires exactly that root set and reconstructs each definition from its extracted expression. It validates operation signatures, raw-bit/scale ranges, anchor identities, result types and dominance, reconnects fixed consumers when a pure root becomes an anchor or a dominating original pure operand, and materializes new pure dependencies before their use. It preserves original source spelling for unchanged expressions, including comments, properties, metadata and shared SSA definitions. The result is structurally reparsed before return. Exact equivalence remains the caller's obligation; this API deliberately permits a different splat so tests can prove that extraction controls the output.

The optimization path can request `decode_program(program, extracted_roots, prune_dead_pure=True)`. A backward use/definition walk then removes pure definitions unreachable from fixed operations, including dependencies made dead by a rewrite. State, DMA, boundary I/O, scalar controls and stateful MXU operations always remain. This option can change an unchanged extraction if its source already contains dead pure definitions; its default is `False` so no-rewrite roundtrips retain source and instruction identity.

New subexpressions are materialized near each changed root. The decoder reuses dominating anchors and that root's original operands; it does not introduce sharing across different roots. This bounds new constant sharing while preserving existing source sharing. It does not establish physical register allocation or schedule feasibility: verify and lower every candidate through the Atlas compiler before execution.

Parsing and structural decoding require only Python's standard library. Egglog extraction uses the optional pinned dependency in `tools/requirements-egglog.txt`. Run `python -m unittest discover -s test -p test_egglog_ir.py -v`; the three backend tests skip when egglog is unavailable. These tests cover actual union/extraction followed by elimination, extracted-value mutation, new dependencies, fixed-consumer reconnection, bounded rematerialization, contracts as opaque data, malformed signatures, resource units, accumulator reuse and forged/non-dominating anchors. Compiler byte-equivalence and per-fixture exclusions belong to the separate retained integration manifest; these unit tests make no model or RTL execution claim.
