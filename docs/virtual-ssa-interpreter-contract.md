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

[`atlas_virtual_evaluator.py`](../tools/atlas_virtual_evaluator.py) executes the
[admitted forms](#admitted-forms-and-numerical-sources) with audited `npu-model`
numerics; selected RTL and matching CIRCT artifacts, not that model's timing
assumptions, establish target behavior.

`parse_program` checks signatures, types, attributes and modes in every block
and keeps xDSL SSA/block identities (reparse after mutating the IR).
`validate_inputs` requires exactly the declared indices/formats, including
those on untaken paths, and controls in entry argument order.
`evaluate(program, inputs, max_steps=10000)` first checks dominance, edges,
state flow, handle rules and DMA address proofs in every block, then counts one
step per executed operation, applies effects and numerical checks only on the
chosen path and returns only executed outputs; unlike lowering, it accepts
multiple returns and path-dependent outputs. `compare_results` checks output
indices, raw BF16 bits and mapped memory bytes, ignoring output/region order and
adjacent region partitioning but distinguishing unmapped bytes from zero.

A `Tile` holds 1,024 raw BF16 words or FP8 E4M3 bytes, a `Scalar` unsigned
i1/i32 bits (-1 is `Scalar(32, 0xffffffff)`), and a `MemoryRegion` a nonempty
byte snapshot within 32-bit DRAM; supplied regions cannot overlap. Logical tiles are
independent of transport layout: BF16 pair-halves transport places a word at
byte offset `(col // 16) * 1024 + (row * 16 + col % 16) * 2`, and FP8 payloads
are 1,024 row-major bytes. Weights are `W[N,K]`, so contraction is
`A[M,K] @ W[N,K].T`; do not derive expected values from compiler relayout.

Install `tools/requirements-virtual-evaluator.txt` in the pinned model's
environment (Python 3.14, Torch 2.11.0, NumPy 2.4.4) with the model source root
on `PYTHONPATH`; compiler-backed checks also need `ATLAS_OOT_BIN_DIR` and
`ATLAS_LLVM_BIN`.

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

## Explicit DMA completion and scalar values

Channel-free DMA tile operations take ordinary i32 SSA values for DRAM byte addresses and byte lengths. The current admission proves these operands from constants and unflagged wrapping additions, requires complete FP8/BF16 tiles (1,024/2,048 bytes), and checks 32-byte DRAM alignment and the selected 32-bit address range. No scalar register is encoded by the SSA name. The contract fixes the upper DRAM half at zero and does not yet admit runtime DMA addresses or lengths.

An interpreter must distinguish a pending transfer from a ready tensor. `virtual_dma_load_fp8/bf16` creates a pending-load identity; its matching `virtual_dma_await_fp8/bf16` produces the usable tile. `virtual_dma_store_fp8/bf16` captures an immutable source tile into transfer-owned staging, and `virtual_dma_wait` establishes completion of the external write. Treat the staging as a private logical buffer owned through completion, not as an assigned VMEM window. An untimed interpreter may perform the copy eagerly internally, but must preserve these visibility and handle-lifetime rules.

Up to two pending transfers may complete in either order within their defining block ([admission](dialect-reference.md#channel-free-virtual-dma-and-scalar-ssa)); awaiting B exposes B, not A. Two transfers whose proven spans share DRAM bytes while either writes are never pending together, as the dialect verifier requires. Keep transfer identities independent of channels, staging windows and scalar helpers; scalar capture at launch does not release source memory (physical ownership and release belong to [issue #10](https://github.com/ucb-bar/atlas-mlir/issues/10)).

Every executed transfer span must lie in supplied regions or ABI payload mappings without holes, and loads must read initialized bytes. Boundary I/O and explicit DMA share execution-owned bytes, returned as final host-visible snapshots; unwritten output bytes and FP8 slot padding are undefined unless supplied.

The environment must keep external load sources stable and exclude conflicting accesses to transfer ranges until completion. The IR checks do not prove host-side synchronization.

## Explicit MXU handle extension

Weight and accumulator-version handles are logical identities, not physical slots; the [handle rules](dialect-reference.md#explicit-virtual-mxu-resources) admit up to two weights and two accumulator chains per unit, which an interpreter must keep distinct.

Each explicit MXU operation advances the virtual state token. Reset starts a contraction, not merely a zero accumulator.

FP8 readout takes an immutable `!atlas.virtual_scale` produced by the pure `virtual_scale_constant` operation. Keep its raw code (`0..255`) in the value environment rather than treating it as a mutable physical scale register. The current slice admits constant scale definitions with ordinary dominance, but no scale block arguments or runtime scale inputs. A numerical interpreter must use the selected MXU converter, including its special-code behavior; VPU packing is not a substitute for MXU FP8 readout. FP8 readout produces the logical row-major tile layout used by virtual FP8 inputs.

Reuse audited `RtlNumerics` components with MXU0's per-MAC BF16 rounding and MXU1's anchor accumulation, not a generic matmul.

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

## Admitted forms and numerical sources

The evaluator executes these 23 `atlas.virtual_*` forms plus `arith.constant/addi/cmpi` (all ten predicates), `cf.br/cond_br` and `func.return`. `S/B/F/E` are state/BF16/FP8/scale, `L8/L16/D` pending FP8-load/BF16-load/store handles and `W<u>/A<u>` weight/accumulator handles on unit `u = 0..1`; final letters name audit rows.

| Suffix | Operands → results | Admission / meaning |
| --- | --- | --- |
| `start` | `() → S` | Unique first entry operation |
| `input_bf16` / `input_fp8` | `S → S, B/F` | Nonnegative `index`; snapshot input |
| `output_bf16` | `S, B → S` | Unique `index`; publish snapshot |
| `vpu_unary` | `B → B` | `kind`: `mov`, `relu`; V1 |
| `vpu_binary` | `B, B → B` | `kind`: `add`; V2 |
| `pack_fp8` | `B → F` | `scale_code = 127` (`RtlNumerics.to_fp8`); P |
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

In required mode on 2026-10-09, every row of `test/test_virtual_evaluator_core.py`, including scheduled variants, matched the reference on the selected-core Arc model, which is not cross-checked against Verilator or VCS; the evidence is bounded by its fidelity to the selected RTL, and [issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9) closes only when the team accepts that bound.

| ID | Existing evidence | Selected-core rows | Remaining audit |
| --- | --- | --- | --- |
| V1 | MOV raw copy and ReLU sign-bit rule, all 65,536 encodings; [ReLU](vpu-relu-observation.md) | `shared_relu`, `vpu`, `dma_bf16_relu`, `cfg_relu_loop_*`; ReLU→MOV mutation detected | Patterned tiles, not all 65,536 encodings, on the core |
| V2 | FP32 RNE/BF16 chop path with canonical NaN and directed signed-zero, subnormal, overflow, infinity and NaN cases; [addition](vpu-add-observation.md) | `vpu` and `dma_pending_work` (ADD on patterned tiles) | The `1 + 3/512` chop-versus-nearest-even literal and the directed special cases on the core |
| P | Scale-127 converter: ties, carry, saturation, underflow, specials, logical order; [E8M0 pack](vpu-e8m0-pack-observation.md) | `mxu{0,1}_legacy_pack` and the physical converter/transport probe | PACK→MXU chains beyond `legacy_pack` |
| M | Per-MAC MXU0 and anchor MXU1 adapters: orientation, rounding, reset/continuation, seeds, handles; [discriminator](mxu-arithmetic-discriminator-observation.md) | `mxu{0,1}_{continuation,seed_bf16,seed_fp8}`, `dma_mxu_*`, `two_chains_*` | Exceptional BF16 accumulator encodings on the core |
| R | MXU converter, special scale codes, reserved rounded ±480; [MXU1 continuation](mxu1-continuation-observation.md) | None (no row reads out FP8) | Selected-core comparison; VPU pack divides by scale while MXU pop multiplies |
| D | Owned snapshots, matching completion, raw serialization, mapped spans, guards, two-handle cases; [pointer lifetime](dma-pointer-lifetime-observation.md) | `dma_pending_work`, `two_dma_reverse_await`, `dma_fp8_copy`, traffic counts | Pending-write conflicts and mixed boundary/memory write aliases unqualified |

Pins: selected RTL `0079c0541111197741a231c002e3843fa6f545b2` and `npu-model` `0c4a1f9ee508c9e81fc9f21354229fa3a51c86e6`. The base reference suites (`test_*_reference.py`, handoff and inventory tests) assert this RTL pin, but the model was built from a checkout reporting `2ae0bef`, so those pin assertions fail until refreshed; their executions on the model pass.

### Core comparison prerequisites

- Select the AtlasCore closure from the Chipyard FIRRTL and run CIRCT firtool-1.75.0's `arcilator` with its full LLVM pipeline (`--observe-registers --observe-memories --observe-named-values --state-file`; that release lacks `--hw-convert-bitcasts` and `--arc-lower-arrays`). Split the emitted LLVM IR into basic blocks of at most 128 instructions before `llc -O0 -filetype=obj -relocation-model=pic` (unsplit, instruction selection on the 800k-instruction `AtlasCore_passthrough` never finishes; split, about 35 s), then link with `gcc -shared -fPIC -Wl,--no-undefined`. The splitting helper lives outside this repository with the selected-core artifacts.
- The test drives the model through ModeLIR `mlc.backends.cosim_atlas.run_program` with explicit `halt_signal="scalar/halt_now"`.
- Environment: `ATLAS_REQUIRE_VIRTUAL_CORE=1`, `ATLAS_OOT_BIN_DIR`, `ATLAS_LLVM_BIN`, `ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`; required mode fails on missing prerequisites and ordinary discovery skips these checks.
