# Virtual evaluator coverage

[Issue #9](https://github.com/ucb-bar/atlas-mlir/issues/9) now has execution for `start`, BF16 boundary input/output, ReLU, and the scalar/CFG forms below, alongside the broader parser/input interface. Other execution forms fail explicitly. Recognition, compiler admission, and numerical qualification are separate claims. Runtime ownership and execution rules are in the [interpreter contract](virtual-ssa-interpreter-contract.md).

## Baselines and evidence

| Source | Exact revision | Role |
| --- | --- | --- |
| Local compiler | `7b1e3dab55c2d0bd05d51c8f6ec48e55bcedeb58` | Implementation/inventory baseline |
| `feat/virtual-scheduler`, [PR #13](https://github.com/ucb-bar/atlas-mlir/pull/13) head | `23fdbf51f89a829bfcd24bbd32fa179616e26d9f` | Compatibility target; not included in local baseline |
| `handwritten-implementation`, [PR #13](https://github.com/ucb-bar/atlas-mlir/pull/13) base | `66e8756befb30161bab9bc75a6380c9bbc47d6d4` | Default-branch baseline |
| Historical selected RTL | `0079c0541111197741a231c002e3843fa6f545b2` | Existing bounded compiler observations |
| Historically examined model | `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` | Historical comparisons, not current numerical policy |
| Current `npu-model` | `0c4a1f9ee508c9e81fc9f21354229fa3a51c86e6` | Numerical source to audit/reuse |
| Surrounding RTL checkout | `2ae0bef209df6db78c3de18e8f651bb43855cce9` | Distinct current source baseline |

The current ReLU LUT and lane/vector-engine source hashes match the model's [RTL provenance](../../../../../npu-model/tests/rtl/provenance.json). The adapter preserves raw bits through `RtlNumerics.unary`; execution currently admits finite normal BF16 values and positive zero. Special encodings remain unsupported by ReLU, while boundary-only copies preserve them. Matching the selected-core executable to this source evidence remains outstanding; historical observations qualify only their recorded inputs/configuration.

Schemas: [AtlasOps.td](../include/Atlas/AtlasOps.td), [AtlasTypes.td](../include/Atlas/AtlasTypes.td). Checks: [AtlasOps.cpp](../lib/AtlasOps.cpp), [AtlasVirtualVerification.cpp](../lib/AtlasVirtualVerification.cpp), and [AtlasVirtualToMachine.cpp](../lib/AtlasVirtualToMachine.cpp).

## All 23 Atlas forms

Names below follow `atlas.virtual_`. `S/B/F/E` denote virtual state/BF16/FP8/scale; `L8/L16/D` denote pending FP8-load/BF16-load/store handles; `W<u>/A<u>` denote weight/accumulator handles for unit `u = 0..1`. Tiles are logical 32×32 raw encodings. Attributes are i32 unless marked string; DMA address/length are i32 SSA operands. Evidence IDs refer to the numerical table below.

| Suffix | Operands → results | Admission / meaning |
| --- | --- | --- |
| `start` | `() → S` | Unique first entry operation |
| `input_bf16` | `S → S, B` | Nonnegative `index`; snapshot input |
| `input_fp8` | `S → S, F` | Nonnegative `index`; snapshot input |
| `output_bf16` | `S, B → S` | Nonnegative unique `index`; publish snapshot |
| `vpu_unary` | `B → B` | String `kind`: `mov`, `relu`; V1 |
| `vpu_binary` | `B, B → B` | String `kind`: `add`; V2 |
| `pack_fp8` | `B → F` | `scale_code = 127`; P |
| `scale_constant` | `() → E` | Raw `code = 0..255`; immutable |
| `mxu_matmul` | `F, F → B` | `unit = 0..1`; reset contraction/readout; M |
| `dma_load_fp8` | `S, i32, i32 → S, L8` | Length 1,024; pending load; D |
| `dma_load_bf16` | `S, i32, i32 → S, L16` | Length 2,048; pending load; D |
| `dma_await_fp8` | `S, L8 → S, F` | Consume handle; expose tile; D |
| `dma_await_bf16` | `S, L16 → S, B` | Consume handle; expose tile; D |
| `dma_store_fp8` | `S, F, i32, i32 → S, D` | Length 1,024; capture source; D |
| `dma_store_bf16` | `S, B, i32, i32 → S, D` | Length 2,048; capture source; D |
| `dma_wait` | `S, D → S` | Consume handle; complete external output; D |
| `mxu_load_weight` | `S, F → S, W<u>` | `unit` matches handle; resident identity; M |
| `mxu_load_acc_fp8` | `S, F → S, A<u>` | `unit` matches handle; unscaled seed; M |
| `mxu_load_acc_bf16` | `S, B → S, A<u>` | `unit` matches handle; BF16 seed; M |
| `mxu_reset` | `S, F, W<u> → S, A<u>` | Matching units; initial contraction; M |
| `mxu_accumulate` | `S, F, W<u>, A<u> → S, A<u>` | Matching units; consume version; M |
| `mxu_readout_bf16` | `S, A<u> → S, B` | Consume version, retain weights; M |
| `mxu_readout_fp8` | `S, A<u>, E → S, F` | Constant scale; selected converter; R |

The dialect also recognizes unary `recip`, `exp`, `exp2`, `square`, `cube`, `sin`, `cos`, `tanh`, `log2`, `sqrt` and binary `sub`, `mul`, `min`, `max`; these and pack scales other than 127 are unsupported by initial evaluator admission and physical lowering.

## Scalar and CFG forms

| Form | Signature / policy |
| --- | --- |
| `arith.constant` | `() → i1/i32`; matching integer attribute |
| `arith.addi` | `i32, i32 → i32`; modulo `2^32`, flags absent or `none` |
| `arith.cmpi` | `i32, i32 → i1`; `eq/ne/slt/sgt/sle/sge/ult/ugt/ule/uge` |
| `cf.br` | Current state first on edge; simultaneous argument binding |
| `cf.cond_br` | i1 condition; select one edge, both statically carry current state |
| `func.return` | Return exactly current state; lowering additionally requires a unique return block |

Parsing accepts flat IR or a selected function; physical lowering requires exactly one function returning state with i1/i32 entry controls. Non-entry arguments are state followed by BF16/i1/i32, excluding FP8, scales, and resource handles. Compiler verification requires reachable blocks and one return block containing CFG outputs/stores/waits. Dominating tensor/scale definitions remain allowed independently of block-argument restrictions.

Execution independently checks reachable blocks, SSA dominance/order, edge arity/types, and current-state identities on both conditional edges. It binds arguments simultaneously and evaluates only the chosen path, with a positive `max_steps` budget counting every executed operation. Unsupported operation modes are rejected before execution; numerical-domain checks occur only when the operation executes. Multiple returns and path-dependent outputs are meaningful to the evaluator even when outside the compiler's lowering subset.

The contract details DMA address proof/alignment/range and handle rules. Both compiler snapshots expose all 23 forms; local bounds are one pending DMA and one weight/live accumulator per unit, while the target uses `kMaxPendingVirtualDMA = 2` and `kVirtualMXUSlots = 2`. Completion/version tracking is logical and block-local; physical placement is not an input. `atlas.input_dram_base`, `atlas.output_dram_base`, and optional `atlas.control_dram_base` govern compiler materialization, not logical boundary indices.

## Numerical sources to audit

Current model [numerics and layouts](../../../../../npu-model/docs/rtl-timing.md#numerical-behavior-and-layouts) supply reusable components. ReLU's raw-bit adapter is tested; target/configuration compatibility remains required for selected-core comparison.

| ID | Existing evidence | Remaining audit |
| --- | --- | --- |
| V1 | Provenance-matched ReLU table and raw-bit adapter; historical [ReLU](vpu-relu-observation.md) | Selected-core comparison; negative zero/subnormals/NaN/infinity; bit-exact `mov` |
| V2 | FP32 add/BF16 chop; [addition](vpu-add-observation.md) | Canonical NaN and raw-bit adapter; `1 + 3/512` distinguishes chop `0x3f80` from nearest-even `0x3f81` |
| P | Model pack converter; [E8M0 pack](vpu-e8m0-pack-observation.md) | Rounding/saturation/underflow/special codes; logical output versus physical PACK permutation |
| M | Unit-specific integer numerics; [discriminator](mxu-arithmetic-discriminator-observation.md), [MXU0 continuation](mxu0-k-continuation-observation.md), [MXU1 continuation](mxu1-continuation-observation.md) | MXU0 per-MAC versus MXU1 anchor behavior; geometry, seeds, product order, continuation/readout, exceptional encodings, W[N,K] orientation |
| R | Model MXU converter; [explicit MXU reference](dialect-reference.md) | Special scale codes; VPU pack divides by scale, MXU pop multiplies; distinct treatment of rounded FP8 `0x7f` |
| D | Model DMA; [pointer lifetime](dma-pointer-lifetime-observation.md) | Little-endian serialization, source stability, snapshots, conflicting ranges, visibility, two-transfer ordering |

At the current model pin, relevant specifications are [parameters](../../../../../npu-model/npu_spec/02_system_parameters/README.md), [registers/state](../../../../../npu-model/npu_spec/03_registers_and_execution_state/README.md), [functional units](../../../../../npu-model/npu_spec/04_functional_units/README.md), [memory](../../../../../npu-model/npu_spec/05_memory_model/README.md), and [instructions](../../../../../npu-model/npu_spec/06_instruction_set/README.md). They describe BF16 register pairs, two MXU slots, scale payloads/alignment, and current single-delay-slot handling. Instruction pseudocode alone does not establish rounding. Historical [discrepancies](source-discrepancies.md), including older two-delay-slot text and model arithmetic, must not substitute for current policy. Virtual CFG has no machine delay slots.

## Validation boundary

[Interface tests](../test/test_virtual_evaluator_interface.py) cover parsing and immutable inputs. [Execution tests](../test/test_virtual_evaluator_execution.py) check shared-input preservation, fresh inputs, finite BF16 ReLU, state/SSA identity, and unsupported modes. [CFG tests](../test/test_virtual_evaluator_cfg.py) cover integer boundaries, diamonds, untaken effects, loops, simultaneous swaps, malformed SSA/edges, and step limits.

The [shared-input comparison](../test/test_virtual_evaluator_core.py) includes two outputs, input/guard preservation, and a ReLU-to-MOV mutation. The [CFG comparisons](../test/test_virtual_evaluator_cfg_core.py) reuse compiled words across fresh tiles and controls for branches, all ten predicates, wrapping addition, and loops. Evaluator expectations and lowering/LLVM word agreement passed for 16 CFG programs; actual core execution is pending external ARC/ModeLIR assets. Skipped core tests do not complete these comparisons.

Remaining VPU/pack modes, MXU chains, DMA completion, and original/scheduled comparisons remain future work. These checks do not qualify matrix layout, resource release, or timing.
