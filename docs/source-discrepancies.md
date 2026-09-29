# Source discrepancies and qualification obligations

Source observations for RTL `0079c0541111197741a231c002e3843fa6f545b2`
and local model `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`. These
are primarily static checks, with bounded branch, DMA, MXU0, MXU1, VPU, and XLU
diagnostic `AtlasCore` ARC executions. A fresh elaboration of the selected source copy
and a fresh ARC rebuild now link that standalone core model to the selected
Atlas Scala bytes. Fresh checkouts at the selected Atlas Git submodule pins
matched all 45 `sp26-fp-units` files and all 26 present `fpex` files byte for
byte; three `fpex` test files are absent from the Chipyard copy. Full SoC
execution remains unqualified.

| Subject | Selected RTL | Examined model | Handling here |
| --- | --- | --- | --- |
| VR operand fields | `ScalarDecoder.scala` uses `vd[12:7]`, `vs1[18:13]`, `vs2[24:19]`, six bits each | `npu_model/isa.py::VRType.to_bytecode` masks `vs1` to seven bits and `vs2` to five bits | Emitter uses RTL layout; test exercises a nonzero secondary operand. Model assembly encoding cannot be the sole oracle. |
| DMA config | `Instructions.scala::DMA_CONFIG_ANY` fixes funct7 to zero; `DMA_WAIT_ANY` fixes it to one | `DMA_CONFIG_CH0..7` set funct7 to one | Emitter uses RTL config word and reserves other fields at zero; integrated DMA config/wait behavior still needs testing. |
| CSRRCI | `Instructions.scala::CSRRCI` has funct3 `111` | Model class has funct3 `100` | Word and LLVM lowering use selected RTL bits; integrated CSR behavior remains unqualified. |
| Square/cube VPU encodings | `VSQUARE_BF16` uses funct7 `0x46`; `VCUBE_BF16` uses `0x47` | Model classes use `0x4e` and `0x4f` | Emitter uses selected RTL; numerical execution and dispatch still need qualification. |
| VPU BF16 pair alignment | `ScalarCore.scala` asserts even source and destination banks for VPU pair operations | Model helper checks only that pair base is below register 63 | Verifier requires even pair bases in VPU paths. The bounded VRELU stream checked both halves over two finite panels. |
| Selective VLI raw immediate | `VectorLoadImm.scala` uses `imm.asUInt`; ROW writes row zero of both pair halves, while COL/ONE write one physical register | `VLI_ROW/COL/ONE.exec` numerically assign integer `imm` into a BF16 tensor before a bit view | A [bounded selected-core check](vli-selective-observation.md) distinguished raw `0x4000` from numeric BF16(16384), and checked `0x8000`, all output bytes, preserved register, input, and guard for six streams. Full-domain and temporal behavior remain open. |
| VPU move width | `ScalarCore.scala` treats `VMOV` as a pair read/write | Model `VMOV.exec` reads and writes one BF16 register | Verifier treats it as a pair. A selected-core VRELU-to-VMOV mutation preserved both input halves for one finite panel; general semantics remain open. |
| VPU addition rounding | `AddSubSumVec.scala` widens BF16 operands to a 32-bit add and takes bits 31:16 of the 32-bit result | `VADD_BF16.exec` uses Torch BF16 pair addition and writes BF16 values | A bounded exact-FP32-sum panel on selected standalone core matched independent rational addition followed by bit chop. Directed `1 + 3/512` produced `0x3f80`, whereas BF16 nearest-even yields `0x3f81`. Model bit agreement was not measured and broader numerical behavior is unresolved. |
| VPU row-sum order | `SumRedu.scala` reduces adjacent lane pairs in an FP32 rounded tree and rounds once to BF16, then broadcasts per row | `VREDSUM_ROW_BF16.exec` uses Torch row-wise `sum(dim=1)` and BF16 conversion | A bounded selected-core run matched an independent five-level FP32 tree oracle across two full panels. Directed `2^25,1,-2^25,1,2` produced BF16 2; serial FP32 accumulation produces 3. Torch/model reduction order was not measured. |
| MXU arithmetic | MXU0 uses `PEArchitecture.CustomFMA` (`E4M3FMA`) while MXU1 uses an anchor-aligned integer reduction tree; they need separate contracts | Model `_vmatmul` uses FP16 matmul and converts to BF16 for both units | No model/RTL arithmetic equivalence is claimed. The 43-case exact-rational MXU0 check covers one cell and every finite normal E4M3 encoding at least once. A separate two-K MXU0 check compared all 1,024 cells for dense and mixed finite inputs against ordered BF16 steps. Bounded MXU1 tests compared all 1,024 cells against a round-once exact-rational reference, and a half-ULP tie distinguished MXU1 (`0x3f81`) from MXU0 (`0x3f80`) on the selected standalone core. |
| XLU transpose dispatch | `IDecode.scala` routes opcode `0x6b`/function zero to the separate XLU; `XLU.scala` reads a full 32-by-32 byte matrix before writing its transpose | `VTRPOSE_XLU` declares `EXU.VECTOR` and uses a contiguous 32-by-32 FP8 view transpose | The hand OOT path uses the RTL encoding and checks an independent byte-index permutation. The selected standalone core is exercised with distinct and identical source/destination registers; general temporal overlap remains unqualified. |
| Branch delay | `PcControl.scala` describes/implements one delay slot in the inspected RTL | `npu_spec/04_functional_units/README.md` requires two | Emitter requires one following non-redirecting instruction. A typed branch stream and changed-target mutation exhibited one-slot behavior on the rebuilt standalone `AtlasCore` ARC model; integrated SoC execution remains open. |
| CSR address aliases | `CSRFile.scala` maps only `0xC00..0xC03` and `0xC10..0xC11`; an unmapped scalar CSR address defaults to the cycle-counter index | The ISA names a scale-register CSR range, but the selected internal CSR file does not route it | Verifier allows only the six mapped addresses and zero-source reads of read-only status/illegal registers. Integrated CSR behavior still needs testing. |
| Scale interpretation | RTL VPU pack/unpack uses an E8M0 exponent shift | This local model revision contains a later E8M0 fix; earlier inspected model revision did not | `scale_reg` is explicit for pack/unpack and FP8 MXU pop; no quantization semantics are certified. |

The word emitter implements the selected RTL's **encoding choice** where these
sources disagree. That choice is not a proof that the instruction is legal in
the selected full core, that its timing is safe, or that the model operation
matches RTL numerically. Required follow-up tests must use selected RTL
execution and independent expected state/output checks.
