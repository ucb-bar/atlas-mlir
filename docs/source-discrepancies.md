# Source discrepancies and qualification obligations

Source observations for RTL `0079c0541111197741a231c002e3843fa6f545b2`
and local model `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`. These
are static checks, not evidence of integrated execution.

| Subject | Selected RTL | Examined model | Handling here |
| --- | --- | --- | --- |
| VR operand fields | `ScalarDecoder.scala` uses `vd[12:7]`, `vs1[18:13]`, `vs2[24:19]`, six bits each | `npu_model/isa.py::VRType.to_bytecode` masks `vs1` to seven bits and `vs2` to five bits | Emitter uses RTL layout; test exercises a nonzero secondary operand. Model assembly encoding cannot be the sole oracle. |
| DMA config | `Instructions.scala::DMA_CONFIG_ANY` fixes funct7 to zero; `DMA_WAIT_ANY` fixes it to one | `DMA_CONFIG_CH0..7` set funct7 to one | Emitter uses RTL config word and reserves other fields at zero; integrated DMA config/wait behavior still needs testing. |
| CSRRCI | `Instructions.scala::CSRRCI` has funct3 `111` | Model class has funct3 `100` | Scalar/CSR lowering is outstanding. |
| Square/cube VPU encodings | `VSQUARE_BF16` uses funct7 `0x46`; `VCUBE_BF16` uses `0x47` | Model classes use `0x4e` and `0x4f` | Emitter uses selected RTL; numerical execution and dispatch still need qualification. |
| VPU BF16 pair alignment | `ScalarCore.scala` asserts even source and destination banks for VPU pair operations | Model helper checks only that pair base is below register 63 | Verifier conservatively requires even pair bases in VPU paths. |
| VPU move width | `ScalarCore.scala` treats `VMOV` as a pair read/write | Model `VMOV.exec` reads and writes one BF16 register | Verifier treats it as a pair; semantics unresolved. |
| MXU arithmetic | RTL unit implementations must be checked independently | Model `_vmatmul` uses FP16 matmul and converts to BF16 | No arithmetic equivalence is claimed. An ordered BF16 reduction is a separate numerical policy. |
| Branch delay | `PcControl.scala` describes/implements one delay slot in the inspected RTL | `npu_spec/04_functional_units/README.md` requires two | Emitter requires one following non-redirecting instruction; integrated side-effect microtest is still needed. |
| CSR address aliases | `CSRFile.scala` maps only `0xC00..0xC03` and `0xC10..0xC11`; an unmapped scalar CSR address defaults to the cycle-counter index | The ISA names a scale-register CSR range, but the selected internal CSR file does not route it | Verifier allows only the six mapped addresses and zero-source reads of read-only status/illegal registers. Integrated CSR behavior still needs testing. |
| Scale interpretation | RTL VPU pack/unpack uses an E8M0 exponent shift | This local model revision contains a later E8M0 fix; earlier inspected model revision did not | `scale_reg` is explicit for pack/unpack and FP8 MXU pop; no quantization semantics are certified. |

The word emitter implements the selected RTL's **encoding choice** where these
sources disagree. That choice is not a proof that the instruction is legal in
the selected full core, that its timing is safe, or that the model operation
matches RTL numerically. Required follow-up tests must use selected RTL
execution and independent expected state/output checks.
