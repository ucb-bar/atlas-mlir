# Selected EE290 BF16 multiply timing

The selected provider supports `VMUL.BF16`: two BF16 MREG pairs produce one BF16 MREG pair. A separately selected compute receipt binds its row streams to the retained AtlasCore and the shared timing interface. Selection remains conditional and serialized; it does not qualify other VPU operations, overlap, aliasing or exceptional BF16 arithmetic. See the [selection guide](selected-evidence.md) for the options.

The selected artifact is `build/rtl-timing/ee290-portable-helper-check/ee290.hw.mlir`, SHA-256 `49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6`. The matching source inventory is `build/rtl-timing/ee290-source-inventory-001/inventory.json`; its `snapshots/` copies identify the Scala inputs below. These ignored build paths are local evidence identities, not committed fixtures.

| Inventory snapshot | Source | SHA-256 |
| --- | --- | --- |
| `007.scala` | `atlas/common/MregParams.scala` | `006b5190cbd5f457718d699222b1b304a6c5c78981706a6af4d3c8c123a1a93c` |
| `013.scala` | `atlas/mreg/MregFile.scala` | `4b9349ccfff146b5322180ab9e88e7a3adcc777a993beee8b863404ad3ddb79f` |
| `028.scala` | `atlas/scalar/IDecode.scala` | `b308974f2be31848fde180c9528f0e595df434232b70b6681b266f0c64c8cd31` |
| `033.scala` | `atlas/scalar/ScalarCore.scala` | `3ae39e67520d7d684d3aaa57c060ca4f7b191f30be8b783ae3b0f4d8f0aeca3a` |
| `036.scala` | `atlas/scalar/ScalarISA.scala` | `a67fd220305d1de18847461059c5d8a91c6ac296d10aa5df38c5759687de3ab8` |
| `039.scala` | `atlas/vector/VectorEngine.scala` | `356da155a2aa64aee5badb7496df43f80a46e21dcc017824e137cfc179f8270c` |
| `040.scala` | `atlas/vector/VectorEngineTop.scala` | `92a79cac21fc3d73f9f35faf234d904437e1675ab9b7a7d693a215e5ce37389a` |
| `041.scala` | `atlas/vector/VectorFSM.scala` | `8b199ab477a840067b70bf31f79022cc8f6ff0cb6235fecfd36a70c32254d598` |
| `052.scala` | arithmetic `MulRec.scala` | `210f0b7d17a9081a147a9cdfd41058c65298753fa7897ea84ce78e32826006a4` |
| `070.scala` | `AtlasCore.scala` | `cac182a402e37a27f688a1d714b238e7b4ad5189c0b19f3f3e21580e9e60c78b` |

## Command and bounded admission

The typed instruction is `"atlas.vpu_binary"(%state) {kind = "mul", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32} : (!atlas.state) -> !atlas.state`; its assembly is `VMUL.BF16 4, 0, 2`. `IDecode.scala:112` selects `VPU_MUL`. The external command selector is 3 (`ScalarISA.scala:114`); `VectorEngineTop.scala:74` subtracts one to obtain internal enum value 2. Issue admission uses external `issueBusy` bit 3.

Use a quiescent engine and even pair bases in `[0,62]`. The selected profile requires all three pair bases to be distinct modulo 32, for example lhs 0, rhs 2, destination 4. This rejects mirrored input, in-place output, and distinct architectural pairs that share physical banks; those require additional evidence before admission. No VPU operation may overlap this multiply. A conservative next VPU launch is age 66 or later.

ScalarCore issues the VPU command directly on `s1_fire` (`ScalarCore.scala:493,516-522`, `AtlasCore.scala:218-219`). Its actual stall expression contains delay and DMA wait only (`ScalarCore.scala:248-249`). MREG hazards and VPU busy are software-scheduling assertions at lines 285-291; they do not provide automatic command backpressure. An invalid schedule cannot be repaired by assuming the core waits for these resources.

The multiply datapath consumes MREG operands only. Although the common command carries `scaleE8M0` and an immediate, `VectorEngine.scala` connects multiply inputs to the two MREG vectors, sets rounding mode 0 and enables all 16 lanes. Multiply does not consume ERF scale or scalar-register values. The numerical replay uses signed finite-normal powers of two with normal results; exact expected multiplication requires no rounding approximation. Broader RNE tests can reuse the independent rational oracle in `test/test_vpu_mul_reference.py` after tying their execution to this selected artifact.

## Row stream and latency

An age is the edge on which a request is consumed, relative to the edge consuming the VPU command. Signals must be sampled before that edge. One architectural MREG contains 32 rows of 256 bits, or 512 BF16 values. A pair contains 64 rows and 1024 values.

For pair bases `lhs`, `rhs`, and `dst`, define `half = floor(k / 32)` and `row = k % 32`, for `k` in `[0,63]`:

| Event | Age | Architectural MREG | Row |
| --- | --- | --- | --- |
| Source request port 0 | `k` | `lhs + half` | `row` |
| Source request port 1 | `k` | `rhs + half` | `row` |
| Source responses | `k + 1` | corresponding source requests | corresponding row |
| Destination write port 0 | `k + 2` | `dst + half` | `row` |
| Destination write port 1 | none | none | none |

`VectorFSM.scala:260-280` uses the incoming command for early reads, initializes each read counter to 1, and emits the first row at launch age 0. The double-input path advances both counters from the primary response, stops requests after counter 63, and uses counter bit 5 to switch to pair+1 (`VectorFSM.scala:372-429`). Thus source first halves occupy ages 0-31 and second halves ages 32-63.

`MregFile.scala:247-261,294-305` implements a single registered SRAM response and routes it directly to the consumer; the VPU wrapper adds no ingress register. `MulRec.scala` specifies one intermediate stage and registers the valid and raw multiplication result. The selected HW independently contains one `seq.firreg` for `commonState_0_valid` in `@MulRec` and one stage of raw-product registers. `VectorEngine.scala` routes that response directly to FSM write data and valid, and `VectorEngineTop.scala:130-140` routes writes directly to MREG. Destination first halves therefore occupy ages 2-33 and second halves ages 34-65. These constants derive from the selected source and HW; their agreement with existing model row ages is not timing evidence by itself.

The FSM holds both source pairs active while `readDone` is false and both destination MREGs active while `writeDone` is false (`VectorFSM.scala:443-480`). With uninterrupted one-cycle source responses, source activity lasts through request age 63, and destination activity lasts through write age 65. The selected footprint sets `mregReads={lhs,lhs+1,rhs,rhs+1}`, `readRelease=64`, `mregWrites={dst,dst+1}`, `writeRelease=65`, and `doneAge=65`. It conservatively retains source operands through the final synchronous response, one edge beyond the observed last request. Both VLS paths and the VPU are reserved through age 65, so every next selected fixed-engine command starts at age 66 or later.

`canIssueOp` in the double state uses `doubleDone`, which includes the final output firing combinationally (`VectorFSM.scala:213-237`). Its busy bit can clear on the final write edge, age 65. The initial policy should use age 66 for the next VPU command, avoiding same-edge reuse. Replay must distinguish busy-bit deassertion from conservative policy release.

## Physical MREG conflicts and alias limits

There are 64 architectural MREGs and 32 physical banks. Physical bank is `mregId & 31`; physical row concatenates `mregId[5]` with the logical five-bit row. Thus m0 and m32 occupy different row regions of one physical bank. Each bank accepts at most one read and one write per cycle; multiple same-direction requests assert and have no retry/backpressure (`MregFile.scala:201-246`). Whole-register RAW/WAR/WAW tracking uses architectural MREG identities; the reservation table must additionally enforce physical bank ports.

Two different source pair bases equal modulo 32 collide on both read banks each cycle. `VectorEngineTop.scala:96-120` suppresses the second request only when both architectural bank and row match exactly, then broadcasts the delayed first response. This proves a mirror mechanism exists, but does not justify admitting mirror through the generic selected reservation policy. Initial distinct physical pair admission avoids that special sharing rule. Destination/source physical aliases and in-place updates also remain excluded from the first finite profile.

Independent engines may touch the same physical bank even when their architectural registers differ. The selected VMUL footprint includes every timed `Res::MReg` row access, so the existing physical port checker sees both halves of each pair and conflicts with other selected engines. Finite replay should include an adjacent safe producer/consumer boundary, not just a widely spaced standalone multiply.

## Required shared policy and replay contract

`TargetTiming::ConservativeRTL` rejects every VPU instruction by default. Its explicit `computePolicies` set may contain `ComputePolicy::VmulBf16` only after separate evidence loading; `allowsOperation` then requires both `Engine::Vpu` and mnemonic `vmul.bf16`. The provider must still enforce the bounded pair constraints above. All unary, pack/unpack, reduction, immediate, and other binary operations remain unsupported.

Whole-VPU serialization uses an explicit capacity-one `Unit::Vpu` hold for ages 0-65. The current `vpuLive` reservation uses the half-open range `[0,vpuLive)`, so a value of 65 would omit final age 65; a serialized value would be 66. More importantly, `ReservationTable::extendForWait` extends ordinary holds and physical port windows but does not extend its separate VPU slot map. The dedicated VPU hold uses the existing conservative unknown-wait handling and avoids relying on the model overlap table. VMUL also reserves both VLS paths, which XLU and VLS reserve as well. This serializes the admitted fixed engines and drains VMUL before a DMA launch. A matching wait is required before VMUL when DMA is pending.

The [selected-AtlasCore replay](../../tools/replay-ee290-compute.py) records external command valid/op/pair bases, `issueBusy` bit 3, both MREG read requests and responses, both MREG write ports, active pair masks, and retirement/publication boundaries. The retained instance is `vpu`; wrapper port names are `io_cmd_valid`, `io_cmd_bits_op`, `io_cmd_bits_vd`, `io_cmd_bits_vs1`, `io_cmd_bits_vs2`, `io_mregReadReq0/1_valid`, `io_mregReadReq0/1_bits_mregId`, `io_mregReadReq0/1_bits_row`, and corresponding `io_mregWriteReq0/1_*`. Generated VCD hierarchy must be resolved from the actual replay build, rather than assumed from source names.

The captured multiply fixture host-preloads four 1024-byte input tiles, VLOADs m0–m3, multiplies pairs 0 and 2 into pair 4, and VSTOREs m4/m5. It compares all 2048 output bytes and the remainder of an 8192-byte input/guard window with an independent exact reference. The checker decodes every issued word and binds both halves of both read ports, one-cycle responses, product writes, observed release, marker/halt and post-halt drain. Separate transpose and DMA/transpose cases use the same captured model. These finite, conservatively spaced executions support the loader; they do not themselves establish a compiler-produced schedule or full EE290 system behavior.

Select `ATLAS_RTL_VMUL_EVIDENCE_REPORT` alongside the VLS receipt to run `test/test_rtl_vmul_timing.py`. The suite exercises scheduling, delay insertion, final checks/export, pair bounds and aliases, age-65/66 reuse, DMA waits, LLVM reconstruction, and rehashed receipt/event mutations. Compiler tests and finite numerical traces remain distinct evidence layers. Other BF16 arithmetic domains, mirrored/in-place pairs, concurrent engines and MXU operations remain outside this profile.
