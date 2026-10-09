# XLU transpose capture and timing

The selected XLU reads a complete 32×32 byte matrix into an internal buffer, then writes its transpose to one matrix register. Its command captures only while idle; busy launches are ignored without retry. This reference derives a conditional rule from selected CIRCT and source snapshots, corroborated by the component replay below. It does not qualify another memory implementation or full EE290 execution. The [extraction walkthrough](extraction-walkthrough.md) explains the common method, and [admission](admission.md) covers frontend and other-engine obligations.

## Selected evidence

All HW locators refer to `build/rtl-timing/ee290-portable-helper-check/ee290.hw.mlir`, SHA-256 `49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6`, selected by its manifest from FIRRTL SHA-256 `5c6eb7ce8d0005354eb9a2d3f0c0d52bc105c6261ff25b7081825428b93780fa`. The relevant definitions begin at HW 259509 (`ScalarDecoder`), 260678 (`ScalarCore`), 305832 (`XluEngine`), 310194 (`MregFile`), and 316694 (`AtlasCore`). Selection must bind these definitions, the SRAM implementation and the source/build links; module names and locators alone are insufficient.

Source locators use `build/rtl-timing/ee290-source-inventory-001/snapshots/`, selected by inventory SHA-256 `939fee9cb515f3f1b62a8c6d6743a312284856efd9e8a51140f83c24b7782a23`. They are captured current Scala sources, with module-local equations independently checked against the retained HW. They do not reconstruct the original Chisel execution edge.

| Snapshot | Source | SHA-256 |
| --- | --- | --- |
| `065.scala` | `atlas/xlu/XLU.scala` | `9015a665d5bdd11fb94766e32353b4b394fc0f561e1c24415c13003b0277a03a` |
| `013.scala` | `atlas/mreg/MregFile.scala` | `4b9349ccfff146b5322180ab9e88e7a3adcc777a993beee8b863404ad3ddb79f` |
| `007.scala` | `atlas/common/MregParams.scala` | `006b5190cbd5f457718d699222b1b304a6c5c78981706a6af4d3c8c123a1a93c` |
| `029.scala` | `atlas/scalar/Instructions.scala` | `83dd5f1467a0293aff0d2c47ed52f98165660558a1ebe0add4b354d2a4c28265` |
| `035.scala` | `atlas/scalar/ScalarDecoder.scala` | `a4feaa3ffa60234ded981fd99a0cc168fa58257d5e5989f476c3b75ffe77630e` |

## Instruction, capture and storage mapping

The one catalog operation is `vtrpose.xlu dst, src`. Selected instruction pattern is `0000000_??????_??????_??????_1101011` (snapshot 029:158): top seven bits zero, low seven bits opcode `0x6b`. Decoder VR fields are destination `[12:7]`, source `[18:13]`, and unused second source `[24:19]` (snapshot 035:76–80). ScalarDecoder selects the XLU mode at HW 260084/260234–260235 and emits it at 260295. Initial compiler support should emit canonical zero for the unused second-source field and reject other modes rather than generalizing from ignored engine inputs.

Scalar launch is `s1_fire && decoded_xlu_cmd` (HW 261219), with `s1_fire` requiring valid stage 1, no delay/DMA-wait stall and no halt (HW 260778–260781). The root directly wires source/destination IDs and valid to XLU at HW 316714. There are no scalar-register pointer operands, byte addresses, scale operands, queue slots or variable-size modes. Both matrix IDs are six bits, covering 0–63.

`XluEngine` captures both IDs on `state==Idle && cmd.valid` (HW 306905–306908; active-command registers at 305874–305875), resets request/response counts and enters Read (snapshot 065:137–143). Command valid while Read or Write does not capture or restart the operation. The complete selected XLU has no assertion operation or ready output. Its unused source-level busy output is eliminated from the retained interface because ScalarCore does not consume it; compiler engine occupancy must therefore prevent another launch even when the new IDs name unrelated free banks.

Each MREG is 32 rows of 256 bits, representing 32 byte elements per row. Physical SRAM bank is `id & 31`, and physical row is `{id[5], logicalRow[4:0]}` (snapshot 007:36–40/72–78; XLU read mapping at HW 310281/310296–310297). IDs `m` and `m+32` share a bank but occupy separate halves. Physical bank conflicts require checks even when logical IDs differ.

In Read, the request counter emits source rows 0–31 once each, without a ready/grant predicate (HW 306912–306930; snapshot 065:147–153). Every valid response places its 32 bytes into the next buffered row, and the 32nd response changes state to Write (HW 306931–306938 and 310097–310104; snapshot 065:155–166). In Write, output row `j` takes byte `i` from `buf[i][j]`: HW 310116–310180 selects a common column index from all 32 buffered rows and concatenates them with element zero in the low byte. Destination rows 0–31 write consecutively (HW 310105–310115/310184–310191; snapshot 065:173–189).

Source and destination equality does not select a different pipeline. All input rows are buffered before any output write, so an isolated in-place transpose is structurally supported. Distinct IDs sharing one physical bank also use separate read and write phases. Neither fact establishes safe overlap with another engine or same-address read/write behavior.

## Conditional ages and provider mapping

Use the accepted idle command's capture edge as age zero, with subsequent request/write events sampled before their clock edge. The selected MregFile registers per-bank read-valid and client tags for one cycle (HW 314165 onward; snapshot 013:264–267), routes client index seven to XLU (HW 315672–315675), and directly outputs the resulting valid/data at HW 316604. Its source declares SyncReadMem and preserves one SRAM response stage (snapshot 013:70–72/253–267/299 onward). The SRAM body is external at the retained boundary; selected behavioral-memory correspondence and actual event observation must substantiate the assumed data latency.

Under one-cycle MREG read response, initialized memory and uncontended ports, the equations yield:

| Event | Ages, inclusive | Proof |
| --- | --- | --- |
| Source row requests 0–31 | 1–32 | Capture enters Read; counter starts at zero and emits while `<32`. |
| Source responses and buffer capture | 2–33 | One-cycle selected MREG response; response counter ends at 31. |
| Destination row writes 0–31 | 34–65 | Last response enters Write; write index starts zero and advances every Write cycle. |
| Active source-read report | 1–33 | `state==Read`, HW 306904. |
| Active destination-write report | 1–65 | `state!=Idle`, HW 306903. |
| Idle / next accepted transpose | 66 | Final write index 31 selects Idle, HW 310187–310191. |

The selected provider expresses source MREG rows as `first=32*src, count=32, age=1, step=1` and destination rows as `first=32*dst, count=32, age=34, step=1`, with `mregReads={src}`, `mregWrites={dst}`, `readRelease=33`, `writeRelease=65`, and an inclusive XLU hold `0..65`. Physical MREG read-port requests occupy ages 1–32 and write-port requests 34–65. Conservative register holds begin at capture rather than waiting for the active-report rise. These rules happen to agree with the inherited compiler table, but are derived here from selected state/register equations; agreement is not execution qualification.

If the response latency changes to `L>=1` while one ordered response arrives per cycle, writes instead begin at `33+L` and last at `64+L`. Missing or losing responses can postpone Write indefinitely because requests are not retried. The fixed ages therefore require the memory and arbitration assumptions; they must not be applied to an unknown response implementation.

## Small useful domain and interactions

The smallest complete workload is serialized VLOAD → transpose → VSTORE, with legal six-bit IDs, full byte-matrix layout, initialized source data, quiescent entry and no other MREG engine active during transpose. Admit out-of-place, in-place and the `m`/`m+32` physical-bank case under this same serialization. Reject an XLU launch before age 66 even with unrelated banks. Require preceding producer completion and following consumers/markers/halt after the applicable release event. Keep reset/host restart outside the running interval because START does not drain engines.

For a first mixed domain, reserve both VLS paths through the XLU hold, and require their prior holds to end before XLU capture. This deliberately serializes otherwise independent VLS traffic and avoids importing an unreviewed cross-engine arbitration policy. Later overlap can use the exact row streams and bank capacities: MregFile has fixed port priority and asserts at most one reader and writer per physical bank (snapshot 013:189–217); XLU has no grant/retry to recover a lost request. Shared active-bank reports are logical-register reports and do not replace physical-bank checks.

DMA uses VMEM rather than MREG, so transpose has no direct DMA port conflict. Nevertheless a composed load-to-VMEM → VLOAD → transpose → VSTORE → store-from-VMEM must retain the [matching DMA wait](dma-timing.md) before either VLS access or transfer reuse. The initial domain can prohibit transpose while DMA is pending to keep one serialized memory/engine envelope. Relaxing that prohibition requires its own explicit supported predicate and mixed execution evidence; XLU completion never releases DMA ranges.

Useful discriminating witnesses should check numerical byte permutation and preserved guards for distinct IDs, in-place, IDs separated by 32 and first/last IDs; observe all request/response/write rows and first-free age; reject a second transpose at age 65 and accept one at 66; and reject a busy launch on unrelated free IDs. VLOAD/transpose and transpose/VSTORE boundaries need observed producer/consumer events. Module-local simulation success is separate from integrated-system qualification, and assertions must remain enabled for arbitration cases.

## Component replay and reproduction

The [checker](../../tools/check-rtl-xlu.py) slices the complete selected XluEngine from retained CIRCT, preserving the module, macro fragments and any assertions while removing only location-alias references. The selected XLU contains no assertions; none are invented to protect busy launches. CIRCT verifies and exports the slice, and assertion-enabled Verilator compiles it with the [independent responder/transpose harness](../../test/rtl-xlu-replay.cpp). Tools, executable, raw/exported hardware, producer sources, reviewed Scala copies, commands and event logs are hash-bound in the receipt. A new output directory is required; prior evidence is not overwritten.

The final local receipt `build/rtl-timing/xlu-check-002/report.json`, schema `atlas.conditional_xlu_hw_check.v0`, SHA-256 `52c1fc88024aaf8f0d312102189473434a90a8512ac58777f94e983e25d82898`, reports `conditional_checks_passed` for all nine cases. Its raw XLU slice SHA-256 is `9a0ed59aaeba553d4ba7e984bfa5c170a18f62c229180d5ce79f9ae29b31d60a`. Distinct, in-place, paired-bank and both endpoint cases pass full byte permutation and whole-memory guard checks. Busy launches on unrelated IDs at ages 1 and 65 produce only the original transfer; a launch at 66 captures and executes a second transfer. Inactive command operands change immediately after launch without retargeting captured IDs. All normal cases show read ages 1–32, response ages 2–33, write ages 34–65 and first-free age 66. The two-cycle-response case instead shows write ages 35–66 and first-free age 67, corroborating the response-latency applicability condition.

The [seven focused tests](../../test/test_rtl_xlu_evidence.py) pass rejection checks for mutated register/row/age streams, missing responses and cycles, premature releases, incorrect numerical/count summaries, busy decode mistaken for capture, dropped assertion operations during slicing, mismatched HW identities and unreviewed/missing source identities. These synthetic tests check the evidence validator; the nine cases above execute the actual selected XLU slice.

From the compiler checkout, use explicit selected inputs/tools:

```bash
python3 tools/check-rtl-xlu.py \
  --manifest build/rtl-timing/ee290-portable-helper-check/manifest.json \
  --source-inventory build/rtl-timing/ee290-source-inventory-001/inventory.json \
  --output build/rtl-timing/xlu-check-new \
  --circt-opt /path/to/circt-opt \
  --verilator /path/to/verilator \
  --verilator-root /path/to/share/verilator \
  --cxx /path/to/c++ --make /path/to/make --ar /path/to/ar
python3 test/test_rtl_xlu_evidence.py
```

The receipt retains `scheduling_qualified=false` and `integrated_target_execution_qualified=false`. It executes XLU with injected ordered MREG responses; MregFile arbitration and SRAM data latency are not executed. Physical-bank cases validate the modeled address mapping and XLU's nonoverlapping phases, not general integrated port safety. This component validation does not execute the full EE290 system. VLS/XLU and DMA composition require observed integrated boundaries before broadening qualification.

The implemented [selected-evidence binding](selected-evidence.md) accepts optional `xlu-evidence` and `xlu-evidence-sha256` alongside the selected VLS receipt. `loadXLUEvidence` verifies matching hardware/manifest identities, the complete selected slice, reviewed source and harness copies, tools/commands and all nine observed streams before enabling `atlas.vls_xlu.serialized.v1` or the combined `atlas.vls_dma_xlu.serialized.v1` resolver. Both VLS paths and XLU remain reserved through age 65, and pending DMA requires its matching WAIT before transpose. The final verifier rechecks the rewritten instruction stream at emission and LLVM handoff boundaries.

The [seven compiler integration tests](../../test/test_rtl_xlu_timing.py) passed against the local `compiler-dma` build with explicit VLS007, XLU002 and DMA006 receipts. They exercise both scheduling consumers, register admission, the age-65/66 boundary, VLS and DMA serialization, final emission after delay removal, and rejection of rehashed claims, command recipes and mutated observed logs. Compiler test success preserves the receipt's conditional execution scope.

## Next useful families

Existing workload sources make two scoped follow-ons useful. A serialized `VMUL.BF16` tile is the smallest VPU path exercised by `smolvla_parameterized_elementwise_mul.S:88`; it needs BF16 pair geometry, both read ports, destination writes and candidate-specific issue release. For matrix workloads, audit MXU0 weight push, overwrite matmul and BF16 pop as a complete isolated tile, then add accumulator reuse and `matmul.acc` for the two-K chain. `smolvla_matmul_k_chain_mxu0.S:48/52/54/74/78/80/83` specifically transposes B, pushes weights, overwrites then accumulates and pops BF16. MXU1 requires a separate sequencer proof rather than inheriting MXU0 admission.

After those isolated tiles, RMS normalization supplies a concrete VPU expansion: square, row sum, multiply, add, sqrt and reciprocal (`smolvla_parameterized_rms_norm.S:66–78`). Mixed overlap, generic loops and fused attention should follow established primitive storage/release contracts. These are workload-based audit priorities, not enabled timing rules or a Merlin implementation plan.
