# Command admission and capture

This reference records static review of a selected retained HW/Comb/Seq artifact for the logical `EE290SimConfig` target. It supplies no enabled timing rules, resolver bindings, measured instruction spacing, or execution qualification. An actual evidence bundle must bind the exact hardware bytes, source/build identity, configuration and lowering provenance, together with the operation locators used below. The logical target name does not establish equivalence with a public elaboration. Follow the [retention](retained-hw.md) and [evidence indexing](hw-evidence.md) requirements before applying these observations to another artifact.

Instruction retirement, command-valid assertion, command capture, first/last storage access, engine release and observed completion are distinct events. The reviewed engine command interfaces are Valid-only: a blocked or ignored command has no ready/valid retry handshake with the scalar frontend. A predicate sufficient for metadata capture is not sufficient for safe execution. Assertions express software obligations and do not provide backpressure; selected non-synthesis assertions may be absent from synthesized hardware. The descriptions below assume reset is deasserted and normal execution; host restart and reset require separate analysis.

## Coverage and frontend

The [exact mnemonic catalog](operation-catalog.json) and [operation coverage table](operation-coverage.md) remain the denominator: 127 mnemonic variants and 32 classes. This partition covers each catalog variant once, including SELI under scalar/control and SELD under scalar memory. It does not admit arbitrary encoded words, reserved fields, operands or aliases.

| Domain | Catalog classes | Exact mnemonic count |
| --- | --- | ---: |
| Scalar/control | Alu, Csr, Branch, Jump, Delay, Halt, Fence, ScaleImm | 40 |
| Scalar memory | ScalarLoad, ScalarStore, ScaleLoad | 9 |
| Vector LSU | VLoad, VStore | 2 |
| MXU0 | WeightPush, AccPushFp8, AccPushBf16, PopFp8, PopBf16, MatMul, MatMulAcc, each with `.mxu0` | 7 |
| MXU1 | The same seven classes, each with `.mxu1` | 7 |
| VPU | VpuElementwise, VpuPack, VpuUnpack, VpuRowReduce, VpuColReduce, VpuLoadImmPair, VpuLoadImmSingle | 29 |
| XLU | Transpose | 1 |
| DMA | DmaLoad, DmaStore, DmaConfig, DmaWait, each expanded over `.ch0` through `.ch7` | 32 |

In `ScalarCore`, `s1_fire = pc_ctrl.io_s1_valid && !stall && !halt_now`. The selected `stall` cone contains only `delayCounter != 0` and `dma_wait_stall`; the latter is `s1_valid && decoded DMA WAIT && io_dma_busy[selectedChannel]`. `halt_now` combines registered halt, host stop, illegal instruction or redirect in a delay slot, ECALL and EBREAK. These conditions exactly describe scalar fire in the reviewed artifact, not safe engine acceptance.

`AtlasCore.scalar` receives VPU `issueBusy`, LSU `scalarBusy/vloadBusy/vstoreBusy` and `MregBankTracker.readBusy/writeBusy`. Those signals and scalar `memLoadPending` feed assertions rather than the stall cone. Scalar checks cover MREG RAW/WAR/WAW, VPU issue availability and pair parity, vector path availability, scalar load pending/busy and response/writeback collisions. There is no scalar command-ready input from either MXU or XLU. Shared active-bank tracking does not describe all engine occupancy or internal slot hazards.

## Capture, obligations and stalls

Module and signal names below are evidence selectors. Their exact operation locations and identities belong in the bound bundle; names alone are insufficient evidence for an enabled compiler rule.

| Domain | Actual launch/capture in the reviewed command path | Required software obligations and unresolved safety |
| --- | --- | --- |
| Scalar/control | `ScalarCore.s1WritesScalarRd = s1_fire && rd_wen && rd!=0`; `io_csrPort_valid = s1_fire && is_csr`; SELI writes on `s1_fire && is_seli`. DELAY loads the low 12 immediate bits into `delayCounter`. Redirect is gated by `!stall && !halt_now`. Terminal/illegal sidebands follow valid decode and registered halt rather than `s1_fire`. FENCE supplies no engine-drain command in this path. | Scalar/scale load responses must avoid competing scalar or SELI writes. DELAY and matching DMA WAIT stall ordinary retirement; terminal sidebands can act while that retirement is stalled. Halt acceptance does not wait for pending engines or memory. CSR address semantics, PC/control-flow effects and a complete halt/restart protocol need separate evidence. |
| Scalar memory | `ScalarCore.issueScalarLoad/issueScalarStore` latch command operands on `s1_fire`; registered `scalarMemCmdValid` subsequently drives `LSU.io_scalarCmd_valid`. LSU load capture is valid and not-store; store request is valid and store. There is no command-ready gate. | Scalar asserts that a new load sees neither `memLoadPending` nor LSU `scalarBusy`; the latter reflects LSU `scalarLoadPending`, not every frontend/response stage. Stores still require bank compatibility. Capacity, effective address/masks, writeback availability and visibility must be resolved independently. Free pending state is insufficient for safe shared memory access. |
| Vector LSU | `LSU.issueVloadCmd/issueVstoreCmd` are valid plus selected opcode. Metadata capture additionally requires the respective `vloadState/vstoreState` to be idle. A command presented while that state is occupied is ignored by capture. | `vloadBusy` includes nonidle state, response pending and write pending; `vstoreBusy` includes nonidle state and both response pipeline stages. Thus idle alone is insufficient during a tail. Scalar and LSU assert full path freedom, 1 KiB base alignment and a valid range within one VMEM bank. MREG and shared-bank obligations remain necessary. |
| MXU0 | `AtlasCore.mxu0` selects `SystolicArrayTop` and `SystolicArraySequencer`. Its `acceptCompute/acceptPushP0/acceptPushP1/acceptBF16Push/acceptPopFP8/acceptPopBF16` actually gate dispatch and port metadata capture; predicates are detailed below. | Rejected valid commands cause sequencer assertions and are not retried by scalar. Acceptance does not include every MREG, slot and downstream storage obligation. The accumulator row-0 condition is distinct from full-result completion. |
| MXU1 | `AtlasCore.mxu1` selects `InnerProductTreesTop` and `InnerProductTreesSequencer`. It has corresponding dispatch predicates, with different FIFO capacity and selected-slot hazards from MXU0. | Rejected commands assert and are not retried. MXU0 predicates cannot be substituted. Complete storage conflicts, alias legality, result visibility and release still require evidence. |
| VPU | `VectorEngineTop.io_cmd_valid` becomes `VectorEngine.io_inst_valid`, then raw `VectorFSM.io_in_instFire`. It is not ANDed with `VEReady`. State-dependent `newInputToSINGLE/newInputToDOUBLE` and completed slots determine metadata capture. | Scalar and top assert `!issueBusy[encodedOp]`; core asserts `VEReady`. These are software obligations, not a handshake. Invalid issue can overwrite sequencer state or fail to allocate a slot. Pair parity, mirrored-read row equality, simultaneous dual responses, FP8 unpack queue capacity and MREG hazards must also hold. |
| XLU | `XluEngine` latches source/destination metadata only on `state==Idle && io_cmd_valid`. Its read/write states continue the existing command while ignoring a new command. | The complete reviewed XluEngine definition has no assertion operation or command-ready output, and ScalarCore has no direct XLU occupancy check. With unrelated available banks, MREG checks can permit scalar retirement while XLU silently ignores the command. Engine idle is necessary for capture but insufficient for storage/alias safety; safe reuse needs an independently established release event. |
| DMA | Scalar LOAD/STORE launch asserts `io_dmaCmd_valid`; CONFIG updates global `dmaBaseReg` without enqueueing; WAIT sends no DMA command. `DmaEngine` captures valid into `commandQueue[enqueueIdx]` and advances the eight-slot index modulo eight, independently of slot/channel occupancy. | The reviewed enqueue path has no command-ready, queue-full or occupied-slot gate. Range assertions do not establish ring safety. `channelBusy` is one bit per channel, not a generation count; channel freedom alone does not establish enqueue-slot freedom. Matching WAIT stalls scalar until its integrated busy bit clears. Ring wrap, channel generations and dynamic memory completion require explicit obligations or conservative rejection. |

LSU and VMEM independently assert pairwise same-bank conflicts among scalar load/store and vector load/store requests. Their path-free conditions do not imply safe simultaneous bank use. DMA has actual downstream VMEM grants and TileLink ready/valid flow control; these regulate progress after command capture and do not turn its launch into a backpressured command interface. Unknown addresses, aliases and external-memory effects remain conservative.

## MXU dispatch predicates

Let `P0/P1/W0/W1` denote each port's boundary: `!portCmdValid || (portRow+1)>=rowLimit`. Boundary includes the final useful cycle, not just idle. Let `A` mean no valid in-flight entry for the selected accumulator has `rowsWritten==0`. The reviewed SA tracker has three metadata entries and the IPT tracker two; these declarations do not establish a global instruction capacity. Every acceptance expression also requires incoming command valid and the corresponding operation class.

| Operation | MXU0: SystolicArraySequencer | MXU1: InnerProductTreesSequencer |
| --- | --- | --- |
| Matmul or matmul.acc | `P0 && !saTrackerFull && A` | `P0 && !fifoFull && A && !pushWActive[weightSlot] && !pushAccActive[accSel]` |
| Weight push | Available read route and `wbufReady[weightSlot]`, where the selected countdown equals zero | Available read route and `!wslotComputeReading[weightSlot] && !pushWActive[weightSlot]` |
| FP8 accumulator push | Available read route and `A` | Available read route and `A && !pushAccActive[accSel]` |
| BF16 accumulator push | `P0 && P1 && A` | `P0 && P1 && A` |
| FP8 accumulator pop | `W0 && A` | `W0 && A` |
| BF16 accumulator pop | `W0 && W1 && A` | `W0 && W1 && A` |

A single-port push prefers P1; it falls back to P0 only when P1 is unavailable. The available-route predicate is therefore `P1 || (!P1 && P0)`, with mutually exclusive routing, not simultaneous reservation of both ports. IPT's weight-feed and active-weight-push hazards exclude the final useful row; its active accumulator-push hazard includes active FP8 pushes on either read port and BF16 push on P0. The selected SA compute predicate lacks those IPT-style active-push gates; that absence does not prove arbitrary overlap safe.

Both sequencers independently assert conflicts between incoming pop destinations and active read-port banks, or incoming push sources and active pop write-port banks. Those checks do not enter the dispatch predicates above. Passing dispatch is necessary and sufficient for the corresponding metadata latch under the reviewed state, while these additional assertions and complete storage constraints remain separate requirements. Neither row-0 readiness nor a port boundary proves last write visibility or whole-engine completion.

## VPU availability and DMA completion

`VectorFSM.VEReady` evaluates candidate-specific `canIssueOp`. Idle permits issue. In single state, both completed slots permit issue; one completed slot permits only a single-port operation that shares no logic with the live slot. With neither completed it rejects issue. Double state requires `doubleDone`, which requires both write paths for row reduction and the primary write path for binary operations. Slot completion includes a final output firing at its write limit, allowing a same-cycle boundary. This is an admission expression, not a fixed spacing measurement.

VPU shared logic groups include identical operation, add/sub/row-sum, exp/exp2, sin/cos, square/cube, pair-max/column-max, pair-min/column-min and all VLI forms. Binary operations and row reductions require both read ports. Readiness varies with the candidate and live operation; it does not imply pair parity, bank/row compatibility or response/queue safety. Because raw valid drives `instFire` and next-state selection, the VEReady assertion cannot be treated as hardware protection against an unsafe launch.

`AtlasCore` feeds scalar DMA busy as `DmaEngine.channelBusy[channel] || dmaLaunched[channel]`, retaining the launch/busy observation boundary. A wait at decode while busy is attempted issue; successful `s1_fire` follows the busy bit clearing and the other frontend conditions permitting retirement. This does not supply a fixed external-memory completion bound. DMA completion uses captured slot metadata, dispatched state, outstanding replies and accepted responses. Unconditionally writing an occupied enqueue slot can replace metadata needed by that protocol; a Boolean channel indication cannot distinguish multiple pending generations. Globally serialized launches with matching waits are a candidate conservative scope to validate, not an enabled or generally sufficient DMA safety rule.

## Remaining qualification

Bounded tests must observe frontend retirement and engine capture separately, with assertions enabled, exact identities and operand domains recorded. Useful discriminating cases include vector idle versus response-tail launch; each MXU boundary versus selected-slot/FIFO hazards; VPU same-logic versus distinct-logic overlap; busy XLU with unrelated banks; and DMA ring wrap or repeated same-channel launches under delayed downstream progress. Visibility and final results need independent observers. Such tests do not replace catalog/mode coverage, full alias and reserved-field analysis, symbolic completion handling, or extraction of first/last access and release events.
