# RTL timing and resource contract for EE290SimConfig

**Draft v0; the broad contract has no compiler loader or qualified target profile.** This document specifies the handoff from RTL analysis to Atlas scheduling, delay insertion, allocation constraints, and final hazard verification. The [operation coverage table](operation-coverage.md), [machine mnemonic catalog](operation-catalog.json), [JSON Schema](contract.schema.json), and [draft bundle](examples/contract.json) accompany it. The draft covers the operation surface while leaving unavailable evidence explicit; schema validity is not timing qualification.

The [admission audit](admission.md) distinguishes command presentation, capture, assertions and stalls across all engines. The [bounded VLOAD/VSTORE analysis](vls-timing.md) maps a selected LSU slice into the existing timing API and provides a conditional hardware replay and compiler comparison.

The implemented [selected-evidence workflow](selected-evidence.md) separately loads `atlas.conditional_vls_hw_check.v0` replay receipts through `atlas.vls.conservative.v1`. Explicitly selected conditional rules now feed scheduling, delay insertion and an independent final timed verifier for a small straight-line subset. This backend rejects the broad draft bundle and unsupported instances; it does not qualify integrated EE290SimConfig execution or the full operation catalog.

## Scope and ownership

The contract target is named **`EE290SimConfig`**, referring to [bringup-chipyard's system configuration](https://github.com/ucb-ee194-tapeout/bringup-chipyard/blob/main/generators/chipyard/src/main/scala/config/AtlasConfigs.scala#L12-L25). `target.config` and `target.reference` identify that public configuration, independent of local checkout or input filenames. The helper receives explicit paths and never translates a hidden local configuration name. A run records its actual producer label, if supplied, in provenance and `build.config`; local input mappings and measured bundles stay in ignored artifacts. A shared logical target does not establish equality of sources, parameters or generated hardware.

Atlas-mlir owns the compiler-facing contract, versioned interpretation of its rules, and resolved footprint API. Merlin's Atlas adapter consumes the exported facts and compiler checks for candidate evaluation; it must preserve their coverage and identities. The permanent home of the extraction implementation remains to be coordinated with Agustin. Reuse the historical profiles and evidence where applicable, with explicit conversion from their schemas and hardware identities.

This contract concerns cycle-level instruction behavior and resource availability. Functional numerical semantics, encoding, and runtime launch protocols retain their own specifications, with references and compatibility requirements here. A timing profile cannot certify numerical correctness, a callable host ABI, physical propagation delays, or a maximum chip frequency.

## Hardware input versus simulation output

The intended extraction path is:

```text
selected Chisel/configuration
  -> FIRRTL + annotations
  -> retained CIRCT HW/Comb/Seq MLIR
  -> hardware graph / RTL analysis and evidence
  -> target contract + operand-resolved footprints
  -> atlas-mlir scheduling, delay insertion, and verification
```

The VCS Makefile instead retains generated SystemVerilog and builds a simulator after firtool lowering. It does not currently save the required HW/Comb/Seq MLIR snapshot. The separate [retention helper](retained-hw.md) produces that snapshot from explicit input paths and records its lowering command, tool identity and verification result, preserving the sequential boundaries needed by extraction. FIRRTL, lowered hardware MLIR, generated SV, and simulator bytes have different identities; none of their hashes substitutes for another. Existing adopted historical HW IR has incomplete lowering provenance.

The simulator is a separate validation tool: it can exercise emitted instructions and check results, guards, and timing observations as required by [issue #11](https://github.com/ucb-bar/atlas-mlir/issues/11). A simulator build alone establishes neither extraction nor execution coverage. Contract design and source audit can proceed before hardware-MLIR extraction or simulation is ready.

## Inventory and completeness

At compiler revision `7b1e3dab55c2d0bd05d51c8f6ec48e55bcedeb58`, [AtlasTiming.cpp](../../lib/AtlasTiming.cpp) has **127 exact mnemonic variants, 32 operation classes, and seven engines**. The [selected decoder census](../selected-variant-census.md) has **99 decoder modes**: the 28-count difference comes from expanding four DMA families over eight channels. These are different denominators, not evidence of extra qualified operations. The dialect also has structural and virtual operations that must be mapped through lowering rather than assigned arbitrary physical instruction latencies.

Every catalog mnemonic must occur in exactly one rule group. Class grouping is a compact representation: each mnemonic and each parameterized mode must still resolve independently. A new mnemonic, removed decoder mode, changed field interpretation, or changed lowering must fail the catalog compatibility check until the contract is updated. No catch-all engine latency or implicit empty footprint is permitted.

Coverage must include these dimensions, even when their status is unknown:

| Family | Required distinctions |
| --- | --- |
| Scalar ALU and upper/PC-relative operations | Register versus immediate sources; x0; signed widths; scalar writeback; PC units and implicit PC reads |
| CSR and scale immediates | CSR address and read/modify/write mode, immediate-versus-register source, volatile counters/host-visible state, exponent-register write timing |
| Scalar and scale memory | LB/LBU/LH/LHU/LW, SB/SH/SW, SELD; byte enables, sign extension dependencies, address masking/alignment, load writeback and VMEM-bank arbitration |
| Vector memory | VLOAD/VSTORE; effective base/immediate units, line and tile geometry, row streams, register release, bank crossings and simultaneous scalar/DMA traffic |
| MXU0 and MXU1 | Weight and accumulator push; FP8/BF16 push/pop; overwrite versus accumulate matmul; exponent operands; unit/slot ownership, port alternatives, accumulator hazards and in-flight capacities |
| VPU | Each unary/binary arithmetic function; pack/unpack; row versus column sum/min/max; VLI all/row/column/one; paired registers, selective writes, repeated passes, scale inputs, shared logic groups and asymmetric overlap |
| XLU | Transpose source/destination mapping, aliasing, read/write streams, issue admission and engine release |
| DMA | Load/store/config/wait on every channel; captured effective operands/base, memory ranges, queue-slot reuse, channel generations, dynamic completion and backpressure |
| Frontend/control | DELAY encoding and issue duration, branch/JAL/JALR targets and delay slots, path-dependent stalls, FENCE semantics, ECALL/EBREAK and pending-work drain |
| Structural/virtual lowering | State tokens, block arguments and CFG, physical allocation/aliasing, compiler annotations, inserted copies/transfers/delays/waits, and final emitted instruction words |

The inventory is compiler-wide; evidence may initially support only a bounded subset. Unsupported addressing, register overlap, scale codes, shapes, modes, or control flow must be explicit results, not extrapolations from a test of the same mnemonic.

## Bundle structure

The proposed JSON format is `atlas.rtl_timing.contract.v0`. V0 is a review draft, not compatible input to the historical `atlas.rtlgraph.contract.v1` importer. The schema describes a target profile; a future compiler separately exports program-specific resolved footprints tied to that profile's hash.

| Field | Meaning |
| --- | --- |
| `target.config`, `target.reference` | Canonical name `EE290SimConfig` and its public bringup-chipyard configuration definition |
| `build` | Actual producer configuration, executed command and provenance when bound; exact source inputs, FIRRTL/annotations, retained hardware IR, tools and optional simulator identities |
| `catalog` | Versioned operation catalog path, byte hash, compiler revision and selected decoder reference |
| `conventions` | Clock domain, issue-age origin, inclusive holds, event mapping and explicit units |
| `policy` | Rejection of mismatches, missing rules, unqualified safety facts and unimplemented resolvers; separation of costs from legality |
| `resources` | Storage, ports, capacities, queues and control state, including unit/index/address interpretation and alias domains |
| `rule_groups` | Exhaustive mnemonic coverage, per-mnemonic resolver bindings, required rule dimensions, evidence-backed parameter facts and unresolved obligations |
| `coverage_records` | Reusable, explicitly referenced coverage records; use of a shared record never broadens its applicability |
| `evidence` | Stable IDs, source/artifact locators, method, applicability and validation scope |

A fact carries its value, unit, basis, qualification scope, and evidence references. A model assumption or historical result can be recorded without being enabled as a selected-target rule. Unknown values are `null` with an explanation; they never mean zero, no access, free resource, or completed transfer. A `not_applicable` coverage decision needs a reason and evidence just as a positive claim does. `unsupported` means the consumer has no implementation for the declared operation/domain; `unknown` means a required fact remains unresolved. Both may be present in an exhaustive catalog without preventing use of separately supported operations.

The draft bundle deliberately contains no qualified timing constants or enabled resolver bindings. Its complete rule-group inventory is a list of obligations, not a scheduler configuration that is safe to apply. The provider must reject it for evidence-backed scheduling until the relevant scopes are established.

Relative artifact paths resolve against the bundle's directory; absolute paths identify local artifacts. Paths are locators, not identities or executable commands. An exported bundle must include accessible artifacts or rewrite their locators while retaining byte hashes. When present, `build.command` records the executed artifact producer invocation, with its working directory in provenance; `build.provenance` retains exact upstream build labels/commands, while tool-specific `arguments` record individual lowering invocations. Neither field authorizes execution. The checked-in example is an unbound template: build artifacts, command and provenance are null, source selection is incomplete, and target parameters are unknown. It retains only the compiler catalog/interface evidence and does not describe a completed public build. Per-run bundles under ignored `build/rtl-timing/` retain the actual source labels, measured facts and artifact identities. Do not rename private build evidence into a public-source claim or infer equivalence from `target.config`.

## Values, parameter interpretation, and resolvers

An operation resolver is a named, versioned, pure compiler function taking the decoded instruction, known/unknown scalar state, allocation identities and selected facts. The JSON names the resolver and supplies typed parameters; it is not executable Python, C++, or an unrestricted expression language. Unknown resolver IDs/versions are errors. A resolver change that affects address calculation, event timing, footprint shape, admission, or completion requires a new identity and compatibility check.

The resolver registry must document the meaning and units of every required parameter and its applicability predicates. Scoped coverage binds a versioned predicate ID, typed arguments, and implementation identity; `applies_when` text explains that predicate to a reviewer and is not executable proof. Unknown predicate versions/arguments are errors. Evaluate the predicate for the actual mnemonic, fields, allocation and pending state, including each sibling in a grouped rule. No resolver may obtain an unrecorded safety constant from the old hardcoded implementation. Initial adapters can wrap existing computations, but inherited constants must remain explicitly labeled and must not silently become RTL-derived facts.

Inputs include instruction class and exact mnemonic, MXU/channel selection, typed physical operand fields, immediate widths/sign extension, register-pair and slot allocation, known/unknown values, and pending operation identities. Discriminate layout and numerical modes even where the current implementation shares one `OpClass`. In particular, VPU logic sharing is operation-dependent; MXU overwrite and accumulation differ; DMA CONFIG shares global state despite its channel suffix.

Address transforms record the encoded operand unit, scale, sign extension, masking/truncation, alignment and resulting address space. Canonical memory ranges are half-open byte intervals `[begin, end)` after those transforms; row and VMEM-line projections are explicit derived views. This convention does not change the hardware's word-addressed operands. Keep unknown addresses and unknown lengths conservative, preserve wrap/truncation semantics, and do not fabricate ranges from a nominal tile shape.

## Resolved result required from the provider

The existing [AtlasTiming interface](../../include/Atlas/AtlasTiming.h) is the starting point. The table below describes required semantics; additions beyond today's C++ types are proposals and must not be advertised as implemented.

| Result | Required content | Existing mapping / gap |
| --- | --- | --- |
| Admission | Legal modes, operand constraints, acceptance conditions and permitted outstanding work | Existing `Footprint.error` covers some constraints; complete frontend/queue admission is additional work |
| Storage accesses | Resource and index, read/write, exact range or conservative unknown, element order/stride, first event and per-element interval | `Access {res, write, first, count, age, step, anywhere, atCompletion}` supports regular issue-relative row streams; selective masks, address stride and external memory need explicit extension or conservative projection |
| Resource reservations | Pool/index, capacity demand, interval, alternative choices, and justified sharing predicate | `Hold` and `ReservationTable`; don't equate an alternate port with two simultaneous reservations or arbitrary sharing |
| Releases | Last read/write, allocation lifetime, engine release, and first subsequent safe use | `readRelease`, `writeRelease`, `mregReads/Writes`, `writeDuringRead`; distinguish inclusive last occupied cycle from first free cycle |
| Completion | Fixed proven event or symbolic event, pending operation token, observer/wait and effects discharged | `atCompletion` is a useful marker but does not alone express the full async protocol or its CFG lifetime |
| Control/order effects | Scalar issue/stall behavior, redirect/delay slot, ordered CSR/host interactions, halt/drain obligations | Requires coordination with stream/CFG verification and finalization beyond ordinary data footprints |
| Cost | Optional latency/throughput estimate and conditions | `dmaCycles`/critical-path estimates may rank candidates; they cannot prove memory completion or resource release |
| Explanation | Exact rule and evidence IDs used, applicability decisions and unresolved effects | Required for mismatch diagnostics, exported schedules and Merlin feedback |

The proposed provider result is `ResolvedRule {accesses, holds, releases, admission, pendingEffects, controlEffects, costs, explanation}` plus a status of `supported`, `unsupported_instance`, `unknown_safety_fact`, `illegal_instruction`, or `identity_mismatch`. This is a proposed interface, not an existing C++ declaration. Launch/config/wait effects describe explicit transitions of pending analysis state; a versioned CFG transfer/join operation merges that state. Keep these transitions separate from the pure per-instruction footprint computation. Events may anchor access streams, reservation endpoints and releases as well as completion. Where the present fixed-age C++ representation cannot express such an event, the adapter must reject the scope or use a separately justified conservative projection.

The current resource enumeration includes XReg, EReg, MReg, Acc, Weight, Vmem and DmaBase. The contract additionally needs explicit scope for DRAM ranges, CSR/PC/control state, DMA channel and ring-slot ownership, and any shared frontend/physical resources absent from that enumeration. Representing those names in JSON does not implement their enforcement.

Register-pair aliases, MXU-local accumulator/weight slots, overlapping VMEM projections, and DMA/host memory aliases must meet in the same resource model. Static allocation lifetime is separate from each instruction's timed accesses. First output visibility is separate from last output visibility and acceptance of another instruction.

## Time and events

* Numeric ages count Atlas-core cycles from a specified instruction-issue event; all parties must identify the same RTL signal/edge for age zero.
* Instruction issue, engine command acceptance, first operand read, last read, first result write, last result write, engine release and observed completion are distinct events. Record their mappings and any assumptions about stalls/backpressure.
* `conventions.events` binds named events to an RTL signal/locator, clock edge, versioned acceptance predicate, and origin/fixed-offset/dynamic relation to another event. Unresolved locators, offsets and conditions remain null in the draft. A signal name without its condition and edge is insufficient.
* For the existing regular stream representation, element `i` is touched at `age + i * step`. Count, element indexing and operation-specific physical order are independently specified. Repeated reduction passes require multiple streams; selective VLI writes require exact masks or an explicitly conservative superset.
* Resource holds `[from, to]` are inclusive. Storage byte ranges are half-open. Record whether a reported number is a last occupied age or a first free age; no implicit off-by-one conversion is allowed.
* A timing relation conditional on successful command acceptance is not automatically an instruction-issue spacing rule. The frontend acceptance and operand-availability conditions must also be established.
* Read-during-write, same-cycle writeback/bypass, and shared-port read behavior require explicit policy/evidence. Equal ages alone do not resolve them.
* Unbounded dynamic completion remains symbolic. A clock frequency or a transfer-size estimate cannot turn it into a fixed cycle deadline.

## Operation-specific obligations

| Operation group | Facts the resolver must bind |
| --- | --- |
| Scalar ALU/CSR/upper/scale immediate | Capture/writeback events; implicit state accesses; actual issue interval; special x0/immediate behavior; volatile and externally visible effects |
| Scalar/scale loads and stores | Effective address/byte enables; load/store pipeline stages; bank/port arbitration; result writeback; pending effects at halt and restart |
| VLOAD/VSTORE | Effective address transform; row count/order; VMEM and MREG streams; bank/path occupancy; producer-consumer forwarding and allocation release |
| MXU push/pop | Unit/slot selection; source pairs/scales; accumulator/weight streams; read/write ports, alternate choices and slot lifetime |
| MXU matmul and accumulate | Operand read streams, result rows, overwrite-versus-read/modify/write accumulator behavior, compute capacity, shared ports and completion/release conditions for each unit |
| VPU arithmetic | Exact unary/binary op; source/destination pairs; row timing; logic group and slot occupancy; same-op and cross-op admission/overlap |
| VPU conversion/reduction/immediate | Pack/unpack stride and scale reads; row/column physical layout; multi-pass reads; selective masks; mode-specific result and release timing |
| XLU | Permuted source/destination coordinates, read/write streams, legal aliasing, engine occupancy and next issue |
| DMA transfer/config/wait | The protocol below, separately for all channels and effective operand domains |
| Branch/jump/delay/fence/halt | PC units and targets, mandatory delay slot, dynamic branch paths, issue stalls, actual ordering/drain semantics, and asynchronous state at exit |

### DMA illustrates symbolic completion

A transfer captures its effective addresses, size and channel at launch. Represent the assembled DRAM address, including the captured global base, rather than a live expression that changes when the source register or base is later overwritten. Memory hazards persist through transfer completion; a matching channel wait establishes completion to the program. This is a lifetime constraint and does not claim that all bytes physically move in one completion cycle.

Use a pending transfer identity tied to a launch instance, channel and queue-slot ownership. At a CFG merge, conservatively merge possible pending transfers and captured ranges; across loops, distinguish launch generations. A channel number alone must not let a later launch erase an earlier outstanding transfer. Unknown queue admission, ring-wrap behavior or pending state is a rejection/conservative-policy decision, not permission to overlap arbitrarily.

DMA.CONFIG updates global base state and is not a transfer enqueue in the audited RTL. Preserve ordering between config and subsequent launch, but do not allocate transfer busy state to config merely because it is in the DMA instruction family. Attempted wait issue, successful wait issue, and wait return are separate named events until their relationship is established from RTL; reaching a stalled wait does not discharge pending effects. DMA source/destination hazards include DRAM as well as VMEM; current C++ DRAM checking is a gap.

### Control and completion are first-class

The contract must bind PC units and delayed redirects to the checked instruction stream. Scheduling cannot move an instruction across a branch, delay slot, wait, volatile CSR access, publication event, or halt merely because an ordinary data footprint is empty. Incoming async work, joins and backedges require state carried through the CFG. A `fence` is not assumed to drain engines; ECALL/EBREAK is not assumed to mean all outstanding writes completed.

Compiler-only annotations such as completion markers require a verified lowering/use path and explicit semantics. Their presence in a helper `Instr` field is not proof that physical MLIR lowering sets that field or enforces a host completion protocol. Host start/stop/restart, memory ownership/publication and output observation belong to the runtime boundary and must be compatible with this instruction-level completion model.

## Evidence, applicability and acceptance

Evidence records retain exact source/artifact identity, the analysis or observation method, signal/operation locators, assumptions, operand/configuration domains, and what was checked. Historical reports, current Python-model rules, standalone-core observations, integrated-system observations, and structural extraction are separate bases. A bounded execution is evidence only for its declared scope; matching output does not prove every timing boundary.

Before using a rule, a consumer must check all of the following:

1. Schema/version, unique IDs and references; artifact byte hashes; canonical target and actual source/build identity; supported tool/lowering assumptions.
2. Catalog equality against the active compiler and selected decoder/lowering inventory, including all mode/channel/unit expansions. A family-level row cannot hide a missing mnemonic.
3. Resolver version and required parameters; every required safety dimension is supported for this instruction instance or explicitly proven not applicable.
4. Address units, physical resources and aliases; admission preconditions; unknown-value handling; event mapping and endpoint conventions.
5. CFG and runtime boundary conditions, final instruction ordering and encoded-word identity. Evidence and resolved footprints must refer to the program actually emitted/loaded.

Schema, identity, reference, or catalog inconsistency rejects the whole profile. Unknown/unqualified safety facts, unsupported resolvers/domains, or failed admission reject the instruction instance/program that needs them. Other catalog entries may remain explicitly disabled; no full-ISA timing proof is required to enable a supported subset. Illegal encodings/operands remain distinct from legal-but-unsupported instructions. A separately selected conservative mode can exist, but its rule set and limitations must be explicit and validated; a large guessed delay is not a generic safe fallback. Costs and legality are separate APIs. Artifact agreement alone does not qualify an instruction or schedule.

JSON Schema checks structural shape. Hash verification, catalog equality, referenced-evidence existence, duplicate coverage, required parameter semantics, predicate sufficiency, CFG reasoning and execution qualification need semantic validation. A draft may retain null artifact identities for review; an enabled profile must reject missing identities and unresolved required facts.

## Illustrative resolutions beyond DMA

These examples are symbolic, not measured timing values. `R`, `W`, `H` and capacities must come from applicable facts before scheduling.

* **VLOAD:** a captured word-addressed base resolves to a VMEM byte range and line stream; VMEM reads start at `R`, destination MREG rows become visible starting at `W`, and the load path/bank reservations end at their separately established events. A consumer of row 0 and reuse of the entire register can have different constraints.
* **MXU overwrite matmul:** MREG and selected weight-slot reads, accumulator writes and compute/port occupancy are separate. An accumulator data-read or read-port hold must be justified for overwrite mode independently from accumulate mode. MXU0 and MXU1 have separate fact bindings.
* **VPU column reduction:** both registers of a pair can be read in multiple passes; the physical reduction layout, result masks, logic-group occupancy and overlap with a different VPU function must be represented. One first-result latency cannot encode those obligations.
* **Scalar load followed by halt:** a scalar destination writeback can remain pending while the frontend reaches ECALL. The contract must establish the required retirement/drain ordering; a modeled delay instruction count alone is insufficient evidence.

An illustrative VLOAD result makes the required structure concrete. Symbols below are unresolved parameters, so this is explanatory pseudodata, not a supported result or schema instance:

```yaml
status: unknown_safety_fact
instruction: vload
admission: {predicate: vload_effective_range_and_bank_v0, result: unresolved}
accesses:
  - {resource: vmem, range: captured_byte_range, projection: vmem_lines,
     count: N, address_stride_bytes: line_bytes,
     mode: read, anchor: engine_accept, first_offset: R, cycle_step: S_read}
  - {resource: mreg, register: decoded_rd, first_row: 0, count: N,
     mode: write, anchor: engine_accept, first_offset: W, cycle_step: S_write}
holds:
  - {resource: vload_path, index: 0, demand: 1,
     from_event: engine_accept, to_event: load_path_last_occupied, end: inclusive}
releases:
  - {resource: mreg, register: decoded_rd, event: last_destination_row_visible}
pendingEffects: [retain_destination_write_until_visibility_event]
evidence_refs: [applicable_selected_rtl_stream_and_arbitration_evidence_pending]
```

## Integration and validation plan

The [bounded selected-evidence backend](selected-evidence.md) supplies an identity-checked provider and final timed verification for serialized VLS with minimal scalar setup and completion control. Its conditional module replay does not discharge system execution, SRAM implementation or original source/build-linkage obligations. Resolve the remaining coverage gaps against retained CIRCT hardware IR and selected source, keeping unsupported scopes visible. Agree on additional resolver/parameter interfaces with Jeremy and resource/CFG obligations with the verification work. Export profile identity and resolved instruction facts for Merlin instead of maintaining independent timing constants there; that active-compiler feedback adapter remains future work.

Validation should target discriminating boundaries: one cycle before/at/after availability; legal versus illegal same-register overlap; alternative-port and capacity saturation; cross-family MREG/VMEM conflicts; each VPU overlap group and reduction/VLI mode; DMA capture, config ordering, missing waits and ring reuse; branches/loops/joins with pending work; CSR publication and halt/restart. Include unknown-address and mismatched-source cases. Recheck numerical outputs and guard memory for the supported integrated program; record actual RTL cycles separately from model estimates.

Issue #11 remains open until the source/elaboration audit, mismatch rejection, and selected EE290 execution requirements are met. Completion of this draft or the VCS build does not complete that issue.
