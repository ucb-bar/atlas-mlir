# Compiler timing consumer contract

Selected RTL timing is conditional evidence for an explicit bounded instruction domain. A successful schedule or final timing check does not qualify unsupported operations, arithmetic, control flow, external memory accessibility, or a different hardware configuration. The selected evidence and manifest/hardware identities must survive every physical rewrite and LLVM reconstruction.

## Shared interface

`TargetTiming` supplies footprint resolution and chooses the surrounding engine policy. Its default `LegacyModel` policy preserves the named, unqualified `npu_model` rules. Constructing it from a nonempty `FootprintResolver` selects `ConservativeRTL`; this policy admits scalar, LSU, DMA and XLU engines only, with each concrete instruction still subject to the resolver's instance checks. Selected MXU and VPU instructions fail before their callback is used because reviewed pair sequencing, overlap and capacity policies are absent.

Pass the same target to `buildGraph`, `checkAtlasStream`, `dependence` and `ReservationTable`. A resolver callback alone must not select footprints while dependencies or reservations continue with a default legacy target. Conservative dependencies retain ordinary access ordering, release constraints, DMA launch/wait ordering and fixed Atlas MREG port checks; they omit model-specific MXU sequencing. Conservative reservation capacities are one and model-specific VPU overlap/read sharing is unavailable. Compute admission requires a future explicit policy extension and evidence; returning a compute footprint does not enable it.

`Footprint::dmaAsync` identifies commands completed by a matching channel wait independently of `dmaCycles`. `exclusiveVmemUntilWait` identifies the selected conservative DMA admission. `Access::atCompletion` represents asynchronous memory ownership rather than a fixed access age. Legacy `dmaCycles` is only a scheduling priority hint; selected DMA has no assumed completion latency.

## DMA and wait coordinates

The selected DMA profile captures scalar operands and the global configured DRAM base at launch. Configuration is synchronous and creates no pending transfer. Scalar operands and the configured base may be reused after capture; transferred VMEM and DRAM effects remain asynchronous. An external DRAM access may conservatively cover all DRAM when an exact widened span is unavailable.

Only one selected DMA transfer may be pending globally. Its matching channel wait consumes that generation; empty, wrong-channel and stale waits fail. All VMEM accesses, including disjoint ranges and read/read pairs, are excluded while that transfer is pending. A new DMA launch drains earlier finite VMEM work. Completion publication and terminal halt require every pending transfer to have been consumed by its matching wait. Delays cannot substitute for a wait.

Scheduling and delay insertion use nominal issue coordinates that are lower bounds across an unknown-duration wait. Finite resource reservations extend conservatively across the wait. Final verification independently reconstructs pending transfers and their matching waits from the actual rewritten instruction stream. DMA export must identify wait-relative issue epochs and matched completion events; it must not present lower-bound issue coordinates or completion-time memory effects as fixed physical cycles.

The reviewed XLU footprint reads rows at ages 1–32, writes rows at ages 34–65 and occupies its single engine through age 65. Generic access and engine reservations release a subsequent transpose at issue gap 66. This statement describes the proposed instance contract; the selected resolver must independently admit XLU only with matching evidence.

## Verification-session coordination

The verification branch retains source-derived DMA, MXU and tile correspondence expectations and independently decodes actual physical commands. Reuse `verifyAtlasGeneratedDMAMemory`, `buildAtlasDMAContract` and `verifyAtlasGeneratedDMAContract` for their established obligations; scheduler footprints and dependency edges are not correspondence or allocation oracles. That branch's multiple-transfer/disjoint-range policy is broader than this selected DMA profile and cannot establish its physical queue or overlap admission.

The active untimed integration snapshot already accepts supplied footprint rules in `verifyAtlasTimedStream`/`verifyAtlasTiming`, but its `atlas.timing_provider` metadata currently recognizes only `npu-model-rtl-match-v1`. Integrating selected RTL requires the same `TargetTiming` policy in scheduling and final checking, explicit selected identity retention, and fail-closed rejection of missing provider rules. Rewriting must preserve synchronization, source contracts and command tags while treating timing validity as a property to recheck after mutation.

Standalone ARC arithmetic/effect tests and the selected EE290 timing evidence have distinct artifacts, configurations and qualification scopes. Neither test route implicitly supplies the other's target identity or missing timing rules.
