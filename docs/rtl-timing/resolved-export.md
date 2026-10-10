# Exporting resolved RTL timing

`atlas-emit --rtl-timing-json final.mlir` exports the selected timing provider's facts for the actual final physical instruction stream. The input must carry a valid explicit RTL evidence selection and pass final timing verification. The command rejects an unsafe stream, unsupported operation or operand domain, missing or changed evidence, and claims of stronger qualification. It emits no JSON on failure. Ordinary `atlas-emit`, `--map-json` and `--program-json` also recheck timing when the input carries an RTL selection; unselected behavior is unchanged.

The exporter uses the same `RTLEvidence::resolve` footprints and final check as scheduling, delay insertion and `--verify-atlas-rtl-timing`. It does not contain another timing table or infer facts from scheduler annotations. The shared `TargetTiming` interface couples footprint resolution with the selected admission policy; the export lets evidence tools, verification consumers and Merlin inspect its resolved facts. Independent transformation and allocation verification continue to own their source/placement/program analyses.

```sh
atlas-opt selected.mlir --schedule-atlas-stream --verify-atlas-rtl-timing -o final.mlir
atlas-emit --rtl-timing-json final.mlir > resolved.json
```

Select `selected.mlir` with the explicit identities and conditional opt-in described in [Selecting conditional RTL evidence](selected-evidence.md). No local target or report is selected implicitly.

The `atlas.resolved_rtl_timing.v0` object contains:

- The public target `EE290SimConfig`, resolver/version and selected evidence, manifest and hardware-IR SHA-256 identities.
- Final encoded words, their count and the SHA-256 of concatenated little-endian 32-bit words. This hash is distinct from a textual `.words` file hash; compare decoded words or use the declared hash encoding.
- Each instruction's decoded operands, logical issue cycle, accesses, resource holds, MREG usage, release fields and completion age from the shared provider.
- Operand applicability, environment assumptions and unsupported operation/concurrency domains.

Cycles form a conditional model timeline relative to the first instruction, not an observed elapsed-time measurement. Access element `i` occurs at `age + i * step`; MREG elements use `register * 32 + row`, VMEM elements are 32-byte lines and scalar elements are register numbers. Hold endpoints are inclusive. For the current serialized VLS provider, a last occupied age of 34 means the next VLS can issue at age 35. The provider's `read_release`/`write_release` fields describe its MREG lifetime policy; use the individual access streams for memory read/write ages. ECALL is labeled terminal acceptance, because it suppresses scalar fire and is not a retired instruction. Marker publication and terminal ordering are checked by the existing verifier.

With separately selected XLU evidence, the same fixed-age schema includes transpose footprints and `xlu_evidence_sha256`. That selection does not change the VLS-only schema or silently admit other compute modes.

## DMA completion coordinates

Selecting DMA evidence uses `atlas.resolved_rtl_timing.v1`. It includes `dma_evidence_sha256` and replaces each `logical_issue_cycle` with `minimum_issue_cycle`, `issue_epoch` and `epoch_offset`. A matching wait starts a new epoch at offset zero. Offsets can be compared within an epoch; the minimum issue coordinate across waits is a lower bound, not an elapsed-cycle prediction. The wait is labeled `matching_wait_acceptance` and records its channel.

DMA scalar/base accesses have launch-relative age zero. Memory accesses instead carry `at_completion: true`, `lifetime: launch_through_matching_wait` and null `age`/`step`. The transfer's completion and release ages and `dma_cycles` are also null; its completion rule is `matching_dma_wait`. Null means unknown fixed latency, never zero latency. `dma_async`, `exclusive_vmem_until_wait` and `serialize_with_dma` expose the shared policy obligations. External DRAM effects currently conservatively cover all DRAM; they are not an exact alias analysis.

The result explicitly remains `qualification: conditional` and `scheduling_qualified: false`. The [integrated observation checker](integrated-observation-check.md) compares fixed-age VLS exports with captured system events for the exact executed programs; it does not consume the DMA extension automatically. Those finite observations do not qualify every operand admitted by a provider. The export states remaining assumptions, including quiescent competing memory requesters and one-cycle SRAM responses.

The JSON omits workspace paths and does not modify the input or evidence artifacts. Keep generated exports in the ignored artifact directory alongside immutable program, compiler and observation identities.

Optional VMUL selection retains `vmul_evidence_sha256` in the evidence object. Its three MREG streams each span 64 rows (two reads and one write), including both architectural halves of each pair. The export uses the same evidence-gated target policy as scheduling and final verification; adding VMUL does not turn dynamic DMA completion into a fixed cycle.
