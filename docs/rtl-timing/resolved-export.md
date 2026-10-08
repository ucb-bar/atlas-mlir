# Exporting resolved RTL timing

`atlas-emit --rtl-timing-json final.mlir` exports the selected timing provider's facts for the actual final physical instruction stream. The input must carry a valid explicit RTL evidence selection and pass final timing verification. The command rejects an unsafe stream, unsupported operation or operand domain, missing or changed evidence, and claims of stronger qualification. It emits no JSON on failure. Ordinary `atlas-emit`, `--map-json` and `--program-json` retain their existing behavior.

The exporter uses the same `RTLEvidence::resolve` footprints and final check as scheduling, delay insertion and `--verify-atlas-rtl-timing`. It does not contain another timing table or infer facts from scheduler annotations. The existing `FootprintResolver` remains the compiler interface; the export lets evidence tools, verification consumers and Merlin inspect its resolved facts. Independent transformation and allocation verification continue to own their source/placement/program analyses.

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

The result explicitly remains `qualification: conditional` and `scheduling_qualified: false`. Validating finite component observations does not qualify every operand admitted by this conditional provider or establish full-system timing. The export states the remaining assumptions, including quiescent competing memory requesters and one-cycle SRAM responses. Broader evidence acceptance must validate those conditions instead of copying a success flag. Dynamic DMA completion is outside this provider's domain; adding an interface for it should accompany an evidenced rule and its consumers.

The JSON omits workspace paths and does not modify the input or evidence artifacts. Keep generated exports in the ignored artifact directory alongside immutable program, compiler and observation identities.
