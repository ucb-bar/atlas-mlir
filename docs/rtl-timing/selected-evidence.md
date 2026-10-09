# Selecting conditional RTL evidence

The compiler explicitly selects RTL evidence for scheduling, delay insertion and final timed verification. Selection begins with the [bounded VLS replay](vls-timing.md), schema `atlas.conditional_vls_hw_check.v0`, and can add separately checked [DMA](dma-timing.md) and [XLU](xlu-timing.md) receipts for the same hardware. The broad [contract v0](README.md) remains a draft and is not accepted by this loader. All implemented selections remain conditional. The [extraction walkthrough](extraction-walkthrough.md) explains where the rules come from; this guide covers selecting and using them.

## Selection and compatibility

Supply three independently selected SHA-256 digests: the replay report, its retention manifest and the full retained HW/Comb/Seq artifact. Obtain the expected values from the exact reviewed artifacts; copying whatever identity an arbitrary report declares is not an independent selection. The loader checks actual bytes, referenced artifacts and their crosslinks, including retained input snapshots, replay traces and compiler comparisons. Keep the report's `retention-manifest.json` snapshot and referenced artifacts accessible. Moving artifacts does not waive locator or identity checks; rewriting metadata changes its digest and requires a newly reviewed selection.

Resolver v1 additionally pins the reviewed AtlasCore transitive instance closure: 123 module definitions, including frontend, root wiring, LSU, CSR, storage wrappers and external declarations. The `atlas.module.locations_only.v1` normalizer removes only horizontal whitespace followed by `loc(#locN)` outside quoted strings; it preserves semantic attributes, quoted text and other bytes. A changed module body, memory geometry or frontend/wiring is rejected even when the caller supplies matching new artifact hashes. Renew the static review and resolver version before updating compiled compatibility bindings. The [fingerprint helper](../../tools/fingerprint-rtl-modules.py) prints review candidates; it does not authorize or install a new target.

`allow-conditional=true` is mandatory. The selected module records `atlas.rtl_evidence` and `atlas.rtl_qualification = "conditional"`. A malformed selection, incompatible hardware, changed digest, missing required artifact or unsupported operation/domain fails; selected consumers never substitute legacy model footprints. The loader rejects a qualification annotation without selection and claims of a stronger qualification status. Invocations without a selection retain the separate legacy timing mode; `verify-atlas-rtl-timing` always requires selected evidence.

## Supported program domain

The base provider supports `addi`, `lui`, `delay`, `ecall`, VLOAD/VSTORE and a completion marker consisting of `csrrw x0, 0xC10, rs1`. Scheduling and insertion accept one delay-free straight-line machine stream ending in ECALL. CFG, scalar memory, other CSR modes and operations without separately selected rules are rejected. The final program, including inserted instructions, must fit the reviewed 32,768-word instruction memory. Scalar register values initially remain unknown except x0; each VLS base must become known through supported setup.

VLS operates on a raw 1,024-byte tile, with MREG IDs 0–63 and a known effective line address `L = (((B + 32 * sext12(I)) mod 2^32) >> 3) & 0xffff`. L must be a multiple of 32 in 0–49,120 and the 32-line tile must lie wholly within one of six VMEM banks. The scalar base B counts 32-bit words. Both VLS paths are reserved through age 34 for every command, so **every subsequent VLS command requires an issue gap of at least 35**, including cross-path commands that the conditional replay found could overlap in narrower cases. Rolling overwrite and overlapping engine traffic are not enabled.

The public MLIR `offset` attribute uses the signed range −2048–2047. Emission encodes its 12-bit representation: −2048 becomes `0x800`, and −1 becomes `0xfff`. Raw positive attributes 2048–4095 are rejected by the dialect even though the decoded instruction API interprets those bit patterns through sign extension. Keep the source-level attribute and encoded field distinct when constructing boundary programs.

The environment remains conditional: reset is deasserted, the accelerator is quiescent at entry, other engines and external traffic stay quiescent, and SRAM responses arrive one cycle after requests. START does not drain or reset pending engine work. Loaded instructions must remain unchanged, host accesses must not compete for VMEM or rewrite DBG0 during execution, and callers supply the required initial operand data. Pinning external SRAM declarations does not validate their implementations. Source requests occur at ages 1–32 and destination writes at 3–34. Marker publication and terminal acceptance must occur after prior asynchronous writes finish. DELAY N places its ordinary successor N+1 cycles later; a terminal immediately after DELAY can act during that stall, so inserted terminal drain sequences end on a NOP. The final checker rejects adjacent DELAY→ECALL and incomplete work, verifies dependencies and reservations, and bounds the total issue timeline to one million cycles.

## Optional DMA and transpose evidence

Add either or both pairs of options to the same `--select-atlas-rtl-evidence` argument. A path without its expected digest, incompatible hardware, incomplete replay or modified event stream fails selection.

| Options | Additional supported domain | Resolver ID |
| --- | --- | --- |
| None | Base VLS/scalar/completion domain | `atlas.vls.conservative.v1` |
| `dma-evidence=/path/to/report.json dma-evidence-sha256=DMA_SHA256` | Load/store/config/wait on channels 0–7; one pending transfer globally | `atlas.vls_dma.serialized.v1` |
| `xlu-evidence=/path/to/report.json xlu-evidence-sha256=XLU_SHA256` | Serialized full-byte transpose with source/destination MREGs 0–63 | `atlas.vls_xlu.serialized.v1` |
| Both pairs | Both domains, with DMA and fixed engines serialized | `atlas.vls_dma_xlu.serialized.v1` |

DMA requires known aligned operands, full 32-byte beats, sizes 32–4096 bytes within one VMEM bank, a five-bit configured DRAM base and no low-address wrap. Configuration takes effect synchronously. A transfer captures scalar operands and the configured base at launch, allowing later scalar/base changes; its memory lifetime ends only at the matching channel wait. Empty, wrong-channel and stale waits fail. Other VMEM work, transpose, another transfer, marker publication and halt require the pending transfer to be waited. DELAY never substitutes for completion. Fresh START and quiescent DMA entry are explicit assumptions; START alone does not drain queues.

XLU reads source rows at ages 1–32 and writes transposed rows at 34–65 under the selected one-cycle MREG response assumption. Every XLU command reserves both VLS paths and its engine through age 65, so its next fixed-engine successor issues no earlier than age 66. In-place and physically aliased source/destination IDs are admitted only with this serialization. The [XLU guide](xlu-timing.md) gives the derivation and component evidence; mixed-engine and full-system qualification remain separate.

The same [target policy](compiler-consumers.md) is used by dependencies, reservations and final checking. VPU and MXU modes remain unavailable unless an explicit compute policy and matching resolver evidence are supplied; adding a receipt for one family never enables another.

## Build, compile and check

Use matching LLVM/MLIR packages and tools from one installation. Build this checkout in its own ignored directory; do not reuse another checkout's CMake cache.

```bash
cmake -S . -B build/rtl-timing/compiler \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/path/to/python3
cmake --build build/rtl-timing/compiler --parallel 4 \
  --target atlas-opt atlas-emit atlas-rtl-evidence-test
```

From the checkout root, replace the paths and three expected digests below. The [copy example](../../test/examples/ee290_vls_copy.mlir) uses host-preloaded VMEM and needs no DMA instruction. Use either `--insert-atlas-delays` or `--schedule-atlas-stream`, then verify the final rewritten stream before emission.

```bash
build/rtl-timing/compiler/bin/atlas-opt \
  test/examples/ee290_vls_copy.mlir \
  '--select-atlas-rtl-evidence=evidence=/path/to/report.json evidence-sha256=REPORT_SHA256 manifest-sha256=MANIFEST_SHA256 hardware-ir-sha256=HW_SHA256 allow-conditional=true' \
  --insert-atlas-delays --verify-atlas-rtl-timing \
  -o build/rtl-timing/final-vls.mlir
build/rtl-timing/compiler/bin/atlas-opt \
  build/rtl-timing/final-vls.mlir --verify-atlas-rtl-timing
build/rtl-timing/compiler/bin/atlas-emit \
  build/rtl-timing/final-vls.mlir > build/rtl-timing/final-vls.words
```

The final verifier checks the actual encoded machine stream and decoded operands afresh, including inserted DELAYs; it does not trust schedule annotations. Selected `atlas-emit` invocations and both LLVM conversion paths rerun this check; structured LLVM finalization also checks the reconstructed stream. Re-run verification after any instruction change. Selection establishes artifact compatibility and conditional applicability, not exact compiler-binary or emitted-program provenance; retain those identities separately in the execution receipt.

Run the provider and selected-consumer regressions with explicit evidence:

```bash
build/rtl-timing/compiler/bin/atlas-rtl-evidence-test \
  /path/to/report.json REPORT_SHA256 MANIFEST_SHA256 HW_SHA256
ATLAS_OOT_BIN_DIR="$PWD/build/rtl-timing/compiler/bin" \
  ATLAS_RTL_EVIDENCE_REPORT=/path/to/report.json \
  /path/to/python3 -m unittest discover -s test -p test_rtl_timing.py -v
```

These checks cover selection rejection, unsupported instances, both consumers, final emission, serialized admission boundaries and premature marker/halt cases. The evidence replay is still 20 hardware cases; a larger compiler API matrix is not additional hardware replay coverage.

The DMA and XLU compiler suites are `test_rtl_dma_timing.py` and `test_rtl_xlu_timing.py`. In addition to the environment above, select `ATLAS_RTL_DMA_EVIDENCE_REPORT` and/or `ATLAS_RTL_XLU_EVIDENCE_REPORT` explicitly. The XLU suite uses the DMA receipt for its combined pending-transfer case. The guides for each family describe producing the hardware evidence; compiler tests do not substitute for those executions.

## Integrated witness and remaining qualification

The [witness runner](../../tools/run-ee290-vls-witness.py) emits the supplied final program, builds the [RISC-V host](../../test/ee290-vls-host.c), and launches a selected EE290SimConfig simulator. Supply the already verified final stream, a verified retention manifest, the simulator and its explicit source file list, a compatible host compiler with HTIF support, and the required simulator runtime/license environment. The output directory must be new beneath ignored `build/rtl-timing/`.

```bash
/path/to/python3 tools/run-ee290-vls-witness.py \
  --program build/rtl-timing/final-vls.mlir \
  --atlas-emit build/rtl-timing/compiler/bin/atlas-emit \
  --manifest /path/to/manifest.json \
  --simulator /path/to/ee290-simulator \
  --sim-source-list /path/to/simulator-sources.f \
  --host-cc /path/to/riscv64-unknown-elf-gcc \
  --output build/rtl-timing/witness \
  --max-cycles 2000000 --timeout-seconds 300
```

The host loads instructions, preloads VMEM, starts Atlas, observes terminal state and the completion marker, and checks three data panels: 256 output words plus 1,280 preserved input/guard words per panel in a 6,144-byte window. The runner retains commands, binaries, hashes, logs and its `atlas.ee290_vls_witness.v0` receipt. It does not perform the selected scheduling check itself. Observer cycle counts include host polling and are not kernel timing measurements.

A local integrated run passed both a conservative baseline, with DELAY 255 after each VLS instruction, and the program produced by `--schedule-atlas-stream`. Both final 10-word streams passed selected timing verification before execution. Each arm passed all three data panels, preserved the checked inputs/guards, and reached ECALL status 5 with completion marker 1 and no assertion failure. Executed programs, encoded words and host/runner snapshots matched their saved identities; both arms used the same selected simulator and evidence. These are finite numerical execution witnesses, not measurements of internal request/response timing or kernel speedup.

The receipts retain `scheduling_qualified = false`. Subsequent selected-source/CIRCT correspondence and direct AtlasCore/SRAM observations strengthen the evidence, but they do not retroactively capture the historical full-system build or qualify every operand admitted by the conditional resolver. A `runtime_license_unavailable` receipt remains a blocked execution and cannot substitute for a passing witness. These integrated programs exercise VLS; additional DMA and XLU selections retain their separate component scopes. Full-ISA rules, DMA lifetimes across CFG, general restart/overlap behavior and an active Merlin feedback adapter remain outside this backend. The emitted-program output/guard portion of [issue #11](https://github.com/ucb-bar/atlas-mlir/issues/11) has a bounded integrated witness; the remaining selected-source, build-linkage and qualification obligations remain explicit.

Use the [provenance checker](build-provenance.md) to audit the retention/execution crosslinks and optional current source inventory. Its successful result verifies recorded artifact relationships while explicitly leaving the missing historical build relationships and timing qualification open.

## Evidence available for the handoff

These layers have different scopes. Keep their selected identities and applicability together; no single success label enables a qualified compiler profile.

| Layer | Established result | Remaining boundary |
| --- | --- | --- |
| Selected source to CIRCT | [Fresh source compilation/elaboration and correspondence](captured-builds.md) reproduce all 123 reviewed AtlasCore HW definitions after location-only normalization. | Selected Atlas/configuration source overlay against a pinned opaque Chipyard dependency; no claim of complete dependency-source reconstruction or upstream configuration equivalence. |
| Retained CIRCT to compiler | The conditional report, manifest, HW artifact and reviewed module semantics select the common VLS resolver used by scheduling, insertion and final verification. | Conditional environment and supported instruction/operand domain above; the broad draft contract is not an accepted profile. |
| Emitted program to EE290 system | Baseline and scheduled programs pass the three-panel integrated numerical/guard fixture and settled completion checks. | The historical simulator's complete source/build chain and internal timing trace remain unestablished. Prospective [captured simulator builds](captured-builds.md) record a new build rather than repairing an old receipt. |
| Captured EE290 system to internal events | A new captured simulator build and [native VPD observation](vcs-observation.md) pass both numerical programs and boundary decoding, with all 356 selected signals visible. The three panels per arm reproduce gaps 257/35 and the component request/response/write/release ages. | Finite bank-0 lines 32/96 and MREG4; independent aggregate evidence and physical SRAM/competing-request checks remain separate from the capture producer's success state. |
| Selected AtlasCore to observed events | The [component replay and SRAM observer](vls-observation.md#selected-atlascore-replay-with-behavioral-srams) bind the compiled model, executed words and trace. Both arms pass numerical checks and actual frontend/LSU/SRAM events; observed VLOAD-to-VSTORE gaps are 257 and 35 cycles. | Finite bank-0 copy operands and selected behavioral SRAMs, direct TileLink host, two-state Verilator. This is separate from full EE290 CPU/system execution. |
| Resolved compiler export | `atlas-emit --rtl-timing-json` reruns final timing verification and exports the shared provider's exact footprints, encoded words, identities and applicability. See [resolved export](resolved-export.md). | Conditional model facts; the export does not qualify the hardware or widen the provider's domain. |
| Finite component applicability | The [bounded checker](bounded-applicability.md) recomputes boundary/SRAM observations and binds operands, issue gaps, accesses, release and completion to the selected compiler export. | Recorded finite operands/windows only; source-to-simulation completeness, competing-request assumptions and full-system timing remain explicit. |
| Finite integrated applicability | The [integrated checker](integrated-observation-check.md) revalidates the captured pair, source/HW content correspondence and fresh shared-provider exports. Both arms pass physical memory, clock-transition and competing-request checks throughout their observed entry-to-halt windows. | Exact captured programs and operands; host fixture assumptions, opaque dependencies, the unrecorded Verilog-generation execution edge and general domain qualification remain explicit. |
| Qualified compiler acceptance | Not yet implemented. Existing selection remains explicitly conditional. | Establish the admitted scope and validate its build/trace/domain evidence before changing qualification policy. |

The [walkthrough](extraction-walkthrough.md) connects the observed event sequence to the shared footprint; [VLS timing](vls-timing.md) specifies the exact ages and boundaries. Extending qualification beyond the finite observations requires targeted checks or a reviewed structural-equivalence argument for address/bank mapping, MREG selection and admitted transitions. Compiler rejection tests remain separate from hardware coverage.

A bounded delivery can retain the conditional consumer while supplying these artifacts and their reproduction tools. A future qualified path must validate the evidence rather than trust a copied qualification flag. Extend the existing resolver and final checker once that acceptance policy is established; do not create a second independent timing table for Merlin or silently enable the draft rule groups.

## Bounded handoff

The current deliverable is the conditional VLS provider together with independently validated observations of the captured baseline and scheduled programs. To reproduce and review it:

1. Select the reviewed replay, manifest and hardware identities, then compile and verify the final stream using the commands above.
2. Export its resolved facts with [`--rtl-timing-json`](resolved-export.md). Keep the final program, encoded words and selected artifact identities together.
3. Run the [integrated checker](integrated-observation-check.md) with its source, fresh-export and system-memory options. All five `coverage` fields must be true for this combined handoff; a run without those options establishes a narrower result.

Supply the selected evidence, final programs, exports and aggregate report with their referenced artifacts. Keep `allow-conditional=true`: the validated finite cases do not establish every operand admitted by the provider. Extending operation coverage and connecting Merlin can reuse this interface without another scheduler or timing table.
