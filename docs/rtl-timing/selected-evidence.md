# Selecting conditional RTL evidence

The compiler can explicitly consume a passing [bounded VLS replay](vls-timing.md) for scheduling, delay insertion and final timed verification. The implemented input is `atlas.conditional_vls_hw_check.v0`, with resolver `atlas.vls.conservative.v1`; the broad [contract v0](README.md) remains a draft and is not accepted by this loader. Compiler integration preserves the conditional status of the evidence.

## Selection and compatibility

Supply three independently selected SHA-256 digests: the replay report, its retention manifest and the full retained HW/Comb/Seq artifact. Obtain the expected values from the exact reviewed artifacts; copying whatever identity an arbitrary report declares is not an independent selection. The loader checks actual bytes, referenced artifacts and their crosslinks, including retained input snapshots, replay traces and compiler comparisons. Keep the report's `retention-manifest.json` snapshot and referenced artifacts accessible. Moving artifacts does not waive locator or identity checks; rewriting metadata changes its digest and requires a newly reviewed selection.

Resolver v1 additionally pins the reviewed AtlasCore transitive instance closure: 123 module definitions, including frontend, root wiring, LSU, CSR, storage wrappers and external declarations. The `atlas.module.locations_only.v1` normalizer removes only horizontal whitespace followed by `loc(#locN)` outside quoted strings; it preserves semantic attributes, quoted text and other bytes. A changed module body, memory geometry or frontend/wiring is rejected even when the caller supplies matching new artifact hashes. Renew the static review and resolver version before updating compiled compatibility bindings. The [fingerprint helper](../../tools/fingerprint-rtl-modules.py) prints review candidates; it does not authorize or install a new target.

`allow-conditional=true` is mandatory. The selected module records `atlas.rtl_evidence` and `atlas.rtl_qualification = "conditional"`. A malformed selection, incompatible hardware, changed digest, missing required artifact or unsupported operation/domain fails; selected consumers never substitute legacy model footprints. The loader rejects a qualification annotation without selection and claims of a stronger qualification status. Invocations without a selection retain the separate legacy timing mode; `verify-atlas-rtl-timing` always requires selected evidence.

## Supported program domain

The current provider supports `addi`, `lui`, `delay`, `ecall`, VLOAD/VSTORE and a completion marker consisting of `csrrw x0, 0xC10, rs1`. Scheduling and insertion accept one delay-free straight-line machine stream ending in ECALL. They reject CFG, DMA, scalar memory, other CSR modes and other engines. The final program, including inserted instructions, must fit the reviewed 32,768-word instruction memory. Scalar register values initially remain unknown except x0; each VLS base must become known through supported setup.

VLS operates on a raw 1,024-byte tile, with MREG IDs 0–63 and a known effective line address `L = (((B + 32 * sext12(I)) mod 2^32) >> 3) & 0xffff`. L must be a multiple of 32 in 0–49,120 and the 32-line tile must lie wholly within one of six VMEM banks. The scalar base B counts 32-bit words. Both VLS paths are reserved through age 34 for every command, so **every subsequent VLS command requires an issue gap of at least 35**, including cross-path commands that the conditional replay found could overlap in narrower cases. Rolling overwrite and overlapping engine traffic are not enabled.

The environment remains conditional: reset is deasserted, the accelerator is quiescent at entry, other engines and external traffic stay quiescent, and SRAM responses arrive one cycle after requests. START does not drain or reset pending engine work. Loaded instructions must remain unchanged, host accesses must not compete for VMEM or rewrite DBG0 during execution, and callers supply the required initial operand data. Pinning external SRAM declarations does not validate their implementations. Source requests occur at ages 1–32 and destination writes at 3–34. Marker publication and terminal acceptance must occur after prior asynchronous writes finish. DELAY N places its ordinary successor N+1 cycles later; a terminal immediately after DELAY can act during that stall, so inserted terminal drain sequences end on a NOP. The final checker rejects adjacent DELAY→ECALL and incomplete work, verifies dependencies and reservations, and bounds the total issue timeline to one million cycles.

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

The final verifier checks the actual encoded machine stream and decoded operands afresh, including inserted DELAYs; it does not trust schedule annotations. `atlas-emit` performs encoding checks, not this selected timing check. Re-run final verification after any instruction change. Selection establishes artifact compatibility and conditional applicability, not exact compiler-binary or emitted-program provenance; retain those identities separately in the execution receipt.

Run the provider and selected-consumer regressions with explicit evidence:

```bash
build/rtl-timing/compiler/bin/atlas-rtl-evidence-test \
  /path/to/report.json REPORT_SHA256 MANIFEST_SHA256 HW_SHA256
ATLAS_OOT_BIN_DIR="$PWD/build/rtl-timing/compiler/bin" \
  ATLAS_RTL_EVIDENCE_REPORT=/path/to/report.json \
  /path/to/python3 -m unittest discover -s test -p test_rtl_timing.py -v
```

These checks cover selection rejection, unsupported instances, both consumers, final emission, serialized admission boundaries and premature marker/halt cases. The evidence replay is still 20 hardware cases; a larger compiler API matrix is not additional hardware replay coverage.

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

The host build has been exercised; integrated execution remains unestablished because the available runtime attempt could not acquire its simulator license. A `runtime_license_unavailable` receipt is a blocked execution, not a passing numerical witness. Even a later passing witness leaves original Chisel/source-to-simulator build linkage, selected SRAM response timing and internal boundary observations to qualify separately. Full-ISA rules, dynamic DMA/CFG lifetimes, general restart/overlap behavior and an active Merlin feedback adapter remain outside this backend. [Issue #11](https://github.com/ucb-bar/atlas-mlir/issues/11) stays open until its selected-source, mismatch and system-execution requirements are met.
