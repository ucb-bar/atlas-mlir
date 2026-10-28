# Complete ResNet50 host graph

The generators support a batch-one ResNet50 plan with 177 ordered nodes and 54
Atlas contractions. Convolution and linear geometry, padding, groups, parameter
extents, finite values, topology and asset hashes are checked before C
generation. Host operations use separate FP32 multiply/add rounding and
sequential average pooling. The generated graph packs each activation from the
current C tensor and consumes actual device replies. Captured successor
activations are never embedded as graph inputs.

The device callback stages live packed A with immutable B, patches one admitted
rescale word, and launches the sequenced raw runtime. It checks input/weight
preservation and output guards before returning the physical BF16 result.
Descriptors, staging buffers and the device context must remain alive after
failure: timeout can retain device ownership until external quiescence and
reinitialization. The selected DMA address is `0x100000000`; emulation may map a
different CPU staging pointer through the memory callback.

## Regenerate the programs and host ELF

Run from the Atlas checkout with the sibling Merlin repository and its
development Python environment. A source manifest identifies the frozen
checkpoint, image stream and selected image; those immutable data inputs remain
external. This reproduction route requires externally supplied fixed-input
fixtures and their hash-bound assets. The checkout does not distribute those
fixtures or provide an automated fresh-checkpoint capture runner. The public
Merlin entrypoints export the complete operator inventory, host plan and
arithmetic reference:

```sh
MERLIN=../merlin
REFRESH="$MERLIN/examples/atlas/comparisons/refresh"
PACKING_SHA=$(sha256sum tools/compile_workload_inputs.py | cut -d ' ' -f1)
VECTOR_SHA=$(sha256sum tools/atlas_vector_arithmetic.py | cut -d ' ' -f1)
SCALAR_SHA=$(sha256sum tools/atlas_tile_arithmetic.py | cut -d ' ' -f1)
SOURCE_SHA=$(sha256sum "$SOURCE_MANIFEST" | cut -d ' ' -f1)
python "$REFRESH/partition_resnet.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --out out/resnet-inventory
INVENTORY_SHA=$(sha256sum out/resnet-inventory/partition.json | cut -d ' ' -f1)
python "$REFRESH/export_full_host_plan.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --inventory out/resnet-inventory/partition.json \
  --inventory-sha256 "$INVENTORY_SHA" --tile-shape 32 32 32 --atlas-tools tools \
  --packing-provider-sha256 "$PACKING_SHA" --vector tools/atlas_vector_arithmetic.py \
  --vector-sha256 "$VECTOR_SHA" --scalar tools/atlas_tile_arithmetic.py \
  --scalar-sha256 "$SCALAR_SHA" --out out/resnet-provider
PLAN_SHA=$(sha256sum out/resnet-provider/host-plan.json | cut -d ' ' -f1)
python "$REFRESH/export_full_reference.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --host-plan out/resnet-provider/host-plan.json \
  --host-plan-sha256 "$PLAN_SHA" --atlas-tools tools \
  --packing-provider-sha256 "$PACKING_SHA" --vector tools/atlas_vector_arithmetic.py \
  --vector-sha256 "$VECTOR_SHA" --scalar tools/atlas_tile_arithmetic.py \
  --scalar-sha256 "$SCALAR_SHA" --out out/resnet-provider/reference
PROVIDER=out/resnet-provider
```

If a captured manifest was relocated, the exporters accept explicit repeatable
`--relocate OLD_PREFIX=NEW_PREFIX` mappings and continue checking asset hashes.
The reference exporter performs offline target arithmetic, not Core, RTL,
platform or task-quality execution. The selected model checkout must match the
revision checked by `run_model_tile.py`. Compiling the following templates uses
its assembler without constructing a Core or executing a program.

```sh
python tools/compile_resnet_templates.py --model-root "$MODEL" \
  --host-plan "$PROVIDER/host-plan.json" --out out/resnet-templates
python tools/generate_resnet_graph.py --plan "$PROVIDER/host-plan.json" \
  --templates out/resnet-templates/receipt.json --out out/resnet-graph
python tools/check_resnet_graph_replay.py --generated out/resnet-graph \
  --math runtime/atlas_resnet_graph_math.c --reference "$PROVIDER/reference" \
  --cc "$NATIVE_CC" --out out/resnet-replay
python tools/build_resnet_graph_host.py --generated out/resnet-graph \
  --replay-gate out/resnet-replay/graph-replay.json \
  --reference "$PROVIDER/reference" --templates out/resnet-templates/receipt.json \
  --rescale-table out/resnet-templates/rescale-words.bin --atlas-root . \
  --clang "$RISCV_CLANG" --lld "$LLD" --out out/resnet-host
```

The native replay checks all 54 C-produced activation streams and scales and the
1,000 final target logit words against hash-bound saved device replies. It
executes host C, not device instructions. The ELF builder snapshots the
graph/runtime and audits every embedded asset and ELF load segment. Its
RV64GC/LP64D startup enables FP, sets RNE, and supplies a 64 KiB stack, aligned
mailbox and high-address DMA arena. Linking does not execute the ELF.

The optional `check_resnet_graph_device.py --build out/resnet-host --reference
"$PROVIDER/reference" --cc "$NATIVE_CC" --out out/resnet-device-check` checks
the actual C callback and raw runtime against an MMIO mock, with controls for
sequence, capacity, scaling, patch corruption, guards, timeout ownership and
aliasing. This remains saved-reply conformance.

## Prepare and run the C54 RTL harness

C54 preparation additionally requires the reviewed, completed model-partition
fixture, including its parent receipt, all 54 child receipts and their
hash-bound raw artifacts. Supply the matching reference and fresh generated host
assets. The fixture must declare `passed_model_partition`, 54 ordered exact
contractions, unchanged producer/model/input bookends, and the same candidate
logits as the reference. This fixture is an external prerequisite; the checkout
does not generate it.

To relocate a fixture, pass explicit repeatable `--relocate
OLD_PREFIX=NEW_PREFIX` mappings. The preparation command hashes the original
receipt bytes before constructing a relocated view of absolute path values and
keys. It rejects unsafe paths and mapping collisions and verifies each
referenced file's original hash at its new location. Fresh host, reference and
template paths are supplied directly and are not rewritten.

To link, also supply an AtlasCore generated object directory and matching
Verilator headers. Preparation reconstructs every initial/expected DRAM image,
immutable program, DMA count and VMEM guard. It does not run a model or RTL
simulator.

```sh
python tools/prepare_resnet_graph_rtl.py --host out/resnet-host \
  --model-partition "$MODEL_PARTITION_FIXTURE/receipt.json" \
  --reference "$PROVIDER/reference" --templates out/resnet-templates/receipt.json \
  --rtl-object-dir "$RTL_OBJECTS" --verilator-include "$VERILATOR_INCLUDE" \
  --cc "$NATIVE_CC" --cxx "$NATIVE_CXX" --out out/resnet-rtl
# Explicit numerical execution, after allocating resources:
timeout 21600 out/resnet-rtl/resnet_graph_runner out/resnet-rtl/graph.plan 200000000
```

The harness uses the complete C graph, raw host runtime and Verilated AtlasCore
with one reset and 54 sequenced starts. Its declared synthetic TileLink
responder supplies backpressure and 43-cycle responses. It checks live inputs,
patched programs, DMA traffic, guards, markers, a quiet completion margin and
1,000 exact target logit words. Waveforms and DMA transaction logs are off by
default; `--vcd PATH` and `--dma-trace` are explicit diagnostic options.

The RTL harness measures device execution with a synthetic memory responder;
host CPU computation is outside that RTL clock. The [host-ISS
runner](host-iss.md) provides a separate RV64 execution path for a compatible
checksum-bound fixture bundle. The commands above do not produce that bundle
format. Exact target arithmetic and platform timing are separate from
task-quality accuracy. Task-quality admission and integrated platform execution
require independent evidence.

When a model receipt binds a separate producer source snapshot, pass it with
`--model-producer-tools`. Every named helper is checked against the unchanged
receipt digest; the current templates and all prepared launch bytes are still
compared to the recorded model payloads.

## EE290 SoC kernels

The frozen per-kernel network-model fixture supplies `kernel-*/model/{program.bin,initial-dram.bin,expected-bf16-tiles.bin}`. Build one independent ELF per kernel with a RISC-V GCC toolchain that supplies `htif_nano.specs` and `htif.ld`; the builder links the shared host runtime in place. Provide physical paths without symbolic links for all inputs and toolchain executables. LOADMEM places the initial device image at `0x100000000`; HTIF reports completion and the ELF checks every output word, both guards and the runtime sequence markers.

```sh
python tools/chipyard/resnet_kernels.py build --network-model "$NETWORK_MODEL" \
  --cc "$RISCV_CC" --out out/resnet-soc-kernels
# Use an already built EE290GenericIOSimConfig simulator; set its library/license environment separately.
export ATLAS_SOC_SIMULATOR="$EE290_SIMULATOR"
export ATLAS_DRAMSIM_INI="$DRAMSIM_INI"  # Enable +dramsim to reproduce the recorded cycle totals.
python tools/chipyard/resnet_kernels.py run --build-root out/resnet-soc-kernels \
  --out out/resnet-soc-runs --jobs 1
python tools/chipyard/resnet_kernels.py results --build-root out/resnet-soc-kernels \
  --runs-root out/resnet-soc-runs --out out/resnet-soc-results.json
```

The runner records logs, exit status and ELF identity with finite cycle and wall limits; select them with `--max-cycles` and `--wall-seconds`. Increase `--jobs` only after allocating simulator licenses and CPU resources. Results require a matching ELF identity and runner status in `run.json`, zero exit, no timeout and exactly one matching PASS for every built kernel; the complete fixture must report 54/54 exact. No simulator is built by these tools.

Measured on EE290GenericIOSimConfig with VCS: all 54 kernels are exact, totaling 200,874,497 Atlas cycles, about 3.2 times the standalone AtlasCore total of 63,480,310 because the SoC uses real DRAM. A full host run takes about 4.09 billion host instructions even without the startup BSS clear; at about 2,300 simulated cycles/second (measured on the segment runs) it is impractical in RTL simulation. Coverage combines per-kernel exactness, segment stitching on QEMU and short SoC segments; a full end-to-end run remains for FPGA or FireSim.

For a full host loaded by a loader that zero-fills every PT_LOAD memory tail, `build_resnet_graph_host.py --skip-bss-clear` omits the redundant startup clear. The default still clears BSS.

## Segmented EE290 host graph

Segment k executes the unchanged generated graph statements from the pack feeding launch k up to the pack feeding launch k+1. Segment 0 also copies the input; segment 53 adds the FC bias and writes the logits. All 286 statements and 54 launches are covered once. Each ELF links the shared host runtime, loads captured live-in tensors and DMA state, then checks its live-out tensors, Atlas output and runtime state. The final segment also compares all 1,000 logits with the full-host reference.

Use the complete full-host build above as `FULL_HOST`, the matching frozen network-model directory as `NETWORK_MODEL`, and its successful full-host/model-device receipt as `MODEL_EPOCH`. Capture verifies every staged DMA image and immutable program against the receipt before replaying model outputs. `--npu-model "$MODEL"` instead executes the live model device during capture. QEMU capture and stitching establish state and statement coverage; they do not measure SoC timing.

```sh
python tools/host_iss/segment_capture.py capture --elf "$FULL_HOST/host.elf" \
  --host "$FULL_HOST" --model-root "$NETWORK_MODEL" --epoch "$MODEL_EPOCH" \
  --qemu "$QEMU" --out out/resnet-segment-capture
python tools/chipyard/resnet_segments.py build --host "$FULL_HOST" \
  --capture out/resnet-segment-capture --out out/resnet-soc-segments \
  --clang "$RISCV_CLANG" --lld "$LLD" --start "$FULL_HOST/start.S" \
  --runtime "$FULL_HOST/runtime"
python tools/chipyard/resnet_segments.py stitch --host "$FULL_HOST" \
  --capture out/resnet-segment-capture --out out/resnet-soc-segments
# Check a segment in QEMU; repeat for indices 0..53 for complete host instruction coverage.
python tools/host_iss/segment_capture.py check \
  --elf out/resnet-soc-segments/seg-53/host.elf --host "$FULL_HOST" \
  --model-root "$NETWORK_MODEL" --epoch "$MODEL_EPOCH" --qemu "$QEMU" \
  --out out/resnet-segment-qemu/seg-53
# Use the same prebuilt simulator and environment as the per-kernel runs.
python tools/chipyard/run_sim.py --elf out/resnet-soc-segments/seg-53/host.elf \
  --out out/resnet-segment-runs/seg-53 --max-cycles 100000000000 --wall-seconds 604800
python tools/chipyard/resnet_segments.py results --host "$FULL_HOST" \
  --out out/resnet-soc-segments --runs out/resnet-segment-runs --segments 53
```

`build --segments K ...` builds a subset; stitching requires all 54. The builder binds graph text, embedded assets and runtime sources to the supplied full-host build. `results` requires the matching ELF identity in `run.json`, a zero process exit, no timeout or stop marker, exactly one successful final report, every statement marker and every expected region check. Interrupted or incomplete runs are not exact passes. Provide physical paths without symbolic links for all inputs and tools.

Recorded evidence: QEMU passed all 54 segment ELFs; stitching matched 69 producer-to-consumer tensor regions with every statement covered once. A live npu_model capture matched the replay capture in all 356 boundary regions, context fields and final logits.

Short SoC segment results on EE290GenericIOSimConfig with VCS: segments 52 and 53 pass with exact end state. Segment 52 (pack, launch 52, decode, affine, residual add, ReLU, average pool) matched tensor_174 (1,024 words) and the Atlas output (32,768 words) with zero mismatches in 49,851,169 CPU cycles and 2,516,093 Atlas cycles. Segment 53 (pack, launch 53, decode, FC bias, logits) matched the Atlas output (8,192 words) and all 1,000 logits against both the captured and the reference values with zero mismatches in 54,827,923 CPU cycles and 2,358,785 Atlas cycles. Runtime state (completion count, sequence, markers, scale and exponent) was correct in both. Wall time was about 6.4 and 7.1 hours at about 2,300 simulated cycles per second.

## Opt-in full-graph traffic reuse

`compile_resnet_templates.py --traffic-reuse` selects the qualified public generator dispatch for every declared contraction while preserving the plan, physical operand layout and arithmetic references. This remains compile-only preparation. The graph runtime admits at most 32,763 raw words, reserving five additional words for the sequenced completion wrapper inside the 32,768-word instruction memory; relocated programs are checked again after expansion.

Generate the graph and run the native captured-reply gate with the selected templates as above, then build the host with an explicit physical DRAM contract. For a generic QEMU UART profile, pass `--uart-kind ns16550 --uart-base 0x10000000`; a hardware UART profile produces a separately bound ELF. The physical DRAM options validate all load segments, allocated sections and stack space, and `--arena-base` relocates the program, runtime layout and linker reservation together.

```sh
python tools/compile_resnet_templates.py --traffic-reuse --model-root "$MODEL" \
  --host-plan "$PROVIDER/host-plan.json" --out out/resnet-reuse-templates
# Generate, replay-check and link using this template receipt and rescale table.
python tools/prepare_resnet_graph_bundle.py --host out/resnet-reuse-host \
  --templates out/resnet-reuse-templates/receipt.json --reference "$PROVIDER/reference" \
  --model-root "$MODEL" --out out/resnet-reuse-bundle
python tools/host_iss/run.py --workload resnet --elf out/resnet-reuse-bundle/host/host.elf \
  --bundle-root out/resnet-reuse-bundle --model-root "$MODEL" --qemu "$QEMU" \
  --wall-seconds 21600 --out out/resnet-reuse-live
```

The bundle preparer regenerates each scaled program from the current public dispatch, checks it against the host image, preserves all 1,000 expected logit words, and binds captured inputs, immutable weights, guards and expected replies with checksums. Its new kernel receipts use `atlas.resnet_kernel_preparation.v1` and `PREPARED_NOT_EXECUTED`; these are inputs for future execution, never successful numerical receipts. The live bridge accepts this preparation schema as well as legacy `passed_model_only` payloads, then executes every launch and checks its actual DMA counts, preserved bytes and output. Only a fresh successful whole-host run qualifies the reuse graph.
