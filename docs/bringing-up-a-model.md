# Bringing up a model on Atlas

For Agustin, Jeremy, and Loren bringing up Gemma2 or pi0, start with the TinyLlama and SmolVLA examples below. [atlas-kernels](https://github.com/ucb-bar/atlas-kernels) owns kernel generators, shared references, and model coverage; atlas-mlir pins it at `third_party/atlas-kernels` and owns the compiler backend, host runtime, model builders, and QEMU tooling. [Merlin comparisons](https://github.com/ucb-bar/Merlin/tree/sota/bringup-docs/examples/atlas/comparisons) owns captures and hand-written model C providers; the TinyLlama full graph also has sources in atlas-mlir. The separately pinned npu_model supplies instruction arithmetic and device execution. Keep checkpoint, input, source, program, toolchain, and ELF hashes with each result.

## Two routes and their connection

Agustin's compiler route on `feat/atlas-model-compilation` takes PyTorch through Model2MLIR, `merlin-targetgen stage-capture`, Merlin's semantic compiler, Atlas virtual MLIR, `atlas-opt`, and `atlas-emit --program-json`, and generates C dispatch. See the branch's [memory and device lowering](https://github.com/ucb-bar/Merlin/blob/feat/atlas-model-compilation/docs/guides/model_memory_and_device_lowering.md), [C dispatch](https://github.com/ucb-bar/Merlin/blob/feat/atlas-model-compilation/docs/guides/dispatch_c.md), and [Atlas binding](https://github.com/ucb-bar/atlas-mlir/blob/feat/atlas-model-compilation/docs/native-merlin-binding.md). The established evidence at Merlin `62ab24eda` and atlas-mlir `ae8581f` covers small static authored workloads, with largest proven single program M×K×N = 128×512×128. It does not establish compilation of these complete models.

The model route in PR #26 and Merlin PR #55 uses PyTorch `torch.export` for the op and shape inventory, Merlin captures for real values, atlas-kernels Python generators for real device shapes, per-launch npu_model exactness, a hand-written host graph, a host ELF, and QEMU execution. FireSim packaging is a subsequent static step. The generators emit hand-scheduled assembly; translating that assembly to machine IR does not make their schedules or the host graph compiler-generated.

Track A's probe translated every kit device program to machine IR, ran `atlas-opt --verify-atlas-machine-stream`, encoded through `atlas-emit`, and compared the words with npu_model's assembler: 1,589 of 1,589 programs and 1,487,694 words matched. TODO(track-a): replace this probe-only statement with the final backend wrapper command, required tool configuration, receipt location, and integrated build/check status from DONE-A.md. The checked-out bringup builders still call the assembler directly until that integration lands.

The intended connection is to select these library kernels behind the compiler's provider adapter, by op, shape, and arithmetic policy. On Agustin's branch the seam is `AtlasApplicationAdapter.plan`, `compile_region`, and `export_region` in `python/atlas_mlir_native/application.py`. Region matching, a named precision policy, weight-window handling, and a merged host runtime remain integration work. His resident window is at most 1 MiB; a TinyLlama 2048×5632 FP8 weight matrix contains 11,534,336 bytes before metadata.

This follows the Radiance separation: RTL in radiance, kernels and references in radiance-kernels, compiler in radiance-mlir, and experiments in Merlin. Merlin's [Radiance materializer](https://github.com/ucb-bar/Merlin/blob/sota/bringup-docs/examples/radiance/phase1/kernel_library/materialize.py) exports a pinned kernel commit with `git archive`; its adjacent `selection.yaml` maps op, dtype, and shape rules. Atlas currently uses an atlas-mlir submodule instead. The coverage mapping records where the qualified builds put work; it is not that compiler selector. The bringup tooling stays in atlas-mlir, and `examples/atlas/comparisons` keeps its current path.

## Worked sequence

Commands below run from atlas-mlir unless a different repository is named. Variables name physical absolute paths to independent checkouts, authenticated external inputs, or fresh outputs; use the interpreter and pinned packages required by each linked runbook. `MODEL47` means unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`; `MODEL9` means `9e6313bddecb8ba936206eb347f5cd4e5335282c`. Native PyTorch capture, instruction execution, and host replay can require different Python environments. Initialize `third_party/atlas-kernels` as described in the [build instructions](../README.md#build).

### 1. Inventory the executed ops and shapes

In atlas-kernels, follow [coverage regeneration](https://github.com/ucb-bar/atlas-kernels/blob/track-c/coverage/coverage/COVERAGE.md#regenerate) with the exact checkpoint and input used by the capture:

```sh
PYTHONPATH="$TINY_PACKAGES" "$TINY_PYTHON" coverage/export_ops.py tinyllama --checkpoint "$TINY_CHECKPOINT" \
  --corpus "$TINY_CORPUS" --out coverage/out/tinyllama --fixture coverage/fixture/tinyllama.json
"$SMOL_PYTHON" coverage/export_ops.py smolvla --policy "$SMOL_POLICY" --backbone "$SMOL_BACKBONE" \
  --observation "$SMOL_OBSERVATION" --out coverage/out/smolvla --fixture coverage/fixture/smolvla.json
python3 coverage/build_table.py
```

`torch.export.export(strict=False)` yields 4,540 ATen records for TinyLlama and 18,923 for SmolVLA; committed fixtures aggregate these into 227 and 392 rows. It checks structure, shapes, and dtypes with fake tensors, not numerical values. The coverage guide names the checkpoint revisions, packages, input pins, and optional `--capture-receipt` cross-check. For a new model, add its loader and placement rules rather than editing the fixture by hand.

### 2. Capture real values

From Merlin, the existing native capture commands are:

```sh
python examples/atlas/comparisons/capture_tinyllama_native.py \
  --checkpoint "$CHECKPOINT" --corpus "$CORPUS" --full-query --out "$TINY_NATIVE"
python examples/atlas/comparisons/capture_smolvla_policy.py \
  --source-receipt "$DYNAMIC_RECEIPT" --observation-input "$OBSERVATIONS" --out "$SMOL_NATIVE"
```

The [Merlin capture README](https://github.com/ucb-bar/Merlin/blob/sota/bringup-docs/examples/atlas/comparisons/README.md) distinguishes TinyLlama's first-layer Q capture from `capture_tinyllama_full.py`, which covers all decoder layers and greedy feedback. Use Merlin `d1-capture-c-v2.py` with its projection worker and atlas-mlir `tools/tiny_full/attention_v2_worker.py` for the current C-graph route, following [TinyLlama reproduction](../tools/tiny_full/REPRODUCING.md). Inspect `python "$MERLIN/examples/atlas/comparisons/d1-capture-c-v2.py" --help` for its external parameter and scale inputs. Preserve the full staged operands, references, ordered invocation ledger, and source hashes.

For SmolVLA's instruction-backed C-graph capture, follow [Smol attention capture](smol_attention_capture.md), including its immutable compiled-observation bundle and baseline providers. Its maintained command is `tools/capture_smolvla_attention.py` with `--prepare --execute`; `python tools/capture_smolvla_attention.py --help` lists those inputs. This regenerates host math, attention, and the complete denoising trajectory. Capture creates references; qualification must still check them.

### 3. Map every op to a kernel or host operation

Use atlas-kernels `coverage/mapping.yaml` and `coverage/build_table.py` for both models. From atlas-kernels, run `python3 coverage/build_table.py --check-cites "$ATLAS" "$MERLIN" "$KERNELS"` to check citations against the integrated repositories. Start kernel additions with [Adding a kernel](https://github.com/ucb-bar/atlas-kernels/blob/feat/kernel-library/README.md#adding-a-kernel), the family's `kernel.yaml`, and its `gen.py`. Reuse needs matching shape, layout, dtype, scale policy, padding, and arithmetic order, not only an op name.

TinyLlama maps 465 logical projection calls, 66 attention launches, and 132 RMSNorm launches. SmolVLA maps 1,421 logical projection calls, 200 attention launches, and 24 fused fc1 calls. Logical calls and physical panels differ: SmolVLA has 2,715 projection panels. TinyLlama's current kit has 1,038 launches; earlier projection-plus-attention documentation describes 906 before the 132 norm launches. Read [TinyLlama program checking](tiny-fast-build.md) and [Smol placement and admission](smol-full-host.md) for the device plan details.

The plans remain hand-written: `tools/smol_full_plan.py`, `tools/smol_attention_assets.py`, and `tools/tiny_full/capture_boundaries.py` encode model geometry and ordering. Track C's `test/test_model_coverage.py` checks their logical geometry and ordering against the export fixtures, including mutation controls. That test is on Track C's atlas-mlir branch pending integration into this checkout; the proof does not generate C graphs or prove their arithmetic.

### 4. Check kernels at the actual shapes

In atlas-kernels run `make test PYTHON="$MODEL_PYTHON" NPU_MODEL="$MODEL47"`. The library's [family documentation](https://github.com/ucb-bar/atlas-kernels/tree/feat/kernel-library/kernels) identifies each generator, tests, and golden references. TinyLlama's launch executor is described in [attention execution](../tools/tiny_full/ATTENTION_EXECUTION.md); SmolVLA's small real-model check is `tools/smol_capture_smoke.py`, with the full command in [Smol capture](smol_attention_capture.md#capture-and-small-original-model-check).

Compare complete output bytes and preservation regions with npu_model RTL numerics, using the real padded shapes, scales, and DMA layout. Separately measure error against PyTorch or an FP32 reference. Track C's per-family standalone numeric references and benchmark results are still in progress in the inspected handoff; do not treat the coverage FLOP share as a latency result. Functional `ArchState` execution in `atlas_execution.py` does not measure Core cycles.

### 5. Export programs and qualify every launch

TinyLlama's [export runbook](tiny-attention-export.md) gives this complete template, with the actual platform DRAM range substituted:

```sh
python tools/tiny_full/export_capture.py \
  --capture "$TINY_CAPTURE" --capture-sha256 "$TINY_CAPTURE_SHA256" \
  --model-root "$MODEL47" --out "$TINY_ASSETS" \
  --dram-base 0x80000000 --dram-bytes 0x80000000 --qualify-launches
python tools/tiny_full/check_programs.py --assets "$TINY_ASSETS" \
  --contract-sha256 "$TINY_CONTRACT_SHA256" --model-root "$MODEL47"
```

The illustrated DRAM range is 2 GiB beginning at `0x80000000`; the arena and complete ELF must fit the selected target. The [current TinyLlama reproduction runbook](../tools/tiny_full/REPRODUCING.md) also uses `tools/tiny_full/d1-v2-qualify.py` for final-program qualification. Keep its output under the runbook's required qualification root, and audit saved records before reuse. `check_programs.py` regenerates sources and checks encoded words; it does not rerun numerical execution.

For SmolVLA, follow [export and original-model backfill](smol_attention_capture.md):

```sh
python tools/export_smol_policy_kernels.py --native-capture \
  --replay "$SMOL_CAPTURE/run/replay.json" --attention-capture "$SMOL_CAPTURE/run/manifest.json" \
  --model-root "$MODEL47" --out "$SMOL_ASSETS"
python tools/smol_capture_backfill.py --python "$MODEL_PYTHON" --model-root "$MODEL47" \
  --ledger "$SMOL_CAPTURE/run/device-events.jsonl" --out "$SMOL_AUDIT" --workers 16
```

A full Smol gate covers 2,915 events, including 2,715 projections and 200 attention launches. It binds original-model execution, source and program identity, staged inputs, complete final arenas, preservation, and DMA. Reply reuse must retain its narrower declared scope. TODO(track-a): insert the final explicit backend wrapper around these encode/check commands so the emitted words come from `atlas-emit`, retain the assembler cross-check, and record the actual verifier tier for every distinct program.

### 6. Adapt the host graph and boundaries

For TinyLlama, atlas-mlir `tools/tiny_full/tiny_full_model.c` supplies full-model orchestration, `tools/tiny_full/tiny_full_projection.c` supplies projection transport, and `tools/tiny_full/tinyllama_main.c` drives the attention host. Merlin supplies the arithmetic providers; its `tiny_block.c` is the separate block graph. Use `tools/tiny_full/capture_boundaries.py` as described in [Tiny reference boundaries](full-model-contracts.md#tiny-reference-boundaries); `python tools/tiny_full/capture_boundaries.py --help` lists the capture, gate, provider, compiler, and checksum inputs. It checks the production C graph with captured replies and prepares semantic segment boundaries.

For SmolVLA, adapt Merlin `smol_vision_graph.c`, `smol_policy_graph.c`, and the `smol_host_*.c` arithmetic providers. `tools/capture_smolvla_attention.py --prepare --execute` produces the phase reference required by `tools/build_smol_full_host.py`; follow [Smol host admission](smol-full-host.md). The host performs the operations placed on it by coverage, including norms, RoPE, cache handling, and Euler updates where applicable. These C graphs are written by hand, and their references must be regenerated when the graph or precision changes.

### 7. Build default and fast ELFs

For TinyLlama, use `tools/tiny_full/build_attention.py` with assets, the qualified receipt and its hash, Merlin comparisons, platform support, and the RISC-V toolchain. The [default build](tiny-attention-build.md) gives the complete command; `python tools/tiny_full/build_attention.py --help` lists the arguments. Then follow the [fast build](tiny-fast-build.md):

```sh
python tools/tiny_full/build_fast.py --assets "$TINY_ASSETS" --model-root "$MODEL47" \
  --default "$TINY_DEFAULT" --platform-root "$PLATFORM_ROOT" \
  --toolchain-bin "$RISCV_BIN" --out "$TINY_FAST"
```

For SmolVLA, `tools/build_smol_full_host.py` creates the standalone bundle, `tools/check_smol_full_bundle.py` audits it, and `tools/smol_sims/package.py` plus `tools/smol_sims/build-smolvla.sh` create SIMS builds. Use the complete commands and explicit toolchain/provider inputs in [Smol capture and packaging](smol_attention_capture.md#export-and-standalone-build). For the fusion-capable fast route, use [Smol fast builds](smol-fast-build.md):

```sh
python tools/generate_smol_fast_assets.py --bundle "$SMOL_BUNDLE" \
  --elf "$SMOL_DEFAULT_ELF" --out "$SMOL_FAST_ASSETS"
python tools/build_smol_fast.py --help
```

The fast builder consumes the admitted default CMake recipes and generated assets. Fast programs read immutable weights in place; their addresses bind them to that ELF layout. A successful build or static audit establishes neither guest execution nor a numerical pass. Preserve the exact source, toolchain, generated files, receipts, and ELF hash together.

### 8. Replay real host code in QEMU and test corruption controls

Follow [host ISS](host-iss.md) with the separate `MODEL9` environment and the bundle matching the exact ELF:

```sh
"$REPLAY_PYTHON" -B tools/host_iss/run.py --workload tiny-full \
  --elf "$TINY_BUNDLE/host/host.elf" --bundle-root "$TINY_BUNDLE" \
  --model-root "$MODEL9" --qemu "$QEMU_RISCV64" --out "$TINY_ISS" --wall-seconds "$TINY_WALL_SECONDS"
"$REPLAY_PYTHON" -B tools/host_iss/run.py --workload smol-full \
  --elf "$SMOL_BUNDLE/host/host.elf" --bundle-root "$SMOL_BUNDLE" \
  --model-root "$MODEL9" --qemu "$QEMU_RISCV64" --out "$SMOL_ISS" --wall-seconds "$SMOL_WALL_SECONDS"
```

A successful continuous replay records `PASS_RV64_HOST_WITH_MODEL_DEVICE`; runner support alone is not a completed replay. [Full-model contracts](full-model-contracts.md) explain the alternative segment routes, `tools/host_iss/run_tiny_segment.py` and `tools/host_iss/run_smol_segment.py`. Each restores an admitted boundary into a fresh guest/device and compares live-out bytes. Every ordered link is required before claiming complete segmented qualification.

For both models, inspect the segment runner's `--help`, run a valid segment, and repeat into a fresh output with `--corrupt-region REGION --corrupt-offset BYTE` using a region declared by that bundle. Tiny admits `hidden_0` or `hidden_1`; Smol validates its chain's region names. Require the corrupted run to fail. Retain the first failing boundary, launch ledger, actual readbacks, UART, and source hashes. Match the ELF's UART profile; a SiFive image needs the explicit `--sifive-uart-bridge` described in the runbook.

### 9. Package for FireSim

Use `tools/tiny_full/build_host.py --help` for TinyLlama's full-host packaging route and `tools/build_smol_full_host.py --help` for SmolVLA. [Full-model contracts](full-model-contracts.md) document their `--target firesim`, DRAM, arena, and UART arguments. These are separate build profiles with their own required inputs; do not append this flag to `build_attention.py` or assume an existing fast SIMS ELF has been qualified for the new target.

Both builders call `tools/chipyard/firesim_target.py` to emit `firesim-contract.json` and `marshal.json`. The packager records `PASS_STATIC_ONLY` and `PROPOSED_NOT_ELABORATED_OR_BUILT`. It supplies no hardware configuration, bitstream, or FireSim execution result. Check the platform memory and console contract before running the exact packaged ELF on an available target.

## What the evidence proves

| Check | Claim and limit |
| --- | --- |
| Export and coverage | Every recorded op/shape is placed; exported structure agrees with the hand-written device plans. No numerical proof. |
| Per-launch original47 qualification | Bit-exact against unmodified npu_model `47b8599` with RTL numerics at the captured shapes, scales, inputs, and program words, including full-arena preservation and requested DMA. No claim for arbitrary inputs or shapes. |
| Machine stream verifier and encoding | Valid instruction fields, state chain, delay slots, and branch targets; backend words match the assembler. No inference that timing or whole-model compilation passed. |
| Timing verifier | TODO(track-a): insert final tier counts, all diagnosed rules, and the limitations of fail-fast timing verification from DONE-A.md. The probe's timing acceptance is narrower than its stream acceptance. |
| Guest replay | Actual RV64 host instructions and device starts produce the admitted buffers for the identified ELF and input; segmented evidence must say segmented. No SoC interconnect or FPGA timing proof. |
| PyTorch comparison | The end-to-end outputs were checked against PyTorch for the fixed examples; they differ because the Atlas arithmetic policy differs. See [task accuracy](task-accuracy.md). |
| ELF identity and packaging | Hash and byte identity preserve an artifact; static packaging checks placement and metadata. Neither executes the artifact. |

The fixed-input PyTorch comparison gives TinyLlama tokens `[278, 3711, 5652]` versus Atlas `[278, 14263, 658]`, and SmolVLA action relative L2 error of 20.4%. These are accuracy measurements on those inputs, separate from bit-exact device correctness. The [task accuracy doc](task-accuracy.md) also labels its multi-input precision experiments as offline CPU emulation.

## Pins, timing, and remaining work

`third_party/atlas-kernels/lib/atlas_execution.py` enforces original47 for functional per-launch arithmetic, as described in [attention execution](../tools/tiny_full/ATTENTION_EXECUTION.md). The second pin, `9e6313bd`, is used by `tools/run_model_tile.py` and `tools/host_iss/model_device.py` for the persistent `npu_model.Core` behind QEMU MMIO. It uses `numerics="rtl"` and the `curve` DMA memory backend. `tools/host_iss/segment_boundary.py` requires that revision, backend, and curve digest for segment chains. These pins support different gates; a replay on the second pin does not replace qualification on original47.

Core cycle counts are npu_model model cycles. QEMU host cycles, model device cycles, standalone RTL cycles, and FireSim target cycles measure different systems and must be reported separately. In particular, functional executor instruction counts and coverage FLOPs are not device latency. The compiler route's small-kernel standalone RTL evidence does not establish full-model RTL or FireSim timing.

The host graphs and device plan files are still hand-written, although Track C checks the plan geometry against the real export. New observations and inputs need regenerated authenticated capture and boundary providers. Source-hash checks can reject historical receipts after source movement or edits; explicit relocation preserves hashes and does not authorize changed bytes. The compiler branch's `atlas_host` API has `ensure_loaded`, `invalidate_program`, `reset_completed`, and `set_clock`, while this bringup runtime has raw loading and launch ranges. That divergence must be reconciled before combining the routes.

TODO(track-a): confirm final integration gates, ELF identities, backend selection without silent fallback, timing-tier evidence, and any attention halt/DELAY follow-up in DONE-A.md. Do not promote a probe result to an integrated build claim.

For Gemma2, begin by checking whether projection, RMSNorm, RoPE, gated epilogues, and attention families fit the exported layouts and arithmetic. Check attention masking, scaling, soft-capping, cache geometry, and normalization explicitly; the TinyLlama example does not prove them. For pi0, inventory vision, text, action, attention, and denoising/control work independently before borrowing SmolVLA families. A new op or incompatible layout belongs in the coverage work list until a kernel or explicit host implementation and its checks exist.

Merlin's [held-out workload policy](https://github.com/ucb-bar/Merlin/blob/sota/bringup-docs/examples/workloads/README.md) reserves TinyLlama, SmolVLA, and ResNet50 for headline validation. Keep these bringup captures, frequencies, and results out of capsule derivation and performance selection. Agree on the provider integration and experiment location before treating this hand-built coverage as compiler selection evidence.
