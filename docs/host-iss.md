# RV64 host execution with a model-backed Atlas device

`tools/host_iss/run.py` executes a complete RV64 host ELF under QEMU with a live instruction-model Atlas device. It admits the fixed 54-kernel ResNet50 graph, the TinyLlama first decoder block with 128-token prefill and one teacher-forced decode step, or the selected SmolVLA vision-language layer. Each supplied ELF and payload set requires a fresh successful execution receipt. CPU-only admission and protocol tests do not qualify numerical execution or task accuracy.

The host runs RV64GC/LP64D instructions under generic QEMU `virt`. The bridge intercepts the actual load/store instructions inside `atlas_host_direct_read32` and `atlas_host_direct_write32`. QEMU stops while a single `npu_model.Core` executes a kernel. Core remains idle while QEMU executes host arithmetic, memory operations, fences and completion checks. This is ISA and model integration, with no Chipyard, TileLink, FPGA or platform timing claim.

## Inputs and dependencies

ResNet uses an extracted bundle with schema `atlas.full_resnet_bundle.v3`, `SHA256SUMS`, `provenance.json`, `host/build.json`, `host/host.elf`, and all `kernels/00..53/model/` payloads. The graph build described in `resnet-full-host.md` produces an ELF, templates and reference artifacts; it does not produce this runner's bundle format.

Tiny and Smol use schema `atlas.block_bundle.v1`. Their provenance declares `workload` (`tiny` or `smol`), `kernel_count` (24 or seven), ordered `kernels`, `contract` and `contract_sha256`. Every launch descriptor names checksum-listed `source`, `original`, `initial` and `expected` members, with `dram_base`, `output_offset` and `requested_dma`. The payloads contain the original assembly, original program words, complete initial DRAM image and physical expected output. Requested DMA binds load and store bytes; store bytes must equal the physical output extent. The fixed staging address is `0x100000000`.

Block bundles also contain the unchanged original `host/build.json` and `host/host.elf`. The workload contract binds their identity to the captured launch-data contract, fixed geometry, embedded programs, weights and final references. Tiny uses the retained 24-panel first-block contract; Smol uses `atlas.smol_selected_layer1.epoch3_launch_data.v1` and checksum-listed `data/*.bin` assets. Workload-specific build status and source bookends must pass before QEMU starts. The reader never follows original absolute fixture paths recorded in those contracts. These are external fixed-input fixtures; this checkout does not create, distribute or authenticate their original captures.

The reader checks every listed payload, rejects duplicate or unsafe paths and bundle symlinks, and checks the supplied ELF against the bundle hash. Checksums bind content; supply an independently authenticated expected archive or checksum digest. They do not establish who published a bundle.

The pinned model revision is `9e6313bddecb8ba936206eb347f5cd4e5335282c`, selected through `run_model_tile.configure_model`. Install that model's Python dependencies in a compatible environment; its `pyproject.toml` requires Python >=3.14. Torch uses one intra/inter-op thread. The bridge requires the default `curve` memory backend measured on the EE290 SoC and records its parameters and curve digest. `fifo43` describes standalone RTL and is not the whole-host qualification backend.

Git and `qemu-system-riscv64` must be available. The bridge records and rechecks imported model sources, revision, tracked-source state, runtime layout, workload contract and runner helpers. The harness uses the Python standard library plus the repository tooling and the selected model dependencies. It needs no RTL simulator or native Atlas archive. Owned-QEMU cleanup and absolute wall deadlines are local to the runner.

## Run

Select tools with environment variables and supply physical artifact paths. The ELF must match the checksum-listed bundle ELF; the bundle supplies captured inputs and references. Use an immutable source checkout during a run because successful receipts require unchanged source bookends.

```sh
"$MODEL_PYTHON" -B tools/host_iss/run.py \
  --workload resnet --elf "$RESNET_HOST/host.elf" \
  --bundle-root "$RESNET_BUNDLE" --model-root "$NPU_MODEL_ROOT" \
  --qemu "$QEMU_RISCV64" --out out/resnet-iss --wall-seconds 21500
"$MODEL_PYTHON" -B tools/host_iss/run.py \
  --workload tiny --elf "$TINY_BUNDLE/host/host.elf" \
  --bundle-root "$TINY_BUNDLE" --model-root "$NPU_MODEL_ROOT" \
  --qemu "$QEMU_RISCV64" --out out/tiny-iss --wall-seconds 21600
"$MODEL_PYTHON" -B tools/host_iss/run.py \
  --workload smol --elf "$SMOL_BUNDLE/host/host.elf" \
  --bundle-root "$SMOL_BUNDLE" --model-root "$NPU_MODEL_ROOT" \
  --qemu "$QEMU_RISCV64" --out out/smol-iss --wall-seconds 21600
```

`--workload` defaults to `resnet`. The finite wall limit includes input verification and execution. Default per-start model cycle caps are 20000000 for ResNet, 200000000 for Tiny and 100000000 for Smol; `--kernel-cycle-cap` can set a different positive bound without changing exactness checks. Python `-O` is rejected because runtime checks require assertions.

QEMU uses one TCG CPU, 8 GiB guest RAM, no BIOS or peripheral console, and a local all-stop GDB socket. Guest RAM includes the external Atlas staging mapping at `0x100000000`; this allocation is not an FPGA memory contract. MMIO instruction memory is `0x20000..0x3ffff`; control/status is based at `0x40000` with START at `+24`. Runtime assembly replaces the final raw `ecall` with the exact six-word DBG1-to-DBG0 completion epilogue and checks reconstructed bytes.

One device instance preserves registers, VMEM, accumulators, DRAM and counters between ordered starts. Each START checks actual host-written IMEM and the entire actual guest input buffer. Model execution checks DMA totals, every output byte, preserved inputs and guards, drain state, flags and completion sequence, then copies the complete actual device buffer back into guest memory. Captured successor activations are never supplied in place of host computations.

ResNet completion checks all 54 starts, mailbox state, sequence ledger, zero trap record, `tohost == 1`, and every final logit byte. Tiny checks all 24 starts, its 128-byte mailbox and trap loop, both prefill/decode outputs, and both steps' key and value caches. Smol checks all seven starts, its 120-byte mailbox, completed/cache-valid state, selected-layer output and both caches. The fixed block contracts do not establish full Tiny inference, a full Smol policy or application accuracy.

`receipt.json` contains input, source and tool digests and a fresh verdict. Outputs include bounded per-kernel summaries, actual mailbox/output/cache readbacks and QEMU diagnostics. Per-cycle, DMA and verbose RSP traces are disabled. A successful run reports `PASS_RV64_HOST_WITH_MODEL_DEVICE`; failures retain the first failing boundary. Existing input receipts remain unchanged.

## CPU-only checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test -p 'test_host_iss*.py' -v
```

These checks use synthetic payloads and an in-memory duplex transport. They cover checksum/path and source bookends, exact epilogue reconstruction, actual MMIO instruction decoding, workload geometry and ELF bindings, CSR/IMEM refusals, live-buffer/DMA/output/guard failures, completion/trap/sequence/cache failures, and RSP framing, acknowledgments, checksum and deadlines. They launch neither QEMU nor a model Core and provide no numerical execution verdict.

## Independent EE290 SoC kernel checks

`tools/chipyard/block_kernels.py` builds independent LOADMEM ELFs from a hash-bound manifest without invoking Chipyard's build. The manifest has schema `atlas.block_soc_kernels.v1` and a nonempty `programs` list. Each entry declares a unique `id`, `program_bin`, `program_sha256`, `initial_dram_bin`, `initial_dram_sha256`, `expected_dram_bin`, `expected_dram_sha256`, aligned `dram_base`, and disjoint `output_ranges` with integer `offset` and `size`. Relative payload paths resolve beneath the manifest. Both DRAM images cover the full region; expected bytes outside the output ranges must equal the initial bytes. Supply the complete admitted inventory for a workload, 24 kernels for Tiny or seven for Smol, when claiming full block kernel coverage.

The optional `build --fast-check` flag checks stable output/input/guard spans after completion and ownership return. It uses aligned volatile word XORs and an OR reduction, byte prefixes/tails and a failure-only byte scan. A disappearing mismatch fails closed; fast logs count bytes while default logs retain mixed word/byte units. The flag defaults off and binds `runtime/atlas_host_check.h` into the receipt. `ATLAS_KERNEL_CHECK_TIMING` measures only the check interval; interpreting its guest counter depends on the platform.

The builder links the current shared raw runtime, preserves the original program bytes, binds the completion epilogue separately, and audits the actual ELF load mappings and embedded assets. The C harness checks every output byte, preserved inputs and guards, one launch, drained completion and sequence markers. These are independent kernel checks; host graph arithmetic requires separate whole-host or segment execution.

```sh
python3 tools/chipyard/block_kernels.py build \
  --manifest "$BLOCK_SOC_MANIFEST" --cc "$RISCV_CC" --out out/block-soc-build
# Set the simulator's library and license environment before running.
python3 tools/chipyard/block_kernels.py run \
  --build-root out/block-soc-build --out out/block-soc-runs --jobs 1 \
  --simulator "$EE290_SIMULATOR" --dramsim-ini "$DRAMSIM_INI" \
  --max-cycles 2000000000 --wall-seconds 172800
python3 tools/chipyard/block_kernels.py results \
  --build-root out/block-soc-build --runs-root out/block-soc-runs \
  --out out/block-soc-results.json
```

Allocate simulator licenses and CPU resources before increasing `--jobs`, which is bounded at 18 per invocation; enforce the aggregate limit across invocations separately. `--resume` reuses only completed exact runs with matching ELF/status evidence and refuses incomplete or failed existing directories. A successful report requires every listed kernel, zero process exits, no timeout, matching ELF hashes, exact output/preservation checks and completion markers. Interrupted runs are not exact results. Use the curve model and real SoC cycle totals before standalone RTL timing when comparing platforms.

The device bridge accepts an optional Python API `launch_range=(start, end)` for a nonempty global launch interval with an exclusive end. The workload segment harness must admit the entry and full live-state inventory before constructing it, and must restore the real host invocation sequence to the global start cursor. The bridge prepares only the selected programs, requires global `DBG1/DBG0` sequences, and emits only actually executed global-index kernel receipts; it never fills earlier entries. `kernel_count` remains the whole-model count while `launch_count` is the selected interval size. Omitting the range preserves ordinary continuous execution. This API is shared infrastructure and does not itself provide a segment-capable ELF, boundary producer or segmented qualification.
