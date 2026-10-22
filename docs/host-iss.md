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
