# RV64 host execution with a model-backed Atlas device

`tools/host_iss/run.py` executes a complete fixed 54-kernel ResNet host ELF with
an instruction-model Atlas device. Its CPU-only tests cover admission and
protocol handling; they do not execute the whole ELF. A fresh execution receipt
is required to qualify each supplied ELF and payload set. Task-quality accuracy
requires separate evidence.

The host ELF runs as RV64GC/LP64D instructions under generic QEMU `virt`. The
bridge intercepts the two actual load/store instructions inside
`atlas_host_direct_read32` and `atlas_host_direct_write32`. QEMU stops while the
single `npu_model.Core` executes a kernel. Core remains idle while QEMU executes
host arithmetic, memory operations, fences and completion checks. This is ISA
and model integration, with no Chipyard, TileLink, FPGA or platform timing
claim. Tiny and Smol block contracts are not supported by this ResNet-only entry
point.

## Inputs and dependencies

Use an extracted bundle with schema `atlas.full_resnet_bundle.v3` and
`SHA256SUMS`, `provenance.json`, `host/build.json`, `host/host.elf`, and all
`kernels/00..53/model/` payloads. The reader checks every listed payload,
rejects duplicate/unsafe paths and bundle symlinks, and checks the supplied ELF
against the bundle hash. Checksums bind content; they do not establish who
published a bundle. Supply an independently authenticated expected
archive/checksum digest. The checkout does not create or distribute this bundle.
The graph build in [resnet-full-host.md](resnet-full-host.md) produces an ELF,
templates and reference artifacts; it does not produce this runner's bundle
format.

The pinned model revision is `9e6313bddecb8ba936206eb347f5cd4e5335282c`,
selected through the existing `run_model_tile.configure_model` API. Install that
model's Python dependencies in a compatible environment; its `pyproject.toml`
requires Python >=3.14. Torch runs with one intra/inter-op thread. The bridge
records and rechecks imported model sources, revision, tracked-source state and
memory curve through `model_binding`. Git and `qemu-system-riscv64` must be
available. The harness itself uses the Python standard library and the
repository's `atlas_path_guard`, `run_model_tile`, and `atlas_tile_arithmetic`
helpers. It needs neither an RTL simulator nor a native Atlas archive, compiler,
debugger CLI, reservation ledger or workload controller. Ordinary owned-QEMU
cleanup and absolute wall deadlines are local to the runner.

## Run

Select tools with environment variables and use repository-relative artifact
paths. The supplied ELF must match the bundle ELF checksum; the bundle supplies
captured inputs and references:

```sh
"$MODEL_PYTHON" -B tools/host_iss/run.py \
  --elf out/resnet-host/host.elf \
  --bundle-root out/resnet-bundle \
  --model-root "$NPU_MODEL_ROOT" \
  --qemu "$QEMU_RISCV64" \
  --out out/resnet-iss \
  --wall-seconds 21500
```

The finite wall limit includes input verification and execution; each device
start also has a 20,000,000-cycle cap. Python `-O` is rejected: strict runtime
checks require assertions. QEMU is configured for one TCG CPU, 8 GiB guest RAM,
no BIOS or peripheral console, and a local all-stop GDB socket. Guest RAM
includes the external Atlas staging mapping at `0x100000000`; this allocation is
not an FPGA memory contract. MMIO instruction memory is `0x20000..0x3ffff`;
control/status is based at `0x40000` with START at `+24`. Runtime assembly
replaces the final raw `ecall` with the exact six-word DBG1-to-DBG0 completion
epilogue and checks the reconstructed bytes.

One device instance preserves registers, VMEM, accumulators, DRAM and counters
between ordered starts. Each START checks actual host-written IMEM and the whole
actual guest input buffer; model execution checks DMA totals, output, guards,
drain state, flags and completion sequence, then copies the actual full device
buffer back into guest memory. The runner checks all 54 starts, mailbox state,
sequence ledger, zero trap record, `tohost == 1`, and every final logit byte.

`receipt.json` contains input/source/tool digests and a fresh verdict. Outputs
include 54 bounded kernel summaries, mailbox/logit readbacks and QEMU
diagnostics. Per-cycle, DMA and verbose RSP traces are disabled. A successful
run reports `PASS_RV64_HOST_WITH_MODEL_DEVICE`; failures retain the first
failing boundary. Existing input receipts remain unchanged.

## CPU-only checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test -p test_host_iss.py -v
```

These tests use synthetic payloads and an in-memory duplex transport. They cover
checksum/path and source bookends, exact epilogue reconstruction, actual MMIO
instruction
decoding, CSR/IMEM refusals, completion/trap/sequence/logit failures and RSP
framing, acknowledgments, checksum and absolute-deadline handling. They launch
neither QEMU nor a model Core and provide no numerical execution verdict.
