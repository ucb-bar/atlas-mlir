# Full-model host contracts

Full ResNet uses `atlas.full_resnet_bundle.v3`; Tiny and Smol use `atlas.full_bundle.v1`. Bundles bind the ELF, build, capture, programs, weights, inputs, and expected outputs by checksum. Preparation receipts are not numerical passes.

The three full-host builders accept `--arena-base`, `--dram-base`, `--dram-bytes`, and `--target firesim`. For VCU118 use `--arena-base 0x90000000 --dram-base 0x80000000 --dram-bytes 0x80000000`; Tiny also needs `--rodata-after-arena`. Fit includes load segments, BSS, stack, and arena. FireSim software packaging can instead admit 16 GiB with `--dram-bytes 0x400000000`; a package does not supply the target configuration or establish hardware execution.

Full hosts report `ATLAS_KERNEL_BEGIN`, `ATLAS_KERNEL_PROGRESS`, and a final `ATLAS_KERNEL_PASS` or `ATLAS_KERNEL_FAIL`, alongside mailbox/tohost. FAIL names the failed check. Native QEMU uses NS16550 at `0x10000000`. An admitted SiFive board profile at `0x10020000` requires explicit `--sifive-uart-bridge`; the bridge checks real guest MMIO and initialization, with no board or baud-timing qualification. See [result output](host-result.md).

Numerical runs require pinned Model `9e6313bddecb8ba936206eb347f5cd4e5335282c`, backend `curve`, and the recorded curve hash. `run.py` checks actual staged inputs, programs, outputs, guards, DMA, drained completion, and final host state. A changed ELF needs its own execution evidence.

Segment bundles use `atlas.segment_chain.v1`. Each link restores its complete workload-admitted semantic state into a fresh guest/device, executes real host code, and compares every live-out byte to the next captured boundary. `run_tiny_segment.py` and `run_smol_segment.py` produce exact per-kernel ledgers and boundary receipts; `SegmentChain.verify_receipts` requires every ordered link before summing measured cycles. This is segmented qualification, not continuous execution or hardware timing. Native reference controls do not replace device execution.

Tiny full and segment entries share arithmetic but produce different ELFs; identify both explicitly. Smol's default selector executes the full policy, while an admitted phase request selects a restored segment. Reproducing the current Smol host consumes the authenticated archived boundary reference described in [Smol host inputs](smol-full-host.md).
