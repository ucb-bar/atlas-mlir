# Full-host result output

Full-host builders use the shared `runtime/atlas_result.h` and generated UART sink. The default board profile is SiFive at `0x10020000`; use `--uart-kind ns16550 --uart-base 0x10000000` for native QEMU. The image must match the target UART. Baud configuration is preserved unless `--uart-divisor` is explicit; no clock or baud estimate is embedded.

Programs print BEGIN, monotonically increasing PROGRESS with completed/expected launch counts, and PASS or FAIL through actual UART MMIO. FAIL includes the check, observed value, and expected value, and remains sticky. Mailbox/tohost completion remains available separately. Names and check labels must be nonempty whitespace-free strings. A UART PASS alone does not establish numerical correctness.

The generic runner captures `uart.log`. Native NS16550 uses QEMU serial output; an admitted SiFive image requires `--sifive-uart-bridge` to intercept and validate actual guest accesses. Positive and forced-failure guest diagnostics qualify functional output only, not board wiring, baud timing, accelerator arithmetic, or platform timing. See [full-model execution](full-model-contracts.md) for numerical and segmented qualification.
