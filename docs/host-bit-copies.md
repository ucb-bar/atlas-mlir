# Host BF16 bit copies

The selected Smol layer builder emits `runtime/smol_host_bits.h` for freestanding conversion without `memcpy` calls.

C union copies preserve FP32 representation and the existing BF16 rounding policy without a compiler intrinsic.

Widening signaling NaNs preserves sign and payload; narrowing sets the BF16 quiet bit.

Tests cover all 65,536 BF16 patterns, FP32 rounding boundaries and undefined symbols in a freestanding RV64 object.

Run `PYTHONPATH=tools:test python -m unittest test_host_check` with the configured RV64 compiler.
