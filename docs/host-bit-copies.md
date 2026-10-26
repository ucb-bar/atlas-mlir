# Host BF16 bit copies

The selected Smol layer builder emits the repository's `runtime/smol_host_bits.h`. Its C union copies preserve the existing FP32 representation and BF16 rounding policy without a compiler intrinsic. Signaling NaNs preserve sign and payload when widened; narrowing sets the BF16 quiet bit, as before. The other 14 emitted layer headers have no four-byte `memcpy` conversion sites.

With the retained RV64 Clang recipe and `-ffreestanding -fno-builtin`, the helpers use `fmv.w.x` and `fmv.x.w` rather than external `memcpy` calls. The complete layer ELF is byte-identical to the previously measured intrinsic variant. Fresh QEMU instruction counts are 5,500,736,825 before and 1,554,369,422 after, a 71.743% reduction. Both runs validate all seven staged inputs/programs, complete physical DRAM checks, final output and key/value caches exactly.

These are host instruction counts with captured device outputs and intercepted HTIF console writes, not Rocket cycles, NPU execution or whole-system timing. The new tests check all 65,536 BF16 patterns, rounding boundaries for every upper FP32 half, and the absence of undefined symbols in a freestanding RV64 helper object. All 122 existing gates and both added checks pass without skips.
