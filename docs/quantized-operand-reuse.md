# Quantized activation-row reuse

Set `ATLAS_QUANTIZED_OPERAND_REUSE=row-cache` before compiling the
already-quantized workflow to retain an FP8 activation row across output-column
tiles. The default, `ordered`, keeps the existing counted schedule. Unknown
values are errors. The option selects a transport schedule; it does not select
a different precision policy or recode weights.

Merlin's `semantic_compiler.operand_reuse.plan_operand_reuse` decides whether a
complete immutable operand fits the caller's capacity and expansion limits and
has multiple consumers. Its byte accounting is independent of target geometry.
The Atlas provider establishes invariance across the N loop and supplies tile
sizes and limits. The provider requires the Merlin change
`perf(compiler): plan bounded invariant operand reuse` on the companion
`perf/model-compilation-followups` branch based on Merlin PR #63.

The initial target policy admits at most 64 K32 tiles / 64 KiB of activation
cache and requires at least two output-column tiles. This bounds static code
expansion and leaves the remaining selected VMEM spill frame for restoration
and ordinary allocation. A larger row or a single consumer retains the original
schedule. Normal instruction-memory, dynamic-instruction, allocation, memory
initialization, DMA, timing and source-publication checks still decide whether
the emitted candidate is admissible; the capacity estimate does not bypass them.

For each batch and M tile row, the emitter loads each activation tile once and
saves it in a private FP8 VMEM slot before entering N. Every output column restores
those bits and streams its own weights, staged in groups of four. The complete
K body is expanded with static slot identities. Each output retains its original
ascending K32 sequence, MXU1 arithmetic, BF16 accumulation boundaries, and final
exponent restoration. Existing checked accumulator retention removes redundant
readout/reload pairs. Cache storage is refreshed for every row and batch; no
state is assumed to survive a launch.

Source buffers, physical DRAM layout, generated C transport, exponent admission,
sticky status, and atomic host publication are unchanged. Both shared and
per-batch weights use the same mechanism. VMEM cache contents never escape as
logical output.

For `B` batches and tile counts `(Mc, Nc, Kc)`, activation traffic changes from
`B * Mc * Nc * Kc * 1024` to `B * Mc * Kc * 1024` bytes. Weight traffic is
unchanged. This is a traffic reduction, not a prediction of cycle savings.

`test_quantized_operand_reuse.py` compares the original and cached programs on
the unchanged cycle model with randomized initial state and poisoned VMEM. It
checks full DRAM equality, immutable surrounding memory, exact saved DMA bytes,
shared/per-batch weights, signed and cap-448 codes, uneven M/N/K tails, K=1,
varied exponents and rejected exponents. Removing a cache writer must fail the
standard memory proofs. Separate tests cover resource refusals, native export,
the optional multiply epilogue and its status behavior.

The policy participates in the application materialization identity. Counted
image sizing also hashes the complete emitted virtual source, so changing the
policy cannot reuse an ordered image as a cached one or vice versa.

This extension is restricted to the quantized MXU1 path. It adds no MXU0 operand
cache, PACK repair, compiler cache, or program residency implementation; those
are provided by the companion upstream work.
