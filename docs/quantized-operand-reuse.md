# Quantized activation-row reuse

Set `ATLAS_QUANTIZED_OPERAND_REUSE=row-cache` before compiling the
already-quantized workflow to retain an FP8 activation row across output-column
tiles. The default, `ordered`, keeps the existing counted schedule. Unknown
values are errors. The option selects a transport schedule; it does not select
a different precision policy or recode weights.

Atlas's `quantized_operand_reuse.activation_residency` decides whether a
complete immutable operand fits the target's capacity and expansion limits and
has multiple consumers. The Atlas provider establishes invariance across the N
loop and supplies tile sizes and limits. The capacity bound lives in the Atlas provider and needs no companion Merlin module.

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
A row-cache request is recorded in the execution plan, which is unchanged for
the default ordered policy, and replayed during sizing,
binding, source verification, and C export; those replays and the generated runtime
do not read this environment variable.

This extension is restricted to the quantized MXU1 path. It adds no MXU0 operand
cache, PACK repair, compiler cache, or program residency implementation; those
are provided by the companion upstream work.

## Bounded prefixes for wider reductions

`ATLAS_QUANTIZED_OPERAND_REUSE=prefix-cache` also admits rows wider than 64 K32
tiles. It retains the first 64 tiles (64 KiB) across output columns and streams
the remaining tiles through the existing ordered counted loop. For shorter rows
it selects the existing full-row schedule. A single consumer still uses the
ordered schedule. The default and `row-cache` policies are unchanged.

The prefix is refreshed for every M tile row and batch. The accumulator crosses
the resident/streamed boundary as the same ordered BF16 value; exponent restoration
runs only after the complete K reduction. No reduction terms are reordered and
no extra accumulator precision is retained. Only the bounded prefix is expanded,
so code size does not grow linearly with the entire K dimension.

For a partially cached row the execution plan records `operand_reuse=prefix-cache`
and `operand_reuse_cached_tiles`. Replay derives and checks that count from the
source shape and requested policy before emitting code. Sizing, source binding
and C export continue to use the recorded policy independently of the environment.
All allocation, memory and instruction-budget checks remain in force.

The avoided activation reads are
`B * Mc * (Nc - 1) * min(Kc, 64) * 1024` bytes. Weight reads and output stores are
unchanged. This is a bounded prefix cache, not a complete blocked-panel schedule
or weight reuse across token rows.

`test_quantized_prefix_reuse.py` covers the first streamed tile, counted groups
and tails, K=5632, M/N padding, batches, varied scales, exponent rejection, full
protected-memory equality, exact DMA-byte savings, instruction bounds and
native sizing/binding/export with a changed environment policy.

## Wider bounded activation rows

`ATLAS_QUANTIZED_OPERAND_REUSE=wide-cache` admits up to 192 resident K32
activation tiles (192 KiB). This covers the whole K=5632 down-projection row;
longer reductions cache a bounded prefix and stream the remainder in order.
The original `ordered`, `row-cache`, and `prefix-cache` policies are unchanged.

The 192-tile cap is a construction limit, not a larger hardware budget. Native
allocation, instruction-count, timing and program-size checks still apply to
every candidate. The existing 240 KiB spill frame is unchanged. The explicit
wide policy and exact cache count are recorded and checked when replaying an
execution plan. Generated code expands only the capped resident prefix.

`test_quantized_wide_reuse.py` covers full rows, the first streamed tile after
192 resident tiles, M/N padding, multiple batches, nonuniform scales, refusal,
protected memory, exact read savings and native export/rebinding. Twenty-four
legacy virtual programs were also checked byte-identical across MXU0/MXU1,
cached/streamed schedules and K unrolling.

## Share weight panels across token rows

`ATLAS_QUANTIZED_OPERAND_REUSE=paired-cache` processes pairs of M32 tile rows.
For each panel of up to 32 K32 tiles it loads both activation rows once, then
loads each weight column once and uses it for both rows. The resident operand
slots occupy at most 96 KiB: 64 KiB of activations and 32 KiB of weights. Normal
allocation still accounts for restoration, spills and staging separately.

Each output follows the original ascending K sequence. A completed BF16 partial
sum is stored in the existing output buffer between panels and loaded as the
next panel's seed. The first panel resets the accumulator without reading that
buffer. After the final panel, the original restoration hook applies scales and
checks admission. No partial sum escapes through the C interface: the binding
checks the shared status before publishing its final dense result. This uses
the existing output permissions and allocates no extra DRAM buffer.

The candidate requires one batch, an even number of M32 tiles, at least two N32
tiles, and 32 through 192 K32 tiles. Quantized multiply epilogues retain the wide
activation schedule. Short reductions, odd row counts, larger K and multiple
batches also fall back to `wide-cache`; no input is rejected merely because it
cannot use paired panels. This shape admission avoids the small-K case where
partial-result traffic costs more than the reused weights. It is not a promise
of lower cycles for every admitted shape.

The requested policy, selected schedule, two M tiles and 32 K tiles per panel
are recorded in the execution plan and checked during replay. Native resource,
timing, memory and instruction proofs remain mandatory, including on fallback.

For admitted shapes, compared with caching the complete activation row, weight
reads decrease by `Mc/2 * Nc * Kc * 1024` bytes. Let `P = ceil(Kc/32)`: partial
sum handling adds `P * Mc * Nc * 2048` bytes of reads and the same number of
writes. The net traffic benefit must include both costs. Activation reads and
exponent-page reads are unchanged.

`test_quantized_panel_reuse.py` checks fallback, plan tampering, canonical native
publication/rebinding, padded rows/columns, multiple row pairs, incomplete K
panels, nonuniform exponents, late refusal, protected memory and exact traffic.
The optimized pipeline also executes against an ordered reference on the NPU
model. Both requested policies remain opt-in; `ordered` is still the default.
