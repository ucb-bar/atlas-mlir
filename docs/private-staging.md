# Keep private staged BF16 values in VMEM

Set `ATLAS_STAGING_STORAGE=vmem` when compiling to promote eligible private
edges of a staged BF16 device graph into invocation-local VMEM slots. The
default `dram` retains the previous staging schedule. The setting participates
in both native tuning and application materialization identities.

Merlin's existing semantic partition and liveness allocation identify each
value and its lifetime. Atlas uses that allocation to change the physical
transport: a private BF16 tile store/load becomes a typed `virtual_save` /
`virtual_restore`. Public inputs, constants and final outputs keep their DRAM
ABI. Arithmetic operations and their order are unchanged. This belongs in the
target backend; no model-specific recognition or new Merlin graph rewrite is
needed.

A DRAM slot can hold a private temporary early and a public result later. The
compiler marks individual transfers from their source values, rather than
promoting every use of an address. Promoted values share the original liveness
allocation, and native RAW/WAR/WAW scheduling preserves slot dependencies.

The candidate admits at most 96 BF16 slots, reserving at most 192 KiB of the
selected 240 KiB temporary pool and leaving space for ordinary spills. This is
a conservative selection limit, not a resource proof. Larger candidates retain
their DRAM transfers and record that decision. The native allocator and source,
buffer, initialization, timing and IMEM verifiers still check every accepted
program. Native failures are not ignored or retried as successful compilations.

The execution plan records `private_staging`, including the requested and
selected transport, original staging addresses, tile extent and transfer
counts. The existing host-visible arena is retained; unused private DRAM bytes
are not promised as reclaimed application storage. Mixed FP8/BF16 staging is
outside this option's current scope.

## Repeated immutable inputs

`ATLAS_STAGING_STORAGE=vmem-inputs` additionally caches repeatedly read input
and constant tiles. The checked source boundary identifies immutable full
BF16 tiles; any writer, partial transfer, dynamic address or address outside
that boundary is refused. Single-use tiles retain their original transfers.

Private intermediates take priority within the same 96-slot candidate budget.
The remaining slots cache inputs in descending read-count order, with address
as a deterministic tie-breaker. Each selected tile is loaded once and saved
in VMEM, then restored at its original uses. The prefetch block runs inside
every counted row iteration, after its guard. Nothing is assumed resident
between row groups or dispatches. Slot declarations alone remain in the entry
preheader.

The execution receipt records the candidate and selected addresses, available
slots, prefetches and replaced loads. Both compilation identities include the
actual storage policy, keeping `dram`, `vmem` and `vmem-inputs` distinct.

## Complete-layer results

| Identical fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 77,504,366 | 77,504,366 | 59 → 59 |
| SmolVLA vision block B=1, 1,024 tokens | 741,793,870 | 621,089,999 | 122 → 122 |

The matched baseline is [row dispatch budgeting](row-dispatch-budget.md), with
Nicolas's existing common-value elimination enabled for SmolVLA. Private staging
saves **120,703,871 cycles (16.27%)** across the complete vision block. Its
softmax falls from 180,321,836 to 59,617,965 cycles. This is the only changed
instruction image among the 122 launches; all other images and cycle counts
are identical. TinyLlama's short softmax fits the existing register schedule,
so all 59 instruction images and counts remain identical.

All 786,432 SmolVLA and 131,072 TinyLlama BF16 output words match the independent
original-FX Atlas-arithmetic references. Weights, inputs, selected arithmetic,
and the [SmolVLA vision-block calibration scope](dispatch-reduction.md) are
unchanged. This is not a full SmolVLA policy quality result.

Executed traffic accounting confirms 84,148,224 fewer DRAM read bytes and
75,497,472 fewer write bytes: **152.25 MiB** removed. SmolVLA's remaining device
traffic is 628,017,760 bytes. The same-work serial-launch bound is 467,346,412
cycles and useful MAC utilization is 1.39% of one MXU's peak. The bound is not a
predicted speedup; this transformation reduced the work itself.

Both runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL
numerics, `register_rows`, 64 MiB I/O and a 128,000,000-instruction execution
budget. They measure simulated device cycles, excluding host latency and
silicon timing. Retained artifacts under `perf-studies/smolvla-dispatch` are
`tinyllama-vmem-staging`, `application-vmem-staging`,
`rust-tinyllama-vmem-staging` and `rust-vmem-staging`.

## Immutable-input cache results

| Identical fixture | Private intermediates only | Plus immutable inputs | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 77,504,366 | 77,504,366 | 59 → 59 |
| SmolVLA vision block B=1, 1,024 tokens | 621,089,999 | 606,944,206 | 122 → 122 |

`vmem-inputs` saves another **14,145,793 cycles (2.28%)** and **24 MiB of
DRAM reads** in the complete SmolVLA block. Only its softmax image changes:
59,617,965 → 45,472,172 cycles. The other 121 images and their counts remain
identical. All 59 TinyLlama images remain identical. Every output word still
matches the same independent references used above.

SmolVLA now reads 441,597,952 bytes and writes 161,253,984 bytes. Its same-work
serial-launch bound is 450,307,928 cycles, with 1.43% useful MAC utilization
relative to one MXU. Runtime instruction count increases from 80,645,992 to
80,697,064 because cache initialization adds instructions; the reduction in
DRAM traffic outweighs that cost. These remain device-only simulation results.

Retained artifacts are `application-readonly-staging`,
`tinyllama-readonly-staging-v2`, `rust-readonly-staging` and
`rust-tinyllama-readonly-staging-v2`. The first TinyLlama compilation hit an
intermittent native buffer-verifier crash during an unchanged contraction.
The failing input was retained; three direct replays produced identical valid
images, and a fresh complete compilation and simulation passed. This feature
does not claim to fix that compiler crash.

## Validation

Generated-C/Rust tests cover plain, masked and guarded softmax, f32 and BF16
outputs, partial rows and columns, multiple row groups, fully masked rows and
invalid inputs late in the tensor. Shapes include 65×65, 33×129 and 33×1024.
Every output word is compared with the independent selected-arithmetic
reference; inputs and unrelated memory remain immutable. Native checks still
reject uninitialized slot reads and an insufficient temporary-memory budget.

Source tests distinguish private and escaping values at the same physical
address, reject unknown/partial/misaligned transfers, preserve transfers when
the candidate limit is exceeded, and separate cached compilations by storage
policy. Existing randomized native/Rust slot-scheduling tests cover RAW, WAR
and WAW hazards with independent transfers.

For immutable input caching, directed tests also exercise zero and partial
remaining cache capacity, deterministic selection among repeated reads,
immutable-prefix violations, unexpected device writers and dynamic addresses.
Generated-C/Rust tests run both `vmem` and `vmem-inputs`, including varying rows
to detect a cache incorrectly retained across iterations. A zero-slot fallback
executes the original DRAM path and still matches every reference word.
