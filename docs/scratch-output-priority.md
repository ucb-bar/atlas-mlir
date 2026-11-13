# Output priority through scratch dependencies

The native Atlas scheduler now propagates first-output priority through its
existing dependency graph, including scratch RAW, WAR and WAW edges. Previously
the heuristic followed only tensor SSA operands, so a cache save could lose its
connection to the output served by a later restore. Among operations with equal
earliest start times, work needed by the earliest output is preferred.

This changes ready-operation ranking, not dependencies or legality. Scratch
alias checks, allocation, admission, machine verification and timing checks
remain mandatory. This belongs in Atlas: the heuristic depends on native
scratch operations and the target scheduler. Merlin's semantic graph and
ownership proofs are unchanged.

## Complete-layer measurements

| Same fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 82,772,819 | 81,247,163 | 59 → 59 |
| SmolVLA vision block B=1, 1,024 tokens | 766,046,001 | 766,023,249 | 147 → 147 |

TinyLlama saves 1,525,656 cycles (1.84%); SmolVLA saves 22,752 cycles
(0.003%). All 131,072 and 786,432 BF16 output words respectively match the
independent original-FX Atlas-arithmetic reference. These incremental results
include the preceding [activation boundary folding](pointwise-boundary-folding.md)
and shape-selected operand panels. They do not imply that every model gains
equally from this scheduling heuristic.

Measurements use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU,
RTL numerics, `register_rows`, the optimized native pipeline and `panel-cache`.
Limits remain 64 MiB I/O, 128,000,000 executed instructions and 128 KiB IMEM.
These are simulated device cycles, excluding host costs; the existing
[SmolVLA calibration scope](dispatch-reduction.md) applies.

## Validation

The scheduler/operand-panel regression suite passed 50 tests and 383 subtests.
The directed priority regression fails with the old heuristic and passes with
the new one. Randomized schedules check scratch hazards, complete DRAM contents
and immutable bytes; rejection cases retain overlapping-slot and exponent
checks. Both complete layers passed their independent reference comparisons.

The release tools recompiled and verified all 65 distinct virtual programs from
the two executed applications. Every emitted image was byte-identical to the
image run through Rust. Evidence is retained in
`out/smolvla-audit/release-slot-priority-images-v2.json` in the SmolVLA audit
worktree. Full applications and comparison receipts are
`tinyllama-slot-priority` / `rust-tinyllama-slot-priority` and
`application-slot-priority` / `rust-slot-priority` under
`perf-studies/smolvla-dispatch`.

An earlier parallel release-image replay crashed in the compiler; the fresh
serial replay passed. A separate four-request compilation also encountered a
native verifier crash before a later attempt reached an instruction-memory
capacity refusal. The crashes remain unexplained. This scheduler change does
not claim to fix them, and compiler or verification failures are not accepted
as successful builds.
