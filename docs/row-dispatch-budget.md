# Softmax batching follows the selected execution budget

Row-softmax previously limited each launch to 512 column tiles even when the
application selected a larger execution-instruction budget. It now scales that
candidate envelope with the selected budget. Complete rows stay together, and
the group count divides the total so every invocation uses the same checked
program. RMSNorm retains its existing whole-row-loop strategy.

This is an Atlas scheduling decision. Merlin already owns the optional
application budget; no new model recognizer or generic compiler setting is
needed. The example runtime in Merlin accepts the new versioned publication
while checking the same exact mutable and readonly memory windows.

The envelope is a candidate heuristic, not a resource proof. Each compilation
still checks its actual bounded dynamic instruction count, complete I/O frame,
native allocation, timing and physical IMEM capacity. Increasing the execution
budget does not enlarge the 128 KiB instruction memory. A candidate that fails
these checks is refused. This change does not introduce a retry search for a
smaller row batch.

## Publication and compatibility

The default 1,000,000-instruction strategy retains the v1 row-plan and execution
records. Other budgets use v2 and record `dispatch_instruction_budget`.
Verification reconstructs the saved strategy rather than silently rebatching
under the current invocation. This also preserves old v1 artifacts compiled
with larger execution bounds but the original fixed batching envelope.
Canonical materialization rebinding carries the saved budget forward and
checks the unchanged physical image. The current invocation must still admit
the saved instruction bound.

## Complete-layer measurements

| Matched fixture | Before cycles / dispatches | After cycles / dispatches |
| --- | ---: | ---: |
| TinyLlama B=1, 64 tokens | 77,504,366 / 59 | 77,504,366 / 59 |
| SmolVLA vision block B=1, 1,024 tokens | 741,794,882 / 145 | 741,793,870 / 122 |

SmolVLA's softmax uses one launch instead of 24. This saves 23 launches and
1,012 simulated device cycles; it does not remove the softmax's memory traffic.
Host dispatch latency is not included in these numbers. TinyLlama's shorter
softmax already used one launch. All 131,072 TinyLlama and 786,432 SmolVLA BF16
output words match their independent original-FX Atlas-arithmetic references.

The SmolVLA baseline enables Nicolas's existing Merlin
`--application-common-values` pass. It removes two identical static Q/K/V
input quantizers, saving another 3,740,952 cycles and two launches relative to
the 745,535,834 / 147 [panel-readout result](panel-readout.md). This is an
evaluation of an existing pass, not a new implementation. The retained
TinyLlama graph has no additional common values for that pass to remove.

Both runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL
numerics and `register_rows`, with 64 MiB I/O and a 128,000,000-instruction
execution budget. The independently calibrated SmolVLA vision-block scope in
[dispatch reduction](dispatch-reduction.md) still applies. These are simulation
results, not chip measurements.

The refreshed executed-traffic census records 83,692,352 device DRAM bytes for
TinyLlama and 787,663,456 for SmolVLA. Useful MAC utilization is 3.57% and 1.17%
of one MXU's peak, respectively; both-MXU utilization is half of those values.
The same-work serial-launch lower bounds are 57,575,542 and 575,434,290 cycles.
These bounds are not speedup predictions. Removing intermediate traffic changes
the work itself and requires separate exact-arithmetic checks.

Retained applications are `tinyllama-row-budget` and `application-row-budget`
under `perf-studies/smolvla-dispatch`. Full comparison and traffic receipts are
in `rust-tinyllama-row-budget-v2` and `rust-row-budget-v2`. The first runs refused
the unknown v2 execution schema before softmax; the example runner was then
updated and both applications rerun from the start with memory checks enabled.

## Validation

Directed generated-C/Rust tests cover complete and partial rows and columns,
plain, masked and guarded softmax, f32 and BF16 outputs, fully masked rows, and
invalid values late in the input. Invalid inputs are rejected before device
I/O. Separate native tests reject an insufficient instruction budget and a
complete frame that exceeds the I/O window. Both v1 and v2 materializations
rebind under a larger invocation, preserve their images, and reject changed
semantics or modified metadata.

The Merlin example tests verify identical v1/v2 output and scratch windows,
readonly memory protection, invalid budgets, unknown versions and out-of-bounds
frames. No memory-write checks are disabled for the new schema.
