# Additional model compilation optimizations

These changes are part of Atlas PR #50 and Merlin PR #63.

| Additional feature | Owner | Selection |
| --- | --- | --- |
| Share identical pure values, preserving provider groups and public outputs | Merlin | `--application-common-values` |
| Compose adjacent exact dense reshapes | Merlin | `--application-compose-reshapes` |
| Bound invariant operand residency by caller-supplied storage and expansion limits | Atlas | Target-local API |
| Remove per-element loop/address overhead in exact casts and sign flips | Atlas | `--exact-float-unroll` (off by default) |
| Reuse quantized activation rows across output columns, with ordered K arithmetic and exponent restoration | Atlas MXU1 | `ATLAS_QUANTIZED_OPERAND_REUSE=row-cache` |

The Merlin flags apply to `native-compile --application` and the corresponding
source verification. Both source transforms are off by default and preserve
replayable receipts. MXU1 row caching is also off by default and uses the
target-local operand planner. The default ordered path and the cast changes
remain usable without that planner.

The existing MXU0 A/W cache, PACK repair, compiler materialization cache, counted
host layer loop, program residency, and buffer-overlap checks are not reimplemented
here. The MXU1 extension uses existing scheduling, accumulator-retention,
allocation, instruction-bound, and publication checks. It adds no weight cache
or cross-launch residency.

The cast change preserves exact source arithmetic and complete input validation.
BF16 add/sub and their exceptional scalar fallback are unchanged. Expanded unary
images cost more instruction bytes; the default keeps compact kernels regardless of the image cap.
No model instruction or memory limit is raised.

See [exact float regions](exact-float-device.md) and
[quantized operand reuse](quantized-operand-reuse.md) for contracts and tests.

## Validation status

The implementation commits are ready for code review. The historical full-layer
replay audit remains unresolved after reproducing host-side corruption
independently of the NPU. Do not treat the provisional timing below as an error-free benchmark.

The historical Atlas source passed 53 focused tests and 19 subtests, all 28 native
CTests, and 14 additional quantized integration tests with 15 subtests under
`row-cache`. Two MXU0 integration tests also passed with their required legacy
model pin (20 subtests). The historical Merlin source passed 194 focused tests and 14 subtests, structure,
lint, dependency, source-layout, artifact-layout and documentation checks.
The access-registry bwrap test and 18 wiring-gate findings reproduce unchanged on
the unmodified Merlin #63 base; they are not claimed as passing.

The layer audit freshly compiled the first TinyLlama decoder layer at batch 1,
64 tokens, hidden width 2048 and MLP width 5632. It replayed source verification
for all 90 regions and checked that all 705 final runtime-source, parameter and
program files matched the measured build. Weights were unchanged. The NPU model
was unmodified at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with RTL numerics,
random initial state and poisoned VMEM. This covers one decoder layer, not all 22.

| Successful-replay totals (provisional) | PR #50 / #63 baseline | Additional optimizations |
| --- | ---: | ---: |
| Device cycles | 334,277,079 | 226,725,080 |
| Exact float-region cycles | 159,631,502 | 103,051,174 |
| Quantized matrix-region cycles | 143,462,190 | 95,266,326 |
| Device launches | 601 | 586 |

The provisional reduction is 107,551,999 cycles (32.17%). Cycles exclude host
copies/arithmetic, program uploads and MMIO. The final output has 131,072 BF16
words, SHA256 `9de10cb80864257a8b300b533eaafb0685d2f4f040d39010d6e1b4f7c29f7bd9`.

Two complete replay attempts each failed one distinct state of an unchanged
BF16-add image. The first reported an immutable-memory change, but its subsequent
byte-difference diagnostic found no changed bytes outside the permitted range;
128 diagnostic replays passed. A stricter replay with immutable snapshots,
independent protected-byte hashes and byte comparisons passed 581 of 582 states;
its failed state passed in isolation. The table uses that isolated successful
replay and retains both original failures. The baseline also has an unresolved
replay anomaly. These observations do not establish a root cause or prove that
the failures are harmless. A mixed-state diagnostic replay also failed its input
snapshot SHA256 assertion before starting the simulator (state 307, repeat 1).
An independent Python-standard-library reproducer then performed 2,975 snapshot
checks without importing Torch, Merlin, Atlas or the NPU model. One check captured
a single changed byte at offset 8,265,811 (`0xdd` became `0xdf`) in a 64 MiB
snapshot. Both retained copies consistently hashed to the changed value; a fresh
decode of the original file matched the expected hash. This establishes that
host-side corruption can occur independently of these compiler changes. The
specific host fault and the earlier replay failures are not individually resolved.

The mixed-state diagnostics retained 1,187 results
(one pre-simulation snapshot-hash failure), and a separate mixed-kernel diagnostic
retained 86 passing results. These partial runs are not presented as completed
suites. No simulator, arithmetic, memory guard or verification check was relaxed
to obtain a passing result.

Historical evidence consists of results, original progress/error logs, final
compilation equivalence, and focused-test and build logs. That local evidence,
generated model data, and binary artifacts are not included in the source commits.
