# Measured candidate selection proposal

`tools/select_atlas_candidates.py` runs a bounded implementation shortlist through compilation and exact target measurement. It accepts a JSON manifest and a fresh output directory; all compiler and target details live in explicit argv adapters. It uses only the Python standard library, invokes no shell, and installs no dependencies.

The producer supplies semantic admission, candidate source hashes, an immutable proof-receipt hash and an estimated cost. The driver verifies those identities; it does not prove an equality or infer a source domain. An optional extractor command writes `candidates.json` before measurements begin. The baseline is always retained, and the remaining budget is filled deterministically by estimated cost and identifier. This is a bounded shortlist, not exact top-k enumeration or a global optimum guarantee.

## Adapter protocol

The manifest has schema `atlas.measured-selection.v1`, a baseline ID, positive `max_candidates`, timezone-qualified `deadline_utc`, optional coordinated `stop_files`, and uniquely named targets. A target describes its backend and memory setting; it is copied unchanged into every measurement receipt. `kind` is `model` or `rtl` for the corresponding ranking groups. Sources and output paths must be direct paths without symlink traversal.

| Field | Substitutions | Required output |
| --- | --- | --- |
| `extract_argv` (optional) | `{output}` | `candidates.json`, with a `candidates` array |
| `compile_argv` | `{source}`, `{output}`, `{candidate_id}` | `receipt.json` with `status: PASS`, `program`, `program_sha256`, `source_sha256` |
| `measure_argv` | `{program}`, `{target}`, `{output}`, `{candidate_id}` | `receipt.json` with `status: PASS`, `program_sha256`, exact `target`, invocation `cycles` and `correctness` |

Each candidate has `id`, `source`, `source_sha256`, `admitted`, `estimated_cost`, and `proof: {path, sha256}`. Numerical admission remains the producer/target validator's responsibility. Each measurement must explicitly qualify `correctness.outputs`, `correctness.oracle`, and `correctness.guards` as true. The adapter owns unchanged input/oracle images, resource preconditions, instruction decode checks, complete requested memory guards and target-specific cycle-window qualification. Cold and retained invocations stay separate in `cycles`; ranking uses the first invocation only. Reused evidence must be labeled and bound to the complete program, inputs, guards, target configuration and lifecycle, rather than inferred from matching cycle counts.

A reported byte, oracle or guard difference writes `CORRECTNESS_STOP.json` and stops selection. Any missing receipt, nonzero measurement exit, incomplete qualification or identity mismatch also stops the workflow immediately for diagnosis; it is not silently treated as a rejected implementation. An infrastructure error is recorded as `ERROR`, not relabeled as a numerical difference. Compiler rejections remain visible and do not qualify a candidate. Coordinated stops and the deadline are polled while an adapter runs; cancellation terminates that adapter's own process group.

## Comparing targets

The report gives the cold-cycle winner on each target, componentwise domination, worst ratio against the baseline, and worst normalized regret `max_target(candidate_cycles / best_measured_cycles - 1)`. The minimax candidate minimizes that last quantity on the declared target grid. It also lists candidates within five percent of the measured winner everywhere. Model-only ranking is separate; RTL-only manifests are supported. Incomplete candidates cannot win a robustness ranking, and an incomplete experiment remains explicitly incomplete.

These quantities describe only the measured shortlist and memory policies. A synthetic responder is not an FPGA measurement, and two memory backends with the same unloaded latency need not have the same bandwidth or arbitration. Summed operation costs rank candidates for compilation; scheduled target execution determines the measured winner.

Run focused controller tests with `python -m unittest discover -s test -p test_candidate_selection.py`. They cover rank reversals, incomplete evidence, binary binding, explicit guard failures, missing-receipt termination and path confinement. A real integration supplies target-owned adapters; the experimental combined-program adapter and evidence are session artifacts rather than installed compiler functionality.
