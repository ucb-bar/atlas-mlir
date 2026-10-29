# NPU Model timing facts

Generate compiler facts from an npu_model source checkout and its default
configuration with one command:

```sh
python -B tools/atlas_model_facts.py --npu-model /path/to/npu_model \
  --out /path/to/new-facts
```

The output contains model-timing.json, model-overlap.json, facts.lock.json,
and a compact validation-report.json. The compiler reads the timing and
overlap files; the lock selects their hashes. Regeneration derives operation
ages, reservation releases, architecture, operands, frontend spacing, DMA
costs, weight windows, engine admission, replay edges, and overlap rules.
The supported source and configuration must pass the extractors' structural
checks. Unsupported control flow, geometry, operand mappings, or timing
expressions refuse with model-update guidance. Recognized latency and
capacity changes are extracted from the model rather than fitted to runs.

One native VPU row-sweep probe checks extracted port-event ages and drain.
It rejects a wrong timing fact. This is a representative extraction check,
not exhaustive numerical qualification. Extraction does not run separate
runtime, DMA, replay, admission, weight, or overlap validation suites and
does not publish receipts claiming that those suites ran.

The compiler retains configuration and semantic payload checks when loading
facts. It checks fact structure and scheduling obligations independently of
regeneration. It does not require producer manifests, native-run counters,
validation hashes, or qualification reports. Existing bundles should be
regenerated when updating the compiler.

The extracted events describe the supported model's isolated operations.
Overlap pairs and composition classes govern concurrent admission. DMA
completion remains dynamic and requires matching waits, channel ownership,
and the separate eight-command RTL ring rule. DMA curve costs are
conservative proposal bounds under drained, no-denial preconditions.
Scratchpad response latency has no independent model event; its compatibility
assumption is inactive when complete reservation releases are available.

Compile and select using the generated facts:

```sh
python -B tools/lower_atlas_program.py input.mlir \
  --output-dir selected --atlas-bin-dir build/bin \
  --provider model --npu-model /path/to/npu_model \
  --model-facts /path/to/new-facts
```

Omit --model-facts to regenerate. Candidate search and proposal policy are
compiler policy, defined in model-search-policy.json, rather than source
timing facts. Every production candidate still passes compiler verification
and native scoring through drained ECALL; winning words are the scored words.
Supplied expected regions provide the numerical oracle. The curve profile
uses the configured backend; fifo43 is a synthetic scoring target.

## Updating for NPU Model changes

The model is not an RTL timing proof. Structural checks can conservatively
refuse benign refactors, and the representative native check does not prove
arbitrary mixed streams or arithmetic equivalence. After extractor changes,
regenerate, run compiler tests, compare consumed timing fields, and rerun
representative selections with exact numerical checks.
