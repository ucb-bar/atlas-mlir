# Model timing

The opt-in `model` provider consumes derived isolated operation windows and reservation releases; missing coverage, unsupported configuration and stale evidence fail closed. `builtin` remains the default.

Run `python -B tools/atlas_model_facts.py --npu-model MODEL --out NEW_FACTS` with the selected model environment to derive and validate facts. Select their SHA-256 using `--select-atlas-model-evidence="op-timing=FILE op-timing-sha256=SHA dma=wait"` before timing with `provider=model`.

Derivation reads the selected model's `DefaultHardwareConfig`; successful derivation does not imply compiler support for a changed configuration. Compilation accepts only the reviewed default configuration: 32×32 geometry and 64 MREGs; register/weight/CSR and VMEM/DRAM sizes and bases; fetch width; DMA channels, beat/transfer/in-flight limits; VMEM bank/bus and off-chip link geometry and pacing; RTL numerics, deterministic initialization and fatal runtime errors; the exact execution-unit classes and `IsaSpec`; and the curve backend with only `curves=ee290sim_vcs_probe.json` and null latency/beat overrides. Compilation and overlap evidence require the lock's configuration identity.

Pair and chain overlap use a supplied simulation-measured catalog pinned to model revision `9e6313b` and the locked configuration, plus handwritten capacity, port, weight/accumulator and alias rules. These tools project/replay the supplied catalog; this repository cannot regenerate its measurements. Other revisions have no overlap evidence and must use isolated lifetimes. Re-projection is allowed only for unchanged timing/model content, and binds the supplied catalog to the exact new facts digest. Wall-clock validation times and tool hashes make that digest change even when timing content is unchanged.

The eight-slot DMA command ring is an RTL contract absent from the model. Under `model`, launch n waits for a still-pending launch n-8 before reusing its slot. Builtin applies no ring rule; its emitted words are unchanged on the checked corpus, while recorded provider metadata uses `builtin`.

Composed overlap requires the supplied pinned master, pair and chain replay receipts, queries and holdouts. `tools/atlas_model_composed_projection.py` produces the side projection; it does not produce a new measured catalog.
