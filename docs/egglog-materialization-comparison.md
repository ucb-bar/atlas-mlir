# Comparing bounded materialization choices

`tools/compare_atlas_materializations.py` extends the hand/egglog comparison with identical optional splat-sharing policies. Sharing reuses a structurally identical pure BF16 constant, including all attributes and its raw bits; it introduces no floating-point arithmetic identity. The default extraction pipeline still rematerializes constants near their uses. This optional comparison explores no sharing, a 16-operation/two-constant window, and a 64-operation/two-constant window after each strategy produces virtual IR.

Each enabled policy probes both its baseline and proposed materialization with the actual virtual allocator. Rejection preserves the exact baseline source and records zero effective shares. Distance and cache capacity are search bounds, not a register-pressure proof. The usual scheduled compilation, physical verifier, independent output oracle, guard checks, and measured selection are still mandatory after generation.

Hand enumeration applies the directed exact local laws and every native/unit0/unit1 assignment for at most six automatically guarded PACK sites. Egglog supplies the actual bounded extraction results. Both receive the same policy set and the same budget of distinct cost-ranked source candidates; original native and every legacy uniform-engine control/policy are common mandatory controls. Source aliases are measured once while retaining both strategies' identities. The complete generated pool is also saved for exhaustive evaluation of this explicitly bounded product, which is not a global algebraic or scheduling optimum.

Use an explicit Merlin checkout containing the shared e-graph and selection modules, an observed timing-facts file, and a qualified Atlas compiler. Set `TMPDIR` to an owned output directory for allocation probes.

```sh
PYTHONPATH="$MERLIN_SRC:$ATLAS_TOOLS" python "$ATLAS_TOOLS/compare_atlas_materializations.py" source.mlir --timing-facts model-timing.json --atlas-opt "$ATLAS_OPT" --output comparison --budget 4 --attempts 64
```

The output contains `pool.json`, the shortlist `candidates.json`, all unique virtual sources, `comparison.json` with policy decisions and strategy aliases, and a source-bound provider declaration in `proof.json`. The declaration pins the implementation, unary qualification data, sharing helper, hand comparator, and allocator executable. It is a derivation record, not an independent theorem checker. Pass the shortlist or bounded full pool to `select_atlas_candidates.py` with explicit compile/measurement adapters, metric, target tiers, and independent byte oracles.

The static cost remains an additive observed-operation and pressure heuristic. In particular, deleting splats can affect physical scheduling without changing the modeled critical path. Report measured model and RTL results separately and include equal-policy hand controls before attributing any benefit to egglog search. Candidate `families` identifies provider membership and effective sharing; it does not assert that every available local rule fired.
