# Exact unary value facts

Atlas VPU unary arithmetic uses RTL tables with finite fixed-point wrap behavior. General finite or nonpositive exponential inputs do not imply finite outputs. The shared analyzer admits a transfer only when the entire existing raw ordered BF16 input interval is inside one qualified inclusive domain.

| Source kind | Input ordered keys | Mathematical endpoints | Full-domain output ordered keys |
| --- | --- | --- | --- |
| `exp` | `3c58..8000` | −334 through +0, including −0 | `8000..bf80` (+0 through 1) |
| `exp2` | `3c7f..8000` | −256 through +0, including −0 | `8000..bf80` (+0 through 1) |
| `recip` | `bf80..c200` | 1 through 32 | `bd00..bf80` (1/32 through 1) |

These are conservative admission domains rather than claims about all finite regions. `exp(-336)`, `exp2(-258)` and reciprocal of maximum finite BF16 produce infinity. Some inputs beyond the negative exponential boundary wrap to finite values; those are also refused. Unknown inputs and intervals extending even one key outside a domain stay unknown. Source-derived raw min/max clamps can establish the domains without caller contracts.

`tools/atlas_egglog_unary_tables.py` authenticates provider-owned compressed and decompressed table identities and computes the exact ordered image hull over every key in a query interval. It does not interpolate endpoints or assume monotonicity. Immutable segment-tree caches include the operation, BF16 dtype, table hash and arithmetic target identity; the compressed artifact is reauthenticated before each query. Missing or changed artifacts fail closed. Caller contract opt-in, current-program identity and invocation-owned fact caches retain their existing checks.

The data provenance records the retained RTL-generated tables and 17 matching relevant source hashes, not a complete transitive RTL dependency closure. Fresh retained `VAtlasCore` sweeps additionally qualify all 65,536 raw input encodings per operation against the exact raw table outputs, including NaNs and both signed zeros. Each operation uses two 32-tile batches with two repeated invocations in each backend, full DRAM checks, VMEM-complement guards, decoded operation checks, drained completion and unchanged binary hashes. A 64-tile attempt was preserved when the compiler rejected its 2,109 words against its 2,048-word generated-program policy; the two batches preserve the full domain and all numerical criteria.

Qualification artifacts and reproducible private drivers live in `sessions/oct18/egglog/unary-facts/`; `sweep-audit.json`, `table-gate.json` and `provenance-gate.json` distinguish fresh execution, exhaustive table/model checks and source authentication. `pack-execution/` qualifies a consumed-output fixture with derived raw clamps, each unary, and PACK, using automatically derived facts and the frozen integrated PACK recipe library. There is no change to resource eclasses, scheduling semantics or general arithmetic rewrites.

Integration into certificate-producing search must add this unary helper, its three compressed table files and provenance/qualification records to the certificate implementation pins. The local branch does not edit the frozen integrated search provider; its existing fixed module list predates this extension.
