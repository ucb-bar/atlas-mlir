# Proof work measurement

Set `ATLAS_REPORT_PROOF_WORK=1` when running `atlas-opt` or `atlas-emit` to emit
one `atlas-proof-work-v1` JSON object to stderr when the process exits. The
report accumulates the entire process, including repeated verification passes;
it does not reset at a function, source operation, tile, or helper fragment.
Normal stdout, generated metadata, and emitted program words are unchanged.

The six contract rows are `dma`, `mxu`, `tile`, `cfg`, `source_memory`, and
`buffer`. DMA includes its launch/extent contract and in-flight memory checks.
`invocations` counts checker entries. `record_visits` counts captured/source
contract records examined or derived and machine instruction tag scans.
Shared source record derivation used during contract construction is included.
`transfer_visits` counts abstract instruction transfers, including every
fixpoint revisit, diagnostic replay, and PACK helper-loop iteration. A transfer
is one interpreter/checker step, regardless of its number of register or byte
facts; the counters are work units, not elapsed time or CPU instruction counts.

Auxiliary rows separately count source and captured scalar address expression
transfers, source and captured output-initialization instruction/expression
transfers, and initialization state-graph reachability visits. `six_contract_work`
is the sum of record and transfer visits in the six standard rows;
`auxiliary_work` sums the remaining rows; `total_work` sums both. Invocation
counts are not added to these totals.

These are measurement counters only. They never grant, reset, split, consume,
or multiply a proof allowance. The existing default scalar proof work limit
remains 1,000,000, and all existing refusal behavior remains in force. Reporting
is disabled by default; disabled event sites perform only a boolean check.
