# Device instruction bounds

Instruction proofs use `atlas.restoration_instruction_bound.v2`. The path-aware
method charges continuing and exiting header visits separately; exports still
recompute the original all-word proof for authenticated v1 publications. Unknown
schemas fail closed. The functional cap comes from Atlas's single
`DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT` constant (1,000,000). Merlin's
`SearchLimits` has no functional execution limit. Lower target API caps are
optional Atlas policy; the v2 record declares both the default and the applied
cap. This does not introduce Merlin core vocabulary.
