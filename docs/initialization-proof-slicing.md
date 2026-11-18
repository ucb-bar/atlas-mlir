# Output-initialization proof slicing

Large counted matrix programs repeatedly execute restoration arithmetic that
cannot affect DMA addresses or control flow. Replaying those unrelated facts
could exhaust the one-million-step proof budget before reaching the final
output store. Increasing the instruction-image or execution budget does not
change that independent compiler proof limit.

The source initialization verifier computes a conservative set of scalar
values it can resolve, then slices their definitions from every DMA address,
DMA size and control-flow operand. It still visits every transfer and control
edge. Tensor-derived unknown values cannot establish an address or branch
condition; an unknown branch still explores both successors. Loop arguments
remain eligible whenever any incoming edge can provide a known value.

The captured-contract verifier independently derives its required scalar
values from transfer pointers and branch conditions, including all incoming
edges of required block arguments. It validates all operation records once
and replays every DMA operation in source order, so unrelated loop-carried
tensor data does not distinguish its memory states.

The live-IR proof also discards dead scalar facts at each block entry using
MLIR liveness; facts used later, including outer values and current block
arguments, remain. The captured-CFG proof stores one bit per aligned 32-byte
DMA beat, with unchanged coverage and intersection rules.

Both verifiers retain the original work limit, initialization intersections,
unknown-address rejection and nonterminating-cycle checks. This changes proof
work, not the emitted program's arithmetic, memory layout or scheduling.

Validation covers known and unknown loop-carried facts, late loss of a known
value, unknown DMA addresses, uninitialized reads, joins, contract tampering
and proof exhaustion.
