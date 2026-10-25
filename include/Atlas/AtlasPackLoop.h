#ifndef ATLAS_PACK_LOOP_H
#define ATLAS_PACK_LOOP_H

#include "Atlas/AtlasLoop.h"

namespace mlir::atlas {
// Compiler code-size policy: unroll the model-provider copy loop by two rows.
// This is neither derived from BF16 register spans nor a model timing fact.
constexpr unsigned kPackRelayoutUnrollFactor = 2;
constexpr unsigned atlasPackRelayoutRowsPerIteration(bool modelProvider) {
  return modelProvider ? kPackRelayoutUnrollFactor : 1;
}

// Additional provenance checks for the PACK layout contract. DMA overlap uses
// AtlasLoop directly and does not consult these annotations.
bool isAtlasPackRelayoutLoop(llvm::ArrayRef<Operation *> ops,
                            const AtlasLoop &loop);

struct AtlasStream;
// Timing contexts remain restricted to PACK provenance and a decided finite
// replay covering 32 rows, possibly grouped within each iteration. Generic
// loop DMA access proofs grant no scalar overlap. Shorter replays repeat their
// exact cases to fill the immutable 32-entry context at shared ordinals.
std::vector<timing::TimingContext> proveAtlasPackLoopTimingContexts(const AtlasStream &stream, const AtlasLoop &loop,
                                                                    const timing::RegValues &entry);

// Proof for two copy loads of a PACK relayout loop, as lowerPack issues
// them back to back: word 0 of each 512-byte half of the staged tile, 16
// bytes further each row. For virtual costs of the loop not yet lowered.
std::pair<timing::TimingContext, timing::TimingContext>
atlasPackRelayoutCopyContexts(const timing::Instr &first, const timing::Instr &second);

} // namespace mlir::atlas
#endif
