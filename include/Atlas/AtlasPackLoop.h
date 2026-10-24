#ifndef ATLAS_PACK_LOOP_H
#define ATLAS_PACK_LOOP_H

#include "Atlas/AtlasLoop.h"

namespace mlir::atlas {
// Additional provenance checks for the PACK layout contract. DMA overlap uses
// AtlasLoop directly and does not consult these annotations.
bool isAtlasPackRelayoutLoop(llvm::ArrayRef<Operation *> ops,
                            const AtlasLoop &loop);

struct AtlasStream;
// Timing contexts remain restricted to PACK provenance and exactly 32
// decided iterations. Generic loop DMA access proofs grant no scalar overlap.
std::vector<timing::TimingContext> proveAtlasPackLoopTimingContexts(const AtlasStream &stream, const AtlasLoop &loop,
                                                                    const timing::RegValues &entry);

// Proof for two copy loads of a PACK relayout loop, as lowerPack issues
// them back to back: word 0 of each 512-byte half of the staged tile, 16
// bytes further each row. For virtual costs of the loop not yet lowered.
std::pair<timing::TimingContext, timing::TimingContext>
atlasPackRelayoutCopyContexts(const timing::Instr &first, const timing::Instr &second);

} // namespace mlir::atlas
#endif
