#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

#include "Atlas/AtlasStream.h"

namespace mlir::atlas {

// Delays valid under every entry context group. Absent `order` keeps source
// order; a supplied absolute-index order preserves values and the terminator
// and delay slot. `tail` drains a block without a branch or halt.
LogicalResult atlasBlockDelays(const AtlasStream &stream, size_t block, const timing::TimingProvider &provider,
    std::vector<DelayInsertion> &before, DelayInsertion &after, int &tail,
    std::optional<llvm::ArrayRef<size_t>> order = std::nullopt);

// Register the delay insertion pass with atlas-opt.
void registerInsertAtlasDelaysPass();

} // namespace mlir::atlas

#endif
