#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

#include "Atlas/AtlasStream.h"

namespace mlir::atlas {

// Source-order delays valid under every entry context group.
// `tail` drains a block without a branch or halt.
LogicalResult atlasInOrderBlockDelays(const AtlasStream &stream, size_t block,
                                      const timing::TimingProvider &provider,
                                      std::vector<DelayInsertion> &before,
                                      DelayInsertion &after, int &tail);

// Delays for an absolute-index order preserving scalar values and the
// terminator/delay slot, valid under every entry context group.
LogicalResult atlasOrderedBlockDelays(const AtlasStream &stream, size_t block,
                                      const timing::TimingProvider &provider,
                                      llvm::ArrayRef<size_t> order,
                                      std::vector<DelayInsertion> &before,
                                      DelayInsertion &after, int &tail);

// Register the delay insertion pass with atlas-opt.
void registerInsertAtlasDelaysPass();

} // namespace mlir::atlas

#endif
