#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

#include "Atlas/AtlasStream.h"

namespace mlir::atlas {

// Delays that keep block `block` in source order under every entry context
// group of the provider (atlasBlockContexts); `tail` receives the idle time
// that drains a block without a branch or halt.
LogicalResult atlasInOrderBlockDelays(const AtlasStream &stream, size_t block,
                                      const timing::TimingProvider &provider,
                                      std::vector<DelayInsertion> &before,
                                      DelayInsertion &after, int &tail);

// Plan one block in a chosen order, given as absolute stream indices. The
// order must preserve scalar values and the final terminator/delay slot.
// One delay plan satisfies every entry context group of the provider.
LogicalResult atlasOrderedBlockDelays(const AtlasStream &stream, size_t block,
                                      const timing::TimingProvider &provider,
                                      llvm::ArrayRef<size_t> order,
                                      std::vector<DelayInsertion> &before,
                                      DelayInsertion &after, int &tail);

// Register the delay insertion pass with atlas-opt.
void registerInsertAtlasDelaysPass();

} // namespace mlir::atlas

#endif
