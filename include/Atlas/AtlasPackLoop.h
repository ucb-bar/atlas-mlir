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

} // namespace mlir::atlas
#endif
