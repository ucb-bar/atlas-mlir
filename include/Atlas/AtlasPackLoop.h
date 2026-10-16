#ifndef ATLAS_PACK_LOOP_H
#define ATLAS_PACK_LOOP_H

#include "Atlas/AtlasLoop.h"

namespace mlir::atlas {
// Additional provenance checks for the PACK layout contract. DMA overlap uses
// AtlasLoop directly and does not consult these annotations.
bool isAtlasPackRelayoutLoop(llvm::ArrayRef<Operation *> ops,
                            const AtlasLoop &loop);
}
#endif
