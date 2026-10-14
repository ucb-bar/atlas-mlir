#ifndef ATLAS_DMA_WAIT_INSERTION_H
#define ATLAS_DMA_WAIT_INSERTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {

// Completes a generated stream that virtual-to-machine lowering left without
// DMA waits (marked kAtlasDMAWaitsPending): inserts each explicit transfer's
// DMA.WAIT immediately before the first instruction that needs it complete,
// then verifies the generated artifact. Streams without the mark are left
// unchanged.
LogicalResult insertAtlasDMAWaits(ModuleOp module);

// Register the DMA wait insertion pass with atlas-opt.
void registerInsertAtlasDMAWaitsPass();

} // namespace mlir::atlas

#endif
