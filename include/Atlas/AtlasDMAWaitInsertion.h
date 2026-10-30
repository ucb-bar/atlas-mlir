#ifndef ATLAS_DMA_WAIT_INSERTION_H
#define ATLAS_DMA_WAIT_INSERTION_H

#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {

// Completes a generated stream that virtual-to-machine lowering left without
// DMA waits (marked kAtlasDMAWaitsPending): inserts each explicit transfer's
// DMA.WAIT immediately before the first instruction that needs it complete,
// then verifies the generated artifact. With `waitReusedSlots`, launch n also
// completes launch n-8, whose command slot it reuses. Streams without the mark
// are left unchanged.
LogicalResult insertAtlasDMAWaits(ModuleOp module, bool waitReusedSlots = false);

// Register the DMA wait insertion pass with atlas-opt.
void registerInsertAtlasDMAWaitsPass();
// The selected provider decides whether launches wait for reused command slots.
std::unique_ptr<Pass> createInsertAtlasDMAWaitsPass(llvm::StringRef provider = {},
                                                    bool waitReusedSlots = false);

} // namespace mlir::atlas

#endif
