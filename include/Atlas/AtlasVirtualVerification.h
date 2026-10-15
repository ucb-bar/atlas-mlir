#ifndef ATLAS_VIRTUAL_VERIFICATION_H
#define ATLAS_VIRTUAL_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {

// Per-unit MXU weights/accumulators the virtual stage may hold at once.
constexpr unsigned kVirtualMXUSlots = 2;

// Whether two virtual DMA operations may touch the same DRAM bytes while one
// of them writes, by the address and size their op verifiers prove. Their
// state order must then be kept.
bool virtualDMAConflict(Operation *a, Operation *b);

// Check a pre-allocation SSA candidate and its CFG/state obligations.
LogicalResult verifyAtlasVirtualModule(ModuleOp module);
void registerVerifyAtlasVirtualStreamPass();

} // namespace mlir::atlas

#endif
