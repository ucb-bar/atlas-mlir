#ifndef ATLAS_VIRTUAL_VERIFICATION_H
#define ATLAS_VIRTUAL_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {

// What the virtual stage may hold at once: explicit DMA transfers in flight,
// each with its own channel and staging window (the DMA latches its
// registers at launch); and weights and accumulators per MXU unit, one per
// hardware slot.
constexpr unsigned kMaxPendingVirtualDMA = 2;
constexpr unsigned kVirtualMXUSlots = 2;

// Check a pre-allocation SSA candidate and its CFG/state obligations.
LogicalResult verifyAtlasVirtualModule(ModuleOp module);
void registerVerifyAtlasVirtualStreamPass();

} // namespace mlir::atlas

#endif
