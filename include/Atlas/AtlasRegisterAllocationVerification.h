#ifndef ATLAS_REGISTER_ALLOCATION_VERIFICATION_H
#define ATLAS_REGISTER_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasDMAAllocationVerification.h"

namespace mlir::atlas {

struct VirtualRegisterAssignment {
  Value value;
  unsigned reg;
};

// Atomic DMA always rematerializes: lowering places each staging VLOAD
// before the load's first use and sets its base there. The policy is kept
// for callers and no longer changes the check.
enum class DMAAwaitBasePolicy {
  Preserved,
  Rematerialized // Each VLOAD rematerializes its staging base (PR #13).
};

// Requires a verified, admitted virtual CFG and live source IR. Checks the
// supplied assignments independently of the allocator's interference graph.
// DMA assignments must cover every source transfer; helper writes must spare
// live scalars and uncaptured operands. With explicit DMA, each staging
// VLOAD's base and the 1024-byte helper are checked by constant propagation
// over source lowering effects.
LogicalResult verifyAtlasRegisterAllocation(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> assignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments = {},
    DMAAwaitBasePolicy awaitBasePolicy = DMAAwaitBasePolicy::Preserved);

} // namespace mlir::atlas

#endif
