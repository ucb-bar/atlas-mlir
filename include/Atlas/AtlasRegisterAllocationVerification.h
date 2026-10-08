#ifndef ATLAS_REGISTER_ALLOCATION_VERIFICATION_H
#define ATLAS_REGISTER_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasDMAAllocationVerification.h"

namespace mlir::atlas {

struct VirtualRegisterAssignment {
  Value value;
  unsigned reg;
};

enum class DMAAwaitBasePolicy {
  Preserved,      // The first VLOAD uses the base established at launch.
  Rematerialized // Each VLOAD rematerializes its staging base (PR #13).
};

// Requires a verified, admitted virtual CFG and live source IR. Checks the
// supplied assignments independently of the allocator's interference graph.
// DMA assignments must cover every source transfer. Checks ordered DMA helper
// writes against virtual scalar liveness and preserves operands until capture.
// Choose the await policy to match the lowering being checked.
// Does not establish persistent fixed-helper contents, pending staging-base
// contents, emitted correspondence, or physical completion/timing.
LogicalResult verifyAtlasRegisterAllocation(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> assignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments = {},
    DMAAwaitBasePolicy awaitBasePolicy = DMAAwaitBasePolicy::Preserved);

} // namespace mlir::atlas

#endif
