#ifndef ATLAS_REGISTER_ALLOCATION_VERIFICATION_H
#define ATLAS_REGISTER_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

struct VirtualRegisterAssignment {
  Value value;
  unsigned reg;
};

// Requires a verified, admitted virtual CFG and live source IR. Checks the
// supplied assignments independently of the allocator's interference graph.
LogicalResult verifyAtlasRegisterAllocation(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> assignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs);

} // namespace mlir::atlas

#endif
