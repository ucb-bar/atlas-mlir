#ifndef ATLAS_TILE_CONTRACT_VERIFICATION_H
#define ATLAS_TILE_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {

// Requires admitted, verified live virtual SSA, independently checked register
// placement claims and valid boundary ABI bindings. Derives static transfer
// expectations without consulting the lowering's planned commands.
FailureOr<ArrayAttr> buildAtlasTileContract(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> tensorRegisters,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs);

// Checks all generated VLOAD/VSTORE/DMA/WAIT commands and tagged mailbox LW
// against source records, including required predecessor commands on emitted
// paths. Requires separate structural/lifecycle and DMA/MXU contract checks.
// Excludes tensor contents, PACK's scalar relayout, source/emitted CFG
// correspondence, dynamic execution counts, and physical completion/release.
LogicalResult verifyAtlasGeneratedTileContract(ModuleOp module);

} // namespace mlir::atlas

#endif
