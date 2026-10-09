#ifndef ATLAS_TILE_CONTRACT_VERIFICATION_H
#define ATLAS_TILE_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Tile commands and DMA launches that one source operation expands to, keyed
// by operation name. The scalar-argument mailbox prelude (DMA launch and wait)
// precedes the per-argument LW commands and every operation's commands.
struct AtlasTileExpansion { unsigned commands, launches; };
AtlasTileExpansion atlasTileExpansion(llvm::StringRef sourceOp);
constexpr int32_t kMailboxPreludeCommands = 2;

// Builds static transfer records from live source, checked placements and the
// boundary ABI, without consulting the lowering's planned commands.
FailureOr<ArrayAttr> buildAtlasTileContract(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> tensorRegisters,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs);

// Checks a generated artifact's VLOAD/VSTORE/DMA/WAIT and mailbox LW commands
// and their emitted-path predecessors. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedTileContract(const AtlasVerificationContext &ctx);

} // namespace mlir::atlas

#endif
