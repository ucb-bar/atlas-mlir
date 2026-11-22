#ifndef ATLAS_TILE_CONTRACT_VERIFICATION_H
#define ATLAS_TILE_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Tile commands and DMA launches one source operation expands to. The mailbox
// prelude (DMA launch, wait) precedes the per-argument LWs and all of these.
struct AtlasTileExpansion { unsigned commands, launches; };
AtlasTileExpansion atlasTileExpansion(llvm::StringRef sourceOp);
AtlasTileExpansion atlasTileExpansion(Operation *op);
constexpr int32_t kMailboxPreludeCommands = 2;

// Source-derived partial loads whose allocator-owned staging suffix remains
// zero from an earlier load in the same block. Final buffer verification still
// checks every surviving word origin in the emitted stream.
llvm::DenseSet<Operation *> findAtlasReusedDMAPrefills(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments);

// Builds static transfer records from live source, checked placements and the
// boundary ABI, without consulting the lowering's planned commands.
FailureOr<ArrayAttr> buildAtlasTileContract(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> tensorRegisters,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs,
    llvm::ArrayRef<VirtualRegisterAssignment> scalarRegisters = {});

// Checks a generated artifact's VLOAD/VSTORE/DMA/WAIT and mailbox LW commands
// and their emitted-path predecessors. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedTileContract(const AtlasVerificationContext &ctx);

} // namespace mlir::atlas

#endif
