#ifndef ATLAS_DMA_CONTRACT_VERIFICATION_H
#define ATLAS_DMA_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasDMAAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {

// Derives expectations from live virtual source and independently validated
// placements. Numeric fields are signless i32 bit patterns; direction is a
// load/store string. Records are sorted by nonnegative transfer id.
FailureOr<ArrayAttr> buildAtlasDMAContract(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> assignments);

// Checks retained source expectations against captured tagged machine DMA operands.
// Requires separate structural and DMA lifecycle checks (the generated-schedule
// entry point runs those first); correspondence alone does not prove ordering.
// Legacy UnitAttr markers require no contract. This does not prove tensor
// contents, untagged DMA correspondence, or source-to-emitted CFG correspondence.
LogicalResult verifyAtlasGeneratedDMAContract(ModuleOp module);

} // namespace mlir::atlas

#endif
