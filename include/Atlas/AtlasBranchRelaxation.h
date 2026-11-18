#ifndef ATLAS_BRANCH_RELAXATION_H
#define ATLAS_BRANCH_RELAXATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
namespace mlir::atlas {
// Run whenever layout changes, before encoding. Preserves the original slot.
LogicalResult relaxAtlasBranches(ModuleOp module);
LogicalResult verifyAtlasRelaxedBranches(ModuleOp module);
}
#endif
