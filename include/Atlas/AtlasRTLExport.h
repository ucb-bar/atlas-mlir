#ifndef ATLAS_RTL_EXPORT_H
#define ATLAS_RTL_EXPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"

namespace mlir::atlas {
// Export the shared provider's resolved facts only after final timing checks.
// The export preserves conditional applicability and is not a qualification.
FailureOr<llvm::json::Object> exportAtlasRTLTiming(ModuleOp module);
} // namespace mlir::atlas

#endif
