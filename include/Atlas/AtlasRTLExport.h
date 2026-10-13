#ifndef ATLAS_RTL_EXPORT_H
#define ATLAS_RTL_EXPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"

namespace mlir::atlas {
// Export final-stream timing only after executable and timing verification.
FailureOr<llvm::json::Object> exportAtlasRTLTiming(ModuleOp module);
} // namespace mlir::atlas

#endif
