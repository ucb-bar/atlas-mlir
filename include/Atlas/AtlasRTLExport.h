#ifndef ATLAS_RTL_EXPORT_H
#define ATLAS_RTL_EXPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"

namespace mlir::atlas {
// Resolved timing of the final stream, exported only after it rechecks.
FailureOr<llvm::json::Object> exportAtlasRTLTiming(ModuleOp module);
} // namespace mlir::atlas

#endif
