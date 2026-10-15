#include "Atlas/AtlasTypes.h"
#include "Atlas/AtlasDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::atlas;

static mlir::LogicalResult
verifyMXUUnit(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
              unsigned unit) {
  if (unit > 1)
    return emitError() << "MXU unit must be in [0, 1], got " << unit;
  return mlir::success();
}

mlir::LogicalResult VirtualMXUWeightType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, unsigned unit) {
  return verifyMXUUnit(emitError, unit);
}

mlir::LogicalResult VirtualMXUAccType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, unsigned unit) {
  return verifyMXUUnit(emitError, unit);
}

#define GET_TYPEDEF_CLASSES
#include "Atlas/AtlasOpsTypes.cpp.inc"

void AtlasDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Atlas/AtlasOpsTypes.cpp.inc"
      >();
}
