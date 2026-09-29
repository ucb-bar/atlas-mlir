#include "Atlas/AtlasTypes.h"
#include "Atlas/AtlasDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::atlas;

#define GET_TYPEDEF_CLASSES
#include "Atlas/AtlasOpsTypes.cpp.inc"

void AtlasDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Atlas/AtlasOpsTypes.cpp.inc"
      >();
}
