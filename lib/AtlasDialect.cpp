#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"

using namespace mlir::atlas;

#include "Atlas/AtlasOpsDialect.cpp.inc"

void AtlasDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Atlas/AtlasOps.cpp.inc"
      >();
  registerTypes();
}
