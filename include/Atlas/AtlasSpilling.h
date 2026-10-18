#ifndef ATLAS_SPILLING_H
#define ATLAS_SPILLING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
bool isAtlasSpillStore(Operation *op);
bool isAtlasSpillLoad(Operation *op);
unsigned atlasSpillWords(Operation *store);
StringRef atlasSlotKind(Type type);
bool isAtlasSlotAccess(Operation *op);
bool isAtlasMemoryControl(Value value);
LogicalResult verifyAtlasSlotOperation(Operation *op);
LogicalResult verifyAtlasSpillOperation(Operation *op);
LogicalResult verifyAtlasSpillSlots(func::FuncOp function);
void registerSpillAtlasVirtualPass();
} // namespace mlir::atlas

#endif
