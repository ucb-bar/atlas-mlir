#ifndef ATLAS_SPILLING_H
#define ATLAS_SPILLING_H

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
bool isAtlasSpillStore(Operation *op);
bool isAtlasSpillLoad(Operation *op);
unsigned atlasSpillWords(Operation *store);
StringRef atlasSlotKind(Type type);
bool isAtlasSlotAccess(Operation *op);
// The slot a save or restore accesses, or a slot copy reads.
Value atlasAccessedSlot(Operation *op);
// The value a save writes to its slot or a restore reads back; null for a
// slot copy.
Value atlasSlotValue(Operation *op);
// The spill value a spill store produces and its reload reads.
Value atlasSpillSaved(Operation *op);
// The tile a spill store saves or a reload restores.
Value atlasSpillTile(Operation *op);
// A spill store of `tile` at `slotWord`, and a reload of `saved` as a tile of
// `type`, each consuming `state`.
Operation *createAtlasSpillStore(OpBuilder &builder, Location loc, Value state,
                                 Value tile, uint32_t slotWord);
Operation *createAtlasReload(OpBuilder &builder, Location loc, Value state,
                             Value saved, Type type);
bool isAtlasMemoryControl(Value value);
LogicalResult verifyAtlasSlotOperation(Operation *op);
LogicalResult verifyAtlasSpillOperation(Operation *op);
LogicalResult placeAtlasSpillSlots(func::FuncOp function, unsigned words);
LogicalResult verifyAtlasSpillSlots(func::FuncOp function);
// Preserve completed MXU accumulators and weights through raw tile saves.
LogicalResult spillAtlasMXUState(func::FuncOp function, unsigned maxSpills,
                                unsigned &spills);
void registerSpillAtlasMXUStatePass();
std::unique_ptr<Pass> createSpillAtlasMXUStatePass();
void registerSpillAtlasVirtualPass();
std::unique_ptr<Pass> createSpillAtlasVirtualPass();
} // namespace mlir::atlas

#endif
