#ifndef ATLAS_VIRTUAL_VERIFICATION_H
#define ATLAS_VIRTUAL_VERIFICATION_H

#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir::atlas {

// Per-unit MXU weights/accumulators the virtual stage may hold at once.
constexpr unsigned kVirtualMXUSlots = 2;

// A virtual DMA load's or store's DRAM byte address and byte count operands,
// and its tile: what a load produces or a store reads.
struct VirtualDMATransfer {
  OpOperand *dram, *size;
  Value tile;
  bool store;
};
// Null for any other operation.
std::optional<VirtualDMATransfer> virtualDMATransfer(Operation *op);

// The accumulator a virtual MXU readout pops; null for any other operation.
Value virtualMXUReadoutAcc(Operation *op);

// Whether two virtual DMA operations may touch the same DRAM bytes while one
// of them writes, by the address and size their op verifiers prove. Their
// state order must then be kept.
bool virtualDMAConflict(Operation *a, Operation *b);

// Whether `value` is an i32 arith.constant used only as the DRAM address or
// byte count of virtual DMA. Lowering builds it in the transfer's command
// register at each launch, so it takes no scalar register and is no CFG
// value; the DMA contract still checks what each launch captures.
bool isVirtualDMAImmediate(Value value);

// Whether `value` is the tile a virtual DMA load produces. It stays in its
// staging window until lowering's staging VLOADs, right before its first use,
// a later launch given that window, or its block's terminator; its registers
// are held from there.
bool isVirtualDMALoadedTile(Value value);

// Check a pre-allocation SSA candidate and its CFG/state obligations.
// Spilling may defer only slot-capacity checks. State linearity, current
// accumulator versions and unit ownership remain mandatory. The optional CFG
// form requires each live accumulator exactly once on every successor edge;
// only the MXU spill pass admits it before materializing completed state.
LogicalResult verifyAtlasVirtualModule(ModuleOp module,
                                      bool allowMXUPressure = false,
                                      bool allowMXUCFG = false);
void registerVerifyAtlasVirtualStreamPass();
std::unique_ptr<Pass> createVerifyAtlasVirtualStreamPass();

} // namespace mlir::atlas

#endif
