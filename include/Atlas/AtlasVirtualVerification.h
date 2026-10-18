#ifndef ATLAS_VIRTUAL_VERIFICATION_H
#define ATLAS_VIRTUAL_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {

// Per-unit MXU weights/accumulators the virtual stage may hold at once.
constexpr unsigned kVirtualMXUSlots = 2;

// Whether two virtual DMA operations may touch the same DRAM bytes while one
// of them writes, by the address and size their op verifiers prove. Their
// state order must then be kept.
bool virtualDMAConflict(Operation *a, Operation *b);

// Whether `value` is an i32 arith.constant used only as the DRAM address or
// byte count of virtual DMA. Lowering builds it in the transfer's command
// register at each launch, so it takes no scalar register and is no CFG
// value; the DMA contract still checks what each launch captures.
bool isVirtualDMAImmediate(Value value);

// Check a pre-allocation SSA candidate and its CFG/state obligations.
// Spilling may defer only slot-capacity checks. State linearity, current
// accumulator versions and unit ownership remain mandatory. The optional CFG
// form requires each live accumulator exactly once on every successor edge;
// only the MXU spill pass admits it before materializing completed state.
LogicalResult verifyAtlasVirtualModule(ModuleOp module,
                                      bool allowMXUPressure = false,
                                      bool allowMXUCFG = false);
void registerVerifyAtlasVirtualStreamPass();

} // namespace mlir::atlas

#endif
