#ifndef ATLAS_OPS_H
#define ATLAS_OPS_H

#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cstdint>
#include <optional>
#define GET_OP_CLASSES
#include "Atlas/AtlasOps.h.inc"

namespace mlir::atlas {

// The value of an i32 built from arith.constant and unflagged arith.addi, as
// the virtual DMA verifiers prove DRAM addresses and sizes; nullopt otherwise.
std::optional<uint32_t> provenI32(Value value);

} // namespace mlir::atlas

#endif
