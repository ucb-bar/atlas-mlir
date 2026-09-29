#include "Atlas/AtlasOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::atlas;

// Until unit-specific temporal effects are qualified, every issued machine
// instruction conservatively reads and writes the physical state resource.
#define ATLAS_MACHINE_EFFECTS(OP)                                             \
  void OP::getEffects(                                                        \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>     \
          &effects) {                                                         \
    effects.emplace_back(MemoryEffects::Read::get());                         \
    effects.emplace_back(MemoryEffects::Write::get());                        \
  }

ATLAS_MACHINE_EFFECTS(VLoadOp)
ATLAS_MACHINE_EFFECTS(VStoreOp)
ATLAS_MACHINE_EFFECTS(DMAOp)
ATLAS_MACHINE_EFFECTS(DMAWaitOp)
ATLAS_MACHINE_EFFECTS(DMAConfigOp)
ATLAS_MACHINE_EFFECTS(MXUPushOp)
ATLAS_MACHINE_EFFECTS(MXUMatmulOp)
ATLAS_MACHINE_EFFECTS(MXUPopOp)
ATLAS_MACHINE_EFFECTS(VPUBinaryOp)
ATLAS_MACHINE_EFFECTS(VPUUnaryOp)
ATLAS_MACHINE_EFFECTS(VPUPackOp)
ATLAS_MACHINE_EFFECTS(VPUReduceOp)
ATLAS_MACHINE_EFFECTS(VLIOp)
ATLAS_MACHINE_EFFECTS(XLUTransposeOp)

static LogicalResult inRange(Operation *op, StringRef name, int64_t value,
                             int64_t first, int64_t last) {
  if (value >= first && value <= last)
    return success();
  return op->emitOpError() << name << " must be in [" << first << ", " << last
                           << "], got " << value;
}

static LogicalResult matrixReg(Operation *op, StringRef name, int64_t value,
                               bool bf16Pair = false) {
  if (failed(inRange(op, name, value, 0, bf16Pair ? 62 : 63)))
    return failure();
  if (bf16Pair && (value & 1))
    return op->emitOpError() << name << " BF16 pair base must be even";
  return success();
}

static LogicalResult scalarReg(Operation *op, StringRef name, int64_t value) {
  return inRange(op, name, value, 0, 31);
}

static LogicalResult mxuSlot(Operation *op, StringRef name, int64_t value) {
  return inRange(op, name, value, 0, 1);
}

LogicalResult VLoadOp::verify() {
  if (getFormat() != "raw")
    return emitOpError("format must be raw; one instruction moves 1024 bytes");
  if (failed(matrixReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "base", getBase())))
    return failure();
  return inRange(*this, "offset (32-byte units)",
                 getOffsetAttr().getValue().getSExtValue(), -2048, 2047);
}

LogicalResult VStoreOp::verify() {
  if (getFormat() != "raw")
    return emitOpError("format must be raw; one instruction moves 1024 bytes");
  if (failed(matrixReg(*this, "src", getSrc())) ||
      failed(scalarReg(*this, "base", getBase())))
    return failure();
  return inRange(*this, "offset (32-byte units)",
                 getOffsetAttr().getValue().getSExtValue(), -2048, 2047);
}

LogicalResult DMAOp::verify() {
  if (getDirection() != "load" && getDirection() != "store")
    return emitOpError("direction must be load or store");
  if (failed(inRange(*this, "channel", getChannel(), 0, 7)))
    return failure();
  if (failed(scalarReg(*this, "reg", getReg())) ||
      failed(scalarReg(*this, "dram", getDram())))
    return failure();
  return scalarReg(*this, "size", getSize());
}

LogicalResult DMAWaitOp::verify() {
  return inRange(*this, "channel", getChannel(), 0, 7);
}

LogicalResult DMAConfigOp::verify() {
  if (failed(inRange(*this, "channel", getChannel(), 0, 7)))
    return failure();
  return scalarReg(*this, "base_reg", getBaseReg());
}

LogicalResult MXUPushOp::verify() {
  if (failed(inRange(*this, "unit", getUnit(), 0, 1)) ||
      failed(mxuSlot(*this, "slot", getSlot())))
    return failure();
  bool bf16 = getKind() == "acc_bf16";
  if (getKind() != "weight_fp8" && getKind() != "acc_fp8" && !bf16)
    return emitOpError("kind must be weight_fp8, acc_fp8, or acc_bf16");
  return matrixReg(*this, "src", getSrc(), bf16);
}

LogicalResult MXUMatmulOp::verify() {
  if (failed(inRange(*this, "unit", getUnit(), 0, 1)) ||
      failed(mxuSlot(*this, "weight_slot", getWeightSlot())) ||
      failed(mxuSlot(*this, "acc_slot", getAccSlot())))
    return failure();
  return matrixReg(*this, "src", getSrc());
}

LogicalResult MXUPopOp::verify() {
  if (failed(inRange(*this, "unit", getUnit(), 0, 1)) ||
      failed(mxuSlot(*this, "slot", getSlot())) ||
      failed(inRange(*this, "scale_reg", getScaleReg(), 0, 31)))
    return failure();
  if (getFormat() != "fp8" && getFormat() != "bf16")
    return emitOpError("format must be fp8 or bf16");
  if (getFormat() == "bf16" && getScaleReg() != 0)
    return emitOpError("unused BF16 scale_reg must be zero");
  // BF16 fills the two 1024-byte halves of one 32x32 tile.
  return matrixReg(*this, "dst", getDst(), getFormat() == "bf16");
}

LogicalResult VPUBinaryOp::verify() {
  StringRef kind = getKind();
  if (kind != "add" && kind != "sub" && kind != "mul" &&
      kind != "min" && kind != "max")
    return emitOpError("unknown binary VPU kind");
  if (failed(matrixReg(*this, "dst", getDst(), true)) ||
      failed(matrixReg(*this, "lhs", getLhs(), true)))
    return failure();
  return matrixReg(*this, "rhs", getRhs(), true);
}

LogicalResult VPUUnaryOp::verify() {
  StringRef kind = getKind();
  if (kind != "mov" && kind != "recip" && kind != "exp" &&
      kind != "exp2" && kind != "square" && kind != "cube" &&
      kind != "relu" && kind != "sin" && kind != "cos" &&
      kind != "tanh" && kind != "log2" && kind != "sqrt")
    return emitOpError("unknown unary VPU kind");
  bool pair = true;
  if (failed(matrixReg(*this, "dst", getDst(), pair)))
    return failure();
  return matrixReg(*this, "src", getSrc(), pair);
}

LogicalResult VPUPackOp::verify() {
  if (getDirection() != "bf16_to_fp8" &&
      getDirection() != "fp8_to_bf16")
    return emitOpError("unknown pack direction");
  if (failed(inRange(*this, "scale_reg", getScaleReg(), 0, 31)))
    return failure();
  bool srcPair = getDirection() == "bf16_to_fp8";
  if (failed(matrixReg(*this, "src", getSrc(), srcPair)))
    return failure();
  return matrixReg(*this, "dst", getDst(), !srcPair);
}

LogicalResult VPUReduceOp::verify() {
  StringRef kind = getKind();
  if (kind != "col_sum" && kind != "col_min" && kind != "col_max" &&
      kind != "row_sum" && kind != "row_min" && kind != "row_max")
    return emitOpError("unknown reduce VPU kind");
  if (failed(matrixReg(*this, "dst", getDst(), true)))
    return failure();
  return matrixReg(*this, "src", getSrc(), true);
}

LogicalResult VLIOp::verify() {
  if (getMode() != "all" && getMode() != "row" &&
      getMode() != "col" && getMode() != "one")
    return emitOpError("mode must be all, row, col, or one");
  bool pair = getMode() == "all" || getMode() == "row";
  if (failed(matrixReg(*this, "dst", getDst(), pair)))
    return failure();
  return inRange(*this, "immediate (raw 16 bits)", getImmediate(), 0, 65535);
}

LogicalResult XLUTransposeOp::verify() {
  if (failed(matrixReg(*this, "dst", getDst())))
    return failure();
  return matrixReg(*this, "src", getSrc());
}

#define GET_OP_CLASSES
#include "Atlas/AtlasOps.cpp.inc"
