#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdint>
#include <optional>

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
ATLAS_MACHINE_EFFECTS(ALURegOp)
ATLAS_MACHINE_EFFECTS(ALUImmOp)
ATLAS_MACHINE_EFFECTS(BranchOp)
ATLAS_MACHINE_EFFECTS(JumpOp)
ATLAS_MACHINE_EFFECTS(DelayOp)
ATLAS_MACHINE_EFFECTS(UpperOp)
ATLAS_MACHINE_EFFECTS(CSROp)
ATLAS_MACHINE_EFFECTS(TrapOp)
ATLAS_MACHINE_EFFECTS(FenceOp)
ATLAS_MACHINE_EFFECTS(ScalarLoadOp)
ATLAS_MACHINE_EFFECTS(ScalarStoreOp)

// Conservatively model virtual MXU resource transitions as read/write effects.
ATLAS_MACHINE_EFFECTS(VirtualMXULoadWeightOp)
ATLAS_MACHINE_EFFECTS(VirtualMXULoadAccFP8Op)
ATLAS_MACHINE_EFFECTS(VirtualMXULoadAccBF16Op)
ATLAS_MACHINE_EFFECTS(VirtualMXUResetOp)
ATLAS_MACHINE_EFFECTS(VirtualMXUAccumulateOp)
ATLAS_MACHINE_EFFECTS(VirtualMXUReadoutBF16Op)
ATLAS_MACHINE_EFFECTS(VirtualMXUReadoutFP8Op)
ATLAS_MACHINE_EFFECTS(VirtualDMALoadFP8Op)
ATLAS_MACHINE_EFFECTS(VirtualDMALoadBF16Op)
ATLAS_MACHINE_EFFECTS(VirtualDMAAwaitFP8Op)
ATLAS_MACHINE_EFFECTS(VirtualDMAAwaitBF16Op)
ATLAS_MACHINE_EFFECTS(VirtualDMAStoreFP8Op)
ATLAS_MACHINE_EFFECTS(VirtualDMAStoreBF16Op)
ATLAS_MACHINE_EFFECTS(VirtualDMAWaitOp)

void VirtualInputBF16Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void VirtualInputFP8Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void VirtualOutputBF16Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

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

static std::optional<uint32_t>
constantI32(Value value, llvm::DenseMap<Value, std::optional<uint32_t>> &constants) {
  auto [entry, inserted] = constants.try_emplace(value, std::nullopt);
  if (!inserted)
    return entry->second;
  if (!value.getType().isInteger(32))
    return std::nullopt;
  std::optional<uint32_t> result;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      result = static_cast<uint32_t>(integer.getValue().getZExtValue());
  } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
    auto lhs = constantI32(add.getLhs(), constants);
    auto rhs = constantI32(add.getRhs(), constants);
    if (lhs && rhs)
      result = static_cast<uint32_t>(static_cast<uint64_t>(*lhs) + *rhs);
  }
  constants[value] = result;
  return result;
}

static LogicalResult verifyVirtualDMAAttributes(Operation *op) {
  if (op->hasAttr("channel"))
    return op->emitOpError("virtual DMA does not select a physical channel");
  return success();
}

static LogicalResult verifyVirtualDMATransfer(Operation *op, Value dramByte,
                                             Value sizeBytes,
                                             uint32_t tileBytes) {
  if (failed(verifyVirtualDMAAttributes(op)))
    return failure();
  llvm::DenseMap<Value, std::optional<uint32_t>> constants;
  auto address = constantI32(dramByte, constants);
  auto size = constantI32(sizeBytes, constants);
  if (!address || !size)
    return op->emitOpError(
        "DRAM address and byte length must be proven by i32 arith.constant/arith.addi expressions");
  if (*size != tileBytes)
    return op->emitOpError("byte length must equal the complete tile size ")
           << tileBytes;
  if (*address < 0x80000000u || *address % 32)
    return op->emitOpError(
        "DRAM address must be a 32-byte-aligned selected-memory address");
  if (static_cast<uint64_t>(*address) + *size > (1ULL << 32))
    return op->emitOpError("DRAM transfer span exceeds the 32-bit address space");
  return success();
}

LogicalResult VirtualDMALoadFP8Op::verify() {
  return verifyVirtualDMATransfer(getOperation(), getDramByte(), getSizeBytes(), 1024);
}

LogicalResult VirtualDMALoadBF16Op::verify() {
  return verifyVirtualDMATransfer(getOperation(), getDramByte(), getSizeBytes(), 2048);
}

LogicalResult VirtualDMAAwaitFP8Op::verify() {
  return verifyVirtualDMAAttributes(getOperation());
}

LogicalResult VirtualDMAAwaitBF16Op::verify() {
  return verifyVirtualDMAAttributes(getOperation());
}

LogicalResult VirtualDMAStoreFP8Op::verify() {
  return verifyVirtualDMATransfer(getOperation(), getDramByte(), getSizeBytes(), 1024);
}

LogicalResult VirtualDMAStoreBF16Op::verify() {
  return verifyVirtualDMATransfer(getOperation(), getDramByte(), getSizeBytes(), 2048);
}

LogicalResult VirtualDMAWaitOp::verify() {
  return verifyVirtualDMAAttributes(getOperation());
}

LogicalResult VirtualInputBF16Op::verify() {
  if (getIndexAttr().getValue().getSExtValue() < 0)
    return emitOpError("input index must be nonnegative");
  return success();
}

LogicalResult VirtualInputFP8Op::verify() {
  if (getIndexAttr().getValue().getSExtValue() < 0)
    return emitOpError("input index must be nonnegative");
  return success();
}

LogicalResult VirtualMXUMatmulOp::verify() {
  return inRange(getOperation(), "unit", getUnit(), 0, 1);
}

LogicalResult VirtualScaleConstantOp::verify() {
  return inRange(getOperation(), "code", getCodeAttr().getValue().getSExtValue(),
                 0, 255);
}

LogicalResult VirtualMXULoadWeightOp::verify() {
  if (failed(inRange(getOperation(), "unit", getUnit(), 0, 1)))
    return failure();
  if (cast<VirtualMXUWeightType>(getWeight().getType()).getUnit() != getUnit())
    return emitOpError("weight handle unit must match the selected unit");
  return success();
}

static LogicalResult verifyVirtualAccLoad(Operation *op, Value acc,
                                          int64_t unit) {
  if (failed(inRange(op, "unit", unit, 0, 1)))
    return failure();
  if (cast<VirtualMXUAccType>(acc.getType()).getUnit() != unit)
    return op->emitOpError(
        "accumulator handle unit must match the selected unit");
  return success();
}

LogicalResult VirtualMXULoadAccFP8Op::verify() {
  return verifyVirtualAccLoad(getOperation(), getAcc(), getUnit());
}

LogicalResult VirtualMXULoadAccBF16Op::verify() {
  return verifyVirtualAccLoad(getOperation(), getAcc(), getUnit());
}

LogicalResult VirtualMXUResetOp::verify() {
  if (cast<VirtualMXUWeightType>(getWeight().getType()).getUnit() !=
      cast<VirtualMXUAccType>(getAcc().getType()).getUnit())
    return emitOpError("weight and accumulator must use the same MXU unit");
  return success();
}

LogicalResult VirtualMXUAccumulateOp::verify() {
  unsigned unit = cast<VirtualMXUAccType>(getAcc().getType()).getUnit();
  if (cast<VirtualMXUWeightType>(getWeight().getType()).getUnit() != unit ||
      cast<VirtualMXUAccType>(getNextAcc().getType()).getUnit() != unit)
    return emitOpError(
        "weight and accumulator versions must use the same MXU unit");
  return success();
}

LogicalResult VirtualPackFP8Op::verify() {
  return inRange(getOperation(), "scale_code", getScaleCode(), 0, 255);
}

LogicalResult VirtualOutputBF16Op::verify() {
  if (getIndexAttr().getValue().getSExtValue() < 0)
    return emitOpError("output index must be nonnegative");
  return success();
}

LogicalResult VirtualVPUUnaryOp::verify() {
  StringRef kind = getKind();
  if (kind != "mov" && kind != "recip" && kind != "exp" &&
      kind != "exp2" && kind != "square" && kind != "cube" &&
      kind != "relu" && kind != "sin" && kind != "cos" &&
      kind != "tanh" && kind != "log2" && kind != "sqrt")
    return emitOpError("unknown virtual unary VPU kind");
  return success();
}

LogicalResult VirtualVPUBinaryOp::verify() {
  StringRef kind = getKind();
  if (kind != "add" && kind != "sub" && kind != "mul" &&
      kind != "min" && kind != "max")
    return emitOpError("unknown virtual binary VPU kind");
  return success();
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

LogicalResult ALURegOp::verify() {
  StringRef kind = getKind();
  if (kind != "add" && kind != "sub" && kind != "sll" &&
      kind != "slt" && kind != "sltu" && kind != "xor" &&
      kind != "srl" && kind != "sra" && kind != "or" &&
      kind != "and")
    return emitOpError("unknown register ALU kind");
  if (failed(scalarReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "lhs", getLhs())))
    return failure();
  return scalarReg(*this, "rhs", getRhs());
}

LogicalResult ALUImmOp::verify() {
  StringRef kind = getKind();
  if (kind != "addi" && kind != "slti" && kind != "sltiu" &&
      kind != "xori" && kind != "ori" && kind != "andi" &&
      kind != "slli" && kind != "srli" && kind != "srai")
    return emitOpError("unknown immediate ALU kind");
  if (failed(scalarReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "src", getSrc())))
    return failure();
  if (kind == "slli" || kind == "srli" || kind == "srai")
    return inRange(*this, "shift amount", getImmediate(), 0, 31);
  return inRange(*this, "signed immediate",
                 getImmediateAttr().getValue().getSExtValue(), -2048, 2047);
}

LogicalResult BranchOp::verify() {
  StringRef kind = getKind();
  if (kind != "beq" && kind != "bne" && kind != "blt" &&
      kind != "bge" && kind != "bltu" && kind != "bgeu")
    return emitOpError("unknown branch kind");
  if (failed(scalarReg(*this, "lhs", getLhs())) ||
      failed(scalarReg(*this, "rhs", getRhs())))
    return failure();
  auto offset = getOffsetBytesAttr().getValue().getSExtValue();
  if (offset & 1)
    return emitOpError("branch byte offset must be even");
  return inRange(*this, "branch byte offset", offset, -4096, 4094);
}

LogicalResult JumpOp::verify() {
  if (getKind() != "jal" && getKind() != "jalr")
    return emitOpError("kind must be jal or jalr");
  if (failed(scalarReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "base", getBase())))
    return failure();
  auto offset = getOffsetAttr().getValue().getSExtValue();
  if (getKind() == "jal") {
    if (getBase() != 0)
      return emitOpError("jal has no base register; base must be zero");
    if (offset & 1)
      return emitOpError("jal byte offset must be even");
    return inRange(*this, "jal byte offset", offset, -1048576, 1048574);
  }
  return inRange(*this, "jalr word offset", offset, -2048, 2047);
}

LogicalResult DelayOp::verify() {
  return inRange(*this, "cycles", getCycles(), 0, 4095);
}

LogicalResult UpperOp::verify() {
  if (getKind() != "lui" && getKind() != "auipc")
    return emitOpError("kind must be lui or auipc");
  if (failed(scalarReg(*this, "dst", getDst())))
    return failure();
  return inRange(*this, "raw 20-bit immediate", getImmediate(), 0, 1048575);
}

LogicalResult CSROp::verify() {
  StringRef kind = getKind();
  if (kind != "rrw" && kind != "rrs" && kind != "rrc" &&
      kind != "rrwi" && kind != "rrsi" && kind != "rrci")
    return emitOpError("unknown CSR kind");
  if (failed(scalarReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "source/zimm", getSource())))
    return failure();
  // CSRFile.scala maps only these six internal addresses. An unmapped CSR
  // aliases cycle counter via its default index, so accepting it would be
  // especially misleading. Scale registers use SELI/SELD, not this CSR port.
  uint32_t address = getAddress();
  if (address != 0xc00 && address != 0xc01 && address != 0xc02 &&
      address != 0xc03 && address != 0xc10 && address != 0xc11)
    return emitOpError("CSR address is not implemented by selected CSRFile");
  if (address == 0xc02 || address == 0xc03) {
    bool readOnlyAccess = (kind == "rrs" || kind == "rrc" ||
                           kind == "rrsi" || kind == "rrci") &&
                          getSource() == 0;
    if (!readOnlyAccess)
      return emitOpError("read-only CSR requires zero-source read form");
  }
  return success();
}

LogicalResult TrapOp::verify() {
  if (getKind() != "ecall" && getKind() != "ebreak")
    return emitOpError("kind must be ecall or ebreak");
  return success();
}

LogicalResult ScalarLoadOp::verify() {
  StringRef kind = getKind();
  if (kind != "lb" && kind != "lh" && kind != "lw" &&
      kind != "lbu" && kind != "lhu" && kind != "seld" &&
      kind != "seli")
    return emitOpError("unknown scalar/scale load kind");
  if (failed(scalarReg(*this, "dst", getDst())) ||
      failed(scalarReg(*this, "base", getBase())))
    return failure();
  auto offset = getOffsetAttr().getValue().getSExtValue();
  if (kind == "seli") {
    if (getBase() != 0)
      return emitOpError("seli has no base register; base must be zero");
    return inRange(*this, "raw E8M0 code", offset, 0, 255);
  }
  return inRange(*this, "signed byte offset", offset, -2048, 2047);
}

LogicalResult ScalarStoreOp::verify() {
  StringRef kind = getKind();
  if (kind != "sb" && kind != "sh" && kind != "sw")
    return emitOpError("kind must be sb, sh, or sw");
  if (failed(scalarReg(*this, "src", getSrc())) ||
      failed(scalarReg(*this, "base", getBase())))
    return failure();
  return inRange(*this, "signed byte offset",
                 getOffsetAttr().getValue().getSExtValue(), -2048, 2047);
}

#define GET_OP_CLASSES
#include "Atlas/AtlasOps.cpp.inc"
