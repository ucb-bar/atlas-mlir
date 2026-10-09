#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <optional>

using namespace mlir;
using namespace mlir::atlas;

static uint32_t vr(unsigned funct7, unsigned vd, unsigned vs1,
                   unsigned vs2, unsigned opcode) {
  // ScalarDecoder.scala: vd[12:7], vs1[18:13], vs2[24:19].
  return (funct7 << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | opcode;
}

static uint32_t r(unsigned funct7, unsigned rd, unsigned rs1,
                  unsigned rs2, unsigned funct3, unsigned opcode) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t i(unsigned imm, unsigned rd, unsigned rs1,
                  unsigned funct3, unsigned opcode) {
  return ((imm & 0xfff) << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t s(unsigned imm, unsigned rs1, unsigned rs2,
                  unsigned funct3, unsigned opcode) {
  imm &= 0xfff;
  return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | ((imm & 31) << 7) | opcode;
}

static uint32_t b(unsigned imm, unsigned rs1, unsigned rs2,
                  unsigned funct3) {
  imm &= 0x1fff;
  return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 63) << 25) |
         (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (((imm >> 1) & 15) << 8) | (((imm >> 11) & 1) << 7) | 0x63;
}

static uint32_t j(unsigned imm, unsigned rd) {
  imm &= 0x1fffff;
  return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 1023) << 21) |
         (((imm >> 11) & 1) << 20) | (((imm >> 12) & 255) << 12) |
         (rd << 7) | 0x6f;
}

static std::optional<unsigned> valueFor(StringRef key,
                                         std::initializer_list<std::pair<StringRef, unsigned>> values) {
  for (auto [name, value] : values)
    if (key == name)
      return value;
  return std::nullopt;
}

FailureOr<uint32_t> mlir::atlas::encodeMachineWord(Operation *op) {
  if (auto x = dyn_cast<VLoadOp>(op))
    return ((x.getOffset() & 0xfff) << 20) | (x.getBase() << 15) |
           (x.getDst() << 7) | 0x07;
  if (auto x = dyn_cast<VStoreOp>(op))
    return ((x.getOffset() & 0xfff) << 20) | (x.getBase() << 15) |
           (1 << 13) | (x.getSrc() << 7) | 0x07;
  if (auto x = dyn_cast<DMAOp>(op)) {
    unsigned isStore = x.getDirection() == "store";
    unsigned rd = isStore ? x.getDram() : x.getReg();
    unsigned rs1 = isStore ? x.getReg() : x.getDram();
    return (isStore << 25) | (x.getSize() << 20) | (rs1 << 15) |
           (x.getChannel() << 12) | (rd << 7) | 0x7b;
  }
  if (auto x = dyn_cast<DMAWaitOp>(op))
    return (1u << 25) | (x.getChannel() << 12) | 0x7f;
  if (auto x = dyn_cast<DMAConfigOp>(op))
    return (x.getBaseReg() << 15) | (x.getChannel() << 12) | 0x7f;
  if (auto x = dyn_cast<MXUPushOp>(op)) {
    auto base = valueFor(x.getKind(), {{"weight_fp8", 0}, {"acc_fp8", 2},
                                       {"acc_bf16", 4}});
    if (!base) return failure();
    return vr(*base + x.getUnit(), x.getSlot(), x.getSrc(), 0, 0x77);
  }
  if (auto x = dyn_cast<MXUMatmulOp>(op)) {
    unsigned f7 = (x.getAccumulate() ? 12 : 10) + x.getUnit();
    return vr(f7, x.getAccSlot(), x.getSrc(), x.getWeightSlot(), 0x77);
  }
  if (auto x = dyn_cast<MXUPopOp>(op)) {
    unsigned f7 = (x.getFormat() == "bf16" ? 8 : 6) + x.getUnit();
    unsigned scale = x.getFormat() == "fp8" ? x.getScaleReg() : 0;
    return vr(f7, x.getDst(), scale, x.getSlot(), 0x77);
  }
  if (auto x = dyn_cast<VPUBinaryOp>(op)) {
    auto f7 = valueFor(x.getKind(), {{"add", 0}, {"sub", 2}, {"mul", 3},
                                      {"min", 4}, {"max", 6}});
    if (!f7) return failure();
    return vr(*f7, x.getDst(), x.getLhs(), x.getRhs(), 0x57);
  }
  if (auto x = dyn_cast<VPUUnaryOp>(op)) {
    auto f7 = valueFor(x.getKind(), {{"mov", 0x40}, {"recip", 0x41},
                                      {"exp", 0x42}, {"exp2", 0x43},
                                      {"square", 0x46}, {"cube", 0x47},
                                      {"relu", 0x48}, {"sin", 0x49},
                                      {"cos", 0x4a}, {"tanh", 0x4b},
                                      {"log2", 0x4c}, {"sqrt", 0x4d}});
    if (!f7) return failure();
    return vr(*f7, x.getDst(), x.getSrc(), 0, 0x57);
  }
  if (auto x = dyn_cast<VPUPackOp>(op)) {
    unsigned f7 = x.getDirection() == "bf16_to_fp8" ? 0x44 : 0x45;
    return vr(f7, x.getDst(), x.getScaleReg(), x.getSrc(), 0x57);
  }
  if (auto x = dyn_cast<VPUReduceOp>(op)) {
    auto f7 = valueFor(x.getKind(), {{"col_sum", 1}, {"col_min", 5},
                                      {"col_max", 7}, {"row_sum", 0x21},
                                      {"row_min", 0x24}, {"row_max", 0x26}});
    if (!f7) return failure();
    return vr(*f7, x.getDst(), x.getSrc(), 0, 0x57);
  }
  if (auto x = dyn_cast<VLIOp>(op)) {
    auto mode = valueFor(x.getMode(), {{"all", 0}, {"row", 1},
                                       {"col", 2}, {"one", 3}});
    if (!mode) return failure();
    return (x.getImmediate() << 16) | (*mode << 13) |
           (x.getDst() << 7) | 0x5f;
  }
  if (auto x = dyn_cast<XLUTransposeOp>(op))
    return vr(0, x.getDst(), x.getSrc(), 0, 0x6b);
  if (auto x = dyn_cast<ALURegOp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"add", 0}, {"sub", 0},
                                           {"sll", 1}, {"slt", 2},
                                           {"sltu", 3}, {"xor", 4},
                                           {"srl", 5}, {"sra", 5},
                                           {"or", 6}, {"and", 7}});
    if (!funct3) return failure();
    unsigned funct7 = x.getKind() == "sub" || x.getKind() == "sra" ? 0x20 : 0;
    return r(funct7, x.getDst(), x.getLhs(), x.getRhs(), *funct3, 0x33);
  }
  if (auto x = dyn_cast<ALUImmOp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"addi", 0}, {"slli", 1},
                                           {"slti", 2}, {"sltiu", 3},
                                           {"xori", 4}, {"srli", 5},
                                           {"srai", 5}, {"ori", 6},
                                           {"andi", 7}});
    if (!funct3) return failure();
    unsigned imm = x.getImmediate();
    if (x.getKind() == "srai") imm |= 0x400;
    return i(imm, x.getDst(), x.getSrc(), *funct3, 0x13);
  }
  if (auto x = dyn_cast<BranchOp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"beq", 0}, {"bne", 1},
                                           {"blt", 4}, {"bge", 5},
                                           {"bltu", 6}, {"bgeu", 7}});
    if (!funct3) return failure();
    return b(x.getOffsetBytes(), x.getLhs(), x.getRhs(), *funct3);
  }
  if (auto x = dyn_cast<JumpOp>(op)) {
    if (x.getKind() == "jal") return j(x.getOffset(), x.getDst());
    return i(x.getOffset(), x.getDst(), x.getBase(), 0, 0x67);
  }
  if (auto x = dyn_cast<DelayOp>(op))
    return i(x.getCycles(), 0, 0, 1, 0x67);
  if (auto x = dyn_cast<UpperOp>(op))
    return (x.getImmediate() << 12) | (x.getDst() << 7) |
           (x.getKind() == "lui" ? 0x37 : 0x17);
  if (auto x = dyn_cast<CSROp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"rrw", 1}, {"rrs", 2},
                                           {"rrc", 3}, {"rrwi", 5},
                                           {"rrsi", 6}, {"rrci", 7}});
    if (!funct3) return failure();
    return i(x.getAddress(), x.getDst(), x.getSource(), *funct3, 0x73);
  }
  if (auto x = dyn_cast<TrapOp>(op))
    return x.getKind() == "ecall" ? 0x00000073u : 0x00100073u;
  if (isa<FenceOp>(op)) return 0x0000000fu;
  if (auto x = dyn_cast<ScalarLoadOp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"lb", 0}, {"lh", 1},
                                           {"lw", 2}, {"lbu", 4},
                                           {"lhu", 5}, {"seld", 6},
                                           {"seli", 7}});
    if (!funct3) return failure();
    return i(x.getOffset(), x.getDst(), x.getBase(), *funct3, 0x03);
  }
  if (auto x = dyn_cast<ScalarStoreOp>(op)) {
    auto funct3 = valueFor(x.getKind(), {{"sb", 0}, {"sh", 1}, {"sw", 2}});
    if (!funct3) return failure();
    return s(x.getOffset(), x.getBase(), x.getSrc(), *funct3, 0x23);
  }
  return failure();
}

// ScalarCore.scala uses the scalar register value plus a signed I-immediate
// as a *word-index* PC for JALR. The single LLVM inline-assembly block can
// only contain the jump if that register value is proven from the linear
// prefix. This deliberately recognizes only synchronous LUI/ADDI constant
// materialization. Prior redirects, asynchronous scalar loads, and unknown
// effects make the proof inconclusive rather than silently accepting a
// possibly escaping target.
static FailureOr<uint32_t> proveJalrTarget(
    llvm::ArrayRef<Operation *> ops, size_t index, JumpOp jump) {
  std::array<std::optional<uint32_t>, 32> values{};
  values[0] = 0;
  auto write = [&](unsigned dst, std::optional<uint32_t> value) {
    if (dst != 0) values[dst] = value;
  };
  for (size_t i = 0; i < index; ++i) {
    Operation *op = ops[i];
    if (isa<BranchOp, JumpOp, TrapOp, CSROp, ScalarLoadOp>(op)) {
      jump.emitError("JALR target proof cannot cross earlier control flow, CSR effects, or an asynchronous scalar load");
      return failure();
    }
    if (auto alu = dyn_cast<ALUImmOp>(op)) {
      std::optional<uint32_t> result;
      if (alu.getKind() == "addi" && values[alu.getSrc()]) {
        auto immediate = alu.getImmediateAttr().getValue().getSExtValue();
        result = *values[alu.getSrc()] + static_cast<uint32_t>(immediate);
      }
      write(alu.getDst(), result);
    } else if (auto upper = dyn_cast<UpperOp>(op)) {
      write(upper.getDst(), upper.getKind() == "lui"
                                 ? std::optional<uint32_t>(upper.getImmediate() << 12)
                                 : std::nullopt);
    } else if (auto alu = dyn_cast<ALURegOp>(op)) {
      write(alu.getDst(), std::nullopt);
    } else if (!isa<DelayOp, FenceOp>(op)) {
      jump.emitError("JALR target proof has an unmodeled preceding instruction effect");
      return failure();
    }
  }
  if (!values[jump.getBase()]) {
    jump.emitError("JALR base register has no proven constant word-index value");
    return failure();
  }
  auto offset = jump.getOffsetAttr().getValue().getSExtValue();
  int64_t target = static_cast<int64_t>(*values[jump.getBase()]) + offset;
  if (target < 0 || target >= static_cast<int64_t>(ops.size())) {
    jump.emitError("JALR register-indirect target escapes the LLVM inline assembly block");
    return failure();
  }
  return static_cast<uint32_t>(target);
}

LogicalResult mlir::atlas::collectAtlasWords(
    ModuleOp module, llvm::SmallVectorImpl<uint32_t> &words, bool llvmBlock,
    bool skipGeneratedCheck) {
  if (failed(verify(module)) || failed(verifyAtlasTimingState(module))) return failure();
  bool generated =
      module->hasAttr("atlas.generated_from_virtual") || module->hasAttr("atlas.virtual_dma_contract") ||
       module->hasAttr("atlas.virtual_mxu_contract") || module->hasAttr("atlas.virtual_tile_contract") ||
       llvm::any_of(module.getBody()->getOperations(), [](Operation &op) {
         return op.hasAttr("atlas.virtual_mxu_command") || op.hasAttr("atlas.virtual_dma_transfer") || op.hasAttr("atlas.virtual_tile_command");
       });
  if (!skipGeneratedCheck && generated &&
      failed(verifyAtlasGeneratedSchedule(module)))
    return failure();
  auto timingState = module->getAttrOfType<StringAttr>("atlas.timing_state");
  if (!skipGeneratedCheck && !generated && timingState &&
      timingState.getValue() == "timed" &&
      failed(verifyAtlasTiming(module, timing::footprintOf)))
    return failure();
  llvm::SmallVector<uint32_t> collected;
  llvm::SmallVector<Operation *> encodedOps;
  Value previous;
  bool started = false;
  bool needsDelaySlot = false;
  for (Operation &op : module.getBody()->getOperations()) {
    if (isa<StartOp>(op)) {
      if (started || previous || !collected.empty()) {
        op.emitError("one atlas.start is required at the beginning");
        return failure();
      }
      started = true;
      previous = op.getResult(0);
      continue;
    }
    if (!started || op.getNumRegions() != 0 || op.getNumOperands() != 1 ||
        op.getOperand(0) != previous || op.getNumResults() != 1) {
      op.emitError("expected a flat, linear Atlas state chain");
      return failure();
    }
    bool redirects = isa<BranchOp, JumpOp>(op);
    if (needsDelaySlot && redirects) {
      op.emitError("branch or jump in selected RTL delay slot");
      return failure();
    }
    needsDelaySlot = redirects;
    auto word = encodeMachineWord(&op);
    if (failed(word)) {
      op.emitError("has no selected RTL encoding");
      return failure();
    }
    collected.push_back(*word);
    encodedOps.push_back(&op);
    previous = op.getResult(0);
  }
  if (!started || collected.empty()) {
    module.emitError("module requires atlas.start and at least one encoded instruction");
    return failure();
  }
  if (needsDelaySlot) {
    module.emitError("branch or jump lacks its required delay-slot instruction");
    return failure();
  }
  if (llvmBlock) {
    for (auto [index, op] : llvm::enumerate(encodedOps)) {
      if (auto jump = dyn_cast<JumpOp>(op)) {
        if (jump.getKind() == "jalr") {
          if (failed(proveJalrTarget(encodedOps, index, jump)))
            return failure();
          continue;
        }
      }
      int64_t offsetBytes = 0;
      // The generated I32 accessors expose raw unsigned bits. Interpret the
      // encoded displacement as signed before validating backward targets.
      if (auto branch = dyn_cast<BranchOp>(op))
        offsetBytes = branch.getOffsetBytesAttr().getValue().getSExtValue();
      else if (auto jump = dyn_cast<JumpOp>(op))
        offsetBytes = jump.getOffsetAttr().getValue().getSExtValue();
      else
        continue;
      // ScalarCore.scala:269-270 applies the encoded byte displacement >> 1
      // to its instruction-index PC. All targets must stay in this asm block.
      int64_t target = static_cast<int64_t>(index) + offsetBytes / 2;
      if (target < 0 || target >= static_cast<int64_t>(collected.size())) {
        op->emitError("PC-relative target escapes the LLVM inline assembly block");
        return failure();
      }
    }
  }
  words.append(collected.begin(), collected.end());
  return success();
}
