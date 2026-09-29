#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::atlas;

static uint32_t vr(unsigned funct7, unsigned vd, unsigned vs1,
                   unsigned vs2, unsigned opcode) {
  // ScalarDecoder.scala: vd[12:7], vs1[18:13], vs2[24:19].
  return (funct7 << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | opcode;
}

static std::optional<unsigned> valueFor(StringRef key,
                                         std::initializer_list<std::pair<StringRef, unsigned>> values) {
  for (auto [name, value] : values)
    if (key == name)
      return value;
  return std::nullopt;
}

static FailureOr<uint32_t> encode(Operation *op) {
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
  return failure();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: atlas-emit <flat-mlir-module>\n";
    return 2;
  }
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module))) return 1;

  Value previous;
  bool started = false;
  llvm::SmallVector<uint32_t> words;
  for (Operation &op : module->getBody()->getOperations()) {
    if (isa<StartOp>(op)) {
      if (started || previous) {
        op.emitError("one atlas.start is required at the beginning");
        return 1;
      }
      started = true;
      previous = op.getResult(0);
      continue;
    }
    if (!started || op.getNumOperands() != 1 ||
        op.getOperand(0) != previous || op.getNumResults() != 1) {
      op.emitError("expected a linear Atlas state chain");
      return 1;
    }
    auto word = encode(&op);
    if (failed(word)) {
      op.emitError("has no selected RTL encoding");
      return 1;
    }
    words.push_back(*word);
    previous = op.getResult(0);
  }
  if (!started) {
    llvm::errs() << "module has no atlas.start\n";
    return 1;
  }
  for (uint32_t word : words)
    llvm::outs() << llvm::format_hex_no_prefix(word, 8) << '\n';
  return 0;
}
