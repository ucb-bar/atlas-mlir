#include "Atlas/AtlasVirtualBoundaryToDMA.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Boundary slots are 2 KiB apart: an FP8 input fills the first half of its
// slot, a BF16 tile all of it.
constexpr uint64_t kSlotBytes = 2048;

uint64_t slotIndex(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
}

// One function's boundary tiles and the DRAM windows they address.
struct Boundary {
  SmallVector<Operation *> inputs, outputs;
  uint64_t inputBase = 0, outputBase = 0;
};

// Checks the boundary ABI before conversion removes the slot indices, so it
// cannot hide an overflowing or aliased window from the machine lowerer.
FailureOr<Boundary> readBoundary(func::FuncOp function) {
  Boundary boundary;
  uint64_t inputSlots = 0, outputSlots = 1;
  function.walk([&](Operation *op) {
    if (isa<VirtualInputFP8Op, VirtualInputBF16Op>(op)) {
      boundary.inputs.push_back(op);
      inputSlots = std::max(inputSlots, slotIndex(op) + 1);
    } else if (isa<VirtualOutputBF16Op>(op)) {
      boundary.outputs.push_back(op);
      outputSlots = std::max(outputSlots, slotIndex(op) + 1);
    }
  });
  if (boundary.inputs.empty() && boundary.outputs.empty())
    return boundary;
  auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
  auto output = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
  if (!input || !output)
    return function.emitOpError("boundary-to-DMA conversion requires explicit "
                                "atlas.input_dram_base and atlas.output_dram_base");
  int64_t in = input.getValue().getSExtValue();
  int64_t out = output.getValue().getSExtValue();
  if (in < 0x80000000LL || out < 0x80000000LL || in >= (1LL << 32) ||
      out >= (1LL << 32) || in % 1024 || out % 1024)
    return function.emitOpError("DRAM bases must be aligned 32-bit selected-memory addresses");
  uint64_t inputEnd = in + inputSlots * kSlotBytes;
  uint64_t outputEnd = out + outputSlots * kSlotBytes;
  if (inputEnd > (1ULL << 32) || outputEnd > (1ULL << 32) ||
      (uint64_t(in) < outputEnd && uint64_t(out) < inputEnd))
    return function.emitOpError("input and output DRAM spans overlap or overflow");
  boundary.inputBase = uint64_t(in);
  boundary.outputBase = uint64_t(out);
  return boundary;
}

// Replaces `op` with the DMA operation `name` on its state (and, for a store,
// its tile), addressing `bytes` at DRAM `address`. The state chain is kept;
// the virtual scheduler reorders it by its proven DMA alias dependencies.
void replaceWithDMA(Operation *op, StringRef name, uint64_t address, int64_t bytes) {
  OpBuilder builder(op);
  auto dram = arith::ConstantIntOp::create(builder, op->getLoc(), int64_t(address), 32);
  auto size = arith::ConstantIntOp::create(builder, op->getLoc(), bytes, 32);
  OperationState state(op->getLoc(), name);
  state.addOperands(op->getOperands());
  state.addOperands({dram.getResult(), size.getResult()});
  state.addTypes(op->getResultTypes());
  Operation *dma = builder.create(state);
  op->replaceAllUsesWith(dma);
  op->erase();
}

struct LowerAtlasVirtualBoundaryToDMAPass
    : PassWrapper<LowerAtlasVirtualBoundaryToDMAPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAtlasVirtualBoundaryToDMAPass)
  StringRef getArgument() const final { return "lower-atlas-virtual-boundary-to-dma"; }
  StringRef getDescription() const final {
    return "Expose boundary input and output transfers as schedulable virtual DMA";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    for (auto function : module.getOps<func::FuncOp>()) {
      FailureOr<Boundary> boundary = readBoundary(function);
      if (failed(boundary))
        return signalPassFailure();
      for (Operation *input : boundary->inputs) {
        bool fp8 = isa<VirtualInputFP8Op>(input);
        replaceWithDMA(input, fp8 ? "atlas.virtual_dma_load_fp8" : "atlas.virtual_dma_load_bf16",
                       boundary->inputBase + slotIndex(input) * kSlotBytes, fp8 ? 1024 : 2048);
      }
      for (Operation *output : boundary->outputs)
        replaceWithDMA(output, "atlas.virtual_dma_store_bf16",
                       boundary->outputBase + slotIndex(output) * kSlotBytes, 2048);
    }
    if (failed(verifyAtlasVirtualModule(module)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasVirtualBoundaryToDMAPass() {
  PassRegistration<LowerAtlasVirtualBoundaryToDMAPass>();
}
