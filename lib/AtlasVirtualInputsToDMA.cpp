#include "Atlas/AtlasVirtualInputsToDMA.h"
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
struct LowerAtlasVirtualInputsToDMAPass
    : PassWrapper<LowerAtlasVirtualInputsToDMAPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAtlasVirtualInputsToDMAPass)
  StringRef getArgument() const final { return "lower-atlas-virtual-inputs-to-dma"; }
  StringRef getDescription() const final {
    return "Expose boundary input transfers as schedulable virtual DMA loads";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    // Check the original boundary ABI before removing input indices. In
    // particular, conversion must not hide an overflowing or aliased window
    // from the machine lowerer's ABI checks.
    for (auto function : module.getOps<func::FuncOp>()) {
      SmallVector<Operation *> inputs;
      uint64_t inputSlots = 0, outputSlots = 1;
      function.walk([&](Operation *op) {
        if (isa<VirtualInputFP8Op, VirtualInputBF16Op>(op)) {
          inputs.push_back(op);
          inputSlots = std::max(inputSlots,
              op->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue() + 1);
        }
        if (auto output = dyn_cast<VirtualOutputBF16Op>(op))
          outputSlots = std::max(outputSlots,
              output.getIndexAttr().getValue().getZExtValue() + 1);
      });
      if (inputs.empty())
        continue;
      auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
      auto output = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
      if (!input || !output) {
        function.emitOpError("input-to-DMA conversion requires explicit atlas.input_dram_base and atlas.output_dram_base");
        return signalPassFailure();
      }
      int64_t in = input.getValue().getSExtValue();
      int64_t out = output.getValue().getSExtValue();
      if (in < 0x80000000LL || out < 0x80000000LL ||
          in >= (1LL << 32) || out >= (1LL << 32) ||
          in % 1024 || out % 1024) {
        function.emitOpError("DRAM bases must be aligned 32-bit selected-memory addresses");
        return signalPassFailure();
      }
      uint64_t inputEnd = in + inputSlots * 2048;
      uint64_t outputEnd = out + outputSlots * 2048;
      if (inputEnd > (1ULL << 32) || outputEnd > (1ULL << 32) ||
          (static_cast<uint64_t>(in) < outputEnd &&
           static_cast<uint64_t>(out) < inputEnd)) {
        function.emitOpError("input and output DRAM spans overlap or overflow");
        return signalPassFailure();
      }
      for (Operation *op : inputs) {
        OpBuilder builder(op);
        bool fp8 = isa<VirtualInputFP8Op>(op);
        uint64_t index = op->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
        auto address = arith::ConstantIntOp::create(
            builder, op->getLoc(), static_cast<int64_t>(in + index * 2048), 32);
        auto size = arith::ConstantIntOp::create(
            builder, op->getLoc(), fp8 ? 1024 : 2048, 32);
        // Keep the state chain. The virtual scheduler rewrites it according
        // to its proven DMA alias dependencies; no ordering is dropped here.
        OperationState state(op->getLoc(), fp8 ? "atlas.virtual_dma_load_fp8"
                                              : "atlas.virtual_dma_load_bf16");
        state.addOperands({op->getOperand(0), address.getResult(), size.getResult()});
        state.addTypes(op->getResultTypes());
        Operation *load = builder.create(state);
        op->replaceAllUsesWith(load);
        op->erase();
      }
    }
    if (failed(verifyAtlasVirtualModule(module)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasVirtualInputsToDMAPass() {
  PassRegistration<LowerAtlasVirtualInputsToDMAPass>();
}
