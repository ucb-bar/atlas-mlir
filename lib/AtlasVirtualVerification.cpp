#include "Atlas/AtlasVirtualVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
struct VerifyAtlasVirtualStreamPass
    : PassWrapper<VerifyAtlasVirtualStreamPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasVirtualStreamPass)

  StringRef getArgument() const final { return "verify-atlas-virtual-stream"; }
  StringRef getDescription() const final {
    return "Check pre-allocation BF16 SSA values, boundary order, and stage isolation";
  }

  void runOnOperation() override {
    Value state;
    llvm::DenseSet<int64_t> outputIndices;
    unsigned starts = 0;
    unsigned outputs = 0;
    for (Operation &operation : getOperation().getBody()->getOperations()) {
      if (auto start = dyn_cast<VirtualStartOp>(operation)) {
        if (++starts != 1 || state) {
          start.emitOpError("virtual stream requires exactly one start");
          return signalPassFailure();
        }
        state = start.getNext();
        continue;
      }
      if (auto input = dyn_cast<VirtualInputBF16Op>(operation)) {
        if (!state || input.getState() != state) {
          input.emitOpError("nonlinear virtual state chain");
          return signalPassFailure();
        }
        state = input.getNext();
        continue;
      }
      if (auto output = dyn_cast<VirtualOutputBF16Op>(operation)) {
        if (!state || output.getState() != state) {
          output.emitOpError("nonlinear virtual state chain");
          return signalPassFailure();
        }
        int64_t index = output.getIndexAttr().getValue().getSExtValue();
        if (!outputIndices.insert(index).second) {
          output.emitOpError("duplicate virtual output index ") << index;
          return signalPassFailure();
        }
        state = output.getNext();
        ++outputs;
        continue;
      }
      if (isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp>(operation))
        continue;
      operation.emitOpError("operation is outside the virtual Atlas stage");
      return signalPassFailure();
    }
    if (starts != 1 || outputs == 0) {
      getOperation().emitOpError(
          "virtual stream requires one start and at least one output");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::atlas::registerVerifyAtlasVirtualStreamPass() {
  PassRegistration<VerifyAtlasVirtualStreamPass>();
}
