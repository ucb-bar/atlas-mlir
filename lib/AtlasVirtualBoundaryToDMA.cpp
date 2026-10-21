#include "Atlas/AtlasVirtualBoundaryToDMA.h"
#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasImmutableWeightReuse.h"
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
uint64_t slotIndex(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
}

// One function's boundary tiles and the DRAM windows they address.
struct Boundary {
  SmallVector<Operation *> inputs, outputs;
  AtlasBoundaryABI abi;
};

// Checks the boundary ABI before conversion removes the slot indices, so it
// cannot hide an overflowing or aliased window from the machine lowerer.
FailureOr<Boundary> readBoundary(func::FuncOp function) {
  Boundary boundary;
  function.walk([&](Operation *op) {
    if (isa<VirtualInputFP8Op, VirtualInputBF16Op>(op))
      boundary.inputs.push_back(op);
    else if (isa<VirtualOutputBF16Op>(op))
      boundary.outputs.push_back(op);
  });
  if (boundary.inputs.empty() && boundary.outputs.empty())
    return boundary;
  FailureOr<AtlasBoundaryABI> abi = AtlasBoundaryABI::read(function);
  if (failed(abi) || failed(abi->verifySpans(function)))
    return failure();
  boundary.abi = *abi;
  return boundary;
}

// Replaces the boundary tile `op` with the DMA operation `DMAOp` on its state
// (and, for a store, its tile), addressing `bytes` at DRAM `address`. The state
// chain is kept; the virtual scheduler reorders it by its proven DMA alias
// dependencies.
template <typename DMAOp>
void replaceWithDMA(Operation *op, uint64_t address, int64_t bytes) {
  OpBuilder builder(op);
  auto dram = arith::ConstantIntOp::create(builder, op->getLoc(), int64_t(address), 32);
  auto size = arith::ConstantIntOp::create(builder, op->getLoc(), bytes, 32);
  SmallVector<Value> operands(op->getOperands());
  operands.append({dram, size});
  auto dma = DMAOp::create(builder, op->getLoc(), op->getResultTypes(), operands);
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
      reuseAtlasImmutableWeights(function);
      FailureOr<Boundary> boundary = readBoundary(function);
      if (failed(boundary))
        return signalPassFailure();
      for (Operation *input : boundary->inputs) {
        uint64_t address = boundary->abi.inputAddress(slotIndex(input));
        if (isa<VirtualInputFP8Op>(input))
          replaceWithDMA<VirtualDMALoadFP8Op>(input, address, kAtlasRegisterBytes);
        else
          replaceWithDMA<VirtualDMALoadBF16Op>(input, address, kAtlasBF16TileBytes);
      }
      for (Operation *output : boundary->outputs)
        replaceWithDMA<VirtualDMAStoreBF16Op>(
            output, boundary->abi.outputAddress(slotIndex(output)), kAtlasBF16TileBytes);
    }
    if (failed(verifyAtlasVirtualModule(module)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasVirtualBoundaryToDMAPass() {
  PassRegistration<LowerAtlasVirtualBoundaryToDMAPass>();
}

std::unique_ptr<Pass> mlir::atlas::createLowerAtlasVirtualBoundaryToDMAPass() {
  return std::make_unique<LowerAtlasVirtualBoundaryToDMAPass>();
}
