#include "Atlas/AtlasImmutableWeightReuse.h"
#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;

namespace {
using Span = std::pair<uint64_t, uint64_t>;
// Do not merge boundary and explicit loads: erasing the last boundary at an
// index must never conceal the original lowerer's ABI/scratch bounds checks.
using Key = std::pair<bool, uint64_t>;
struct Input {
  Operation *op;
  Key key;
  Span span;
  // Its state operand, which removing an earlier duplicate redirects, and
  // its results.
  OpOperand *state;
  Value next, value;
};

bool overlaps(Span a, Span b) {
  return a.first < b.second && b.first < a.second;
}

bool onlyWeightUses(Value value) {
  return !value.use_empty() && llvm::all_of(value.getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(user))
      return &use == &matmul.getWeightMutable();
    if (auto load = dyn_cast<VirtualMXULoadWeightOp>(user))
      return &use == &load.getSrcMutable();
    return false;
  });
}

std::optional<Span> dmaSpan(Operation *op) {
  std::optional<VirtualDMATransfer> transfer = virtualDMATransfer(op);
  auto address = provenI32(transfer->dram->get());
  auto size = provenI32(transfer->size->get());
  if (!address || !size || uint64_t(*address) + *size > (1ULL << 32))
    return std::nullopt;
  return Span{*address, uint64_t(*address) + *size};
}

SmallVector<Input> candidates(func::FuncOp function) {
  // The required virtual-module verification has a closed operation whitelist:
  // only the boundary output and explicit DMA stores below can write DRAM.
  // Extend this effect scan when admitting another DRAM-writing operation to
  // that verifier; an unclassified write must prevent reuse.
  SmallVector<Input> inputs;
  SmallVector<Span> writes;
  // Validate the complete original boundary spans before considering reuse.
  // This does not replace the lowering ABI checks (including VMEM placement).
  AtlasBoundaryABI abi;
  if (hasAtlasBoundaryTiles(function)) {
    FailureOr<AtlasBoundaryABI> read = AtlasBoundaryABI::read(function, /*diagnose=*/false);
    if (failed(read) || failed(read->verifySpans(function, /*diagnose=*/false)))
      return inputs;
    abi = *read;
  }
  for (Operation &op : function.front()) {
    if (isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
      auto span = dmaSpan(&op);
      if (!span)
        return {};
      writes.push_back(*span);
    } else if (auto output = dyn_cast<VirtualOutputBF16Op>(op)) {
      uint64_t first = abi.outputAddress(output.getIndexAttr().getValue().getZExtValue());
      writes.push_back({first, first + kAtlasBF16TileBytes});
    } else if (auto input = dyn_cast<VirtualInputFP8Op>(op)) {
      if (onlyWeightUses(input.getValue())) {
        uint64_t index = input.getIndexAttr().getValue().getZExtValue();
        uint64_t first = abi.inputAddress(index);
        inputs.push_back({&op, {false, index}, {first, first + kAtlasRegisterBytes},
                          &input.getStateMutable(), input.getNext(), input.getValue()});
      }
    } else if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      // Restrict to direct constants, so removing a transfer can clean up
      // precisely its newly dead address/size constants, without global DCE.
      if (!onlyWeightUses(load.getValue()) ||
          !load.getDramByte().getDefiningOp<arith::ConstantOp>() ||
          !load.getSizeBytes().getDefiningOp<arith::ConstantOp>())
        continue;
      auto span = dmaSpan(&op);
      if (span && span->second - span->first == kAtlasRegisterBytes)
        inputs.push_back({&op, {true, span->first}, *span, &load.getStateMutable(),
                          load.getNext(), load.getValue()});
    }
  }
  llvm::erase_if(inputs, [&](const Input &input) {
    return llvm::any_of(writes, [&](Span write) { return overlaps(input.span, write); });
  });
  return inputs;
}

bool hasRepeat(ArrayRef<Input> inputs) {
  std::map<Key, unsigned> seen;
  for (const Input &input : inputs)
    if (++seen[input.key] > 1)
      return true;
  return false;
}

void rewrite(ArrayRef<Input> inputs) {
  std::map<Key, Value> seen;
  for (Input input : inputs) {
    Operation *op = input.op;
    auto [entry, first] = seen.emplace(input.key, input.value);
    if (first)
      continue;
    SmallVector<Operation *> constants;
    if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op))
      for (Value operand : {load.getDramByte(), load.getSizeBytes()}) {
        Operation *constant = operand.getDefiningOp();
        if (!llvm::is_contained(constants, constant))
          constants.push_back(constant);
      }
    input.value.replaceAllUsesWith(entry->second);
    input.next.replaceAllUsesWith(input.state->get());
    op->erase();
    for (Operation *constant : constants)
      if (constant->use_empty())
        constant->erase();
  }
}

struct ReuseAtlasImmutableWeightsPass
    : PassWrapper<ReuseAtlasImmutableWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseAtlasImmutableWeightsPass)
  StringRef getArgument() const final { return "reuse-atlas-immutable-weights"; }
  StringRef getDescription() const final {
    return "Reuse provably unwritten FP8 weight inputs across token tiles when they allocate";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      reuseAtlasImmutableWeights(function);
    if (failed(verifyAtlasVirtualModule(module)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::reuseAtlasImmutableWeights(func::FuncOp function) {
  if (!llvm::hasSingleElement(function.getBody()) || !hasRepeat(candidates(function)))
    return;
  OwningOpRef<func::FuncOp> trial(cast<func::FuncOp>(function->clone()));
  rewrite(candidates(*trial));
  // Reuse extends register lifetimes. Trial the actual greedy allocator and
  // its independent checks rather than assuming a cached-weight count fits.
  ScopedDiagnosticHandler suppress(function.getContext(), [](Diagnostic &) {
    return success();
  });
  VirtualAllocationPlan plan;
  if (failed(plan.allocate(*trial)) || failed(plan.verify()))
    return;
  function.getBody().takeBody(trial->getBody());
}

void mlir::atlas::registerReuseAtlasImmutableWeightsPass() {
  PassRegistration<ReuseAtlasImmutableWeightsPass>();
}
