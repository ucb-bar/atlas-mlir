#include "Atlas/AtlasImmutableWeightReuse.h"
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
};

bool overlaps(Span a, Span b) {
  return a.first < b.second && b.first < a.second;
}

bool onlyWeightUses(Value value) {
  return !value.use_empty() && llvm::all_of(value.getUses(), [](OpOperand &use) {
    return use.getOperandNumber() == 1 &&
           isa<VirtualMXUMatmulOp, VirtualMXULoadWeightOp>(use.getOwner());
  });
}

std::optional<Span> dmaSpan(Operation *op, bool store) {
  auto address = provenI32(op->getOperand(store ? 2 : 1));
  auto size = provenI32(op->getOperand(store ? 3 : 2));
  if (!address || !size || uint64_t(*address) + *size > (1ULL << 32))
    return std::nullopt;
  return Span{*address, uint64_t(*address) + *size};
}

// Validate the complete original boundary spans before considering reuse.
// This does not replace the lowering ABI checks (including VMEM placement).
bool boundaryBases(func::FuncOp function, uint64_t &input, uint64_t &output) {
  uint64_t inputSlots = 1, outputSlots = 1;
  bool hasBoundary = false;
  for (Operation &op : function.front()) {
    if (isa<VirtualInputFP8Op, VirtualInputBF16Op, VirtualOutputBF16Op>(op)) {
      hasBoundary = true;
      uint64_t slots = op.getAttrOfType<IntegerAttr>("index").getValue().getZExtValue() + 1;
      auto &maximum = isa<VirtualOutputBF16Op>(op) ? outputSlots : inputSlots;
      maximum = std::max(maximum, slots);
    }
  }
  if (!hasBoundary)
    return true;
  auto in = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
  auto out = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
  if (!in || !out)
    return false;
  int64_t a = in.getValue().getSExtValue(), b = out.getValue().getSExtValue();
  if (a < 0x80000000LL || b < 0x80000000LL || a >= (1LL << 32) ||
      b >= (1LL << 32) || a % 1024 || b % 1024)
    return false;
  input = a;
  output = b;
  Span reads{input, input + inputSlots * 2048};
  Span writes{output, output + outputSlots * 2048};
  return reads.second <= (1ULL << 32) && writes.second <= (1ULL << 32) &&
         !overlaps(reads, writes);
}

SmallVector<Input> candidates(func::FuncOp function) {
  // The required virtual-module verification has a closed operation whitelist:
  // only the boundary output and explicit DMA stores below can write DRAM.
  // Extend this effect scan when admitting another DRAM-writing operation to
  // that verifier; an unclassified write must prevent reuse.
  SmallVector<Input> inputs;
  SmallVector<Span> writes;
  uint64_t inputBase = 0, outputBase = 0;
  if (!boundaryBases(function, inputBase, outputBase))
    return inputs;
  for (Operation &op : function.front()) {
    if (isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
      auto span = dmaSpan(&op, true);
      if (!span)
        return {};
      writes.push_back(*span);
    } else if (auto output = dyn_cast<VirtualOutputBF16Op>(op)) {
      uint64_t first = outputBase + output.getIndexAttr().getValue().getZExtValue() * 2048;
      writes.push_back({first, first + 2048});
    } else if (auto input = dyn_cast<VirtualInputFP8Op>(op)) {
      if (onlyWeightUses(input.getValue())) {
        uint64_t index = input.getIndexAttr().getValue().getZExtValue();
        uint64_t first = inputBase + index * 2048;
        inputs.push_back({&op, {false, index}, {first, first + 1024}});
      }
    } else if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      // Restrict to direct constants, so removing a transfer can clean up
      // precisely its newly dead address/size constants, without global DCE.
      if (!onlyWeightUses(load.getValue()) ||
          !load.getDramByte().getDefiningOp<arith::ConstantOp>() ||
          !load.getSizeBytes().getDefiningOp<arith::ConstantOp>())
        continue;
      auto span = dmaSpan(&op, false);
      if (span && span->second - span->first == 1024)
        inputs.push_back({&op, {true, span->first}, *span});
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
  for (const Input &input : inputs) {
    Operation *op = input.op;
    auto [entry, first] = seen.emplace(input.key, op->getResult(1));
    if (first)
      continue;
    SmallVector<Operation *> constants;
    if (isa<VirtualDMALoadFP8Op>(op))
      for (unsigned operand : {1u, 2u}) {
        Operation *constant = op->getOperand(operand).getDefiningOp();
        if (!llvm::is_contained(constants, constant))
          constants.push_back(constant);
      }
    op->getResult(1).replaceAllUsesWith(entry->second);
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
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
