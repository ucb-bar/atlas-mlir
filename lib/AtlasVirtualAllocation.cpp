#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr unsigned kTensorPairTemporary = 62;
constexpr unsigned kFp8Registers = 32; // m0-m31 when FP8 values are present
constexpr unsigned kFirstScalarValue = 10;
constexpr unsigned kFirstScalarValueWithPack = 18; // x10-x17 serve the pack
constexpr unsigned kLastScalarValue = 26;

FixedResourcePlacement selectedResources() {
  FixedResourcePlacement resources;
  resources.tensorTemporary = kTensorPairTemporary;
  resources.scalarTemporary = 27;
  resources.oneReg = 28;
  resources.zeroReg = 5;
  resources.halfSizeReg = 2;
  resources.haltReg = 1;
  resources.inputBaseReg = 6;
  resources.inputDramReg = 1;
  resources.outputBaseReg = 8;
  resources.outputDramReg = 3;
  resources.loadChannel = 0;
  resources.storeChannel = 1;
  resources.inputWord = 0;
  resources.outputWord = 65536;
  resources.mailboxWord = 0;
  resources.inputWindowWords = 65536;
  resources.outputWindowWords = 65536;
  resources.packWord = 32768;
  resources.packRelayoutWord = 33024;
  resources.stagingWord = 131072;
  resources.dmaBaseReg = 4;
  resources.dmaDramReg = 7;
  resources.dmaSizeReg = 9;
  resources.scaleReg = 3;
  resources.packSourceRegs = {10, 11};
  resources.packDestinationReg = 12;
  resources.packRowReg = 13;
  resources.packRowsReg = 14;
  resources.packTemporaryRegs = {16, 17};
  resources.mxuWeightSlot = 0;
  resources.mxuAccSlot = 0;
  return resources;
}
} // namespace

RegisterBudget mlir::atlas::registerBudget(RegisterKind kind, bool mixedFp8,
                                          bool hasPack) {
  switch (kind) {
  case RegisterKind::BF16: {
    unsigned first = mixedFp8 ? kFp8Registers : 0;
    return {first, (kTensorPairTemporary - first) / 2, 2};
  }
  case RegisterKind::FP8:
    return {0, kFp8Registers, 1};
  case RegisterKind::Scalar: {
    unsigned first = hasPack ? kFirstScalarValueWithPack : kFirstScalarValue;
    return {first, kLastScalarValue - first + 1, 1};
  }
  }
  llvm_unreachable("unknown register kind");
}

VirtualAllocationPlan::VirtualAllocationPlan()
    : fixedResources(selectedResources()) {}

LogicalResult VirtualAllocationPlan::placeResources(func::FuncOp function) {
  this->function = function;
  tileRegs.clear();
  fp8Regs.clear();
  scalarRegs.clear();
  mxuResources.clear();
  dmaTransfers.clear();
  scalarArgumentRegs.clear();
  mixedFp8 = false;
  hasPack = false;
  function.walk([&](Operation *op) {
    hasPack |= isa<VirtualPackFP8Op>(op);
    for (Value result : op->getResults()) {
      mixedFp8 |= isa<VirtualFP8Type>(result.getType());
      // Verified lifetimes permit slot 0 in each unit's weight/accumulator bank.
      if (auto weight = dyn_cast<VirtualMXUWeightType>(result.getType()))
        mxuResources[result] = {weight.getUnit(), fixedResources.mxuWeightSlot};
      if (auto acc = dyn_cast<VirtualMXUAccType>(result.getType()))
        mxuResources[result] = {acc.getUnit(), fixedResources.mxuAccSlot};
    }
  });

  unsigned nextTransfer = 0;
  for (Block &block : function.getBody()) {
    for (Operation &op : block) {
      Value transfer;
      unsigned channel = fixedResources.loadChannel;
      unsigned halves = 1;
      if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op))
        transfer = load.getTransfer();
      else if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
        transfer = load.getTransfer();
        halves = 2;
      } else if (auto store = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
        transfer = store.getTransfer();
        channel = fixedResources.storeChannel;
      } else if (auto store = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
        transfer = store.getTransfer();
        channel = fixedResources.storeChannel;
        halves = 2;
      }
      if (transfer)
        dmaTransfers[transfer] = {
            channel, halves, nextTransfer++, fixedResources.stagingWord,
            fixedResources.dmaBaseReg, fixedResources.dmaDramReg,
            fixedResources.dmaSizeReg};
    }
  }
  return success();
}

LogicalResult VirtualAllocationPlan::allocate(func::FuncOp function) {
  if (failed(placeResources(function)) ||
      failed(colorValues(RegisterKind::BF16)) ||
      failed(colorValues(RegisterKind::FP8)) ||
      failed(colorValues(RegisterKind::Scalar)))
    return failure();
  for (BlockArgument arg : function.getArguments())
    scalarArgumentRegs.push_back(scalar(arg));
  return success();
}

LogicalResult VirtualAllocationPlan::colorValues(RegisterKind kind) {
  using Set = llvm::DenseSet<Value>;
  auto selected = [&](Value value) {
    Type type = value.getType();
    if (kind == RegisterKind::BF16)
      return isa<VirtualBF16Type>(type);
    if (kind == RegisterKind::FP8)
      return isa<VirtualFP8Type>(type);
    return type.isInteger(1) || type.isInteger(32);
  };
  SmallVector<Value> values;
  llvm::DenseMap<Block *, Set> uses, defs, liveIn, liveOut;
  for (Block &block : function.getBody()) {
    Set &defined = defs[&block];
    Set &used = uses[&block];
    for (BlockArgument arg : block.getArguments())
      if (selected(arg)) {
        values.push_back(arg);
        defined.insert(arg);
      }
    for (Operation &op : block) {
      // A branch's tile/block-argument operands are edge uses, not uses on
      // both paths. They enter the fixed-point equation below per successor.
      if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
        Value condition = branch.getCondition();
        if (selected(condition) && !defined.contains(condition))
          used.insert(condition);
      } else if (!isa<cf::BranchOp>(op))
        for (Value operand : op.getOperands())
          if (selected(operand) && !defined.contains(operand))
            used.insert(operand);
      for (Value result : op.getResults())
        if (selected(result)) {
          values.push_back(result);
          defined.insert(result);
        }
    }
  }
  auto same = [](const Set &a, const Set &b) {
    if (a.size() != b.size())
      return false;
    for (Value value : a)
      if (!b.contains(value))
        return false;
    return true;
  };
  bool changed;
  do {
    changed = false;
    for (Block &block : llvm::reverse(function.getBody())) {
      Set out;
      auto edge = [&](Block *successor, ValueRange operands) {
        for (Value value : liveIn[successor])
          out.insert(value);
        for (Value operand : operands)
          if (selected(operand))
            out.insert(operand);
      };
      Operation *terminator = block.getTerminator();
      if (auto branch = dyn_cast<cf::BranchOp>(terminator))
        edge(branch.getDest(), branch.getDestOperands());
      if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
        edge(branch.getTrueDest(), branch.getTrueDestOperands());
        edge(branch.getFalseDest(), branch.getFalseDestOperands());
      }
      Set in = uses[&block];
      for (Value value : out)
        if (!defs[&block].contains(value))
          in.insert(value);
      if (!same(out, liveOut[&block]) || !same(in, liveIn[&block])) {
        liveOut[&block] = std::move(out);
        liveIn[&block] = std::move(in);
        changed = true;
      }
    }
  } while (changed);

  llvm::DenseMap<Value, Set> neighbors;
  for (Value value : values)
    neighbors[value];
  auto interfere = [&](Value first, Value second) {
    if (first == second)
      return;
    neighbors[first].insert(second);
    neighbors[second].insert(first);
  };
  auto clique = [&](const Set &set) {
    for (Value first : set)
      for (Value second : set)
        interfere(first, second);
  };
  // All runtime controls are loaded from the mailbox before the entry
  // block executes. Their physical registers therefore overlap at staging,
  // even if their SSA uses are disjoint later in the CFG.
  if (kind == RegisterKind::Scalar) {
    Set stagedArguments;
    for (BlockArgument arg : function.getArguments())
      stagedArguments.insert(arg);
    clique(stagedArguments);
  }
  for (Block &block : function.getBody()) {
    Set live = liveOut[&block];
    clique(live);
    for (Operation &op : llvm::reverse(block.getOperations())) {
      if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
        Value condition = branch.getCondition();
        if (selected(condition)) {
          live.insert(condition);
          clique(live);
        }
        continue;
      }
      if (isa<cf::BranchOp>(op))
        continue;
      SmallVector<Value> results, operands;
      for (Value result : op.getResults())
        if (selected(result))
          results.push_back(result);
      for (Value operand : op.getOperands())
        if (selected(operand))
          operands.push_back(operand);
      for (Value result : results) {
        for (Value other : live)
          interfere(result, other);
        // No in-place tensor or scalar update is assumed by this allocator.
        for (Value operand : operands)
          interfere(result, operand);
        live.erase(result);
      }
      for (Value operand : operands)
        live.insert(operand);
      clique(live);
    }
    SmallVector<Value> args;
    for (BlockArgument arg : block.getArguments())
      if (selected(arg))
        args.push_back(arg);
    for (Value arg : args)
      for (Value other : live)
        interfere(arg, other);
    for (Value first : args)
      for (Value second : args)
        interfere(first, second);
  }

  // Stable degree-first coloring reduces pressure without depending on
  // textual SSA names. Any coloring is checked against the graph below.
  std::stable_sort(values.begin(), values.end(), [&](Value a, Value b) {
    return neighbors[a].size() > neighbors[b].size();
  });
  llvm::DenseMap<Value, unsigned> colors;
  RegisterBudget budget = registerBudget(kind, mixedFp8, hasPack);
  for (Value value : values) {
    bool assigned = false;
    for (unsigned color = 0; color < budget.count; ++color) {
      bool conflict = llvm::any_of(neighbors[value], [&](Value other) {
        auto found = colors.find(other);
        return found != colors.end() && found->second == color;
      });
      if (!conflict) {
        colors[value] = color;
        assigned = true;
        break;
      }
    }
    if (!assigned)
      return function.emitOpError(
          kind == RegisterKind::BF16
              ? (mixedFp8 ? "mixed virtual BF16 interference exceeds 15 physical pairs"
                          : "virtual BF16 interference exceeds 31 physical pairs")
              : kind == RegisterKind::FP8
                    ? "virtual FP8 interference exceeds 32 physical registers"
                    : (hasPack
                           ? "virtual control interference exceeds 9 scalar registers with FP8 pack"
                           : "virtual control interference exceeds 17 scalar registers"));
  }
  for (Value value : values) {
    for (Value other : neighbors[value])
      if (colors[value] == colors[other])
        return function.emitOpError("internal register-coloring overlap");
    unsigned reg = budget.reg(colors[value]);
    if (kind == RegisterKind::BF16)
      tileRegs[value] = reg;
    else if (kind == RegisterKind::FP8)
      fp8Regs[value] = reg;
    else
      scalarRegs[value] = reg;
  }
  return success();
}

LogicalResult VirtualAllocationPlan::verify() const {
  SmallVector<VirtualRegisterAssignment> assignments;
  for (const auto &[value, reg] : tileRegs)
    assignments.push_back({value, reg});
  for (const auto &[value, reg] : fp8Regs)
    assignments.push_back({value, reg});
  for (const auto &[value, reg] : scalarRegs)
    assignments.push_back({value, reg});
  return verifyAtlasRegisterAllocation(function, assignments, fixedResources,
                                      scalarArgumentRegs);
}

unsigned VirtualAllocationPlan::tile(Value value) const {
  return tileRegs.find(value)->second;
}

unsigned VirtualAllocationPlan::fp8(Value value) const {
  return fp8Regs.find(value)->second;
}

unsigned VirtualAllocationPlan::scalar(Value value) const {
  return scalarRegs.find(value)->second;
}

MXUPlacement VirtualAllocationPlan::mxu(Value value) const {
  return mxuResources.find(value)->second;
}

const DMATransferPlacement &VirtualAllocationPlan::dma(Value value) const {
  return dmaTransfers.find(value)->second;
}

llvm::ArrayRef<int32_t> VirtualAllocationPlan::scalarArguments() const {
  return scalarArgumentRegs;
}

const FixedResourcePlacement &VirtualAllocationPlan::fixed() const {
  return fixedResources;
}
