#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr unsigned kTensorPairTemporary = 62;
constexpr unsigned kFirstScalarValue = 10;
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

VirtualAllocationPlan::VirtualAllocationPlan()
    : fixedResources(selectedResources()) {}

LogicalResult VirtualAllocationPlan::allocate(func::FuncOp function) {
  this->function = function;
  tileRegs.clear();
  fp8Regs.clear();
  scalarRegs.clear();
  mxuResources.clear();
  dmaTransfers.clear();
  scalarArgumentRegs.clear();
  sourceBlocks.clear();
  sourceOperations.clear();
  sourceIR.clear();
  hasPack = false;
  function.walk([&](Operation *op) {
    hasPack |= isa<VirtualPackFP8Op>(op);
    for (Value result : op->getResults()) {
      // Verified lifetimes permit slot 0 in each unit's weight/accumulator bank.
      if (auto weight = dyn_cast<VirtualMXUWeightType>(result.getType()))
        mxuResources[result] = {weight.getUnit(), fixedResources.mxuWeightSlot};
      if (auto acc = dyn_cast<VirtualMXUAccType>(result.getType()))
        mxuResources[result] = {acc.getUnit(), fixedResources.mxuAccSlot};
    }
  });
  if (failed(colorValues(RegisterKind::Tensor)) ||
      failed(colorValues(RegisterKind::Scalar)))
    return failure();
  for (BlockArgument arg : function.getArguments())
    scalarArgumentRegs.push_back(scalar(arg));

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
  captureSourceSnapshot();
  return success();
}

void VirtualAllocationPlan::captureSourceSnapshot() {
  for (Block &block : function.getBody()) {
    sourceBlocks.push_back(&block);
    for (Operation &op : block)
      sourceOperations.push_back(&op);
  }
  llvm::raw_string_ostream stream(sourceIR);
  function.print(stream);
}

LogicalResult VirtualAllocationPlan::verifySourceSnapshot() const {
  func::FuncOp current = function;
  unsigned blockIndex = 0, opIndex = 0;
  for (Block &block : current.getBody()) {
    if (blockIndex >= sourceBlocks.size() || sourceBlocks[blockIndex++] != &block)
      return current.emitOpError(
          "virtual allocation plan is stale after source change; rerun allocation");
    for (Operation &op : block) {
      if (opIndex >= sourceOperations.size() || sourceOperations[opIndex] != &op)
        return current.emitOpError(
            "virtual allocation plan is stale after source change; rerun allocation");
      ++opIndex;
    }
  }
  if (blockIndex != sourceBlocks.size() || opIndex != sourceOperations.size())
    return current.emitOpError(
        "virtual allocation plan is stale after source change; rerun allocation");
  std::string currentIR;
  llvm::raw_string_ostream stream(currentIR);
  current.print(stream);
  stream.flush();
  if (currentIR != sourceIR)
    return current.emitOpError(
        "virtual allocation plan is stale after source change; rerun allocation");
  return success();
}

LogicalResult VirtualAllocationPlan::colorValues(RegisterKind kind) {
  using Set = llvm::DenseSet<Value>;
  auto selected = [&](Value value) {
    Type type = value.getType();
    if (kind == RegisterKind::Tensor)
      return isa<VirtualBF16Type, VirtualFP8Type>(type);
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
    if (neighbors[a].size() != neighbors[b].size())
      return neighbors[a].size() > neighbors[b].size();
    // Wider values have fewer legal placements. Preserve source order within
    // a width so unrelated SSA names do not affect placement.
    return isa<VirtualBF16Type>(a.getType()) &&
           !isa<VirtualBF16Type>(b.getType());
  });
  llvm::DenseMap<Value, unsigned> colors;
  unsigned firstScalar = hasPack ? 18 : kFirstScalarValue;
  unsigned count = kLastScalarValue - firstScalar + 1;
  auto lastReg = [](Value value, unsigned base) {
    return base + unsigned(isa<VirtualBF16Type>(value.getType()));
  };
  for (Value value : values) {
    bool assigned = false;
    unsigned limit = kind == RegisterKind::Tensor ? kTensorPairTemporary : count;
    for (unsigned color = 0; color < limit; ++color) {
      if (kind == RegisterKind::Tensor &&
          isa<VirtualBF16Type>(value.getType()) && (color & 1))
        continue;
      if (kind == RegisterKind::Tensor && lastReg(value, color) >= limit)
        continue;
      bool conflict = llvm::any_of(neighbors[value], [&](Value other) {
        auto found = colors.find(other);
        if (found == colors.end())
          return false;
        if (kind == RegisterKind::Scalar)
          return found->second == color;
        return color <= lastReg(other, found->second) &&
               found->second <= lastReg(value, color);
      });
      if (!conflict) {
        colors[value] = color;
        assigned = true;
        break;
      }
    }
    if (!assigned)
      return function.emitOpError(
          kind == RegisterKind::Tensor
              ? "virtual tensor greedy placement failed in 62-register file (31 BF16 pairs)"
              : (hasPack
                           ? "virtual control interference exceeds 9 scalar registers with FP8 pack"
                           : "virtual control interference exceeds 17 scalar registers"));
  }
  for (Value value : values) {
    for (Value other : neighbors[value])
      if (kind == RegisterKind::Scalar
              ? colors[value] == colors[other]
              : colors[value] <= lastReg(other, colors[other]) &&
                    colors[other] <= lastReg(value, colors[value]))
        return function.emitOpError("internal register-coloring overlap");
    if (kind == RegisterKind::Tensor && isa<VirtualBF16Type>(value.getType()))
      tileRegs[value] = colors[value];
    else if (kind == RegisterKind::Tensor)
      fp8Regs[value] = colors[value];
    else
      scalarRegs[value] = firstScalar + colors[value];
  }
  return success();
}

LogicalResult VirtualAllocationPlan::verify() const {
  if (!function || failed(verifySourceSnapshot()))
    return failure();
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
