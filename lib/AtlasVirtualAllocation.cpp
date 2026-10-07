#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr unsigned kTensorPairTemporary = 62;
constexpr unsigned kFirstScalarValue = 10;
constexpr unsigned kLastScalarValue = 26;
// The DMA engine's channels. Each transfer in flight holds one, and a staging
// window the size of a BF16 tile.
constexpr unsigned kDMAChannels = 8;
constexpr uint32_t kStagingWindowWords = 512;
static_assert(kMaxPendingVirtualDMA <= kDMAChannels,
              "each pending DMA needs its own channel");

// Whether `function` has FP8 values, which keep m0-m31 and move BF16 pairs
// above them, and a pack, which keeps x10-x17 for its relayout loop.
void findReservations(func::FuncOp function, bool &mixedFp8, bool &hasPack) {
  mixedFp8 = false;
  hasPack = false;
  function.walk([&](Operation *op) {
    hasPack |= isa<VirtualPackFP8Op>(op);
    for (Type type : op->getResultTypes())
      mixedFp8 |= isa<VirtualFP8Type>(type);
  });
}

unsigned capacity(RegisterKind kind, bool mixedFp8, bool hasPack) {
  unsigned firstScalar = hasPack ? 18 : kFirstScalarValue;
  return kind == RegisterKind::BF16
             ? (mixedFp8 ? 15 : kTensorPairTemporary / 2)
             : kind == RegisterKind::FP8 ? 32
                                         : kLastScalarValue - firstScalar + 1;
}

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

unsigned mlir::atlas::registerCapacity(func::FuncOp function,
                                       RegisterKind kind) {
  bool mixedFp8, hasPack;
  findReservations(function, mixedFp8, hasPack);
  return capacity(kind, mixedFp8, hasPack);
}

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
  usedDMAChannels.clear();
  findReservations(function, mixedFp8, hasPack);
  if (failed(colorValues(RegisterKind::BF16)) ||
      failed(colorValues(RegisterKind::FP8)) ||
      failed(colorValues(RegisterKind::Scalar)))
    return failure();
  for (BlockArgument arg : function.getArguments())
    scalarArgumentRegs.push_back(scalar(arg));

  // MXU and DMA state never crosses a block, so each block places its own.
  unsigned nextTransfer = 0;
  for (Block &block : function.getBody())
    if (failed(placeMXU(block)) || failed(placeDMA(block, nextTransfer)))
      return failure();
  return success();
}

// Each weight takes the lowest slot of its unit free of live weights and
// frees it after its last use; each accumulator chain takes the lowest free
// accumulator slot from its start to its readout. The verifier bounds both by
// the slot count and keeps both inside the block; the checks here state what
// the lowering relies on.
LogicalResult VirtualAllocationPlan::placeMXU(Block &block) {
  using Slots = std::array<Value, kVirtualMXUSlots>;
  std::array<Slots, 2> weightSlots{}, accSlots{};
  llvm::DenseMap<Value, unsigned> remainingUses;
  auto take = [&](Operation &op, Slots &slots, Value value,
                  unsigned unit) -> LogicalResult {
    for (unsigned slot = 0; slot < kVirtualMXUSlots; ++slot)
      if (!slots[slot]) {
        slots[slot] = value;
        mxuResources[value] = {unit, slot};
        return success();
      }
    return op.emitOpError("needs an MXU slot while every slot of its unit "
                          "is live");
  };
  auto release = [](Slots &slots, Value value) {
    for (Value &owner : slots)
      if (owner == value)
        owner = Value{};
  };
  // The live accumulator `acc` replaces or ends, and where it is.
  auto liveAccumulator = [&](Operation &op,
                             Value acc) -> FailureOr<MXUPlacement> {
    auto placed = mxuResources.find(acc);
    if (placed == mxuResources.end() ||
        accSlots[placed->second.unit][placed->second.slot] != acc)
      return op.emitOpError("uses an accumulator that is not live");
    return placed->second;
  };
  for (Operation &op : block) {
    for (Value operand : op.getOperands())
      if (auto weight = dyn_cast<VirtualMXUWeightType>(operand.getType()))
        if (--remainingUses[operand] == 0)
          release(weightSlots[weight.getUnit()], operand);
    if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
      Value weight = load.getWeight();
      unsigned unit = cast<VirtualMXUWeightType>(weight.getType()).getUnit();
      if (failed(take(op, weightSlots[unit], weight, unit)))
        return failure();
      remainingUses[weight] =
          std::distance(weight.use_begin(), weight.use_end());
      if (weight.use_empty())
        release(weightSlots[unit], weight);
    } else if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
      // The next version stays in its chain's slot.
      FailureOr<MXUPlacement> placement =
          liveAccumulator(op, accumulate.getAcc());
      if (failed(placement))
        return failure();
      mxuResources[accumulate.getNextAcc()] = *placement;
      accSlots[placement->unit][placement->slot] = accumulate.getNextAcc();
    } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
      FailureOr<MXUPlacement> placement = liveAccumulator(op, op.getOperand(1));
      if (failed(placement))
        return failure();
      accSlots[placement->unit][placement->slot] = Value{};
    } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op,
                   VirtualMXUResetOp>(op)) {
      Value acc = op.getResult(1);
      unsigned unit = cast<VirtualMXUAccType>(acc.getType()).getUnit();
      if (failed(take(op, accSlots[unit], acc, unit)))
        return failure();
    } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
      // The legacy form uses fixed slots of its unit, which must be free.
      unsigned unit = matmul.getUnit();
      if (weightSlots[unit][fixedResources.mxuWeightSlot] ||
          accSlots[unit][fixedResources.mxuAccSlot])
        return op.emitOpError("overwrites MXU slots that are live");
    }
  }
  for (const Slots &slots : accSlots)
    if (llvm::any_of(slots, [](Value acc) { return bool(acc); }))
      return block.getTerminator()->emitOpError(
          "leaves an MXU accumulator live at the block end");
  return success();
}

// Each transfer holds a channel and a staging window from its launch to its
// completion. The DMA latches its registers at launch, so every transfer
// uses the same three. A load takes the load channel when it is free and a
// store the store channel, as a lone transfer always has; otherwise the
// lowest free channel. A transfer takes the lowest free window.
LogicalResult VirtualAllocationPlan::placeDMA(Block &block,
                                              unsigned &nextTransfer) {
  std::array<Value, kDMAChannels> channels{}, windows{};
  for (Operation &op : block) {
    if (isa<VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op, VirtualDMAWaitOp>(
            op)) {
      Value transfer = op.getOperand(1);
      auto channel = llvm::find(channels, transfer);
      auto window = llvm::find(windows, transfer);
      if (channel == channels.end() || window == windows.end())
        return op.emitOpError("completes a DMA transfer that is not in flight");
      *channel = Value{};
      *window = Value{};
      continue;
    }
    bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
    if (!load && !isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op))
      continue;
    unsigned channel =
        load ? fixedResources.loadChannel : fixedResources.storeChannel;
    if (channels[channel])
      channel = llvm::find(channels, Value{}) - channels.begin();
    auto window = llvm::find(windows, Value{});
    if (channel == kDMAChannels || window == windows.end())
      return op.emitOpError("launches a DMA transfer while every channel is "
                            "in flight");
    Value transfer = op.getResult(1);
    channels[channel] = transfer;
    *window = transfer;
    if (!llvm::is_contained(usedDMAChannels, channel))
      usedDMAChannels.push_back(channel);
    unsigned halves =
        isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(op) ? 2 : 1;
    uint32_t stagingWord =
        fixedResources.stagingWord +
        static_cast<uint32_t>(window - windows.begin()) * kStagingWindowWords;
    dmaTransfers[transfer] = {channel,
                              halves,
                              nextTransfer++,
                              stagingWord,
                              fixedResources.dmaBaseReg,
                              fixedResources.dmaDramReg,
                              fixedResources.dmaSizeReg};
  }
  if (llvm::any_of(channels, [](Value transfer) { return bool(transfer); }))
    return block.getTerminator()->emitOpError(
        "leaves a DMA transfer in flight at the block end");
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
  unsigned firstScalar = hasPack ? 18 : kFirstScalarValue;
  unsigned count = capacity(kind, mixedFp8, hasPack);
  for (Value value : values) {
    bool assigned = false;
    for (unsigned color = 0; color < count; ++color) {
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
    if (kind == RegisterKind::BF16)
      tileRegs[value] = (mixedFp8 ? 32 : 0) + 2 * colors[value];
    else if (kind == RegisterKind::FP8)
      fp8Regs[value] = colors[value];
    else
      scalarRegs[value] = firstScalar + colors[value];
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
