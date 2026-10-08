#include "Atlas/AtlasMXUAllocationVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// WeightBuffers.scala and AccumulationBuffers.scala define two separate slots
// in each bank of each MXU. This is geometry, not a physical release rule.
constexpr unsigned kUnits = 2;
constexpr unsigned kSlots = 2;
using Owners = std::array<std::array<Value, kSlots>, kUnits>;

bool isWeight(Value value) {
  return isa<VirtualMXUWeightType>(value.getType());
}

bool isAccumulator(Value value) {
  return isa<VirtualMXUAccType>(value.getType());
}

unsigned handleUnit(Value value) {
  if (auto weight = dyn_cast<VirtualMXUWeightType>(value.getType()))
    return weight.getUnit();
  return cast<VirtualMXUAccType>(value.getType()).getUnit();
}

class MXUAllocationVerifier {
public:
  MXUAllocationVerifier(func::FuncOp function,
                        ArrayRef<VirtualMXUAssignment> assignments,
                        const FixedResourcePlacement &fixed)
      : function(function), assignments(assignments), fixed(fixed),
        assemblyState(function) {}

  LogicalResult verify() {
    for (Block &block : function.getBody()) {
      for (BlockArgument argument : block.getArguments())
        known.insert(argument);
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          known.insert(result);
          if (isWeight(result) || isAccumulator(result))
            handles.push_back(result);
        }
        // Recompute uses from source operations, never allocator summaries.
        for (Value operand : op.getOperands())
          if (isWeight(operand))
            ++remainingUses[operand];
      }
    }
    for (const VirtualMXUAssignment &assignment : assignments) {
      Value handle = assignment.handle;
      if (!handle || !known.contains(handle))
        return function.emitOpError("MXU assignment refers to a foreign, stale, or untracked value");
      if ((!isWeight(handle) && !isAccumulator(handle)) || !handle.getDefiningOp())
        return assignmentError(handle, "MXU assignment does not refer to a source handle result");
      if (placements.contains(handle))
        return assignmentError(handle, "duplicate MXU assignment");
      const MXUPlacement &placement = assignment.placement;
      if (placement.unit >= kUnits || placement.unit != handleUnit(handle))
        return assignmentError(handle, "MXU placement unit must match its handle and be in [0, 1]");
      if (placement.slot >= kSlots)
        return assignmentError(handle, "MXU slot is outside [0, 1]");
      placements[handle] = placement;
    }
    for (Value handle : handles)
      if (!placements.contains(handle))
        return assignmentError(handle, "missing MXU assignment");
    for (Block &block : function.getBody())
      if (failed(verifyBlock(block)))
        return failure();
    return success();
  }

private:
  void noteHandle(InFlightDiagnostic &diagnostic, Value handle, StringRef label) {
    std::string operand;
    llvm::raw_string_ostream stream(operand);
    handle.printAsOperand(stream, assemblyState);
    auto &note = diagnostic.attachNote(handle.getLoc());
    note << label << " " << stream.str();
    auto found = placements.find(handle);
    if (found != placements.end())
      note << "; unit " << found->second.unit << ", "
           << (isWeight(handle) ? "weight" : "accumulator") << " slot "
           << found->second.slot;
  }

  LogicalResult assignmentError(Value handle, StringRef message) {
    auto diagnostic = function.emitOpError(message);
    noteHandle(diagnostic, handle, "assigned handle");
    return failure();
  }

  LogicalResult ownershipError(Operation &op, Value handle, Value owner,
                               StringRef message) {
    auto diagnostic = op.emitOpError(message);
    noteHandle(diagnostic, handle, "used or produced handle");
    if (owner)
      noteHandle(diagnostic, owner, "current owner");
    return failure();
  }

  LogicalResult claim(Operation &op, Value handle, Owners &owners) {
    MXUPlacement placement = placements.lookup(handle);
    Value &owner = owners[placement.unit][placement.slot];
    if (owner)
      return ownershipError(op, handle, owner, "MXU placement overwrites a logically live slot");
    owner = handle;
    return success();
  }

  LogicalResult useWeight(Operation &op, Value handle, Owners &weights) {
    MXUPlacement placement = placements.lookup(handle);
    Value &owner = weights[placement.unit][placement.slot];
    if (owner != handle || remainingUses.lookup(handle) == 0)
      return ownershipError(op, handle, owner, "MXU weight use requires its current block-local slot owner");
    if (--remainingUses[handle] == 0)
      owner = Value{};
    return success();
  }

  LogicalResult verifyBlock(Block &block) {
    Owners weights{}, accumulators{};
    for (Operation &op : block) {
      for (Value operand : op.getOperands())
        if ((isWeight(operand) || isAccumulator(operand)) && operand.getParentBlock() != &block)
          return op.emitOpError("MXU handles cannot cross CFG blocks");
      if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
        Value weight = load.getWeight();
        if (failed(claim(op, weight, weights)))
          return failure();
        // A dead load still writes its claimed slot before becoming dead.
        if (remainingUses.lookup(weight) == 0) {
          MXUPlacement placement = placements.lookup(weight);
          weights[placement.unit][placement.slot] = Value{};
        }
      } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
        if (failed(claim(op, op.getResult(1), accumulators)))
          return failure();
      } else if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
        if (failed(useWeight(op, reset.getWeight(), weights)) ||
            failed(claim(op, reset.getAcc(), accumulators)))
          return failure();
      } else if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
        Value previous = accumulate.getAcc(), next = accumulate.getNextAcc();
        MXUPlacement oldPlacement = placements.lookup(previous);
        MXUPlacement newPlacement = placements.lookup(next);
        Value &owner = accumulators[oldPlacement.unit][oldPlacement.slot];
        if (owner != previous)
          return ownershipError(op, previous, owner, "MXU accumulation requires the current accumulator version");
        if (oldPlacement.unit != newPlacement.unit || oldPlacement.slot != newPlacement.slot)
          return ownershipError(op, next, previous, "MXU accumulator continuation must retain its unit and slot");
        if (failed(useWeight(op, accumulate.getWeight(), weights)))
          return failure();
        owner = next;
      } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
        Value acc = op.getOperand(1);
        MXUPlacement placement = placements.lookup(acc);
        Value &owner = accumulators[placement.unit][placement.slot];
        if (owner != acc)
          return ownershipError(op, acc, owner, "MXU readout requires the current accumulator version");
        owner = Value{};
      } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
        unsigned unit = matmul.getUnit();
        if (unit >= kUnits || fixed.mxuWeightSlot >= kSlots || fixed.mxuAccSlot >= kSlots)
          return op.emitOpError("legacy MXU fixed unit/slots are outside [0, 1]");
        if (Value owner = weights[unit][fixed.mxuWeightSlot]) {
          auto diagnostic = op.emitOpError("legacy MXU matmul overwrites a logically live weight slot");
          noteHandle(diagnostic, owner, "current owner");
          return failure();
        }
        if (Value owner = accumulators[unit][fixed.mxuAccSlot]) {
          auto diagnostic = op.emitOpError("legacy MXU matmul overwrites a logically live accumulator slot");
          noteHandle(diagnostic, owner, "current owner");
          return failure();
        }
      }
    }
    for (const auto &unit : accumulators)
      for (Value owner : unit)
        if (owner) {
          auto diagnostic = block.getTerminator()->emitOpError("MXU accumulator ownership remains live at block exit");
          noteHandle(diagnostic, owner, "current owner");
          return failure();
        }
    return success();
  }

  func::FuncOp function;
  ArrayRef<VirtualMXUAssignment> assignments;
  const FixedResourcePlacement &fixed;
  llvm::DenseSet<Value> known;
  SmallVector<Value> handles;
  llvm::DenseMap<Value, MXUPlacement> placements;
  llvm::DenseMap<Value, unsigned> remainingUses;
  AsmState assemblyState;
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasMXUAllocation(
    func::FuncOp function, ArrayRef<VirtualMXUAssignment> assignments,
    const FixedResourcePlacement &fixed) {
  return MXUAllocationVerifier(function, assignments, fixed).verify();
}
