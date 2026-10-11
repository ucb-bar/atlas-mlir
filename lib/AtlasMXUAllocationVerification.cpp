#include "Atlas/AtlasMXUAllocationVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasMXUOwnership.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr unsigned kUnits = MXUOwnership::kUnits, kSlots = MXUOwnership::kSlots;
constexpr int32_t kFree = MXUOwnership::kFree;

bool isWeight(Value value) { return isa<VirtualMXUWeightType>(value.getType()); }
bool isAccumulator(Value value) { return isa<VirtualMXUAccType>(value.getType()); }

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
    known = sourceValues(function);
    for (Block &block : function.getBody()) {
      for (Operation &op : block) {
        for (Value result : op.getResults())
          if (isWeight(result) || isAccumulator(result))
            handles.push_back(result);
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

  LogicalResult ownershipError(Operation &op, Value handle, int32_t owner,
                               StringRef message) {
    auto diagnostic = op.emitOpError(message);
    noteHandle(diagnostic, handle, "used or produced handle");
    if (owner != kFree)
      noteHandle(diagnostic, values[owner], "current owner");
    return failure();
  }

  LogicalResult ownerError(Operation &op, int32_t owner, StringRef message) {
    auto diagnostic = op.emitOpError(message);
    noteHandle(diagnostic, values[owner], "current owner");
    return failure();
  }

  // Owner ids are assigned on first use so no two values ever share one.
  int32_t id(Value value) {
    auto [found, inserted] = ids.try_emplace(value, values.size());
    if (inserted)
      values.push_back(value);
    return found->second;
  }

  LogicalResult useWeight(Operation &op, Value weight) {
    MXUPlacement at = placements.lookup(weight);
    unsigned &remaining = remainingUses[weight];
    if (auto violation = owners.useWeight(at.unit, at.slot, id(weight), remaining == 1))
      return ownershipError(op, weight, *violation, "MXU weight use requires its current block-local slot owner");
    --remaining;
    return success();
  }

  LogicalResult startAccumulator(Operation &op, Value acc) {
    MXUPlacement at = placements.lookup(acc);
    if (auto violation = owners.startAccumulator(at.unit, at.slot, id(acc)))
      return ownershipError(op, acc, *violation, "MXU placement overwrites a logically live slot");
    return success();
  }

  LogicalResult verifyBlock(Block &block) {
    owners = MXUOwnership{};
    for (Operation &op : block) {
      for (Value operand : op.getOperands())
        if ((isWeight(operand) || isAccumulator(operand)) && operand.getParentBlock() != &block)
          return op.emitOpError("MXU handles cannot cross CFG blocks");
      if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
        Value weight = load.getWeight();
        MXUPlacement at = placements.lookup(weight);
        if (auto violation = owners.pushWeight(at.unit, at.slot, id(weight), remainingUses.lookup(weight) != 0))
          return ownershipError(op, weight, *violation, "MXU placement overwrites a logically live slot");
      } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
        if (failed(startAccumulator(op, op.getResult(1))))
          return failure();
      } else if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
        if (failed(useWeight(op, reset.getWeight())) || failed(startAccumulator(op, reset.getAcc())))
          return failure();
      } else if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
        Value previous = accumulate.getAcc(), next = accumulate.getNextAcc();
        MXUPlacement at = placements.lookup(previous), to = placements.lookup(next);
        if (auto violation = owners.continueAccumulator(at.unit, at.slot, id(previous), id(next)))
          return ownershipError(op, previous, *violation, "MXU accumulation requires the current accumulator version");
        if (at.unit != to.unit || at.slot != to.slot)
          return ownershipError(op, next, id(previous), "MXU accumulator continuation must retain its unit and slot");
        if (failed(useWeight(op, accumulate.getWeight())))
          return failure();
      } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
        Value acc = op.getOperand(1);
        MXUPlacement at = placements.lookup(acc);
        if (auto violation = owners.readoutAccumulator(at.unit, at.slot, id(acc)))
          return ownershipError(op, acc, *violation, "MXU readout requires the current accumulator version");
      } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
        unsigned unit = matmul.getUnit();
        if (unit >= kUnits || fixed.mxuWeightSlot >= kSlots || fixed.mxuAccSlot >= kSlots)
          return op.emitOpError("legacy MXU fixed unit/slots are outside [0, 1]");
        if (auto violation = owners.legacyMatmul(unit, fixed.mxuWeightSlot, fixed.mxuAccSlot))
          return ownerError(op, *violation, isWeight(values[*violation])
                                                ? "legacy MXU matmul overwrites a logically live weight slot"
                                                : "legacy MXU matmul overwrites a logically live accumulator slot");
      }
    }
    if (auto violation = owners.blockExit())
      return ownerError(*block.getTerminator(), *violation, "MXU accumulator ownership remains live at block exit");
    return success();
  }

  func::FuncOp function;
  ArrayRef<VirtualMXUAssignment> assignments;
  const FixedResourcePlacement &fixed;
  llvm::DenseSet<Value> known;
  SmallVector<Value> handles;
  llvm::DenseMap<Value, MXUPlacement> placements;
  llvm::DenseMap<Value, unsigned> remainingUses;
  llvm::DenseMap<Value, int32_t> ids;
  SmallVector<Value> values;
  MXUOwnership owners;
  AsmState assemblyState;
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasMXUAllocation(
    func::FuncOp function, ArrayRef<VirtualMXUAssignment> assignments,
    const FixedResourcePlacement &fixed) {
  return MXUAllocationVerifier(function, assignments, fixed).verify();
}
