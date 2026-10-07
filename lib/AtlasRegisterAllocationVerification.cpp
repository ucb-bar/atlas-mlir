#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
using ValueSet = llvm::DenseSet<Value>;

bool isScalar(Value value) {
  return value.getType().isInteger(1) || value.getType().isInteger(32);
}

bool isTracked(Value value) {
  return isScalar(value) ||
         isa<VirtualBF16Type, VirtualFP8Type>(value.getType());
}

class AllocationVerifier {
public:
  AllocationVerifier(func::FuncOp function,
                     llvm::ArrayRef<VirtualRegisterAssignment> assignments,
                     const FixedResourcePlacement &fixed,
                     llvm::ArrayRef<int32_t> scalarArgumentRegs)
      : function(function), assignments(assignments), fixed(fixed),
        scalarArgumentRegs(scalarArgumentRegs), assemblyState(function) {}

  LogicalResult verify() {
    for (Block &block : function.getBody()) {
      for (BlockArgument arg : block.getArguments())
        collect(arg);
      for (Operation &op : block) {
        hasPack |= isa<VirtualPackFP8Op>(op);
        for (Value result : op.getResults())
          collect(result);
      }
    }
    if (failed(verifyAssignments()) || failed(verifyArgumentABI()))
      return failure();
    computeLiveness();

    SmallVector<Value> entryArgs;
    for (BlockArgument arg : function.getArguments())
      entryArgs.push_back(arg);
    // The mailbox stages every entry argument before executing the body.
    if (failed(checkClique(entryArgs, function.getLoc(),
                           "entry argument staging")))
      return failure();

    for (Block &block : function.getBody()) {
      if (failed(verifyBlock(block)))
        return failure();
    }
    return success();
  }

private:
  void collect(Value value) {
    known.insert(value);
    if (isTracked(value)) {
      order[value] = values.size();
      values.push_back(value);
    }
  }

  void noteValue(InFlightDiagnostic &diagnostic, Value value,
                 llvm::StringRef label) {
    std::string operand;
    llvm::raw_string_ostream stream(operand);
    value.printAsOperand(stream, assemblyState);
    auto &note = diagnostic.attachNote(value.getLoc());
    note << label << " " << stream.str() << " : " << value.getType();
    if (auto arg = dyn_cast<BlockArgument>(value))
      note << " (block argument " << arg.getArgNumber() << ")";
    else
      note << " (defined by " << value.getDefiningOp()->getName() << ")";
  }

  LogicalResult assignmentError(Value value, const llvm::Twine &message) {
    auto diagnostic = function.emitOpError(message);
    noteValue(diagnostic, value, "assigned value");
    return failure();
  }

  bool reservedScalar(unsigned reg) const {
    for (unsigned reserved :
         {fixed.scalarTemporary, fixed.oneReg, fixed.zeroReg, fixed.halfSizeReg,
          fixed.haltReg, fixed.inputBaseReg, fixed.inputDramReg,
          fixed.outputBaseReg, fixed.outputDramReg})
      if (reg == reserved)
        return true;
    for (const DMASlotPlacement &slot : fixed.dmaSlots)
      if (reg == slot.baseReg || reg == slot.dramReg || reg == slot.sizeReg)
        return true;
    if (!hasPack)
      return false;
    // x15 is intentionally reserved along with the pack helper's x10..x17.
    if (reg >= 10 && reg <= 17)
      return true;
    return reg == fixed.packSourceRegs[0] || reg == fixed.packSourceRegs[1] ||
           reg == fixed.packDestinationReg || reg == fixed.packRowReg ||
           reg == fixed.packRowsReg || reg == fixed.packTemporaryRegs[0] ||
           reg == fixed.packTemporaryRegs[1];
  }

  LogicalResult verifyAssignments() {
    for (const VirtualRegisterAssignment &assignment : assignments) {
      Value value = assignment.value;
      // Test pointer membership before inspecting an externally supplied Value;
      // a stale Value may no longer have valid type or location storage.
      if (!value || !known.contains(value))
        return function.emitOpError(
            "register assignment refers to a foreign, stale, or untracked value");
      if (!isTracked(value))
        return assignmentError(value,
                               "register assignment has unsupported value type");
      if (registers.contains(value))
        return assignmentError(value, "duplicate register assignment");
      unsigned reg = assignment.reg;
      if (isScalar(value)) {
        if (reg < 10 || reg > 26)
          return assignmentError(
              value, llvm::Twine("x") + llvm::Twine(reg) +
                         " is outside scalar allocation pool x10..x26");
        if (reservedScalar(reg))
          return assignmentError(
              value, llvm::Twine("reserved scalar register x") + llvm::Twine(reg));
      } else {
        bool pair = isa<VirtualBF16Type>(value.getType());
        if (reg >= 64 || (pair && reg >= 63))
          return assignmentError(
              value, llvm::Twine("tensor register ") + llvm::Twine(reg) +
                         " is outside tensor register bank [0, 63]");
        if (pair && reg % 2)
          return assignmentError(
              value, llvm::Twine("BF16 requires an even tensor register; got ") +
                         llvm::Twine(reg));
        unsigned last = reg + (pair ? 1 : 0);
        if ((reg <= fixed.tensorTemporary && last >= fixed.tensorTemporary) ||
            (reg <= uint64_t(fixed.tensorTemporary) + 1 &&
             last >= uint64_t(fixed.tensorTemporary) + 1))
          return assignmentError(
              value, llvm::Twine("reserved tensor temporary pair ") +
                         llvm::Twine(fixed.tensorTemporary) + ":" +
                         llvm::Twine(uint64_t(fixed.tensorTemporary) + 1));
      }
      registers[value] = reg;
    }
    for (Value value : values)
      if (!registers.contains(value))
        return assignmentError(value, "missing register assignment");
    return success();
  }

  LogicalResult verifyArgumentABI() {
    if (scalarArgumentRegs.size() != function.getNumArguments())
      return function.emitOpError(
          "scalar argument ABI register count does not match entry arguments");
    for (auto indexed : llvm::enumerate(function.getArguments())) {
      Value arg = indexed.value();
      if (scalarArgumentRegs[indexed.index()] < 0 ||
          unsigned(scalarArgumentRegs[indexed.index()]) != registers.lookup(arg))
        return assignmentError(
            arg,
            llvm::Twine("scalar argument ABI register does not match assignment x") +
                llvm::Twine(registers.lookup(arg)));
    }
    return success();
  }

  void addOperands(ValueSet &live, ValueRange operands) {
    for (Value operand : operands)
      if (order.contains(operand))
        live.insert(operand);
  }

  template <typename Callback>
  void forEachEdge(Operation *terminator, Callback callback) {
    if (auto branch = dyn_cast<cf::BranchOp>(terminator))
      callback(branch.getDest(), branch.getDestOperands());
    else if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
      callback(branch.getTrueDest(), branch.getTrueDestOperands());
      callback(branch.getFalseDest(), branch.getFalseDestOperands());
    }
  }

  void transfer(Operation &op, ValueSet &live) {
    if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
      live.insert(branch.getCondition());
      return;
    }
    if (isa<cf::BranchOp>(op))
      return;
    for (Value result : op.getResults())
      live.erase(result);
    addOperands(live, op.getOperands());
  }

  void computeLiveness() {
    bool changed;
    do {
      changed = false;
      for (Block &block : llvm::reverse(function.getBody())) {
        ValueSet out;
        forEachEdge(block.getTerminator(), [&](Block *dest, ValueRange operands) {
          out.insert(liveIn[dest].begin(), liveIn[dest].end());
          // Every incoming value participates in emitted parallel copies,
          // including values passed to otherwise unused destination arguments.
          addOperands(out, operands);
        });
        ValueSet in = out;
        for (Operation &op : llvm::reverse(block.getOperations()))
          transfer(op, in);
        for (BlockArgument arg : block.getArguments())
          in.erase(arg);
        if (in != liveIn[&block] || out != liveOut[&block]) {
          liveIn[&block] = std::move(in);
          liveOut[&block] = std::move(out);
          changed = true;
        }
      }
    } while (changed);
  }

  SmallVector<Value> ordered(const ValueSet &set) {
    SmallVector<Value> result(set.begin(), set.end());
    llvm::sort(result,
               [&](Value a, Value b) { return order.lookup(a) < order.lookup(b); });
    return result;
  }

  std::string physical(Value value) {
    unsigned reg = registers.lookup(value);
    if (isScalar(value))
      return (llvm::Twine("x") + llvm::Twine(reg)).str();
    if (isa<VirtualBF16Type>(value.getType()))
      return (llvm::Twine("tensor pair ") + llvm::Twine(reg) + ":" +
              llvm::Twine(reg + 1))
          .str();
    return (llvm::Twine("tensor register ") + llvm::Twine(reg)).str();
  }

  LogicalResult checkPair(Value first, Value second, Location location,
                          llvm::StringRef context) {
    if (first == second || isScalar(first) != isScalar(second))
      return success();
    unsigned a = registers.lookup(first), b = registers.lookup(second);
    unsigned aLast = a + isa<VirtualBF16Type>(first.getType());
    unsigned bLast = b + isa<VirtualBF16Type>(second.getType());
    if (a > bLast || b > aLast)
      return success();
    auto diagnostic = emitError(location);
    diagnostic << "register allocation overlap in " << context << ": "
               << physical(first) << " conflicts with " << physical(second);
    noteValue(diagnostic, first, "first value");
    noteValue(diagnostic, second, "conflicting value");
    return failure();
  }

  LogicalResult checkClique(llvm::ArrayRef<Value> live, Location location,
                            llvm::StringRef context) {
    for (unsigned i = 0; i < live.size(); ++i)
      for (unsigned j = i + 1; j < live.size(); ++j)
        if (failed(checkPair(live[i], live[j], location, context)))
          return failure();
    return success();
  }

  LogicalResult verifyBlock(Block &block) {
    ValueSet live = liveOut[&block];
    if (failed(checkClique(ordered(live), block.getTerminator()->getLoc(),
                           "live values at block exit")))
      return failure();
    LogicalResult edges = success();
    forEachEdge(block.getTerminator(), [&](Block *dest, ValueRange) {
      if (failed(edges))
        return;
      auto through = ordered(liveIn[dest]);
      for (BlockArgument arg : dest->getArguments()) {
        if (!order.contains(arg))
          continue;
        // Copy sources may alias destinations and form cycles; only values
        // needed after the copies must survive writes to all destinations.
        for (Value value : through)
          if (failed(checkPair(arg, value, block.getTerminator()->getLoc(),
                               "edge destinations and live-through values"))) {
            edges = failure();
            return;
          }
      }
    });
    if (failed(edges))
      return failure();

    for (Operation &op : llvm::reverse(block.getOperations())) {
      if (!isa<cf::BranchOp, cf::CondBranchOp>(op)) {
        SmallVector<Value> results, operands;
        for (Value result : op.getResults())
          if (order.contains(result))
            results.push_back(result);
        for (Value operand : op.getOperands())
          if (order.contains(operand))
            operands.push_back(operand);
        if (failed(checkClique(results, op.getLoc(), "operation results")))
          return failure();
        auto after = ordered(live);
        for (Value result : results) {
          for (Value value : after)
            if (failed(checkPair(result, value, op.getLoc(),
                                 "operation results and live-after values")))
              return failure();
          for (Value operand : operands)
            if (failed(checkPair(result, operand, op.getLoc(),
                                 "operation results and operands")))
              return failure();
        }
      }
      transfer(op, live);
      if (failed(checkClique(ordered(live), op.getLoc(), "live values")))
        return failure();
    }

    SmallVector<Value> args;
    for (BlockArgument arg : block.getArguments())
      if (order.contains(arg))
        args.push_back(arg);
    if (failed(checkClique(args, block.getParentOp()->getLoc(),
                           "block arguments")))
      return failure();
    auto through = ordered(liveIn[&block]);
    for (Value arg : args)
      for (Value value : through)
        if (failed(checkPair(arg, value, block.getParentOp()->getLoc(),
                             "block arguments and live-through values")))
          return failure();
    return success();
  }

  func::FuncOp function;
  llvm::ArrayRef<VirtualRegisterAssignment> assignments;
  const FixedResourcePlacement &fixed;
  llvm::ArrayRef<int32_t> scalarArgumentRegs;
  AsmState assemblyState;
  bool hasPack = false;
  ValueSet known;
  SmallVector<Value> values;
  llvm::DenseMap<Value, unsigned> order, registers;
  llvm::DenseMap<Block *, ValueSet> liveIn, liveOut;
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasRegisterAllocation(
    func::FuncOp function,
    llvm::ArrayRef<VirtualRegisterAssignment> assignments,
    const FixedResourcePlacement &fixed,
    llvm::ArrayRef<int32_t> scalarArgumentRegs) {
  return AllocationVerifier(function, assignments, fixed, scalarArgumentRegs)
      .verify();
}
