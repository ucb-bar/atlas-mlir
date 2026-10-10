#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <deque>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
using ValueSet = llvm::DenseSet<Value>;
using HelperRegisters = std::array<std::optional<uint32_t>, 32>;
using HelperWriters = std::array<Operation *, 32>;

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
                     llvm::ArrayRef<int32_t> scalarArgumentRegs,
                     llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments,
                     DMAAwaitBasePolicy awaitBasePolicy)
      : function(function), assignments(assignments), fixed(fixed),
        scalarArgumentRegs(scalarArgumentRegs), dmaAssignments(dmaAssignments),
        awaitBasePolicy(awaitBasePolicy), assemblyState(function) {}

  LogicalResult verify() {
    if (failed(verifyAtlasDMAAllocation(function, dmaAssignments)))
      return failure();
    for (const VirtualDMAAssignment &assignment : dmaAssignments)
      dmaPlacements[assignment.transfer] = &assignment.placement;
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
    return verifyHelperContents();
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
         {fixed.scalarTemporary, fixed.halfSizeReg, fixed.haltReg,
          fixed.inputBaseReg, fixed.inputDramReg, fixed.outputBaseReg,
          fixed.outputDramReg, fixed.dmaBaseReg, fixed.dmaDramReg,
          fixed.dmaSizeReg})
      if (reg == reserved)
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

  LogicalResult checkHelperWrite(Operation &op, Value transfer, unsigned reg, const ValueSet &live, StringRef phase) {
    for (Value value : ordered(live)) {
      if (!isScalar(value) || registers.lookup(value) != reg)
        continue;
      auto diagnostic = op.emitOpError("DMA helper write clobbers live scalar value");
      diagnostic << ": " << phase << " writes x" << reg;
      noteValue(diagnostic, transfer, "assigned transfer");
      noteValue(diagnostic, value, "live-after scalar value");
      return failure();
    }
    return success();
  }

  LogicalResult verifyDMAWrites(Operation &op, const ValueSet &live) {
    Value transfer, dram, size;
    if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      transfer = load.getTransfer();
      dram = load.getDramByte();
      size = load.getSizeBytes();
    } else if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
      transfer = load.getTransfer();
      dram = load.getDramByte();
      size = load.getSizeBytes();
    } else if (auto store = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
      transfer = store.getTransfer();
      dram = store.getDramByte();
      size = store.getSizeBytes();
    } else if (auto store = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
      transfer = store.getTransfer();
      dram = store.getDramByte();
      size = store.getSizeBytes();
    }
    if (transfer) {
      const DMATransferPlacement &placement = *dmaPlacements.lookup(transfer);
      // Lowering captures DRAM before size, then materializes staging. A
      // self-copy emits no write.
      if (placement.dramReg != registers.lookup(dram)) {
        if (placement.dramReg == registers.lookup(size) && dram != size) {
          auto diagnostic = op.emitOpError("DMA DRAM helper write clobbers size operand before capture");
          diagnostic << ": writes x" << placement.dramReg;
          noteValue(diagnostic, transfer, "assigned transfer");
          noteValue(diagnostic, size, "uncaptured size operand");
          return failure();
        }
        if (failed(checkHelperWrite(op, transfer, placement.dramReg, live, "DRAM capture")))
          return failure();
      }
      if (placement.sizeReg != registers.lookup(size) &&
          failed(checkHelperWrite(op, transfer, placement.sizeReg, live, "size capture")))
        return failure();
      return checkHelperWrite(op, transfer, placement.stagingReg, live, "launch staging materialization");
    }

    bool writesBase = false;
    if (auto await = dyn_cast<VirtualDMAAwaitBF16Op>(op)) {
      transfer = await.getTransfer();
      writesBase = true; // Both policies materialize the second half's base.
    } else if (auto await = dyn_cast<VirtualDMAAwaitFP8Op>(op)) {
      transfer = await.getTransfer();
      writesBase = awaitBasePolicy == DMAAwaitBasePolicy::Rematerialized;
    }
    if (writesBase)
      return checkHelperWrite(op, transfer, dmaPlacements.lookup(transfer)->stagingReg, live,
                              "await staging materialization");
    return success();
  }

  LogicalResult helperRead(Operation &op, const HelperRegisters &contents,
                           unsigned reg, uint32_t expected, StringRef message,
                           const HelperWriters *writers, Value handle = {}) {
    if (reg == 0 || reg >= contents.size())
      return op.emitOpError(message) << ": helper must be in x1..x31";
    if (contents[reg] && *contents[reg] == expected)
      return success();
    auto diagnostic = op.emitOpError(message);
    diagnostic << ": x" << reg << " requires " << expected << ", got ";
    if (contents[reg])
      diagnostic << *contents[reg];
    else
      diagnostic << "unknown";
    if (handle)
      noteValue(diagnostic, handle, "pending load transfer");
    if (writers && (*writers)[reg])
      diagnostic.attachNote((*writers)[reg]->getLoc())
          << "last source operation writing x" << reg;
    return failure();
  }

  std::optional<uint32_t> boundaryAddress(StringRef attribute, uint64_t offset) {
    auto base = function->getAttrOfType<IntegerAttr>(attribute);
    if (!base || base.getValue().getBitWidth() > 64)
      return std::nullopt;
    return uint32_t(base.getValue().getZExtValue() + offset);
  }

  // These are scalar effects of the current lowering, not hardware timing:
  // launch copies DRAM, then size, then materializes staging; an await reads
  // its first half before materializing the second. SELI writes the E bank.
  LogicalResult helperEffects(Operation &op, HelperRegisters &contents,
                              bool diagnose, HelperWriters *writers = nullptr) {
    auto write = [&](unsigned reg, std::optional<uint32_t> value) {
      if (reg == 0 || reg >= contents.size())
        return;
      contents[reg] = value;
      if (writers)
        (*writers)[reg] = &op;
    };
    auto halfSizeRead = [&]() {
      return diagnose ? helperRead(op, contents, fixed.halfSizeReg, 1024,
          "persistent DMA half-size helper is not preserved", writers) : success();
    };
    Value transfer, dram, size;
    bool store = false;
    if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      transfer = load.getTransfer(); dram = load.getDramByte(); size = load.getSizeBytes();
    } else if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
      transfer = load.getTransfer(); dram = load.getDramByte(); size = load.getSizeBytes();
    } else if (auto launch = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
      transfer = launch.getTransfer(); dram = launch.getDramByte(); size = launch.getSizeBytes(); store = true;
    } else if (auto launch = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
      transfer = launch.getTransfer(); dram = launch.getDramByte(); size = launch.getSizeBytes(); store = true;
    }
    if (transfer) {
      const DMATransferPlacement &p = *dmaPlacements.lookup(transfer);
      if (p.dramReg != registers.lookup(dram))
        write(p.dramReg, contents[registers.lookup(dram)]);
      if (p.sizeReg != registers.lookup(size))
        write(p.sizeReg, contents[registers.lookup(size)]);
      write(p.stagingReg, p.stagingWord);
      if (store && p.halves > 1) {
        write(p.stagingReg, p.stagingWord + 256);
        write(p.stagingReg, p.stagingWord);
      }
      return success();
    }
    if (auto await = dyn_cast<VirtualDMAAwaitFP8Op>(op))
      transfer = await.getTransfer();
    else if (auto await = dyn_cast<VirtualDMAAwaitBF16Op>(op))
      transfer = await.getTransfer();
    if (transfer) {
      const DMATransferPlacement &p = *dmaPlacements.lookup(transfer);
      if (awaitBasePolicy == DMAAwaitBasePolicy::Rematerialized)
        write(p.stagingReg, p.stagingWord);
      else if (diagnose && failed(helperRead(op, contents, p.stagingReg,
                   p.stagingWord, "pending DMA staging base is not preserved",
                   writers, transfer)))
        return failure();
      if (p.halves > 1)
        write(p.stagingReg, p.stagingWord + 256);
      return success();
    }

    bool input = isa<VirtualInputFP8Op, VirtualInputBF16Op>(op);
    bool output = isa<VirtualOutputBF16Op>(op);
    if (input || output) {
      uint64_t index = cast<IntegerAttr>(op.getAttr("index")).getValue().getZExtValue();
      unsigned halves = isa<VirtualInputFP8Op>(op) ? 1 : 2;
      for (unsigned half = 0; half < halves; ++half) {
        write(input ? fixed.inputBaseReg : fixed.outputBaseReg,
              uint32_t((input ? fixed.inputWord : fixed.outputWord) + index * 512 + half * 256));
        write(input ? fixed.inputDramReg : fixed.outputDramReg,
              boundaryAddress(input ? "atlas.input_dram_base" : "atlas.output_dram_base",
                              index * 2048 + half * 1024));
        if (failed(halfSizeRead()))
          return failure();
      }
      return success();
    }
    if (isa<VirtualPackFP8Op>(op)) {
      write(fixed.outputBaseReg, fixed.packWord);
      write(fixed.packSourceRegs[0], fixed.packWord * 4);
      write(fixed.packSourceRegs[1], fixed.packWord * 4 + 512);
      write(fixed.packDestinationReg, fixed.packRelayoutWord * 4);
      write(fixed.packRowReg, 0);
      write(fixed.packRowsReg, 32);
      // The relayout loop loads unknown memory and advances its pointers and
      // row counter. Do not assume a fixed iteration count for aliased helpers.
      for (unsigned reg : {fixed.packSourceRegs[0], fixed.packSourceRegs[1],
                           fixed.packDestinationReg, fixed.packRowReg,
                           fixed.packTemporaryRegs[0], fixed.packTemporaryRegs[1]})
        write(reg, std::nullopt);
      write(fixed.inputBaseReg, fixed.packRelayoutWord);
      return success();
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      write(registers.lookup(constant.getResult()), integer
          ? std::optional<uint32_t>(uint32_t(integer.getValue().getSExtValue()))
          : std::nullopt);
    } else if (auto add = dyn_cast<arith::AddIOp>(op)) {
      auto lhs = contents[registers.lookup(add.getLhs())];
      auto rhs = contents[registers.lookup(add.getRhs())];
      write(registers.lookup(add.getResult()), lhs && rhs
          ? std::optional<uint32_t>(uint32_t(uint64_t(*lhs) + *rhs)) : std::nullopt);
    } else if (isa<func::ReturnOp>(op)) {
      write(fixed.haltReg, 1);
    } else {
      for (Value result : op.getResults())
        if (isScalar(result))
          write(registers.lookup(result), std::nullopt);
    }
    return success();
  }

  FailureOr<HelperRegisters> helperEdge(Operation *terminator,
      Block *destination, ValueRange operands, HelperRegisters contents) {
    SmallVector<std::pair<unsigned, unsigned>> copies;
    for (auto [argument, incoming] : llvm::zip(destination->getArguments(), operands))
      if (isScalar(argument) && registers.lookup(argument) != registers.lookup(incoming))
        copies.emplace_back(registers.lookup(argument), registers.lookup(incoming));
    while (!copies.empty()) {
      bool copied = false;
      for (unsigned i = 0; i < copies.size(); ++i) {
        unsigned dst = copies[i].first;
        if (llvm::any_of(copies, [&](auto copy) { return copy.second == dst; }))
          continue;
        contents[dst] = contents[copies[i].second];
        copies.erase(copies.begin() + i);
        copied = true;
        break;
      }
      if (copied)
        continue;
      unsigned saved = copies.front().first;
      // Exactly the cycle-breaking write emitted by scalar parallel copies.
      if (fixed.scalarTemporary == 0 || fixed.scalarTemporary >= contents.size()) {
        terminator->emitOpError("scalar parallel-copy helper requires x1..x31");
        return failure();
      }
      contents[fixed.scalarTemporary] = contents[saved];
      for (auto &copy : copies)
        if (copy.second == saved)
          copy.second = fixed.scalarTemporary;
    }
    return contents;
  }

  LogicalResult verifyHelperContents() {
    // Existing non-DMA callers need not provide a complete fixed-helper ABI.
    if (dmaAssignments.empty() || function.getBody().empty())
      return success();
    HelperRegisters initial{};
    initial[0] = 0;
    if (fixed.halfSizeReg > 0 && fixed.halfSizeReg < 32)
      initial[fixed.halfSizeReg] = 1024;
    if (function.getNumArguments()) {
      if (fixed.inputBaseReg > 0 && fixed.inputBaseReg < 32)
        initial[fixed.inputBaseReg] = fixed.mailboxWord;
      if (fixed.inputDramReg > 0 && fixed.inputDramReg < 32)
        initial[fixed.inputDramReg] = boundaryAddress("atlas.control_dram_base", 0);
      if (failed(helperRead(*function.getOperation(), initial, fixed.halfSizeReg,
              1024, "persistent DMA half-size helper is not preserved", nullptr)))
        return failure();
      for (BlockArgument argument : function.getArguments())
        initial[registers.lookup(argument)].reset();
    }

    llvm::DenseMap<Block *, HelperRegisters> entries;
    Block *entry = &function.getBody().front();
    entries[entry] = initial;
    std::deque<Block *> work = {entry};
    while (!work.empty()) {
      Block *block = work.front();
      work.pop_front();
      HelperRegisters contents = entries.lookup(block);
      for (Operation &op : *block)
        (void)helperEffects(op, contents, false);
      LogicalResult edges = success();
      forEachEdge(block->getTerminator(), [&](Block *next, ValueRange operands) {
        if (failed(edges))
          return;
        auto incoming = helperEdge(block->getTerminator(), next, operands, contents);
        if (failed(incoming)) {
          edges = failure();
          return;
        }
        auto [found, inserted] = entries.try_emplace(next, *incoming);
        bool changed = inserted;
        if (!inserted)
          for (unsigned reg = 0; reg < incoming->size(); ++reg)
            if (found->second[reg] && found->second[reg] != (*incoming)[reg]) {
              found->second[reg].reset();
              changed = true;
            }
        if (changed)
          work.push_back(next);
      });
      if (failed(edges))
        return failure();
    }
    for (Block &block : function.getBody()) {
      if (!entries.contains(&block))
        continue;
      HelperRegisters contents = entries.lookup(&block);
      HelperWriters writers{};
      for (Operation &op : block)
        if (failed(helperEffects(op, contents, true, &writers)))
          return failure();
    }
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
      if (failed(verifyDMAWrites(op, live)))
        return failure();
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
  llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments;
  DMAAwaitBasePolicy awaitBasePolicy;
  llvm::DenseMap<Value, const DMATransferPlacement *> dmaPlacements;
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
    llvm::ArrayRef<int32_t> scalarArgumentRegs,
    llvm::ArrayRef<VirtualDMAAssignment> dmaAssignments,
    DMAAwaitBasePolicy awaitBasePolicy) {
  return AllocationVerifier(function, assignments, fixed, scalarArgumentRegs,
                            dmaAssignments, awaitBasePolicy)
      .verify();
}
