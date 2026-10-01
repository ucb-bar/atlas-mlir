#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// The selected RTL has 64 1-KiB tensor registers. BF16 uses even/odd pairs;
// pair 62 is reserved for cycle breaking in CFG edge copies. Scalar x27 is the
// corresponding edge-copy temporary. Live values are colored on separate
// physical locations; dead locations may be reused across CFG regions.
constexpr unsigned kTensorPairTemporary = 62;
constexpr unsigned kScalarTemporary = 27;
constexpr unsigned kFirstScalarValue = 10;
constexpr unsigned kLastScalarValue = 26;
constexpr uint64_t kTileBytes = 2048;
constexpr uint64_t kHalfBytes = 1024;
constexpr uint64_t kScratchBankWords = 65536; // selected 256-KiB VMEM bank
constexpr unsigned kDiagnosticDelay = 256;

struct PlannedOp {
  std::string name;
  SmallVector<NamedAttribute> attrs;
  Location loc;
  std::optional<unsigned> targetLabel;
};

class VirtualPlanner {
public:
  VirtualPlanner(ModuleOp module, func::FuncOp function)
      : module(module), function(function), attrs(module.getContext()) {}

  LogicalResult plan() {
    if (failed(readABI()) || failed(allocateValues()))
      return failure();
    Location loc = function.getLoc();
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(28)}, {"src", i32(0)},
         {"immediate", i32(1)}});
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(5)}, {"src", i32(0)},
         {"immediate", i32(0)}});
    for (unsigned channel : {0u, 1u})
      add("atlas.dma_config", loc,
          {{"channel", i32(channel)}, {"base_reg", i32(5)}});
    materializeScalar(2, kHalfBytes, loc);
    if (controlBase)
      stageScalarArguments(loc);

    for (Block &block : function.getBody()) {
      mark(labelFor(&block));
      for (Operation &op : block) {
        if (failed(lowerOperation(op)))
          return failure();
      }
    }
    if (planned.size() > 2048)
      return function.emitOpError("lowered program exceeds the 2048-word policy limit");
    return resolveTargets();
  }

  LogicalResult materialize() {
    for (Operation &op : llvm::make_early_inc_range(
             llvm::reverse(module.getBody()->getOperations())))
      op.erase();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    OperationState startState(module.getLoc(), "atlas.start");
    startState.addTypes(StateType::get(module.getContext()));
    Value state = builder.create(startState)->getResult(0);
    for (const PlannedOp &step : planned) {
      OperationState machineState(step.loc, step.name);
      machineState.addOperands(state);
      machineState.addTypes(StateType::get(module.getContext()));
      machineState.addAttributes(step.attrs);
      state = builder.create(machineState)->getResult(0);
    }
    module->setAttr("atlas.generated_from_virtual", builder.getUnitAttr());
    module->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(inputBase));
    module->setAttr("atlas.output_dram_base", builder.getI64IntegerAttr(outputBase));
    module->setAttr("atlas.scalar_arg_regs",
                    builder.getDenseI32ArrayAttr(scalarArgumentRegs));
    if (controlBase)
      module->setAttr("atlas.control_dram_base",
                      builder.getI64IntegerAttr(*controlBase));
    if (failed(verify(module)))
      return failure();
    llvm::SmallVector<uint32_t> words;
    return collectAtlasWords(module, words, /*llvmBlock=*/true);
  }

private:
  using Fields = std::initializer_list<std::pair<StringRef, Attribute>>;

  IntegerAttr i32(int64_t value) { return attrs.getI32IntegerAttr(value); }
  StringAttr str(StringRef value) { return attrs.getStringAttr(value); }
  BoolAttr boolean(bool value) { return attrs.getBoolAttr(value); }

  void add(StringRef name, Location loc, Fields fields = {},
           std::optional<unsigned> target = std::nullopt) {
    PlannedOp step{name.str(), {}, loc, target};
    for (const auto &[key, value] : fields)
      step.attrs.emplace_back(StringAttr::get(module.getContext(), key), value);
    planned.push_back(std::move(step));
  }

  unsigned newLabel() { return nextLabel++; }
  unsigned labelFor(Block *block) {
    auto it = blockLabels.find(block);
    if (it != blockLabels.end())
      return it->second;
    unsigned label = newLabel();
    blockLabels[block] = label;
    return label;
  }
  void mark(unsigned label) { labelPC[label] = planned.size(); }

  LogicalResult readABI() {
    auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
    auto output = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
    if (!input || !output)
      return function.emitOpError(
          "lowering requires explicit atlas.input_dram_base and atlas.output_dram_base");
    int64_t in = input.getValue().getSExtValue();
    int64_t out = output.getValue().getSExtValue();
    if (in < 0x80000000LL || out < 0x80000000LL ||
        in > std::numeric_limits<uint32_t>::max() ||
        out > std::numeric_limits<uint32_t>::max() ||
        (in % kHalfBytes) || (out % kHalfBytes))
      return function.emitOpError("DRAM bases must be aligned 32-bit selected-memory addresses");
    inputBase = static_cast<uint64_t>(in);
    outputBase = static_cast<uint64_t>(out);

    uint64_t maxInput = 0, maxOutput = 0;
    function.walk([&](Operation *op) {
      if (auto x = dyn_cast<VirtualInputBF16Op>(op))
        maxInput = std::max(maxInput,
                            static_cast<uint64_t>(x.getIndexAttr().getValue().getZExtValue()));
      if (auto x = dyn_cast<VirtualOutputBF16Op>(op))
        maxOutput = std::max(maxOutput,
                             static_cast<uint64_t>(x.getIndexAttr().getValue().getZExtValue()));
    });
    if (maxInput >= kScratchBankWords * 4 / kTileBytes ||
        maxOutput >= kScratchBankWords * 4 / kTileBytes)
      return function.emitOpError("input or output tile index exceeds its VMEM bank");
    uint64_t inputEnd = inputBase + (maxInput + 1) * kTileBytes;
    uint64_t outputEnd = outputBase + (maxOutput + 1) * kTileBytes;
    if (inputEnd > (1ULL << 32) || outputEnd > (1ULL << 32) ||
        (inputBase < outputEnd && outputBase < inputEnd))
      return function.emitOpError("input and output DRAM spans overlap or overflow");
    if (function.getNumArguments()) {
      auto control =
          function->getAttrOfType<IntegerAttr>("atlas.control_dram_base");
      if (!control)
        return function.emitOpError(
            "scalar arguments require atlas.control_dram_base mailbox binding");
      int64_t address = control.getValue().getSExtValue();
      if (address < 0x80000000LL ||
          address > std::numeric_limits<uint32_t>::max() ||
          address % kHalfBytes)
        return function.emitOpError(
            "control mailbox must have an aligned 32-bit DRAM address");
      controlBase = static_cast<uint64_t>(address);
      uint64_t controlEnd = *controlBase + kHalfBytes;
      if (controlEnd > (1ULL << 32) ||
          (*controlBase < inputEnd && inputBase < controlEnd) ||
          (*controlBase < outputEnd && outputBase < controlEnd))
        return function.emitOpError(
            "control mailbox overlaps a tensor buffer or overflows DRAM");
    }
    return success();
  }

  LogicalResult allocateValues() {
    if (failed(colorValues(/*tensor=*/true)) ||
        failed(colorValues(/*tensor=*/false)))
      return failure();
    for (BlockArgument arg : function.getArguments())
      scalarArgumentRegs.push_back(scalar(arg));
    return success();
  }

  LogicalResult colorValues(bool tensor) {
    using Set = llvm::DenseSet<Value>;
    auto selected = [&](Value value) {
      Type type = value.getType();
      return tensor ? isa<VirtualBF16Type>(type)
                    : type.isInteger(1) || type.isInteger(32);
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
    if (!tensor) {
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
    unsigned count = tensor ? kTensorPairTemporary / 2
                            : kLastScalarValue - kFirstScalarValue + 1;
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
        return function.emitOpError(tensor
            ? "virtual BF16 interference exceeds 31 physical pairs"
            : "virtual control interference exceeds 17 scalar registers");
    }
    for (Value value : values) {
      for (Value other : neighbors[value])
        if (colors[value] == colors[other])
          return function.emitOpError("internal register-coloring overlap");
      if (tensor)
        tileRegs[value] = 2 * colors[value];
      else
        scalarRegs[value] = kFirstScalarValue + colors[value];
    }
    return success();
  }

  unsigned tile(Value value) const { return tileRegs.find(value)->second; }
  unsigned scalar(Value value) const { return scalarRegs.find(value)->second; }

  void materializeScalar(unsigned dst, uint32_t value, Location loc) {
    uint32_t upper = ((static_cast<uint64_t>(value) + 0x800) >> 12) & 0xfffff;
    int32_t lower = static_cast<int32_t>(value & 0xfff);
    if (lower >= 2048)
      lower -= 4096;
    if (upper)
      add("atlas.upper", loc,
          {{"kind", str("lui")}, {"dst", i32(dst)},
           {"immediate", i32(upper)}});
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(dst)},
         {"src", i32(upper ? dst : 0)}, {"immediate", i32(lower)}});
  }

  void delay(Location loc, StringRef reason) {
    add("atlas.delay", loc,
        {{"cycles", i32(kDiagnosticDelay)},
         {"atlas.delay_reason", str(reason)}});
  }

  void stageScalarArguments(Location loc) {
    // The first 1-KiB VMEM window is a temporary mailbox. Tensor input DMA
    // may reuse it only after every asynchronous scalar LW has completed.
    materializeScalar(6, 0, loc);
    materializeScalar(1, static_cast<uint32_t>(*controlBase), loc);
    add("atlas.dma", loc,
        {{"direction", str("load")}, {"channel", i32(0)},
         {"reg", i32(6)}, {"dram", i32(1)}, {"size", i32(2)}});
    add("atlas.dma_wait", loc, {{"channel", i32(0)}});
    for (auto [index, arg] : llvm::enumerate(function.getArguments())) {
      unsigned reg = scalar(arg);
      add("atlas.scalar_load", loc,
          {{"kind", str("lw")}, {"dst", i32(reg)}, {"base", i32(0)},
           {"offset", i32(4 * index)}});
      add("atlas.delay", loc,
          {{"cycles", i32(8)},
           {"atlas.delay_reason", str("scalar_load_completion")}});
      if (arg.getType().isInteger(1))
        add("atlas.alu_imm", loc,
            {{"kind", str("andi")}, {"dst", i32(reg)},
             {"src", i32(reg)}, {"immediate", i32(1)}});
    }
  }

  void emitCopy(unsigned dst, unsigned src, bool tensor, Location loc) {
    if (dst == src)
      return;
    if (tensor) {
      add("atlas.vpu_unary", loc,
          {{"kind", str("mov")}, {"dst", i32(dst)}, {"src", i32(src)}});
      delay(loc, "cfg_tensor_copy");
    } else {
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(dst)},
           {"src", i32(src)}, {"immediate", i32(0)}});
    }
  }

  void parallelCopies(SmallVector<std::pair<unsigned, unsigned>> copies,
                      bool tensor, Location loc) {
    copies.erase(std::remove_if(copies.begin(), copies.end(),
                                [](auto copy) { return copy.first == copy.second; }),
                 copies.end());
    while (!copies.empty()) {
      bool progressed = false;
      for (size_t i = 0; i < copies.size(); ++i) {
        unsigned dst = copies[i].first;
        bool neededAsSource = llvm::any_of(copies, [&](auto copy) {
          return copy.second == dst;
        });
        if (!neededAsSource) {
          emitCopy(dst, copies[i].second, tensor, loc);
          copies.erase(copies.begin() + i);
          progressed = true;
          break;
        }
      }
      if (progressed)
        continue;
      unsigned saved = copies.front().first;
      unsigned temporary = tensor ? kTensorPairTemporary : kScalarTemporary;
      emitCopy(temporary, saved, tensor, loc);
      for (auto &copy : copies)
        if (copy.second == saved)
          copy.second = temporary;
    }
  }

  void edgeCopies(Block *dest, ValueRange operands, Location loc) {
    SmallVector<std::pair<unsigned, unsigned>> tensors, scalars;
    for (auto [arg, incoming] : llvm::zip(dest->getArguments(), operands)) {
      if (isa<VirtualStateType>(arg.getType()))
        continue;
      if (isa<VirtualBF16Type>(arg.getType()))
        tensors.emplace_back(tile(arg), tile(incoming));
      else
        scalars.emplace_back(scalar(arg), scalar(incoming));
    }
    parallelCopies(std::move(tensors), true, loc);
    parallelCopies(std::move(scalars), false, loc);
  }

  void jump(unsigned target, Location loc) {
    add("atlas.jump", loc,
        {{"kind", str("jal")}, {"dst", i32(0)}, {"base", i32(0)}}, target);
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(0)}, {"src", i32(0)},
         {"immediate", i32(0)}});
  }

  void inputHalf(unsigned dst, uint64_t index, unsigned half, Location loc) {
    uint32_t scratchWord = static_cast<uint32_t>(index * 512 + half * 256);
    uint32_t dramByte = static_cast<uint32_t>(inputBase + index * kTileBytes +
                                              half * kHalfBytes);
    materializeScalar(6, scratchWord, loc);
    materializeScalar(1, dramByte, loc);
    add("atlas.dma", loc,
        {{"direction", str("load")}, {"channel", i32(0)},
         {"reg", i32(6)}, {"dram", i32(1)}, {"size", i32(2)}});
    add("atlas.dma_wait", loc, {{"channel", i32(0)}});
    add("atlas.vload", loc,
        {{"dst", i32(dst)}, {"base", i32(6)}, {"offset", i32(0)},
         {"format", str("raw")}});
    delay(loc, "vload_completion");
  }

  void outputHalf(unsigned src, uint64_t index, unsigned half, Location loc) {
    uint32_t scratchWord = static_cast<uint32_t>(kScratchBankWords +
                                                 index * 512 + half * 256);
    uint32_t dramByte = static_cast<uint32_t>(outputBase + index * kTileBytes +
                                              half * kHalfBytes);
    materializeScalar(8, scratchWord, loc);
    add("atlas.vstore", loc,
        {{"src", i32(src)}, {"base", i32(8)}, {"offset", i32(0)},
         {"format", str("raw")}});
    delay(loc, "vstore_completion");
    materializeScalar(3, dramByte, loc);
    add("atlas.dma", loc,
        {{"direction", str("store")}, {"channel", i32(1)},
         {"reg", i32(8)}, {"dram", i32(3)}, {"size", i32(2)}});
    add("atlas.dma_wait", loc, {{"channel", i32(1)}});
  }

  LogicalResult lowerOperation(Operation &op) {
    Location loc = op.getLoc();
    if (isa<VirtualStartOp>(op))
      return success();
    if (auto input = dyn_cast<VirtualInputBF16Op>(op)) {
      uint64_t index = input.getIndexAttr().getValue().getZExtValue();
      for (unsigned half = 0; half < 2; ++half)
        inputHalf(tile(input.getValue()) + half, index, half, loc);
      return success();
    }
    if (auto output = dyn_cast<VirtualOutputBF16Op>(op)) {
      uint64_t index = output.getIndexAttr().getValue().getZExtValue();
      for (unsigned half = 0; half < 2; ++half)
        outputHalf(tile(output.getValue()) + half, index, half, loc);
      return success();
    }
    if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op)) {
      if (unary.getKind() != "mov" && unary.getKind() != "relu")
        return unary.emitOpError(
            "virtual-to-machine admission currently requires mov or relu");
      add("atlas.vpu_unary", loc,
          {{"kind", unary.getKindAttr()}, {"dst", i32(tile(unary.getDst()))},
           {"src", i32(tile(unary.getSrc()))}});
      delay(loc, "vpu_completion");
      return success();
    }
    if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op)) {
      return binary.emitOpError(
          "virtual-to-machine binary VPU modes require separate numerical and timing qualification");
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto value = dyn_cast<IntegerAttr>(constant.getValue());
      if (!value)
        return constant.emitOpError("only integer control constants lower to Atlas");
      materializeScalar(scalar(constant.getResult()),
                        static_cast<uint32_t>(value.getValue().getSExtValue()), loc);
      return success();
    }
    if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      add("atlas.alu_reg", loc,
          {{"kind", str("add")}, {"dst", i32(scalar(addi.getResult()))},
           {"lhs", i32(scalar(addi.getLhs()))},
           {"rhs", i32(scalar(addi.getRhs()))}});
      return success();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      return lowerCompare(cmp);
    if (auto branch = dyn_cast<cf::BranchOp>(op)) {
      edgeCopies(branch.getDest(), branch.getDestOperands(), loc);
      jump(labelFor(branch.getDest()), loc);
      return success();
    }
    if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
      unsigned trueEdge = newLabel();
      add("atlas.branch", loc,
          {{"kind", str("bne")}, {"lhs", i32(scalar(branch.getCondition()))},
           {"rhs", i32(0)}}, trueEdge);
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(0)}, {"src", i32(0)},
           {"immediate", i32(0)}});
      edgeCopies(branch.getFalseDest(), branch.getFalseDestOperands(), loc);
      jump(labelFor(branch.getFalseDest()), loc);
      mark(trueEdge);
      edgeCopies(branch.getTrueDest(), branch.getTrueDestOperands(), loc);
      jump(labelFor(branch.getTrueDest()), loc);
      return success();
    }
    if (isa<func::ReturnOp>(op)) {
      materializeScalar(1, 1, loc);
      add("atlas.csr", loc,
          {{"kind", str("rrw")}, {"dst", i32(0)}, {"source", i32(1)},
           {"address", i32(0xc10)}});
      add("atlas.trap", loc, {{"kind", str("ecall")}});
      return success();
    }
    return op.emitOpError("has no virtual-to-machine lowering");
  }

  LogicalResult lowerCompare(arith::CmpIOp cmp) {
    Location loc = cmp.getLoc();
    unsigned dst = scalar(cmp.getResult());
    unsigned lhs = scalar(cmp.getLhs());
    unsigned rhs = scalar(cmp.getRhs());
    auto emitReg = [&](StringRef kind, unsigned first, unsigned second) {
      add("atlas.alu_reg", loc,
          {{"kind", str(kind)}, {"dst", i32(dst)},
           {"lhs", i32(first)}, {"rhs", i32(second)}});
    };
    auto invert = [&]() {
      add("atlas.alu_imm", loc,
          {{"kind", str("xori")}, {"dst", i32(dst)},
           {"src", i32(dst)}, {"immediate", i32(1)}});
    };
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      emitReg("xor", lhs, rhs);
      if (cmp.getPredicate() == arith::CmpIPredicate::eq)
        add("atlas.alu_imm", loc,
            {{"kind", str("sltiu")}, {"dst", i32(dst)},
             {"src", i32(dst)}, {"immediate", i32(1)}});
      else
        emitReg("sltu", 0, dst);
      return success();
    case arith::CmpIPredicate::slt: emitReg("slt", lhs, rhs); return success();
    case arith::CmpIPredicate::sgt: emitReg("slt", rhs, lhs); return success();
    case arith::CmpIPredicate::sle: emitReg("slt", rhs, lhs); invert(); return success();
    case arith::CmpIPredicate::sge: emitReg("slt", lhs, rhs); invert(); return success();
    case arith::CmpIPredicate::ult: emitReg("sltu", lhs, rhs); return success();
    case arith::CmpIPredicate::ugt: emitReg("sltu", rhs, lhs); return success();
    case arith::CmpIPredicate::ule: emitReg("sltu", rhs, lhs); invert(); return success();
    case arith::CmpIPredicate::uge: emitReg("sltu", lhs, rhs); invert(); return success();
    }
    return cmp.emitOpError("unsupported integer comparison predicate");
  }

  LogicalResult resolveTargets() {
    for (size_t pc = 0; pc < planned.size(); ++pc) {
      PlannedOp &step = planned[pc];
      if (!step.targetLabel)
        continue;
      auto found = labelPC.find(*step.targetLabel);
      if (found == labelPC.end() || found->second >= planned.size())
        return function.emitOpError("unresolved virtual CFG target");
      int64_t offset = 2 * (static_cast<int64_t>(found->second) -
                            static_cast<int64_t>(pc));
      bool branch = step.name == "atlas.branch";
      int64_t bound = branch ? 4096 : 1048576;
      if (offset < -bound || offset > bound - 2)
        return function.emitOpError("virtual CFG target exceeds selected branch range");
      step.attrs.emplace_back(
          StringAttr::get(module.getContext(), branch ? "offset_bytes" : "offset"),
          i32(offset));
    }
    return success();
  }

  ModuleOp module;
  func::FuncOp function;
  Builder attrs;
  llvm::DenseMap<Value, unsigned> tileRegs, scalarRegs;
  llvm::DenseMap<Block *, unsigned> blockLabels;
  llvm::DenseMap<unsigned, size_t> labelPC;
  std::vector<PlannedOp> planned;
  unsigned nextLabel = 0;
  uint64_t inputBase = 0, outputBase = 0;
  std::optional<uint64_t> controlBase;
  SmallVector<int32_t> scalarArgumentRegs;
};

struct LowerAtlasVirtualToMachinePass
    : PassWrapper<LowerAtlasVirtualToMachinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAtlasVirtualToMachinePass)

  StringRef getArgument() const final { return "lower-atlas-virtual-to-machine"; }
  StringRef getDescription() const final {
    return "Allocate a bounded Atlas virtual BF16 CFG and emit a delayed physical word stream";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    if (!llvm::hasSingleElement(module.getBody()->getOperations())) {
      module.emitError("virtual-to-machine lowering requires exactly one function");
      return signalPassFailure();
    }
    auto function = dyn_cast<func::FuncOp>(module.getBody()->front());
    if (!function) {
      module.emitError("virtual-to-machine lowering requires func.func CFG form");
      return signalPassFailure();
    }
    VirtualPlanner planner(module, function);
    if (failed(planner.plan()) || failed(planner.materialize()))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasVirtualToMachinePass() {
  PassRegistration<LowerAtlasVirtualToMachinePass>();
}
