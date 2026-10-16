#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr uint64_t kTileBytes = 2048;
constexpr uint64_t kHalfBytes = 1024;

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
    if (failed(readABI()))
      return failure();
    allocated = true;
    if (failed(allocation.allocate(function)) || failed(allocation.verify()))
      return failure();
    int32_t nextValue = 0, nextBlock = 0, nextOperation = 0;
    for (Block &block : function.getBody()) {
      cfgBlocks[&block] = nextBlock++;
      for (Value v : block.getArguments())
        if (cfgTracked(v)) cfgValues[v] = nextValue++;
      for (Operation &op : block) {
        cfgOperations[&op] = nextOperation++;
        for (Value v : op.getResults())
          if (cfgTracked(v)) cfgValues[v] = nextValue++;
      }
      if (isa<cf::BranchOp, cf::CondBranchOp>(block.getTerminator()))
        for (unsigned n = 0; n < block.getNumSuccessors(); ++n)
          cfgEdges[&block].push_back(nextCFGEdge++);
    }
    Location loc = function.getLoc();
    // A DMA addresses DRAM as Cat(dmaBaseReg, low word), and DMA_CONFIG sets
    // that one upper word whatever its channel field (ScalarCore.scala). The
    // ABI keeps DRAM below 4 GiB, so one configuration from x0 states the
    // zero upper word every transfer relies on.
    add("atlas.dma_config", loc,
        {{"channel", i32(fixed().loadChannel)}, {"base_reg", i32(0)}});
    materializeScalar(fixed().halfSizeReg, kHalfBytes, loc);
    if (controlBase)
      stageScalarArguments(loc);

    for (Block &block : function.getBody()) {
      currentCFGBlock = cfgBlocks.lookup(&block);
      mark(labelFor(&block));
      for (Operation &op : block) {
        // A load's staging VLOADs go right before its first use, or before a
        // launch that reuses its staging window; the block's terminator
        // flushes the rest. They belong to the load.
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          flushLoads(op.getLoc());
        } else {
          flushReusedWindow(op);
          for (Value operand : op.getOperands())
            flushLoad(operand, op.getLoc());
        }
        size_t first = planned.size();
        if (failed(lowerOperation(op)))
          return failure();
        if (planned.size() != first && !op.hasTrait<OpTrait::IsTerminator>()) {
          for (size_t n = first; n < planned.size(); ++n) {
            planned[n].attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGSource), i32(cfgOperations.lookup(&op)));
            if (isa<VirtualPackFP8Op>(op))
              planned[n].attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGHelper), str("pack"));
          }
          planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGOperation), i32(cfgOperations.lookup(&op)));
          for (Value result : op.getResults()) {
            if (!cfgTracked(result) || pendingLoads.count(result)) continue;
            StringRef name = result.getType().isInteger(1) || result.getType().isInteger(32)
                ? kAtlasTagScalarResult : kAtlasTagTensorResult;
            planned.back().attrs.emplace_back(attrs.getStringAttr(name), i32(cfgValues.lookup(result)));
          }
        }
      }
    }
    loweredWords = planned.size();
    if (planned.size() > kAtlasGeneratedWordLimit)
      return function.emitOpError("lowered program exceeds the 2048-word policy limit");
    return resolveTargets();
  }

  LogicalResult materialize() {
    SmallVector<VirtualDMAAssignment> dmaAssignments;
    SmallVector<VirtualMXUAssignment> mxuAssignments;
    SmallVector<VirtualRegisterAssignment> registers, cfgRegisters;
    auto recordPlacement = [&](Value value) {
      Type type = value.getType();
      if (isa<VirtualBF16Type, VirtualFP8Type>(type)) {
        unsigned reg = isa<VirtualBF16Type>(type) ? allocation.tile(value) : allocation.fp8(value);
        registers.push_back({value, reg});
        cfgRegisters.push_back({value, reg});
      } else if (cfgTracked(value)) {
        cfgRegisters.push_back({value, allocation.scalar(value)});
      } else if (isa<VirtualMXUWeightType, VirtualMXUAccType>(type)) {
        mxuAssignments.push_back({value, allocation.mxu(value)});
      }
    };
    for (Block &block : function.getBody()) {
      for (Value argument : block.getArguments())
        recordPlacement(argument);
      for (Operation &op : block) {
        for (Value result : op.getResults())
          recordPlacement(result);
        if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
          Value transfer = op.getResult(0);
          dmaAssignments.push_back({transfer, allocation.dma(transfer)});
        }
      }
    }
    auto cfgContract = buildAtlasCFGContract(function, cfgRegisters);
    auto dmaContract = buildAtlasDMAContract(function, dmaAssignments);
    auto mxuContract = buildAtlasMXUContract(function, registers, mxuAssignments, fixed());
    auto tileContract = buildAtlasTileContract(function, registers, dmaAssignments, fixed(), allocation.scalarArguments());
    if (failed(cfgContract) || failed(dmaContract) || failed(mxuContract) || failed(tileContract))
      return failure();
    auto sourceMemoryContract = buildAtlasSourceMemoryEffectContract(function, *cfgContract, *tileContract);
    auto bufferContract = buildAtlasBufferContract(function, *cfgContract, *tileContract);
    if (failed(sourceMemoryContract) || failed(bufferContract))
      return failure();
    OwningOpRef<ModuleOp> emitted = ModuleOp::create(module.getLoc());
    (*emitted)->setAttrs(module->getAttrs());
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(emitted->getBody());
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
    (*emitted)->setAttr(kAtlasGeneratedMarker, builder.getStringAttr(kAtlasGeneratedVersion));
    (*emitted)->setAttr(kAtlasTimingState, builder.getStringAttr("untimed"));
    // Explicit DMA launches without waits; insert-atlas-dma-waits adds them.
    if (!dmaAssignments.empty())
      (*emitted)->setAttr(kAtlasDMAWaitsPending, builder.getUnitAttr());
    (*emitted)->removeAttr(kAtlasTimingProvider);
    (*emitted)->setAttr(kAtlasCFGContract, *cfgContract);
    (*emitted)->setAttr(kAtlasDMAContract, *dmaContract);
    (*emitted)->setAttr(kAtlasMXUContract, *mxuContract);
    (*emitted)->setAttr(kAtlasTileContract, *tileContract);
    (*emitted)->setAttr(kAtlasSourceMemoryContract, *sourceMemoryContract);
    (*emitted)->setAttr(kAtlasBufferContract, *bufferContract);
    (*emitted)->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(inputBase));
    (*emitted)->setAttr("atlas.output_dram_base", builder.getI64IntegerAttr(outputBase));
    (*emitted)->setAttr("atlas.scalar_arg_regs", builder.getDenseI32ArrayAttr(allocation.scalarArguments()));
    if (controlBase)
      (*emitted)->setAttr("atlas.control_dram_base", builder.getI64IntegerAttr(*controlBase));
    llvm::SmallVector<uint32_t> words;
    if (failed(verifyAtlasArtifact(*emitted, /*llvmBlock=*/true, words,
                                   /*allowPendingDMAWaits=*/true)))
      return failure();
    module->setAttrs((*emitted)->getAttrs());
    module.getBodyRegion().takeBody(emitted->getBodyRegion());
    return success();
  }

  // The allocation plan, once allocation has run, and the untimed program
  // size once every operation has lowered. Call before materialize(), which
  // replaces the function; the pass adds the final lowering status afterward.
  llvm::json::Object report() {
    llvm::json::Object report;
    if (allocated)
      report = allocation.report();
    report["function"] = function.getSymName();
    if (loweredWords)
      report["untimed_words"] =
          llvm::json::Object{{"words", *loweredWords},
                             {"limit", kAtlasGeneratedWordLimit}};
    return report;
  }

private:
  using Fields = std::initializer_list<std::pair<StringRef, Attribute>>;
  bool cfgTracked(Value v) const {
    Type t = v.getType();
    return t.isInteger(1) || t.isInteger(32) || isa<VirtualBF16Type, VirtualFP8Type>(t);
  }

  IntegerAttr i32(int64_t value) { return attrs.getI32IntegerAttr(value); }
  StringAttr str(StringRef value) { return attrs.getStringAttr(value); }
  BoolAttr boolean(bool value) { return attrs.getBoolAttr(value); }

  void add(StringRef name, Location loc, Fields fields = {},
           std::optional<unsigned> target = std::nullopt,
           std::optional<unsigned> tileCommand = std::nullopt) {
    PlannedOp step{name.str(), {}, loc, target};
    step.attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGBlock), i32(currentCFGBlock));
    if (currentCFGEdge)
      step.attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGEdge), i32(*currentCFGEdge));
    for (const auto &[key, value] : fields)
      step.attrs.emplace_back(StringAttr::get(module.getContext(), key), value);
    if (name == "atlas.mxu_push" || name == "atlas.mxu_matmul" || name == "atlas.mxu_pop")
      step.attrs.emplace_back(StringAttr::get(module.getContext(), kAtlasTagMXUCommand), i32(nextMXUCommand++));
    planned.push_back(std::move(step));
    if (name == "atlas.vload" || name == "atlas.vstore" || name == "atlas.dma" || name == "atlas.dma_wait")
      tagTileCommand(tileCommand);
  }

  // Commands are numbered in source expansion order; an atomic DMA reserves
  // its wait and VLOAD commands at the launch and issues them later.
  void tagTileCommand(std::optional<unsigned> id = std::nullopt) {
    planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagTileCommand),
                                      i32(id ? *id : nextTileCommand++));
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

  std::optional<uint32_t> constantAddress(Value value) {
    auto [entry, inserted] = constantAddresses.try_emplace(value, std::nullopt);
    if (!inserted)
      return entry->second;
    std::optional<uint32_t> result;
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        result = static_cast<uint32_t>(integer.getValue().getZExtValue());
    } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      auto lhs = constantAddress(add.getLhs());
      auto rhs = constantAddress(add.getRhs());
      if (lhs && rhs)
        result = static_cast<uint32_t>(static_cast<uint64_t>(*lhs) + *rhs);
    }
    constantAddresses[value] = result;
    return result;
  }

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
    bool hasPack = false;
    function.walk([&](Operation *op) {
      if (auto x = dyn_cast<VirtualInputBF16Op>(op))
        maxInput = std::max(maxInput,
                            static_cast<uint64_t>(x.getIndexAttr().getValue().getZExtValue()));
      if (auto x = dyn_cast<VirtualInputFP8Op>(op))
        maxInput = std::max(maxInput,
                            static_cast<uint64_t>(x.getIndexAttr().getValue().getZExtValue()));
      if (auto x = dyn_cast<VirtualOutputBF16Op>(op))
        maxOutput = std::max(maxOutput,
                             static_cast<uint64_t>(x.getIndexAttr().getValue().getZExtValue()));
      hasPack |= isa<VirtualPackFP8Op>(op);
    });
    if (maxInput >= fixed().inputWindowWords * 4 / kTileBytes ||
        maxOutput >= fixed().outputWindowWords * 4 / kTileBytes)
      return function.emitOpError("input or output tile index exceeds its VMEM bank");
    if (hasPack && maxInput >= (fixed().packWord - fixed().inputWord) / 512)
      return function.emitOpError(
          "FP8 pack scratch window requires input indexes below 64");
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
      WalkResult checked = function.walk([&](Operation *op) {
        bool fp8 = isa<VirtualDMALoadFP8Op, VirtualDMAStoreFP8Op>(op);
        bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
        if (!fp8 && !load && !isa<VirtualDMAStoreBF16Op>(op))
          return WalkResult::advance();
        auto address = constantAddress(op->getOperand(load ? 1 : 2));
        if (!address) {
          op->emitOpError("DMA address is not statically known");
          return WalkResult::interrupt();
        }
        uint64_t end = static_cast<uint64_t>(*address) +
                       (fp8 ? kHalfBytes : kTileBytes);
        if (*address < controlEnd && *controlBase < end) {
          op->emitOpError("explicit DMA span overlaps the control mailbox");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (checked.wasInterrupted())
        return failure();
    }
    return success();
  }

  const FixedResourcePlacement &fixed() const { return allocation.fixed(); }
  unsigned tile(Value value) const { return allocation.tile(value); }
  unsigned fp8(Value value) const { return allocation.fp8(value); }
  unsigned scalar(Value value) const { return allocation.scalar(value); }
  MXUPlacement mxu(Value value) const { return allocation.mxu(value); }

  // Loads `value` with the fewest instructions, as RISC-V `li` expands: LUI
  // sets the upper 20 bits, rounded because ADDI's 12-bit immediate is signed,
  // and each is left out when its part is zero.
  void materializeScalar(unsigned dst, uint32_t value, Location loc) {
    uint32_t upper = ((static_cast<uint64_t>(value) + 0x800) >> 12) & 0xfffff;
    int32_t lower = static_cast<int32_t>(value & 0xfff);
    if (lower >= 2048)
      lower -= 4096;
    if (upper)
      add("atlas.upper", loc,
          {{"kind", str("lui")}, {"dst", i32(dst)},
           {"immediate", i32(upper)}});
    if (lower || !upper)
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(dst)},
           {"src", i32(upper ? dst : 0)}, {"immediate", i32(lower)}});
  }

  void stageScalarArguments(Location loc) {
    // The first 1-KiB VMEM window is a temporary mailbox. Tensor input DMA
    // may reuse it only after every asynchronous scalar LW has completed.
    materializeScalar(fixed().inputBaseReg, fixed().mailboxWord, loc);
    materializeScalar(fixed().inputDramReg, static_cast<uint32_t>(*controlBase), loc);
    add("atlas.dma", loc,
        {{"direction", str("load")}, {"channel", i32(fixed().loadChannel)},
         {"reg", i32(fixed().inputBaseReg)},
         {"dram", i32(fixed().inputDramReg)},
         {"size", i32(fixed().halfSizeReg)}});
    add("atlas.dma_wait", loc, {{"channel", i32(fixed().loadChannel)}});
    for (auto [index, arg] : llvm::enumerate(function.getArguments())) {
      unsigned reg = scalar(arg);
      add("atlas.scalar_load", loc,
          {{"kind", str("lw")}, {"dst", i32(reg)}, {"base", i32(0)},
           {"offset", i32(4 * (fixed().mailboxWord + index))}});
      tagTileCommand();
      planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagScalarArgument), i32(cfgValues.lookup(arg)));
      if (arg.getType().isInteger(1))
        add("atlas.alu_imm", loc,
            {{"kind", str("andi")}, {"dst", i32(reg)},
             {"src", i32(reg)}, {"immediate", i32(1)}});
      planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagScalarResult), i32(cfgValues.lookup(arg)));
    }
  }

  void emitCopy(unsigned dst, unsigned src, bool tensor, Location loc) {
    if (dst == src)
      return;
    if (tensor) {
      add("atlas.vpu_unary", loc,
          {{"kind", str("mov")}, {"dst", i32(dst)}, {"src", i32(src)}});
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
      unsigned temporary =
          tensor ? fixed().tensorTemporary : fixed().scalarTemporary;
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
    uint32_t scratchWord = static_cast<uint32_t>(
        fixed().inputWord + index * 512 + half * 256);
    uint32_t dramByte = static_cast<uint32_t>(inputBase + index * kTileBytes +
                                              half * kHalfBytes);
    materializeScalar(fixed().inputBaseReg, scratchWord, loc);
    materializeScalar(fixed().inputDramReg, dramByte, loc);
    add("atlas.dma", loc,
        {{"direction", str("load")}, {"channel", i32(fixed().loadChannel)},
         {"reg", i32(fixed().inputBaseReg)},
         {"dram", i32(fixed().inputDramReg)},
         {"size", i32(fixed().halfSizeReg)}});
    add("atlas.dma_wait", loc, {{"channel", i32(fixed().loadChannel)}});
    add("atlas.vload", loc,
        {{"dst", i32(dst)}, {"base", i32(fixed().inputBaseReg)}, {"offset", i32(0)},
         {"format", str("raw")}});
  }

  void outputHalf(unsigned src, uint64_t index, unsigned half, Location loc) {
    uint32_t scratchWord = static_cast<uint32_t>(fixed().outputWord +
                                                 index * 512 + half * 256);
    uint32_t dramByte = static_cast<uint32_t>(outputBase + index * kTileBytes +
                                              half * kHalfBytes);
    materializeScalar(fixed().outputBaseReg, scratchWord, loc);
    add("atlas.vstore", loc,
        {{"src", i32(src)}, {"base", i32(fixed().outputBaseReg)}, {"offset", i32(0)},
         {"format", str("raw")}});
    materializeScalar(fixed().outputDramReg, dramByte, loc);
    add("atlas.dma", loc,
        {{"direction", str("store")}, {"channel", i32(fixed().storeChannel)},
         {"reg", i32(fixed().outputBaseReg)},
         {"dram", i32(fixed().outputDramReg)},
         {"size", i32(fixed().halfSizeReg)}});
    add("atlas.dma_wait", loc, {{"channel", i32(fixed().storeChannel)}});
  }

  void launchDMA(Value transfer, Value dramByte, Value sizeBytes,
                 std::optional<unsigned> src, Location loc) {
    const DMATransferPlacement &placement = allocation.dma(transfer);
    emitCopy(placement.dramReg, scalar(dramByte), false, loc);
    emitCopy(placement.sizeReg, scalar(sizeBytes), false, loc);
    materializeScalar(placement.stagingReg, placement.stagingWord, loc);
    if (src) {
      for (unsigned half = 0; half < placement.halves; ++half) {
        if (half)
          materializeScalar(placement.stagingReg,
                            placement.stagingWord + half * 256, loc);
        add("atlas.vstore", loc,
            {{"src", i32(*src + half)}, {"base", i32(placement.stagingReg)},
             {"offset", i32(0)}, {"format", str("raw")}});
      }
      if (placement.halves > 1)
        materializeScalar(placement.stagingReg, placement.stagingWord, loc);
    }
    add("atlas.dma", loc,
        {{"direction", str(src ? "store" : "load")},
         {"channel", i32(placement.channel)}, {"reg", i32(placement.stagingReg)},
         {"dram", i32(placement.dramReg)}, {"size", i32(placement.sizeReg)},
         {"atlas.virtual_dma_transfer", i32(placement.id)}});
  }

  // A launched load whose staging VLOADs wait for its tile's first use.
  struct PendingLoad {
    const DMATransferPlacement *placement;
    unsigned dst;
    unsigned firstVLoad;
    int32_t source, result;
  };

  void launchLoad(Operation &op, Value tile, unsigned dst) {
    Value transfer = op.getResult(0);
    launchDMA(transfer, op.getOperand(1), op.getOperand(2), std::nullopt, op.getLoc());
    const DMATransferPlacement &placement = allocation.dma(transfer);
    ++nextTileCommand; // its DMA.WAIT, which insert-atlas-dma-waits issues
    pendingLoads[tile] = {&placement, dst, nextTileCommand,
                          int32_t(cfgOperations.lookup(&op)),
                          int32_t(cfgValues.lookup(tile))};
    pendingOrder.push_back(tile);
    nextTileCommand += placement.halves;
  }

  void flushLoad(Value tile, Location loc) {
    auto found = pendingLoads.find(tile);
    if (found == pendingLoads.end())
      return;
    PendingLoad load = found->second;
    pendingLoads.erase(found);
    llvm::erase(pendingOrder, tile);
    size_t first = planned.size();
    // Transfers share the staging register, and the DMA latched its own copy
    // at launch, so another launch may have moved it since. Set it for each
    // VLOAD, as every scratch register is set where it is read.
    const DMATransferPlacement &placement = *load.placement;
    for (unsigned half = 0; half < placement.halves; ++half) {
      materializeScalar(placement.stagingReg, placement.stagingWord + half * 256, loc);
      add("atlas.vload", loc,
          {{"dst", i32(load.dst + half)}, {"base", i32(placement.stagingReg)},
           {"offset", i32(0)}, {"format", str("raw")}},
          std::nullopt, load.firstVLoad + half);
    }
    for (size_t n = first; n < planned.size(); ++n)
      planned[n].attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGSource), i32(load.source));
    planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagTensorResult), i32(load.result));
  }

  // placeDMA gives a launch an unread load's window only when all are taken.
  void flushReusedWindow(Operation &op) {
    if (!isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op))
      return;
    uint32_t window = allocation.dma(op.getResult(0)).stagingWord;
    for (Value tile : SmallVector<Value>(pendingOrder))
      if (pendingLoads.lookup(tile).placement->stagingWord == window)
        flushLoad(tile, op.getLoc());
  }

  void flushLoads(Location loc) {
    while (!pendingOrder.empty())
      flushLoad(pendingOrder.front(), loc);
  }

  LogicalResult lowerPack(VirtualPackFP8Op pack) {
    // Only the E8M0 scale codes observed on the selected RTL are admitted.
    if (pack.getScaleCode() < 122 || pack.getScaleCode() > 128)
      return pack.emitOpError(
          "virtual-to-machine pack admits observed E8M0 scale codes 122..128 only");
    Location loc = pack.getLoc();
    add("atlas.scalar_load", loc,
        {{"kind", str("seli")}, {"dst", i32(fixed().scaleReg)}, {"base", i32(0)},
         {"offset", i32(pack.getScaleCode())}});
    add("atlas.vpu_pack", loc,
        {{"direction", str("bf16_to_fp8")},
         {"dst", i32(fp8(pack.getResult()))}, {"src", i32(tile(pack.getSrc()))},
         {"scale_reg", i32(fixed().scaleReg)}});

    // The selected PACK joins 64 physical BF16 rows of 16 lanes. The MXU
    // expects 32 logical rows of 32 FP8 lanes. Stage the raw packed result
    // in the free upper half of input VMEM and interleave its two 512-byte
    // halves through a bounded 32-row scalar loop.
    materializeScalar(fixed().outputBaseReg, fixed().packWord, loc);
    add("atlas.vstore", loc,
        {{"src", i32(fp8(pack.getResult()))}, {"base", i32(fixed().outputBaseReg)},
         {"offset", i32(0)}, {"format", str("raw")}});
    materializeScalar(fixed().packSourceRegs[0], fixed().packWord * 4, loc);
    materializeScalar(fixed().packSourceRegs[1], fixed().packWord * 4 + 512, loc);
    materializeScalar(fixed().packDestinationReg, fixed().packRelayoutWord * 4, loc);
    materializeScalar(fixed().packRowReg, 0, loc);
    materializeScalar(fixed().packRowsReg, 32, loc);
    unsigned loop = newLabel();
    mark(loop);
    // A row copies eight words: copy i moves word i / 2 of half i % 2, from
    // that half's source to the row at 16 * half + 4 * word, through
    // temporary i % 2. Each store follows the next copy's load instead of its
    // own: it issues while the scalar load path is still busy, after its own
    // load has landed, and the next load reuses the temporary it just read.
    // The row counter advances in the first load's busy cycles, and each
    // pointer once its last word is copied.
    static_assert(kPackRelayoutRowWords == 8,
                  "the copy loop moves two halves of four words a row");
    auto copyLoad = [&](unsigned copy) {
      add("atlas.scalar_load", loc,
          {{"kind", str("lw")}, {"dst", i32(fixed().packTemporaryRegs[copy % 2])},
           {"base", i32(fixed().packSourceRegs[copy % 2])},
           {"offset", i32(4 * (copy / 2))}});
    };
    auto copyStore = [&](unsigned copy) {
      add("atlas.scalar_store", loc,
          {{"kind", str("sw")}, {"src", i32(fixed().packTemporaryRegs[copy % 2])},
           {"base", i32(fixed().packDestinationReg)},
           {"offset", i32(4 * (copy / 2) + 16 * (copy % 2))}});
    };
    auto advance = [&](unsigned reg, int32_t step) {
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(reg)}, {"src", i32(reg)},
           {"immediate", i32(step)}});
    };
    copyLoad(0);
    advance(fixed().packRowReg, 1);
    for (unsigned copy = 1; copy < kPackRelayoutRowWords; ++copy) {
      copyLoad(copy);
      copyStore(copy - 1);
      if (copy >= kPackRelayoutRowWords - 2)
        advance(fixed().packSourceRegs[copy % 2], 16);
    }
    copyStore(kPackRelayoutRowWords - 1);
    advance(fixed().packDestinationReg, 32);
    add("atlas.branch", loc,
        {{"kind", str("blt")}, {"lhs", i32(fixed().packRowReg)},
         {"rhs", i32(fixed().packRowsReg)}},
        loop);
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(0)}, {"src", i32(0)},
         {"immediate", i32(0)}});
    materializeScalar(fixed().inputBaseReg, fixed().packRelayoutWord, loc);
    add("atlas.vload", loc,
        {{"dst", i32(fp8(pack.getResult()))}, {"base", i32(fixed().inputBaseReg)},
         {"offset", i32(0)}, {"format", str("raw")}});
    return success();
  }

  LogicalResult lowerOperation(Operation &op) {
    Location loc = op.getLoc();
    if (isa<VirtualStartOp, VirtualScaleConstantOp>(op))
      return success();
    if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      launchLoad(op, load.getValue(), fp8(load.getValue()));
      return success();
    }
    if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
      launchLoad(op, load.getValue(), tile(load.getValue()));
      return success();
    }
    if (auto store = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
      launchDMA(store.getNext(), store.getDramByte(), store.getSizeBytes(),
                fp8(store.getSrc()), loc);
      ++nextTileCommand; // its DMA.WAIT, which insert-atlas-dma-waits issues
      return success();
    }
    if (auto store = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
      launchDMA(store.getNext(), store.getDramByte(), store.getSizeBytes(),
                tile(store.getSrc()), loc);
      ++nextTileCommand; // its DMA.WAIT, which insert-atlas-dma-waits issues
      return success();
    }
    if (auto input = dyn_cast<VirtualInputBF16Op>(op)) {
      uint64_t index = input.getIndexAttr().getValue().getZExtValue();
      for (unsigned half = 0; half < 2; ++half)
        inputHalf(tile(input.getValue()) + half, index, half, loc);
      return success();
    }
    if (auto input = dyn_cast<VirtualInputFP8Op>(op)) {
      uint64_t index = input.getIndexAttr().getValue().getZExtValue();
      // Each boundary index reserves a 2-KiB slot. An FP8 tile occupies its
      // first half, keeping mixed FP8/BF16 boundary addressing unambiguous.
      inputHalf(fp8(input.getValue()), index, 0, loc);
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
      MXUPlacement weight = mxu(load.getWeight());
      add("atlas.mxu_push", loc,
          {{"kind", str("weight_fp8")}, {"unit", i32(weight.unit)},
           {"src", i32(fp8(load.getSrc()))}, {"slot", i32(weight.slot)}});
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccFP8Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      add("atlas.mxu_push", loc,
          {{"kind", str("acc_fp8")}, {"unit", i32(acc.unit)},
           {"src", i32(fp8(load.getSrc()))}, {"slot", i32(acc.slot)}});
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccBF16Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      add("atlas.mxu_push", loc,
          {{"kind", str("acc_bf16")}, {"unit", i32(acc.unit)},
           {"src", i32(tile(load.getSrc()))}, {"slot", i32(acc.slot)}});
      return success();
    }
    if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
      MXUPlacement weight = mxu(reset.getWeight());
      MXUPlacement acc = mxu(reset.getAcc());
      add("atlas.mxu_matmul", loc,
          {{"unit", i32(acc.unit)}, {"src", i32(fp8(reset.getActivation()))},
           {"weight_slot", i32(weight.slot)}, {"acc_slot", i32(acc.slot)},
           {"accumulate", boolean(false)}});
      return success();
    }
    if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
      MXUPlacement weight = mxu(accumulate.getWeight());
      MXUPlacement acc = mxu(accumulate.getAcc());
      add("atlas.mxu_matmul", loc,
          {{"unit", i32(acc.unit)},
           {"src", i32(fp8(accumulate.getActivation()))},
           {"weight_slot", i32(weight.slot)}, {"acc_slot", i32(acc.slot)},
           {"accumulate", boolean(true)}});
      return success();
    }
    if (auto readout = dyn_cast<VirtualMXUReadoutBF16Op>(op)) {
      MXUPlacement acc = mxu(readout.getAcc());
      add("atlas.mxu_pop", loc,
          {{"format", str("bf16")}, {"unit", i32(acc.unit)},
           {"dst", i32(tile(readout.getValue()))},
           {"slot", i32(acc.slot)}, {"scale_reg", i32(0)}});
      return success();
    }
    if (auto readout = dyn_cast<VirtualMXUReadoutFP8Op>(op)) {
      auto scale = readout.getScale().getDefiningOp<VirtualScaleConstantOp>();
      if (!scale)
        return readout.emitOpError(
            "FP8 readout currently requires a virtual_scale_constant");
      MXUPlacement acc = mxu(readout.getAcc());
      // Rematerialize each use: readouts and VPU pack share scratch e3.
      add("atlas.scalar_load", loc,
          {{"kind", str("seli")}, {"dst", i32(fixed().scaleReg)}, {"base", i32(0)},
           {"offset", i32(scale.getCode())}});
      add("atlas.mxu_pop", loc,
          {{"format", str("fp8")}, {"unit", i32(acc.unit)},
           {"dst", i32(fp8(readout.getValue()))},
           {"slot", i32(acc.slot)}, {"scale_reg", i32(fixed().scaleReg)}});
      return success();
    }
    if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
      unsigned unit = static_cast<unsigned>(matmul.getUnit());
      add("atlas.mxu_push", loc,
          {{"kind", str("weight_fp8")}, {"unit", i32(unit)},
           {"src", i32(fp8(matmul.getWeight()))},
           {"slot", i32(fixed().mxuWeightSlot)}});
      add("atlas.mxu_matmul", loc,
          {{"unit", i32(unit)}, {"src", i32(fp8(matmul.getActivation()))},
           {"weight_slot", i32(fixed().mxuWeightSlot)},
           {"acc_slot", i32(fixed().mxuAccSlot)},
           {"accumulate", boolean(false)}});
      add("atlas.mxu_pop", loc,
          {{"format", str("bf16")}, {"unit", i32(unit)},
           {"dst", i32(tile(matmul.getResult()))},
           {"slot", i32(fixed().mxuAccSlot)}, {"scale_reg", i32(0)}});
      return success();
    }
    if (auto pack = dyn_cast<VirtualPackFP8Op>(op))
      return lowerPack(pack);
    if (auto output = dyn_cast<VirtualOutputBF16Op>(op)) {
      uint64_t index = output.getIndexAttr().getValue().getZExtValue();
      for (unsigned half = 0; half < 2; ++half)
        outputHalf(tile(output.getValue()) + half, index, half, loc);
      return success();
    }
    if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op)) {
      add("atlas.vpu_unary", loc,
          {{"kind", unary.getKindAttr()}, {"dst", i32(tile(unary.getDst()))},
           {"src", i32(tile(unary.getSrc()))}});
      return success();
    }
    if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op)) {
      add("atlas.vpu_binary", loc,
          {{"kind", binary.getKindAttr()},
           {"dst", i32(tile(binary.getDst()))},
           {"lhs", i32(tile(binary.getLhs()))},
           {"rhs", i32(tile(binary.getRhs()))}});
      return success();
    }
    if (auto reduce = dyn_cast<VirtualVPUReduceOp>(op)) {
      add("atlas.vpu_reduce", loc,
          {{"kind", reduce.getKindAttr()}, {"dst", i32(tile(reduce.getDst()))},
           {"src", i32(tile(reduce.getSrc()))}});
      return success();
    }
    if (auto splat = dyn_cast<VirtualVPUSplatOp>(op)) {
      add("atlas.vli", loc,
          {{"mode", str("all")}, {"dst", i32(tile(splat.getDst()))},
           {"immediate", i32(splat.getBits())}});
      return success();
    }
    if (auto transpose = dyn_cast<VirtualXLUTransposeOp>(op)) {
      add("atlas.xlu_transpose", loc,
          {{"dst", i32(fp8(transpose.getDst()))},
           {"src", i32(fp8(transpose.getSrc()))}});
      return success();
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
      currentCFGEdge = cfgEdges.lookup(op.getBlock())[0];
      edgeCopies(branch.getDest(), branch.getDestOperands(), loc);
      jump(labelFor(branch.getDest()), loc);
      currentCFGEdge.reset();
      return success();
    }
    if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
      unsigned trueEdge = newLabel();
      add("atlas.branch", loc,
          {{"kind", str("bne")}, {"lhs", i32(scalar(branch.getCondition()))},
           {"rhs", i32(0)}}, trueEdge);
      planned.back().attrs.emplace_back(attrs.getStringAttr(kAtlasTagCFGBranch), i32(currentCFGBlock));
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(0)}, {"src", i32(0)},
           {"immediate", i32(0)}});
      currentCFGEdge = cfgEdges.lookup(op.getBlock())[1];
      edgeCopies(branch.getFalseDest(), branch.getFalseDestOperands(), loc);
      jump(labelFor(branch.getFalseDest()), loc);
      mark(trueEdge);
      currentCFGEdge = cfgEdges.lookup(op.getBlock())[0];
      edgeCopies(branch.getTrueDest(), branch.getTrueDestOperands(), loc);
      jump(labelFor(branch.getTrueDest()), loc);
      currentCFGEdge.reset();
      return success();
    }
    if (isa<func::ReturnOp>(op)) {
      materializeScalar(fixed().haltReg, 1, loc);
      add("atlas.csr", loc,
          {{"kind", str("rrw")}, {"dst", i32(0)}, {"source", i32(fixed().haltReg)},
           {"address", i32(0xc10)}, {"atlas.complete", boolean(true)}});
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
  VirtualAllocationPlan allocation;
  llvm::DenseMap<Value, std::optional<uint32_t>> constantAddresses;
  llvm::DenseMap<Block *, unsigned> blockLabels;
  llvm::DenseMap<unsigned, size_t> labelPC;
  std::vector<PlannedOp> planned;
  llvm::DenseMap<Value, int32_t> cfgValues;
  llvm::DenseMap<Block *, int32_t> cfgBlocks;
  llvm::DenseMap<Operation *, int32_t> cfgOperations;
  llvm::DenseMap<Block *, SmallVector<int32_t, 2>> cfgEdges;
  int32_t currentCFGBlock = 0, nextCFGEdge = 0;
  std::optional<int32_t> currentCFGEdge;
  unsigned nextLabel = 0;
  unsigned nextMXUCommand = 0;
  unsigned nextTileCommand = 0;
  llvm::DenseMap<Value, PendingLoad> pendingLoads;
  SmallVector<Value> pendingOrder;
  uint64_t inputBase = 0, outputBase = 0;
  std::optional<uint64_t> controlBase;
  bool allocated = false;
  std::optional<size_t> loweredWords;
};

struct LowerAtlasVirtualToMachinePass
    : PassWrapper<LowerAtlasVirtualToMachinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAtlasVirtualToMachinePass)

  LowerAtlasVirtualToMachinePass() = default;
  LowerAtlasVirtualToMachinePass(const LowerAtlasVirtualToMachinePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "lower-atlas-virtual-to-machine"; }
  StringRef getDescription() const final {
    return "Allocate a bounded Atlas virtual CFG and emit an untimed physical "
           "instruction stream with its resource contracts; insert-atlas-delays "
           "or schedule-atlas-stream then times it";
  }

  Option<std::string> allocationReport{
      *this, "allocation-report",
      llvm::cl::desc("Write the allocation plan as JSON to this file, also "
                     "when it does not fit, with every error the lowering "
                     "reports")};

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
    if (allocationReport.empty()) {
      if (failed(planner.plan()) || failed(planner.materialize()))
        signalPassFailure();
      return;
    }
    llvm::json::Array failures;
    LogicalResult lowered = failure();
    llvm::json::Object report;
    {
      // Errors still reach the usual handler.
      ScopedDiagnosticHandler capture(&getContext(), [&](Diagnostic &d) {
        if (d.getSeverity() == DiagnosticSeverity::Error)
          failures.push_back(locationText(d.getLocation()) + ": " + d.str());
        return failure();
      });
      lowered = planner.plan();
      report = planner.report();
      if (succeeded(lowered))
        lowered = planner.materialize();
    }
    // Planning can fit but materialization can still reject its contracts.
    // Never publish a successful report for a failed lowering.
    report["fits"] = succeeded(lowered);
    report["failures"] = std::move(failures);
    std::string error;
    std::unique_ptr<llvm::ToolOutputFile> file =
        openOutputFile(allocationReport, &error);
    if (!file) {
      module.emitError(error);
      return signalPassFailure();
    }
    file->os() << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)))
               << "\n";
    file->os().flush();
    if (file->os().has_error()) {
      module.emitError("failed to write allocation report: ")
          << file->os().error().message();
      file->os().clear_error();
      return signalPassFailure();
    }
    file->keep();
    if (failed(lowered))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasVirtualToMachinePass() {
  PassRegistration<LowerAtlasVirtualToMachinePass>();
}
