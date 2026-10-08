#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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
constexpr uint64_t kTileBytes = 2048;
constexpr uint64_t kHalfBytes = 1024;
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
    if (failed(readABI()) || failed(allocation.allocate(function)) || failed(allocation.verify()))
      return failure();
    Location loc = function.getLoc();
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(fixed().oneReg)}, {"src", i32(0)},
         {"immediate", i32(1)}});
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(fixed().zeroReg)}, {"src", i32(0)},
         {"immediate", i32(0)}});
    for (unsigned channel : {fixed().loadChannel, fixed().storeChannel})
      add("atlas.dma_config", loc,
          {{"channel", i32(channel)}, {"base_reg", i32(fixed().zeroReg)}});
    materializeScalar(fixed().halfSizeReg, kHalfBytes, loc);
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
    SmallVector<VirtualDMAAssignment> dmaAssignments;
    SmallVector<VirtualMXUAssignment> mxuAssignments;
    SmallVector<VirtualRegisterAssignment> registers;
    auto recordPlacement = [&](Value value) {
      if (isa<VirtualBF16Type>(value.getType()))
        registers.push_back({value, allocation.tile(value)});
      else if (isa<VirtualFP8Type>(value.getType()))
        registers.push_back({value, allocation.fp8(value)});
      else if (isa<VirtualMXUWeightType, VirtualMXUAccType>(value.getType()))
        mxuAssignments.push_back({value, allocation.mxu(value)});
    };
    for (Block &block : function.getBody()) {
      for (Value argument : block.getArguments())
        recordPlacement(argument);
      for (Operation &op : block) {
        for (Value result : op.getResults())
          recordPlacement(result);
        if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
          Value handle = op.getResult(1);
          dmaAssignments.push_back({handle, allocation.dma(handle)});
        }
      }
    }
    auto dmaContract = buildAtlasDMAContract(function, dmaAssignments);
    auto mxuContract = buildAtlasMXUContract(function, registers, mxuAssignments, fixed());
    if (failed(dmaContract) || failed(mxuContract))
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
    (*emitted)->setAttr("atlas.generated_from_virtual", builder.getStringAttr("resource-contract-v1"));
    (*emitted)->setAttr("atlas.virtual_dma_contract", *dmaContract);
    (*emitted)->setAttr("atlas.virtual_mxu_contract", *mxuContract);
    (*emitted)->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(inputBase));
    (*emitted)->setAttr("atlas.output_dram_base", builder.getI64IntegerAttr(outputBase));
    (*emitted)->setAttr("atlas.scalar_arg_regs", builder.getDenseI32ArrayAttr(allocation.scalarArguments()));
    if (controlBase)
      (*emitted)->setAttr("atlas.control_dram_base", builder.getI64IntegerAttr(*controlBase));
    llvm::SmallVector<uint32_t> words;
    if (failed(collectAtlasWords(*emitted, words, /*llvmBlock=*/true)))
      return failure();
    module->setAttrs((*emitted)->getAttrs());
    module.getBodyRegion().takeBody(emitted->getBodyRegion());
    return success();
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
    if (name == "atlas.mxu_push" || name == "atlas.mxu_matmul" || name == "atlas.mxu_pop")
      step.attrs.emplace_back(StringAttr::get(module.getContext(), "atlas.virtual_mxu_command"), i32(nextMXUCommand++));
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
    delay(loc, "vload_completion");
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
    delay(loc, "vstore_completion");
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
        delay(loc, "vstore_completion");
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

  LogicalResult completeDMA(Value transfer, std::optional<unsigned> dst,
                            Operation &op) {
    const DMATransferPlacement &placement = allocation.dma(transfer);
    Location loc = op.getLoc();
    add("atlas.dma_wait", loc,
        {{"channel", i32(placement.channel)},
         {"atlas.virtual_dma_transfer", i32(placement.id)}});
    if (dst) {
      for (unsigned half = 0; half < placement.halves; ++half) {
        if (half)
          materializeScalar(placement.stagingReg,
                            placement.stagingWord + half * 256, loc);
        add("atlas.vload", loc,
            {{"dst", i32(*dst + half)}, {"base", i32(placement.stagingReg)},
             {"offset", i32(0)}, {"format", str("raw")}});
        delay(loc, "vload_completion");
      }
    }
    return success();
  }

  LogicalResult lowerPack(VirtualPackFP8Op pack) {
    if (pack.getScaleCode() != 127)
      return pack.emitOpError(
          "virtual-to-machine pack currently admits unit E8M0 scale code 127 only");
    Location loc = pack.getLoc();
    add("atlas.scalar_load", loc,
        {{"kind", str("seli")}, {"dst", i32(fixed().scaleReg)}, {"base", i32(0)},
         {"offset", i32(pack.getScaleCode())}});
    add("atlas.vpu_pack", loc,
        {{"direction", str("bf16_to_fp8")},
         {"dst", i32(fp8(pack.getResult()))}, {"src", i32(tile(pack.getSrc()))},
         {"scale_reg", i32(fixed().scaleReg)}});
    delay(loc, "vpu_pack_completion");

    // The selected PACK joins 64 physical BF16 rows of 16 lanes. The MXU
    // expects 32 logical rows of 32 FP8 lanes. Stage the raw packed result
    // in the free upper half of input VMEM and interleave its two 512-byte
    // halves through a bounded 32-row scalar loop.
    materializeScalar(fixed().outputBaseReg, fixed().packWord, loc);
    add("atlas.vstore", loc,
        {{"src", i32(fp8(pack.getResult()))}, {"base", i32(fixed().outputBaseReg)},
         {"offset", i32(0)}, {"format", str("raw")}});
    delay(loc, "pack_vstore_completion");
    materializeScalar(fixed().packSourceRegs[0], fixed().packWord * 4, loc);
    materializeScalar(fixed().packSourceRegs[1], fixed().packWord * 4 + 512, loc);
    materializeScalar(fixed().packDestinationReg, fixed().packRelayoutWord * 4, loc);
    materializeScalar(fixed().packRowReg, 0, loc);
    materializeScalar(fixed().packRowsReg, 32, loc);
    unsigned loop = newLabel();
    mark(loop);
    for (unsigned word = 0; word < 4; ++word) {
      for (unsigned half = 0; half < 2; ++half) {
        unsigned source = fixed().packSourceRegs[half];
        unsigned temp = fixed().packTemporaryRegs[half];
        add("atlas.scalar_load", loc,
            {{"kind", str("lw")}, {"dst", i32(temp)},
             {"base", i32(source)}, {"offset", i32(4 * word)}});
        add("atlas.delay", loc,
            {{"cycles", i32(8)},
             {"atlas.delay_reason", str("scalar_load_completion")}});
        add("atlas.scalar_store", loc,
            {{"kind", str("sw")}, {"src", i32(temp)},
             {"base", i32(fixed().packDestinationReg)},
             {"offset", i32(4 * word + 16 * half)}});
      }
    }
    for (unsigned reg : {fixed().packSourceRegs[0], fixed().packSourceRegs[1],
                         fixed().packDestinationReg})
      add("atlas.alu_imm", loc,
          {{"kind", str("addi")}, {"dst", i32(reg)}, {"src", i32(reg)},
           {"immediate", i32(reg == fixed().packDestinationReg ? 32 : 16)}});
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(fixed().packRowReg)},
         {"src", i32(fixed().packRowReg)},
         {"immediate", i32(1)}});
    add("atlas.branch", loc,
        {{"kind", str("blt")}, {"lhs", i32(fixed().packRowReg)},
         {"rhs", i32(fixed().packRowsReg)}},
        loop);
    add("atlas.alu_imm", loc,
        {{"kind", str("addi")}, {"dst", i32(0)}, {"src", i32(0)},
         {"immediate", i32(0)}});
    delay(loc, "scalar_relayout_completion");
    materializeScalar(fixed().inputBaseReg, fixed().packRelayoutWord, loc);
    add("atlas.vload", loc,
        {{"dst", i32(fp8(pack.getResult()))}, {"base", i32(fixed().inputBaseReg)},
         {"offset", i32(0)}, {"format", str("raw")}});
    delay(loc, "pack_relayout_vload_completion");
    return success();
  }

  LogicalResult lowerOperation(Operation &op) {
    Location loc = op.getLoc();
    if (isa<VirtualStartOp, VirtualScaleConstantOp>(op))
      return success();
    if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
      launchDMA(load.getTransfer(), load.getDramByte(), load.getSizeBytes(),
                std::nullopt, loc);
      return success();
    }
    if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
      launchDMA(load.getTransfer(), load.getDramByte(), load.getSizeBytes(),
                std::nullopt, loc);
      return success();
    }
    if (auto await = dyn_cast<VirtualDMAAwaitFP8Op>(op))
      return completeDMA(await.getTransfer(), fp8(await.getValue()), op);
    if (auto await = dyn_cast<VirtualDMAAwaitBF16Op>(op))
      return completeDMA(await.getTransfer(), tile(await.getValue()), op);
    if (auto store = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
      launchDMA(store.getTransfer(), store.getDramByte(), store.getSizeBytes(),
                fp8(store.getSrc()), loc);
      return success();
    }
    if (auto store = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
      launchDMA(store.getTransfer(), store.getDramByte(), store.getSizeBytes(),
                tile(store.getSrc()), loc);
      return success();
    }
    if (auto wait = dyn_cast<VirtualDMAWaitOp>(op))
      return completeDMA(wait.getTransfer(), std::nullopt, op);
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
      delay(loc, "mxu_weight_completion");
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccFP8Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      add("atlas.mxu_push", loc,
          {{"kind", str("acc_fp8")}, {"unit", i32(acc.unit)},
           {"src", i32(fp8(load.getSrc()))}, {"slot", i32(acc.slot)}});
      delay(loc, "mxu_accumulator_completion");
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccBF16Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      add("atlas.mxu_push", loc,
          {{"kind", str("acc_bf16")}, {"unit", i32(acc.unit)},
           {"src", i32(tile(load.getSrc()))}, {"slot", i32(acc.slot)}});
      delay(loc, "mxu_accumulator_completion");
      return success();
    }
    if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
      MXUPlacement weight = mxu(reset.getWeight());
      MXUPlacement acc = mxu(reset.getAcc());
      add("atlas.mxu_matmul", loc,
          {{"unit", i32(acc.unit)}, {"src", i32(fp8(reset.getActivation()))},
           {"weight_slot", i32(weight.slot)}, {"acc_slot", i32(acc.slot)},
           {"accumulate", boolean(false)}});
      delay(loc, "mxu_matmul_completion");
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
      delay(loc, "mxu_matmul_completion");
      return success();
    }
    if (auto readout = dyn_cast<VirtualMXUReadoutBF16Op>(op)) {
      MXUPlacement acc = mxu(readout.getAcc());
      add("atlas.mxu_pop", loc,
          {{"format", str("bf16")}, {"unit", i32(acc.unit)},
           {"dst", i32(tile(readout.getValue()))},
           {"slot", i32(acc.slot)}, {"scale_reg", i32(0)}});
      delay(loc, "mxu_readout_completion");
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
      delay(loc, "mxu_readout_completion");
      return success();
    }
    if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
      unsigned unit = static_cast<unsigned>(matmul.getUnit());
      add("atlas.mxu_push", loc,
          {{"kind", str("weight_fp8")}, {"unit", i32(unit)},
           {"src", i32(fp8(matmul.getWeight()))},
           {"slot", i32(fixed().mxuWeightSlot)}});
      delay(loc, "mxu_weight_completion");
      add("atlas.mxu_matmul", loc,
          {{"unit", i32(unit)}, {"src", i32(fp8(matmul.getActivation()))},
           {"weight_slot", i32(fixed().mxuWeightSlot)},
           {"acc_slot", i32(fixed().mxuAccSlot)},
           {"accumulate", boolean(false)}});
      delay(loc, "mxu_matmul_completion");
      add("atlas.mxu_pop", loc,
          {{"format", str("bf16")}, {"unit", i32(unit)},
           {"dst", i32(tile(matmul.getResult()))},
           {"slot", i32(fixed().mxuAccSlot)}, {"scale_reg", i32(0)}});
      delay(loc, "mxu_readout_completion");
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
      if (binary.getKind() != "add")
        return binary.emitOpError(
            "virtual-to-machine binary VPU admission currently requires add");
      add("atlas.vpu_binary", loc,
          {{"kind", binary.getKindAttr()},
           {"dst", i32(tile(binary.getDst()))},
           {"lhs", i32(tile(binary.getLhs()))},
           {"rhs", i32(tile(binary.getRhs()))}});
      delay(loc, "vpu_completion");
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
      materializeScalar(fixed().haltReg, 1, loc);
      add("atlas.csr", loc,
          {{"kind", str("rrw")}, {"dst", i32(0)}, {"source", i32(fixed().haltReg)},
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
  VirtualAllocationPlan allocation;
  llvm::DenseMap<Value, std::optional<uint32_t>> constantAddresses;
  llvm::DenseMap<Block *, unsigned> blockLabels;
  llvm::DenseMap<unsigned, size_t> labelPC;
  std::vector<PlannedOp> planned;
  unsigned nextLabel = 0;
  unsigned nextMXUCommand = 0;
  uint64_t inputBase = 0, outputBase = 0;
  std::optional<uint64_t> controlBase;
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
