#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasImmutableWeightReuse.h"
#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasStream.h"
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

class VirtualPlanner {
public:
  VirtualPlanner(ModuleOp module, func::FuncOp function, ArrayRef<unsigned> storeWindowRanks = {},
                 ArrayRef<unsigned> mregRanks = {}, bool packReloadAfterWeight = false, bool packWeightBeforePack = false)
      : module(module), function(function), attrs(module.getContext()), emitter(module.getContext()),
        allocation(storeWindowRanks, mregRanks), packReloadAfterWeight(packReloadAfterWeight),
        packWeightBeforePack(packWeightBeforePack) {}

  LogicalResult plan() {
    if (failed(readABI()))
      return failure();
    emitted = ModuleOp::create(module.getLoc());
    emitter.setInsertionPointToEnd(emitted->getBody());
    state = StartOp::create(emitter, module.getLoc(), stateType()).getResult();
    allocated = true;
    if (failed(allocation.allocate(function)) || failed(allocation.verify()))
      return failure();
    for (Block &block : function.getBody())
      for (Value argument : block.getArguments())
        if (isa<VirtualFP8Type>(argument.getType()))
          return function.emitOpError("FP8 edge values require --spill-atlas-virtual");
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
    emit<DMAConfigOp>(loc, /*channel=*/fixed().loadChannel, /*base_reg=*/0);
    materializeScalar(fixed().halfSizeReg, kAtlasRegisterBytes, loc);
    if (controlBase)
      stageScalarArguments(loc);

    for (Block &block : function.getBody()) {
      packReload.reset();
      currentCFGBlock = cfgBlocks.lookup(&block);
      mark(labelFor(&block));
      for (Operation &op : block) {
        // A load's staging VLOADs go right before its first use, or before a
        // launch that reuses its staging window; the block's terminator
        // flushes the rest. They belong to the load.
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          flushLoads(op.getLoc());
        } else {
          reorderPackReload(op);
          flushReusedWindow(op);
          for (Value operand : op.getOperands())
            flushLoad(operand, op.getLoc());
          preloadPackWeight(op);
        }
        size_t first = planned.size();
        if (failed(lowerOperation(op)))
          return failure();
        if (planned.size() != first && !op.hasTrait<OpTrait::IsTerminator>()) {
          for (size_t n = first; n < planned.size(); ++n) {
            planned[n]->setAttr(kAtlasTagCFGSource, i32(cfgOperations.lookup(&op)));
            if (isa<VirtualPackFP8Op>(op))
              planned[n]->setAttr(kAtlasTagCFGHelper, str("pack"));
          }
          planned.back()->setAttr(kAtlasTagCFGOperation, i32(cfgOperations.lookup(&op)));
          for (Value result : op.getResults()) {
            if (!cfgTracked(result) || pendingLoads.count(result)) continue;
            StringRef name = registerKindOf(result) == RegisterKind::Scalar
                ? kAtlasTagScalarResult : kAtlasTagTensorResult;
            planned.back()->setAttr(name, i32(cfgValues.lookup(result)));
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
    SmallVector<VirtualRegisterAssignment> registers, cfgRegisters, scalarRegisters;
    auto recordPlacement = [&](Value value) {
      std::optional<RegisterKind> kind = registerKindOf(value);
      if (kind == RegisterKind::Scalar) {
        cfgRegisters.push_back({value, allocation.scalar(value)});
        scalarRegisters.push_back({value, allocation.scalar(value)});
      } else if (kind) {
        unsigned reg = kind == RegisterKind::BF16 ? allocation.tile(value) : allocation.fp8(value);
        registers.push_back({value, reg});
        cfgRegisters.push_back({value, reg});
      } else if (isa<VirtualMXUWeightType, VirtualMXUAccType>(value.getType())) {
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
    auto tileContract = buildAtlasTileContract(function, registers, dmaAssignments, fixed(), allocation.scalarArguments(), scalarRegisters);
    if (failed(cfgContract) || failed(dmaContract) || failed(mxuContract) || failed(tileContract))
      return failure();
    auto sourceMemoryContract = buildAtlasSourceMemoryEffectContract(function, *cfgContract, *tileContract);
    auto bufferContract = buildAtlasBufferContract(function, *cfgContract, *tileContract);
    if (failed(sourceMemoryContract) || failed(bufferContract))
      return failure();
    (*emitted)->setAttrs(module->getAttrs());
    Builder builder(module.getContext());
    (*emitted)->setAttr(kAtlasGeneratedMarker, builder.getStringAttr(kAtlasGeneratedVersion));
    (*emitted)->setAttr(kAtlasTimingState, builder.getStringAttr("untimed"));
    // Explicit DMA launches without waits; insert-atlas-dma-waits adds them.
    if (!dmaAssignments.empty())
      (*emitted)->setAttr(kAtlasDMAWaitsPending, builder.getUnitAttr());
    // Branches and jumps without delay slots; the timing passes fill them.
    if (!redirects.empty())
      (*emitted)->setAttr(kAtlasDelaySlotsPending, builder.getUnitAttr());
    (*emitted)->removeAttr(kAtlasTimingProvider);
    (*emitted)->setAttr(kAtlasCFGContract, *cfgContract);
    (*emitted)->setAttr(kAtlasDMAContract, *dmaContract);
    (*emitted)->setAttr(kAtlasMXUContract, *mxuContract);
    (*emitted)->setAttr(kAtlasTileContract, *tileContract);
    (*emitted)->setAttr(kAtlasSourceMemoryContract, *sourceMemoryContract);
    (*emitted)->setAttr(kAtlasBufferContract, *bufferContract);
    (*emitted)->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(abi.inputBase));
    (*emitted)->setAttr("atlas.output_dram_base", builder.getI64IntegerAttr(abi.outputBase));
    (*emitted)->setAttr("atlas.scalar_arg_regs", builder.getDenseI32ArrayAttr(allocation.scalarArguments()));
    if (packWeightBeforePack)
      (*emitted)->setAttr("atlas.model_preloaded_weights", builder.getDenseI32ArrayAttr(preloadedWeights));
    if (controlBase)
      (*emitted)->setAttr("atlas.control_dram_base", builder.getI64IntegerAttr(*controlBase));
    if (function->hasAttr("atlas.memory_controls")) {
      SmallVector<int32_t> types;
      for (Type type : function.getArgumentTypes()) types.push_back(cast<IntegerType>(type).getWidth());
      (*emitted)->setAttr("atlas.control_argument_types", builder.getDenseI32ArrayAttr(types));
    }
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
  bool cfgTracked(Value v) const { return registerKindOf(v).has_value(); }

  IntegerAttr i32(int64_t value) { return attrs.getI32IntegerAttr(value); }
  StringAttr str(StringRef value) { return attrs.getStringAttr(value); }
  BoolAttr boolean(bool value) { return attrs.getBoolAttr(value); }

  Type stateType() { return StateType::get(module.getContext()); }

  // Appends the machine op `OpTy`, built from its fields, to the state chain.
  template <typename OpTy, typename... Fields>
  OpTy emit(Location loc, Fields &&...fields) {
    OpTy op = OpTy::create(emitter, loc, stateType(), state,
                           std::forward<Fields>(fields)...);
    record(op);
    return op;
  }

  // Continues the state chain from `op` and tags it with its CFG block and
  // edge and its MXU or tile command. Commands are numbered in source
  // expansion order; an atomic DMA reserves its wait and VLOAD commands at the
  // launch and issues them later as `tileCommand`.
  void record(Operation *op, std::optional<unsigned> tileCommand = std::nullopt) {
    state = op->getResult(0);
    planned.push_back(op);
    op->setAttr(kAtlasTagCFGBlock, i32(currentCFGBlock));
    if (currentCFGEdge)
      op->setAttr(kAtlasTagCFGEdge, i32(*currentCFGEdge));
    if (isa<MXUPushOp, MXUMatmulOp, MXUPopOp>(op))
      op->setAttr(kAtlasTagMXUCommand, i32(nextMXUCommand++));
    if (isa<VLoadOp, VStoreOp, DMAOp, DMAWaitOp>(op))
      op->setAttr(kAtlasTagTileCommand,
                  i32(tileCommand ? *tileCommand : nextTileCommand++));
  }

  // A mailbox or slot load or store that is a tile command too.
  void tagTileCommand() {
    planned.back()->setAttr(kAtlasTagTileCommand, i32(nextTileCommand++));
  }

  // A branch or jump to `label`, whose offset resolveTargets sets.
  void redirect(Operation *op, unsigned label) {
    assert(op == planned.back() && "redirects the op just emitted");
    redirects.push_back({planned.size() - 1, label});
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
    FailureOr<AtlasBoundaryABI> read = AtlasBoundaryABI::read(function);
    if (failed(read))
      return failure();
    abi = *read;
    // The boundary tiles' VMEM windows, and the PACK scratch after the input.
    bool hasPack = false;
    function.walk([&](Operation *op) { hasPack |= isa<VirtualPackFP8Op>(op); });
    if (abi.inputSlots > fixed().inputWindowWords / kAtlasBF16TileWords ||
        abi.outputSlots > fixed().outputWindowWords / kAtlasBF16TileWords)
      return function.emitOpError("input or output tile index exceeds its VMEM bank");
    if (hasPack &&
        abi.inputSlots > (fixed().packWord - fixed().inputWord) / kAtlasBF16TileWords)
      return function.emitOpError(
          "FP8 pack scratch window requires input indexes below 64");
    if (failed(abi.verifySpans(function)))
      return failure();
    if (function.getNumArguments()) {
      auto control =
          function->getAttrOfType<IntegerAttr>("atlas.control_dram_base");
      if (!control)
        return function.emitOpError(
            "scalar arguments require atlas.control_dram_base mailbox binding");
      int64_t address = control.getValue().getSExtValue();
      if (address < 0x80000000LL ||
          address > std::numeric_limits<uint32_t>::max() ||
          address % kAtlasRegisterBytes)
        return function.emitOpError(
            "control mailbox must have an aligned 32-bit DRAM address");
      // The mailbox is one register's worth of controls.
      controlBase = static_cast<uint64_t>(address);
      uint64_t controlEnd = *controlBase + kAtlasRegisterBytes;
      if (controlEnd > (1ULL << 32) ||
          (*controlBase < abi.inputEnd() && abi.inputBase < controlEnd) ||
          (*controlBase < abi.outputEnd() && abi.outputBase < controlEnd))
        return function.emitOpError(
            "control mailbox overlaps a tensor buffer or overflows DRAM");
      WalkResult checked = function.walk([&](Operation *op) {
        std::optional<VirtualDMATransfer> transfer = virtualDMATransfer(op);
        if (!transfer)
          return WalkResult::advance();
        auto address = constantAddress(transfer->dram->get());
        if (!address) {
          op->emitOpError("DMA address is not statically known");
          return WalkResult::interrupt();
        }
        bool fp8 = isa<VirtualDMALoadFP8Op, VirtualDMAStoreFP8Op>(op);
        uint64_t end = static_cast<uint64_t>(*address) +
                       (fp8 ? kAtlasRegisterBytes : kAtlasBF16TileBytes);
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
      emit<UpperOp>(loc, "lui", dst, /*immediate=*/upper);
    if (lower || !upper)
      emit<ALUImmOp>(loc, "addi", dst, /*src=*/upper ? dst : 0, /*immediate=*/lower);
  }

  void stageScalarArguments(Location loc) {
    // The first 1-KiB VMEM window is a temporary mailbox. Tensor input DMA
    // may reuse it only after every asynchronous scalar LW has completed.
    materializeScalar(fixed().inputBaseReg, fixed().mailboxWord, loc);
    materializeScalar(fixed().inputDramReg, static_cast<uint32_t>(*controlBase), loc);
    emit<DMAOp>(loc, "load", /*channel=*/fixed().loadChannel,
        /*reg=*/fixed().inputBaseReg, /*dram=*/fixed().inputDramReg,
        /*size=*/fixed().halfSizeReg);
    emit<DMAWaitOp>(loc, /*channel=*/fixed().loadChannel);
    if (function->hasAttr("atlas.memory_controls")) return;
    for (auto [index, arg] : llvm::enumerate(function.getArguments())) {
      unsigned reg = scalar(arg);
      emit<ScalarLoadOp>(loc, "lw", /*dst=*/reg, /*base=*/0,
          /*offset=*/4 * (fixed().mailboxWord + index));
      tagTileCommand();
      planned.back()->setAttr(kAtlasTagScalarArgument, i32(cfgValues.lookup(arg)));
      if (arg.getType().isInteger(1))
        emit<ALUImmOp>(loc, "andi", /*dst=*/reg, /*src=*/reg, /*immediate=*/1);
      planned.back()->setAttr(kAtlasTagScalarResult, i32(cfgValues.lookup(arg)));
    }
  }

  void emitCopy(unsigned dst, unsigned src, bool tensor, Location loc) {
    if (dst == src)
      return;
    if (tensor) {
      emit<VPUUnaryOp>(loc, "mov", dst, src);
    } else {
      emit<ALUImmOp>(loc, "addi", dst, src, /*immediate=*/0);
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
    redirect(emit<JumpOp>(loc, "jal", /*dst=*/0, /*base=*/0, /*offset=*/0), target);
  }

  void inputHalf(unsigned dst, uint64_t index, unsigned half, Location loc) {
    uint32_t scratchWord = static_cast<uint32_t>(
        fixed().inputWord + index * kAtlasBF16TileWords + half * kAtlasRegisterWords);
    uint32_t dramByte = static_cast<uint32_t>(abi.inputAddress(index, half));
    materializeScalar(fixed().inputBaseReg, scratchWord, loc);
    materializeScalar(fixed().inputDramReg, dramByte, loc);
    emit<DMAOp>(loc, "load", /*channel=*/fixed().loadChannel,
        /*reg=*/fixed().inputBaseReg, /*dram=*/fixed().inputDramReg,
        /*size=*/fixed().halfSizeReg);
    emit<DMAWaitOp>(loc, /*channel=*/fixed().loadChannel);
    emit<VLoadOp>(loc, dst, /*base=*/fixed().inputBaseReg, /*offset=*/0, "raw");
  }

  void outputHalf(unsigned src, uint64_t index, unsigned half, Location loc) {
    uint32_t scratchWord = static_cast<uint32_t>(
        fixed().outputWord + index * kAtlasBF16TileWords + half * kAtlasRegisterWords);
    uint32_t dramByte = static_cast<uint32_t>(abi.outputAddress(index, half));
    materializeScalar(fixed().outputBaseReg, scratchWord, loc);
    emit<VStoreOp>(loc, src, /*base=*/fixed().outputBaseReg, /*offset=*/0, "raw");
    materializeScalar(fixed().outputDramReg, dramByte, loc);
    emit<DMAOp>(loc, "store", /*channel=*/fixed().storeChannel,
        /*reg=*/fixed().outputBaseReg, /*dram=*/fixed().outputDramReg,
        /*size=*/fixed().halfSizeReg);
    emit<DMAWaitOp>(loc, /*channel=*/fixed().storeChannel);
  }

  void launchDMA(Value transfer, Value dramByte, Value sizeBytes,
                 std::optional<unsigned> src, Location loc) {
    const DMATransferPlacement &placement = allocation.dma(transfer);
    // A command register takes its operand's register, or its immediate.
    auto capture = [&](unsigned reg, Value operand) {
      if (isVirtualDMAImmediate(operand))
        materializeScalar(reg, *constantAddress(operand), loc);
      else
        emitCopy(reg, scalar(operand), false, loc);
    };
    capture(placement.dramReg, dramByte);
    capture(placement.sizeReg, sizeBytes);
    materializeScalar(placement.stagingReg, placement.stagingWord, loc);
    if (src) {
      for (unsigned half = 0; half < placement.halves; ++half) {
        if (half)
          materializeScalar(placement.stagingReg,
                            placement.stagingWord + half * kAtlasRegisterWords, loc);
        emit<VStoreOp>(loc, /*src=*/*src + half, /*base=*/placement.stagingReg,
            /*offset=*/0, "raw");
      }
      if (placement.halves > 1)
        materializeScalar(placement.stagingReg, placement.stagingWord, loc);
    }
    emit<DMAOp>(loc, src ? "store" : "load", /*channel=*/placement.channel,
                /*reg=*/placement.stagingReg, /*dram=*/placement.dramReg,
                /*size=*/placement.sizeReg)
        ->setAttr(kAtlasTagDMATransfer, i32(placement.id));
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
    std::optional<VirtualDMATransfer> dma = virtualDMATransfer(&op);
    launchDMA(transfer, dma->dram->get(), dma->size->get(), std::nullopt, op.getLoc());
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
      materializeScalar(placement.stagingReg, placement.stagingWord + half * kAtlasRegisterWords, loc);
      // Its tile command was reserved at the launch.
      record(VLoadOp::create(emitter, loc, stateType(), state, load.dst + half,
                             placement.stagingReg, /*offset=*/0, "raw"),
             load.firstVLoad + half);
    }
    for (size_t n = first; n < planned.size(); ++n)
      planned[n]->setAttr(kAtlasTagCFGSource, i32(load.source));
    planned.back()->setAttr(kAtlasTagTensorResult, i32(load.result));
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

  void reorderPackReload(Operation &op) {
    auto matmul = dyn_cast<VirtualMXUMatmulOp>(op);
    if (!packReloadAfterWeight || !packReload || !matmul || packReload->block != op.getBlock() ||
        matmul.getActivation() != packReload->value || planned.size() != packReload->end ||
        !pendingLoads.count(matmul.getWeight()))
      return;
    size_t begin = packReload->begin;
    if (llvm::any_of(labelPC, [&](const auto &label) { return label.second >= begin; }))
      return;
    // Preserve the completed PACK's tags while loading an independent weight before its trailing reload.
    std::vector<Operation *> reload(planned.begin() + begin, planned.end());
    planned.erase(planned.begin() + begin, planned.end());
    packReload.reset();
    state = reload.front()->getOperand(0);
    emitter.setInsertionPoint(reload.front());
    flushLoad(matmul.getWeight(), op.getLoc());
    reload.front()->setOperand(0, state);
    state = reload.back()->getResult(0);
    emitter.setInsertionPointToEnd(emitted->getBody());
    planned.insert(planned.end(), reload.begin(), reload.end());
  }

  void preloadPackWeight(Operation &op) {
    auto pack = dyn_cast<VirtualPackFP8Op>(op);
    if (!packWeightBeforePack || !pack)
      return;
    auto matmul = dyn_cast_or_null<VirtualMXUMatmulOp>(op.getNextNode());
    if (!matmul || matmul.getActivation() != pack.getResult())
      return;
    auto load = pendingLoads.find(matmul.getWeight());
    if (load == pendingLoads.end())
      return;
    unsigned weight = load->second.dst, source = tile(pack.getSrc());
    // The original allocation did not keep the weight live across PACK.
    if (weight == source || weight == source + 1 || weight == fp8(pack.getResult()))
      return;
    preloadedWeights.push_back(load->second.placement->id);
    flushLoad(matmul.getWeight(), op.getLoc());
  }

  LogicalResult lowerPack(VirtualPackFP8Op pack) {
    // Only the E8M0 scale codes observed on the selected RTL are admitted.
    if (pack.getScaleCode() < 122 || pack.getScaleCode() > 128)
      return pack.emitOpError(
          "virtual-to-machine pack admits observed E8M0 scale codes 122..128 only");
    Location loc = pack.getLoc();
    emit<ScalarLoadOp>(loc, "seli", /*dst=*/fixed().scaleReg, /*base=*/0,
        /*offset=*/pack.getScaleCode());
    emit<VPUPackOp>(loc, "bf16_to_fp8", /*dst=*/fp8(pack.getResult()),
        /*src=*/tile(pack.getSrc()), /*scale_reg=*/fixed().scaleReg);

    // The selected PACK joins 64 physical BF16 rows of 16 lanes. The MXU
    // expects 32 logical rows of 32 FP8 lanes. Stage the raw packed result
    // in the free upper half of input VMEM and interleave its two 512-byte
    // halves through a bounded 32-row scalar loop.
    materializeScalar(fixed().outputBaseReg, fixed().packWord, loc);
    emit<VStoreOp>(loc, /*src=*/fp8(pack.getResult()), /*base=*/fixed().outputBaseReg,
        /*offset=*/0, "raw");
    materializeScalar(fixed().packSourceRegs[0], fixed().packWord * 4, loc);
    materializeScalar(fixed().packSourceRegs[1], fixed().packWord * 4 + kAtlasRegisterBytes / 2, loc);
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
      emit<ScalarLoadOp>(loc, "lw", /*dst=*/fixed().packTemporaryRegs[copy % 2],
          /*base=*/fixed().packSourceRegs[copy % 2], /*offset=*/4 * (copy / 2));
    };
    auto copyStore = [&](unsigned copy) {
      emit<ScalarStoreOp>(loc, "sw", /*src=*/fixed().packTemporaryRegs[copy % 2],
          /*base=*/fixed().packDestinationReg,
          /*offset=*/4 * (copy / 2) + 16 * (copy % 2));
    };
    auto advance = [&](unsigned reg, int32_t step) {
      emit<ALUImmOp>(loc, "addi", /*dst=*/reg, /*src=*/reg, /*immediate=*/step);
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
    redirect(emit<BranchOp>(loc, "blt", /*lhs=*/fixed().packRowReg,
                            /*rhs=*/fixed().packRowsReg, /*offset_bytes=*/0), loop);
    size_t reloadBegin = planned.size();
    materializeScalar(fixed().inputBaseReg, fixed().packRelayoutWord, loc);
    emit<VLoadOp>(loc, /*dst=*/fp8(pack.getResult()), /*base=*/fixed().inputBaseReg, /*offset=*/0, "raw");
    packReload = PackReload{pack.getResult(), pack->getBlock(), reloadBegin, planned.size()};
    return success();
  }

  LogicalResult lowerOperation(Operation &op) {
    Location loc = op.getLoc();
    if (auto control = dyn_cast<VirtualControlOp>(op)) {
      unsigned reg = scalar(control.getValue());
      emit<ScalarLoadOp>(loc, "lw", /*dst=*/reg, /*base=*/0,
          /*offset=*/4 * (fixed().mailboxWord + control.getIndex()));
      tagTileCommand();
      if (control.getValue().getType().isInteger(1))
        emit<ALUImmOp>(loc, "andi", /*dst=*/reg, /*src=*/reg, /*immediate=*/1);
      return success();
    }
    if (auto slot = dyn_cast<VirtualSlotOp>(op)) {
      if (slot.getSlotWordAttr().getInt() < 0)
        return op.emitOpError("unallocated VMEM slot; run --spill-atlas-virtual first");
      return success();
    }
    if (isa<VirtualStartOp, VirtualScaleConstantOp>(op))
      return success();
    if (isAtlasSlotAccess(&op)) {
      auto source = atlasAccessedSlot(&op).getDefiningOp<VirtualSlotOp>();
      auto copyOp = dyn_cast<VirtualSlotCopyOp>(op);
      bool copy = bool(copyOp), save = isa<VirtualSaveOp>(op);
      auto destination = copy ? copyOp.getDst().getDefiningOp<VirtualSlotOp>() : source;
      bool scalarValue = source.getWords() == 1;
      Value value = atlasSlotValue(&op);
      unsigned reg = copy ? (scalarValue ? fixed().scalarTemporary : fixed().tensorTemporary) :
          scalarValue ? scalar(value) : isa<VirtualBF16Type>(value.getType()) ? tile(value) : fp8(value);
      auto access = [&](bool write, uint32_t word) {
        materializeScalar(fixed().dmaBaseReg, scalarValue ? word * 4 : word, loc);
        unsigned base = fixed().dmaBaseReg;
        if (scalarValue) {
          if (write)
            emit<ScalarStoreOp>(loc, "sw", /*src=*/reg, base, /*offset=*/0);
          else
            emit<ScalarLoadOp>(loc, "lw", /*dst=*/reg, base, /*offset=*/0);
          tagTileCommand();
        } else if (write) {
          emit<VStoreOp>(loc, /*src=*/reg, base, /*offset=*/0, "raw");
        } else {
          emit<VLoadOp>(loc, /*dst=*/reg, base, /*offset=*/0, "raw");
        }
      };
      for (unsigned half = 0; half < (scalarValue ? 1 : source.getWords() / kAtlasRegisterWords); ++half) {
        if (copy) {
          access(false, source.getSlotWord() + half * kAtlasRegisterWords);
          access(true, destination.getSlotWord() + half * kAtlasRegisterWords);
        } else {
          if (!scalarValue) reg = (isa<VirtualBF16Type>(value.getType()) ? tile(value) : fp8(value)) + half;
          access(save, source.getSlotWord() + half * kAtlasRegisterWords);
        }
      }
      return success();
    }
    if (isAtlasSpillStore(&op) || isAtlasSpillLoad(&op)) {
      bool store = isAtlasSpillStore(&op);
      Operation *writer = atlasSpillSaved(&op).getDefiningOp();
      uint32_t word = writer->getAttrOfType<IntegerAttr>("slot_word").getInt();
      unsigned halves = atlasSpillWords(writer) / kAtlasRegisterWords;
      Value value = atlasSpillTile(&op);
      unsigned reg = isa<VirtualBF16Type>(value.getType()) ? tile(value) : fp8(value);
      for (unsigned half = 0; half < halves; ++half) {
        materializeScalar(fixed().dmaBaseReg, word + half * kAtlasRegisterWords, loc);
        if (store)
          emit<VStoreOp>(loc, /*src=*/reg + half, /*base=*/fixed().dmaBaseReg,
              /*offset=*/0, "raw");
        else
          emit<VLoadOp>(loc, /*dst=*/reg + half, /*base=*/fixed().dmaBaseReg,
              /*offset=*/0, "raw");
      }
      return success();
    }
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
      emit<MXUPushOp>(loc, "weight_fp8", /*unit=*/weight.unit,
          /*src=*/fp8(load.getSrc()), /*slot=*/weight.slot);
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccFP8Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      emit<MXUPushOp>(loc, "acc_fp8", /*unit=*/acc.unit, /*src=*/fp8(load.getSrc()),
          /*slot=*/acc.slot);
      return success();
    }
    if (auto load = dyn_cast<VirtualMXULoadAccBF16Op>(op)) {
      MXUPlacement acc = mxu(load.getAcc());
      emit<MXUPushOp>(loc, "acc_bf16", /*unit=*/acc.unit, /*src=*/tile(load.getSrc()),
          /*slot=*/acc.slot);
      return success();
    }
    if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
      MXUPlacement weight = mxu(reset.getWeight());
      MXUPlacement acc = mxu(reset.getAcc());
      emit<MXUMatmulOp>(loc, /*unit=*/acc.unit, /*src=*/fp8(reset.getActivation()),
          /*weight_slot=*/weight.slot, /*acc_slot=*/acc.slot, /*accumulate=*/false);
      return success();
    }
    if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
      MXUPlacement weight = mxu(accumulate.getWeight());
      MXUPlacement acc = mxu(accumulate.getAcc());
      emit<MXUMatmulOp>(loc, /*unit=*/acc.unit, /*src=*/fp8(accumulate.getActivation()),
          /*weight_slot=*/weight.slot, /*acc_slot=*/acc.slot, /*accumulate=*/true);
      return success();
    }
    if (auto readout = dyn_cast<VirtualMXUReadoutBF16Op>(op)) {
      MXUPlacement acc = mxu(readout.getAcc());
      emit<MXUPopOp>(loc, "bf16", /*unit=*/acc.unit, /*dst=*/tile(readout.getValue()),
          /*slot=*/acc.slot, /*scale_reg=*/0);
      return success();
    }
    if (auto readout = dyn_cast<VirtualMXUReadoutFP8Op>(op)) {
      auto scale = readout.getScale().getDefiningOp<VirtualScaleConstantOp>();
      if (!scale)
        return readout.emitOpError(
            "FP8 readout currently requires a virtual_scale_constant");
      MXUPlacement acc = mxu(readout.getAcc());
      // Rematerialize each use: readouts and VPU pack share scratch e3.
      emit<ScalarLoadOp>(loc, "seli", /*dst=*/fixed().scaleReg, /*base=*/0,
          /*offset=*/scale.getCode());
      emit<MXUPopOp>(loc, "fp8", /*unit=*/acc.unit, /*dst=*/fp8(readout.getValue()),
          /*slot=*/acc.slot, /*scale_reg=*/fixed().scaleReg);
      return success();
    }
    if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
      unsigned unit = static_cast<unsigned>(matmul.getUnit());
      emit<MXUPushOp>(loc, "weight_fp8", unit, /*src=*/fp8(matmul.getWeight()),
          /*slot=*/fixed().mxuWeightSlot);
      emit<MXUMatmulOp>(loc, unit, /*src=*/fp8(matmul.getActivation()),
          /*weight_slot=*/fixed().mxuWeightSlot, /*acc_slot=*/fixed().mxuAccSlot,
          /*accumulate=*/false);
      emit<MXUPopOp>(loc, "bf16", unit, /*dst=*/tile(matmul.getResult()),
          /*slot=*/fixed().mxuAccSlot, /*scale_reg=*/0);
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
      emit<VPUUnaryOp>(loc, unary.getKind(), /*dst=*/tile(unary.getDst()),
          /*src=*/tile(unary.getSrc()));
      return success();
    }
    if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op)) {
      emit<VPUBinaryOp>(loc, binary.getKind(), /*dst=*/tile(binary.getDst()),
          /*lhs=*/tile(binary.getLhs()), /*rhs=*/tile(binary.getRhs()));
      return success();
    }
    if (auto reduce = dyn_cast<VirtualVPUReduceOp>(op)) {
      emit<VPUReduceOp>(loc, reduce.getKind(), /*dst=*/tile(reduce.getDst()),
          /*src=*/tile(reduce.getSrc()));
      return success();
    }
    if (auto splat = dyn_cast<VirtualVPUSplatOp>(op)) {
      emit<VLIOp>(loc, "all", /*dst=*/tile(splat.getDst()),
          /*immediate=*/splat.getBits());
      return success();
    }
    if (auto transpose = dyn_cast<VirtualXLUTransposeOp>(op)) {
      emit<XLUTransposeOp>(loc, /*dst=*/fp8(transpose.getDst()),
          /*src=*/fp8(transpose.getSrc()));
      return success();
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto value = dyn_cast<IntegerAttr>(constant.getValue());
      if (!value)
        return constant.emitOpError("only integer control constants lower to Atlas");
      if (isVirtualDMAImmediate(constant.getResult()))
        return success(); // each launch builds it (launchDMA)
      materializeScalar(scalar(constant.getResult()),
                        static_cast<uint32_t>(value.getValue().getSExtValue()), loc);
      return success();
    }
    if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      emit<ALURegOp>(loc, "add", /*dst=*/scalar(addi.getResult()),
          /*lhs=*/scalar(addi.getLhs()), /*rhs=*/scalar(addi.getRhs()));
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
      redirect(emit<BranchOp>(loc, "bne", /*lhs=*/scalar(branch.getCondition()),
                              /*rhs=*/0, /*offset_bytes=*/0),
               trueEdge);
      planned.back()->setAttr(kAtlasTagCFGBranch, i32(currentCFGBlock));
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
      emit<CSROp>(loc, "rrw", /*dst=*/0, /*source=*/fixed().haltReg, /*address=*/0xc10)
          ->setAttr("atlas.complete", boolean(true));
      emit<TrapOp>(loc, "ecall");
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
      emit<ALURegOp>(loc, kind, dst, /*lhs=*/first, /*rhs=*/second);
    };
    auto invert = [&]() {
      emit<ALUImmOp>(loc, "xori", dst, /*src=*/dst, /*immediate=*/1);
    };
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      emitReg("xor", lhs, rhs);
      if (cmp.getPredicate() == arith::CmpIPredicate::eq)
        emit<ALUImmOp>(loc, "sltiu", dst, /*src=*/dst, /*immediate=*/1);
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
    for (auto [pc, label] : redirects) {
      auto found = labelPC.find(label);
      if (found == labelPC.end() || found->second >= planned.size())
        return function.emitOpError("unresolved virtual CFG target");
      int64_t offset = 2 * (static_cast<int64_t>(found->second) -
                            static_cast<int64_t>(pc));
      auto branch = dyn_cast<BranchOp>(planned[pc]);
      int64_t bound = branch ? 4096 : 1048576;
      if (offset < -bound || offset > bound - 2)
        return function.emitOpError("virtual CFG target exceeds selected branch range");
      if (branch)
        branch.setOffsetBytesAttr(i32(offset));
      else
        cast<JumpOp>(planned[pc]).setOffsetAttr(i32(offset));
    }
    return success();
  }

  ModuleOp module;
  func::FuncOp function;
  Builder attrs;
  // The machine program, built on one state chain as lowering goes: its ops
  // in order, and its branches and jumps by index with the labels they target.
  OwningOpRef<ModuleOp> emitted;
  OpBuilder emitter;
  Value state;
  std::vector<Operation *> planned;
  SmallVector<std::pair<size_t, unsigned>> redirects;
  VirtualAllocationPlan allocation;
  bool packReloadAfterWeight;
  bool packWeightBeforePack;
  SmallVector<int32_t> preloadedWeights;
  struct PackReload {
    Value value;
    Block *block;
    size_t begin, end;
  };
  std::optional<PackReload> packReload;
  llvm::DenseMap<Value, std::optional<uint32_t>> constantAddresses;
  llvm::DenseMap<Block *, unsigned> blockLabels;
  llvm::DenseMap<unsigned, size_t> labelPC;
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
  AtlasBoundaryABI abi;
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

  Option<std::string> provider{
      *this, "provider", llvm::cl::desc("Timing provider for model selection candidates"), llvm::cl::init("")};
  Option<std::string> storeWindows{
      *this, "store-windows", llvm::cl::desc("Comma-separated free staging window ranks per DMA store (model only)"),
      llvm::cl::init("")};

  Option<std::string> mregRanks{
      *this, "mreg-ranks", llvm::cl::desc("Cyclic free MREG color ranks (model only)"), llvm::cl::init("")};
  Option<bool> packReloadAfterWeight{
      *this, "pack-reload-after-weight", llvm::cl::desc("Load a pending weight before an adjacent PACK reload (model only)"),
      llvm::cl::init(false)};
  Option<bool> packWeightBeforePack{
      *this, "pack-weight-before-pack", llvm::cl::desc("Preload a disjoint pending weight before adjacent PACK (model only)"),
      llvm::cl::init(false)};

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
    if (packReloadAfterWeight && packWeightBeforePack) {
      module.emitError("PACK weight placement options are mutually exclusive");
      return signalPassFailure();
    }
    if (packReloadAfterWeight || packWeightBeforePack) {
      auto selected = selectAtlasTimingProvider(module, provider.getValue());
      if (failed(selected))
        return signalPassFailure();
      if (selected->id != timing::kModelTimingProviderId) {
        module.emitError() << (packWeightBeforePack ? "pack-weight-before-pack" : "pack-reload-after-weight")
                           << " requires the model timing provider";
        return signalPassFailure();
      }
    }
    SmallVector<unsigned> storeWindowRanks;
    if (!storeWindows.empty()) {
      auto selected = selectAtlasTimingProvider(module, provider.getValue());
      if (failed(selected))
        return signalPassFailure();
      if (selected->id != timing::kModelTimingProviderId) {
        module.emitError("store-windows requires the model timing provider");
        return signalPassFailure();
      }
      SmallVector<StringRef> ranks;
      StringRef(storeWindows.getValue()).split(ranks, ',', -1, true);
      for (StringRef text : ranks) {
        unsigned rank;
        if (text.empty() || text.getAsInteger(10, rank)) {
          module.emitError("store-windows must contain comma-separated unsigned integers");
          return signalPassFailure();
        }
        storeWindowRanks.push_back(rank);
      }
    }
    SmallVector<unsigned> registerRanks;
    if (!mregRanks.empty()) {
      SmallVector<StringRef> ranks;
      StringRef(mregRanks.getValue()).split(ranks, ',', -1, true);
      for (StringRef text : ranks) {
        unsigned rank;
        if (text.empty() || text.getAsInteger(10, rank)) {
          module.emitError("mreg-ranks must contain comma-separated unsigned integers");
          return signalPassFailure();
        }
        registerRanks.push_back(rank);
      }
      auto selected = selectAtlasTimingProvider(module, provider.getValue());
      if (failed(selected))
        return signalPassFailure();
      if (selected->id != timing::kModelTimingProviderId) {
        module.emitError("mreg-ranks requires the model timing provider");
        return signalPassFailure();
      }
    }
    reuseAtlasImmutableWeights(function);
    VirtualPlanner planner(module, function, storeWindowRanks, registerRanks, packReloadAfterWeight, packWeightBeforePack);
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

std::unique_ptr<Pass>
mlir::atlas::createLowerAtlasVirtualToMachinePass(StringRef allocationReport, StringRef provider,
                                                                   StringRef storeWindows, StringRef mregRanks,
                                                                   bool packReloadAfterWeight, bool packWeightBeforePack) {
  auto pass = std::make_unique<LowerAtlasVirtualToMachinePass>();
  pass->allocationReport = allocationReport.str();
  pass->provider = provider.str();
  pass->storeWindows = storeWindows.str();
  pass->mregRanks = mregRanks.str();
  pass->packReloadAfterWeight = packReloadAfterWeight;
  pass->packWeightBeforePack = packWeightBeforePack;
  return pass;
}
