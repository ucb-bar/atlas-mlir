#include "Atlas/AtlasRTLVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

// Bounds explicit DELAY streams before reservation entries are allocated.
static constexpr int kMaximumIssueCycle = 1000000;

LogicalResult mlir::atlas::verifySelectedAtlasRTLTiming(ModuleOp module) {
  return module->hasAttr("atlas.rtl_evidence") ? verifyAtlasRTLTiming(module)
                                                : success();
}

LogicalResult mlir::atlas::verifyAtlasRTLTiming(ModuleOp module,
                                              ResolvedRTLProgram *resolved) {
  if (resolved)
    *resolved = {};
  SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(module, words, /*llvmBlock=*/false)))
    return failure();
  auto selected = getSelectedRTLEvidence(module);
  if (failed(selected))
    return failure();
  if (!*selected)
    return module.emitError("RTL timing verification requires selected evidence");
  if (failed(checkSelectedRTLProgramSize(module)))
    return failure();

  // Each basic block is checked from an idle state: a block may hand nothing
  // but scalar register values to its successors, so all of its work must be
  // complete by the earliest cycle a successor can issue.
  FailureOr<AtlasStream> stream = readAtlasStream(module, /*allowDelays=*/true);
  if (failed(stream) || failed(checkSelectedControlFlow(module, *stream)))
    return failure();
  TargetTiming target = (*selected)->targetTiming();
  std::vector<ResolvedRTLInstruction> issued;
  std::vector<ResolvedRTLBlock> blocks;

  for (size_t block = 0; block < stream->starts.size(); ++block) {
    const size_t begin = stream->starts[block], end = stream->blockEnd(block);
    const size_t blockFirst = issued.size();
    RegValues registers = stream->entry[block];
    ReservationTable reservations(target);
    int cycle = 0;
    int fixedEngineAvailable = 0;
    int asynchronousDone = -1;
    std::optional<Footprint> pendingDMA;
    int pendingChannel = -1;
    Operation *pendingLaunch = nullptr;
    int epoch = 0, epochOrigin = 0;
    // A DELAY before the block start can only fall through into it.
    bool previousWasDelay =
        begin > 0 && stream->instrs[begin - 1].op->opClass == OpClass::Delay;

    for (size_t index = begin; index < end; ++index) {
      Operation &operation = *stream->ops[index];
      const Instr &instruction = stream->instrs[index];
      Footprint footprint = target.resolve(instruction, registers);
      if (!footprint.error.empty())
        return operation.emitOpError(footprint.error);
      if (cycle > kMaximumIssueCycle)
        return operation.emitOpError("exceeds the bounded RTL timing cycle limit");

      const OpClass opClass = instruction.op->opClass;
      const bool vectorMemory =
          opClass == OpClass::VLoad || opClass == OpClass::VStore;
      const bool fixedEngine = vectorMemory || opClass == OpClass::Transpose ||
                               instruction.op->engine == Engine::Vpu;
      const bool terminal = opClass == OpClass::Halt;
      const bool publication = opClass == OpClass::Csr;
      const bool wait = opClass == OpClass::DmaWait;
      if (pendingDMA) {
        if (footprint.dmaAsync)
          return operation.emitOpError("requires a matching DMA wait before another transfer");
        if (terminal || publication)
          return operation.emitOpError("requires a matching DMA wait before completion publication or halt");
        EdgeKind kind;
        if (conflictsAtCompletion(*pendingDMA, footprint, kind))
          return operation.emitOpError("accesses memory retained by pending DMA; matching wait required");
      }
      if (wait && (!pendingDMA || instruction.op->channel != pendingChannel))
        return operation.emitOpError("DMA wait has no matching pending transfer");
      if (terminal && previousWasDelay)
        return operation.emitOpError(
            "can halt while DELAY is stalled; an intervening NOP is required");
      if (isControlFlow(*instruction.op) && previousWasDelay)
        return operation.emitOpError(
            "can redirect while DELAY is stalled; an intervening NOP is required");
      if (fixedEngine && cycle < fixedEngineAvailable)
        return operation.emitOpError("violates the selected serialized engine admission")
               << ": issue cycle " << cycle << ", first permitted cycle "
               << fixedEngineAvailable;
      if ((terminal || publication) && cycle <= asynchronousDone)
        return operation.emitOpError(
                   "requires all prior asynchronous writes to be complete")
               << ": issue cycle " << cycle << ", first permitted cycle "
               << asynchronousDone + 1;

      for (size_t p = blockFirst; p < issued.size(); ++p) {
        const ResolvedRTLInstruction &prior = issued[p];
        Dependence dependency = dependence(prior.instruction, prior.footprint,
                                           instruction, footprint, target);
        if (cycle - prior.cycle < dependency.distance)
          return operation.emitOpError("violates selected RTL dependence: ")
                 << dependency.reason << "; gap " << cycle - prior.cycle
                 << ", required gap " << dependency.distance;
      }
      std::string conflict = reservations.conflict(instruction, footprint, cycle);
      if (!conflict.empty())
        return operation.emitOpError("violates selected RTL reservation: ")
               << conflict;
      reservations.reserve(instruction, footprint, cycle);
      if (wait) {
        pendingDMA.reset();
        pendingChannel = -1;
        pendingLaunch = nullptr;
        reservations.extendForWait(cycle);
        ++epoch;
        epochOrigin = cycle;
      }
      issued.push_back({instruction, footprint, cycle, epoch,
                        cycle - epochOrigin, static_cast<int>(block)});
      if (footprint.dmaAsync) {
        pendingDMA = footprint;
        pendingChannel = instruction.op->channel;
        pendingLaunch = &operation;
      }
      if (footprint.doneAge > 0)
        asynchronousDone = std::max(asynchronousDone, cycle + footprint.doneAge);
      if (fixedEngine)
        fixedEngineAvailable = cycle + footprint.doneAge + 1;
      applyScalar(instruction, registers);
      previousWasDelay = opClass == OpClass::Delay;
      const int gap = naturalGap(instruction);
      if (gap > kMaximumIssueCycle - cycle)
        return operation.emitOpError("exceeds the bounded RTL timing cycle limit");
      cycle += gap;
    }

    blocks.push_back({begin, end, {stream->succs[block].begin(),
                                   stream->succs[block].end()},
                      stream->endsInHalt(block) ? -1 : cycle,
                      stream->endsInBranch(block)});
    if (stream->endsInHalt(block))
      continue;
    // `cycle` is the earliest issue of any successor: the instruction after
    // the delay slot, or the next instruction on a fall-through.
    Operation *exit = stream->ops[end - (stream->endsInBranch(block) ? 2 : 1)];
    if (pendingDMA)
      return pendingLaunch->emitOpError(
          "requires a matching DMA wait before its block ends");
    for (size_t p = blockFirst; p < issued.size(); ++p) {
      const ResolvedRTLInstruction &prior = issued[p];
      const int drained = prior.cycle + prior.footprint.doneAge + 1;
      if (drained > cycle)
        return exit->emitOpError(
                   "leaves its block before prior work completes; selected "
                   "RTL timing starts every block idle")
               << ": successor issue cycle " << cycle
               << ", first permitted cycle " << drained;
    }
  }
  if (resolved) {
    resolved->evidence = *selected;
    resolved->words.assign(words.begin(), words.end());
    resolved->instructions = std::move(issued);
    resolved->blocks = std::move(blocks);
  }
  return success();
}

namespace {
struct VerifyAtlasRTLTimingPass
    : PassWrapper<VerifyAtlasRTLTimingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasRTLTimingPass)
  StringRef getArgument() const final { return "verify-atlas-rtl-timing"; }
  StringRef getDescription() const final {
    return "Verify the final stream against the selected RTL timing";
  }
  void runOnOperation() override {
    if (failed(verifyAtlasRTLTiming(getOperation())))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerVerifyAtlasRTLTimingPass() {
  PassRegistration<VerifyAtlasRTLTimingPass>();
}
