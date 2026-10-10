#include "Atlas/AtlasRTLVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

// Keep explicit DELAY streams bounded before allocating reservation entries.
constexpr int kMaximumIssueCycle = 1000000;

} // namespace

LogicalResult mlir::atlas::verifySelectedAtlasRTLTiming(ModuleOp module) {
  if (module->hasAttr("atlas.rtl_evidence") || module->hasAttr("atlas.rtl_qualification"))
    return verifyAtlasRTLTiming(module);
  return success();
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
  if (words.size() > RTLEvidence::maximumProgramWords())
    return module.emitError("selected RTL program exceeds the 32768-word instruction memory");

  RegValues registers = unknownRegs();
  TargetTiming target = (*selected)->targetTiming();
  ReservationTable reservations(target);
  std::vector<ResolvedRTLInstruction> issued;
  int cycle = 0;
  int fixedEngineAvailable = 0;
  int asynchronousDone = -1;
  std::optional<Footprint> pendingDMA;
  int pendingChannel = -1;
  int epoch = 0, epochOrigin = 0;
  bool halted = false;
  bool previousWasDelay = false;

  for (Operation &operation : module.getBody()->getOperations()) {
    if (isa<StartOp>(operation))
      continue;
    if (halted)
      return operation.emitOpError("follows the terminal instruction");
    auto decoded = atlasInstruction(&operation);
    if (failed(decoded))
      return failure();
    const Instr &instruction = *decoded;
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
    if (fixedEngine && cycle < fixedEngineAvailable)
      return operation.emitOpError("violates the selected serialized engine admission")
             << ": issue cycle " << cycle << ", first permitted cycle "
             << fixedEngineAvailable;
    if ((terminal || publication) && cycle <= asynchronousDone)
      return operation.emitOpError(
                 "requires all prior asynchronous writes to be complete")
             << ": issue cycle " << cycle << ", first permitted cycle "
             << asynchronousDone + 1;

    for (const ResolvedRTLInstruction &prior : issued) {
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
      reservations.extendForWait(cycle);
      ++epoch;
      epochOrigin = cycle;
    }
    issued.push_back({instruction, footprint, cycle, epoch, cycle - epochOrigin});
    if (footprint.dmaAsync) {
      pendingDMA = footprint;
      pendingChannel = instruction.op->channel;
    }
    if (fixedEngine) {
      asynchronousDone = std::max(asynchronousDone, cycle + footprint.doneAge);
      fixedEngineAvailable = cycle + footprint.doneAge + 1;
    }
    applyScalar(instruction, registers);
    previousWasDelay = opClass == OpClass::Delay;
    halted = terminal;
    const int gap = naturalGap(instruction);
    if (gap > kMaximumIssueCycle - cycle)
      return operation.emitOpError("exceeds the bounded RTL timing cycle limit");
    cycle += gap;
  }
  if (!halted)
    return module.emitError("selected RTL timing stream requires a terminal instruction");
  if (resolved) {
    resolved->evidence = *selected;
    resolved->words.assign(words.begin(), words.end());
    resolved->instructions = std::move(issued);
  }
  return success();
}

namespace {
struct VerifyAtlasRTLTimingPass
    : PassWrapper<VerifyAtlasRTLTimingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasRTLTimingPass)

  StringRef getArgument() const final { return "verify-atlas-rtl-timing"; }
  StringRef getDescription() const final {
    return "Verify final straight-line machine timing against selected bounded RTL evidence";
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
