#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <array>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

llvm::Error selectionError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// Bounds explicit DELAY streams before reservation entries are allocated.
constexpr int kMaximumIssueCycle = 1000000;
constexpr int kMaximumPermittedSearch = 4096;

// The DMA engine queues launches into eight command slots indexed by launch
// count; the RTL protects neither slot reuse nor channel reuse.
constexpr size_t kDMASlots = 8;

bool isLaunch(const Instr &in) {
  return in.op->opClass == OpClass::DmaLoad || in.op->opClass == OpClass::DmaStore;
}

// MXU commands were not swept beside a pending transfer and stay serialized
// with DMA; VLS, VPU and XLU kept their isolated events in every swept run.
bool holdsMxu(const Footprint &f) {
  return std::any_of(f.holds.begin(), f.holds.end(), [](const Hold &h) {
    return h.unit == Unit::MxuCompute || h.unit == Unit::MxuPort || h.unit == Unit::MxuAccRead ||
           h.unit == Unit::MxuAccWrite || h.unit == Unit::MxuWeightStream || h.unit == Unit::MxuAccStream;
  });
}

bool isVls(const Footprint &f) {
  return std::any_of(f.holds.begin(), f.holds.end(), [](const Hold &h) {
    return h.unit == Unit::VloadPath || h.unit == Unit::VstorePath;
  });
}

// A VLS tile may not touch a pending transfer's VMEM window at all; the queued
// transfer reads or writes it whenever it reaches the head of the queue.
bool vlsOverlapsWindow(const Footprint &dma, const Footprint &f) {
  for (const Access &window : dma.accesses) {
    if (window.res != Res::Vmem || !window.atCompletion) continue;
    for (const Access &tile : f.accesses)
      if (tile.res == Res::Vmem && tile.first < window.first + window.count && window.first < tile.first + tile.count)
        return true;
  }
  return false;
}

class RTLReservations final : public TimingReservations {
public:
  TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &f, int cycle) const override {
    if (cycle > kMaximumIssueCycle)
      return {{}, "exceeds the bounded RTL timing cycle limit"};
    std::string busy = table.conflict(in, f, cycle);
    if (busy.empty())
      return {};
    int permitted = cycle + 1;
    while (permitted - cycle < kMaximumPermittedSearch &&
           !table.conflict(in, f, permitted).empty())
      ++permitted;
    return {busy + "; issue cycle " + std::to_string(cycle) +
                ", first permitted cycle " + std::to_string(permitted),
            {}};
  }
  std::string reserve(const Instr &in, const Footprint &f, int cycle) override {
    table.reserve(in, f, cycle);
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    table.extendForWait(cycle);
    return {};
  }

private:
  ReservationTable table{ReservationPolicy::Exclusive};
};

// Every block is verified from an idle state, so every path must reach ECALL
// through blocks ending in a halt, a delay-slotted redirect to an instruction,
// or a fall-through. Transfers pend one per channel and at most eight, each
// waited on its own channel within its block.
std::string validateScope(const AtlasStream &s) {
  if (s.ops.size() > RTLEvidence::maximumProgramWords())
    return "program exceeds the 32768-word instruction memory";
  size_t blocks = s.starts.size();
  if (blocks == 0)
    return "selected RTL timing requires a terminal instruction";
  std::vector<bool> reached(blocks, false);
  std::vector<size_t> work = {0};
  reached[0] = true;
  while (!work.empty()) {
    size_t b = work.back();
    work.pop_back();
    for (size_t next : s.succs[b])
      if (!reached[next]) {
        reached[next] = true;
        work.push_back(next);
      }
  }
  auto at = [&](size_t i, const std::string &why) {
    return "word " + std::to_string(i) + " (" + s.ops[i]->getName().getStringRef().str() + ") " + why;
  };
  for (size_t b = 0; b < blocks; ++b) {
    size_t begin = s.starts[b], end = s.blockEnd(b);
    if (!reached[b])
      return at(begin, b > 0 && s.endsInHalt(b - 1) && !s.endsInBranch(b - 1)
                           ? "follows the terminal instruction"
                           : "is unreachable; selected RTL timing verifies only reachable blocks");
    if (s.endsInBranch(b)) {
      if (!s.targetOf.lookup(s.ops[end - 2]))
        return at(end - 2, "targets the stream end; selected RTL timing requires every path to reach ECALL");
      if (s.instrs[end - 1].op->opClass != OpClass::Alu)
        return at(end - 1, "is a delay-slot instruction; selected RTL timing admits only ADDI or LUI in a delay slot");
    } else if (!s.endsInHalt(b) && end == s.ops.size()) {
      return at(end - 1, "ends the stream without ECALL; selected RTL timing requires a terminal instruction");
    }
    struct Launch { size_t index; int channel; bool waited; };
    std::vector<Launch> launches;
    std::array<int, 8> pendingOn;
    pendingOn.fill(-1);
    for (size_t i = begin; i < end; ++i) {
      const Instr &in = s.instrs[i];
      int channel = in.op->channel;
      if (isLaunch(in)) {
        if (pendingOn[channel] >= 0)
          return at(i, "launches on channel " + std::to_string(channel) + " while its transfer at word " +
                           std::to_string(launches[pendingOn[channel]].index) +
                           " has not been waited; the RTL does not protect channel reuse, so add atlas.dma_wait first");
        if (launches.size() >= kDMASlots && !launches[launches.size() - kDMASlots].waited)
          return at(i, "is launch " + std::to_string(launches.size()) + " of its block while launch " +
                           std::to_string(launches.size() - kDMASlots) + " (word " +
                           std::to_string(launches[launches.size() - kDMASlots].index) +
                           ") has not been waited; the DMA engine has eight command slots");
        pendingOn[channel] = static_cast<int>(launches.size());
        launches.push_back({i, channel, false});
      } else if (in.op->opClass == OpClass::DmaWait) {
        if (pendingOn[channel] < 0)
          return at(i, "has no pending transfer to wait for on channel " + std::to_string(channel));
        launches[pendingOn[channel]].waited = true;
        pendingOn[channel] = -1;
      }
    }
    for (const Launch &launch : launches)
      if (!launch.waited)
        return at(launch.index, "launches a DMA transfer that is still pending when its block ends; selected RTL timing requires atlas.dma_wait on channel " +
                                    std::to_string(launch.channel) + " within the block");
  }
  return {};
}

} // namespace

TimingProvider mlir::atlas::rtlTimingProvider(std::shared_ptr<const RTLEvidence> evidence) {
  TimingProvider p;
  p.id = kAtlasRTLTimingProviderId.str();
  p.redirectWaitsForDelay = false;
  p.concreteEntryContexts = true;
  p.validateScope = validateScope;
  p.footprint = [evidence](const Instr &in, const RegValues &regs) {
    return evidence->resolve(in, regs);
  };
  p.dependence = [](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    Dependence d = dataDependence(a, fa, b, fb);
    if (isLaunch(b) && holdsMxu(fa) && fa.doneAge + 1 > d.distance)
      d = {fa.doneAge + 1, EdgeKind::Order, "selected DMA launch waits for MXU work to drain"};
    // Launches keep queue order, so a wait kept ahead of every later launch
    // keeps each launch's command slot free, as the scope requires.
    if (isLaunch(b) && a.op->opClass == OpClass::DmaWait && d.distance < 1)
      d = {1, EdgeKind::Order, "selected DMA launch stays behind an earlier wait (eight command slots)"};
    return TimingRuleResult<Dependence>{d, {}};
  };
  // Pending windows stay disjoint from each other and from VLS tiles; DRAM
  // spans may be shared only by reads; MXU commands wait for every transfer.
  p.dmaConflict = [](const Footprint &dma, const Footprint &f) {
    DMACompletionConflict result;
    result.conflict = conflictsAtCompletion(dma, f, result.kind);
    if (!result.conflict && (holdsMxu(f) || (isVls(f) && vlsOverlapsWindow(dma, f))))
      result = {true, EdgeKind::Order};
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  p.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  p.createReservations = [] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<RTLReservations>();
    return result;
  };
  return p;
}

TimingRuleResult<TimingProvider> mlir::atlas::rtlTimingProviderFactory(ModuleOp module) {
  if (!module)
    return {{}, "the RTL timing provider requires a module carrying selected atlas.rtl_evidence"};
  auto evidence = loadSelectedRTLEvidence(module);
  if (!evidence)
    return {{}, llvm::toString(evidence.takeError())};
  if (!*evidence)
    return {{}, "the RTL timing provider requires selected evidence (--select-atlas-rtl-evidence)"};
  return {rtlTimingProvider(std::move(*evidence)), {}};
}

llvm::Expected<std::shared_ptr<RTLEvidence>>
mlir::atlas::loadSelectedRTLEvidence(ModuleOp module) {
  Attribute raw = module->getAttr(kAtlasRTLEvidence);
  if (!raw)
    return std::shared_ptr<RTLEvidence>();
  auto attr = dyn_cast<DictionaryAttr>(raw);
  if (!attr)
    return selectionError("atlas.rtl_evidence must be a selection dictionary");
  auto path = attr.getAs<StringAttr>("op_timing");
  auto hash = attr.getAs<StringAttr>("op_timing_sha256");
  auto hardware = attr.getAs<StringAttr>("hardware_ir_sha256");
  auto provider = attr.getAs<StringAttr>("provider");
  if (!path || !hash || !hardware || !provider)
    return selectionError("incomplete RTL evidence selection");
  bool dma = false;
  if (Attribute policy = attr.get("dma")) {
    auto text = dyn_cast<StringAttr>(policy);
    if (!text || text.getValue() != "wait")
      return selectionError("unsupported DMA policy; only dma=wait is defined");
    dma = true;
  }
  if (provider.getValue() != kAtlasRTLTimingProviderId)
    return selectionError("unsupported RTL evidence provider: " + provider.getValue().str());
  auto loaded = loadRTLTimingFacts(path.getValue(), hash.getValue(), dma);
  if (!loaded)
    return selectionError("RTL evidence selection failed: " + llvm::toString(loaded.takeError()));
  if (loaded->hardwareIRSha256() != hardware.getValue())
    return selectionError("selected facts describe a different hardware IR than recorded");
  return std::make_shared<RTLEvidence>(std::move(*loaded));
}

FailureOr<std::shared_ptr<RTLEvidence>>
mlir::atlas::getSelectedRTLEvidence(ModuleOp module) {
  auto evidence = loadSelectedRTLEvidence(module);
  if (!evidence)
    return module.emitError(llvm::toString(evidence.takeError()));
  return std::move(*evidence);
}

namespace {
struct SelectAtlasRTLEvidencePass
    : PassWrapper<SelectAtlasRTLEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasRTLEvidencePass)
  SelectAtlasRTLEvidencePass() = default;
  SelectAtlasRTLEvidencePass(const SelectAtlasRTLEvidencePass &other)
      : PassWrapper(other) {}
  Option<std::string> opTiming{*this, "op-timing",
      llvm::cl::desc("RTL-computed operation timing facts (merlin.op_timing.v1)")};
  Option<std::string> opTimingHash{*this, "op-timing-sha256",
      llvm::cl::desc("Expected SHA-256 of the facts file")};
  Option<std::string> dmaPolicy{*this, "dma",
      llvm::cl::desc("Optional DMA policy: wait")};
  StringRef getArgument() const final { return "select-atlas-rtl-evidence"; }
  StringRef getDescription() const final {
    return "Select SHA-256-identified RTL-computed timing facts for the RTL timing provider";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!dmaPolicy.empty() && dmaPolicy != "wait") {
      module.emitError("unsupported DMA policy; only dma=wait is defined");
      signalPassFailure();
      return;
    }
    auto loaded = loadRTLTimingFacts(opTiming, opTimingHash, dmaPolicy == "wait");
    if (!loaded) {
      module.emitError("RTL evidence selection failed: ") << llvm::toString(loaded.takeError());
      signalPassFailure();
      return;
    }
    if (auto retained = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
        retained && retained.getValue() != kAtlasRTLTimingProviderId) {
      module.emitError("selected RTL evidence contradicts the retained timing provider ")
          << retained.getValue() << "; time the stream again after selection or drop the selection";
      signalPassFailure();
      return;
    }
    Builder builder(module.getContext());
    NamedAttrList fields;
    fields.set("op_timing", builder.getStringAttr(opTiming));
    fields.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    fields.set("hardware_ir_sha256", builder.getStringAttr(loaded->hardwareIRSha256()));
    if (loaded->dmaWait())
      fields.set("dma", builder.getStringAttr("wait"));
    fields.set("provider", builder.getStringAttr(kAtlasRTLTimingProviderId));
    module->setAttr(kAtlasRTLEvidence, fields.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasRTLEvidencePass() {
  PassRegistration<SelectAtlasRTLEvidencePass>();
}
