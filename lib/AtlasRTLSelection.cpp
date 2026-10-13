#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <array>
#include <optional>
#include <set>

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

int holdEnd(const Footprint &f, Unit unit) {
  int end = -1;
  for (const Hold &h : f.holds)
    if (h.unit == unit) end = std::max(end, h.to);
  return end;
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

bool isEngineHold(const Hold &h) { return h.unit == Unit::Vpu || h.unit == Unit::Xlu; }

Footprint holdsWhere(const Footprint &f, bool engine) {
  Footprint out = engine ? Footprint{} : f;
  out.holds.clear();
  for (const Hold &h : f.holds)
    if (isEngineHold(h) == engine) out.holds.push_back(h);
  return out;
}

// Physical MREG banks (register mod 32) a command names.
std::set<int> mregBanks(const Footprint &f) {
  std::set<int> banks;
  for (const std::vector<int> *regs : {&f.mregReads, &f.mregWrites})
    for (int reg : *regs) banks.insert(reg & 31);
  return banks;
}

Footprint withoutHold(const Footprint &f, Unit unit) {
  Footprint out = f;
  out.holds.erase(std::remove_if(out.holds.begin(), out.holds.end(), [&](const Hold &h) { return h.unit == unit; }),
                  out.holds.end());
  return out;
}

// Selected pair facts admit commands of one engine at once: a later command
// may issue while an earlier one is live only where a pair fact for the two
// (in issue order) covers that gap and their operands satisfy its relation,
// and at most the engine's observed live capacity run together. A VPU command
// is live from issue until its next_issue_age; it keeps its VPU and XLU holds
// in `engines`, which excludes XLU work as one table would. An MXU command on
// a unit the facts cover is live from issue through its busy end (its last
// event, MREG reads through their response; never shorter than the facts'
// last-event window) and replaces its own-unit hold, so without a fact the
// next command on that MXU waits exactly as the hold made it wait.
class PairSlots {
public:
  explicit PairSlots(const RTLEvidence &evidence) : evidence(evidence), facts(*evidence.pairFacts()) {}

  struct Command {
    const OpInfo *op = nullptr;
    std::string block, engine, holder;
    std::set<int> banks;
    std::optional<int> acc, slot;
    int issue = 0, window = 0;
  };

  std::optional<Command> command(const Instr &in, const Footprint &f, int cycle) const {
    if (!in.op) return std::nullopt;
    if (in.op->engine == Engine::Vpu) {
      const char *name = RTLEvidence::vpuBlockName(in.op->name);
      const OpTimingBlock *b = name ? evidence.block(name) : nullptr;
      if (!b || !b->nextIssueAge) return std::nullopt;
      return Command{in.op, name, b->engine, "VPU slot", mregBanks(f), std::nullopt, std::nullopt, cycle,
                     *b->nextIssueAge};
    }
    std::string name = RTLEvidence::mxuBlockName(*in.op);
    const OpTimingBlock *b = name.empty() ? nullptr : evidence.block(name);
    auto window = facts.window.find(b ? b->engine : "");
    int busy = holdEnd(f, Unit::MxuCompute);
    if (!b || window == facts.window.end() || window->second != "last_event_age" || busy < 0)
      return std::nullopt;
    Command command{in.op, name, b->engine, "MXU in-flight matmuls " + std::to_string(in.op->mxu),
                    mregBanks(f), std::nullopt, std::nullopt, cycle, busy + 1};
    switch (in.op->opClass) {
    case OpClass::WeightPush: command.slot = in.rd; break;
    case OpClass::AccPushFp8:
    case OpClass::AccPushBf16: command.acc = in.rd; break;
    case OpClass::PopFp8:
    case OpClass::PopBf16: command.acc = in.rs2; break;
    default: command.acc = in.rd; command.slot = in.rs2; break; // matmul
    }
    return command;
  }

  std::string conflict(const Command &next) const {
    std::vector<const Command *> overlapping;
    for (const Command &live : commands) {
      if (live.engine != next.engine) continue;
      const Command &first = live.issue <= next.issue ? live : next;
      const Command &second = live.issue <= next.issue ? next : live;
      int gap = second.issue - first.issue;
      if (gap >= first.window) continue;
      overlapping.push_back(&live);
      if (!admits(first, second, gap))
        return live.holder + " held by " + live.op->name + " issued at cycle " + std::to_string(live.issue) +
               " until cycle " + std::to_string(live.issue + live.window) + "; no pair fact admits " +
               second.op->name + " " + std::to_string(gap) + " cycles after " + first.op->name +
               " with these operands";
    }
    auto limit = facts.capacity.find(next.engine);
    int capacity = limit == facts.capacity.end() ? 1 : limit->second;
    for (int start = next.issue; start < next.issue + next.window; ++start) {
      int live = 1;
      for (const Command *other : overlapping)
        live += other->issue <= start && start < other->issue + other->window;
      if (live > capacity)
        return "all " + std::to_string(capacity) + " " + (next.op->engine == Engine::Vpu ? "VPU" : next.engine) +
               " slots busy at cycle " + std::to_string(start) + " (live capacity from the pair facts)";
    }
    return {};
  }

  void reserve(Command command) { commands.push_back(std::move(command)); }

private:
  // A pair fact covers gaps from its stable gap through the end of its sweep,
  // which must reach the end of the first command's live window.
  bool admits(const Command &first, const Command &second, int gap) const {
    auto found = facts.pairs.find({first.block, second.block});
    if (found == facts.pairs.end() || gap < 1) return false;
    for (const OpPairFact &fact : found->second)
      if (fact.stableGap <= gap && fact.lastGap + 1 >= first.window && satisfied(fact, first, second))
        return true;
    return false;
  }
  // Every relation must be stated in a known vocabulary and hold, and it must
  // constrain every operand kind both commands name: physical MREG banks
  // always, the accumulator and the weight slot where both use one.
  static bool satisfied(const OpPairFact &fact, const Command &first, const Command &second) {
    bool banks = false, acc = false, slot = false;
    auto same = [](const std::optional<int> &a, const std::optional<int> &b, const std::string &value) {
      return a && b && (value == "same" || value == "other") && ((*a == *b) == (value == "same"));
    };
    for (const auto &[name, value] : fact.relation) {
      if (name == "mreg_banks" && value == "disjoint") {
        for (int bank : first.banks)
          if (second.banks.count(bank)) return false;
        banks = true;
      } else if (name == "accumulator" && same(first.acc, second.acc, value)) {
        acc = true;
      } else if (name == "weight_slot" && same(first.slot, second.slot, value)) {
        slot = true;
      } else {
        return false;
      }
    }
    return banks && (acc || !(first.acc && second.acc)) && (slot || !(first.slot && second.slot));
  }

  const RTLEvidence &evidence;
  const RTLPairFacts &facts;
  std::vector<Command> commands;
};

class RTLReservations final : public TimingReservations {
public:
  explicit RTLReservations(std::shared_ptr<const RTLEvidence> evidence) : evidence(std::move(evidence)) {
    if (this->evidence && this->evidence->pairFacts()) slots.emplace(*this->evidence);
  }
  TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &f, int cycle) const override {
    if (cycle > kMaximumIssueCycle)
      return {{}, "exceeds the bounded RTL timing cycle limit"};
    std::string busy = busyAt(in, f, cycle);
    if (busy.empty())
      return {};
    int permitted = cycle + 1;
    while (permitted - cycle < kMaximumPermittedSearch && !busyAt(in, f, permitted).empty())
      ++permitted;
    return {busy + "; issue cycle " + std::to_string(cycle) +
                ", first permitted cycle " + std::to_string(permitted),
            {}};
  }
  std::string reserve(const Instr &in, const Footprint &f, int cycle) override {
    if (auto command = slots ? slots->command(in, f, cycle) : std::nullopt) {
      if (in.op->engine == Engine::Vpu) {
        table.reserve(in, holdsWhere(f, false), cycle);
        engines.reserve(in, holdsWhere(f, true), cycle);
      } else {
        table.reserve(in, withoutHold(f, Unit::MxuCompute), cycle);
      }
      slots->reserve(std::move(*command));
      return {};
    }
    table.reserve(in, f, cycle);
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    table.extendForWait(cycle);
    engines.extendForWait(cycle);
    return {};
  }

private:
  // Without pair facts every unit, the VPU and each MXU included, is exclusive
  // in `table`. Slot commands keep no own-engine hold there, and a later
  // stall only widens the gaps the facts admit.
  std::string busyAt(const Instr &in, const Footprint &f, int cycle) const {
    std::string busy = table.conflict(in, f, cycle);
    if (!busy.empty() || !slots)
      return busy;
    if (auto command = slots->command(in, f, cycle))
      return slots->conflict(*command);
    return engines.conflict(in, holdsWhere(f, true), cycle);
  }

  std::shared_ptr<const RTLEvidence> evidence;
  ReservationTable table{ReservationPolicy::Exclusive};
  ReservationTable engines{ReservationPolicy::Exclusive};
  std::optional<PairSlots> slots;
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
    // The LSU reports a VSTORE's source register as read until the VSTORE
    // drains; every writer but a VLOAD asserts on a pending read
    // (MregBankTracker, pairs/core_mxu_cross.yaml).
    if (a.op->opClass == OpClass::VStore && b.op->opClass != OpClass::VLoad)
      for (int r : fa.mregReads)
        if (std::find(fb.mregWrites.begin(), fb.mregWrites.end(), r) != fb.mregWrites.end() &&
            holdEnd(fa, Unit::VstorePath) + 1 > d.distance)
          d = {holdEnd(fa, Unit::VstorePath) + 1, EdgeKind::WAR,
               "m" + std::to_string(r) + " reported read until the VSTORE drains at age " +
                   std::to_string(holdEnd(fa, Unit::VstorePath))};
    // Launches keep queue order, so a wait kept ahead of every later launch
    // keeps each launch's command slot free, as the scope requires.
    if (isLaunch(b) && a.op->opClass == OpClass::DmaWait && d.distance < 1)
      d = {1, EdgeKind::Order, "selected DMA launch stays behind an earlier wait (eight command slots)"};
    return TimingRuleResult<Dependence>{d, {}};
  };
  // Pending windows stay disjoint from each other and from VLS tiles; DRAM
  // spans may be shared only by reads. VLS, VPU, XLU and MXU commands kept
  // their isolated events beside a pending transfer in every swept run.
  p.dmaConflict = [](const Footprint &dma, const Footprint &f) {
    DMACompletionConflict result;
    result.conflict = conflictsAtCompletion(dma, f, result.kind);
    if (!result.conflict && isVls(f) && vlsOverlapsWindow(dma, f))
      result = {true, EdgeKind::Order};
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  p.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  p.createReservations = [evidence] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<RTLReservations>(evidence);
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
  auto pairs = attr.getAs<StringAttr>("op_pairs");
  auto pairsHash = attr.getAs<StringAttr>("op_pairs_sha256");
  if ((attr.get("op_pairs") || attr.get("op_pairs_sha256")) && !(pairs && pairsHash))
    return selectionError("incomplete RTL pair facts selection");
  if (pairs)
    if (llvm::Error error = loaded->selectPairFacts(pairs.getValue(), pairsHash.getValue()))
      return selectionError("RTL evidence selection failed: " + llvm::toString(std::move(error)));
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
  Option<std::string> opPairs{*this, "op-pairs",
      llvm::cl::desc("Optional pair facts (merlin.op_pairs.v1) derived from the same facts")};
  Option<std::string> opPairsHash{*this, "op-pairs-sha256",
      llvm::cl::desc("Expected SHA-256 of the pair facts file")};
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
    if (opPairs.empty() != opPairsHash.empty()) {
      module.emitError("op-pairs and op-pairs-sha256 are selected together");
      signalPassFailure();
      return;
    }
    if (!opPairs.empty())
      if (llvm::Error error = loaded->selectPairFacts(opPairs, opPairsHash)) {
        module.emitError("RTL evidence selection failed: ") << llvm::toString(std::move(error));
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
    if (!opPairs.empty()) {
      fields.set("op_pairs", builder.getStringAttr(opPairs));
      fields.set("op_pairs_sha256", builder.getStringAttr(opPairsHash));
    }
    fields.set("provider", builder.getStringAttr(kAtlasRTLTimingProviderId));
    module->setAttr(kAtlasRTLEvidence, fields.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasRTLEvidencePass() {
  PassRegistration<SelectAtlasRTLEvidencePass>();
}
