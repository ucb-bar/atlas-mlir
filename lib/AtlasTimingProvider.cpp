#include "Atlas/AtlasTimingProvider.h"
#include "Atlas/AtlasModelTables.h"
#include "llvm/ADT/StringMap.h"
#include <cassert>
#include <limits>
#include <mutex>

using namespace mlir::atlas::timing;

namespace {
class RuleReservations final : public TimingReservations {
public:
  explicit RuleReservations(std::shared_ptr<const TimingTables> tables)
      : table(std::move(tables)) {}
  TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &f, int cycle) const override {
    return {table.conflict(in, f, cycle), {}};
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
  ReservationTable table;
};
} // namespace

std::unique_ptr<TimingReservations>
mlir::atlas::timing::createTimingReservations(
    std::shared_ptr<const TimingTables> tables) {
  return std::make_unique<RuleReservations>(std::move(tables));
}


namespace {
std::pair<Instr, Instr> representativeProbes(const OpInfo *op,
                                           const TimingTables *tables) {
  const bool mxu = op->engine == Engine::Mxu0 || op->engine == Engine::Mxu1;
  if (!tables) {
    if (mxu)
      return {{op, 0, 0, 0}, {op, 1, 2, 1}};
    return {{op, 0, 2, 4}, {op, 6, 8, 10}};
  }
  const int stride = tables->bf16RegistersPerTile;
  Instr first{op}, second{op};
  if (op->engine == Engine::Scalar)
    second.rd = 3 * stride;
  // Field positions retain the compiler's disjoint probe placement. The span
  // comes from the model's register geometry; ISA metadata selects live fields.
  auto place = [&](int Instr::*field, int index,
                   std::initializer_list<const char *> tokens) {
    for (const char *token : tokens) {
      if (!hasOperand(*op, token)) continue;
      if (mxu) {
        second.*field = token[0] == 'm' ? stride : token[0] == 'e' ? 0 : 1;
      } else {
        first.*field = index * stride;
        second.*field = (index + 3) * stride;
      }
      return;
    }
  };
  place(&Instr::rd, 0, {"md", "wd", "ad", "ed", "xd"});
  place(&Instr::rs1, 1, {"m1", "e1", "x1", "@"});
  place(&Instr::rs2, 2, {"m2", "w2", "a2", "x2"});
  return {first, second};
}
} // namespace

TimingRuleResult<MachineOpTiming> mlir::atlas::timing::machineTimingOf(
    const TimingProvider &p, const std::string &name) {
  auto error = [&](const std::string &why) {
    return TimingRuleResult<MachineOpTiming>{{}, p.diagnostic(why)};
  };
  const OpInfo *op = findOp(name);
  if (!op) return error("unknown machine operation");
  auto [first, second] = representativeProbes(op, p.model.get());
  Footprint a = p.footprint(first, unknownRegs());
  Footprint b = p.footprint(second, unknownRegs());
  for (const Footprint *f : {&a, &b}) {
    if (!f->error.empty()) return {{}, f->error};
    if (f->doneAge < 0 || f->doneAge == std::numeric_limits<int>::max())
      return error("completion age exceeds positive cycle domain");
  }
  auto gap = p.issueGap(first);
  if (!gap.error.empty()) return {{}, gap.error};
  if (gap.value <= 0) return error("issue gap must be positive");
  auto table = p.createReservations();
  if (!table.error.empty()) return {{}, table.error};
  if (!table.value) return error("timing provider returned no reservation state");
  auto isolated = table.value->conflict(first, a, 0);
  if (!isolated.error.empty()) return {{}, isolated.error};
  if (!isolated.value.empty()) return error("representative instruction can never issue: " + isolated.value);
  std::string why = table.value->reserve(first, a, 0);
  if (!why.empty()) return {{}, why};
  // Compiler policy: bound the same-unit issue interval search.
  constexpr int kMaximumInterval = 100000;
  for (int interval = gap.value; interval <= kMaximumInterval; ++interval) {
    auto conflict = table.value->conflict(second, b, interval);
    if (!conflict.error.empty()) return {{}, conflict.error};
    if (conflict.value.empty())
      // Issue-edge-zero ages include the final event/release edge. The
      // source-bound frontend advances one edge before the work is drained.
      return {{a.doneAge + 1, interval}, {}};
  }
  return error("same-unit issue interval exceeds bounded timing search");
}

std::string mlir::atlas::timing::fillMachineTiming(TimingProvider &p) {
  std::string missing = validateTimingProvider(p);
  if (!missing.empty()) return missing;
  std::map<std::string, MachineOpTiming> timings;
  auto add = [&](const std::string &name) {
    auto result = machineTimingOf(p, name);
    if (!result.error.empty()) return name + ": " + result.error;
    timings.emplace(name, result.value);
    return std::string{};
  };
  for (const OpInfo &op : machineOperations()) {
    if (op.engine == Engine::Dma ||
        (op.engine == Engine::Scalar && op.name != "addi" &&
         op.name != "lui" && op.name != "seli") ||
        (op.opClass == OpClass::ScalarLoad && op.name != "lw") ||
        (op.opClass == OpClass::ScalarStore && op.name != "sw"))
      continue;
    if (auto error = add(op.name); !error.empty()) return error;
  }
  p.machineTiming = std::move(timings);
  return {};
}

std::string TimingProvider::diagnostic(llvm::StringRef error) const {
  return model ? modelTimingDiagnostic(error) : error.str();
}

std::string mlir::atlas::timing::validateTimingProvider(
    const TimingProvider &p) {
  auto error = [&](const char *why) { return p.diagnostic(why); };
  if (p.id.empty()) return error("timing provider requires an explicit policy id");
  if (!p.validateScope) return error("timing provider lacks program scope coverage");
  if (!p.footprint) return error("timing provider lacks footprint rules");
  if (!p.dependence) return error("timing provider lacks dependence rules");
  if (!p.dmaConflict) return error("timing provider lacks DMA completion conflict rules");
  if (!p.issueGap) return error("timing provider lacks issue gap rules");
  if (!p.createReservations) return error("timing provider lacks reservation policy");
  return {};
}

TimingProvider mlir::atlas::timing::builtinTimingProvider() {
  TimingProvider p;
  p.id = kBuiltinTimingProviderId.str();
  p.fillPendingDelaySlots = true;
  p.scalarSlotProposalCycles = 8;
  p.validateScope = [](const mlir::atlas::AtlasStream &) { return std::string{}; };
  p.footprint = [](const Instr &in, const RegValues &regs) { return footprintOf(in, regs); };
  p.dependence = [](const Instr &a, const Footprint &fa, const Instr &b,
                    const Footprint &fb) {
    return TimingRuleResult<Dependence>{mlir::atlas::timing::dependence(a, fa, b, fb), {}};
  };
  p.dmaConflict = [](const Footprint &a, const Footprint &b) {
    DMACompletionConflict result;
    result.conflict = conflictsAtCompletion(a, b, result.kind);
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  p.issueGap = [](const Instr &in) {
    return TimingRuleResult<int>{naturalGap(in), {}};
  };
  p.createReservations = [] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = createTimingReservations();
    return result;
  };
  p.dmaEstimate = [](int64_t bytes, bool store) {
    return TimingRuleResult<int>{dmaTransferCycles(bytes, store), {}};
  };
  p.dmaProposal = [](int latency, int issue, std::optional<int> previousEnd) {
    // Compiler policy: serialize transfer estimates and issue the matching
    // wait two cycles after the estimated queue end.
    int64_t end = std::max(int64_t(issue) + latency - 1,
                          int64_t(previousEnd.value_or(0)) + latency);
    if (latency < 0 || issue < 0 || end < 0 || end > std::numeric_limits<int>::max() - 2)
      return TimingRuleResult<DMAProposal>{{}, "DMA queue estimate exceeds cycle domain"};
    return TimingRuleResult<DMAProposal>{{int(end), int(end + 2)}, {}};
  };
  static const auto timings = [&] {
    std::string error = fillMachineTiming(p);
    assert(error.empty() && "default machine timing must be complete");
    return std::move(p.machineTiming);
  }();
  p.machineTiming = timings;
  return p;
}

namespace {
struct Registry {
  std::mutex mutex;
  llvm::StringMap<TimingProviderFactory> factories;
};

Registry &registry() {
  static Registry *table = [] {
    auto *r = new Registry;
    r->factories[kBuiltinTimingProviderId] = [](mlir::ModuleOp) {
      return TimingRuleResult<TimingProvider>{builtinTimingProvider(), {}};
    };
    r->factories[kModelTimingProviderId] = modelTimingProviderFactory;
    return r;
  }();
  return *table;
}

// CIRCT evidence resolver ids that select footprints without a complete policy.
bool isFootprintOnly(llvm::StringRef id) {
  return id == "atlas.vls.conservative.v1" || id == "atlas.vls_dma.serialized.v1" || id == "atlas.vls_xlu.serialized.v1" || id == "atlas.vls_dma_xlu.serialized.v1";
}
} // namespace

std::string mlir::atlas::timing::registerAtlasTimingProvider(
    const std::string &id, TimingProviderFactory factory) {
  if (id.empty()) return "timing provider requires an explicit policy id";
  if (!factory) return "timing provider registration requires a factory: " + id;
  auto complete = [factory = std::move(factory)](mlir::ModuleOp module) {
    auto result = factory(module);
    if (!result.error.empty()) return result;
    if (auto error = fillMachineTiming(result.value); !error.empty())
      return TimingRuleResult<TimingProvider>{{}, error};
    return result;
  };
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!r.factories.try_emplace(canonicalTimingProviderId(id), std::move(complete)).second)
    return "timing provider is already registered: " + id;
  return {};
}

llvm::StringRef mlir::atlas::timing::canonicalTimingProviderId(llvm::StringRef id) {
  if (id == "npu-model-rtl-match-v1")
    return kBuiltinTimingProviderId;
  return id;
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::lookupTimingProvider(
    const std::string &id, mlir::ModuleOp module) {
  const std::string canonical = canonicalTimingProviderId(id).str();
  TimingProviderFactory factory;
  {
    Registry &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto found = r.factories.find(canonical);
    if (found != r.factories.end())
      factory = found->second;
  }
  if (!factory) {
    if (isFootprintOnly(id))
      return {{}, "selected CIRCT VLS evidence lacks a complete timing policy; footprint-only coverage cannot borrow model rules"};
    return {{}, "unknown Atlas timing provider: " + id};
  }
  auto result = factory(module);
  if (!result.error.empty())
    return {{}, result.error};
  if (result.value.id != canonical)
    return {{}, "timing provider factory for " + id + " returned policy " + result.value.id};
  std::string missing = validateTimingProvider(result.value);
  if (!missing.empty())
    return {{}, missing};
  return result;
}
