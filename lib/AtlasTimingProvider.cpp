#include "Atlas/AtlasTimingProvider.h"
#include "Atlas/AtlasModelTiming.h"
#include "llvm/ADT/StringMap.h"
#include <limits>
#include <mutex>

using namespace mlir::atlas::timing;

namespace {
class ModelReservations final : public TimingReservations {
public:
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

TimingRuleResult<DepGraph> mlir::atlas::timing::buildTimingGraph(
    const std::vector<Instr> &instrs, const RegValues &entry, const TimingProvider &p, const IncomingDma *incomingDma,
    const std::vector<TimingContext> *contexts) {
  std::string missing = validateTimingProvider(p);
  if (!missing.empty()) return {{}, missing};
  if (contexts && contexts->size() != instrs.size())
    return {{}, "timing context count differs from instruction count"};
  auto context = [&](int i) { return contexts ? (*contexts)[i] : TimingContext{}; };
  DepGraph g;
  g.nodes = instrs;
  int n = static_cast<int>(instrs.size());
  RegValues regs = entry;
  for (const Instr &in : instrs) {
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty()) return {{}, f.error};
    if (f.doneAge < 0 || f.doneAge == std::numeric_limits<int>::max())
      return {{}, "timing provider completion age exceeds positive cycle domain"};
    g.footprints.push_back(std::move(f));
    applyScalar(in, regs);
  }

  std::map<std::pair<int, int>, int> index;
  auto add = [&](int from, int to, int distance, EdgeKind kind, const std::string &reason) {
    auto it = index.find({from, to});
    if (it == index.end()) {
      index[{from, to}] = static_cast<int>(g.edges.size());
      g.edges.push_back({from, to, distance, kind, reason});
    } else if (distance > g.edges[it->second].distance) {
      g.edges[it->second] = {from, to, distance, kind, reason};
    }
  };
  for (int b = 0; b < n; b++)
    for (int a = 0; a < b; a++) {
      auto d = p.dependenceFor(g.nodes[a], g.footprints[a], context(a), g.nodes[b], g.footprints[b], context(b));
      if (!d.error.empty()) return {{}, d.error};
      if (d.value.distance < 0) return {{}, "timing provider returned a negative dependence distance"};
      if (d.value.distance > 0)
        add(a, b, d.value.distance, d.value.kind, d.value.reason);
    }
  auto isLaunch = [&](int i) {
    OpClass c = g.nodes[i].op->opClass;
    return c == OpClass::DmaLoad || c == OpClass::DmaStore;
  };
  // A matching wait establishes DMA completion.
  for (int d = 0; d < n; d++) {
    if (!isLaunch(d))
      continue;
    const OpInfo &op = *g.nodes[d].op;
    int wait = -1;
    for (int k = d + 1; k < n && wait < 0; k++)
      if (g.nodes[k].op->opClass == OpClass::DmaWait && g.nodes[k].op->channel == op.channel)
        wait = k;
    for (int k = d + 1; k < n; k++) {
      if (k == wait)
        continue;
      auto conflict = p.dmaConflict(g.footprints[d], g.footprints[k]);
      if (!conflict.error.empty()) return {{}, conflict.error};
      if (!conflict.value.conflict)
        continue;
      if (wait >= 0 && wait < k)
        add(wait, k, 1, conflict.value.kind,
            std::string(edgeKindName(conflict.value.kind)) + " with " + op.name + " (done once the wait issues)");
      else
        add(d, k, 1, EdgeKind::Order, op.name + " may still be in flight (no dma.wait in between)");
    }
  }
  // Use precise incoming DMA facts when supplied.
  for (int w = 0; w < n; w++) {
    const OpInfo &op = *g.nodes[w].op;
    if (op.opClass != OpClass::DmaWait)
      continue;
    bool local = false;
    for (int d = 0; d < w; d++)
      if (isLaunch(d) && g.nodes[d].op->channel == op.channel)
        local = true;
    if (local)
      continue;
    for (int k = w + 1; k < n; k++) {
      bool guarded = isLaunch(k);
      if (incomingDma) {
        guarded = guarded && g.nodes[k].op->channel == op.channel;
        for (const Footprint &dma : (*incomingDma)[op.channel]) {
          auto conflict = p.dmaConflict(dma, g.footprints[k]);
          if (!conflict.error.empty()) return {{}, conflict.error};
          guarded |= conflict.value.conflict;
        }
      } else {
        for (const Access &a : g.footprints[k].accesses)
          if (a.res == Res::Vmem)
            guarded = true;
      }
      if (guarded)
        add(w, k, 1, EdgeKind::Order, op.name + " guards a transfer started in an earlier block");
    }
  }
  g.in.assign(n, {});
  g.out.assign(n, {});
  for (int e = 0; e < static_cast<int>(g.edges.size()); e++) {
    g.out[g.edges[e].from].push_back(e);
    g.in[g.edges[e].to].push_back(e);
  }
  return {std::move(g), {}};
}

mlir::FailureOr<MachineOpTiming> mlir::atlas::timing::machineTimingOf(
    const TimingProvider &p, const std::string &name, mlir::Operation *at,
    const ProbeContexts &contexts) {
  auto error = [&](const std::string &why) -> mlir::FailureOr<MachineOpTiming> {
    return at->emitOpError("virtual cost of ") << name << ": " << why;
  };
  std::string missing = validateTimingProvider(p);
  if (!missing.empty()) return error(missing);
  const OpInfo *op = findOp(name);
  if (!op) return error("unknown machine operation");
  Instr first{op, 0, 2, 4}, second{op, 6, 8, 10};
  switch (op->opClass) {
  case OpClass::WeightPush:
  case OpClass::AccPushFp8:
  case OpClass::AccPushBf16:
  case OpClass::MatMul:
  case OpClass::MatMulAcc:
  case OpClass::PopFp8:
  case OpClass::PopBf16:
    first = {op, 0, 0, 0};
    second = {op, 1, 2, 1};
    break;
  default:
    break;
  }
  if (p.costProbes) {
    auto probes = p.costProbes(*op);
    if (!probes.error.empty()) return error(probes.error);
    if (probes.value.first.op != op || probes.value.second.op != op)
      return error("virtual cost probes must name the requested operation");
    first = probes.value.first;
    second = probes.value.second;
  }
  auto [firstContext, secondContext] = contexts ? contexts(first, second) : std::pair<TimingContext, TimingContext>{};
  Footprint a = p.footprint(first, unknownRegs());
  Footprint b = p.footprint(second, unknownRegs());
  for (const Footprint *f : {&a, &b}) {
    if (!f->error.empty()) return error(f->error);
    if (f->doneAge < 0 || f->doneAge == std::numeric_limits<int>::max())
      return error("completion age exceeds positive cycle domain");
  }
  auto gap = p.issueGap(first);
  if (!gap.error.empty()) return error(gap.error);
  if (gap.value <= 0) return error("issue gap must be positive");
  auto table = p.createReservations();
  if (!table.error.empty() || !table.value)
    return error(table.error.empty() ? "timing provider returned no reservation state" : table.error);
  auto isolated = table.value->conflictWithContext(first, a, 0, firstContext);
  if (!isolated.error.empty()) return error(isolated.error);
  if (!isolated.value.empty()) return error("representative instruction can never issue: " + isolated.value);
  std::string why = table.value->reserveWithContext(first, a, 0, firstContext);
  if (!why.empty()) return error(why);
  constexpr int kMaximumInterval = 100000;
  for (int interval = gap.value; interval <= kMaximumInterval; ++interval) {
    auto conflict = table.value->conflictWithContext(second, b, interval, secondContext);
    if (!conflict.error.empty()) return error(conflict.error);
    if (conflict.value.empty())
      return MachineOpTiming{a.doneAge + 1, interval};
  }
  return error("same-unit issue interval exceeds bounded timing search");
}

std::string mlir::atlas::timing::validateTimingProvider(
    const TimingProvider &p) {
  if (p.id.empty()) return "timing provider requires an explicit policy id";
  if (!p.validateScope) return "timing provider lacks program scope coverage";
  if (!p.footprint) return "timing provider lacks footprint rules";
  if (!p.dependence) return "timing provider lacks dependence rules";
  if (!p.dmaConflict) return "timing provider lacks DMA completion conflict rules";
  if (!p.issueGap) return "timing provider lacks issue gap rules";
  if (!p.createReservations) return "timing provider lacks reservation policy";
  return {};
}

llvm::StringRef mlir::atlas::timing::canonicalTimingProviderId(llvm::StringRef id) {
  if (id == kLegacyBuiltinTimingProviderId)
    return kBuiltinTimingProviderId;
  return id;
}

TimingProvider mlir::atlas::timing::builtinTimingProvider() {
  TimingProvider p;
  p.id = kBuiltinTimingProviderId.str();
  p.validateScope = [](const mlir::atlas::AtlasStream &) { return std::string{}; };
  p.footprint = footprintOf;
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
    result.value = std::make_unique<ModelReservations>();
    return result;
  };
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
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!r.factories.try_emplace(canonicalTimingProviderId(id), std::move(factory)).second)
    return "timing provider is already registered: " + id;
  return {};
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::lookupTimingProvider(
    const std::string &id, mlir::ModuleOp module) {
  llvm::StringRef canonical = canonicalTimingProviderId(id);
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
  result.value.id = canonicalTimingProviderId(result.value.id).str();
  if (result.value.id != canonical)
    return {{}, "timing provider factory for " + id + " returned policy " + result.value.id};
  std::string missing = validateTimingProvider(result.value);
  if (!missing.empty())
    return {{}, missing};
  return result;
}
