#include "Atlas/AtlasTimingProvider.h"
#include "Atlas/AtlasRTLSelection.h"
#include "llvm/ADT/StringMap.h"
#include <map>
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
    const std::vector<Instr> &instrs, const RegValues &entry,
    const TimingProvider &p) {
  TimingRuleResult<DepGraph> result;
  DepGraph &g = result.value;
  g.nodes = instrs;
  int n = static_cast<int>(instrs.size());
  RegValues regs = entry;
  for (const Instr &in : instrs) {
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty())
      return {{}, f.error};
    g.footprints.push_back(std::move(f));
    applyScalar(in, regs);
  }
  // Adds an edge, or raises the distance of an existing edge between the nodes.
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
      auto d = p.dependence(g.nodes[a], g.footprints[a], g.nodes[b], g.footprints[b]);
      if (!d.error.empty())
        return {{}, d.error};
      if (d.value.distance > 0)
        add(a, b, d.value.distance, d.value.kind, d.value.reason);
    }
  auto isLaunch = [&](int i) {
    OpClass c = g.nodes[i].op->opClass;
    return c == OpClass::DmaLoad || c == OpClass::DmaStore;
  };
  // A DMA's completion is known only once its dma.wait issues, so later
  // conflicting accesses wait for that dma.wait.
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
      if (!conflict.error.empty())
        return {{}, conflict.error};
      if (!conflict.value.conflict)
        continue;
      if (wait >= 0 && wait < k)
        add(wait, k, 1, conflict.value.kind,
            std::string(edgeKindName(conflict.value.kind)) + " with " + op.name + " (done once the wait issues)");
      else
        add(d, k, 1, EdgeKind::Order, op.name + " may still be in flight (no dma.wait in between)");
    }
  }
  // A wait for a transfer from an earlier block guards conflicting accesses
  // and channel reuse; the transfer latched its registers at launch, so only
  // the VMEM it moves at completion may conflict.
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
      for (const Access &a : g.footprints[k].accesses)
        if (a.res == Res::Vmem)
          guarded = true;
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
  return result;
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

TimingProvider mlir::atlas::timing::npuModelTimingProvider() {
  TimingProvider p;
  p.id = kNpuModelTimingProviderId.str();
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
    r->factories[kNpuModelTimingProviderId] = [](mlir::ModuleOp) {
      return TimingRuleResult<TimingProvider>{npuModelTimingProvider(), {}};
    };
    r->factories[kAtlasRTLTimingProviderId] = mlir::atlas::rtlTimingProviderFactory;
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
  if (!r.factories.try_emplace(id, std::move(factory)).second)
    return "timing provider is already registered: " + id;
  return {};
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::lookupTimingProvider(
    const std::string &id, mlir::ModuleOp module) {
  TimingProviderFactory factory;
  {
    Registry &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto found = r.factories.find(id);
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
  if (result.value.id != id)
    return {{}, "timing provider factory for " + id + " returned policy " + result.value.id};
  std::string missing = validateTimingProvider(result.value);
  if (!missing.empty())
    return {{}, missing};
  return result;
}
