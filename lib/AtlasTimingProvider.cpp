#include "Atlas/AtlasTimingProvider.h"

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

TimingRuleResult<TimingProvider> mlir::atlas::timing::lookupTimingProvider(
    const std::string &id) {
  if (id == kNpuModelTimingProviderId)
    return {npuModelTimingProvider(), {}};
  if (id == "atlas.vls.conservative.v1")
    return {{}, "selected CIRCT VLS evidence lacks a complete timing policy; footprint-only coverage cannot borrow model rules"};
  return {{}, "unknown Atlas timing provider: " + id};
}
