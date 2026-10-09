#ifndef ATLAS_TIMING_PROVIDER_H
#define ATLAS_TIMING_PROVIDER_H

#include "Atlas/AtlasTiming.h"
#include <memory>

namespace mlir::atlas {
struct AtlasStream;

namespace timing {

// Missing coverage is an error, distinct from a covered rule that reports no
// dependence or no resource conflict. No generic consumer fills missing rules.
template <typename T> struct TimingRuleResult {
  T value{};
  std::string error;
};

struct DMACompletionConflict {
  bool conflict = false;
  EdgeKind kind = EdgeKind::Order;
};

// A fresh instance belongs to one emitted basic block. The implementation owns
// capacities, port sharing, engine overlap and reservations across a DMA wait.
class TimingReservations {
public:
  virtual ~TimingReservations() = default;
  virtual TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &footprint, int cycle) const = 0;
  virtual std::string reserve(const Instr &in, const Footprint &footprint,
                              int cycle) = 0;
  virtual std::string onWait(const Instr &in, int cycle) = 0;
};

// Complete inputs to final timing verification. Scope validation rejects
// unsupported programs before any timing rule runs. The footprint's doneAge
// supplies completion age; selected-core CFG and host-release obligations stay
// in the common verifier. A provider id names policy, not RTL qualification.
struct TimingProvider {
  std::string id;
  std::function<std::string(const AtlasStream &)> validateScope;
  FootprintResolver footprint;
  std::function<TimingRuleResult<Dependence>(
      const Instr &, const Footprint &, const Instr &, const Footprint &)>
      dependence;
  std::function<TimingRuleResult<DMACompletionConflict>(
      const Footprint &, const Footprint &)> dmaConflict;
  std::function<TimingRuleResult<int>(const Instr &)> issueGap;
  std::function<TimingRuleResult<std::unique_ptr<TimingReservations>>()>
      createReservations;
};

std::string validateTimingProvider(const TimingProvider &provider);
// Explicit adapter for existing unqualified npu-model rtl-match rules.
TimingProvider npuModelTimingProvider();
// Only complete policies are returned. Unknown ids and the current CIRCT
// footprint-only evidence selection fail; they never borrow model rules.
TimingRuleResult<TimingProvider> lookupTimingProvider(const std::string &id);

} // namespace timing
} // namespace mlir::atlas

#endif
