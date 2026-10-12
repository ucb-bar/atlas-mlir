#ifndef ATLAS_TIMING_PROVIDER_H
#define ATLAS_TIMING_PROVIDER_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
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

// Complete inputs to final timing verification; scope validation runs first.
// CFG and host-release obligations stay in the common verifier. A provider id
// names policy, not RTL qualification.
struct TimingProvider {
  std::string id;
  // False: a branch or jump, like a halt, issues while a preceding DELAY still
  // stalls, so the timing passes end such a stall on a NOP and verification
  // rejects a DELAY immediately before a redirect.
  bool redirectWaitsForDelay = true;
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
// One block's dependence graph under `provider`: pair dependences, DMA
// completion conflicts resolved at the matching wait, and waits guarding
// transfers from earlier blocks. A footprint or rule error fails the graph.
TimingRuleResult<DepGraph> buildTimingGraph(const std::vector<Instr> &instrs,
                                            const RegValues &entry,
                                            const TimingProvider &provider);
constexpr llvm::StringLiteral kNpuModelTimingProviderId = "npu-model-rtl-match-v1";
// RTL-computed operation timing facts selected on the module (lib/AtlasRTLEvidence.cpp).
constexpr llvm::StringLiteral kAtlasRTLTimingProviderId = "rtl-op-timing-v1";
// Explicit adapter for existing unqualified npu-model rtl-match rules.
TimingProvider npuModelTimingProvider();
// Builds the provider for one, possibly null, module whose attributes may carry
// evidence; the returned id must equal the registered id.
using TimingProviderFactory =
    std::function<TimingRuleResult<TimingProvider>(mlir::ModuleOp module)>;
// Adds `id` to the process-wide registry, which is seeded with the npu-model
// and RTL providers. Returns an error for an empty, duplicate or null registration.
std::string registerAtlasTimingProvider(const std::string &id,
                                        TimingProviderFactory factory);
// Only complete registered policies are returned; unknown and footprint-only
// CIRCT evidence ids fail rather than borrow model rules.
TimingRuleResult<TimingProvider> lookupTimingProvider(const std::string &id,
                                                      mlir::ModuleOp module = {});

} // namespace timing
} // namespace mlir::atlas

#endif
