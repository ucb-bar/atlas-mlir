#ifndef ATLAS_RTL_SELECTION_H
#define ATLAS_RTL_SELECTION_H

#include "Atlas/AtlasRTLEvidence.h"
#include "Atlas/AtlasTimingProvider.h"
#include "Atlas/AtlasDelayInsertion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::atlas {
// Load hash-checked atlas.rtl_evidence; absent selection is null, invalid is an error.
llvm::Expected<std::shared_ptr<timing::RTLEvidence>>
loadSelectedRTLEvidence(ModuleOp module);
FailureOr<std::shared_ptr<timing::RTLEvidence>>
getSelectedRTLEvidence(ModuleOp module);
// The complete RTL policy: selected footprints, exclusive engines or pair
// reservations, drained blocks, bounded programs and NOP guards before
// redirects. DMA: one pending transfer per channel and at most eight, disjoint
// VMEM windows, DRAM spans shared only by reads, VLS off pending windows, MXU
// serialized with DMA, each completed only at its matching wait in the block.
timing::TimingProvider
rtlTimingProvider(std::shared_ptr<const timing::RTLEvidence> evidence);
// Registered factory: fails without valid selected evidence on the module.
timing::TimingRuleResult<timing::TimingProvider>
rtlTimingProviderFactory(ModuleOp module);
// Scheduler-local heuristics: these never alter provider facts or verification.
struct RTLSchedulingOrder {
  bool dmaEstimate = false;
  bool contention = false;
};
void estimateRTLGraphDMA(timing::DepGraph &graph);
std::vector<int> rtlContentionPriorities(const timing::DepGraph &graph,
                                       const std::vector<int> &issued,
                                       int bodyCount);
using RTLBlockScheduler = std::function<LogicalResult(
    const timing::RegValues &, RTLSchedulingOrder, std::vector<size_t> &,
    std::vector<DelayInsertion> &, int &)>;
LogicalResult selectRTLBlockSchedule(const AtlasStream &stream, size_t block,
                                    const timing::TimingProvider &provider,
                                    bool dmaEstimate,
                                    const RTLBlockScheduler &schedule,
                                    std::vector<size_t> &order,
                                    std::vector<DelayInsertion> &before,
                                    int &tail);
void registerSelectAtlasRTLEvidencePass();
} // namespace mlir::atlas
#endif
