#ifndef ATLAS_RTL_SELECTION_H
#define ATLAS_RTL_SELECTION_H

#include "Atlas/AtlasRTLEvidence.h"
#include "Atlas/AtlasTimingProvider.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::atlas {
// The facts named by atlas.rtl_evidence, loaded and hash-checked; null when
// the module carries no selection. An invalid selection is an error rather
// than a fallback to the model.
llvm::Expected<std::shared_ptr<timing::RTLEvidence>>
loadSelectedRTLEvidence(ModuleOp module);
FailureOr<std::shared_ptr<timing::RTLEvidence>>
getSelectedRTLEvidence(ModuleOp module);
// The complete kAtlasRTLTimingProviderId policy over selected facts: facts
// footprints, data dependences, queued transfers (one per channel, eight
// command slots, disjoint VMEM windows, DRAM spans shared only by reads, VLS
// off pending windows, MXU serialized with DMA) completed only at their
// matching wait within the block, drained basic blocks, exclusive engine
// reservations (with selected pair facts, VPU commands overlap where a pair
// fact admits them, up to the facts' live capacity), the instruction-memory and
// issue-cycle bounds, and NOP guards before redirects.
timing::TimingProvider
rtlTimingProvider(std::shared_ptr<const timing::RTLEvidence> evidence);
// Registered factory: fails without valid selected evidence on the module.
timing::TimingRuleResult<timing::TimingProvider>
rtlTimingProviderFactory(ModuleOp module);
void registerSelectAtlasRTLEvidencePass();
} // namespace mlir::atlas
#endif
