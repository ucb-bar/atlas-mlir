#ifndef ATLAS_RTL_SELECTION_H
#define ATLAS_RTL_SELECTION_H

#include "Atlas/AtlasRTLEvidence.h"
#include "Atlas/AtlasStream.h"
#include <memory>

namespace mlir::atlas {
// Null when nothing is selected; an invalid selection fails rather than
// falling back to the model.
FailureOr<std::shared_ptr<timing::RTLEvidence>>
getSelectedRTLEvidence(ModuleOp module);
LogicalResult checkSelectedRTLProgramSize(ModuleOp module);
// Reads and checks the stream for a timing pass under the selected target, if
// any. A selected stream must satisfy checkSelectedControlFlow.
FailureOr<AtlasStream> readTimedStream(ModuleOp module,
                                       timing::TargetTiming &target);
void registerSelectAtlasRTLEvidencePass();
} // namespace mlir::atlas
#endif
