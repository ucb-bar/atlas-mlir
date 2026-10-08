#ifndef ATLAS_RTL_SELECTION_H
#define ATLAS_RTL_SELECTION_H

#include "Atlas/AtlasRTLEvidence.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::atlas {
// No selection returns a null pointer. An invalid selection always fails;
// consumers must never silently fall back to the legacy timing model.
FailureOr<std::shared_ptr<timing::RTLEvidence>>
getSelectedRTLEvidence(ModuleOp module);
// Selected consumers call this before scheduling and after inserting words.
LogicalResult checkSelectedRTLProgramSize(ModuleOp module);
void registerSelectAtlasRTLEvidencePass();
} // namespace mlir::atlas
#endif
