#ifndef ATLAS_RTL_VERIFICATION_H
#define ATLAS_RTL_VERIFICATION_H

#include "Atlas/AtlasRTLSelection.h"

namespace mlir::atlas {
struct ResolvedRTLInstruction {
  timing::Instr instruction;
  timing::Footprint footprint;
  int cycle;
};

struct ResolvedRTLProgram {
  std::shared_ptr<timing::RTLEvidence> evidence;
  std::vector<uint32_t> words;
  std::vector<ResolvedRTLInstruction> instructions;
};

// Recompute the final stream using the selected shared resolver. Optional
// output is populated only on success and cleared on failure. This is the
// same check used by the pass, not an additional scheduling oracle.
LogicalResult verifyAtlasRTLTiming(ModuleOp module,
                                  ResolvedRTLProgram *resolved = nullptr);
void registerVerifyAtlasRTLTimingPass();
} // namespace mlir::atlas

#endif
