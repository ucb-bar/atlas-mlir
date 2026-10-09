#ifndef ATLAS_RTL_VERIFICATION_H
#define ATLAS_RTL_VERIFICATION_H

#include "Atlas/AtlasRTLSelection.h"

namespace mlir::atlas {
struct ResolvedRTLInstruction {
  timing::Instr instruction;
  timing::Footprint footprint;
  int cycle;
  // With dynamic waits, cycle is only a lower bound. An accepted matching
  // wait starts a new epoch; epoch offsets never imply a fixed wait duration.
  int epoch = 0;
  int epochOffset = 0;
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
// Executable output boundaries call this after transformations. Unselected
// legacy streams retain their existing checks; partial selection must fail.
LogicalResult verifySelectedAtlasRTLTiming(ModuleOp module);
void registerVerifyAtlasRTLTimingPass();
} // namespace mlir::atlas

#endif
