#ifndef ATLAS_RTL_VERIFICATION_H
#define ATLAS_RTL_VERIFICATION_H

#include "Atlas/AtlasRTLSelection.h"

namespace mlir::atlas {
struct ResolvedRTLInstruction {
  timing::Instr instruction;
  timing::Footprint footprint;
  // After a DMA wait, cycle is a lower bound; each matching wait starts a new
  // epoch, and offsets are comparable only within one epoch.
  int cycle;
  int epoch = 0;
  int epochOffset = 0;
  // Cycles and epochs restart at each basic block, which starts idle.
  int block = 0;
};

struct ResolvedRTLProgram {
  std::shared_ptr<timing::RTLEvidence> evidence;
  std::vector<uint32_t> words;
  std::vector<ResolvedRTLInstruction> instructions;
};

// Rechecks the final stream with the selected resolver; `resolved` is filled
// only on success.
LogicalResult verifyAtlasRTLTiming(ModuleOp module,
                                   ResolvedRTLProgram *resolved = nullptr);
// No-op unless the module carries a selection.
LogicalResult verifySelectedAtlasRTLTiming(ModuleOp module);
void registerVerifyAtlasRTLTimingPass();
} // namespace mlir::atlas

#endif
