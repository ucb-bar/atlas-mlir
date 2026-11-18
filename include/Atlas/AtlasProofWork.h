#ifndef ATLAS_PROOF_WORK_H
#define ATLAS_PROOF_WORK_H

#include <cstdint>

namespace mlir::atlas {
// Measurement only; these counters never affect an allowance. A record visit is
// one contract record examined or derived, or one instruction-tag scan step. A
// transfer visit is one abstract instruction or scalar-expression transfer.
enum class AtlasProofKind : unsigned {
  DMA, MXU, Tile, CFG, SourceMemory, Buffer,
  SourceAddress, CapturedAddress, SourceInitialization, CapturedInitialization, InitializationReachability,
  Count
};
enum class AtlasProofEvent : unsigned { Invocation, Record, Transfer };
extern const bool atlasProofWorkReportingEnabled;
void recordAtlasProofWork(AtlasProofKind kind, AtlasProofEvent event,
                          uint64_t count = 1);
inline void atlasProofInvocation(AtlasProofKind kind) {
  if (atlasProofWorkReportingEnabled)
    recordAtlasProofWork(kind, AtlasProofEvent::Invocation);
}
inline void atlasProofRecord(AtlasProofKind kind, uint64_t count = 1) {
  if (atlasProofWorkReportingEnabled)
    recordAtlasProofWork(kind, AtlasProofEvent::Record, count);
}
inline void atlasProofTransfer(AtlasProofKind kind, uint64_t count = 1) {
  if (atlasProofWorkReportingEnabled)
    recordAtlasProofWork(kind, AtlasProofEvent::Transfer, count);
}
} // namespace mlir::atlas
#endif
