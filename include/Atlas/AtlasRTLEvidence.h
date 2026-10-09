#ifndef ATLAS_RTL_EVIDENCE_H
#define ATLAS_RTL_EVIDENCE_H

#include "Atlas/AtlasTiming.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace mlir::atlas::timing {

// Caller-selected byte identities. No field claims full Chisel-source equality.
struct ExpectedEvidenceIdentity {
  std::string evidenceSha256;
  std::string manifestSha256;
  std::string hardwareIRSha256;
};

class RTLEvidence {
public:
  static constexpr const char *resolverID() {
    return "atlas.vls.conservative.v1";
  }
  static constexpr int resolverVersion() { return 1; }
  static constexpr const char *dmaResolverID() {
    return "atlas.vls_dma.serialized.v1";
  }
  static constexpr const char *resolverIDFor(bool dma, bool xlu) {
    return xlu ? (dma ? "atlas.vls_dma_xlu.serialized.v1" :
                       "atlas.vls_xlu.serialized.v1") :
                 (dma ? dmaResolverID() : resolverID());
  }
  const char *selectedResolverID() const {
    return resolverIDFor(hasDMAEvidence(), hasXLUEvidence());
  }
  bool hasDMAEvidence() const { return !dmaIdentity.empty(); }
  const std::string &dmaEvidenceSha256() const { return dmaIdentity; }
  bool hasXLUEvidence() const { return !xluIdentity.empty(); }
  const std::string &xluEvidenceSha256() const { return xluIdentity; }
  // Reviewed ScalarCore fetch uses a 15-bit word index into InstrMem.
  static constexpr unsigned maximumProgramWords() { return 32768; }
  const std::string &evidenceSha256() const { return identity.evidenceSha256; }
  const std::string &manifestSha256() const { return identity.manifestSha256; }
  const std::string &hardwareIRSha256() const { return identity.hardwareIRSha256; }
  const char *qualificationStatus() const { return "conditional"; }

  // Resolves the explicit bounded subset. Unsupported instances carry error.
  // Every VLS reserves BOTH paths, deliberately serializing all vector memory.
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  // Describes resolver policy and assumptions, not newly qualified domains.
  llvm::json::Object applicability() const;
  Footprint footprintOf(const Instr &in, const RegValues &regs) const {
    return resolve(in, regs);
  }

private:
  struct Stream {
    int sourceAge = 0, destinationAge = 0, step = 0, rows = 0;
    int busyLast = 0, sourceLast = 0, destinationLast = 0;
  };
  Stream load, store, transpose;
  ExpectedEvidenceIdentity identity;
  std::string dmaIdentity;
  std::string xluIdentity;
  friend llvm::Expected<RTLEvidence>
  loadRTLEvidence(llvm::StringRef, const ExpectedEvidenceIdentity &, bool);
  friend llvm::Error loadDMAEvidence(RTLEvidence &, llvm::StringRef,
                                     llvm::StringRef);
  friend llvm::Error loadXLUEvidence(RTLEvidence &, llvm::StringRef,
                                     llvm::StringRef);
};

// Draft full-ISA contracts are intentionally not accepted by this loader.
// The report remains conditional; opting in does not qualify the target.
llvm::Expected<RTLEvidence>
loadRTLEvidence(llvm::StringRef path, const ExpectedEvidenceIdentity &expected,
                bool conditionalOptIn);

// Extend an already selected VLS provider only after checking a separate
// selected-core DMA receipt against the same retained hardware identity.
llvm::Error loadDMAEvidence(RTLEvidence &evidence, llvm::StringRef path,
                            llvm::StringRef expectedSha256);
llvm::Error loadXLUEvidence(RTLEvidence &evidence, llvm::StringRef path,
                            llvm::StringRef expectedSha256);

// Exact reviewed semantic module closure for resolver v1. Location aliases
// are ignored; changed frontend, geometry or wiring requires renewed review.
// This checks text already in memory and does not inspect artifact paths.
llvm::Error checkRTLSemanticCompatibility(llvm::StringRef hardwareIR);

} // namespace mlir::atlas::timing
#endif
