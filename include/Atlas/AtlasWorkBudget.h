#ifndef ATLAS_WORK_BUDGET_H
#define ATLAS_WORK_BUDGET_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir { class Operation; }

namespace mlir::atlas {
// A deterministic compiler-work allowance for one named algorithm invocation,
// charged per record, candidate, edge, or repeated probe when examined. Charges
// succeed through the limit inclusively; the next one diagnoses failure.
class AtlasWorkBudget {
public:
  AtlasWorkBudget(Operation *operation, llvm::StringRef name, bool *failureFlag = nullptr);
  ~AtlasWorkBudget();
  AtlasWorkBudget(const AtlasWorkBudget &) = delete;
  AtlasWorkBudget &operator=(const AtlasWorkBudget &) = delete;
  LogicalResult charge(uint64_t count = 1);
  // Trials may suppress diagnostics, but a refused charge is always terminal.
  static bool anyExhausted();
  bool exhausted() const { return exceeded; }
private:
  Operation *operation;
  llvm::StringRef name;
  uint64_t limit, consumed = 0;
  bool exceeded = false;
  bool *failureFlag;
};
} // namespace mlir::atlas
#endif
