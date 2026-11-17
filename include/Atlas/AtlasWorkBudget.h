#ifndef ATLAS_WORK_BUDGET_H
#define ATLAS_WORK_BUDGET_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir { class Operation; }

namespace mlir::atlas {
// A deterministic allowance for one named algorithm invocation. Count records,
// candidate/edge visits, and repeated fixed-point or reservation probes at the
// point they are examined. A charge succeeds through the limit inclusively;
// the next charge diagnoses failure without wrapping the counter. This bounds
// compiler work, not generated program execution or the hardware's capacity.
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
  uint64_t used() const { return consumed; }
private:
  Operation *operation;
  llvm::StringRef name;
  uint64_t limit, consumed = 0;
  bool exceeded = false;
  bool *failureFlag;
};
} // namespace mlir::atlas
#endif
