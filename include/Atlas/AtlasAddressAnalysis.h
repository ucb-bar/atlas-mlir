#ifndef ATLAS_ADDRESS_ANALYSIS_H
#define ATLAS_ADDRESS_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

namespace mlir::atlas {
struct AtlasVerificationContext;
// Check output bounds, preserved inputs and initialized reads of temporaries.
LogicalResult verifyAtlasOutputBuffer(Operation *function);
// Bounded traversal of static scalar control flow. Tensor arithmetic is never
// executed; stores initialize byte ranges, and output reads must be covered.
LogicalResult verifyAtlasOutputInitialization(Operation *function,
                                            uint64_t first, uint64_t end);
struct I32AddressSpan {
  uint32_t first, last; // Minimum and maximum starting byte address.
};
// Constant addresses or canonical pointer/counter induction, guarded by an
// unsigned loop bound. A nonzero constant guard start permits a constant
// positive step, including a pointer guard; its final exit increment must not
// wrap. Unknown recurrences and possible wrapping of observed values are refused.
std::optional<I32AddressSpan> provenI32AddressSpan(Value value, Operation *use);
// Unsigned scalar range at a dominated use; wrapping/unknown cases refuse.
std::optional<I32AddressSpan> provenI32UnsignedRange(Value value, Operation *use);
// Independent replay over captured source CFG records, before accepting any
// dynamic address in a generated stream. Physical origins are checked by CFG.
LogicalResult verifyAtlasBoundedDMAAddresses(const AtlasVerificationContext &ctx);
std::optional<int32_t> atlasCFGValueID(Value value);
} // namespace mlir::atlas
#endif
