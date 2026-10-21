#ifndef ATLAS_BOUNDARY_ABI_H
#define ATLAS_BOUNDARY_ABI_H

#include "Atlas/AtlasTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>

namespace mlir::atlas {

// A virtual function's DRAM boundary: input tile slots from
// atlas.input_dram_base and output slots from atlas.output_dram_base, each
// slot one BF16 tile; an FP8 input fills the first half of its slot. Each
// window spans at least one slot, and through the highest index a boundary
// tile uses.
struct AtlasBoundaryABI {
  uint64_t inputBase = 0, outputBase = 0;
  uint64_t inputSlots = 1, outputSlots = 1;

  // Reads both bases, which must be aligned 32-bit addresses in selected
  // DRAM, and the slots the function's boundary tiles use.
  static FailureOr<AtlasBoundaryABI> read(func::FuncOp function,
                                          bool diagnose = true);
  // Whether the windows neither overlap nor pass 4 GiB.
  LogicalResult verifySpans(func::FuncOp function, bool diagnose = true) const;

  uint64_t inputAddress(uint64_t index, unsigned half = 0) const {
    return inputBase + index * kAtlasBF16TileBytes + half * kAtlasRegisterBytes;
  }
  uint64_t outputAddress(uint64_t index, unsigned half = 0) const {
    return outputBase + index * kAtlasBF16TileBytes + half * kAtlasRegisterBytes;
  }
  uint64_t inputEnd() const { return inputAddress(inputSlots); }
  uint64_t outputEnd() const { return outputAddress(outputSlots); }
};

// Whether `function` has boundary input or output tiles.
bool hasAtlasBoundaryTiles(func::FuncOp function);

} // namespace mlir::atlas

#endif
