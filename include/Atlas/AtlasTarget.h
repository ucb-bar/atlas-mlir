#ifndef ATLAS_TARGET_H
#define ATLAS_TARGET_H

namespace mlir::atlas {

// Facts of the selected Atlas RTL that more than one pass or checker relies
// on. Where a pass places things (which registers, windows or slots) is its
// own policy and stays with it.

// VMEM (ScalarCore.scala AtlasMemMap). DMA addresses it in 4-byte words.
constexpr int kAtlasVmemBytes = 1536 * 1024;

// DMA channels (DmaParams.scala); a transfer holds one while in flight.
constexpr unsigned kAtlasDMAChannels = 8;

// Shared DMA command ring depth (DmaParams.scala), across channels and directions.
constexpr unsigned kAtlasDMACommandSlots = 8;

// Tensor registers m0..m63 of 1 KiB. One VLOAD or VSTORE moves one register,
// 1 KiB-aligned (LSU.scala). An FP8 tile fills one register and a BF16 tile an
// even-based pair, which is also the size of one boundary tile slot in DRAM.
constexpr int kAtlasTensorRegisters = 64;
constexpr int kAtlasRegisterBytes = 1024;
constexpr int kAtlasRegisterWords = kAtlasRegisterBytes / 4;
constexpr int kAtlasBF16TileBytes = 2 * kAtlasRegisterBytes;
constexpr int kAtlasBF16TileWords = kAtlasBF16TileBytes / 4;

} // namespace mlir::atlas

#endif
