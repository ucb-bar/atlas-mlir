#ifndef ATLAS_TARGET_H
#define ATLAS_TARGET_H

namespace mlir::atlas {

// Compiler capabilities for the selected Atlas layout. Builtin uses these
// constants directly; the model provider checks every model-owned value against
// source-derived architecture facts before constructing any timing provider.
// Placement choices remain compiler policy.

// VMEM (ScalarCore.scala AtlasMemMap). DMA addresses it in 4-byte words.
constexpr int kAtlasVmemBytes = 1536 * 1024;

// DMA channels (DmaParams.scala); a transfer holds one while in flight.
constexpr unsigned kAtlasDMAChannels = 8;

// RTL CONTRACT ONLY: shared DMA command ring (DmaParams.scala), across
// channels and directions. The NPU Model does not represent this rule; it
// must not be presented as a model-derived architecture fact.
constexpr unsigned kAtlasDMACommandSlots = 8;

// Tensor registers m0..m63 of 1 KiB. One VLOAD or VSTORE moves one register,
// 1 KiB-aligned (LSU.scala). An FP8 tile fills one register and a BF16 tile an
// even-based pair, which is also the size of one boundary tile slot in DRAM.
constexpr int kAtlasTensorRegisters = 64;
constexpr int kAtlasMregRows = 32;
constexpr int kAtlasMregRowBytes = 32;
constexpr int kAtlasMregBanks = 32;
constexpr int kAtlasBF16Registers = 2;
constexpr int kAtlasVmemBankBytes = 256 * 1024;
constexpr int kAtlasRegisterBytes = kAtlasMregRows * kAtlasMregRowBytes;
constexpr int kAtlasRegisterWords = kAtlasRegisterBytes / 4;
constexpr int kAtlasBF16TileBytes = kAtlasBF16Registers * kAtlasRegisterBytes;
constexpr int kAtlasBF16TileWords = kAtlasBF16TileBytes / 4;

} // namespace mlir::atlas

#endif
