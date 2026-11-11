#ifndef ATLAS_TARGET_H
#define ATLAS_TARGET_H

#include <cstdint>

namespace mlir::atlas {

// Facts of the selected Atlas RTL that more than one pass or checker relies
// on. Where a pass places things (which registers, windows or slots) is its
// own policy and stays with it.

// VMEM (ScalarCore.scala AtlasMemMap). DMA addresses it in 4-byte words.
constexpr int kAtlasVmemBytes = 1536 * 1024;

// VMEM is also an SBUS manager at kAtlasVmemBusBase. Its banks 3-5 form a
// system-memory window; a program whose DMA uses it must keep its own VMEM
// accesses in banks 0-2, below kAtlasProgramVmemBytes.
constexpr uint64_t kAtlasVmemBusBase = 0x20000000ULL;
constexpr uint64_t kAtlasProgramVmemBytes = 768 * 1024;

inline bool atlasVmemBusSpan(uint64_t address, uint64_t bytes) {
  return address >= kAtlasVmemBusBase + kAtlasProgramVmemBytes &&
         address < kAtlasVmemBusBase + kAtlasVmemBytes &&
         bytes <= kAtlasVmemBusBase + kAtlasVmemBytes - address;
}

// Selected EE290 system-memory windows reachable by the Atlas SBUS master.
// Keep MMIO holes excluded and require a nonempty transfer in one window.
// An absent boundary input is handled by the ABI, not a zero-byte DMA span.
inline bool atlasDMAMemorySpan(uint64_t address, uint64_t bytes) {
  constexpr uint64_t limit = uint64_t(1) << 32;
  if (!bytes || address >= limit || bytes > limit - address) return false;
  return address >= 0x80000000ULL ||
         (address >= 0x08000000ULL && address + bytes <= 0x08010000ULL) ||
         atlasVmemBusSpan(address, bytes);
}

// DMA channels (DmaParams.scala); a transfer holds one while in flight.
constexpr unsigned kAtlasDMAChannels = 8;

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
