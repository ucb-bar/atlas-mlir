#include "Atlas/AtlasDMAMemoryVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

struct Span {
  std::optional<uint64_t> start;
  std::optional<uint32_t> bytes;
  bool write;
};

struct Transfer {
  DMAOp launch;
  Span vmem, dram;
};

// AtlasCore slices wordAddr[18:3]; DMA and LSU then address 32-byte lines.
uint64_t vmemStart(uint32_t wordAddress) {
  return uint64_t((wordAddress >> 3) & 0xffff) * 32;
}

std::string describe(const Span &span) {
  return std::string(span.write ? "write" : "read") + " byte start=" +
         (span.start ? std::to_string(*span.start) : "unknown") + " bytes=" +
         (span.bytes ? std::to_string(*span.bytes) : "unknown");
}

LogicalResult compare(Operation *op, const Span &a, const Span &b,
                      StringRef space, DMAOp launch) {
  if (!a.write && !b.write)
    return success();
  bool unknown = !a.start || !b.start || !a.bytes || !b.bytes;
  // Unsigned distances avoid constructing an overflowing interval endpoint.
  if (unknown || uint64_t(*a.start - *b.start) < *b.bytes ||
      uint64_t(*b.start - *a.start) < *a.bytes) {
    auto diagnostic = op->emitOpError(
        unknown ? "cannot prove DMA memory disjointness in "
                : "DMA memory conflict in ");
    diagnostic << space << ": access {" << describe(a) << "}, pending {"
               << describe(b) << "}";
    diagnostic.attachNote(launch.getLoc())
        << "captured DMA launch on channel " << launch.getChannel();
    return failure();
  }
  return success();
}

} // namespace

LogicalResult mlir::atlas::verifyAtlasGeneratedDMAMemory(
    const AtlasVerificationContext &ctx) {
  if (!ctx.hasDMA)
    return success();
  assert(ctx.stream && "a DMA stream is decoded");
  const AtlasStream &s = *ctx.stream;
  const auto &bases = ctx.dmaUpperEntry;
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues regs = s.entry[block];
    auto base = bases[block];
    std::vector<Transfer> pending;
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      Operation *op = s.ops[i];
      if (auto config = dyn_cast<DMAConfigOp>(op)) {
        base = regs[config.getBaseReg()];
      } else if (auto dma = dyn_cast<DMAOp>(op)) {
        bool store = dma.getDirection() == "store";
        Transfer transfer{dma, {std::nullopt, std::nullopt, !store},
                          {std::nullopt, std::nullopt, store}};
        // AtlasCore truncates size to 13 bits; DMA.scala uses floor(size/32).
        if (auto size = regs[dma.getSize()]) {
          uint32_t bytes = (*size & 0x1fff) & ~uint32_t(31);
          if (!bytes)
            return op->emitOpError("DMA memory transfer has zero complete beats");
          transfer.vmem.bytes = transfer.dram.bytes = bytes;
        }
        if (auto address = regs[dma.getReg()])
          transfer.vmem.start = vmemStart(*address);
        // TileLinkAdapter aligns requests down to 32 bytes, then narrows to
        // negotiated addressBits. Only the bounded zero-upper ABI is qualified
        // here; different unqualified upper words cannot prove disjointness.
        if (base && *base == 0 && regs[dma.getDram()])
          transfer.dram.start = uint64_t(*regs[dma.getDram()]) & ~uint64_t(31);
        if (transfer.dram.start && transfer.dram.bytes &&
            *transfer.dram.start + *transfer.dram.bytes > (uint64_t(1) << 32))
          transfer.dram.start.reset();
        if (transfer.vmem.start && transfer.vmem.bytes &&
            *transfer.vmem.start + *transfer.vmem.bytes > kVmemBytes)
          return op->emitOpError("DMA memory transfer exceeds VMEM capacity");
        for (Transfer &other : pending)
          if (failed(compare(op, transfer.vmem, other.vmem, "VMEM", other.launch)) ||
              failed(compare(op, transfer.dram, other.dram, "DRAM", other.launch)))
            return failure();
        pending.push_back(transfer);
      } else if (auto wait = dyn_cast<DMAWaitOp>(op)) {
        auto found = std::find_if(pending.begin(), pending.end(),
                                 [&](Transfer &transfer) {
          return transfer.launch.getChannel() == wait.getChannel();
        });
        if (found == pending.end())
          return op->emitOpError("DMA memory wait has no captured transfer");
        pending.erase(found);
      } else if (isa<VLoadOp, VStoreOp>(op)) {
        const Instr &in = s.instrs[i];
        Span access{std::nullopt, 1024, isa<VStoreOp>(op)};
        if (auto address = regs[in.rs1]) {
          // ScalarDecoder shifts signed imm12 by 5 WORDS, then RV32 ADD wraps.
          uint32_t word = *address +
                          uint32_t(signExtend(in.imm, 12) * 32);
          access.start = vmemStart(word);
          // LSU accepts aligned 1KiB vectors contained in one 256KiB bank.
          if (*access.start % 1024 || *access.start + 1024 > kVmemBytes ||
              *access.start / kVmemBankBytes !=
                  (*access.start + 1023) / kVmemBankBytes)
            return op->emitOpError("vector DMA memory access has invalid VMEM span");
        }
        for (Transfer &other : pending)
          if (failed(compare(op, access, other.vmem, "VMEM", other.launch)))
            return failure();
      }
      // Only scalar ALU operations yield constants. Loads, AUIPC and link
      // results conservatively become unknown; SELI leaves scalar regs alone.
      applyScalar(s.instrs[i], regs);
    }
    if (!pending.empty())
      return pending.front().launch.emitOpError(
          "DMA memory transfer crosses a basic-block boundary");
  }
  return success();
}
