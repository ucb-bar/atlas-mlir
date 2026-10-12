#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
struct Record {
  uint32_t id, channel, stagingWord, dramByte, sizeBytes;
  uint32_t stagingReg, dramReg, sizeReg;
  StringRef direction;
  unsigned launches = 0, waits = 0;
};

bool validGeometry(const Record &r) {
  return r.id <= 0x7fffffff && r.channel < 8 &&
         (r.direction == "load" || r.direction == "store") &&
         validDMATileGeometry(uint64_t(r.stagingWord) * 4, r.dramByte, r.sizeBytes) &&
         r.stagingReg > 0 && r.stagingReg < 32 &&
         r.dramReg > 0 && r.dramReg < 32 &&
         r.sizeReg > 0 && r.sizeReg < 32 &&
         r.stagingReg != r.dramReg && r.stagingReg != r.sizeReg &&
         r.dramReg != r.sizeReg;
}

DictionaryAttr encodeRecord(Builder &builder, const Record &r) {
  return builder.getDictionaryAttr({
      namedI32(builder, "id", r.id), namedI32(builder, "channel", r.channel),
      builder.getNamedAttr("direction", builder.getStringAttr(r.direction)),
      namedI32(builder, "staging_word", r.stagingWord), namedI32(builder, "dram_byte", r.dramByte),
      namedI32(builder, "size_bytes", r.sizeBytes), namedI32(builder, "staging_reg", r.stagingReg),
      namedI32(builder, "dram_reg", r.dramReg), namedI32(builder, "size_reg", r.sizeReg)});
}

FailureOr<Record> parseRecord(ModuleOp module, Attribute attr) {
  auto dictionary = dyn_cast<DictionaryAttr>(attr);
  if (!dictionary || dictionary.size() != 9)
    return module.emitOpError("DMA contract record requires exactly the v1 fields");
  Record r{};
  if (failed(readI32Fields<uint32_t>(module, dictionary, "DMA", {
          {"id", &r.id}, {"channel", &r.channel}, {"staging_word", &r.stagingWord},
          {"dram_byte", &r.dramByte}, {"size_bytes", &r.sizeBytes},
          {"staging_reg", &r.stagingReg}, {"dram_reg", &r.dramReg},
          {"size_reg", &r.sizeReg}})))
    return failure();
  auto direction = dyn_cast_or_null<StringAttr>(dictionary.get("direction"));
  if (!direction)
    return module.emitOpError("DMA contract direction requires a string");
  r.direction = direction.getValue();
  if (!validGeometry(r))
    return module.emitOpError("DMA contract record has invalid v1 geometry or register claims");
  return r;
}
} // namespace

FailureOr<ArrayAttr> mlir::atlas::buildAtlasDMAContract(
    func::FuncOp function, ArrayRef<VirtualDMAAssignment> assignments) {
  if (failed(verifyAtlasDMAAllocation(function, assignments)))
    return failure();
  SmallVector<Record> records;
  for (const VirtualDMAAssignment &assignment : assignments) {
    Operation *source = assignment.transfer.getDefiningOp();
    bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(source);
    bool bf16 = isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(source);
    // Proven from source SSA only, never planned commands; additions wrap in RV32.
    auto address = provenI32(source->getOperand(load ? 1 : 2));
    auto size = provenI32(source->getOperand(load ? 2 : 3));
    if (!address || !size)
      return source->emitOpError("DMA contract requires proven wrapping-i32 source address and size");
    const DMATransferPlacement &p = assignment.placement;
    Record r{p.id, p.channel, p.stagingWord, *address, *size,
             p.stagingReg, p.dramReg, p.sizeReg, load ? "load" : "store"};
    if (*size != (bf16 ? 2048u : 1024u) || !validGeometry(r))
      return source->emitOpError("DMA contract source must describe an aligned complete tile in the supported 32-bit address range");
    records.push_back(r);
  }
  llvm::sort(records, [](const Record &a, const Record &b) { return a.id < b.id; });
  Builder builder(function.getContext());
  SmallVector<Attribute> encoded;
  for (const Record &record : records)
    encoded.push_back(encodeRecord(builder, record));
  return builder.getArrayAttr(encoded);
}

LogicalResult mlir::atlas::verifyAtlasGeneratedDMAContract(
    const AtlasVerificationContext &ctx) {
  assert(ctx.generated && "the structural stage checks transfer tags and launch/wait pairing");
  ModuleOp module = ctx.module;
  if (failed(requireAtlasGeneratedArtifact(module)))
    return failure();
  auto array = dyn_cast<ArrayAttr>(module->getAttr(kAtlasDMAContract));
  if (!array)
    return module.emitOpError("generated resource contract requires an atlas.virtual_dma_contract array");
  llvm::DenseMap<uint32_t, Record> records;
  std::optional<uint32_t> previous;
  for (Attribute attr : array) {
    auto record = parseRecord(module, attr);
    if (failed(record))
      return failure();
    if (previous && record->id <= *previous)
      return module.emitOpError("DMA contract ids must be unique and sorted");
    previous = record->id;
    records[record->id] = *record;
  }

  llvm::DenseMap<Operation *, uint32_t> tagged;
  for (Operation &op : module.getBody()->getOperations()) {
    auto id = contractTag(&op, kAtlasTagDMATransfer);
    if (!id)
      continue;
    uint32_t value = uint32_t(*id);
    auto found = records.find(value);
    if (found == records.end())
      return op.emitOpError("DMA contract has no source record for transfer id ") << value;
    Record &r = found->second;
    if (auto dma = dyn_cast<DMAOp>(op)) {
      ++r.launches;
      if (dma.getChannel() != r.channel || dma.getDirection() != r.direction ||
          dma.getReg() != r.stagingReg || dma.getDram() != r.dramReg ||
          dma.getSize() != r.sizeReg)
        return op.emitOpError("DMA contract launch direction, channel, or helper register mismatch");
      tagged[&op] = value;
    } else {
      ++r.waits;
      if (cast<DMAWaitOp>(op).getChannel() != r.channel)
        return op.emitOpError("DMA contract wait channel mismatch");
    }
  }
  for (auto &entry : records)
    if (entry.second.launches != 1 || entry.second.waits != 1)
      return module.emitOpError("DMA contract requires exactly one tagged launch and wait for transfer id ")
             << entry.first;
  if (tagged.empty())
    return success();

  assert(ctx.stream && "a tagged DMA stream is decoded");
  const AtlasStream &s = *ctx.stream;
  const auto &bases = ctx.dmaUpperEntry;
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues regs = s.entry[block];
    auto base = bases[block];
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      Operation *op = s.ops[i];
      if (auto config = dyn_cast<DMAConfigOp>(op))
        base = regs[config.getBaseReg()];
      if (auto found = tagged.find(op); found != tagged.end()) {
        const Record &r = records.find(found->second)->second;
        // Exact captured values also ensure RTL masking/alignment cannot turn
        // a different command into a silently shortened or shifted transfer.
        if (failed(checkCaptured(op, "DMA", "DRAM upper word", base, 0)) ||
            failed(checkCaptured(op, "DMA", "staging word", regs[r.stagingReg], r.stagingWord)) ||
            failed(checkCaptured(op, "DMA", "DRAM byte address", regs[r.dramReg], r.dramByte)) ||
            failed(checkCaptured(op, "DMA", "byte length", regs[r.sizeReg], r.sizeBytes)))
          return failure();
      }
      applyScalar(s.instrs[i], regs);
    }
  }
  return success();
}

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
        auto found = llvm::find_if(pending, [&](Transfer &transfer) {
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
