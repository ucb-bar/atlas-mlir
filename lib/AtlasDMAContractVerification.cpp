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
  // AtlasCore uses wordAddr[18:3], size[12:0]; DMA consumes complete
  // 32-byte beats. Full FP8/BF16 tiles are 1/2 KiB in selected VMEM.
  return r.id <= 0x7fffffff && r.channel < 8 &&
         (r.direction == "load" || r.direction == "store") &&
         (r.sizeBytes == 1024 || r.sizeBytes == 2048) &&
         r.stagingWord % 256 == 0 &&
         uint64_t(r.stagingWord) * 4 + r.sizeBytes <= kVmemBytes &&
         r.dramByte >= 0x80000000u && r.dramByte % 32 == 0 &&
         uint64_t(r.dramByte) + r.sizeBytes <= (uint64_t(1) << 32) &&
         r.stagingReg > 0 && r.stagingReg < 32 &&
         r.dramReg > 0 && r.dramReg < 32 &&
         r.sizeReg > 0 && r.sizeReg < 32 &&
         r.stagingReg != r.dramReg && r.stagingReg != r.sizeReg &&
         r.dramReg != r.sizeReg;
}

DictionaryAttr encodeRecord(Builder &builder, const Record &r) {
  auto field = [&](StringRef name, uint32_t bits) {
    return builder.getNamedAttr(name, builder.getIntegerAttr(builder.getI32Type(), llvm::APInt(32, bits)));
  };
  return builder.getDictionaryAttr({
      field("id", r.id), field("channel", r.channel),
      builder.getNamedAttr("direction", builder.getStringAttr(r.direction)),
      field("staging_word", r.stagingWord), field("dram_byte", r.dramByte),
      field("size_bytes", r.sizeBytes), field("staging_reg", r.stagingReg),
      field("dram_reg", r.dramReg), field("size_reg", r.sizeReg)});
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
    auto id = op.getAttrOfType<IntegerAttr>(kAtlasTagDMATransfer);
    if (!id)
      continue;
    uint32_t value = uint32_t(id.getValue().getZExtValue());
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
