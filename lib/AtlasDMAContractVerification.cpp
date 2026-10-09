#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
constexpr llvm::StringLiteral kMarker = "atlas.generated_from_virtual";
constexpr llvm::StringLiteral kContract = "atlas.virtual_dma_contract";
constexpr llvm::StringLiteral kTransfer = "atlas.virtual_dma_transfer";
constexpr llvm::StringLiteral kVersion = "dma-contract-v1";

struct Record {
  uint32_t id, channel, stagingWord, dramByte, sizeBytes;
  uint32_t stagingReg, dramReg, sizeReg;
  StringRef direction;
  unsigned launches = 0, waits = 0;
};

// This proof reads only source SSA, not the lowering's planned commands or
// scalar constant cache. Overflow flags are excluded: additions wrap in RV32.
std::optional<uint32_t> sourceI32(
    Value value, llvm::DenseMap<Value, std::optional<uint32_t>> &known) {
  auto [entry, inserted] = known.try_emplace(value, std::nullopt);
  if (!inserted)
    return entry->second;
  if (!value.getType().isSignlessInteger(32))
    return std::nullopt;
  std::optional<uint32_t> result;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue());
        integer && integer.getType().isSignlessInteger(32))
      result = uint32_t(integer.getValue().getZExtValue());
  } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
    auto lhs = sourceI32(add.getLhs(), known);
    auto rhs = sourceI32(add.getRhs(), known);
    if (lhs && rhs)
      result = uint32_t(uint64_t(*lhs) + *rhs);
  }
  known[value] = result;
  return result;
}

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
  auto i32 = [&](uint32_t bits) {
    return builder.getIntegerAttr(builder.getI32Type(), llvm::APInt(32, bits));
  };
  return builder.getDictionaryAttr({
      builder.getNamedAttr("id", i32(r.id)),
      builder.getNamedAttr("channel", i32(r.channel)),
      builder.getNamedAttr("direction", builder.getStringAttr(r.direction)),
      builder.getNamedAttr("staging_word", i32(r.stagingWord)),
      builder.getNamedAttr("dram_byte", i32(r.dramByte)),
      builder.getNamedAttr("size_bytes", i32(r.sizeBytes)),
      builder.getNamedAttr("staging_reg", i32(r.stagingReg)),
      builder.getNamedAttr("dram_reg", i32(r.dramReg)),
      builder.getNamedAttr("size_reg", i32(r.sizeReg))});
}

FailureOr<Record> parseRecord(ModuleOp module, Attribute attr) {
  auto dictionary = dyn_cast<DictionaryAttr>(attr);
  if (!dictionary || dictionary.size() != 9) {
    module.emitOpError("DMA contract record requires exactly the v1 fields");
    return failure();
  }
  Record r{};
  for (auto [name, field] : {
           std::pair<StringRef, uint32_t *>("id", &r.id),
           {"channel", &r.channel}, {"staging_word", &r.stagingWord},
           {"dram_byte", &r.dramByte}, {"size_bytes", &r.sizeBytes},
           {"staging_reg", &r.stagingReg}, {"dram_reg", &r.dramReg},
           {"size_reg", &r.sizeReg}}) {
    auto integer = dyn_cast_or_null<IntegerAttr>(dictionary.get(name));
    if (!integer || !integer.getType().isSignlessInteger(32)) {
      module.emitOpError("DMA contract field requires signless i32: ") << name;
      return failure();
    }
    *field = uint32_t(integer.getValue().getZExtValue());
  }
  auto direction = dyn_cast_or_null<StringAttr>(dictionary.get("direction"));
  if (!direction) {
    module.emitOpError("DMA contract direction requires a string");
    return failure();
  }
  r.direction = direction.getValue();
  if (!validGeometry(r)) {
    module.emitOpError("DMA contract record has invalid v1 geometry or register claims");
    return failure();
  }
  return r;
}

LogicalResult capturedValue(Operation *op, StringRef field,
                            std::optional<uint32_t> actual, uint32_t expected) {
  if (!actual)
    return op->emitOpError("DMA contract cannot prove captured ") << field;
  if (*actual != expected)
    return op->emitOpError("DMA contract captured ")
           << field << " mismatch: expected " << expected << ", got " << *actual;
  return success();
}
} // namespace

FailureOr<ArrayAttr> mlir::atlas::buildAtlasDMAContract(
    func::FuncOp function, ArrayRef<VirtualDMAAssignment> assignments) {
  if (failed(verifyAtlasDMAAllocation(function, assignments)))
    return failure();
  llvm::DenseMap<Value, std::optional<uint32_t>> known;
  SmallVector<Record> records;
  for (const VirtualDMAAssignment &assignment : assignments) {
    Operation *source = assignment.transfer.getDefiningOp();
    bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(source);
    bool bf16 = isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(source);
    auto address = sourceI32(source->getOperand(load ? 1 : 2), known);
    auto size = sourceI32(source->getOperand(load ? 2 : 3), known);
    if (!address || !size) {
      source->emitOpError("DMA contract requires proven wrapping-i32 source address and size");
      return failure();
    }
    const DMATransferPlacement &p = assignment.placement;
    Record r{p.id, p.channel, p.stagingWord, *address, *size,
             p.stagingReg, p.dramReg, p.sizeReg, load ? "load" : "store"};
    if (*size != (bf16 ? 2048u : 1024u) || !validGeometry(r)) {
      source->emitOpError("DMA contract source must describe an aligned complete tile in the supported 32-bit address range");
      return failure();
    }
    records.push_back(r);
  }
  llvm::sort(records, [](const Record &a, const Record &b) { return a.id < b.id; });
  Builder builder(function.getContext());
  SmallVector<Attribute> encoded;
  for (const Record &record : records)
    encoded.push_back(encodeRecord(builder, record));
  return builder.getArrayAttr(encoded);
}

LogicalResult mlir::atlas::verifyAtlasGeneratedDMAContract(ModuleOp module) {
  Attribute marker = module->getAttr(kMarker);
  Attribute contract = module->getAttr(kContract);
  if (isa_and_nonnull<UnitAttr>(marker)) {
    if (contract)
      return module.emitOpError("legacy generated marker cannot carry a DMA contract");
    return success();
  }
  auto version = dyn_cast_or_null<StringAttr>(marker);
  if (!version || (version.getValue() != kVersion && version.getValue() != "resource-contract-v1" && version.getValue() != "resource-contract-v2"))
    return module.emitOpError("expected generated marker version resource-contract-v2, resource-contract-v1, dma-contract-v1 or legacy unit");
  auto array = dyn_cast_or_null<ArrayAttr>(contract);
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
    Attribute attr = op.getAttr(kTransfer);
    if (!attr)
      continue;
    auto id = dyn_cast<IntegerAttr>(attr);
    if (!isa<DMAOp, DMAWaitOp>(op) || !id ||
        !id.getType().isSignlessInteger(32) || id.getValue().isNegative())
      return op.emitOpError("DMA contract transfer tag requires nonnegative i32 on DMA or DMA.WAIT");
    uint32_t value = uint32_t(id.getValue().getZExtValue());
    auto found = records.find(value);
    if (found == records.end())
      return op.emitOpError("DMA contract has no source record for transfer id ") << value;
    Record &r = found->second;
    if (auto dma = dyn_cast<DMAOp>(op)) {
      if (++r.launches != 1)
        return op.emitOpError("DMA contract has duplicate tagged launch");
      if (dma.getChannel() != r.channel || dma.getDirection() != r.direction ||
          dma.getReg() != r.stagingReg || dma.getDram() != r.dramReg ||
          dma.getSize() != r.sizeReg)
        return op.emitOpError("DMA contract launch direction, channel, or helper register mismatch");
      tagged[&op] = value;
    } else {
      auto wait = cast<DMAWaitOp>(op);
      if (++r.waits != 1)
        return op.emitOpError("DMA contract has duplicate tagged wait");
      if (wait.getChannel() != r.channel)
        return op.emitOpError("DMA contract wait channel mismatch");
    }
  }
  for (auto &entry : records)
    if (entry.second.launches != 1 || entry.second.waits != 1)
      return module.emitOpError("DMA contract requires exactly one tagged launch and wait for transfer id ")
             << entry.first;
  if (tagged.empty())
    return success();

  auto stream = readAtlasStream(module, AtlasStreamReadMode::Verification);
  if (failed(stream))
    return failure();
  const AtlasStream &s = *stream;
  auto bases = atlasDMAUpperWordEntries(s);
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
        if (failed(capturedValue(op, "DRAM upper word", base, 0)) ||
            failed(capturedValue(op, "staging word", regs[r.stagingReg], r.stagingWord)) ||
            failed(capturedValue(op, "DRAM byte address", regs[r.dramReg], r.dramByte)) ||
            failed(capturedValue(op, "byte length", regs[r.sizeReg], r.sizeBytes)))
          return failure();
      }
      applyScalar(s.instrs[i], regs);
    }
  }
  return success();
}
