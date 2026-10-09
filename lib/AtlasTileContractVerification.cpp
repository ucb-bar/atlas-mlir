#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
constexpr llvm::StringLiteral kMarker = "atlas.generated_from_virtual";
constexpr llvm::StringLiteral kContract = "atlas.virtual_tile_contract";
constexpr llvm::StringLiteral kCommand = "atlas.virtual_tile_command";
constexpr llvm::StringLiteral kVersion = "resource-contract-v2";

struct Record {
  int32_t id = -1;
  StringRef kind;
  int32_t reg = -1;
  uint32_t vmemByte = 0, dramByte = 0, bytes = 0;
  int32_t channel = -1, transfer = -1;
  SmallVector<int32_t, 2> after;
};

bool isVector(const Record &r) { return r.kind == "vload" || r.kind == "vstore"; }
bool isDMA(const Record &r) { return r.kind == "dma_load" || r.kind == "dma_store"; }

bool validRecord(const Record &r) {
  if (r.id < 0 || r.transfer < -1)
    return false;
  if (r.kind == "dma_wait")
    return r.reg == -1 && r.vmemByte == 0 && r.dramByte == 0 &&
           r.bytes == 0 && r.channel >= 0 && r.channel < 8;
  if (isDMA(r))
    return r.reg == -1 && r.channel >= 0 && r.channel < 8 &&
           (r.bytes == 1024 || r.bytes == 2048) && r.vmemByte % 1024 == 0 &&
           uint64_t(r.vmemByte) + r.bytes <= kVmemBytes &&
           r.dramByte >= 0x80000000u && r.dramByte % 32 == 0 &&
           uint64_t(r.dramByte) + r.bytes <= (uint64_t(1) << 32);
  if (r.channel != -1 || r.dramByte != 0)
    return false;
  if (isVector(r))
    return r.reg >= 0 && r.reg < 64 && r.bytes == 1024 &&
           r.vmemByte % 1024 == 0 && uint64_t(r.vmemByte) + 1024 <= kVmemBytes &&
           r.vmemByte / kVmemBankBytes == (r.vmemByte + 1023) / kVmemBankBytes;
  return r.kind == "mailbox_load" && r.reg > 0 && r.reg < 32 &&
         r.transfer == -1 && r.bytes == 4 && r.vmemByte % 4 == 0 &&
         uint64_t(r.vmemByte) + 4 <= kVmemBytes;
}

DictionaryAttr encodeRecord(Builder &builder, const Record &r) {
  auto field = [&](StringRef name, uint32_t bits) {
    return builder.getNamedAttr(name, builder.getIntegerAttr(
        builder.getI32Type(), llvm::APInt(32, bits)));
  };
  return builder.getDictionaryAttr({
      field("id", r.id), builder.getNamedAttr("kind", builder.getStringAttr(r.kind)),
      field("reg", r.reg), field("vmem_byte", r.vmemByte),
      field("dram_byte", r.dramByte), field("bytes", r.bytes),
      field("channel", r.channel), field("transfer", r.transfer),
      builder.getNamedAttr("after", builder.getDenseI32ArrayAttr(r.after))});
}

FailureOr<Record> parseRecord(ModuleOp module, Attribute attr) {
  auto dictionary = dyn_cast<DictionaryAttr>(attr);
  if (!dictionary || dictionary.size() != 9) {
    module.emitOpError("tile contract record requires exactly the v2 fields");
    return failure();
  }
  Record r;
  for (auto [name, field] : {
           std::pair<StringRef, int32_t *>("id", &r.id), {"reg", &r.reg},
           {"channel", &r.channel}, {"transfer", &r.transfer}}) {
    auto integer = dictionary.getAs<IntegerAttr>(name);
    if (!integer || !integer.getType().isSignlessInteger(32)) {
      module.emitOpError("tile contract field requires signless i32: ") << name;
      return failure();
    }
    *field = int32_t(integer.getValue().getSExtValue());
  }
  for (auto [name, field] : {
           std::pair<StringRef, uint32_t *>("vmem_byte", &r.vmemByte),
           {"dram_byte", &r.dramByte}, {"bytes", &r.bytes}}) {
    auto integer = dictionary.getAs<IntegerAttr>(name);
    if (!integer || !integer.getType().isSignlessInteger(32)) {
      module.emitOpError("tile contract field requires signless i32: ") << name;
      return failure();
    }
    *field = uint32_t(integer.getValue().getZExtValue());
  }
  auto kind = dictionary.getAs<StringAttr>("kind");
  auto after = dictionary.getAs<DenseI32ArrayAttr>("after");
  if (!kind || !after) {
    module.emitOpError("tile contract requires string kind and dense i32 after fields");
    return failure();
  }
  r.kind = kind.getValue();
  r.after.append(after.asArrayRef().begin(), after.asArrayRef().end());
  if (!validRecord(r)) {
    module.emitOpError("tile contract record has invalid v2 fields or geometry");
    return failure();
  }
  return r;
}

LogicalResult checkValue(Operation *op, StringRef field,
                         std::optional<uint64_t> actual, uint64_t expected) {
  if (!actual)
    return op->emitOpError("tile contract cannot prove captured ") << field;
  if (*actual != expected)
    return op->emitOpError("tile contract captured ") << field << " mismatch: expected "
           << expected << ", got " << *actual;
  return success();
}

LogicalResult checkReferences(ModuleOp module, ArrayRef<Record> records) {
  auto contains = [](const Record &outer, const Record &inner) {
    return outer.vmemByte <= inner.vmemByte &&
           uint64_t(inner.vmemByte) + inner.bytes <=
               uint64_t(outer.vmemByte) + outer.bytes;
  };
  for (const Record &r : records) {
    bool valid = false;
    if (r.kind == "dma_load" || r.kind == "vstore") {
      valid = r.after.empty();
    } else if (r.kind == "dma_store") {
      valid = r.after.size() == r.bytes / 1024;
      unsigned covered = 0;
      for (int32_t id : r.after) {
        const Record &store = records[id];
        if (store.kind != "vstore" || store.transfer != r.transfer ||
            !contains(r, store)) {
          valid = false;
          break;
        }
        unsigned half = (store.vmemByte - r.vmemByte) / 1024;
        if (covered & (1u << half)) {
          valid = false;
          break;
        }
        covered |= 1u << half;
      }
      valid &= covered == (1u << (r.bytes / 1024)) - 1;
    } else if (r.kind == "dma_wait") {
      if (r.after.size() == 1) {
        const Record &launch = records[r.after[0]];
        valid = isDMA(launch) && launch.channel == r.channel &&
                launch.transfer == r.transfer;
      }
    } else if (r.after.size() == 1) {
      const Record &predecessor = records[r.after[0]];
      if (r.kind == "vload" && predecessor.kind == "vstore") {
        // PACK's vector endpoints bracket its scalar relayout. Endpoint
        // ordering alone does not prove that relayout.
        valid = r.transfer == -1 && predecessor.transfer == -1 &&
                r.reg == predecessor.reg;
      } else if (predecessor.kind == "dma_wait") {
        // Earlier records have already passed their launch-reference check.
        const Record &launch = records[predecessor.after[0]];
        valid = launch.kind == "dma_load" && r.transfer == launch.transfer &&
                contains(launch, r) &&
                (r.kind == "vload" || r.kind == "mailbox_load");
      }
    }
    if (!valid)
      return module.emitOpError("tile contract has incompatible source expansion predecessors at command ")
             << r.id;
  }
  return success();
}

// Static must-executed facts establish necessary ordering through emitted CFG
// joins and PACK's generated loop. They do not prove iteration correspondence.
LogicalResult checkDependencies(const AtlasStream &s, ArrayRef<Record> records,
                                const llvm::DenseMap<Operation *, int32_t> &tags) {
  const size_t blocks = s.starts.size();
  if (!blocks)
    return success();
  std::vector<bool> reachable(blocks, false);
  SmallVector<size_t> worklist{0};
  reachable[0] = true;
  while (!worklist.empty()) {
    size_t block = worklist.pop_back_val();
    for (size_t successor : s.succs[block])
      if (!reachable[successor]) {
        reachable[successor] = true;
        worklist.push_back(successor);
      }
  }
  std::vector<SmallVector<size_t>> predecessors(blocks);
  for (size_t block = 0; block < blocks; ++block)
    if (reachable[block])
      for (size_t successor : s.succs[block])
        predecessors[successor].push_back(block);
  std::vector<llvm::BitVector> entry(blocks, llvm::BitVector(records.size(), true));
  std::vector<llvm::BitVector> exit = entry;
  auto transfer = [&](size_t block, llvm::BitVector facts) {
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i)
      if (auto found = tags.find(s.ops[i]); found != tags.end())
        facts.set(found->second);
    return facts;
  };
  bool changed;
  do {
    changed = false;
    for (size_t block = 0; block < blocks; ++block) {
      llvm::BitVector incoming(records.size(), block != 0 && reachable[block]);
      if (block != 0 && reachable[block])
        for (size_t predecessor : predecessors[block])
          incoming &= exit[predecessor];
      auto outgoing = transfer(block, incoming);
      if (entry[block] != incoming || exit[block] != outgoing) {
        entry[block] = std::move(incoming);
        exit[block] = std::move(outgoing);
        changed = true;
      }
    }
  } while (changed);
  for (size_t block = 0; block < blocks; ++block) {
    llvm::BitVector facts = entry[block];
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      auto found = tags.find(s.ops[i]);
      if (found == tags.end())
        continue;
      for (int32_t predecessor : records[found->second].after)
        if (!facts.test(predecessor))
          return s.ops[i]->emitOpError("tile contract predecessor is not established on every emitted path: ")
                 << predecessor;
      facts.set(found->second);
    }
  }
  return success();
}
} // namespace

FailureOr<ArrayAttr> mlir::atlas::buildAtlasTileContract(
    func::FuncOp function, ArrayRef<VirtualRegisterAssignment> tensorRegisters,
    ArrayRef<VirtualDMAAssignment> dmaAssignments,
    const FixedResourcePlacement &fixed, ArrayRef<int32_t> scalarArgumentRegs) {
  auto sourceDMA = buildAtlasDMAContract(function, dmaAssignments);
  if (failed(sourceDMA))
    return failure();
  llvm::DenseSet<Value> knownValues;
  for (Block &block : function.getBody()) {
    for (Value argument : block.getArguments())
      knownValues.insert(argument);
    for (Operation &op : block)
      for (Value result : op.getResults())
        knownValues.insert(result);
  }
  llvm::DenseMap<Value, unsigned> registers;
  for (const VirtualRegisterAssignment &assignment : tensorRegisters) {
    if (!assignment.value || !knownValues.contains(assignment.value) ||
        assignment.reg >= 64 ||
        !isa<VirtualBF16Type, VirtualFP8Type>(assignment.value.getType()) ||
        (isa<VirtualBF16Type>(assignment.value.getType()) && assignment.reg % 2) ||
        !registers.try_emplace(assignment.value, assignment.reg).second) {
      function.emitOpError("tile contract requires unique valid tensor register claims");
      return failure();
    }
  }
  llvm::DenseMap<Value, const VirtualDMAAssignment *> transfers;
  llvm::DenseMap<uint32_t, DictionaryAttr> dmaRecords;
  for (const VirtualDMAAssignment &assignment : dmaAssignments)
    transfers[assignment.transfer] = &assignment;
  for (Attribute attr : *sourceDMA) {
    auto record = cast<DictionaryAttr>(attr);
    dmaRecords[uint32_t(record.getAs<IntegerAttr>("id").getInt())] = record;
  }
  auto sourceBase = [&](StringRef name) -> std::optional<uint64_t> {
    auto attr = function->getAttrOfType<IntegerAttr>(name);
    if (!attr || attr.getValue().isNegative() || attr.getValue().getActiveBits() > 32)
      return std::nullopt;
    uint64_t value = attr.getValue().getZExtValue();
    if (value < 0x80000000u || value % 1024)
      return std::nullopt;
    return value;
  };
  auto inputBase = sourceBase("atlas.input_dram_base");
  auto outputBase = sourceBase("atlas.output_dram_base");
  if (!inputBase || !outputBase) {
    function.emitOpError("tile contract requires aligned 32-bit boundary DRAM bases");
    return failure();
  }
  SmallVector<Record> records;
  auto add = [&](StringRef kind, int32_t reg, uint64_t vmem, uint64_t dram,
                 uint32_t bytes, int32_t channel, int32_t transfer,
                 ArrayRef<int32_t> after = {}) -> FailureOr<int32_t> {
    if (vmem > std::numeric_limits<uint32_t>::max() ||
        dram > std::numeric_limits<uint32_t>::max() ||
        records.size() > uint64_t(std::numeric_limits<int32_t>::max())) {
      function.emitOpError("tile contract source transfer address or id overflows its supported range");
      return failure();
    }
    Record r{int32_t(records.size()), kind, reg, uint32_t(vmem), uint32_t(dram),
             bytes, channel, transfer, {}};
    r.after.append(after.begin(), after.end());
    if (!validRecord(r)) {
      function.emitOpError("tile contract source transfer has invalid geometry or placement");
      return failure();
    }
    records.push_back(std::move(r));
    return int32_t(records.size() - 1);
  };
  auto addWait = [&](int32_t launch, int32_t channel, int32_t transfer) {
    return add("dma_wait", -1, 0, 0, 0, channel, transfer, {launch});
  };
  if (scalarArgumentRegs.size() != function.getNumArguments()) {
    function.emitOpError("tile contract requires one scalar register claim per entry argument");
    return failure();
  }
  if (function.getNumArguments()) {
    auto controlBase = sourceBase("atlas.control_dram_base");
    if (!controlBase || function.getNumArguments() > 256) {
      function.emitOpError("tile contract requires an aligned bounded scalar mailbox");
      return failure();
    }
    auto launch = add("dma_load", -1, uint64_t(fixed.mailboxWord) * 4,
                      *controlBase, 1024, fixed.loadChannel, -1);
    if (failed(launch)) {
      function.emitOpError("tile contract has invalid mailbox transfer geometry");
      return failure();
    }
    auto wait = addWait(*launch, fixed.loadChannel, -1);
    if (failed(wait))
      return failure();
    for (auto [index, reg] : llvm::enumerate(scalarArgumentRegs))
      if (failed(add("mailbox_load", reg, (uint64_t(fixed.mailboxWord) + index) * 4,
                     0, 4, -1, -1, {*wait}))) {
        function.emitOpError("tile contract has invalid scalar argument placement");
        return failure();
      }
  }
  llvm::DenseMap<Value, int32_t> launches;
  for (Block &block : function.getBody()) {
    for (Operation &op : block) {
      auto tensorReg = [&](Value value) -> FailureOr<unsigned> {
        auto found = registers.find(value);
        if (found == registers.end()) {
          op.emitOpError("tile contract has no tensor register placement for source value");
          return failure();
        }
        return found->second;
      };
      if (isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(op)) {
        bool output = isa<VirtualOutputBF16Op>(op);
        bool bf16 = !isa<VirtualInputFP8Op>(op);
        Value value;
        if (output)
          value = cast<VirtualOutputBF16Op>(op).getValue();
        else
          value = op.getResult(1);
        auto reg = tensorReg(value);
        auto indexAttr = op.getAttrOfType<IntegerAttr>("index");
        if (failed(reg) || !indexAttr || indexAttr.getValue().isNegative() ||
            indexAttr.getValue().getActiveBits() > 32)
          return failure();
        uint64_t index = indexAttr.getValue().getZExtValue();
        uint64_t window = output ? fixed.outputWindowWords : fixed.inputWindowWords;
        if (index >= window / 512) {
          op.emitOpError("tile contract boundary index exceeds its fixed VMEM window");
          return failure();
        }
        for (unsigned half = 0; half < (bf16 ? 2u : 1u); ++half) {
          uint64_t vmem = (uint64_t(output ? fixed.outputWord : fixed.inputWord) +
                           index * 512 + half * 256) * 4;
          uint64_t dram = *(output ? outputBase : inputBase) + index * 2048 + half * 1024;
          unsigned channel = output ? fixed.storeChannel : fixed.loadChannel;
          if (output) {
            auto store = add("vstore", *reg + half, vmem, 0, 1024, -1, -1);
            if (failed(store))
              return failure();
            auto launch = add("dma_store", -1, vmem, dram, 1024, channel, -1, {*store});
            if (failed(launch) || failed(addWait(*launch, channel, -1)))
              return failure();
          } else {
            auto launch = add("dma_load", -1, vmem, dram, 1024, channel, -1);
            if (failed(launch))
              return failure();
            auto wait = addWait(*launch, channel, -1);
            if (failed(wait) || failed(add("vload", *reg + half, vmem, 0, 1024, -1, -1, {*wait})))
              return failure();
          }
        }
      } else if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
                     VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
        Value handle = op.getResult(1);
        const auto &p = transfers.find(handle)->second->placement;
        DictionaryAttr source = dmaRecords.find(p.id)->second;
        bool store = isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op);
        SmallVector<int32_t, 2> after;
        if (store) {
          auto reg = tensorReg(op.getOperand(1));
          if (failed(reg))
            return failure();
          for (unsigned half = 0; half < p.halves; ++half) {
            auto id = add("vstore", *reg + half,
                          (uint64_t(p.stagingWord) + half * 256) * 4,
                          0, 1024, -1, p.id);
            if (failed(id))
              return failure();
            after.push_back(*id);
          }
        }
        auto launch = add(store ? "dma_store" : "dma_load", -1,
                          uint64_t(p.stagingWord) * 4,
                          uint32_t(source.getAs<IntegerAttr>("dram_byte").getInt()),
                          uint32_t(source.getAs<IntegerAttr>("size_bytes").getInt()),
                          p.channel, p.id, after);
        if (failed(launch))
          return failure();
        launches[handle] = *launch;
      } else if (isa<VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op, VirtualDMAWaitOp>(op)) {
        Value handle = op.getOperand(1);
        auto assignment = transfers.find(handle);
        auto launch = launches.find(handle);
        if (assignment == transfers.end() || launch == launches.end()) {
          op.emitOpError("tile contract completion has no source launch");
          return failure();
        }
        const auto &p = assignment->second->placement;
        auto wait = addWait(launch->second, p.channel, p.id);
        if (failed(wait))
          return failure();
        if (!isa<VirtualDMAWaitOp>(op)) {
          auto reg = tensorReg(op.getResult(1));
          if (failed(reg))
            return failure();
          for (unsigned half = 0; half < p.halves; ++half)
            if (failed(add("vload", *reg + half,
                           (uint64_t(p.stagingWord) + half * 256) * 4,
                           0, 1024, -1, p.id, {*wait})))
              return failure();
        }
      } else if (auto pack = dyn_cast<VirtualPackFP8Op>(op)) {
        auto reg = tensorReg(pack.getResult());
        if (failed(reg))
          return failure();
        auto store = add("vstore", *reg, uint64_t(fixed.packWord) * 4,
                         0, 1024, -1, -1);
        if (failed(store) || failed(add("vload", *reg,
                                      uint64_t(fixed.packRelayoutWord) * 4,
                                      0, 1024, -1, -1, {*store})))
          return failure();
      }
    }
  }
  Builder builder(function.getContext());
  SmallVector<Attribute> encoded;
  for (const Record &r : records)
    encoded.push_back(encodeRecord(builder, r));
  return builder.getArrayAttr(encoded);
}

LogicalResult mlir::atlas::verifyAtlasGeneratedTileContract(
    const AtlasVerificationContext &ctx) {
  ModuleOp module = ctx.module;
  Attribute marker = module->getAttr(kMarker);
  Attribute contract = module->getAttr(kContract);
  bool tagged = false;
  for (Operation &op : module.getBody()->getOperations())
    tagged |= bool(op.getAttr(kCommand));
  auto version = dyn_cast_or_null<StringAttr>(marker);
  if (!version || (version.getValue() != kVersion && version.getValue() != "resource-contract-v3")) {
    if (contract || tagged)
      return module.emitOpError("tile metadata requires generated marker resource-contract-v2/v3");
    if (isa_and_nonnull<UnitAttr>(marker) ||
        (version && (version.getValue() == "dma-contract-v1" ||
                     version.getValue() == "resource-contract-v1")))
      return success();
    return module.emitOpError("expected generated marker resource-contract-v2/v3 or supported legacy marker");
  }
  auto array = dyn_cast_or_null<ArrayAttr>(contract);
  if (!array)
    return module.emitOpError("generated resource contract requires an atlas.virtual_tile_contract array");
  SmallVector<Record> records;
  for (Attribute attr : array) {
    auto record = parseRecord(module, attr);
    if (failed(record))
      return failure();
    if (record->id != int32_t(records.size()))
      return module.emitOpError("tile contract ids must be contiguous in source expansion order");
    llvm::BitVector unique(records.size());
    for (int32_t predecessor : record->after) {
      if (predecessor < 0 || predecessor >= record->id || unique.test(predecessor))
        return module.emitOpError("tile contract predecessor ids must be unique earlier commands");
      unique.set(predecessor);
    }
    records.push_back(std::move(*record));
  }
  if (failed(checkReferences(module, records)))
    return failure();
  llvm::DenseMap<Operation *, int32_t> tags;
  llvm::BitVector seen(records.size());
  for (Operation &op : module.getBody()->getOperations()) {
    Attribute attr = op.getAttr(kCommand);
    bool required = isa<VLoadOp, VStoreOp, DMAOp, DMAWaitOp>(op);
    if (!attr) {
      if (required)
        return op.emitOpError("tile contract requires a command tag on every generated tile transfer");
      continue;
    }
    auto id = dyn_cast<IntegerAttr>(attr);
    if (!id || !id.getType().isSignlessInteger(32) || id.getValue().isNegative() ||
        id.getValue().getZExtValue() >= records.size())
      return op.emitOpError("tile contract command tag requires a known nonnegative i32 id");
    int32_t value = int32_t(id.getValue().getSExtValue());
    if (seen.test(value))
      return op.emitOpError("tile contract has duplicate command id ") << value;
    seen.set(value);
    const Record &r = records[value];
    bool matches = false;
    if (auto load = dyn_cast<VLoadOp>(op))
      matches = r.kind == "vload" && load.getDst() == uint32_t(r.reg) && load.getFormat() == "raw";
    else if (auto store = dyn_cast<VStoreOp>(op))
      matches = r.kind == "vstore" && store.getSrc() == uint32_t(r.reg) && store.getFormat() == "raw";
    else if (auto dma = dyn_cast<DMAOp>(op))
      matches = r.kind == (dma.getDirection() == "load" ? "dma_load" : "dma_store") &&
                dma.getChannel() == uint32_t(r.channel);
    else if (auto wait = dyn_cast<DMAWaitOp>(op))
      matches = r.kind == "dma_wait" && wait.getChannel() == uint32_t(r.channel);
    else if (auto load = dyn_cast<ScalarLoadOp>(op))
      matches = r.kind == "mailbox_load" && load.getKind() == "lw" && load.getDst() == uint32_t(r.reg);
    if (!matches)
      return op.emitOpError("tile contract command kind, format, register, or channel mismatch");
    if (isa<DMAOp, DMAWaitOp>(op)) {
      auto transfer = op.getAttrOfType<IntegerAttr>("atlas.virtual_dma_transfer");
      if ((r.transfer == -1 && transfer) ||
          (r.transfer >= 0 && (!transfer || !transfer.getType().isSignlessInteger(32) ||
                              transfer.getInt() != r.transfer)))
        return op.emitOpError("tile contract explicit transfer identity mismatch");
    }
    tags[&op] = value;
  }
  if (seen.count() != records.size())
    return module.emitOpError("tile contract requires exactly one command for every source record");
  if (tags.empty())
    return success();
  assert(ctx.stream && "a tagged tile stream is decoded");
  const AtlasStream &s = *ctx.stream;
  if (failed(checkDependencies(s, records, tags)))
    return failure();
  const auto &bases = ctx.dmaUpperEntry;
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues regs = s.entry[block];
    auto upper = bases[block];
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      Operation *op = s.ops[i];
      if (auto config = dyn_cast<DMAConfigOp>(op))
        upper = regs[config.getBaseReg()];
      if (auto found = tags.find(op); found != tags.end()) {
        const Record &r = records[found->second];
        if (isVector(r)) {
          unsigned base;
          int64_t offset;
          if (auto load = dyn_cast<VLoadOp>(op)) {
            base = load.getBase();
            offset = load.getOffsetAttr().getValue().getSExtValue();
          } else {
            auto store = cast<VStoreOp>(op);
            base = store.getBase();
            offset = store.getOffsetAttr().getValue().getSExtValue();
          }
          std::optional<uint64_t> address;
          if (regs[base]) {
            // The scalar address ALU wraps at RV32 width. Compare its full
            // result before the LSU masks address bits or drops low words.
            uint32_t word = uint32_t(int64_t(*regs[base]) + offset * 32);
            address = uint64_t(word) * 4;
          }
          if (failed(checkValue(op, "vector VMEM byte address", address, r.vmemByte)))
            return failure();
        } else if (isDMA(r)) {
          auto dma = cast<DMAOp>(op);
          std::optional<uint64_t> address;
          if (regs[dma.getReg()])
            address = uint64_t(*regs[dma.getReg()]) * 4;
          if (failed(checkValue(op, "DRAM upper word", upper, 0)) ||
              failed(checkValue(op, "DMA VMEM byte address", address, r.vmemByte)) ||
              failed(checkValue(op, "DRAM byte address", regs[dma.getDram()], r.dramByte)) ||
              failed(checkValue(op, "byte length", regs[dma.getSize()], r.bytes)))
            return failure();
        } else if (r.kind == "mailbox_load") {
          auto load = cast<ScalarLoadOp>(op);
          std::optional<uint64_t> address;
          if (regs[load.getBase()])
            address = uint32_t(int64_t(*regs[load.getBase()]) +
                               load.getOffsetAttr().getValue().getSExtValue());
          if (failed(checkValue(op, "mailbox VMEM byte address", address, r.vmemByte)))
            return failure();
        }
      }
      applyScalar(s.instrs[i], regs);
    }
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasGeneratedTileContract(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasGeneratedTileContract(*ctx);
}
