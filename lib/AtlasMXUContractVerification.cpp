#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <deque>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr llvm::StringLiteral kMarker = "atlas.generated_from_virtual";
constexpr llvm::StringLiteral kContract = "atlas.virtual_mxu_contract";
constexpr llvm::StringLiteral kCommand = "atlas.virtual_mxu_command";

struct Record {
  int32_t id = -1, block = -1;
  StringRef kind;
  int32_t unit = -1, reg = -1, slot = -1, weightSlot = -1;
  int32_t weight = -1, previous = -1, scaleReg = -1, scale = -1;
};

bool isWeight(const Record &r) { return r.kind == "weight_fp8"; }
bool isSeed(const Record &r) { return r.kind == "acc_fp8" || r.kind == "acc_bf16"; }
bool isMatmul(const Record &r) { return r.kind == "reset" || r.kind == "accumulate"; }
bool isPop(const Record &r) { return r.kind == "pop_bf16" || r.kind == "pop_fp8"; }
bool isAccumulator(const Record &r) { return isSeed(r) || isMatmul(r); }

bool validRecord(const Record &r) {
  if (r.id < 0 || r.block < 0 || r.unit < 0 || r.unit >= 2 ||
      r.reg < 0 || r.reg >= 64 || r.slot < 0 || r.slot >= 2)
    return false;
  if ((r.kind == "acc_bf16" || r.kind == "pop_bf16") && r.reg % 2)
    return false;
  if (isMatmul(r)) {
    if (r.weightSlot < 0 || r.weightSlot >= 2 || r.weight < 0 ||
        (r.kind == "reset" ? r.previous != -1 : r.previous < 0))
      return false;
  } else if (r.weightSlot != -1 || r.weight != -1 ||
             (isPop(r) ? r.previous < 0 : r.previous != -1)) {
    return false;
  }
  if (r.kind == "pop_fp8")
    return r.scaleReg >= 0 && r.scaleReg < 32 && r.scale >= 0 && r.scale <= 255;
  return (isWeight(r) || isSeed(r) || isMatmul(r) || r.kind == "pop_bf16") &&
         r.scaleReg == (r.kind == "pop_bf16" ? 0 : -1) && r.scale == -1;
}

DictionaryAttr encodeRecord(Builder &builder, const Record &r) {
  auto field = [&](StringRef name, int32_t value) {
    return builder.getNamedAttr(name, builder.getI32IntegerAttr(value));
  };
  return builder.getDictionaryAttr({
      field("id", r.id), field("block", r.block),
      builder.getNamedAttr("kind", builder.getStringAttr(r.kind)),
      field("unit", r.unit), field("reg", r.reg), field("slot", r.slot),
      field("weight_slot", r.weightSlot), field("weight", r.weight),
      field("previous", r.previous), field("scale_reg", r.scaleReg),
      field("scale", r.scale)});
}

FailureOr<Record> parseRecord(ModuleOp module, Attribute attr) {
  auto dictionary = dyn_cast<DictionaryAttr>(attr);
  if (!dictionary || dictionary.size() != 11) {
    module.emitOpError("MXU contract record requires exactly the v1 fields");
    return failure();
  }
  Record r;
  for (auto [name, field] : {
           std::pair<StringRef, int32_t *>("id", &r.id),
           {"block", &r.block}, {"unit", &r.unit}, {"reg", &r.reg},
           {"slot", &r.slot}, {"weight_slot", &r.weightSlot},
           {"weight", &r.weight}, {"previous", &r.previous},
           {"scale_reg", &r.scaleReg}, {"scale", &r.scale}}) {
    auto integer = dyn_cast_or_null<IntegerAttr>(dictionary.get(name));
    if (!integer || !integer.getType().isSignlessInteger(32)) {
      module.emitOpError("MXU contract field requires signless i32: ") << name;
      return failure();
    }
    *field = int32_t(integer.getValue().getSExtValue());
  }
  auto kind = dictionary.getAs<StringAttr>("kind");
  if (!kind) {
    module.emitOpError("MXU contract kind requires a string");
    return failure();
  }
  r.kind = kind.getValue();
  if (!validRecord(r)) {
    module.emitOpError("MXU contract record has invalid v1 fields or geometry");
    return failure();
  }
  return r;
}

LogicalResult checkReferences(ModuleOp module, ArrayRef<Record> records) {
  for (const Record &r : records) {
    auto producer = [&](int32_t id) -> const Record * {
      if (id < 0 || id >= r.id || records[id].block != r.block)
        return nullptr;
      return &records[id];
    };
    if (isMatmul(r)) {
      const Record *weight = producer(r.weight);
      if (!weight || !isWeight(*weight) || weight->unit != r.unit ||
          weight->slot != r.weightSlot)
        return module.emitOpError("MXU contract weight must reference its source-block producer and placement");
    }
    if (r.previous >= 0) {
      const Record *previous = producer(r.previous);
      if (!previous || !isAccumulator(*previous) || previous->unit != r.unit ||
          previous->slot != r.slot)
        return module.emitOpError("MXU contract previous version must reference its source-block producer and placement");
    }
  }
  return success();
}

using Owners = std::array<std::array<int32_t, 2>, 2>;
Owners emptyOwners() { return {{{-1, -1}, {-1, -1}}}; }

struct BlockOwnership {
  Owners weights = emptyOwners(), accumulators = emptyOwners();
  llvm::DenseMap<int32_t, unsigned> remaining;
};

LogicalResult checkOwnership(ModuleOp module, ArrayRef<Record> records,
                             ArrayRef<int32_t> order) {
  llvm::DenseMap<int32_t, BlockOwnership> blocks;
  for (const Record &r : records)
    if (isMatmul(r))
      ++blocks[r.block].remaining[r.weight];
  for (int32_t id : order) {
    const Record &r = records[id];
    BlockOwnership &block = blocks[r.block];
    int32_t &weightOwner = block.weights[r.unit][isMatmul(r) ? r.weightSlot : r.slot];
    int32_t &accOwner = block.accumulators[r.unit][r.slot];
    auto error = [&](StringRef message) -> LogicalResult {
      return module.emitOpError("MXU contract ") << message << " at command " << r.id;
    };
    if (isWeight(r)) {
      if (weightOwner != -1)
        return error("overwrites a logically live weight slot");
      weightOwner = block.remaining.lookup(r.id) ? r.id : -1;
    } else if (isSeed(r)) {
      if (accOwner != -1)
        return error("overwrites a logically live accumulator slot");
      accOwner = r.id;
    } else if (isMatmul(r)) {
      if (weightOwner != r.weight || !block.remaining.lookup(r.weight))
        return error("requires the current weight owner");
      if (r.kind == "reset" ? accOwner != -1 : accOwner != r.previous)
        return error("requires the current accumulator version or a free reset slot");
      if (--block.remaining[r.weight] == 0)
        weightOwner = -1;
      accOwner = r.id;
    } else {
      if (accOwner != r.previous)
        return error("readout requires the current accumulator version");
      accOwner = -1;
    }
  }
  for (const auto &entry : blocks)
    for (const auto &unit : entry.second.accumulators)
      for (int32_t owner : unit)
        if (owner != -1)
          return module.emitOpError("MXU contract accumulator remains live at source-block exit");
  return success();
}

// The emitted stream has one physical state, irrespective of source-block ids.
// Unknown owners retain possible liveness. Freshness intersects incoming paths,
// while pending consumers unite them so skipped uses cannot hide live weights.
struct PathOwnership {
  Owners weights = emptyOwners(), accumulators = emptyOwners();
  SmallVector<bool> fresh, pending;

  explicit PathOwnership(size_t records) : fresh(records, false), pending(records, false) {}
};

bool mergeOwnership(PathOwnership &into, const PathOwnership &from) {
  bool changed = false;
  auto mergeOwners = [&](Owners &owners, const Owners &incoming) {
    for (unsigned unit = 0; unit < 2; ++unit)
      for (unsigned slot = 0; slot < 2; ++slot)
        if (owners[unit][slot] != -2 && owners[unit][slot] != incoming[unit][slot]) {
          owners[unit][slot] = -2;
          changed = true;
        }
  };
  mergeOwners(into.weights, from.weights);
  mergeOwners(into.accumulators, from.accumulators);
  for (size_t i = 0; i < into.fresh.size(); ++i) {
    if (into.fresh[i] && !from.fresh[i]) {
      into.fresh[i] = false;
      changed = true;
    }
    if (!into.pending[i] && from.pending[i]) {
      into.pending[i] = true;
      changed = true;
    }
  }
  return changed;
}

LogicalResult checkPathOwnership(ModuleOp module, ArrayRef<Record> records,
                                 const AtlasStream &stream,
                                 const llvm::DenseMap<Operation *, int32_t> &tagged) {
  if (stream.starts.empty())
    return success();
  SmallVector<SmallVector<int32_t>> consumers(records.size());
  for (const Record &r : records)
    if (isMatmul(r))
      consumers[r.weight].push_back(r.id);
  auto transfer = [&](size_t block, PathOwnership &state, bool diagnose) -> LogicalResult {
    for (size_t i = stream.starts[block]; i < stream.blockEnd(block); ++i) {
      auto found = tagged.find(stream.ops[i]);
      if (found == tagged.end())
        continue;
      const Record &r = records[found->second];
      int32_t &weight = state.weights[r.unit][isMatmul(r) ? r.weightSlot : r.slot];
      int32_t &acc = state.accumulators[r.unit][r.slot];
      auto error = [&](StringRef message) -> LogicalResult {
        return stream.ops[i]->emitOpError("MXU contract emitted path ") << message << " at command " << r.id;
      };
      if (isWeight(r)) {
        if (diagnose && weight != -1)
          return error("overwrites a logically live weight slot");
        weight = consumers[r.id].empty() ? -1 : r.id;
        // A repeated consumer needs a producer execution since its prior use;
        // distinct consumers may share one producer within the same iteration.
        for (int32_t consumer : consumers[r.id]) {
          state.fresh[consumer] = true;
          state.pending[consumer] = true;
        }
      } else if (isSeed(r)) {
        if (diagnose && acc != -1)
          return error("overwrites a logically live accumulator slot");
        acc = r.id;
      } else if (isMatmul(r)) {
        if (diagnose && (weight != r.weight || !state.fresh[r.id]))
          return error("requires a fresh current weight owner on every incoming path");
        if (diagnose && (r.kind == "reset" ? acc != -1 : acc != r.previous))
          return error("requires the current accumulator version or a free reset slot on every incoming path");
        state.fresh[r.id] = state.pending[r.id] = false;
        if (llvm::none_of(consumers[r.weight], [&](int32_t id) { return state.pending[id]; }))
          weight = -1;
        acc = r.id;
      } else {
        if (diagnose && acc != r.previous)
          return error("readout requires the current accumulator version on every incoming path");
        acc = -1;
      }
    }
    return success();
  };
  std::vector<PathOwnership> entries(stream.starts.size(), PathOwnership(records.size()));
  std::vector<bool> reached(stream.starts.size(), false);
  std::deque<size_t> work = {0};
  reached[0] = true;
  // Entry owners only become unknown; must-fresh bits only become false and
  // may-pending bits only become true. Producer writes make loop transfer finite.
  while (!work.empty()) {
    size_t block = work.front();
    work.pop_front();
    PathOwnership state = entries[block];
    (void)transfer(block, state, false);
    for (size_t next : stream.succs[block]) {
      if (!reached[next]) {
        entries[next] = state;
        reached[next] = true;
        work.push_back(next);
      } else if (mergeOwnership(entries[next], state)) {
        work.push_back(next);
      }
    }
  }
  for (size_t block = 0; block < stream.starts.size(); ++block) {
    if (!reached[block])
      continue;
    PathOwnership state = entries[block];
    if (failed(transfer(block, state, true)))
      return failure();
    bool exits = stream.succs[block].empty();
    if (stream.endsInBranch(block)) {
      Operation *redirect = stream.ops[stream.blockEnd(block) - 2];
      exits |= !stream.targetOf.lookup(redirect);
      exits |= isa<BranchOp>(redirect) && stream.blockEnd(block) == stream.ops.size();
    }
    if (exits)
      for (const auto &unit : state.accumulators)
        for (int32_t owner : unit)
          if (owner != -1)
            return module.emitOpError("MXU contract accumulator remains live at emitted path exit");
  }
  return success();
}

LogicalResult matchCommand(Operation &op, const Record &r) {
  bool matches = false;
  if (auto push = dyn_cast<MXUPushOp>(op))
    matches = (isWeight(r) || isSeed(r)) && push.getKind() == r.kind &&
              push.getUnit() == r.unit && push.getSrc() == r.reg && push.getSlot() == r.slot;
  else if (auto matmul = dyn_cast<MXUMatmulOp>(op))
    matches = isMatmul(r) && matmul.getUnit() == r.unit && matmul.getSrc() == r.reg &&
              matmul.getWeightSlot() == r.weightSlot && matmul.getAccSlot() == r.slot &&
              matmul.getAccumulate() == (r.kind == "accumulate");
  else if (auto pop = dyn_cast<MXUPopOp>(op))
    matches = isPop(r) && pop.getFormat() == (r.kind == "pop_fp8" ? "fp8" : "bf16") &&
              pop.getUnit() == r.unit && pop.getDst() == r.reg && pop.getSlot() == r.slot &&
              pop.getScaleReg() == r.scaleReg;
  if (!matches)
    return op.emitOpError("MXU contract command kind, unit, register, slot, mode, or scale register mismatch for id ") << r.id;
  return success();
}
} // namespace

FailureOr<ArrayAttr> mlir::atlas::buildAtlasMXUContract(
    func::FuncOp function, ArrayRef<VirtualRegisterAssignment> registers,
    ArrayRef<VirtualMXUAssignment> mxuAssignments,
    const FixedResourcePlacement &fixed) {
  if (failed(verifyAtlasMXUAllocation(function, mxuAssignments, fixed)))
    return failure();
  llvm::DenseSet<Value> known;
  for (Block &block : function.getBody()) {
    for (Value argument : block.getArguments())
      known.insert(argument);
    for (Operation &op : block)
      for (Value result : op.getResults())
        known.insert(result);
  }
  llvm::DenseMap<Value, unsigned> mapped;
  for (const VirtualRegisterAssignment &assignment : registers) {
    Value value = assignment.value;
    if (!value || !known.contains(value) ||
        !isa<VirtualFP8Type, VirtualBF16Type>(value.getType())) {
      function.emitOpError("MXU contract tensor assignment refers to a foreign, stale, or non-tensor value");
      return failure();
    }
    bool pair = isa<VirtualBF16Type>(value.getType());
    if (mapped.contains(value) || assignment.reg >= 64 || (pair && assignment.reg % 2)) {
      function.emitOpError("MXU contract tensor assignment has duplicate value or invalid register geometry");
      return failure();
    }
    mapped[value] = assignment.reg;
  }
  llvm::DenseMap<Value, MXUPlacement> placements;
  for (const VirtualMXUAssignment &assignment : mxuAssignments)
    placements[assignment.handle] = assignment.placement;
  llvm::DenseMap<Value, int32_t> producer;
  SmallVector<Record> records;
  auto append = [&](Record r, Value tensor, Value handle = {}) -> LogicalResult {
    auto found = mapped.find(tensor);
    if (found == mapped.end())
      return tensor.getParentBlock()->getParentOp()->emitOpError("MXU contract missing source tensor assignment");
    r.reg = found->second;
    r.id = records.size();
    if (!validRecord(r))
      return function.emitOpError("MXU contract source has unsupported fields or geometry");
    records.push_back(r);
    if (handle)
      producer[handle] = r.id;
    return success();
  };
  int32_t blockId = 0;
  for (Block &block : function.getBody()) {
    for (Operation &op : block) {
      Record r;
      r.block = blockId;
      Value tensor, handle;
      if (isa<VirtualMXULoadWeightOp, VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
        handle = op.getResult(1);
        MXUPlacement placement = placements.lookup(handle);
        r.unit = placement.unit;
        r.slot = placement.slot;
        r.kind = isa<VirtualMXULoadWeightOp>(op) ? "weight_fp8" :
                 isa<VirtualMXULoadAccFP8Op>(op) ? "acc_fp8" : "acc_bf16";
        tensor = op.getOperand(1);
      } else if (isa<VirtualMXUResetOp, VirtualMXUAccumulateOp>(op)) {
        handle = op.getResult(1);
        Value weight = op.getOperand(2);
        MXUPlacement placement = placements.lookup(handle);
        r.unit = placement.unit;
        r.slot = placement.slot;
        r.weightSlot = placements.lookup(weight).slot;
        r.weight = producer.lookup(weight);
        r.kind = isa<VirtualMXUResetOp>(op) ? "reset" : "accumulate";
        if (r.kind == "accumulate")
          r.previous = producer.lookup(op.getOperand(3));
        tensor = op.getOperand(1);
      } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
        Value acc = op.getOperand(1);
        MXUPlacement placement = placements.lookup(acc);
        r.unit = placement.unit;
        r.slot = placement.slot;
        r.previous = producer.lookup(acc);
        tensor = op.getResult(1);
        r.kind = isa<VirtualMXUReadoutBF16Op>(op) ? "pop_bf16" : "pop_fp8";
        r.scaleReg = 0;
        if (r.kind == "pop_fp8") {
          auto scale = op.getOperand(2).getDefiningOp<VirtualScaleConstantOp>();
          if (!scale) {
            op.emitOpError("MXU contract FP8 readout requires a source scale constant");
            return failure();
          }
          r.scaleReg = fixed.scaleReg;
          r.scale = scale.getCode();
        }
      } else if (auto legacy = dyn_cast<VirtualMXUMatmulOp>(op)) {
        r.unit = legacy.getUnit();
        r.kind = "weight_fp8";
        r.slot = fixed.mxuWeightSlot;
        if (failed(append(r, legacy.getWeight())))
          return failure();
        r.kind = "reset";
        r.weight = records.back().id;
        r.weightSlot = fixed.mxuWeightSlot;
        r.slot = fixed.mxuAccSlot;
        if (failed(append(r, legacy.getActivation())))
          return failure();
        r.kind = "pop_bf16";
        r.previous = records.back().id;
        r.weight = r.weightSlot = -1;
        r.scaleReg = 0;
        if (failed(append(r, legacy.getResult())))
          return failure();
        continue;
      } else {
        continue;
      }
      if (failed(append(r, tensor, handle)))
        return failure();
    }
    ++blockId;
  }
  Builder builder(function.getContext());
  SmallVector<Attribute> encoded;
  for (const Record &r : records)
    encoded.push_back(encodeRecord(builder, r));
  return builder.getArrayAttr(encoded);
}

LogicalResult mlir::atlas::verifyAtlasGeneratedMXUContract(ModuleOp module) {
  Attribute marker = module->getAttr(kMarker), contract = module->getAttr(kContract);
  auto version = dyn_cast_or_null<StringAttr>(marker);
  bool legacy = isa_and_nonnull<UnitAttr>(marker) ||
                (version && version.getValue() == "dma-contract-v1");
  if (legacy) {
    if (contract)
      return module.emitOpError("legacy generated marker cannot carry an MXU contract");
    for (Operation &op : module.getBody()->getOperations())
      if (op.hasAttr(kCommand))
        return op.emitOpError("legacy generated marker cannot carry MXU command tags");
    return success();
  }
  if (!version || (version.getValue() != "resource-contract-v1" && version.getValue() != "resource-contract-v2"))
    return module.emitOpError("expected resource-contract-v1/v2 or a legacy generated marker");
  auto array = dyn_cast_or_null<ArrayAttr>(contract);
  if (!array || !isa_and_nonnull<ArrayAttr>(module->getAttr("atlas.virtual_dma_contract")))
    return module.emitOpError("generated resource contract requires DMA and MXU contract arrays");
  SmallVector<Record> records;
  for (Attribute attr : array) {
    auto record = parseRecord(module, attr);
    if (failed(record))
      return failure();
    if (record->id != int64_t(records.size()) ||
        (!records.empty() && record->block < records.back().block))
      return module.emitOpError("MXU contract ids must be consecutive and source blocks ordered");
    records.push_back(*record);
  }
  if (failed(checkReferences(module, records)))
    return failure();
  SmallVector<int32_t> sourceOrder;
  for (const Record &r : records)
    sourceOrder.push_back(r.id);
  if (failed(checkOwnership(module, records, sourceOrder)))
    return failure();
  llvm::DenseMap<Operation *, int32_t> tagged;
  llvm::DenseSet<int32_t> seen;
  for (Operation &op : module.getBody()->getOperations()) {
    Attribute attr = op.getAttr(kCommand);
    bool command = isa<MXUPushOp, MXUMatmulOp, MXUPopOp>(op);
    if (!attr && !command)
      continue;
    auto id = dyn_cast_or_null<IntegerAttr>(attr);
    if (!command || !id || !id.getType().isSignlessInteger(32) || id.getValue().isNegative())
      return op.emitOpError("MXU contract requires a nonnegative i32 command tag on every MXU command only");
    uint64_t value = id.getValue().getZExtValue();
    if (value >= records.size())
      return op.emitOpError("MXU contract has no source record for command id ") << value;
    if (!seen.insert(value).second)
      return op.emitOpError("MXU contract has duplicate command id ") << value;
    if (failed(matchCommand(op, records[value])))
      return failure();
    tagged[&op] = value;
  }
  if (seen.size() != records.size())
    return module.emitOpError("MXU contract requires exactly one issued command for every source record");
  if (tagged.empty())
    return success();
  auto stream = readAtlasStream(module, AtlasStreamReadMode::Verification);
  if (failed(stream))
    return failure();
  if (failed(checkPathOwnership(module, records, *stream, tagged)))
    return failure();
  llvm::DenseSet<int32_t> readoutScaleRegs;
  for (const Record &r : records)
    if (r.kind == "pop_fp8")
      readoutScaleRegs.insert(r.scaleReg);
  // SELD writes at response arrival, so a later SELI does not prove restoration
  // without independent completion rules. Reject these writes anywhere in the
  // stream, including after readout; current virtual lowering uses only SELI.
  for (size_t i = 0; i < stream->instrs.size(); ++i)
    if (stream->instrs[i].op->opClass == timing::OpClass::ScaleLoad &&
        readoutScaleRegs.contains(stream->instrs[i].rd))
      return stream->ops[i]->emitOpError("MXU contract cannot prove FP8 scale contents with SELD writes to its scale register");
  auto scales = atlasScaleRegisterEntries(*stream);
  for (size_t block = 0; block < stream->starts.size(); ++block) {
    timing::RegValues regs = scales[block];
    for (size_t i = stream->starts[block]; i < stream->blockEnd(block); ++i) {
      auto found = tagged.find(stream->ops[i]);
      if (found != tagged.end()) {
        const Record &r = records[found->second];
        if (r.kind == "pop_fp8" && (!regs[r.scaleReg] || *regs[r.scaleReg] != uint32_t(r.scale)))
          return stream->ops[i]->emitOpError("MXU contract cannot prove the source FP8 scale code at readout");
      }
      applyAtlasScaleRegister(stream->instrs[i], regs);
    }
  }
  return success();
}
