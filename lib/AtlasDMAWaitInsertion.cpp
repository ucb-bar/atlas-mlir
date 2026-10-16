// DMA wait insertion: the last step from virtual to physical assembly.
// Virtual DMA is atomic; lowering emits each transfer's launch and leaves its
// completion to this pass. A transfer stays in flight until the first
// instruction that needs it complete:
//   - a VLOAD or VSTORE touching its staging window (its own staging read, or
//     a later transfer reusing the window);
//   - a launch on its channel, or one whose DRAM bytes overlap where either
//     side writes;
//   - a helper DMA (boundary tile or mailbox), or any instruction the
//     generated interval policy does not admit beside a pending transfer
//     (canOverlapAtlasGeneratedDMA): scalar memory outside a PACK relayout
//     loop, redirects, CSR, the halt;
//   - a PACK relayout loop whose VMEM meets its staging window;
//   - entry to any other branch target, so the wait stays on the fall-through
//     path.
// The wait goes immediately before that instruction. Timing passes may still
// move it earlier within its block.

#include "Atlas/AtlasDMAWaitInsertion.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;

namespace {
struct Span {
  uint64_t begin = 0, end = 0;
  bool overlaps(const Span &other) const {
    return begin < other.end && other.begin < end;
  }
};

// One explicit transfer, from the DMA contract.
struct Transfer {
  int32_t id = -1;
  unsigned channel = 0;
  Span staging, dram;
  bool store = false;
  // Its DMA.WAIT in the tile contract and that command's source owner.
  int32_t waitCommand = -1, ownerOperation = -1, ownerBlock = -1;
};

struct TileRecord {
  StringRef kind;
  Span vmem;
  int32_t transfer = -1;
};

FailureOr<llvm::DenseMap<int32_t, Transfer>> readTransfers(ModuleOp module) {
  auto dma = dyn_cast_or_null<ArrayAttr>(module->getAttr(kAtlasDMAContract));
  auto tiles = dyn_cast_or_null<ArrayAttr>(module->getAttr(kAtlasTileContract));
  auto cfg = dyn_cast_or_null<DictionaryAttr>(module->getAttr(kAtlasCFGContract));
  if (!dma || !tiles || !cfg)
    return module.emitOpError("DMA wait insertion requires the DMA, tile and CFG contracts");
  llvm::DenseMap<int32_t, Transfer> transfers;
  for (Attribute attr : dma) {
    auto d = dyn_cast<DictionaryAttr>(attr);
    auto id = d ? contractI32(d, "id") : std::nullopt;
    auto channel = d ? contractI32(d, "channel") : std::nullopt;
    auto staging = d ? contractI32(d, "staging_word") : std::nullopt;
    auto dram = d ? contractI32(d, "dram_byte") : std::nullopt;
    auto size = d ? contractI32(d, "size_bytes") : std::nullopt;
    if (!id || !channel || !staging || !dram || !size)
      return module.emitOpError("DMA wait insertion cannot read a DMA contract record");
    Transfer t;
    t.id = *id;
    t.channel = unsigned(*channel);
    uint64_t stagingByte = uint64_t(uint32_t(*staging)) * 4, dramByte = uint32_t(*dram);
    t.staging = {stagingByte, stagingByte + uint32_t(*size)};
    t.dram = {dramByte, dramByte + uint32_t(*size)};
    t.store = contractString(d, "direction") == "store";
    transfers[t.id] = t;
  }
  for (Attribute attr : tiles) {
    auto d = dyn_cast<DictionaryAttr>(attr);
    if (!d || contractString(d, "kind") != "dma_wait")
      continue;
    auto id = contractI32(d, "id"), transfer = contractI32(d, "transfer");
    if (!id || !transfer)
      return module.emitOpError("DMA wait insertion cannot read a tile contract record");
    if (*transfer < 0)
      continue; // A helper's wait, which lowering emits with its launch.
    auto found = transfers.find(*transfer);
    if (found == transfers.end() || found->second.waitCommand >= 0)
      return module.emitOpError("tile contract needs exactly one DMA.WAIT per explicit transfer");
    found->second.waitCommand = *id;
  }
  auto operations = cfg.getAs<ArrayAttr>("operations");
  if (!operations)
    return module.emitOpError("DMA wait insertion cannot read the CFG contract operations");
  llvm::DenseMap<int32_t, std::pair<int32_t, int32_t>> owners;
  for (Attribute attr : operations) {
    auto d = dyn_cast<DictionaryAttr>(attr);
    auto id = d ? contractI32(d, "id") : std::nullopt;
    auto block = d ? contractI32(d, "block") : std::nullopt;
    if (!id || !block)
      return module.emitOpError("DMA wait insertion cannot read a CFG contract operation");
    for (int32_t command : contractI32Array(d, "tile_commands"))
      owners[command] = {*id, *block};
  }
  for (auto &[id, t] : transfers) {
    auto owner = owners.find(t.waitCommand);
    if (t.waitCommand < 0 || owner == owners.end())
      return module.emitOpError("explicit DMA transfer ") << id << " has no owned DMA.WAIT command";
    t.ownerOperation = owner->second.first;
    t.ownerBlock = owner->second.second;
  }
  return transfers;
}

FailureOr<llvm::DenseMap<int32_t, TileRecord>> readTiles(ModuleOp module) {
  llvm::DenseMap<int32_t, TileRecord> records;
  for (Attribute attr : cast<ArrayAttr>(module->getAttr(kAtlasTileContract))) {
    auto d = cast<DictionaryAttr>(attr);
    auto id = contractI32(d, "id"), vmem = contractI32(d, "vmem_byte"),
         bytes = contractI32(d, "bytes"), transfer = contractI32(d, "transfer");
    if (!id || !vmem || !bytes || !transfer)
      return module.emitOpError("DMA wait insertion cannot read a tile contract record");
    uint64_t begin = uint32_t(*vmem);
    records[*id] = {contractString(d, "kind"), {begin, begin + uint32_t(*bytes)}, *transfer};
  }
  return records;
}

// The instruction-word distance a redirect encodes, and its attribute name.
std::optional<std::pair<int64_t, StringRef>> redirect(Operation *op) {
  std::optional<int64_t> distance = atlasRedirectDistance(op);
  if (!distance)
    return std::nullopt;
  return std::pair{*distance, StringRef(isa<BranchOp>(op) ? "offset_bytes" : "offset")};
}

// Whether a PACK loop touching `access` meets transfer `t`, which writes its
// staging window if it loads and reads it if it stores.
bool meets(const AtlasVmemAccess &access, const Transfer &t) {
  AtlasVmemRange staging{t.staging.begin, t.staging.end};
  return access.writes.overlaps(staging) || (!t.store && access.reads.overlaps(staging));
}
} // namespace

LogicalResult mlir::atlas::insertAtlasDMAWaits(ModuleOp module) {
  if (!module->hasAttr(kAtlasDMAWaitsPending))
    return success();
  auto transfers = readTransfers(module);
  if (failed(transfers))
    return failure();
  auto tiles = readTiles(module);
  if (failed(tiles))
    return failure();

  // The decoded stream finds each PACK relayout loop and the VMEM it touches.
  auto stream = readAtlasStream(module, AtlasStreamReadMode::Verification);
  if (failed(stream))
    return failure();
  ArrayRef<Operation *> ops = stream->ops;
  ArrayRef<AtlasPackLoop> loops = stream->packLoops;
  size_t n = ops.size();
  std::vector<bool> target(n, false);
  for (size_t pc = 0; pc < n; ++pc)
    if (auto r = redirect(ops[pc])) {
      int64_t to = int64_t(pc) + r->first;
      if (to < 0 || to >= int64_t(n))
        return ops[pc]->emitOpError("redirect leaves the generated stream");
      target[to] = true;
    }

  // Transfers to complete immediately before each instruction, in order.
  std::vector<SmallVector<int32_t, 2>> waitsBefore(n);
  SmallVector<int32_t, 8> pending;
  auto complete = [&](size_t pc, auto &&needed) {
    for (auto it = pending.begin(); it != pending.end();) {
      if (needed(transfers->find(*it)->second)) {
        waitsBefore[pc].push_back(*it);
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
  };
  auto all = [](const Transfer &) { return true; };
  for (size_t pc = 0; pc < n; ++pc) {
    Operation *op = ops[pc];
    // A PACK loop's only other entry is its own back-edge, so transfers stay
    // in flight through it unless it touches their staging windows.
    const AtlasPackLoop *loop = findAtlasPackLoopAt(loops, pc);
    if (loop && loop->begin == pc) {
      const std::optional<AtlasVmemAccess> &touched = stream->factsOf(*loop).access;
      complete(pc, [&](const Transfer &t) { return !touched || meets(*touched, t); });
    } else if (target[pc]) {
      complete(pc, all);
    }
    if (auto dma = dyn_cast<DMAOp>(op)) {
      auto id = contractTag(op, kAtlasTagDMATransfer);
      if (!id) {
        complete(pc, all); // Helper DMA moves memory beside no explicit transfer.
        continue;
      }
      auto found = transfers->find(*id);
      if (found == transfers->end())
        return op->emitOpError("explicit DMA launch has no DMA contract record");
      const Transfer &launch = found->second;
      complete(pc, [&](const Transfer &t) {
        return t.channel == launch.channel || t.staging.overlaps(launch.staging) ||
               ((t.store || launch.store) && t.dram.overlaps(launch.dram));
      });
      pending.push_back(*id);
    } else if (isa<VLoadOp, VStoreOp>(op)) {
      auto command = contractTag(op, kAtlasTagTileCommand);
      auto record = command ? tiles->find(*command) : tiles->end();
      if (record == tiles->end()) {
        complete(pc, all);
        continue;
      }
      Span vmem = record->second.vmem;
      complete(pc, [&](const Transfer &t) { return t.staging.overlaps(vmem); });
    } else if (isa<DMAWaitOp>(op)) {
      if (auto id = contractTag(op, kAtlasTagDMATransfer))
        llvm::erase(pending, *id);
    } else if (!canOverlapAtlasGeneratedDMA(ops, loops, pc)) {
      complete(pc, all);
    }
  }
  if (!pending.empty())
    return module.emitOpError("generated stream ends with a DMA transfer in flight");
  for (size_t pc = 1; pc < n; ++pc)
    if (redirect(ops[pc - 1]) && !waitsBefore[pc].empty())
      return ops[pc]->emitOpError("a delay slot cannot hold a DMA completion");

  // Each original instruction's new index; a redirect still reaches its
  // original target, after the waits placed on the fall-through path.
  std::vector<int64_t> moved(n);
  int64_t inserted = 0;
  for (size_t pc = 0; pc < n; ++pc) {
    inserted += waitsBefore[pc].size();
    moved[pc] = int64_t(pc) + inserted;
  }
  Builder builder(module.getContext());
  for (size_t pc = 0; pc < n; ++pc)
    if (auto r = redirect(ops[pc])) {
      int64_t to = moved[int64_t(pc) + r->first] - moved[pc];
      ops[pc]->setAttr(r->second, builder.getI32IntegerAttr(int32_t(2 * to)));
    }
  OpBuilder rewriter(module.getContext());
  for (size_t pc = 0; pc < n; ++pc) {
    Operation *op = ops[pc];
    for (int32_t id : waitsBefore[pc]) {
      const Transfer &t = transfers->find(id)->second;
      rewriter.setInsertionPoint(op);
      OperationState state(op->getLoc(), "atlas.dma_wait");
      state.addOperands(op->getOperand(0));
      state.addTypes(op->getOperand(0).getType());
      state.addAttribute("channel", rewriter.getI32IntegerAttr(t.channel));
      state.addAttribute(kAtlasTagDMATransfer, rewriter.getI32IntegerAttr(t.id));
      state.addAttribute(kAtlasTagTileCommand, rewriter.getI32IntegerAttr(t.waitCommand));
      state.addAttribute(kAtlasTagCFGBlock, rewriter.getI32IntegerAttr(t.ownerBlock));
      state.addAttribute(kAtlasTagCFGSource, rewriter.getI32IntegerAttr(t.ownerOperation));
      Operation *wait = rewriter.create(state);
      op->setOperand(0, wait->getResult(0));
    }
  }
  module->removeAttr(kAtlasDMAWaitsPending);
  llvm::SmallVector<uint32_t> words;
  return verifyAtlasArtifact(module, /*llvmBlock=*/true, words);
}

namespace {
struct InsertAtlasDMAWaitsPass
    : PassWrapper<InsertAtlasDMAWaitsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertAtlasDMAWaitsPass)

  StringRef getArgument() const final { return "insert-atlas-dma-waits"; }
  StringRef getDescription() const final {
    return "Insert each explicit DMA transfer's wait immediately before the "
           "first instruction that needs it complete; the last step from "
           "virtual-to-machine lowering to an untimed physical stream";
  }

  void runOnOperation() override {
    if (failed(insertAtlasDMAWaits(getOperation())))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerInsertAtlasDMAWaitsPass() {
  PassRegistration<InsertAtlasDMAWaitsPass>();
}
