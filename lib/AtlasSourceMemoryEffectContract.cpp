#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include <array>
#include <map>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
struct Tile { StringRef kind; uint32_t dram, bytes; int32_t channel; std::vector<int32_t> after; };
struct Owner { int32_t source, block; };
struct Effect {
  int32_t id, source, block, launch, completion, channel;
  uint32_t dram, bytes;
  bool write;
  std::vector<int32_t> predecessors;
};
struct SourceEdge { int32_t from, to; };

bool overlap(const Effect &a, const Effect &b) {
  return uint64_t(a.dram) < uint64_t(b.dram) + b.bytes && uint64_t(b.dram) < uint64_t(a.dram) + a.bytes;
}

// Joins the source-derived CFG and tile contracts by stable source identities.
// Effects are ordered by tile id, so tile ids must follow source order: the
// unowned mailbox prelude, then each operation strictly after the previous one.
FailureOr<std::vector<Effect>> deriveEffects(DictionaryAttr cfg, ArrayAttr records) {
  auto blocks = cfg.getAs<ArrayAttr>("blocks");
  auto operations = cfg.getAs<ArrayAttr>("operations");
  if (!blocks || blocks.empty() || !operations) return failure();
  std::vector<Tile> tiles;
  for (Attribute a : records) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) return failure();
    auto id = contractI32(d, "id"), dram = contractI32(d, "dram_byte"), bytes = contractI32(d, "bytes"), channel = contractI32(d, "channel");
    auto after = d.getAs<DenseI32ArrayAttr>("after");
    if (!id || *id != int32_t(tiles.size()) || !dram || !bytes || !channel || !after) return failure();
    Tile t{contractString(d, "kind"), uint32_t(*dram), uint32_t(*bytes), *channel, {after.asArrayRef().begin(), after.asArrayRef().end()}};
    if ((t.kind == "dma_load" || t.kind == "dma_store" || t.kind == "dma_wait") && (t.channel < 0 || t.channel >= 8)) return failure();
    tiles.push_back(t);
  }
  for (const Tile &t : tiles)
    for (int id : t.after) if (id < 0 || id >= int(tiles.size())) return failure();
  std::map<int32_t, Owner> owners;
  int32_t previousSource = -1, previousBlock = -1, previousCommand = -1;
  for (Attribute a : operations) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) return failure();
    auto source = contractI32(d, "id"), block = contractI32(d, "block");
    auto commands = d.getAs<DenseI32ArrayAttr>("tile_commands");
    if (!source || *source <= previousSource || !block || *block < previousBlock || *block < 0 || *block >= int32_t(blocks.size()) || !commands) return failure();
    previousSource = *source;
    previousBlock = *block;
    unsigned launches = 0;
    for (int32_t id : commands.asArrayRef()) {
      if (id <= previousCommand || id < 0 || id >= int32_t(tiles.size()) || !owners.emplace(id, Owner{*source, *block}).second) return failure();
      previousCommand = id;
      launches += tiles[id].kind == "dma_load" || tiles[id].kind == "dma_store";
    }
    if (launches != atlasTileExpansion(contractString(d, "name")).launches) return failure();
  }
  for (int32_t id = 0; id < int32_t(tiles.size()); ++id)
    if (!owners.count(id) && !owners.empty() && id > owners.begin()->first) return failure();
  std::vector<Effect> effects;
  for (unsigned launch = 0; launch < tiles.size(); ++launch) {
    const Tile &t = tiles[launch];
    if (t.kind != "dma_load" && t.kind != "dma_store") continue;
    auto owner = owners.find(launch);
    int32_t source = -1, block = -1;
    if (owner != owners.end()) {
      source = owner->second.source;
      block = owner->second.block;
    } else {
      // Only the scalar-argument mailbox prelude launches without a source operation.
      auto first = dyn_cast<DictionaryAttr>(blocks[0]);
      auto args = first ? first.getAs<DenseI32ArrayAttr>("args") : DenseI32ArrayAttr{};
      if (launch != 0 || t.kind != "dma_load" || !args || args.empty()) return failure();
    }
    int32_t completion = -1;
    for (unsigned id = 0; id < tiles.size(); ++id) {
      if (tiles[id].kind != "dma_wait" || tiles[id].after != std::vector<int32_t>{int32_t(launch)}) continue;
      auto waitOwner = owners.find(id);
      bool owned = waitOwner != owners.end();
      bool placed = source == -1 ? (id == 1 && !owned) : (owned && waitOwner->second.block == block);
      if (completion != -1 || tiles[id].channel != t.channel || !placed) return failure();
      completion = id;
    }
    if (completion < 0) return failure();
    Effect effect{int32_t(effects.size()), source, block, int32_t(launch), completion, t.channel, t.dram, t.bytes, t.kind == "dma_store", {}};
    for (const Effect &earlier : effects)
      if ((earlier.block == effect.block || earlier.source == -1) && (earlier.write || effect.write) && overlap(earlier, effect)) effect.predecessors.push_back(earlier.id);
    effects.push_back(effect);
  }
  return effects;
}
ArrayAttr encodeEffects(Builder &b, ArrayRef<Effect> effects) {
  auto i = [&](int32_t n) { return b.getI32IntegerAttr(n); };
  auto f = [&](StringRef key, Attribute a) { return b.getNamedAttr(key, a); };
  SmallVector<Attribute> records;
  for (const Effect &e : effects)
    records.push_back(b.getDictionaryAttr({
        f("id", i(e.id)), f("source", i(e.source)), f("block", i(e.block)), f("launch", i(e.launch)), f("completion", i(e.completion)),
        f("dram_byte", i(int32_t(e.dram))), f("bytes", i(int32_t(e.bytes))), f("write", b.getBoolAttr(e.write)), f("predecessors", b.getDenseI32ArrayAttr(e.predecessors))}));
  return b.getArrayAttr(records);
}
struct State {
  int32_t sourceBlock = 0;
  llvm::BitVector launched, completed;
  std::array<int32_t, 8> pending;
  explicit State(unsigned effects) : launched(effects), completed(effects) { pending.fill(-1); }
};
// Launches merge as may-facts to detect duplication; completions merge as
// must-facts so a predecessor counts only when every path completed it.
bool merge(State &to, const State &from) {
  bool changed = false;
  if (to.sourceBlock != from.sourceBlock && to.sourceBlock != -2) { to.sourceBlock = -2; changed = true; }
  auto launched = to.launched, completed = to.completed;
  to.launched |= from.launched;
  to.completed &= from.completed;
  changed |= launched != to.launched || completed != to.completed;
  for (unsigned channel = 0; channel < 8; ++channel)
    if (to.pending[channel] != from.pending[channel] && to.pending[channel] != -2) { to.pending[channel] = -2; changed = true; }
  return changed;
}
} // namespace

FailureOr<DictionaryAttr> mlir::atlas::buildAtlasSourceMemoryEffectContract(
    func::FuncOp function, DictionaryAttr cfgContract, ArrayAttr tileContract) {
  auto operations = cfgContract ? cfgContract.getAs<ArrayAttr>("operations") : ArrayAttr{};
  if (!operations || !tileContract) {
    function.emitOpError("source memory contract requires the CFG and tile contracts");
    return failure();
  }
  std::map<int32_t, DictionaryAttr> records;
  for (Attribute a : operations) {
    auto d = dyn_cast<DictionaryAttr>(a);
    auto id = d ? contractI32(d, "id") : std::nullopt;
    if (!id || !records.emplace(*id, d).second) {
      function.emitOpError("source memory contract requires unique source operation records");
      return failure();
    }
  }
  int32_t source = 0, blockID = 0;
  for (Block &block : function.getBody()) {
    for (Operation &op : block) {
      int32_t id = source++;
      if (!atlasTileExpansion(op.getName().getStringRef()).launches) continue;
      auto found = records.find(id);
      if (found == records.end() || contractString(found->second, "name") != op.getName().getStringRef() || contractI32(found->second, "block") != blockID) {
        op.emitOpError("source memory contract requires its live source identity in the CFG contract");
        return failure();
      }
    }
    ++blockID;
  }
  auto effects = deriveEffects(cfgContract, tileContract);
  if (failed(effects)) {
    function.emitOpError("source memory contract cannot derive block-local source spans and completions");
    return failure();
  }
  Builder b(function.getContext());
  return b.getDictionaryAttr({b.getNamedAttr("effects", encodeEffects(b, *effects))});
}

LogicalResult mlir::atlas::verifyAtlasGeneratedSourceMemoryEffectContract(const AtlasVerificationContext &ctx) {
  ModuleOp module = ctx.module;
  if (failed(requireAtlasGeneratedArtifact(module))) return failure();
  auto contract = module->getAttrOfType<DictionaryAttr>(kAtlasSourceMemoryContract);
  auto cfg = module->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
  auto tiles = module->getAttrOfType<ArrayAttr>(kAtlasTileContract);
  auto bad = [&]() -> LogicalResult { return module.emitOpError("source memory contract has malformed or inconsistent source records"); };
  if (!contract || contract.size() != 1 || !cfg || !tiles) return bad();
  auto records = contract.getAs<ArrayAttr>("effects");
  auto effects = deriveEffects(cfg, tiles);
  Builder b(module.getContext());
  if (!records || failed(effects) || records != encodeEffects(b, *effects)) return bad();
  auto cfgEdges = cfg.getAs<ArrayAttr>("edges");
  auto cfgBlocks = cfg.getAs<ArrayAttr>("blocks");
  if (!cfgEdges || !cfgBlocks) return bad();
  std::vector<SourceEdge> edges;
  for (Attribute a : cfgEdges) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) return bad();
    auto id = contractI32(d, "id"), from = contractI32(d, "from"), to = contractI32(d, "to");
    if (!id || *id != int32_t(edges.size()) || !from || *from < 0 || *from >= int32_t(cfgBlocks.size()) || !to || *to < 0 || *to >= int32_t(cfgBlocks.size())) return bad();
    edges.push_back({*from, *to});
  }
  assert(ctx.stream && "a generated artifact's stream is decoded");
  const AtlasStream &s = *ctx.stream;
  if (s.ops.empty()) return module.emitOpError("source memory contract requires a nonempty issued instruction stream");
  std::map<int32_t, int32_t> launchEffect, waitEffect;
  for (const Effect &e : *effects) { launchEffect[e.launch] = e.id; waitEffect[e.completion] = e.id; }
  std::vector<unsigned> launches(effects->size()), waits(effects->size());
  for (Operation *op : s.ops) {
    auto command = contractTag(op, kAtlasTagTileCommand);
    if (!command) continue;
    if (auto found = launchEffect.find(*command); found != launchEffect.end()) ++launches[found->second];
    if (auto found = waitEffect.find(*command); found != waitEffect.end()) ++waits[found->second];
  }
  for (size_t n = 0; n < effects->size(); ++n)
    if (launches[n] != 1 || waits[n] != 1) return module.emitOpError("source memory contract requires one issued launch and completion per effect");
  auto fail = [&](size_t pc, StringRef why) -> LogicalResult { return s.ops[pc]->emitOpError("source memory contract: ") << why; };
  auto drained = [&](const State &state, bool diagnose, size_t pc) -> LogicalResult {
    if (!diagnose) return success();
    if (state.sourceBlock < 0) return fail(pc, "cannot prove the current source-block visit at exit");
    // A pending launch is also incomplete; report it first so both causes stay distinguishable.
    for (int pending : state.pending) if (pending != -1) return fail(pc, "source block exits with a pending DRAM effect");
    for (const Effect &e : *effects)
      if ((e.block == state.sourceBlock || e.source == -1) && !state.completed[e.id]) return fail(pc, "source block exits before its required effects complete");
    return success();
  };
  auto run = [&](size_t block, State &state, bool diagnose) -> LogicalResult {
    for (size_t pc = s.starts[block]; pc < s.blockEnd(block); ++pc) {
      Operation *op = s.ops[pc];
      auto command = contractTag(op, kAtlasTagTileCommand);
      auto launch = command ? launchEffect.find(*command) : launchEffect.end();
      auto wait = command ? waitEffect.find(*command) : waitEffect.end();
      if (launch != launchEffect.end()) {
        const Effect &e = (*effects)[launch->second];
        auto actual = dyn_cast<DMAOp>(op);
        if (!actual || actual.getChannel() != unsigned(e.channel) || (actual.getDirection() == "store") != e.write) return fail(pc, "launch differs from its checked source effect");
        if (diagnose && e.source != -1 && e.block != state.sourceBlock) return fail(pc, "launch belongs to a different source-block visit");
        if (diagnose && state.launched[e.id]) return fail(pc, "effect launched twice within one source visit");
        if (diagnose && state.pending[e.channel] != -1) return fail(pc, "effect channel is not idle");
        for (int predecessor : e.predecessors)
          if (diagnose && !state.completed[predecessor]) return fail(pc, "overlapping source predecessor has not completed in this visit");
        state.launched.set(e.id);
        state.completed.reset(e.id);
        state.pending[e.channel] = e.id;
      } else if (wait != waitEffect.end()) {
        const Effect &e = (*effects)[wait->second];
        auto actual = dyn_cast<DMAWaitOp>(op);
        if (!actual || actual.getChannel() != unsigned(e.channel)) return fail(pc, "completion differs from its checked source effect");
        bool current = state.pending[e.channel] == e.id && state.launched[e.id];
        if (diagnose && !current) return fail(pc, "completion has no matching current launch");
        state.completed[e.id] = current;
        state.pending[e.channel] = -1;
      }
      if (auto edge = contractTag(op, kAtlasTagCFGEdge); edge && s.instrs[pc].op->opClass == OpClass::Jump) {
        if (*edge >= int32_t(edges.size())) return bad();
        const SourceEdge &e = edges[*edge];
        Operation *target = s.targetOf.lookup(op);
        if (!target || contractTag(target, kAtlasTagCFGBlock) != e.to) return fail(pc, "edge target is not its checked source destination");
        if (diagnose && state.sourceBlock != e.from) return fail(pc, "edge does not leave the current source visit");
        if (failed(drained(state, diagnose, pc))) return failure();
        for (const Effect &effect : *effects)
          if (effect.source != -1) { state.launched.reset(effect.id); state.completed.reset(effect.id); }
        state.sourceBlock = e.to;
      }
      if (s.instrs[pc].release && failed(drained(state, diagnose, pc))) return failure();
    }
    return success();
  };
  auto paths = atlasForwardEntries(s, State(effects->size()), [&](size_t block, State &state) { return run(block, state, false); }, merge);
  if (failed(paths)) return failure();
  for (size_t block = 0; block < s.starts.size(); ++block) {
    if (!paths->reached[block]) continue;
    State output = paths->entries[block];
    if (failed(run(block, output, true))) return failure();
    if (atlasBlockExits(s, block) && failed(drained(output, true, s.blockEnd(block) - 1))) return failure();
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasGeneratedSourceMemoryEffectContract(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasGeneratedSourceMemoryEffectContract(*ctx);
}
