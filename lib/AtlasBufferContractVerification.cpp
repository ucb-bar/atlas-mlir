#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
// Bounds the checker's execution of a copy loop; the relayout takes 32 rows.
constexpr unsigned kCopyIterationBound = 256;

struct Tile { StringRef kind; uint32_t vmem, bytes; std::vector<int32_t> after; };
// `command` observes `bytes` at `vmem`: word i is word `word + i` of `writer`,
// or for a PACK relayout the raw word packWord(i).
struct Read { int32_t command, writer; uint32_t vmem, bytes, word; bool pack; };
struct Pack { int32_t source, store, load, scale = -1; };
struct Derived { std::vector<Tile> tiles; std::vector<Read> reads; std::vector<Pack> packs; };

// Relayout word 8r+4h+l holds raw word 128h+4r+l, the selected PACK layout as
// lowerPack describes it; this checks lowering against its own description,
// not against RTL.
uint32_t packWord(uint32_t word) { return (word % 8) / 4 * 128 + word / 8 * 4 + word % 4; }

FailureOr<Derived> derive(DictionaryAttr cfg, ArrayAttr records) {
  auto operations = cfg ? cfg.getAs<ArrayAttr>("operations") : ArrayAttr{};
  if (!operations || !records) return failure();
  Derived d;
  for (Attribute a : records) {
    auto r = dyn_cast<DictionaryAttr>(a);
    int32_t id = int32_t(d.tiles.size());
    if (!r || contractI32(r, "id") != id || !contractI32(r, "vmem_byte") || !contractI32(r, "bytes") || !r.getAs<DenseI32ArrayAttr>("after")) return failure();
    Tile t{contractString(r, "kind"), uint32_t(*contractI32(r, "vmem_byte")), uint32_t(*contractI32(r, "bytes")), contractI32Array(r, "after")};
    for (int32_t p : t.after) if (p < 0 || p >= id) return failure();
    d.tiles.push_back(t);
  }
  const auto &tiles = d.tiles;
  for (Attribute a : operations) {
    auto op = dyn_cast<DictionaryAttr>(a);
    if (!op) return failure();
    if (contractString(op, "name") != VirtualPackFP8Op::getOperationName()) continue;
    auto source = contractI32(op, "id");
    auto commands = contractI32Array(op, "tile_commands");
    if (!source || commands.size() != 2 || commands[0] < 0 || commands[1] <= commands[0] || commands[1] >= int32_t(tiles.size())) return failure();
    const Tile &store = tiles[commands[0]], &load = tiles[commands[1]];
    if (store.kind != "vstore" || load.kind != "vload" || load.after != std::vector<int32_t>{commands[0]} || store.bytes != 1024 || load.bytes != 1024) return failure();
    d.packs.push_back({*source, commands[0], commands[1]});
  }
  auto contains = [](const Tile &outer, const Tile &inner) {
    return outer.vmem <= inner.vmem && uint64_t(inner.vmem) + inner.bytes <= uint64_t(outer.vmem) + outer.bytes && (inner.vmem - outer.vmem) % 4 == 0;
  };
  for (int32_t id = 0; id < int32_t(tiles.size()); ++id) {
    const Tile &t = tiles[id];
    auto copy = [&](int32_t writer, const Tile &range) {
      d.reads.push_back({id, writer, range.vmem, range.bytes, (range.vmem - tiles[writer].vmem) / 4, false});
    };
    if (t.kind == "dma_store") {
      for (int32_t store : t.after) {
        if (tiles[store].kind != "vstore" || !contains(t, tiles[store])) return failure();
        copy(store, tiles[store]);
      }
    } else if (t.kind == "vload" || t.kind == "mailbox_load") {
      if (t.after.size() != 1) return failure();
      int32_t writer = t.after[0];
      if (tiles[writer].kind == "dma_wait" && tiles[writer].after.size() == 1) writer = tiles[writer].after[0];
      bool pack = llvm::any_of(d.packs, [&](const Pack &p) { return p.load == id; });
      bool dma = writer != t.after[0] && tiles[writer].kind == "dma_load";
      // Lowering orders a VLOAD after a VSTORE only inside PACK.
      if (pack) d.reads.push_back({id, writer, t.vmem, t.bytes, 0, true});
      else if (dma && contains(tiles[writer], t)) copy(writer, t);
      else return failure();
    }
  }
  return d;
}

DictionaryAttr encode(Builder &b, const Derived &d) {
  SmallVector<Attribute> reads, packs;
  for (const Read &r : d.reads)
    reads.push_back(b.getDictionaryAttr({
        namedI32(b, "command", r.command), namedI32(b, "writer", r.writer), namedI32(b, "vmem_byte", r.vmem), namedI32(b, "bytes", r.bytes),
        namedI32(b, "word", r.word), b.getNamedAttr("layout", b.getStringAttr(r.pack ? "pack" : "copy"))}));
  for (const Pack &p : d.packs)
    packs.push_back(b.getDictionaryAttr({namedI32(b, "source", p.source), namedI32(b, "store", p.store), namedI32(b, "load", p.load), namedI32(b, "scale_code", p.scale)}));
  return b.getDictionaryAttr({b.getNamedAttr("reads", b.getArrayAttr(reads)), b.getNamedAttr("packs", b.getArrayAttr(packs))});
}

// An opaque 32-bit VMEM word: word `word` of tile command `writer`'s data.
struct Word {
  int32_t writer;
  uint32_t word;
  bool operator==(const Word &o) const { return writer == o.writer && word == o.word; }
  bool operator!=(const Word &o) const { return !(*this == o); }
};
struct State {
  RegValues x = unknownRegs();
  std::array<std::optional<Word>, 32> carried{}; // the word a scalar register holds since its LW
  std::map<uint32_t, Word> memory;               // by VMEM word index; absent is unproved
};
// Must-facts: a word survives a join only when every reaching path agrees.
bool merge(State &to, const State &from) {
  bool changed = false;
  for (unsigned r = 0; r < 32; ++r) {
    if (to.x[r] && to.x[r] != from.x[r]) { to.x[r].reset(); changed = true; }
    if (to.carried[r] && to.carried[r] != from.carried[r]) { to.carried[r].reset(); changed = true; }
  }
  for (auto i = to.memory.begin(); i != to.memory.end();) {
    auto other = from.memory.find(i->first);
    if (other == from.memory.end() || other->second != i->second) { i = to.memory.erase(i); changed = true; }
    else ++i;
  }
  return changed;
}

std::optional<bool> taken(BranchOp branch, const RegValues &x) {
  auto lhs = x[branch.getLhs()], rhs = x[branch.getRhs()];
  if (!lhs || !rhs) return std::nullopt;
  StringRef kind = branch.getKind();
  if (kind == "beq") return *lhs == *rhs;
  if (kind == "bne") return *lhs != *rhs;
  if (kind == "blt") return int32_t(*lhs) < int32_t(*rhs);
  if (kind == "bge") return int32_t(*lhs) >= int32_t(*rhs);
  if (kind == "bltu") return *lhs < *rhs;
  return *lhs >= *rhs;
}
} // namespace

FailureOr<DictionaryAttr> mlir::atlas::buildAtlasBufferContract(
    func::FuncOp function, DictionaryAttr cfgContract, ArrayAttr tileContract) {
  auto derived = derive(cfgContract, tileContract);
  if (failed(derived))
    return function.emitOpError("buffer contract cannot derive tile readers and PACK endpoints");
  // CFG operation ids number every source operation in block order.
  std::map<int32_t, int32_t> scales;
  int32_t source = 0;
  for (Block &block : function.getBody())
    for (Operation &op : block) {
      int32_t id = source++;
      if (auto pack = dyn_cast<VirtualPackFP8Op>(op)) scales[id] = int32_t(pack.getScaleCode());
    }
  for (Pack &p : derived->packs) {
    auto found = scales.find(p.source);
    if (found == scales.end())
      return function.emitOpError("buffer contract requires its source PACK in the CFG contract");
    p.scale = found->second;
  }
  if (scales.size() != derived->packs.size())
    return function.emitOpError("buffer contract requires every source PACK in the CFG contract");
  Builder b(function.getContext());
  return encode(b, *derived);
}

LogicalResult mlir::atlas::verifyAtlasGeneratedBufferContract(const AtlasVerificationContext &ctx) {
  ModuleOp module = ctx.module;
  if (failed(requireAtlasGeneratedArtifact(module))) return failure();
  auto contract = module->getAttrOfType<DictionaryAttr>(kAtlasBufferContract);
  auto bad = [&]() -> LogicalResult { return module.emitOpError("buffer contract has malformed or inconsistent source records"); };
  auto retained = contract ? contract.getAs<ArrayAttr>("packs") : ArrayAttr{};
  auto derived = derive(module->getAttrOfType<DictionaryAttr>(kAtlasCFGContract), module->getAttrOfType<ArrayAttr>(kAtlasTileContract));
  if (!contract || !retained || failed(derived) || retained.size() != derived->packs.size()) return bad();
  for (auto [record, pack] : llvm::zip(retained, derived->packs)) {
    auto d = dyn_cast<DictionaryAttr>(record);
    auto scale = d ? contractI32(d, "scale_code") : std::nullopt;
    if (!scale || *scale < 0 || *scale > 255) return bad();
    pack.scale = *scale;
  }
  Builder b(module.getContext());
  if (contract != encode(b, *derived)) return bad();
  const std::vector<Tile> &tiles = derived->tiles;
  std::map<int32_t, std::vector<const Read *>> readsOf;
  for (const Read &r : derived->reads) readsOf[r.command].push_back(&r);
  assert(ctx.stream && "a generated artifact's stream is decoded");
  const AtlasStream &s = *ctx.stream;
  auto packOf = [&](Operation *op) -> const Pack * {
    auto source = contractTag(op, kAtlasTagCFGSource);
    auto found = llvm::find_if(derived->packs, [&](const Pack &p) { return source == p.source; });
    return found == derived->packs.end() ? nullptr : &*found;
  };

  // The conversion's ERF entry must hold the source code when it issues.
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues scale = ctx.scaleEntry[block];
    for (size_t pc = s.starts[block]; pc < s.blockEnd(block); ++pc) {
      if (auto conversion = dyn_cast<VPUPackOp>(s.ops[pc])) {
        const Pack *p = packOf(conversion);
        if (!p || conversion.getDirection() != "bf16_to_fp8") return conversion.emitOpError("buffer contract: PACK conversion differs from its source PACK");
        if (scale[conversion.getScaleReg()] != uint32_t(p->scale)) return conversion.emitOpError("buffer contract: PACK conversion scale register does not hold its source scale code");
      }
      applyAtlasScaleRegister(s.instrs[pc], scale);
    }
  }
  // PACK's helper branch closes a one-block counted copy loop, which the walk
  // executes from its entry registers instead of joining at its backedge.
  std::map<size_t, const Pack *> helpers;
  for (size_t block = 0; block < s.starts.size(); ++block) {
    if (!s.endsInBranch(block)) continue;
    Operation *branch = s.ops[s.blockEnd(block) - 2];
    if (!isa<BranchOp>(branch) || !branch->hasAttr(kAtlasTagCFGHelper)) continue;
    const Pack *p = packOf(branch);
    if (!p || s.targetOf.lookup(branch) != s.ops[s.starts[block]])
      return branch->emitOpError("buffer contract: PACK copy loop must be one block closed by its own backedge");
    helpers[block] = p;
  }

  auto check = [&](Operation *op, const Read &r, const State &state) -> LogicalResult {
    StringRef kind = tiles[r.command].kind;
    for (uint32_t i = 0; i < r.bytes / 4; ++i) {
      Word expected{r.writer, r.pack ? packWord(i) : r.word + i};
      auto found = state.memory.find(r.vmem / 4 + i);
      if (found != state.memory.end() && found->second == expected) continue;
      StringRef what = kind == "dma_store" ? "DMA store captures a word its VSTORE did not write"
          : kind == "mailbox_load" ? "mailbox LW reads a word its mailbox DMA load did not write"
          : !r.pack ? "VLOAD reads a word its writer did not write"
          : found != state.memory.end() && found->second.writer == r.writer ? "PACK relayout word differs from the selected row interleave"
          : "PACK relayout word is not a copy of its current raw conversion store";
      return op->emitOpError("buffer contract: ") << what << " (VMEM byte " << r.vmem + 4 * i << ", expected word " << expected.word << " of tile command " << r.writer << ")";
    }
    return success();
  };
  auto write = [](State &state, int32_t writer, const Tile &t) {
    for (uint32_t i = 0; i < t.bytes / 4; ++i) state.memory[t.vmem / 4 + i] = {writer, i};
  };
  auto step = [&](size_t pc, State &state, bool diagnose, const Pack *helper) -> LogicalResult {
    Operation *op = s.ops[pc];
    if (auto id = contractTag(op, kAtlasTagTileCommand); id && *id < int32_t(tiles.size())) {
      if (diagnose)
        for (const Read *r : readsOf[*id])
          if (failed(check(op, *r, state))) return failure();
      const Tile &t = tiles[*id];
      if (t.kind == "vstore") write(state, *id, t);
      // A DMA load's words are indeterminate until its WAIT completes it.
      if (t.kind == "dma_load")
        for (uint32_t i = 0; i < t.bytes / 4; ++i) state.memory.erase(t.vmem / 4 + i);
      if (t.kind == "dma_wait" && t.after.size() == 1 && tiles[t.after[0]].kind == "dma_load") write(state, t.after[0], tiles[t.after[0]]);
    }
    auto address = [&](unsigned base, IntegerAttr offset) -> std::optional<uint32_t> {
      if (!state.x[base]) return std::nullopt;
      return uint32_t(int64_t(*state.x[base]) + offset.getValue().getSExtValue());
    };
    std::optional<Word> loaded;
    if (auto load = dyn_cast<ScalarLoadOp>(op); load && load.getKind() == "lw") {
      auto a = address(load.getBase(), load.getOffsetAttr());
      if (a && *a % 4 == 0)
        if (auto found = state.memory.find(*a / 4); found != state.memory.end()) loaded = found->second;
    } else if (auto store = dyn_cast<ScalarStoreOp>(op)) {
      uint32_t width = store.getKind() == "sw" ? 4 : store.getKind() == "sh" ? 2 : 1;
      auto a = address(store.getBase(), store.getOffsetAttr());
      if (helper && diagnose) {
        const Tile &scratch = tiles[helper->load];
        if (!a || *a < scratch.vmem || uint64_t(*a) + width > uint64_t(scratch.vmem) + scratch.bytes)
          return op->emitOpError("buffer contract: PACK copy loop store is not proven inside its relayout scratch");
      }
      // RTL slices VMEM addresses, so an out-of-range store may alias any word.
      if (!a || uint64_t(*a) + width > uint64_t(kVmemBytes)) {
        state.memory.clear();
      } else {
        for (uint64_t word = *a / 4; word <= (uint64_t(*a) + width - 1) / 4; ++word) state.memory.erase(uint32_t(word));
        if (width == 4 && *a % 4 == 0 && state.carried[store.getSrc()]) state.memory[*a / 4] = *state.carried[store.getSrc()];
      }
    }
    const Instr &in = s.instrs[pc];
    OpClass c = in.op->opClass;
    if (in.rd && (c == OpClass::Alu || c == OpClass::Csr || c == OpClass::Jump || c == OpClass::ScalarLoad)) state.carried[in.rd] = loaded;
    applyScalar(in, state.x);
    return success();
  };
  // Without a decidable exit the copy's effect is unknown: nothing survives it.
  auto copyLoop = [&](size_t block, const Pack &p, State &state, bool diagnose) -> LogicalResult {
    size_t end = s.blockEnd(block);
    auto branch = cast<BranchOp>(s.ops[end - 2]);
    for (unsigned iteration = 0; iteration < kCopyIterationBound; ++iteration) {
      std::optional<bool> again;
      for (size_t pc = s.starts[block]; pc < end; ++pc) {
        if (pc == end - 2) again = taken(branch, state.x);
        if (failed(step(pc, state, diagnose, &p))) return failure();
      }
      if (!again) {
        if (diagnose) return branch.emitOpError("buffer contract: PACK copy loop branch depends on an unknown register");
        state = State();
        return success();
      }
      if (!*again) return success();
    }
    if (diagnose) return branch.emitOpError("buffer contract: PACK copy loop does not exit within ") << kCopyIterationBound << " iterations";
    state = State();
    return success();
  };
  auto run = [&](size_t block, State &state, bool diagnose) -> LogicalResult {
    if (auto helper = helpers.find(block); helper != helpers.end()) return copyLoop(block, *helper->second, state, diagnose);
    for (size_t pc = s.starts[block]; pc < s.blockEnd(block); ++pc)
      if (failed(step(pc, state, diagnose, nullptr))) return failure();
    return success();
  };
  auto successors = [&](size_t block) {
    SmallVector<size_t, 2> next(s.succs[block].begin(), s.succs[block].end());
    if (helpers.count(block)) next.erase(std::remove(next.begin(), next.end(), block), next.end());
    return next;
  };
  auto paths = atlasForwardEntries(s, State(), [&](size_t block, State &state) { return run(block, state, false); }, merge, successors);
  if (failed(paths)) return failure();
  for (size_t block = 0; block < s.starts.size(); ++block) {
    if (!paths->reached[block]) continue;
    State state = paths->entries[block];
    if (failed(run(block, state, true))) return failure();
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasGeneratedBufferContract(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasGeneratedBufferContract(*ctx);
}
