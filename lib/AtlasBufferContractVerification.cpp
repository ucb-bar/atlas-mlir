#include "Atlas/AtlasProofWork.h"
#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
struct Tile { StringRef kind; uint32_t vmem, bytes; std::vector<int32_t> after; };
// `command` observes `bytes` at `vmem`: word i is word `word + i` of `writer`,
// or for a PACK relayout the raw word packWord(i).
struct Read { int32_t command, writer; uint32_t vmem, bytes, word; bool pack; };
struct Pack { int32_t source, store, load, scale = -1; };
struct SlotWrite { int32_t writer; uint32_t word; };
struct Derived {
  std::vector<Tile> tiles;
  std::vector<Read> reads;
  std::vector<Pack> packs;
  std::map<int32_t, SlotWrite> slotWrites, copyWrites;
  std::map<int32_t, uint32_t> scalarSlotLoads, scalarSlotStores;
  std::set<int32_t> zeroStores, indexedStores;
  struct Initialization { int32_t block; uint32_t first, bytes; };
  std::vector<Initialization> initializations;
};

// A counted, straight-line store loop can establish a complete footprint at
// its unique exit. Check the captured source recurrence, not the widened
// per-access address envelope: that envelope alone proves no byte was written.
// CFG verification independently authenticates these source operations and
// edges against the emitted machine stream.
void deriveCountedInitializations(DictionaryAttr cfg, Derived &d) {
  auto values = cfg.getAs<ArrayAttr>("values");
  auto blocks = cfg.getAs<ArrayAttr>("blocks");
  auto edges = cfg.getAs<ArrayAttr>("edges");
  auto operations = cfg.getAs<ArrayAttr>("operations");
  if (!values || !blocks || !edges || !operations) return;
  auto record = [](ArrayAttr records, int32_t id) -> DictionaryAttr {
    atlasProofRecord(AtlasProofKind::Buffer);
    return id >= 0 && unsigned(id) < records.size()
        ? dyn_cast<DictionaryAttr>(records[id]) : DictionaryAttr{};
  };
  auto constant = [&](int32_t id) -> std::optional<uint32_t> {
    auto value = record(values, id);
    if (!value || contractString(value, "def") != "arith.constant") return std::nullopt;
    auto bits = contractI32(value, "constant");
    return bits ? std::optional<uint32_t>(uint32_t(*bits)) : std::nullopt;
  };
  auto increment = [&](int32_t id, int32_t argument, uint32_t step, int32_t block) {
    auto value = record(values, id);
    if (!value || contractString(value, "def") != "arith.addi" ||
        contractI32(value, "block") != block) return false;
    auto operands = contractI32Array(value, "operands");
    return operands.size() == 2 &&
        ((operands[0] == argument && constant(operands[1]) == step) ||
         (operands[1] == argument && constant(operands[0]) == step));
  };
  std::map<int32_t, DictionaryAttr> operationOf;
  for (Attribute raw : operations) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto op = dyn_cast<DictionaryAttr>(raw);
    auto id = op ? contractI32(op, "id") : std::nullopt;
    if (!id) return;
    operationOf[*id] = op;
  }
  std::map<int32_t, SmallVector<DictionaryAttr>> incoming;
  for (Attribute raw : edges) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto edge = dyn_cast<DictionaryAttr>(raw);
    auto target = edge ? contractI32(edge, "to") : std::nullopt;
    if (!target) return;
    incoming[*target].push_back(edge);
  }
  for (Attribute raw : blocks) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto header = dyn_cast<DictionaryAttr>(raw);
    auto headerID = header ? contractI32(header, "id") : std::nullopt;
    auto condition = header ? contractI32(header, "condition") : std::nullopt;
    if (!headerID || !condition) continue;
    auto comparison = record(values, *condition);
    auto args = contractI32Array(header, "args"), exits = contractI32Array(header, "edges");
    if (!comparison || contractI32(comparison, "block") != headerID ||
        contractString(comparison, "def") != "arith.cmpi" ||
        contractI32(comparison, "predicate") != 6 || exits.size() != 2) continue; // ult
    auto operands = contractI32Array(comparison, "operands");
    if (operands.size() != 2) continue;
    auto count = constant(operands[1]);
    auto counter = llvm::find(args, operands[0]);
    if (!count || !*count || counter == args.end()) continue;
    unsigned counterIndex = counter - args.begin();
    auto bodyEdge = record(edges, exits[0]), exitEdge = record(edges, exits[1]);
    auto bodyID = bodyEdge ? contractI32(bodyEdge, "to") : std::nullopt;
    auto exitID = exitEdge ? contractI32(exitEdge, "to") : std::nullopt;
    auto body = bodyID ? record(blocks, *bodyID) : DictionaryAttr{};
    if (!body || !exitID || incoming[*bodyID].size() != 1 || incoming[*exitID].size() != 1 ||
        !contractI32Array(body, "args").empty()) continue;
    auto backedges = contractI32Array(body, "edges");
    auto backedge = backedges.size() == 1 ? record(edges, backedges[0]) : DictionaryAttr{};
    if (!backedge || contractI32(backedge, "to") != headerID) continue;
    auto back = contractI32Array(backedge, "incoming");
    auto &predecessors = incoming[*headerID];
    if (predecessors.size() != 2 || back.size() != args.size() ||
        !increment(back[counterIndex], args[counterIndex], 1, *bodyID)) continue;
    DictionaryAttr seedEdge;
    for (auto edge : predecessors)
      if (contractI32(edge, "from") != bodyID) seedEdge = edge;
    if (!seedEdge) continue;
    auto seeds = contractI32Array(seedEdge, "incoming");
    if (seeds.size() != args.size() || constant(seeds[counterIndex]) != 0) continue;
    // The header is pure, and the body contains only scalar arithmetic plus
    // a contiguous group of indexed word stores to one slot.
    bool valid = true;
    for (int32_t id : contractI32Array(header, "operations")) {
      auto found = operationOf.find(id);
      if (found == operationOf.end() || !contractString(found->second, "name").starts_with("arith.")) valid = false;
    }
    int32_t slot = -1, pointer = -1;
    uint32_t slotWords = 0;
    std::map<uint32_t, int32_t> stores;
    for (int32_t id : contractI32Array(body, "operations")) {
      auto found = operationOf.find(id);
      if (found == operationOf.end()) { valid = false; break; }
      auto op = found->second;
      if (contractString(op, "name").starts_with("arith.")) continue;
      auto access = contractI32Array(op, "operands"), commands = contractI32Array(op, "tile_commands");
      auto word = contractI32(op, "slot_word"), words = contractI32(op, "slot_words");
      auto offset = contractI32(op, "byte_offset");
      if (contractString(op, "name") != "atlas.virtual_slot_store" ||
          contractString(op, "kind") != "sw" || access.size() != 2 || commands.size() != 1 ||
          !word || !words || *words <= 0 || !offset || *offset < 0 || *offset % 4 ||
          !d.indexedStores.count(commands[0])) { valid = false; break; }
      if (slot < 0) { slot = *word; slotWords = *words; pointer = access[1]; }
      if (slot != *word || slotWords != uint32_t(*words) || pointer != access[1] ||
          !stores.emplace(uint32_t(*offset), commands[0]).second) { valid = false; break; }
    }
    auto pointerArg = llvm::find(args, pointer);
    if (!valid || stores.empty() || pointerArg == args.end()) continue;
    unsigned pointerIndex = pointerArg - args.begin();
    uint64_t stride = stores.size(), bytes = uint64_t(*count) * stride * 4;
    if (stride > UINT32_MAX || bytes > uint64_t(slotWords) * 4 ||
        constant(seeds[pointerIndex]) != 0 || !increment(back[pointerIndex], pointer, stride, *bodyID)) continue;
    uint32_t expected = 0;
    for (auto [offset, command] : stores) {
      if (offset != expected) valid = false;
      expected += 4;
    }
    if (!valid) continue;
    for (auto [offset, command] : stores)
      d.scalarSlotLoads.erase(command);
    d.initializations.push_back({*exitID, uint32_t(slot) * 4, uint32_t(bytes)});
  }
}

// Relayout word 8r+4h+l holds raw word 128h+4r+l, the selected PACK layout as
// lowerPack describes it; this checks lowering against its own description,
// not against RTL.
uint32_t packWord(uint32_t word) { return (word % 8) / 4 * 128 + word / 8 * 4 + word % 4; }

FailureOr<Derived> derive(DictionaryAttr cfg, ArrayAttr records) {
  auto operations = cfg ? cfg.getAs<ArrayAttr>("operations") : ArrayAttr{};
  if (!operations || !records) return failure();
  Derived d;
  for (Attribute a : records) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto r = dyn_cast<DictionaryAttr>(a);
    int32_t id = int32_t(d.tiles.size());
    if (!r || contractI32(r, "id") != id || !contractI32(r, "vmem_byte") || !contractI32(r, "bytes") || !r.getAs<DenseI32ArrayAttr>("after")) return failure();
    Tile t{contractString(r, "kind"), uint32_t(*contractI32(r, "vmem_byte")), uint32_t(*contractI32(r, "bytes")), contractI32Array(r, "after")};
    for (int32_t p : t.after) if (p < 0 || p >= id) return failure();
    d.tiles.push_back(t);
  }
  const auto &tiles = d.tiles;
  // A raw readback is admitted only for declared spill/reload expansions.
  // Arbitrary VSTORE -> VLOAD records remain invalid. Keep the writer and
  // BF16 half identity, rather than accepting any overlapping VMEM range.
  std::map<int32_t, std::pair<int32_t, unsigned>> spillWriters;
  std::map<int32_t, unsigned> spillWidths;
  std::set<int32_t> spillReads;
  auto fixed = VirtualAllocationPlan().fixed();
  std::set<int32_t> slotReads;
  std::map<uint32_t, std::pair<uint32_t, StringRef>> slots;
  auto values = cfg.getAs<ArrayAttr>("values");
  auto declareSlot = [&](uint32_t word, uint32_t words, StringRef kind) {
    unsigned alignment = words == 1 ? 1 : kAtlasRegisterWords;
    if (word < fixed.spillWord || word % alignment ||
        uint64_t(word) + words > uint64_t(fixed.spillWord) + fixed.spillWindowWords)
      return false;
    for (auto [address, geometry] : slots) {
      if (address == word) return geometry == std::make_pair(words, kind);
      if (uint64_t(word) < uint64_t(address) + geometry.first &&
          uint64_t(address) < uint64_t(word) + words) return false;
    }
    slots[word] = {words, kind};
    return true;
  };
  for (Attribute a : operations) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto op = dyn_cast<DictionaryAttr>(a);
    if (!op) return failure();
    StringRef name = contractString(op, "name");
    bool save = name == "atlas.virtual_save", restore = name == "atlas.virtual_restore",
         copy = name == "atlas.virtual_slot_copy";
    bool byteLoad = name == "atlas.virtual_slot_load", byteStore = name == "atlas.virtual_slot_store";
    if (!save && !restore && !copy && !byteLoad && !byteStore) continue;
    if (!values) return failure();
    auto word = contractI32(op, "slot_word"), words = contractI32(op, "slot_words");
    StringRef kind = contractString(op, "slot_kind");
    if (!word || !words || *words <= 0 ||
        (kind == "bf16"                ? *words != kAtlasBF16TileWords
         : kind == "fp8"               ? *words != kAtlasRegisterWords
         : kind == "i1" || kind == "i32" ? *words != 1
                                       : kind != "raw" || *words % kAtlasRegisterWords) ||
        !declareSlot(*word, *words, kind)) return failure();
    if (byteLoad || byteStore) {
      auto offset = contractI32(op, "byte_offset");
      StringRef access = contractString(op, "kind");
      unsigned bytes = access == "lw" || access == "sw" ? 4 : access == "lhu" || access == "sh" ? 2 : 1;
      if (!offset || *offset < 0 || uint64_t(*offset) + bytes > uint64_t(*words) * 4 || *offset % bytes ||
          (byteLoad ? access != "lw" && access != "lhu" && access != "lbu" : access != "sw" && access != "sh" && access != "sb")) return failure();
      auto commands = contractI32Array(op, "tile_commands");
      if (commands.size() != 1 || commands[0] < 0 || commands[0] >= int32_t(tiles.size())) return failure();
      const Tile &t = tiles[commands[0]];
      auto operands = contractI32Array(op,"operands"), results = contractI32Array(op,"results");
      bool indexed = operands.size() == (byteStore ? 2u : 1u);
      if (operands.size() != (byteStore ? 1u : 0u) + unsigned(indexed) || results.size() != unsigned(byteLoad)) return failure();
      StringRef expected = byteLoad ? (indexed ? "slot_index_load" : "slot_load") : (indexed ? "slot_index_store" : "slot_store");
      if (t.kind != expected || t.vmem != uint32_t(*word) * 4 + *offset || !t.after.empty() ||
          (indexed ? t.bytes < bytes || t.bytes % bytes || uint64_t(*offset) + t.bytes > uint64_t(*words)*4 : t.bytes != bytes)) return failure();
      if (indexed) {
        // A dynamic store preserves an initialized footprint; it cannot
        // establish that all possible destinations have been written.
        d.scalarSlotLoads[commands[0]] = *word;
        if (byteStore) d.indexedStores.insert(commands[0]);
      } else (byteLoad ? d.scalarSlotLoads : d.scalarSlotStores)[commands[0]] = *word;
      slotReads.insert(commands[0]);
      continue;
    }
    uint32_t destination = *word;
    if (copy) {
      auto other = contractI32(op, "destination_word");
      if (!other || *other == *word || !declareSlot(*other, *words, kind)) return failure();
      destination = *other;
    }
    auto commands = contractI32Array(op, "tile_commands");
    bool scalar = *words == 1;
    unsigned halves = scalar ? 1 : *words / kAtlasRegisterWords;
    if (commands.size() != halves * (copy ? 2 : 1)) return failure();
    auto operands = contractI32Array(op, "operands"), results = contractI32Array(op, "results");
    int32_t reg = scalar ? fixed.scalarTemporary : fixed.tensorTemporary;
    if (!copy) {
      auto ids = save ? operands : results;
      if (ids.size() != 1 || ids[0] < 0 || ids[0] >= int32_t(values.size())) return failure();
      auto value = dyn_cast<DictionaryAttr>(values[ids[0]]);
      auto placed = value ? contractI32(value, "reg") : std::nullopt;
      if (!placed || contractString(value, "type") != kind) return failure();
      reg = *placed;
    }
    for (unsigned half = 0; half < halves; ++half) {
      for (unsigned step = 0; step < (copy ? 2u : 1u); ++step) {
        int32_t id = commands[half * (copy ? 2 : 1) + step];
        if (id < 0 || id >= int32_t(tiles.size())) return failure();
        bool write = save || (copy && step == 1);
        uint32_t address = (write ? destination : uint32_t(*word)) + half * kAtlasRegisterWords;
        const Tile &t = tiles[id];
        auto record = cast<DictionaryAttr>(records[id]);
        if (t.kind != (scalar ? (write ? "slot_store" : "slot_load") : (write ? "vstore" : "vload")) ||
            !t.after.empty() || t.bytes != (scalar ? 4 : kAtlasRegisterBytes) ||
            t.vmem != address * 4 ||
            contractI32(record, "reg") != reg + (!copy && !scalar ? half : 0)) return failure();
        if (write) {
          d.slotWrites[id] = {-int32_t(destination) - 1, half * kAtlasRegisterWords};
          if (copy) d.copyWrites[id] = {-*word - 1, half * kAtlasRegisterWords};
        } else {
          slotReads.insert(id);
          d.reads.push_back({id, -*word - 1, t.vmem, t.bytes, half * kAtlasRegisterWords, false});
        }
      }
    }
  }
  for (Attribute a : operations) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto op = dyn_cast<DictionaryAttr>(a);
    if (!op) return failure();
    StringRef name = contractString(op, "name");
    if (name != VirtualSpillBF16Op::getOperationName() &&
        name != VirtualSpillFP8Op::getOperationName()) continue;
    auto source = contractI32(op, "id");
    auto commands = contractI32Array(op, "tile_commands");
    unsigned halves = name == VirtualSpillBF16Op::getOperationName() ? 2 : 1;
    if (!source || commands.size() != halves) return failure();
    uint32_t base = 0;
    for (unsigned half = 0; half < halves; ++half) {
      int32_t id = commands[half];
      if (id < 0 || id >= int32_t(tiles.size())) return failure();
      const Tile &t = tiles[id];
      if (!half) base = t.vmem;
      if (t.kind != "vstore" || t.bytes != kAtlasRegisterBytes ||
          t.vmem != base + half * kAtlasRegisterBytes ||
          base % kAtlasRegisterBytes || base < fixed.spillWord * 4 ||
          uint64_t(base) + halves * kAtlasRegisterBytes >
              uint64_t(fixed.spillWord + fixed.spillWindowWords) * 4 ||
          !spillWriters.emplace(id, std::make_pair(*source, half)).second) return failure();
      for (auto [word, geometry] : slots)
        if (uint64_t(t.vmem) < uint64_t(word + geometry.first) * 4 &&
            uint64_t(word) * 4 < uint64_t(t.vmem) + t.bytes) return failure();
    }
    spillWidths[*source] = halves;
  }
  for (Attribute a : operations) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto op = cast<DictionaryAttr>(a);
    StringRef name = contractString(op, "name");
    if (name != VirtualReloadBF16Op::getOperationName() &&
        name != VirtualReloadFP8Op::getOperationName()) continue;
    auto commands = contractI32Array(op, "tile_commands");
    unsigned halves = name == VirtualReloadBF16Op::getOperationName() ? 2 : 1;
    if (commands.size() != halves) return failure();
    int32_t writer = -1;
    for (unsigned half = 0; half < halves; ++half) {
      int32_t id = commands[half];
      if (id < 0 || id >= int32_t(tiles.size())) return failure();
      const Tile &t = tiles[id];
      if (t.kind != "vload" || t.bytes != kAtlasRegisterBytes || t.after.size() != 1) return failure();
      auto saved = spillWriters.find(t.after[0]);
      if (saved == spillWriters.end()) return failure();
      if (!half) writer = saved->second.first;
      if (saved->second != std::make_pair(writer, half) || spillWidths[writer] != halves ||
          t.vmem != tiles[t.after[0]].vmem) return failure();
      spillReads.insert(id);
    }
  }
  for (Attribute a : operations) {
    atlasProofRecord(AtlasProofKind::Buffer);
    auto op = dyn_cast<DictionaryAttr>(a);
    if (!op) return failure();
    if (contractString(op, "name") == VirtualRawPackFP8Op::getOperationName()) {
      auto commands = contractI32Array(op, "tile_commands");
      if (commands.size() != 2 || commands[0] < 0 || commands[1] != commands[0] + 1 ||
          commands[1] >= int32_t(tiles.size())) return failure();
      const Tile &store = tiles[commands[0]], &load = tiles[commands[1]];
      if (store.kind != "slot_store" || load.kind != "scale_load" ||
          store.vmem != fixed.packWord * 4 || load.vmem != store.vmem ||
          store.bytes != 4 || load.bytes != 4 ||
          load.after != std::vector<int32_t>{commands[0]}) return failure();
      d.slotWrites[commands[0]] = {commands[0], 0};
      d.reads.push_back({commands[1], commands[0], store.vmem, 4, 0, false});
    }
    if (contractString(op, "name") != VirtualPackFP8Op::getOperationName()) continue;
    auto source = contractI32(op, "id");
    auto commands = contractI32Array(op, "tile_commands");
    if (!source || commands.size() != 2 || commands[0] < 0 || commands[1] <= commands[0] || commands[1] >= int32_t(tiles.size())) return failure();
    const Tile &store = tiles[commands[0]], &load = tiles[commands[1]];
    if (store.kind != "vstore" || load.kind != "vload" ||
        load.after != std::vector<int32_t>{commands[0]} ||
        store.bytes != kAtlasRegisterBytes || load.bytes != kAtlasRegisterBytes)
      return failure();
    d.packs.push_back({*source, commands[0], commands[1]});
  }
  auto contains = [](const Tile &outer, const Tile &inner) {
    return outer.vmem <= inner.vmem && uint64_t(inner.vmem) + inner.bytes <= uint64_t(outer.vmem) + outer.bytes && (inner.vmem - outer.vmem) % 4 == 0;
  };
  for (int32_t id = 0; id < int32_t(tiles.size()); ++id) {
    atlasProofRecord(AtlasProofKind::Buffer);
    const Tile &t = tiles[id];
    auto copy = [&](int32_t writer, const Tile &range) {
      d.reads.push_back({id, writer, range.vmem, range.bytes, (range.vmem - tiles[writer].vmem) / 4, false});
    };
    if (t.kind == "dma_store") {
      for (int32_t store : t.after) {
        if (tiles[store].kind != "vstore") return failure();
        const Tile &writer = tiles[store];
        uint32_t first = std::max(t.vmem, writer.vmem);
        uint64_t end = std::min(uint64_t(t.vmem) + t.bytes, uint64_t(writer.vmem) + writer.bytes);
        if (end <= first) continue;
        copy(store, Tile{t.kind, first, uint32_t(end-first), {}});
      }
    } else if (t.kind == "vload" || t.kind == "mailbox_load" || t.kind == "slot_load" || t.kind == "slot_index_load" || t.kind == "slot_index_store") {
      if (slotReads.count(id)) continue;
      if (t.after.size() != 1) return failure();
      int32_t writer = t.after[0];
      if (tiles[writer].kind == "dma_wait" && tiles[writer].after.size() == 1) writer = tiles[writer].after[0];
      bool pack = llvm::any_of(d.packs, [&](const Pack &p) { return p.load == id; });
      bool dma = writer != t.after[0] && tiles[writer].kind == "dma_load";
      // Reloads copy a checked spill store; PACK uses its own relayout map.
      if (pack) d.reads.push_back({id, writer, t.vmem, t.bytes, 0, true});
      else if ((dma || spillReads.count(id)) && contains(tiles[writer], t)) copy(writer, t);
      else if (dma) {
        int32_t prefill = -1;
        for (int32_t earlier = 0; earlier < writer; ++earlier)
          if (tiles[earlier].kind == "vstore" && contains(tiles[earlier], t)) prefill = earlier;
        if (prefill < 0) return failure();
        d.zeroStores.insert(prefill);
        for (uint32_t offset = 0; offset < t.bytes; offset += 4) {
          Tile part{t.kind, t.vmem + offset, 4, {}};
          copy(contains(tiles[writer], part) ? writer : prefill, part);
        }
      } else return failure();
    }
  }
  deriveCountedInitializations(cfg, d);
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
// Compact byte initialization facts keep partial scalar stores precise without
// copying one set node per initialized byte at each CFG program point.
class InitializedRanges {
  std::map<uint32_t, uint32_t> ranges;
public:
  void insert(uint32_t byte) { insert(byte, byte + 1); }
  void insert(uint32_t first, uint32_t end) {
    auto next = ranges.lower_bound(first);
    if (next != ranges.begin() && std::prev(next)->second >= first) next = std::prev(next);
    while (next != ranges.end() && next->first <= end) {
      first = std::min(first, next->first); end = std::max(end, next->second);
      next = ranges.erase(next);
    }
    ranges[first] = end;
  }
  bool count(uint32_t byte) const {
    auto next = ranges.upper_bound(byte);
    return next != ranges.begin() && byte < std::prev(next)->second;
  }
  void erase(uint32_t byte) {
    auto next = ranges.upper_bound(byte);
    if (next == ranges.begin()) return;
    auto found = std::prev(next);
    if (found->second <= byte) return;
    auto [first, end] = *found; ranges.erase(found);
    if (first < byte) ranges[first] = byte;
    if (byte + 1 < end) ranges[byte + 1] = end;
  }
  void clear() { ranges.clear(); }
  bool intersect(const InitializedRanges &other) {
    std::map<uint32_t, uint32_t> common;
    for (auto [first, end] : ranges)
      for (auto [rightFirst, rightEnd] : other.ranges) {
        auto begin = std::max(first, rightFirst), stop = std::min(end, rightEnd);
        if (begin < stop) common[begin] = stop;
      }
    bool changed = common != ranges; ranges = std::move(common); return changed;
  }
};
struct State {
  RegValues x = unknownRegs();
  std::array<std::optional<Word>, 32> carried{}; // the word a scalar register holds since its LW
  std::array<std::optional<Word>, kAtlasTensorRegisters> tensor{}; // first of 256 consecutive words in a raw register
  InitializedRanges initialized;
  std::set<unsigned> zeroTensor;
  std::map<uint32_t, Word> memory;               // by VMEM word index; absent is unproved
};
// Must-facts: a word survives a join only when every reaching path agrees.
bool merge(State &to, const State &from) {
  bool changed = false;
  for (unsigned r = 0; r < kAtlasTensorRegisters; ++r)
    if (to.tensor[r] && to.tensor[r] != from.tensor[r]) { to.tensor[r].reset(); changed = true; }
  for (unsigned r = 0; r < 32; ++r) {
    if (to.x[r] && to.x[r] != from.x[r]) { to.x[r].reset(); changed = true; }
    if (to.carried[r] && to.carried[r] != from.carried[r]) { to.carried[r].reset(); changed = true; }
  }
  for (auto i = to.memory.begin(); i != to.memory.end();) {
    auto other = from.memory.find(i->first);
    if (other == from.memory.end() || other->second != i->second) { i = to.memory.erase(i); changed = true; }
    else ++i;
  }
  changed |= to.initialized.intersect(from.initialized);
  for (auto it = to.zeroTensor.begin(); it != to.zeroTensor.end();)
    if (!from.zeroTensor.count(*it)) { it = to.zeroTensor.erase(it); changed = true; } else ++it;
  return changed;
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
  atlasProofInvocation(AtlasProofKind::Buffer);
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
  auto registerRoundtrip = [&](Operation *op) -> DictionaryAttr {
    auto source = contractTag(op, kAtlasTagCFGSource);
    auto cfg = module->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
    auto records = cfg ? cfg.getAs<ArrayAttr>("operations") : ArrayAttr{};
    if (source && records) for (Attribute raw : records) {
      atlasProofRecord(AtlasProofKind::Buffer);
      auto record = dyn_cast<DictionaryAttr>(raw);
      if (record && contractI32(record, "id") == source &&
          contractString(record, "name") == "atlas.virtual_pack_roundtrip_bf16") return record;
    }
    return {};
  };
  auto rawPack = [&](Operation *op) {
    auto source = contractTag(op, kAtlasTagCFGSource);
    auto cfg = module->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
    auto records = cfg ? cfg.getAs<ArrayAttr>("operations") : ArrayAttr{};
    if (source && records) for (Attribute raw : records) {
      atlasProofRecord(AtlasProofKind::Buffer);
      auto record = dyn_cast<DictionaryAttr>(raw);
      if (record && contractI32(record,"id") == source &&
          contractString(record,"name") == "atlas.virtual_raw_pack_fp8") return true;
    }
    return false;
  };

  // The conversion's ERF entry must hold the source code when it issues.
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues scale = ctx.scaleEntry[block];
    for (size_t pc = s.starts[block]; pc < s.blockEnd(block); ++pc) {
      atlasProofTransfer(AtlasProofKind::Buffer);
      if (auto conversion = dyn_cast<VPUPackOp>(s.ops[pc])) {
        // CFG authenticates the runtime scalar store, SELD and raw PACK;
        // the buffer walk below proves SELD reads that store's current word.
        if (rawPack(conversion)) continue;
        // The CFG contract checks the register-only helper's fields and origins.
        if (auto record = registerRoundtrip(conversion)) {
          uint32_t expected = conversion.getDirection() == "bf16_to_fp8"
              ? uint32_t(*contractI32(record, "scale_code")) : 127;
          if (scale[conversion.getScaleReg()] != expected)
            return conversion.emitOpError("CFG contract PACK roundtrip conversion scale differs");
          continue;
        }
        const Pack *p = packOf(conversion);
        if (!p || conversion.getDirection() != "bf16_to_fp8") return conversion.emitOpError("buffer contract: PACK conversion differs from its source PACK");
        if (scale[conversion.getScaleReg()] != uint32_t(p->scale)) return conversion.emitOpError("buffer contract: PACK conversion scale register does not hold its source scale code");
      }
      applyAtlasScaleRegister(s.instrs[pc], scale);
    }
  }
  // PACK's helper branch closes its relayout loop (AtlasLoop.h), which the
  // walk executes from its entry registers instead of joining at its backedge.
  struct Helper { const AtlasLoop *loop; const Pack *pack; };
  std::map<size_t, Helper> helpers;
  for (size_t block = 0; block < s.starts.size(); ++block) {
    if (!s.endsInBranch(block)) continue;
    Operation *branch = s.ops[s.redirectOf(block)];
    if (!isa<BranchOp>(branch) || !branch->hasAttr(kAtlasTagCFGHelper)) continue;
    const AtlasLoop *loop = s.loop(block);
    const Pack *p = packOf(branch);
    if (!loop || !p || !isAtlasPackRelayoutLoop(s.ops, *loop))
      return branch->emitOpError("buffer contract: PACK helper branch does not close a PACK relayout loop (one block of that PACK's scalar copy, closed by its own backedge)");
    helpers[block] = {loop, p};
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
  auto write = [](State &state, int32_t writer, const Tile &t, uint32_t first = 0) {
    for (uint32_t i = 0; i < t.bytes; ++i) state.initialized.insert(t.vmem + i);
    for (uint32_t i = 0; i < t.bytes / 4; ++i) state.memory[t.vmem / 4 + i] = {writer, first + i};
  };
  auto step = [&](size_t pc, State &state, bool diagnose, const Pack *helper) -> LogicalResult {
    atlasProofTransfer(AtlasProofKind::Buffer);
    Operation *op = s.ops[pc];
    if (auto id = contractTag(op, kAtlasTagTileCommand); id && *id < int32_t(tiles.size())) {
      if (diagnose)
        for (const Read *r : readsOf[*id])
          if (failed(check(op, *r, state))) return failure();
      const Tile &t = tiles[*id];
      if (derived->scalarSlotLoads.count(*id) && diagnose)
        for (uint32_t byte = t.vmem; byte < t.vmem + t.bytes; ++byte)
          if (!state.initialized.count(byte)) return op->emitOpError("buffer contract: scalar slot load reads an uninitialized byte");
      if (derived->zeroStores.count(*id) && diagnose &&
          (!isa<VStoreOp>(op) || !state.zeroTensor.count(cast<VStoreOp>(op).getSrc())))
        return op->emitOpError("buffer contract: partial DMA padding is not a zero tensor");
      if (auto found = derived->scalarSlotStores.find(*id); found != derived->scalarSlotStores.end()) {
        for (uint32_t byte = t.vmem; byte < t.vmem + t.bytes; ++byte) state.initialized.insert(byte);
        uint32_t word = t.vmem / 4;
        bool complete = true;
        for (unsigned byte = 0; byte < 4; ++byte) complete &= state.initialized.count(word * 4 + byte);
        if (complete) state.memory[word] = {-int32_t(found->second)-1, word-found->second};
        else state.memory.erase(word);
      }
      if (auto found = derived->copyWrites.find(*id); found != derived->copyWrites.end()) {
        auto origin = isa<VStoreOp>(op) ? state.tensor[cast<VStoreOp>(op).getSrc()] : state.carried[cast<ScalarStoreOp>(op).getSrc()];
        if (diagnose && origin != std::optional<Word>(Word{found->second.writer, found->second.word}))
          return op->emitOpError("buffer contract: slot copy lost its loaded source bytes");
      }
      if (auto found = derived->slotWrites.find(*id); found != derived->slotWrites.end())
        write(state, found->second.writer, t, found->second.word);
      else if (t.kind == "vstore") write(state, *id, t);
      // A DMA load's words are indeterminate until its WAIT completes it.
      if (t.kind == "dma_load") {
        for (uint32_t i = 0; i < t.bytes / 4; ++i) state.memory.erase(t.vmem / 4 + i);
        for (uint32_t i = 0; i < t.bytes; ++i) state.initialized.erase(t.vmem + i);
      }
      if (t.kind == "dma_wait" && t.after.size() == 1 && tiles[t.after[0]].kind == "dma_load") write(state, t.after[0], tiles[t.after[0]]);
    }
    unsigned zeroReg = 0, zeroWidth = 0;
    if (auto x = dyn_cast<VLoadOp>(op)) { zeroReg = x.getDst(); zeroWidth = 1; }
    if (auto x = dyn_cast<VPUUnaryOp>(op)) { zeroReg = x.getDst(); zeroWidth = 2; }
    if (auto x = dyn_cast<VPUBinaryOp>(op)) { zeroReg = x.getDst(); zeroWidth = 2; }
    if (auto x = dyn_cast<VPUReduceOp>(op)) { zeroReg = x.getDst(); zeroWidth = 2; }
    if (auto x = dyn_cast<XLUTransposeOp>(op)) { zeroReg = x.getDst(); zeroWidth = 1; }
    if (auto x = dyn_cast<MXUPopOp>(op)) { zeroReg = x.getDst(); zeroWidth = x.getFormat() == "bf16" ? 2 : 1; }
    if (auto x = dyn_cast<VPUPackOp>(op)) { zeroReg = x.getDst(); zeroWidth = x.getDirection() == "fp8_to_bf16" ? 2 : 1; }
    if (auto x = dyn_cast<VLIOp>(op)) { zeroReg = x.getDst(); zeroWidth = x.getMode() == "all" || x.getMode() == "row" ? 2 : 1; }
    for (unsigned half = 0; half < zeroWidth; ++half) state.zeroTensor.erase(zeroReg + half);
    if (auto x = dyn_cast<VLIOp>(op); x && x.getMode() == "all" && x.getImmediate() == 0) {
      state.zeroTensor.insert(x.getDst()); state.zeroTensor.insert(x.getDst()+1);
    }
    if (auto load = dyn_cast<VLoadOp>(op)) {
      auto id = contractTag(op, kAtlasTagTileCommand);
      state.tensor[load.getDst()].reset();
      if (id) if (auto found = state.memory.find(tiles[*id].vmem / 4); found != state.memory.end())
        state.tensor[load.getDst()] = found->second;
    } else {
      unsigned reg = 0, width = 0;
      if (auto p = dyn_cast<VPUUnaryOp>(op)) { reg = p.getDst(); width = 2; }
      if (auto p = dyn_cast<VPUBinaryOp>(op)) { reg = p.getDst(); width = 2; }
      if (auto p = dyn_cast<VPUReduceOp>(op)) { reg = p.getDst(); width = 2; }
      if (auto p = dyn_cast<XLUTransposeOp>(op)) { reg = p.getDst(); width = 1; }
      if (auto p = dyn_cast<MXUPopOp>(op)) { reg = p.getDst(); width = p.getFormat() == "bf16" ? 2 : 1; }
      if (auto p = dyn_cast<VPUPackOp>(op)) { reg = p.getDst(); width = p.getDirection() == "fp8_to_bf16" ? 2 : 1; }
      if (auto p = dyn_cast<VLIOp>(op)) { reg = p.getDst(); width = p.getMode() == "all" || p.getMode() == "row" ? 2 : 1; }
      for (unsigned half = 0; half < width; ++half) state.tensor[reg + half].reset();
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
    } else if (auto store = dyn_cast<ScalarStoreOp>(op);
               store && !(contractTag(op, kAtlasTagTileCommand) &&
               (derived->slotWrites.count(*contractTag(op, kAtlasTagTileCommand)) ||
                derived->scalarSlotStores.count(*contractTag(op, kAtlasTagTileCommand)) ||
                derived->indexedStores.count(*contractTag(op, kAtlasTagTileCommand))))) {
      uint32_t width = store.getKind() == "sw" ? 4 : store.getKind() == "sh" ? 2 : 1;
      auto a = address(store.getBase(), store.getOffsetAttr());
      if (helper && diagnose) {
        const Tile &scratch = tiles[helper->load];
        if (!a || *a < scratch.vmem || uint64_t(*a) + width > uint64_t(scratch.vmem) + scratch.bytes)
          return op->emitOpError("buffer contract: PACK copy loop store is not proven inside its relayout scratch");
      }
      // RTL slices VMEM addresses, so an out-of-range store may alias any word.
      if (!a || uint64_t(*a) + width > uint64_t(kAtlasVmemBytes)) {
        state.memory.clear();
        state.initialized.clear();
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
  auto copyLoop = [&](const Helper &helper, State &state, bool diagnose) -> LogicalResult {
    auto exit = runAtlasLoop(s.ops, *helper.loop, [&]() -> const RegValues & { return state.x; },
                                 [&](size_t pc) { return step(pc, state, diagnose, helper.pack); });
    if (failed(exit)) return failure();
    if (*exit == AtlasLoopExit::Exited) return success();
    Operation *branch = s.ops[helper.loop->branch];
    if (diagnose && *exit == AtlasLoopExit::UnknownBranch)
      return branch->emitOpError("buffer contract: PACK copy loop branch depends on an unknown register");
    if (diagnose)
      return branch->emitOpError("buffer contract: PACK copy loop does not exit within ") << kAtlasLoopIterationBound << " iterations";
    state = State();
    return success();
  };
  std::map<int32_t, size_t> sourceEntry;
  for (size_t pc = 0; pc < s.ops.size(); ++pc)
    if (auto source = contractTag(s.ops[pc], kAtlasTagCFGBlock)) sourceEntry.emplace(*source, pc);
  auto run = [&](size_t block, State &state, bool diagnose) -> LogicalResult {
    auto sourceBlock = contractTag(s.ops[s.starts[block]], kAtlasTagCFGBlock);
    for (const auto &initialization : derived->initializations)
      if (sourceBlock == initialization.block && sourceEntry.at(*sourceBlock) == s.starts[block])
        state.initialized.insert(initialization.first, initialization.first + initialization.bytes);
    if (auto helper = helpers.find(block); helper != helpers.end()) return copyLoop(helper->second, state, diagnose);
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
