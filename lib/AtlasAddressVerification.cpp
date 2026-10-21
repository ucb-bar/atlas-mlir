#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;

namespace {
struct SourceValue {
  StringRef def, type;
  int32_t block, predicate;
  uint32_t constant;
  std::vector<int32_t> operands;
};

struct SourceBlock { int32_t condition; std::vector<int32_t> args, edges, predecessors; };
struct SourceEdge { int32_t from, to; std::vector<int32_t> incoming; };

// Reconstruct bounds from the retained source SSA/CFG, independently of the
// live-IR analysis that constructed the DMA record. No generated address or
// solver assignment is used to decide the source interval.
class CapturedAddressProof {
public:
  std::vector<SourceValue> values;
  std::vector<SourceBlock> blocks;
  std::vector<SourceEdge> edges;

  bool parse(DictionaryAttr cfg) {
    auto vr = cfg ? cfg.getAs<ArrayAttr>("values") : ArrayAttr{};
    auto br = cfg ? cfg.getAs<ArrayAttr>("blocks") : ArrayAttr{};
    auto er = cfg ? cfg.getAs<ArrayAttr>("edges") : ArrayAttr{};
    if (!vr || !br || br.empty() || !er) return false;
    for (Attribute a : vr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) return false;
      auto id = contractI32(d, "id");
      auto block = contractI32(d, "block");
      if (!id || *id != int32_t(values.size()) || !block || *block < 0 ||
          *block >= int32_t(br.size()) || !d.getAs<DenseI32ArrayAttr>("operands")) return false;
      values.push_back({contractString(d, "def"), contractString(d,"type"), *block,
          contractI32(d, "predicate").value_or(-1),
          uint32_t(contractI32(d, "constant").value_or(0)), contractI32Array(d, "operands")});
      if (values.back().def == "arith.constant" && !contractI32(d, "constant")) return false;
    }
    for (Attribute a : br) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) return false;
      auto id = contractI32(d, "id");
      auto condition = contractI32(d, "condition");
      if (!id || *id != int32_t(blocks.size()) || !condition ||
          !d.getAs<DenseI32ArrayAttr>("args") || !d.getAs<DenseI32ArrayAttr>("edges")) return false;
      blocks.push_back({*condition, contractI32Array(d,"args"), contractI32Array(d,"edges"), {}});
      for (int arg : blocks.back().args)
        if (!validValue(arg) || values[arg].block != *id || values[arg].def != "argument") return false;
    }
    for (Attribute a : er) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) return false;
      auto id = contractI32(d, "id");
      auto from = contractI32(d, "from");
      auto to = contractI32(d, "to");
      if (!id || *id != int32_t(edges.size()) || !from || !validBlock(*from) ||
          !to || !validBlock(*to) || !d.getAs<DenseI32ArrayAttr>("incoming")) return false;
      edges.push_back({*from,*to,contractI32Array(d,"incoming")});
      if (edges.back().incoming.size() != blocks[*to].args.size()) return false;
      for (int v : edges.back().incoming) if (!validValue(v)) return false;
      blocks[*to].predecessors.push_back(*id);
    }
    for (unsigned block = 0; block < blocks.size(); ++block)
      for (int id : blocks[block].edges)
        if (id < 0 || id >= int(edges.size()) || edges[id].from != int(block)) return false;
    std::set<int> reached;
    std::vector<int> pending{0};
    while (!pending.empty()) {
      int b = pending.back(); pending.pop_back();
      if (!reached.insert(b).second) continue;
      for (int e : blocks[b].edges) pending.push_back(edges[e].to);
    }
    if (reached.size() != blocks.size()) return false;
    dominators.assign(blocks.size(), reached);
    dominators[0] = {0};
    bool change = true;
    while (change) {
      change = false;
      for (unsigned b = 1; b < blocks.size(); ++b) {
        std::set<int> next = reached;
        for (int e : blocks[b].predecessors) {
          std::set<int> intersection;
          const auto &d = dominators[edges[e].from];
          std::set_intersection(next.begin(), next.end(), d.begin(), d.end(),
                                std::inserter(intersection, intersection.end()));
          next = std::move(intersection);
        }
        next.insert(b);
        if (next != dominators[b]) { dominators[b] = std::move(next); change = true; }
      }
    }
    return true;
  }

  std::optional<I32AddressSpan> prove(int value, int useBlock) {
    if (!validValue(value) || values[value].type != "i32" || !validBlock(useBlock) || !active.insert(value).second)
      return std::nullopt;
    auto result = derive(value,useBlock);
    active.erase(value);
    return result;
  }

private:
  std::vector<std::set<int>> dominators;
  std::set<int> active, constants;
  bool validValue(int v) const { return v >= 0 && v < int(values.size()); }
  bool validBlock(int b) const { return b >= 0 && b < int(blocks.size()); }
  bool dominates(int a, int b) const { return dominators[b].count(a); }

  std::optional<uint32_t> literal(int id) {
    if (!validValue(id) || values[id].type != "i32" || !constants.insert(id).second) return std::nullopt;
    const auto &v = values[id];
    std::optional<uint32_t> result;
    if (v.def == "arith.constant") result = v.constant;
    else if (v.def == "arith.addi" && v.operands.size() == 2) {
      auto a = literal(v.operands[0]), b = literal(v.operands[1]);
      if (a && b) result = *a + *b;
    }
    constants.erase(id);
    return result;
  }
  std::optional<uint32_t> increment(int next, int current) {
    if (!validValue(next)) return std::nullopt;
    const auto &v = values[next];
    if (v.def != "arith.addi" || v.operands.size() != 2) return std::nullopt;
    if (v.operands[0] == current) return literal(v.operands[1]);
    if (v.operands[1] == current) return literal(v.operands[0]);
    return std::nullopt;
  }
  std::optional<I32AddressSpan> derive(int id, int useBlock) {
    if (auto n = literal(id)) return I32AddressSpan{*n,*n};
    const auto &v = values[id];
    if (v.def == "arith.addi" && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (a && b && uint64_t(a->last)+b->last <= UINT32_MAX)
        return I32AddressSpan{a->first+b->first,a->last+b->last};
      return std::nullopt;
    }
    if (v.def != "argument") return std::nullopt;
    const auto &h = blocks[v.block];
    if (h.edges.size() != 2 || !validValue(h.condition) || h.predecessors.size() != 2)
      return std::nullopt;
    int body = edges[h.edges[0]].to;
    if (body == edges[h.edges[1]].to || blocks[body].predecessors.size() != 1 ||
        edges[blocks[body].predecessors[0]].from != v.block || !dominates(body,useBlock))
      return std::nullopt;
    const auto &cmp = values[h.condition];
    if (cmp.def != "arith.cmpi" || cmp.block != v.block || cmp.type != "i1" || cmp.predicate != int(arith::CmpIPredicate::ult) ||
        cmp.operands.size() != 2) return std::nullopt;
    int counter = cmp.operands[0];
    auto count = literal(cmp.operands[1]);
    auto p = std::find(h.args.begin(),h.args.end(),id);
    auto c = std::find(h.args.begin(),h.args.end(),counter);
    if (p == h.args.end() || c == h.args.end() || !count || !*count) return std::nullopt;
    int entry = -1, back = -1;
    for (int e : h.predecessors) {
      if (blocks[edges[e].from].edges.size() != 1) return std::nullopt;
      if (dominates(v.block,edges[e].from)) back = e;
      else entry = e;
    }
    if (entry < 0 || back < 0 || !dominates(body,edges[back].from)) return std::nullopt;
    unsigned pi = p-h.args.begin(), ci = c-h.args.begin();
    auto initial = literal(edges[entry].incoming[ci]);
    auto one = increment(edges[back].incoming[ci],counter);
    auto stride = increment(edges[back].incoming[pi],id);
    if (!initial || *initial || !one || *one != 1 || !stride || !*stride || *stride % 32)
      return std::nullopt;
    auto base = prove(edges[entry].incoming[pi],useBlock);
    if (!base) return std::nullopt;
    uint64_t last = uint64_t(base->last)+uint64_t(*count-1)* *stride;
    if (last > UINT32_MAX) return std::nullopt;
    return I32AddressSpan{base->first,uint32_t(last)};
  }
};
// Replay the retained scalar CFG, independently of live-IR initialization
// analysis. The existing CFG and DMA verifiers bind these records to emitted
// instructions. Tensor arithmetic and runtime input values are not evaluated.
bool initializedOutputReads(const AtlasVerificationContext &ctx,
                            CapturedAddressProof &source, DictionaryAttr cfg,
                            ArrayAttr dma, uint64_t first, uint64_t end) {
  std::map<int32_t, DictionaryAttr> transfers, operations;
  for (Attribute a : cfg.getAs<ArrayAttr>("operations")) {
    auto d = dyn_cast<DictionaryAttr>(a);
    auto id = contractI32(d, "id");
    if (!id || !operations.emplace(*id, d).second) return false;
  }
  std::vector<std::vector<int32_t>> operationsByBlock(source.blocks.size());
  for (const auto &[id, record] : operations) {
    auto block = contractI32(record, "block");
    if (!block || *block < 0 || *block >= int32_t(source.blocks.size())) return false;
    operationsByBlock[*block].push_back(id);
  }
  for (Operation *op : ctx.ops) {
    if (!isa<DMAOp>(op)) continue;
    auto owner = contractTag(op, kAtlasTagCFGSource);
    auto id = contractTag(op, kAtlasTagDMATransfer);
    if (!owner || !id || *id < 0 || *id >= int32_t(dma.size()) ||
        !operations.count(*owner) || !transfers.emplace(*owner, cast<DictionaryAttr>(dma[*id])).second)
      return false;
  }
  std::map<int32_t, uint32_t> arguments;
  std::set<uint32_t> initialized; // DMA accesses are aligned 32-byte blocks.
  std::set<int> active;
  unsigned remaining = 1000000;
  std::function<std::optional<uint32_t>(int)> evaluate = [&](int id) -> std::optional<uint32_t> {
    if (!remaining || id < 0 || id >= int(source.values.size()) || !active.insert(id).second) return std::nullopt;
    --remaining;
    const auto &v = source.values[id];
    std::optional<uint32_t> value;
    if (v.def == "argument") {
      auto found = arguments.find(id);
      if (found != arguments.end()) value = found->second;
    } else if (v.def == "arith.constant") value = v.constant;
    else if (v.operands.size() == 2) {
      auto a = evaluate(v.operands[0]), b = evaluate(v.operands[1]);
      if (a && b) {
        if (v.def == "arith.addi") value = *a + *b;
        else if (v.def == "arith.cmpi") {
          switch (arith::CmpIPredicate(v.predicate)) {
          case arith::CmpIPredicate::eq: value = *a == *b; break;
          case arith::CmpIPredicate::ne: value = *a != *b; break;
          case arith::CmpIPredicate::ult: value = *a < *b; break;
          case arith::CmpIPredicate::ule: value = *a <= *b; break;
          case arith::CmpIPredicate::ugt: value = *a > *b; break;
          case arith::CmpIPredicate::uge: value = *a >= *b; break;
          case arith::CmpIPredicate::slt: value = int32_t(*a) < int32_t(*b); break;
          case arith::CmpIPredicate::sle: value = int32_t(*a) <= int32_t(*b); break;
          case arith::CmpIPredicate::sgt: value = int32_t(*a) > int32_t(*b); break;
          case arith::CmpIPredicate::sge: value = int32_t(*a) >= int32_t(*b); break;
          }
        }
      }
    }
    active.erase(id);
    return value;
  };
  int block = 0;
  while (remaining) {
    --remaining;
    for (int id : operationsByBlock[block]) {
      if (!remaining) return false;
      --remaining;
      auto found = transfers.find(id);
      if (found == transfers.end()) continue;
      auto d = found->second;
      auto dynamic = contractI32(d, "dram_value");
      auto address = dynamic ? evaluate(*dynamic) : std::optional<uint32_t>(uint32_t(*contractI32(d, "dram_byte")));
      auto bytes = contractI32(d, "size_bytes");
      if (!address || !bytes || *bytes <= 0 || *address % 32 || *bytes % 32) return false;
      uint64_t stop = uint64_t(*address) + uint32_t(*bytes);
      if (stop > (1ULL << 32)) return false;
      bool store = contractString(d, "direction") == "store";
      if (store && (*address < first || stop > end)) return false;
      if (uint64_t(*address) < end && first < stop) {
        StringRef name = contractString(operations[id], "name");
        if (!name.starts_with(store ? "atlas.virtual_dma_store_" : "atlas.virtual_dma_load_")) return false;
        for (uint64_t at = std::max(uint64_t(*address), first); at < std::min(stop, end); at += 32) {
          if (store) initialized.insert(uint32_t(at));
          else if (!initialized.count(uint32_t(at))) return false;
        }
      }
    }
    const auto &b = source.blocks[block];
    if (b.edges.empty()) return true;
    int edge;
    if (b.edges.size() == 1 && b.condition == -1) edge = b.edges[0];
    else if (b.edges.size() == 2) {
      auto condition = evaluate(b.condition);
      if (!condition) return false;
      edge = b.edges[*condition ? 0 : 1];
    } else return false;
    const auto &next = source.edges[edge];
    std::vector<std::optional<uint32_t>> incoming;
    for (int v : next.incoming) incoming.push_back(evaluate(v));
    for (unsigned i = 0; i < incoming.size(); ++i) {
      int arg = source.blocks[next.to].args[i];
      if (incoming[i]) arguments[arg] = *incoming[i];
      else arguments.erase(arg);
    }
    block = next.to;
  }
  return false;
}
} // namespace

LogicalResult mlir::atlas::verifyAtlasBoundedDMAAddresses(const AtlasVerificationContext &ctx) {
  ModuleOp module = ctx.module;
  auto dma = module->getAttrOfType<ArrayAttr>(kAtlasDMAContract);
  if (!dma) return success(); // The ordinary DMA verifier diagnoses absence.
  bool dynamic = llvm::any_of(dma, [](Attribute a) {
    auto d = dyn_cast<DictionaryAttr>(a);
    return d && (d.get("dram_value") || d.get("dram_span"));
  });
  auto cfg = ctx.module->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
  auto output = cfg ? cfg.getAs<DictionaryAttr>("output_buffer") : DictionaryAttr{};
  bool explicitBuffer = module->hasAttr("atlas.output_dram_bytes") || (cfg && cfg.get("output_buffer"));
  if (!dynamic && !explicitBuffer) return success();
  uint64_t outputFirst = 0, outputEnd = 0;
  bool readsOutput = false;
  if (explicitBuffer) {
    auto base = output ? output.getAs<IntegerAttr>("base") : IntegerAttr{};
    auto bytes = output ? output.getAs<IntegerAttr>("bytes") : IntegerAttr{};
    if (!output || output.size() != 2 || !base || !bytes ||
        !base.getType().isSignlessInteger(64) || !bytes.getType().isSignlessInteger(64) ||
        base.getInt() < 0x80000000LL || base.getInt() > UINT32_MAX || bytes.getInt() <= 0 ||
        base.getInt() % 32 || bytes.getInt() % 32 ||
        uint64_t(base.getInt())+uint64_t(bytes.getInt()) > (1ULL << 32) ||
        module->getAttr("atlas.output_dram_base") != base ||
        module->getAttr("atlas.output_dram_bytes") != bytes)
      return module.emitOpError("explicit output buffer differs from source bounds");
    outputFirst = base.getInt(); outputEnd = outputFirst+bytes.getInt();
    // Indexed boundary loads are tile helpers, not explicit DMA records.
    // Check them too, so the output range cannot alias read-only inputs.
    auto tiles = module->getAttrOfType<ArrayAttr>(kAtlasTileContract);
    if (!tiles) return module.emitOpError("explicit output buffer has no tile footprints");
    for (Attribute a : tiles) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) return module.emitOpError("invalid output-buffer tile footprint");
      StringRef kind = contractString(d,"kind");
      if (kind != "dma_load" && kind != "dma_store") continue;
      auto address = contractI32(d,"dram_byte"), length = contractI32(d,"bytes");
      if (!address || !length) return module.emitOpError("missing output-buffer tile footprint");
      uint64_t first = uint32_t(*address);
      uint64_t end = first+uint32_t(contractI32(d,"dram_span_bytes").value_or(*length));
      if (end <= first || end > (1ULL << 32) ||
          (kind == "dma_store" && (first < outputFirst || end > outputEnd)))
        return module.emitOpError("tile footprint violates source output-buffer bounds");
      readsOutput |= kind == "dma_load" && first < outputEnd && outputFirst < end;
    }
    auto operations = cfg.getAs<ArrayAttr>("operations");
    if (!operations) return module.emitOpError("explicit output buffer has no source operations");
    for (Attribute a : operations) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d || contractString(d,"name") == "atlas.virtual_output_bf16")
        return module.emitOpError("explicit output buffer cannot contain indexed outputs");
    }
  }
  CapturedAddressProof source;
  if (!source.parse(cfg)) return module.emitOpError("bounded DMA requires a valid source CFG");
  auto fail = [&](StringRef reason) { return module.emitOpError("bounded DMA address differs from its guarded source induction: ") << reason; };
  for (Attribute a : dma) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) return fail("invalid DMA record");
    auto value = contractI32(d,"dram_value"), span = contractI32(d,"dram_span");
    if (explicitBuffer) {
      auto start = contractI32(d,"dram_byte"), bytes = contractI32(d,"size_bytes");
      if (!start || !bytes) return fail("missing DMA footprint");
      uint64_t first = uint32_t(*start), end = first+uint32_t(span.value_or(*bytes));
      StringRef direction = contractString(d,"direction");
      if ((direction != "load" && direction != "store") || end <= first || end > (1ULL << 32) ||
          (direction == "store" && (first < outputFirst || end > outputEnd)))
        return module.emitOpError("DMA footprint violates source output-buffer bounds");
    }
    if (!value && !span) continue;
    auto id = contractI32(d,"id"), first = contractI32(d,"dram_byte"), size = contractI32(d,"size_bytes");
    if (!value || !span || !id || !first || !size || *value < 0 || *value >= int(source.values.size())) return fail("invalid source value or span");
    std::optional<int> useBlock;
    for (Operation *op : ctx.ops)
      if (isa<DMAOp>(op) && contractTag(op,kAtlasTagDMATransfer) == id) {
        auto sourceID = contractTag(op,kAtlasTagCFGSource);
        auto operations = cfg.getAs<ArrayAttr>("operations");
        if (!sourceID || !operations) return fail("missing DMA source operation");
        // Source operation IDs include terminators and state-only operations;
        // the retained operation array is sparse in that identity space.
        DictionaryAttr record;
        for (Attribute a : operations) {
          auto candidate = dyn_cast<DictionaryAttr>(a);
          if (candidate && contractI32(candidate,"id") == sourceID) {
            if (record) return fail("duplicate source operation identity");
            record = candidate;
          }
        }
        if (!record) return fail("invalid source operation");
        auto operands = contractI32Array(record,"operands");
        bool store = contractString(record,"name").starts_with("atlas.virtual_dma_store_");
        bool load = contractString(record,"name").starts_with("atlas.virtual_dma_load_");
        if ((!load && !store) || operands.size() < (store ? 2u : 1u) || operands[store ? 1 : 0] != *value) return fail("DMA source operand mismatch");
        useBlock = contractI32(record,"block");
      }
    if (!useBlock) return fail("missing tagged DMA instruction");
    auto bound = source.prove(*value,*useBlock);
    if (!bound || bound->first != uint32_t(*first) || uint64_t(bound->last)+uint32_t(*size) > (1ULL<<32) ||
        uint64_t(bound->last)-bound->first+uint32_t(*size) != uint32_t(*span)) return fail(bound ? "incorrect address interval" : "unproved source recurrence");
  }
  if (readsOutput && !initializedOutputReads(ctx, source, cfg, dma, outputFirst, outputEnd))
    return module.emitOpError("source CFG does not prove output reads initialized within the scalar work limit");
  return success();
}
