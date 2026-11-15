#include "Atlas/AtlasProofWork.h"
#include "Atlas/AtlasProofLimits.h"
#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/BitVector.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <deque>
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
    atlasProofInvocation(AtlasProofKind::CapturedAddress);
    auto vr = cfg ? cfg.getAs<ArrayAttr>("values") : ArrayAttr{};
    auto br = cfg ? cfg.getAs<ArrayAttr>("blocks") : ArrayAttr{};
    auto er = cfg ? cfg.getAs<ArrayAttr>("edges") : ArrayAttr{};
    if (!vr || !br || br.empty() || !er) return false;
    for (Attribute a : vr) {
      atlasProofRecord(AtlasProofKind::CapturedAddress);
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
      atlasProofRecord(AtlasProofKind::CapturedAddress);
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
      atlasProofRecord(AtlasProofKind::CapturedAddress);
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
    atlasProofTransfer(AtlasProofKind::CapturedAddress);
    if (!validValue(value) || values[value].type != "i32" || !validBlock(useBlock) || !active.insert(value).second)
      return std::nullopt;
    auto result = derive(value,useBlock);
    active.erase(value);
    return result;
  }

  std::optional<I32AddressSpan> proveUnsigned(int value, int useBlock) {
    unsignedRange = true;
    auto result = prove(value,useBlock);
    unsignedRange = false;
    return result;
  }

private:
  bool unsignedRange = false;
  std::vector<std::set<int>> dominators;
  std::set<int> active, constants;
  bool validValue(int v) const { return v >= 0 && v < int(values.size()); }
  bool validBlock(int b) const { return b >= 0 && b < int(blocks.size()); }
  bool dominates(int a, int b) const { return dominators[b].count(a); }

  std::optional<uint32_t> literal(int id) {
    atlasProofTransfer(AtlasProofKind::CapturedAddress);
    if (!validValue(id) || (values[id].type != "i32" && values[id].type != "i1") || !constants.insert(id).second) return std::nullopt;
    const auto &v = values[id];
    std::optional<uint32_t> result;
    if (v.def == "arith.constant") result = v.constant;
    else if (v.operands.size() == 2) {
      auto a = literal(v.operands[0]), b = literal(v.operands[1]);
      if (a && b) {
        if (v.def == "arith.addi") result = *a + *b;
        if (v.def == "arith.subi") result = *a - *b;
        if (v.def == "arith.muli") result = *a * *b;
        if (v.def == "arith.andi") result = *a & *b;
        if (v.def == "arith.ori") result = *a | *b;
        if (v.def == "arith.xori") result = *a ^ *b;
        if (v.def == "arith.shli" && *b < 32) result = *a << *b;
        if (v.def == "arith.shrui" && *b < 32) result = *a >> *b;
      }
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
  std::optional<I32AddressSpan> refine(int id, int condition, bool truth, int useBlock) {
    auto range = prove(id,useBlock);
    if (!validValue(condition)) return range;
    const auto &extended = values[condition];
    if (extended.def == "arith.extui" && extended.operands.size() == 1)
      condition = extended.operands[0];
    if (!validValue(condition)) return range;
    const auto &cmp = values[condition];
    if (cmp.def != "arith.cmpi" || cmp.operands.size() != 2 || cmp.operands[0] != id) return range;
    auto bound = literal(cmp.operands[1]);
    if (!bound) return range;
    uint32_t first = range ? range->first : 0, last = range ? range->last : UINT32_MAX;
    switch (arith::CmpIPredicate(cmp.predicate)) {
    case arith::CmpIPredicate::ult:
      if (truth) { if (!*bound) return std::nullopt; last = std::min(last,*bound-1); }
      else first = std::max(first,*bound);
      break;
    case arith::CmpIPredicate::ule:
      if (truth) last = std::min(last,*bound);
      else { if (*bound == UINT32_MAX) return std::nullopt; first = std::max(first,*bound+1); }
      break;
    case arith::CmpIPredicate::eq:
      if (truth) { first = std::max(first,*bound); last = std::min(last,*bound); }
      break;
    default: break;
    }
    return first <= last ? std::optional<I32AddressSpan>{{first,last}} : std::nullopt;
  }
  std::optional<I32AddressSpan> maskedSelect(int id, int useBlock) {
    const auto &out = values[id];
    if (out.def != "arith.xori" || out.operands.size() != 2) return std::nullopt;
    for (int operand : out.operands) if (!validValue(operand)) return std::nullopt;
    for (bool swapOuter : {false,true}) {
      int no = out.operands[swapOuter ? 1 : 0];
      const auto &chosen = values[out.operands[swapOuter ? 0 : 1]];
      if (chosen.def != "arith.andi" || chosen.operands.size() != 2) continue;
      if (!validValue(chosen.operands[0]) || !validValue(chosen.operands[1])) continue;
      for (bool swapMask : {false,true}) {
        const auto &mask = values[chosen.operands[swapMask ? 0 : 1]];
        const auto &delta = values[chosen.operands[swapMask ? 1 : 0]];
        if (mask.def != "arith.subi" || mask.operands.size() != 2 ||
            delta.def != "arith.xori" || delta.operands.size() != 2 || literal(mask.operands[0]) != 0u) continue;
        auto predicate = prove(mask.operands[1],useBlock);
        if (!predicate || predicate->last > 1) continue;
        int yes;
        if (delta.operands[0] == no) yes = delta.operands[1];
        else if (delta.operands[1] == no) yes = delta.operands[0];
        else continue;
        auto a = refine(yes,mask.operands[1],true,useBlock), b = refine(no,mask.operands[1],false,useBlock);
        if (a && b) return I32AddressSpan{std::min(a->first,b->first),std::max(a->last,b->last)};
      }
    }
    return std::nullopt;
  }
  std::optional<I32AddressSpan> derive(int id, int useBlock) {
    if (auto n = literal(id)) return I32AddressSpan{*n,*n};
    const auto &v = values[id];
    if (!unsignedRange && v.def == "arith.shli" && v.operands.size() == 2) {
      // Re-establish both nonwrapping bounds and alignment from the captured
      // expression. An ordinary scalar range alone proves neither that every
      // interior address is aligned nor that scaling stays within i32.
      auto bits = literal(v.operands[1]);
      if (!bits || *bits < 5 || *bits >= 32) return std::nullopt;
      auto index = proveUnsigned(v.operands[0],useBlock);
      if (!index || (uint64_t(index->last) << *bits) > UINT32_MAX)
        return std::nullopt;
      return I32AddressSpan{uint32_t(index->first << *bits),
                            uint32_t(index->last << *bits)};
    }
    if (unsignedRange) if (auto selected = maskedSelect(id,useBlock)) return selected;
    if (v.def == "arith.addi" && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (a && b && uint64_t(a->last)+b->last <= UINT32_MAX)
        return I32AddressSpan{a->first+b->first,a->last+b->last};
      return std::nullopt;
    }
    if (unsignedRange && v.def == "arith.muli" && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (a && b && uint64_t(a->last)*b->last <= UINT32_MAX)
        return I32AddressSpan{a->first*b->first,a->last*b->last};
      return std::nullopt;
    }
    if (unsignedRange && v.def == "arith.extui" && v.operands.size() == 1 && validValue(v.operands[0]) && values[v.operands[0]].type == "i1") {
      if (auto n = literal(v.operands[0])) return I32AddressSpan{*n,*n};
      return I32AddressSpan{0,1};
    }
    if (unsignedRange && v.def == "arith.andi" && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (a || b) return I32AddressSpan{0,a && b ? std::min(a->last,b->last) : a ? a->last : b->last};
    }
    if (unsignedRange && (v.def == "arith.ori" || v.def == "arith.xori") && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (!a || !b) return std::nullopt;
      uint32_t last = std::max(a->last,b->last);
      last |= last >> 1; last |= last >> 2; last |= last >> 4; last |= last >> 8; last |= last >> 16;
      return I32AddressSpan{0,last};
    }
    if (unsignedRange && v.def == "arith.subi" && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (!a || !b || a->first < b->last) return std::nullopt;
      return I32AddressSpan{a->first-b->last,a->last-b->first};
    }
    if (unsignedRange && (v.def == "arith.shli" || v.def == "arith.shrui") && v.operands.size() == 2) {
      auto a = prove(v.operands[0],useBlock), b = prove(v.operands[1],useBlock);
      if (b && b->last < 32) {
        if (v.def == "arith.shrui") {
          if (!a) a = I32AddressSpan{0,UINT32_MAX};
          return I32AddressSpan{a->first >> b->last,a->last >> b->first};
        }
        if (a && (uint64_t(a->last) << b->last) <= UINT32_MAX)
          return I32AddressSpan{uint32_t(a->first << b->first),uint32_t(a->last << b->last)};
      }
      return std::nullopt;
    }
    if (v.def == "arith.cmpi" && v.type == "i1") return I32AddressSpan{0,1};
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
    auto step = increment(edges[back].incoming[ci],counter);
    auto stride = increment(edges[back].incoming[pi],id);
    if (!initial || !step || !*step || !stride || !*stride || (!unsignedRange && *stride % 32))
      return std::nullopt;
    uint64_t iterations = *count;
    if (*initial == 0) {
      if (*step != 1) return std::nullopt;
    } else {
      if (*initial >= *count) return std::nullopt;
      // Independently reconstruct ceil((limit-initial)/step), including the
      // last guard value. A wrapped exit test would invalidate the bound.
      iterations = 1 + (uint64_t(*count) - 1 - *initial) / *step;
      if (uint64_t(*initial) + iterations * *step > UINT32_MAX)
        return std::nullopt;
    }
    auto base = prove(edges[entry].incoming[pi],useBlock);
    if (!base) return std::nullopt;
    uint64_t last = uint64_t(base->last)+(iterations-1)* *stride;
    if (last > UINT32_MAX) return std::nullopt;
    return I32AddressSpan{base->first,uint32_t(last)};
  }
};
// The only unowned DMA admitted by output-initialization replay is the
// canonical argument mailbox. Bind both the retained tile and actual captured
// operands to the declared mailbox ABI; command id zero alone is insufficient.
bool argumentMailbox(const AtlasVerificationContext &ctx, DMAOp dma) {
  auto control = ctx.module->getAttrOfType<IntegerAttr>("atlas.control_dram_base");
  auto records = ctx.module->getAttrOfType<ArrayAttr>(kAtlasTileContract);
  if (!control || !control.getType().isSignlessInteger(64) ||
      !atlasDMAMemorySpan(uint64_t(control.getInt()), kAtlasRegisterBytes) || control.getInt() % 32 ||
      uint64_t(control.getInt()) + kAtlasRegisterBytes > (1ULL << 32) ||
      !records || records.empty() || !ctx.stream)
    return false;
  auto record = dyn_cast<DictionaryAttr>(records[0]);
  auto after = record ? record.getAs<DenseI32ArrayAttr>("after") : DenseI32ArrayAttr{};
  auto fixed = VirtualAllocationPlan().fixed();
  auto field = [&](StringRef name, uint32_t expected) {
    auto value = contractI32(record, name);
    return value && uint32_t(*value) == expected;
  };
  if (!record || contractString(record,"kind") != "dma_load" || !after || !after.empty() ||
      !field("id",0) || !field("reg",uint32_t(-1)) || !field("transfer",uint32_t(-1)) ||
      !field("channel",fixed.loadChannel) || !field("vmem_byte",fixed.mailboxWord * 4) ||
      !field("dram_byte",uint32_t(control.getInt())) || !field("bytes",kAtlasRegisterBytes) ||
      dma.getDirection() != "load" || dma.getChannel() != fixed.loadChannel)
    return false;
  auto found = ctx.pcOf.find(dma.getOperation());
  const auto &stream = *ctx.stream;
  if (found == ctx.pcOf.end() || stream.starts.empty() || found->second >= stream.blockEnd(0) ||
      found->second >= ctx.dmaUpperEntry.size() || ctx.dmaUpperEntry[found->second] != 0u)
    return false;
  auto values = stream.entry[0];
  for (size_t pc = stream.starts[0]; pc < found->second; ++pc) {
    if (isa<DMAOp>(stream.ops[pc])) return false;
    timing::applyScalar(stream.instrs[pc], values);
  }
  return values[dma.getReg()] == fixed.mailboxWord &&
         values[dma.getDram()] == uint32_t(control.getInt()) &&
         values[dma.getSize()] == kAtlasRegisterBytes;
}

// Replay the retained scalar CFG, independently of live-IR initialization
// analysis. The existing CFG and DMA verifiers bind these records to emitted
// instructions. Tensor arithmetic and runtime input values are not evaluated.
bool initializedOutputReads(const AtlasVerificationContext &ctx,
                            CapturedAddressProof &source, DictionaryAttr cfg,
                            ArrayAttr dma, uint64_t first, uint64_t end) {
  atlasProofInvocation(AtlasProofKind::CapturedInitialization);
  std::map<int32_t, DictionaryAttr> transfers, operations;
  for (Attribute a : cfg.getAs<ArrayAttr>("operations")) {
    atlasProofRecord(AtlasProofKind::CapturedInitialization);
    auto d = dyn_cast<DictionaryAttr>(a);
    auto id = contractI32(d, "id");
    if (!id || !operations.emplace(*id, d).second) return false;
  }
  for (Operation *op : ctx.ops) {
    atlasProofRecord(AtlasProofKind::CapturedAddress);
    if (!isa<DMAOp>(op)) continue;
    auto owner = contractTag(op, kAtlasTagCFGSource);
    auto id = contractTag(op, kAtlasTagDMATransfer);
    if (!owner && !id && contractTag(op, kAtlasTagTileCommand) == 0 &&
        argumentMailbox(ctx, cast<DMAOp>(op)))
      continue;
    if (!owner || !id || *id < 0 || *id >= int32_t(dma.size()) ||
        !operations.count(*owner) || !transfers.emplace(*owner, cast<DictionaryAttr>(dma[*id])).second)
      return false;
  }
  // Scalar expressions are evaluated on demand below. Replaying unrelated
  // tensor/scalar operations at every loop iteration adds no memory evidence.
  // Validate all records once, then retain every DMA transfer in source order.
  std::vector<std::vector<int32_t>> operationsByBlock(source.blocks.size());
  for (const auto &[id, record] : operations) {
    auto block = contractI32(record, "block");
    if (!block || *block < 0 || *block >= int32_t(source.blocks.size())) return false;
    if (transfers.count(id)) operationsByBlock[*block].push_back(id);
  }
  std::map<int32_t, uint32_t> arguments;
  // One bit per aligned DMA beat. Copying a tree node per initialized beat
  // made snapshots of wide output panels much larger than their payload.
  // This retains the independent captured-CFG proof's exact beat granularity.
  llvm::BitVector initialized((end - first) / 32);
  std::set<int> active;
  // Every scalar step and output-byte read is still checked exhaustively.
  unsigned remaining = kAtlasScalarProofWorkLimit;
  // Follow only scalar values that can affect a transfer or a control edge.
  // A branch may carry unrelated data; its value cannot distinguish memory
  // states. Derive this slice independently from the retained CFG, including
  // every predecessor of a needed block argument.
  llvm::BitVector needed(source.values.size());
  std::vector<std::vector<int>> incomingValues(source.values.size());
  for (const auto &edge : source.edges)
    for (unsigned i = 0; i < edge.incoming.size(); ++i)
      incomingValues[source.blocks[edge.to].args[i]].push_back(edge.incoming[i]);
  std::vector<int> work;
  auto require = [&](int id) {
    if (id >= 0 && id < int(source.values.size()) && !needed.test(id)) {
      needed.set(id);
      work.push_back(id);
    }
  };
  for (const auto &block : source.blocks) require(block.condition);
  for (const auto &[id, transfer] : transfers)
    if (auto value = contractI32(transfer, "dram_value")) require(*value);
  while (!work.empty()) {
    if (!remaining) return false;
    --remaining;
    atlasProofRecord(AtlasProofKind::CapturedInitialization);
    int id = work.back(); work.pop_back();
    const auto &value = source.values[id];
    const auto &operands = value.def == "argument" ? incomingValues[id] : value.operands;
    for (int operand : operands) require(operand);
  }
  std::function<std::optional<uint32_t>(int)> evaluate = [&](int id) -> std::optional<uint32_t> {
    if (!remaining || id < 0 || id >= int(source.values.size()) || !active.insert(id).second) return std::nullopt;
    --remaining;
    atlasProofTransfer(AtlasProofKind::CapturedInitialization);
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
  struct Pending { int block; std::map<int32_t, uint32_t> arguments; llvm::BitVector initialized; std::optional<unsigned> predecessor; };
  std::deque<Pending> pending;
  std::map<std::vector<uint64_t>, llvm::BitVector> entries;
  std::map<std::vector<uint64_t>, unsigned> stateIDs;
  AtlasProofStateGraph graph;
  pending.push_back({0, arguments, initialized, std::nullopt});
  while (!pending.empty() && remaining) {
    auto current = std::move(pending.front()); pending.pop_front();
    int block = current.block; arguments = std::move(current.arguments); initialized = std::move(current.initialized);
    std::vector<uint64_t> key{uint64_t(block)};
    for (auto [id, bits] : arguments) { key.push_back(uint32_t(id)); key.push_back(bits); }
    auto [state, newState] = stateIDs.emplace(key, stateIDs.size());
    if (newState) graph.addNode();
    unsigned stateID = state->second;
    if (current.predecessor && graph.addEdge(*current.predecessor, stateID, remaining) !=
        AtlasProofStateGraph::EdgeResult::Added) return false;
    auto [entry, fresh] = entries.emplace(key, initialized);
    if (!fresh) {
      auto common = entry->second;
      common &= initialized;
      if (common == entry->second) continue;
      entry->second = common; initialized = std::move(common);
    }
    if (!remaining) return false;
    --remaining;
    atlasProofRecord(AtlasProofKind::CapturedInitialization);
    for (int id : operationsByBlock[block]) {
      if (!remaining) return false;
      --remaining;
      atlasProofTransfer(AtlasProofKind::CapturedInitialization);
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
        unsigned begin = (std::max(uint64_t(*address), first) - first) / 32;
        unsigned limit = (std::min(stop, end) - first) / 32;
        if (store) initialized.set(begin, limit);
        else if (initialized.find_first_unset_in(begin, limit) != -1) return false;
      }
    }
    const auto &b = source.blocks[block];
    if (b.edges.empty()) continue;
    auto enqueue = [&](int edge) {
      const auto &next = source.edges[edge];
      std::vector<std::optional<uint32_t>> incoming;
      for (unsigned i = 0; i < next.incoming.size(); ++i)
        incoming.push_back(needed.test(source.blocks[next.to].args[i]) ?
            evaluate(next.incoming[i]) : std::nullopt);
      auto nextArguments = arguments;
      for (unsigned i = 0; i < incoming.size(); ++i) {
        int arg = source.blocks[next.to].args[i];
        if (incoming[i]) nextArguments[arg] = *incoming[i];
        else nextArguments.erase(arg);
      }
      pending.push_back({next.to, std::move(nextArguments), initialized, stateID});
    };
    int edge;
    if (b.edges.size() == 1 && b.condition == -1) edge = b.edges[0];
    else if (b.edges.size() == 2) {
      auto condition = evaluate(b.condition);
      if (!condition) { enqueue(b.edges[0]); enqueue(b.edges[1]); continue; }
      edge = b.edges[*condition ? 0 : 1];
    } else return false;
    enqueue(edge);
  }
  return pending.empty();
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
  auto operations = cfg ? cfg.getAs<ArrayAttr>("operations") : ArrayAttr{};
  bool indexedSlots = operations && llvm::any_of(operations, [](Attribute a) {
    auto d = dyn_cast<DictionaryAttr>(a);
    StringRef name = contractString(d,"name");
    auto operands = contractI32Array(d,"operands");
    return (name == "atlas.virtual_slot_load" && operands.size() == 1) ||
           (name == "atlas.virtual_slot_store" && operands.size() == 2);
  });
  auto sourceValues = cfg ? cfg.getAs<ArrayAttr>("values") : ArrayAttr{};
  bool scalarShifts = sourceValues && llvm::any_of(sourceValues, [](Attribute a) {
    auto value = dyn_cast<DictionaryAttr>(a);
    StringRef def = contractString(value,"def");
    return def == "arith.shli" || def == "arith.shrui";
  });
  if (!dynamic && !explicitBuffer && !indexedSlots && !scalarShifts) return success();
  uint64_t outputFirst = 0, outputEnd = 0;
  bool readsOutput = false;
  if (explicitBuffer) {
    auto base = output ? output.getAs<IntegerAttr>("base") : IntegerAttr{};
    auto bytes = output ? output.getAs<IntegerAttr>("bytes") : IntegerAttr{};
    if (!output || output.size() != 2 || !base || !bytes ||
        !base.getType().isSignlessInteger(64) || !bytes.getType().isSignlessInteger(64) ||
        !atlasDMAMemorySpan(uint64_t(base.getInt()), uint64_t(bytes.getInt())) || base.getInt() > UINT32_MAX || bytes.getInt() <= 0 ||
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
      atlasProofRecord(AtlasProofKind::CapturedAddress);
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
      atlasProofRecord(AtlasProofKind::CapturedAddress);
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d || contractString(d,"name") == "atlas.virtual_output_bf16")
        return module.emitOpError("explicit output buffer cannot contain indexed outputs");
    }
  }
  CapturedAddressProof source;
  if (!source.parse(cfg)) return module.emitOpError("bounded DMA requires a valid source CFG");
  // Recheck the retained scalar language even when no dynamic memory address
  // uses the shift. Physical scalar shifts mask their distance; arith shifts
  // become poison at32, so matching machine/source mutations must not bypass
  // the virtual language's admission rule.
  for (const auto &value : source.values) {
    if (value.def != "arith.shli" && value.def != "arith.shrui") continue;
    auto word = [&](int id) { return id >= 0 && id < int(source.values.size()) && source.values[id].type == "i32"; };
    auto range = value.type == "i32" && value.operands.size() == 2 && word(value.operands[0]) && word(value.operands[1])
        ? source.proveUnsigned(value.operands[1],value.block) : std::nullopt;
    if (!range || range->last >= 32)
      return module.emitOpError("retained scalar shift distance needs a proven unsigned range below32");
  }
  if (indexedSlots) {
    auto tiles = module->getAttrOfType<ArrayAttr>(kAtlasTileContract);
    if (!tiles) return module.emitOpError("indexed slot has no tile footprint");
    for (Attribute a : operations) {
      atlasProofRecord(AtlasProofKind::CapturedAddress);
      auto record = dyn_cast<DictionaryAttr>(a);
      StringRef name = contractString(record,"name");
      bool write = name == "atlas.virtual_slot_store";
      auto operands = contractI32Array(record,"operands");
      if ((name != "atlas.virtual_slot_load" && !write) || operands.size() != (write ? 2u : 1u)) continue;
      auto block = contractI32(record,"block"), offset = contractI32(record,"byte_offset"),
           words = contractI32(record,"slot_words"), word = contractI32(record,"slot_word");
      StringRef kind = contractString(record,"kind");
      unsigned width = kind == (write ? "sw" : "lw") ? 4 : kind == (write ? "sh" : "lhu") ? 2 : kind == (write ? "sb" : "lbu") ? 1 : 0;
      auto range = block ? source.proveUnsigned(operands[write ? 1 : 0],*block) : std::nullopt;
      auto commands = contractI32Array(record,"tile_commands");
      if (!width || !range || !offset || *offset < 0 || *offset % width || !word || !words || *words <= 0 ||
          uint64_t(*offset) + (uint64_t(range->last)+1)*width > uint64_t(*words)*4 ||
          commands.size() != 1 || commands[0] < 0 || commands[0] >= int32_t(tiles.size()))
        return module.emitOpError("indexed slot needs an independently proven in-bounds source range");
      auto tile = dyn_cast<DictionaryAttr>(tiles[commands[0]]);
      if (contractString(tile,"kind") != (write ? "slot_index_store" : "slot_index_load") ||
          contractI32(tile,"vmem_byte") != uint32_t(*word)*4 + uint32_t(*offset) ||
          !contractI32(tile,"bytes") || *contractI32(tile,"bytes") <= 0 ||
          uint64_t(*contractI32(tile,"bytes")) < (uint64_t(range->last)+1)*width ||
          uint32_t(*contractI32(tile,"bytes")) % width ||
          uint64_t(*offset) + uint32_t(*contractI32(tile,"bytes")) > uint64_t(*words)*4)
        return module.emitOpError("indexed slot tile footprint does not cover its proven source range");
    }
  }
  auto fail = [&](StringRef reason) { return module.emitOpError("bounded DMA address differs from its guarded source induction: ") << reason; };
  for (Attribute a : dma) {
    atlasProofRecord(AtlasProofKind::CapturedAddress);
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
          atlasProofRecord(AtlasProofKind::CapturedAddress);
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
