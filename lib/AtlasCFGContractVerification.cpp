#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
namespace {
bool tracked(Type t) { return t.isInteger(1) || t.isInteger(32) || isa<VirtualBF16Type, VirtualFP8Type>(t); }
bool scalar(Type t) { return t.isInteger(1) || t.isInteger(32); }
std::string typeName(Type t) { return t.isInteger(1) ? "i1" : t.isInteger(32) ? "i32" : isa<VirtualBF16Type>(t) ? "bf16" : "fp8"; }
struct ValueRecord {
  int32_t id, block, reg;
  std::string type, def, kind;
  std::vector<int32_t> operands;
  uint32_t constant = 0;
  int32_t predicate = -1;
};
struct BlockRecord {
  int32_t id, condition;
  std::vector<int32_t> args, live, operations, edges;
};
struct EdgeRecord { int32_t id, from, to; std::vector<int32_t> incoming; };

// Expression roots are checked SSA origins, not recursively expanded source
// expressions. Joining differing physical expressions loses information rather
// than choosing one path. This bounds the state even through helper cycles.
struct Expr;
using E = std::shared_ptr<const Expr>;
struct Expr { std::string key, kind; E a, b; uint32_t literal = 0; int32_t id = -1; };
E node(std::string kind, E a = {}, E b = {}, uint32_t literal = 0, int32_t id = -1) {
  auto e = std::make_shared<Expr>(); e->kind = kind; e->a = a; e->b = b; e->literal = literal; e->id = id;
  e->key = kind + ":" + std::to_string(literal) + ":" + std::to_string(id) + "(" + (a ? a->key : "?") + "," + (b ? b->key : "?") + ")";
  return e;
}
E origin(int id) { return node("origin", {}, {}, 0, id); }
E literal(uint32_t n) { return node("literal", {}, {}, n); }
bool same(E a, E b) { return (!a && !b) || (a && b && a->key == b->key); }
bool isLiteral(E e, uint32_t v) { return e && e->kind == "literal" && e->literal == v; }
E compare(std::string kind, E a, E b) {
  if (!a || !b) return {};
  if (kind == "eq" || kind == "ne") if (a->key > b->key) std::swap(a, b);
  return node(kind, a, b);
}
E invert(E e) {
  if (!e) return {};
  static const std::map<std::string, std::string> opposite = {{"eq","ne"},{"ne","eq"},{"slt","sge"},{"sge","slt"},{"ult","uge"},{"uge","ult"}};
  auto it = opposite.find(e->kind);
  return it == opposite.end() ? node("xor", e, literal(1)) : compare(it->second, e->a, e->b);
}
E binary(std::string kind, E a, E b) {
  if (!a || !b) return {};
  if (kind == "add" && isLiteral(b, 0)) return a;
  if (kind == "add" && isLiteral(a, 0)) return b;
  if (kind == "xor" && isLiteral(b, 1)) return invert(a);
  if (kind == "xor" && isLiteral(a, 1)) return invert(b);
  if (kind == "ult" && isLiteral(b, 1) && a->kind == "xor") return compare("eq", a->a, a->b);
  if (kind == "ult" && isLiteral(a, 0) && b->kind == "xor") return compare("ne", b->a, b->b);
  if (a->kind == "literal" && b->kind == "literal") {
    uint32_t x = a->literal, y = b->literal;
    if (kind == "add") return literal(x + y);
    if (kind == "xor") return literal(x ^ y);
    if (kind == "and") return literal(x & y);
    if (kind == "slt") return literal(int32_t(x) < int32_t(y));
    if (kind == "ult") return literal(x < y);
  }
  if (kind == "add" || kind == "xor" || kind == "and") if (a->key > b->key) std::swap(a, b);
  return node(kind, a, b);
}
E sourceCompare(int p, E a, E b) {
  switch (arith::CmpIPredicate(p)) {
  case arith::CmpIPredicate::eq: return compare("eq", a, b);
  case arith::CmpIPredicate::ne: return compare("ne", a, b);
  case arith::CmpIPredicate::slt: return compare("slt", a, b);
  case arith::CmpIPredicate::sgt: return compare("slt", b, a);
  case arith::CmpIPredicate::sle: return compare("sge", b, a);
  case arith::CmpIPredicate::sge: return compare("sge", a, b);
  case arith::CmpIPredicate::ult: return compare("ult", a, b);
  case arith::CmpIPredicate::ugt: return compare("ult", b, a);
  case arith::CmpIPredicate::ule: return compare("uge", b, a);
  case arith::CmpIPredicate::uge: return compare("uge", a, b);
  }
  return {};
}
struct State {
  std::array<E, 32> x{};
  std::array<int32_t, 64> tensor;
  std::set<int32_t> done;
  int32_t chosen = -1;
  State() { x[0] = literal(0); tensor.fill(-1); }
};
bool merge(State &to, const State &from) {
  bool change = false;
  for (unsigned i = 1; i < 32; ++i) if (to.x[i] && !same(to.x[i], from.x[i])) { to.x[i].reset(); change = true; }
  for (unsigned i = 0; i < 64; ++i) if (to.tensor[i] != -1 && to.tensor[i] != from.tensor[i]) { to.tensor[i] = -1; change = true; }
  for (auto i = to.done.begin(); i != to.done.end();) if (!from.done.count(*i)) { i = to.done.erase(i); change = true; } else ++i;
  if (to.chosen != from.chosen && to.chosen != -2) { to.chosen = -2; change = true; }
  return change;
}
}

FailureOr<DictionaryAttr> mlir::atlas::buildAtlasCFGContract(func::FuncOp function, ArrayRef<VirtualRegisterAssignment> registers) {
  Builder b(function.getContext());
  DenseMap<Value, int32_t> ids, regs;
  DenseMap<Block *, int32_t> blocks;
  DenseMap<Operation *, int32_t> operations;
  int32_t nextValue = 0, nextBlock = 0, nextOperation = 0;
  int32_t nextTile = function.getNumArguments() ? kMailboxPreludeCommands + function.getNumArguments() : 0, nextMXU = 0;
  for (auto r : registers) {
    if (!r.value || !tracked(r.value.getType()) || regs.count(r.value)) return function.emitOpError("CFG contract invalid placement claim");
    regs[r.value] = r.reg;
  }
  for (Block &block : function.getBody()) {
    blocks[&block] = nextBlock++;
    for (Value arg : block.getArguments()) if (tracked(arg.getType())) ids[arg] = nextValue++;
    for (Operation &op : block) {
      operations[&op] = nextOperation++;
      for (Value v : op.getResults()) if (tracked(v.getType())) ids[v] = nextValue++;
    }
  }
  if (regs.size() != ids.size()) return function.emitOpError("CFG contract requires every source scalar and tensor placement");
  auto i = [&](int32_t n) { return b.getI32IntegerAttr(n); };
  auto dictionary = [&](SmallVector<NamedAttribute> fields) { return b.getDictionaryAttr(fields); };
  auto field = [&](StringRef name, Attribute value) { return b.getNamedAttr(name, value); };
  SmallVector<Attribute> values, blockRecords, edgeRecords, operationRecords;
  std::vector<std::set<int32_t>> uses(nextBlock), defs(nextBlock), live(nextBlock);
  std::vector<BlockRecord> summaries(nextBlock);
  std::vector<EdgeRecord> edges;
  auto valueRecord = [&](Value value) -> LogicalResult {
    if (!regs.count(value)) return function.emitOpError("CFG contract missing source placement");
    std::string type = typeName(value.getType());
    unsigned reg = regs.lookup(value), width = type == "bf16" ? 2 : 1;
    if ((scalar(value.getType()) && (reg == 0 || reg >= 32)) || (!scalar(value.getType()) && (reg + width > 64 || (width == 2 && reg % 2)))) return function.emitOpError("CFG contract invalid register geometry");
    Operation *op = value.getDefiningOp();
    SmallVector<int32_t> operands;
    if (op) for (Value v : op->getOperands()) if (tracked(v.getType())) operands.push_back(ids.lookup(v));
    SmallVector<NamedAttribute> f = {field("id",i(ids.lookup(value))), field("block",i(blocks.lookup(value.getParentBlock()))), field("reg",i(reg)), field("type",b.getStringAttr(type)), field("def",b.getStringAttr(op ? op->getName().getStringRef() : "argument")), field("operands",b.getDenseI32ArrayAttr(operands))};
    if (auto c = dyn_cast_or_null<arith::ConstantOp>(op)) f.push_back(field("constant",i(uint32_t(cast<IntegerAttr>(c.getValue()).getValue().getSExtValue()))));
    if (auto cmp = dyn_cast_or_null<arith::CmpIOp>(op)) f.push_back(field("predicate",i(int32_t(cmp.getPredicate()))));
    if (op) if (auto kind = op->getAttrOfType<StringAttr>("kind")) f.push_back(field("kind",kind));
    values.push_back(dictionary(f));
    return success();
  };
  for (Block &block : function.getBody()) {
    int32_t id = blocks.lookup(&block);
    BlockRecord &r = summaries[id]; r.id = id; r.condition = -1;
    for (Value arg : block.getArguments()) if (tracked(arg.getType())) { if (failed(valueRecord(arg))) return failure(); r.args.push_back(ids.lookup(arg)); defs[id].insert(ids.lookup(arg)); }
    for (Operation &op : block) {
      for (Value v : op.getOperands()) if (tracked(v.getType()) && !defs[id].count(ids.lookup(v))) uses[id].insert(ids.lookup(v));
      for (Value v : op.getResults()) if (tracked(v.getType())) { if (failed(valueRecord(v))) return failure(); defs[id].insert(ids.lookup(v)); }
      if (!op.hasTrait<OpTrait::IsTerminator>() && !isa<VirtualStartOp,VirtualScaleConstantOp>(op)) {
        r.operations.push_back(operations.lookup(&op));
        SmallVector<int32_t> operands, results;
        for (Value v : op.getOperands()) if (tracked(v.getType())) operands.push_back(ids.lookup(v));
        for (Value v : op.getResults()) if (tracked(v.getType())) results.push_back(ids.lookup(v));
        // Source format and lifecycle determine command expansion counts;
        // neither planned instructions nor issued tags supply these facts.
        unsigned tileCount = atlasTileExpansion(op.getName().getStringRef()).commands;
        unsigned mxuCount = isa<VirtualMXUMatmulOp>(op) ? 3 :
            isa<VirtualMXULoadWeightOp,VirtualMXULoadAccFP8Op,VirtualMXULoadAccBF16Op,
                VirtualMXUResetOp,VirtualMXUAccumulateOp,VirtualMXUReadoutBF16Op,
                VirtualMXUReadoutFP8Op>(op) ? 1 : 0;
        SmallVector<int32_t> tileCommands, mxuCommands;
        for (unsigned n = 0; n < tileCount; ++n) tileCommands.push_back(nextTile++);
        for (unsigned n = 0; n < mxuCount; ++n) mxuCommands.push_back(nextMXU++);
        operationRecords.push_back(dictionary({field("id",i(operations.lookup(&op))),field("block",i(id)),field("name",b.getStringAttr(op.getName().getStringRef())),field("operands",b.getDenseI32ArrayAttr(operands)),field("results",b.getDenseI32ArrayAttr(results)),field("tile_commands",b.getDenseI32ArrayAttr(tileCommands)),field("mxu_commands",b.getDenseI32ArrayAttr(mxuCommands))}));
      }
    }
    auto edge = [&](Block *dest, ValueRange operands) {
      EdgeRecord e{int32_t(edges.size()), id, blocks.lookup(dest), {}};
      for (Value v : operands) if (tracked(v.getType())) e.incoming.push_back(ids.lookup(v));
      r.edges.push_back(e.id); edges.push_back(e);
    };
    if (auto br = dyn_cast<cf::BranchOp>(block.getTerminator())) edge(br.getDest(), br.getDestOperands());
    else if (auto br = dyn_cast<cf::CondBranchOp>(block.getTerminator())) { r.condition = ids.lookup(br.getCondition()); edge(br.getTrueDest(), br.getTrueDestOperands()); edge(br.getFalseDest(), br.getFalseDestOperands()); }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (int32_t id = nextBlock - 1; id >= 0; --id) {
      auto incoming = uses[id];
      for (int32_t edge : summaries[id].edges) {
        auto &e = edges[edge];
        for (int32_t v : live[e.to]) if (!defs[id].count(v)) incoming.insert(v);
        for (int32_t v : e.incoming) if (!defs[id].count(v)) incoming.insert(v);
      }
      if (incoming != live[id]) { live[id] = std::move(incoming); changed = true; }
    }
  }
  for (auto &r : summaries) {
    SmallVector<int32_t> l(live[r.id].begin(), live[r.id].end());
    blockRecords.push_back(dictionary({field("id",i(r.id)), field("condition",i(r.condition)), field("args",b.getDenseI32ArrayAttr(r.args)), field("live_in",b.getDenseI32ArrayAttr(l)), field("operations",b.getDenseI32ArrayAttr(r.operations)), field("edges",b.getDenseI32ArrayAttr(r.edges))}));
  }
  for (auto &e : edges) edgeRecords.push_back(dictionary({field("id",i(e.id)),field("from",i(e.from)),field("to",i(e.to)),field("incoming",b.getDenseI32ArrayAttr(e.incoming))}));
  return dictionary({field("values",b.getArrayAttr(values)),field("blocks",b.getArrayAttr(blockRecords)),field("edges",b.getArrayAttr(edgeRecords)),field("operations",b.getArrayAttr(operationRecords))});
}

LogicalResult mlir::atlas::verifyAtlasGeneratedCFGContract(
    const AtlasVerificationContext &ctx) {
  ModuleOp module = ctx.module;
  if (failed(requireAtlasGeneratedArtifact(module)))
    return failure();
  auto contract = dyn_cast<DictionaryAttr>(module->getAttr(kAtlasCFGContract));
  auto bad = [&]() { return module.emitOpError("CFG contract malformed source records"); };
  if (!contract || contract.size() != 4) return bad();
  auto vr = contract.getAs<ArrayAttr>("values"), br = contract.getAs<ArrayAttr>("blocks"), er = contract.getAs<ArrayAttr>("edges");
  auto opRecords = contract.getAs<ArrayAttr>("operations");
  if (!vr || !br || !er || !opRecords || br.empty()) return bad();
  std::map<int32_t, DictionaryAttr> sourceOperations;
  std::vector<ValueRecord> values;
  std::vector<BlockRecord> blocks;
  std::vector<EdgeRecord> edges;
  for (Attribute raw : vr) {
    auto d = dyn_cast<DictionaryAttr>(raw); if (!d) return bad();
    auto id = contractI32(d,"id"), block = contractI32(d,"block"), reg = contractI32(d,"reg");
    std::string type = contractString(d,"type").str(), def = contractString(d,"def").str();
    if (!id || *id != int32_t(values.size()) || !block || *block < 0 || *block >= int32_t(br.size()) || !reg || def.empty() || !d.getAs<DenseI32ArrayAttr>("operands")) return bad();
    if ((type == "i1" || type == "i32") ? (*reg <= 0 || *reg >= 32) : (type != "bf16" && type != "fp8") || *reg < 0 || *reg + (type == "bf16" ? 2 : 1) > 64 || (type == "bf16" && *reg % 2)) return bad();
    ValueRecord r{*id,*block,*reg,type,def,contractString(d,"kind").str(),contractI32Array(d,"operands")};
    if (def == "arith.constant") { auto n = contractI32(d,"constant"); if (!n || !r.operands.empty()) return bad(); r.constant = uint32_t(*n); }
    if (def == "arith.cmpi") { auto p = contractI32(d,"predicate"); if (!p || *p < 0 || *p > 9 || r.operands.size() != 2) return bad(); r.predicate = *p; }
    if (def == "arith.addi" && r.operands.size() != 2) return bad();
    values.push_back(r);
  }
  for (auto &v : values) for (int32_t operand : v.operands) if (operand < 0 || operand >= int32_t(values.size())) return bad();
  std::set<int32_t> expectedOperations;
  for (Attribute raw : br) {
    auto d = dyn_cast<DictionaryAttr>(raw); if (!d || d.size() != 6) return bad();
    auto id = contractI32(d,"id"), condition = contractI32(d,"condition");
    for (StringRef name : {"args","live_in","operations","edges"}) if (!d.getAs<DenseI32ArrayAttr>(name)) return bad();
    if (!id || *id != int32_t(blocks.size()) || !condition || *condition < -1 || *condition >= int32_t(values.size())) return bad();
    BlockRecord r{*id,*condition,contractI32Array(d,"args"),contractI32Array(d,"live_in"),contractI32Array(d,"operations"),contractI32Array(d,"edges")};
    if (r.edges.size() > 2 || ((*condition >= 0) != (r.edges.size() == 2))) return bad();
    for (int32_t v : r.args) if (v < 0 || v >= int32_t(values.size()) || values[v].block != *id || values[v].def != "argument") return bad();
    for (int32_t v : r.live) if (v < 0 || v >= int32_t(values.size())) return bad();
    for (int32_t op : r.operations) if (op < 0 || !expectedOperations.insert(op).second) return bad();
    if (*condition >= 0 && values[*condition].type != "i1") return bad();
    blocks.push_back(r);
  }
  for (Attribute raw : er) {
    auto d = dyn_cast<DictionaryAttr>(raw); if (!d || d.size() != 4) return bad();
    auto id = contractI32(d,"id"), from = contractI32(d,"from"), to = contractI32(d,"to");
    if (!id || *id != int32_t(edges.size()) || !from || *from < 0 || *from >= int32_t(blocks.size()) || !to || *to < 0 || *to >= int32_t(blocks.size()) || !d.getAs<DenseI32ArrayAttr>("incoming")) return bad();
    EdgeRecord e{*id,*from,*to,contractI32Array(d,"incoming")};
    if (e.incoming.size() != blocks[e.to].args.size()) return bad();
    for (size_t n = 0; n < e.incoming.size(); ++n) if (e.incoming[n] < 0 || e.incoming[n] >= int32_t(values.size()) || values[e.incoming[n]].type != values[blocks[e.to].args[n]].type) return bad();
    edges.push_back(e);
  }
  std::set<int32_t> ownedEdges;
  for (auto &b : blocks) for (int32_t e : b.edges) if (e < 0 || e >= int32_t(edges.size()) || edges[e].from != b.id || !ownedEdges.insert(e).second) return bad();
  if (ownedEdges.size() != edges.size()) return bad();
  for (Attribute raw : opRecords) {
    auto d = dyn_cast<DictionaryAttr>(raw); if (!d || d.size() != 7) return bad();
    auto id = contractI32(d,"id"), owner = contractI32(d,"block");
    if (!id || !expectedOperations.count(*id) || !owner || *owner < 0 || *owner >= int32_t(blocks.size()) || contractString(d,"name").empty() || !d.getAs<DenseI32ArrayAttr>("operands") || !d.getAs<DenseI32ArrayAttr>("results") || !d.getAs<DenseI32ArrayAttr>("tile_commands") || !d.getAs<DenseI32ArrayAttr>("mxu_commands") || !sourceOperations.emplace(*id,d).second) return bad();
    for (StringRef key : {"operands","results"})
      for (int32_t v : contractI32Array(d,key)) if (v < 0 || v >= int32_t(values.size())) return bad();
  }
  if (sourceOperations.size() != expectedOperations.size()) return bad();
  assert(ctx.stream && "a CFG-contract stream is decoded");
  const std::optional<AtlasStream> &stream = ctx.stream;
  auto &ops = stream->ops;
  if (ops.empty())
    return module.emitOpError("CFG contract requires a nonempty issued instruction stream");
  std::vector<size_t> starts(blocks.size(), ops.size()), edgeStarts(edges.size(), ops.size());
  std::set<int32_t> seenOperations, seenResults, seenBranches, seenEdges;
  for (size_t pc = 0; pc < ops.size(); ++pc) {
    Operation *op = ops[pc];
    for (StringRef name : {kAtlasTagCFGBlock,kAtlasTagCFGEdge,kAtlasTagCFGBranch,kAtlasTagCFGSource,kAtlasTagCFGOperation,kAtlasTagScalarResult,kAtlasTagTensorResult,kAtlasTagScalarArgument}) if (op->hasAttr(name) && !contractTag(op,name)) return op->emitOpError("CFG contract tags require nonnegative i32 identities");
    auto owner = contractTag(op,kAtlasTagCFGBlock);
    if (!owner) { if (!isa<DelayOp>(op) && !isNop(op)) return op->emitOpError("CFG contract instruction lacks source block ownership"); continue; }
    if (*owner >= int32_t(blocks.size())) return bad();
    starts[*owner] = std::min(starts[*owner],pc);
    if (auto e = contractTag(op,kAtlasTagCFGEdge)) { if (*e >= int32_t(edges.size()) || edges[*e].from != *owner) return bad(); edgeStarts[*e] = std::min(edgeStarts[*e],pc); if (isa<JumpOp>(op) && !seenEdges.insert(*e).second) return op->emitOpError("CFG contract repeated edge redirect"); }
    if (auto id = contractTag(op,kAtlasTagCFGSource))
      if (!sourceOperations.count(*id) || contractI32(sourceOperations[*id],"block") != *owner) return bad();
    if (auto id = contractTag(op,kAtlasTagCFGOperation)) if (contractTag(op,kAtlasTagCFGSource) != id || !expectedOperations.count(*id) || !seenOperations.insert(*id).second || std::find(blocks[*owner].operations.begin(),blocks[*owner].operations.end(),*id) == blocks[*owner].operations.end()) return op->emitOpError("CFG contract invalid or duplicate source operation identity");
    for (StringRef name : {kAtlasTagScalarResult,kAtlasTagTensorResult}) if (auto id = contractTag(op,name)) {
      if (*id >= int32_t(values.size()) || values[*id].block != *owner ||
          !seenResults.insert(*id).second ||
          ((name == kAtlasTagScalarResult) != (values[*id].type == "i1" || values[*id].type == "i32")))
        return op->emitOpError("CFG contract invalid or duplicate source result identity");
      auto &v = values[*id];
      auto source = contractTag(op,kAtlasTagCFGSource);
      if (v.def == "argument") {
        if (v.block != 0 || name != kAtlasTagScalarResult || source)
          return op->emitOpError("CFG contract invalid argument result site");
      } else {
        if (!source || !sourceOperations.count(*source))
          return op->emitOpError("CFG contract result lacks its source producer");
        auto results = contractI32Array(sourceOperations[*source],"results");
        if (std::find(results.begin(),results.end(),*id) == results.end())
          return op->emitOpError("CFG contract result belongs to another source producer");
      }
      int32_t destination = -1;
      if (name == kAtlasTagScalarResult) {
        if (auto a = dyn_cast<ALURegOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<ALUImmOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<UpperOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<ScalarLoadOp>(op); a && a.getKind() == "lw") destination = a.getDst();
      } else {
        if (auto a = dyn_cast<VPUUnaryOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<VPUBinaryOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<MXUPopOp>(op)) destination = a.getDst();
        else if (auto a = dyn_cast<VLoadOp>(op)) destination = int32_t(a.getDst()) - (v.type == "bf16" ? 1 : 0);
      }
      if (destination != v.reg)
        return op->emitOpError("CFG contract result tag does not identify its physical producer destination");
    }
    for (auto binding : {std::pair<StringRef,StringRef>{kAtlasTagTileCommand,"tile_commands"}, {kAtlasTagMXUCommand,"mxu_commands"}}) {
      if (auto command = contractTag(op,binding.first)) {
        auto source = contractTag(op,kAtlasTagCFGSource);
        if (source) {
          auto commands = contractI32Array(sourceOperations[*source],binding.second);
          if (std::find(commands.begin(),commands.end(),*command) == commands.end())
            return op->emitOpError("CFG contract issued command belongs to another source operation");
        } else if (binding.first != kAtlasTagTileCommand || *owner != 0 ||
                   blocks[0].args.empty() || *command >= int32_t(kMailboxPreludeCommands + blocks[0].args.size())) {
          return op->emitOpError("CFG contract issued command lacks its source operation");
        }
      }
    }
    if (isa<BranchOp,JumpOp>(op) && !isExactAtlasNop(pc + 1 < ops.size() ? ops[pc+1] : nullptr))
      return op->emitOpError("CFG contract redirect requires an exact x0 NOP delay slot");
    if (auto id = contractTag(op,kAtlasTagCFGBranch)) if (!isa<BranchOp>(op) || *id != *owner || blocks[*id].condition < 0 || !seenBranches.insert(*id).second) return op->emitOpError("CFG contract invalid source branch identity");
    if (Attribute h = op->getAttr(kAtlasTagCFGHelper)) {
      auto name = dyn_cast<StringAttr>(h);
      if (!name || name.getValue() != "pack")
        return op->emitOpError("CFG contract unsupported helper scope");
      auto source = contractTag(op,kAtlasTagCFGSource);
      if (source) {
        if (!sourceOperations.count(*source) ||
            contractString(sourceOperations[*source],"name") != "atlas.virtual_pack_fp8")
          return op->emitOpError("CFG contract PACK helper requires an actual source PACK operation");
      } else if (!isa<DelayOp>(op) && !isNop(op)) {
        return op->emitOpError("CFG contract PACK helper instruction lacks its source PACK ownership");
      }
    }
  }
  if (starts[0] != 0 || contractTag(ops[0],kAtlasTagCFGBlock) != 0)
    return module.emitOpError("CFG contract emitted entry must be source block zero");
  if (seenOperations != expectedOperations || seenEdges.size() != edges.size()) return module.emitOpError("CFG contract requires exactly one issued operation and edge site for every source record");
  for (auto &b : blocks) if (starts[b.id] == ops.size() || (b.condition >= 0 && !seenBranches.count(b.id))) return module.emitOpError("CFG contract missing source block or branch site");
  for (auto &v : values) if ((v.def != "argument" || v.block == 0) && !seenResults.count(v.id)) return module.emitOpError("CFG contract missing source result site");
  auto target = [&](Operation *op) { Operation *t = stream->targetOf.lookup(op); return t ? ctx.pcOf.lookup(t) : ops.size(); };
  auto matches = [&](State &s, const ValueRecord &v, int32_t expected) { return v.type == "i1" || v.type == "i32" ? same(s.x[v.reg],origin(expected)) : s.tensor[v.reg] == expected && (v.type != "bf16" || s.tensor[v.reg+1] == expected); };
  auto install = [&](State &s, const ValueRecord &v) { if (v.type == "i1" || v.type == "i32") s.x[v.reg] = origin(v.id); else { s.tensor[v.reg] = v.id; if (v.type == "bf16") s.tensor[v.reg+1] = v.id; } };
  for (auto &block : blocks) {
    State initial;
    for (int32_t v : block.live) install(initial,values[v]);
    if (block.id != 0) for (int32_t v : block.args) install(initial,values[v]);
    std::vector<State> entries(ops.size()); std::vector<bool> reached(ops.size(),false);
    std::deque<size_t> work; entries[starts[block.id]] = initial; reached[starts[block.id]] = true; work.push_back(starts[block.id]);
    auto propagate = [&](size_t pc, const State &s) { if (pc >= ops.size()) return; if (!reached[pc]) { reached[pc] = true; entries[pc] = s; work.push_back(pc); } else if (merge(entries[pc],s)) work.push_back(pc); };
    auto transfer = [&](size_t pc, State &s, bool diagnose) -> LogicalResult {
      Operation *op = ops[pc];
      auto error = [&](StringRef message) -> LogicalResult {
        if (diagnose) return op->emitOpError("CFG contract ") << message;
        return failure();
      };
      auto owner = contractTag(op,kAtlasTagCFGBlock); if (owner && *owner != block.id) return error("path changes source block without its source edge");
      const State before = s;
      auto tensorRead = [&](unsigned reg, int32_t id) {
        if (id < 0 || id >= int32_t(values.size()) || reg >= 64) return false;
        return before.tensor[reg] == id && (values[id].type != "bf16" || (reg + 1 < 64 && before.tensor[reg+1] == id));
      };
      // Every actual tensor write kills the previous physical origin, even
      // when the instruction has no source identity. These widths follow the
      // machine opcode geometry, independently of timing-provider policies.
      auto killTensor = [&](unsigned reg, unsigned width) {
        for (unsigned half = 0; half < width && reg + half < 64; ++half)
          s.tensor[reg + half] = -1;
      };
      if (auto write = dyn_cast<VLoadOp>(op)) killTensor(write.getDst(),1);
      else if (auto write = dyn_cast<VPUUnaryOp>(op)) killTensor(write.getDst(),2);
      else if (auto write = dyn_cast<VPUBinaryOp>(op)) killTensor(write.getDst(),2);
      else if (auto write = dyn_cast<VPUReduceOp>(op)) killTensor(write.getDst(),2);
      else if (auto write = dyn_cast<XLUTransposeOp>(op)) killTensor(write.getDst(),1);
      else if (auto write = dyn_cast<VLIOp>(op))
        killTensor(write.getDst(),write.getMode() == "all" || write.getMode() == "row" ? 2 : 1);
      else if (auto write = dyn_cast<MXUPopOp>(op))
        killTensor(write.getDst(),write.getFormat() == "bf16" ? 2 : 1);
      else if (auto write = dyn_cast<VPUPackOp>(op))
        killTensor(write.getDst(),write.getDirection() == "fp8_to_bf16" ? 2 : 1);
      if (auto source = contractTag(op,kAtlasTagCFGSource)) {
        auto record = sourceOperations[*source];
        auto operands = contractI32Array(record,"operands"), results = contractI32Array(record,"results");
        std::string name = contractString(record,"name").str();
        if (auto push = dyn_cast<MXUPushOp>(op)) {
          size_t n = name == "atlas.virtual_mxu_matmul" && push.getKind() == "weight_fp8" ? 1 : 0;
          if (n >= operands.size() || !tensorRead(push.getSrc(),operands[n])) return error("MXU producer read lost its source tensor origin");
        }
        if (auto matmul = dyn_cast<MXUMatmulOp>(op))
          if (operands.empty() || !tensorRead(matmul.getSrc(),operands[0])) return error("MXU multiply read lost its source tensor origin");
        if (auto store = dyn_cast<VStoreOp>(op)) {
          int32_t expected = -1;
          if (name == "atlas.virtual_pack_fp8" && !results.empty()) expected = results[0];
          else if (!operands.empty()) expected = operands[0];
          if (expected < 0 || store.getSrc() >= 64 || before.tensor[store.getSrc()] != expected) return error("memory store read lost its source tensor origin");
        }
        if (auto pack = dyn_cast<VPUPackOp>(op)) {
          if (operands.empty() || results.empty() || !tensorRead(pack.getSrc(),operands[0])) return error("PACK read lost its source tensor origin");
          // The owned relayout preserves the logical result identity; the
          // buffer contract checks its conversion scale and memory layout.
        }
      }
      E produced;
      if (auto alu = dyn_cast<ALURegOp>(op)) {
        std::string kind = alu.getKind().str(); if (kind == "sltu") kind = "ult";
        produced = binary(kind,s.x[alu.getLhs()],s.x[alu.getRhs()]); if (alu.getDst()) s.x[alu.getDst()] = produced;
      } else if (auto alu = dyn_cast<ALUImmOp>(op)) {
        std::string kind = alu.getKind().str(); kind = kind == "addi" ? "add" : kind == "xori" ? "xor" : kind == "andi" ? "and" : kind == "sltiu" ? "ult" : kind;
        produced = binary(kind,s.x[alu.getSrc()],literal(uint32_t(alu.getImmediateAttr().getValue().getSExtValue()))); if (alu.getDst()) s.x[alu.getDst()] = produced;
      } else if (auto upper = dyn_cast<UpperOp>(op)) { produced = upper.getKind() == "lui" ? literal(upper.getImmediate() << 12) : E{}; if (upper.getDst()) s.x[upper.getDst()] = produced; }
      else if (auto load = dyn_cast<ScalarLoadOp>(op)) {
        // Every XRF load replaces its previous source origin. SELI and SELD
        // write the separate scale bank and leave scalar origins intact.
        if (load.getKind() != "seli" && load.getKind() != "seld" && load.getDst())
          s.x[load.getDst()].reset();
        if (auto id = contractTag(op,kAtlasTagScalarArgument)) {
          if (*id >= int32_t(values.size()) || values[*id].def != "argument" || values[*id].block != 0 || load.getKind() != "lw" || load.getDst() != unsigned(values[*id].reg) || contractTag(op,kAtlasTagTileCommand) != kMailboxPreludeCommands + *id) return error("invalid mailbox scalar origin");
          produced = node("mailbox",{}, {},0,*id); s.x[load.getDst()] = produced;
        }
      } else if (auto csr = dyn_cast<CSROp>(op)) { if (csr.getDst()) s.x[csr.getDst()].reset(); }
      else if (auto unary = dyn_cast<VPUUnaryOp>(op)) {
        if (unary.getKind() == "mov") {
          s.tensor[unary.getDst()] = before.tensor[unary.getSrc()];
          s.tensor[unary.getDst()+1] = before.tensor[unary.getSrc()+1];
        }
      } else if (auto load = dyn_cast<VLoadOp>(op)) {
        if (auto source = contractTag(op,kAtlasTagCFGSource)) {
          auto results = contractI32Array(sourceOperations[*source],"results");
          if (results.size() != 1) return error("tensor load has no unique source result");
          auto &v = values[results[0]];
          if (load.getDst() < unsigned(v.reg) || load.getDst() >= unsigned(v.reg + (v.type == "bf16" ? 2 : 1)))
            return error("tensor load destination differs from its source result");
          s.tensor[load.getDst()] = v.id;
        }
      }
      else if (auto pack = dyn_cast<VPUPackOp>(op)) {
        if (auto source = contractTag(op,kAtlasTagCFGSource)) {
          auto results = contractI32Array(sourceOperations[*source],"results");
          if (!results.empty()) s.tensor[pack.getDst()] = results[0];
        }
      }
      if (auto id = contractTag(op,kAtlasTagScalarResult)) {
        auto &v = values[*id]; E expected;
        if (v.def == "arith.constant") expected = literal(v.constant);
        else if (v.def == "arith.addi") expected = binary("add",origin(v.operands[0]),origin(v.operands[1]));
        else if (v.def == "arith.cmpi") expected = sourceCompare(v.predicate,origin(v.operands[0]),origin(v.operands[1]));
        else if (v.def == "argument") { expected = node("mailbox",{}, {},0,v.id); if (v.type == "i1") expected = binary("and",expected,literal(1)); }
        if (!expected || !same(s.x[v.reg],expected)) return error("issued scalar expression differs from source definition");
        install(s,v);
      }
      if (auto id = contractTag(op,kAtlasTagTensorResult)) {
        auto &v = values[*id];
        if (v.def == "atlas.virtual_vpu_unary") {
          auto unary = dyn_cast<VPUUnaryOp>(op); if (!unary || unary.getDst() != unsigned(v.reg) || unary.getKind() != v.kind || v.operands.size() != 1) return error("tensor source unary fields differ");
          // The producer may overwrite its source only when placement permits;
          // compare the entry state before the physical write below.
          if (!tensorRead(unary.getSrc(),v.operands[0])) return error("tensor unary operand lost its source origin");
        } else if (v.def == "atlas.virtual_vpu_binary") {
          auto binary = dyn_cast<VPUBinaryOp>(op); if (!binary || binary.getDst() != unsigned(v.reg) || binary.getKind() != v.kind || v.operands.size() != 2 || !tensorRead(binary.getLhs(),v.operands[0]) || !tensorRead(binary.getRhs(),v.operands[1])) return error("tensor binary fields or origins differ");
        } else if (!isa<VLoadOp,MXUPopOp>(op)) return error("unsupported tensor source result site");
        if (isa<VLoadOp>(op) && !matches(s,v,v.id))
          return error("tensor result is claimed before all of its halves are produced");
        install(s,v);
      }
      if (auto id = contractTag(op,kAtlasTagCFGOperation)) { if (s.done.count(*id)) return error("source operation repeats inside one source block visit"); s.done.insert(*id); }
      if (auto branch = dyn_cast<BranchOp>(op)) {
        if (contractTag(op,kAtlasTagCFGBranch)) {
          if (block.edges.size() != 2 || branch.getKind() != "bne" || branch.getRhs() != 0 || !same(s.x[branch.getLhs()],origin(block.condition)) || target(op) != edgeStarts[block.edges[0]]) return error("source branch condition, polarity or true target differs");
          size_t fallthrough = pc + 2; if (fallthrough != edgeStarts[block.edges[1]]) return error("source branch false target differs");
        } else if (!op->hasAttr(kAtlasTagCFGHelper)) return error("unowned conditional redirect changes source visits");
      }
      if (auto jump = dyn_cast<JumpOp>(op)) {
        auto id = contractTag(op,kAtlasTagCFGEdge); if (!id || jump.getKind() != "jal" || jump.getDst() != 0 || edges[*id].from != block.id || target(op) != starts[edges[*id].to]) return error("source edge target differs or redirect is unowned");
        if (block.condition >= 0 && s.chosen != *id) return error("source conditional edge does not match branch outcome");
        for (int32_t expected : block.operations) if (!s.done.count(expected)) return error("source operation is skipped before source edge");
        auto &edge = edges[*id]; auto &dest = blocks[edge.to];
        for (size_t n = 0; n < edge.incoming.size(); ++n) if (!matches(s,values[dest.args[n]],edge.incoming[n])) return error("simultaneous source edge copy lost its incoming origin");
        for (int32_t v : dest.live) if (!matches(s,values[v],v)) return error("source edge clobbers a live-through origin");
      }
      if (isa<TrapOp>(op)) { if (!block.edges.empty()) return error("source block exits before its source edge"); for (int32_t expected : block.operations) if (!s.done.count(expected)) return error("source return skips an operation"); }
      return success();
    };
    while (!work.empty()) {
      size_t pc = work.front(); work.pop_front(); State s = entries[pc];
      if (failed(transfer(pc,s,false))) continue;
      Operation *op = ops[pc];
      if (isa<JumpOp,TrapOp>(op)) continue;
      if (isa<BranchOp>(op)) {
        State taken = s, fall = s;
        if (contractTag(op,kAtlasTagCFGBranch)) { taken.chosen = block.edges[0]; fall.chosen = block.edges[1]; }
        propagate(target(op),taken); propagate(pc+2,fall);
      } else propagate(pc+1,s);
    }
    for (size_t pc = 0; pc < ops.size(); ++pc) if (reached[pc]) { State s = entries[pc]; if (failed(transfer(pc,s,true))) return failure(); }
    if (block.edges.empty()) {
      bool returned = false;
      for (size_t pc = 0; pc < ops.size(); ++pc)
        returned |= reached[pc] && isa<TrapOp>(ops[pc]);
      if (!returned) return module.emitOpError("CFG contract source return has no emitted exit");
    }
    for (int32_t e : block.edges) {
      bool visited = false; for (size_t pc = 0; pc < ops.size(); ++pc) if (reached[pc] && isa<JumpOp>(ops[pc]) && contractTag(ops[pc],kAtlasTagCFGEdge) == e) visited = true;
      if (!visited) return module.emitOpError("CFG contract source edge has no emitted execution path");
    }
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasGeneratedCFGContract(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasGeneratedCFGContract(*ctx);
}
