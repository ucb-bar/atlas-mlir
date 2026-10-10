#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "VerificationTestSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <functional>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral reader = "buffer contract: VLOAD reads a word its writer did not write";
constexpr StringLiteral capture = "buffer contract: DMA store captures a word its VSTORE did not write";
constexpr StringLiteral mailbox = "buffer contract: mailbox LW reads a word its mailbox DMA load did not write";
constexpr StringLiteral interleave = "buffer contract: PACK relayout word differs from the selected row interleave";
constexpr StringLiteral stale = "buffer contract: PACK relayout word is not a copy of its current raw conversion store";
constexpr StringLiteral inconsistent = "buffer contract has malformed or inconsistent source records";
constexpr StringLiteral sourceText = R"mlir(module {
  func.func @buffer(%flag: i1, %count: i32) -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64,
    atlas.control_dram_base = 2415927296 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %p = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %s2 = "atlas.virtual_output_bf16"(%s1, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
})mlir";

const FixedResourcePlacement &fixed() {
  static VirtualAllocationPlan plan;
  return plan.fixed();
}
bool accepts(ModuleOp module) { diagnostics.clear(); return succeeded(verifyAtlasGeneratedBufferContract(module)); }
bool rejects(ModuleOp module, StringRef message) { return !accepts(module) && diagnosed(message); }
int64_t i32(Operation *op, StringRef name) { return op->getAttrOfType<IntegerAttr>(name).getInt(); }
void set(Operation *op, StringRef name, int64_t value) { op->setAttr(name, Builder(op->getContext()).getI32IntegerAttr(value)); }

OwningOpRef<ModuleOp> lowered(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(sourceText, &context);
  PassManager passes(&context);
  if (!module || failed(parsePassPipeline("lower-atlas-virtual-to-machine", passes)) || failed(passes.run(*module))) return nullptr;
  return module;
}
Operation *find(ModuleOp module, function_ref<bool(Operation &)> wanted) {
  for (Operation &op : module.getBody()->getOperations())
    if (wanted(op)) return &op;
  return nullptr;
}
bool helper(Operation &op) { return op.hasAttr(kAtlasTagCFGHelper); }
// Rethreads the state chain: `op` moves to just before `anchor`.
void moveBefore(Operation *op, Operation *anchor) {
  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
  op->moveBefore(anchor);
  op->setOperand(0, anchor->getOperand(0));
  anchor->setOperand(0, op->getResult(0));
}

// Records are retained from source: readers, writers and the PACK's scale code.
void records(MLIRContext &context) {
  auto module = lowered(context);
  check(bool(module), "literal PACK source lowers"); if (!module) return;
  auto contract = (*module)->getAttrOfType<DictionaryAttr>(kAtlasBufferContract);
  auto tiles = (*module)->getAttrOfType<ArrayAttr>(kAtlasTileContract);
  check(contract && tiles, "lowering retains the buffer contract"); if (!contract || !tiles) return;
  auto packs = contract.getAs<ArrayAttr>("packs"), reads = contract.getAs<ArrayAttr>("reads");
  check(packs.size() == 1 && reads.size() == 7, "two mailbox LWs, two input halves, one relayout and two output captures");
  auto pack = cast<DictionaryAttr>(packs[0]);
  auto vmem = [&](int64_t id) { return integer(cast<DictionaryAttr>(tiles[id]), "vmem_byte"); };
  check(integer(pack, "source") == 2 && integer(pack, "scale_code") == 127 && vmem(integer(pack, "store")) == fixed().packWord * 4 && vmem(integer(pack, "load")) == fixed().packRelayoutWord * 4, "PACK source, scale and scratch endpoints");
  auto argument = cast<DictionaryAttr>(reads[1]);
  check(integer(argument, "writer") == 0 && integer(argument, "vmem_byte") == 4 && integer(argument, "bytes") == 4 && integer(argument, "word") == 1, "second argument reads word 1 of the mailbox DMA");
  unsigned relayouts = 0, captures = 0;
  for (Attribute a : reads) {
    auto r = cast<DictionaryAttr>(a);
    relayouts += r.getAs<StringAttr>("layout").getValue() == "pack";
    captures += cast<DictionaryAttr>(tiles[integer(r, "command")]).getAs<StringAttr>("kind").getValue() == "dma_store";
  }
  check(relayouts == 1 && captures == 2, "one relayout read and one capture per output half");
  check(accepts(*module), "lowered PACK, mailbox and boundary I/O preserve their words", diagnostics);

  auto source = parseSourceString<ModuleOp>(sourceText, &context);
  auto function = *source->getOps<func::FuncOp>().begin();
  auto cfg = (*module)->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
  auto rebuilt = buildAtlasBufferContract(function, cfg, tiles);
  check(succeeded(rebuilt) && *rebuilt == contract, "the builder is a function of live source and the two contracts");
  SmallVector<Attribute> operations(cfg.getAs<ArrayAttr>("operations").begin(), cfg.getAs<ArrayAttr>("operations").end());
  for (Attribute &a : operations) {
    auto op = cast<DictionaryAttr>(a);
    if (op.getAs<StringAttr>("name").getValue() != VirtualPackFP8Op::getOperationName()) continue;
    auto commands = op.getAs<DenseI32ArrayAttr>("tile_commands").asArrayRef();
    NamedAttrList fields(op);
    fields.set("tile_commands", Builder(&context).getDenseI32ArrayAttr({commands[1], commands[0]}));
    a = fields.getDictionary(&context);
  }
  NamedAttrList swapped(cfg);
  swapped.set("operations", Builder(&context).getArrayAttr(operations));
  diagnostics.clear();
  check(failed(buildAtlasBufferContract(function, swapped.getDictionary(&context), tiles)) && diagnosed("buffer contract cannot derive tile readers and PACK endpoints"), "PACK endpoints must be its raw VSTORE then its relayout VLOAD");
}

// Each case lowers afresh, corrupts one site and expects one diagnostic.
void pack(MLIRContext &context) {
  const FixedResourcePlacement &f = fixed();
  auto limit = [&](Operation &op) { auto a = dyn_cast<ALUImmOp>(op); return a && helper(op) && a.getDst() == f.packRowsReg; };
  auto backedge = [](Operation &op) { return isa<BranchOp>(op) && helper(op); };
  auto copyLoad = [&](Operation &op) { auto l = dyn_cast<ScalarLoadOp>(op); return l && helper(op) && l.getKind() == "lw"; };
  auto rawStore = [](Operation &op) { return isa<VStoreOp>(op) && helper(op); };
  auto relayout = [](Operation &op) { return isa<VLoadOp>(op) && helper(op); };
  auto scale = [](Operation &op) { auto l = dyn_cast<ScalarLoadOp>(op); return l && helper(op) && l.getKind() == "seli"; };
  struct Case { StringRef name, diagnostic; std::function<void(ModuleOp)> corrupt; };
  const std::vector<Case> cases = {
      {"31-row copy leaves the last relayout row stale", stale, [&](ModuleOp m) { set(find(m, limit), "immediate", 31); }},
      {"33-row copy writes past its relayout scratch", "buffer contract: PACK copy loop store is not proven inside its relayout scratch", [&](ModuleOp m) { set(find(m, limit), "immediate", 33); }},
      {"a copy that stops advancing never exits", "buffer contract: PACK copy loop does not exit within 256 iterations", [&](ModuleOp m) {
         for (Operation &op : m.getBody()->getOperations())
           if (auto a = dyn_cast<ALUImmOp>(op); a && helper(op) && a.getDst() == a.getSrc() && llvm::is_contained({1, 16, 32}, i32(&op, "immediate"))) set(&op, "immediate", 0); }},
      {"inverted exit copies one row", stale, [&](ModuleOp m) { find(m, backedge)->setAttr("kind", StringAttr::get(&context, "bge")); }},
      {"unknown row limit is undecidable", "buffer contract: PACK copy loop branch depends on an unknown register", [&](ModuleOp m) { set(find(m, limit), "src", f.scalarTemporary); }},
      {"shifted source word breaks the interleave", interleave, [&](ModuleOp m) { set(find(m, copyLoad), "offset", 4); }},
      {"forward helper branch is not a counted copy", "buffer contract: PACK copy loop must be one block closed by its own backedge", [&](ModuleOp m) { set(find(m, backedge), "offset_bytes", 4); }},
      {"interior backedge skips the first load on later rows", interleave, [&](ModuleOp m) { Operation *b = find(m, backedge); set(b, "offset_bytes", i32(b, "offset_bytes") + 2); }},
      {"raw store after the copy is stale", stale, [&](ModuleOp m) { moveBefore(find(m, rawStore), find(m, relayout)->getPrevNode()->getPrevNode()); }},
      {"conversion under another scale code", "buffer contract: PACK conversion scale register does not hold its source scale code", [&](ModuleOp m) { set(find(m, scale), "offset", 126); }},
      {"retained scale code is checked against the conversion", "buffer contract: PACK conversion scale register does not hold its source scale code", [&](ModuleOp m) {
         auto contract = m->getAttrOfType<DictionaryAttr>(kAtlasBufferContract);
         auto record = cast<DictionaryAttr>(contract.getAs<ArrayAttr>("packs")[0]);
         NamedAttrList fields(record); fields.set("scale_code", Builder(&context).getI32IntegerAttr(126));
         NamedAttrList dictionary(contract); dictionary.set("packs", Builder(&context).getArrayAttr({fields.getDictionary(&context)}));
         m->setAttr(kAtlasBufferContract, dictionary.getDictionary(&context)); }},
      {"retained relayout layout must match its derivation", inconsistent, [&](ModuleOp m) {
         auto contract = m->getAttrOfType<DictionaryAttr>(kAtlasBufferContract);
         SmallVector<Attribute> reads;
         for (Attribute a : contract.getAs<ArrayAttr>("reads")) {
           NamedAttrList fields(cast<DictionaryAttr>(a)); fields.set("layout", StringAttr::get(&context, "copy")); reads.push_back(fields.getDictionary(&context));
         }
         NamedAttrList dictionary(contract); dictionary.set("reads", Builder(&context).getArrayAttr(reads));
         m->setAttr(kAtlasBufferContract, dictionary.getDictionary(&context)); }},
      {"missing contract fails classification", "resource-contract-v5 artifact requires atlas.virtual_buffer_contract", [&](ModuleOp m) { m->removeAttr(kAtlasBufferContract); }},
  };
  for (const Case &c : cases) {
    auto module = lowered(context);
    if (!module) { check(false, c.name); continue; }
    c.corrupt(*module);
    check(rejects(*module, c.diagnostic), c.name, diagnostics);
  }
  // Independent LW/SW pairs may issue in any order that keeps each pair's data dependence.
  auto module = lowered(context);
  if (!module) return;
  std::vector<Operation *> loads;
  for (Operation &op : module->getBody()->getOperations()) if (copyLoad(op)) loads.push_back(&op);
  for (size_t n = 2; n < loads.size(); n += 2) moveBefore(loads[n], loads[n - 1]);
  check(accepts(*module), "reordered independent copy pairs keep the interleave", diagnostics);
}

// The tensor input reuses the mailbox window; completing it before the argument LWs overwrites their words.
void mailboxReuse(MLIRContext &context) {
  for (bool early : {false, true}) {
    auto module = lowered(context);
    if (!module) { check(false, "mailbox fixture lowers"); continue; }
    Operation *argument = find(*module, [](Operation &op) { return op.hasAttr(kAtlasTagScalarArgument); });
    auto input = contractTag(find(*module, [](Operation &op) { return isa<DMAOp>(op) && op.hasAttr(kAtlasTagCFGSource); }), kAtlasTagCFGSource);
    if (early)
      for (Operation &op : llvm::make_early_inc_range(module->getBody()->getOperations()))
        if (contractTag(&op, kAtlasTagCFGSource) == input) moveBefore(&op, argument);
    check(early ? rejects(*module, mailbox) : accepts(*module), early ? "input completed before the argument LWs overwrites the mailbox" : "input after the argument LWs may reuse the mailbox window", diagnostics);
  }
}

// Hand-issued streams isolate the must-join: a word survives a join or loop header only when every path keeps it.
struct Stream {
  MLIRContext &context;
  OpBuilder b;
  OwningOpRef<ModuleOp> module, source;
  Value state;
  std::vector<Operation *> ops;
  SmallVector<Attribute> tiles;
  explicit Stream(MLIRContext &c) : context(c), b(&c), module(ModuleOp::create(b.getUnknownLoc())),
      source(parseSourceString<ModuleOp>("module { func.func @f() { return } }", &c)) {
    b.setInsertionPointToEnd(module->getBody());
    OperationState start(b.getUnknownLoc(), "atlas.start");
    start.addTypes(StateType::get(&c));
    state = b.create(start)->getResult(0);
  }
  NamedAttribute i(StringRef name, int64_t n) { return b.getNamedAttr(name, b.getI32IntegerAttr(n)); }
  NamedAttribute text(StringRef name, StringRef s) { return b.getNamedAttr(name, b.getStringAttr(s)); }
  Operation *emit(StringRef name, std::initializer_list<NamedAttribute> attributes, int command = -1) {
    OperationState op(b.getUnknownLoc(), name);
    op.addOperands(state); op.addTypes(StateType::get(&context)); op.addAttributes(attributes);
    if (command >= 0) op.addAttribute(kAtlasTagTileCommand, b.getI32IntegerAttr(command));
    Operation *created = b.create(op); state = created->getResult(0); ops.push_back(created); return created;
  }
  int record(StringRef kind, uint32_t vmem, uint32_t bytes, ArrayRef<int32_t> after = {}) {
    tiles.push_back(b.getDictionaryAttr({i("id", tiles.size()), text("kind", kind), i("vmem_byte", vmem), i("bytes", bytes), b.getNamedAttr("after", b.getDenseI32ArrayAttr(after))}));
    return tiles.size() - 1;
  }
  void constant(int reg, uint32_t value) {
    int32_t low = int32_t(value << 20) >> 20;
    emit("atlas.upper", {text("kind", "lui"), i("dst", reg), i("immediate", (value - uint32_t(low)) >> 12)});
    emit("atlas.alu_imm", {text("kind", "addi"), i("dst", reg), i("src", reg), i("immediate", low)});
  }
  // A completed 1-KiB DMA load into `vmem`; returns the WAIT's record.
  int load(uint32_t vmem) {
    int launch = record("dma_load", vmem, 1024), wait = record("dma_wait", 0, 0, {launch});
    constant(4, vmem / 4);
    emit("atlas.dma", {text("direction", "load"), i("channel", 0), i("reg", 4), i("dram", 5), i("size", 9)}, launch);
    emit("atlas.dma_wait", {i("channel", 0)}, wait);
    return wait;
  }
  void vector(StringRef name, int command, uint32_t vmem) {
    constant(6, vmem / 4);
    emit(name, {i(name == "atlas.vload" ? "dst" : "src", 40), i("base", 6), i("offset", 0), text("format", "raw")}, command);
  }
  void nop() { emit("atlas.alu_imm", {text("kind", "addi"), i("dst", 0), i("src", 0), i("immediate", 0)}); }
  Operation *branch() { Operation *op = emit("atlas.branch", {text("kind", "bne"), i("lhs", 18), i("rhs", 0), i("offset_bytes", 0)}); nop(); return op; }
  void aim(Operation *redirect, size_t target) {
    set(redirect, "offset_bytes", 2 * (int64_t(target) - int64_t(std::find(ops.begin(), ops.end(), redirect) - ops.begin())));
  }
  // Derives the contract from the records unless one is `retained`.
  bool verify(DictionaryAttr retained = {}) {
    emit("atlas.trap", {text("kind", "ecall")});
    Builder &builder = b;
    auto cfg = builder.getDictionaryAttr({builder.getNamedAttr("operations", builder.getArrayAttr({}))});
    auto function = *source->getOps<func::FuncOp>().begin();
    FailureOr<DictionaryAttr> contract = retained ? FailureOr<DictionaryAttr>(retained) : buildAtlasBufferContract(function, cfg, builder.getArrayAttr(tiles));
    if (failed(contract)) return false;
    (*module)->setAttr(kAtlasGeneratedMarker, builder.getStringAttr(kAtlasGeneratedVersion));
    (*module)->setAttr(kAtlasTimingState, builder.getStringAttr("untimed"));
    for (StringRef name : {kAtlasDMAContract, kAtlasMXUContract}) (*module)->setAttr(name, builder.getArrayAttr({}));
    (*module)->setAttr(kAtlasSourceMemoryContract, builder.getDictionaryAttr({}));
    (*module)->setAttr(kAtlasCFGContract, cfg);
    (*module)->setAttr(kAtlasTileContract, builder.getArrayAttr(tiles));
    (*module)->setAttr(kAtlasBufferContract, *contract);
    return accepts(*module);
  }
};
constexpr uint32_t A = 0x8000; // a VMEM byte; DMA and vector bases hold word A / 4
void joins(MLIRContext &context) {
  for (bool overwrite : {false, true}) {
    Stream s(context);
    int wait = s.load(A);
    int read = s.record("vload", A, 1024, {wait});
    Operation *skip = s.branch();
    if (overwrite) s.load(A); else s.nop();
    size_t join = s.ops.size();
    s.vector("atlas.vload", read, A);
    s.aim(skip, join);
    check(overwrite ? !s.verify() && diagnosed(reader) : s.verify(), overwrite ? "a word overwritten on one path does not survive the join" : "a word kept on both paths survives the join", diagnostics);
  }
}
void loops(MLIRContext &context) {
  for (bool overwrite : {false, true}) {
    Stream s(context);
    int wait = s.load(A);
    int read = s.record("vload", A, 1024, {wait});
    size_t header = s.ops.size();
    s.vector("atlas.vload", read, A);
    if (overwrite) s.load(A);
    s.aim(s.branch(), header);
    check(overwrite ? !s.verify() && diagnosed(reader) : s.verify(), overwrite ? "a later iteration cannot read a word its body overwrote" : "a word preserved around the loop is read on every iteration", diagnostics);
  }
}
// A stray scalar store between a staging VSTORE and its capture: a known word, a sub-word, elsewhere, beyond
// VMEM (RTL address slicing aliases it into the staged half), or an unknown address.
void strayStores(MLIRContext &context) {
  struct Case { StringRef name, kind; uint32_t address; bool known, accepted; };
  for (const Case &c : {Case{"word store into the staged half", "sw", A + 8, true, false},
                        Case{"byte store into the staged half", "sb", A + 9, true, false},
                        Case{"word store outside every staged half", "sw", A + 4096, true, true},
                        Case{"word store beyond VMEM may alias the staged half", "sw", A + 8 + (1u << 21), true, false},
                        Case{"store at an unknown address", "sw", 0, false, false}}) {
    Stream s(context);
    int store = s.record("vstore", A, 1024);
    int launch = s.record("dma_store", A, 1024, {store});
    s.vector("atlas.vstore", store, A);
    if (c.known) s.constant(7, c.address);
    s.emit("atlas.scalar_store", {s.text("kind", c.kind), s.i("src", 0), s.i("base", c.known ? 7 : 27), s.i("offset", 0)});
    s.constant(4, A / 4);
    s.emit("atlas.dma", {s.text("direction", "store"), s.i("channel", 1), s.i("reg", 4), s.i("dram", 5), s.i("size", 9)}, launch);
    check(c.accepted ? s.verify() : !s.verify() && diagnosed(capture), c.name, diagnostics);
  }
}
// Lowering orders a VLOAD after a VSTORE only inside PACK; a copy readback is neither derived nor retained.
void readback(MLIRContext &context) {
  for (bool retained : {false, true}) {
    Stream s(context);
    int store = s.record("vstore", A, 1024);
    int read = s.record("vload", A, 1024, {store});
    s.vector("atlas.vstore", store, A);
    s.vector("atlas.vload", read, A);
    Builder &b = s.b;
    auto copy = b.getDictionaryAttr({s.i("command", read), s.i("writer", store), s.i("vmem_byte", A), s.i("bytes", 1024), s.i("word", 0), s.text("layout", "copy")});
    auto contract = retained ? b.getDictionaryAttr({b.getNamedAttr("packs", b.getArrayAttr({})), b.getNamedAttr("reads", b.getArrayAttr({copy}))}) : DictionaryAttr{};
    diagnostics.clear();
    check(!s.verify(contract) && diagnosed(retained ? StringRef(inconsistent) : StringRef("buffer contract cannot derive tile readers and PACK endpoints")),
          retained ? "a retained non-PACK readback is inconsistent with the tile records" : "a non-PACK VLOAD after a VSTORE is not derived", diagnostics);
  }
}
void emptyStream(MLIRContext &context) {
  AtlasVerificationContext ctx;
  auto module = parseSourceString<ModuleOp>(R"mlir(module attributes {
    atlas.generated_from_virtual = "resource-contract-v5", atlas.timing_state = "untimed",
    atlas.virtual_dma_contract = [], atlas.virtual_mxu_contract = [], atlas.virtual_tile_contract = [],
    atlas.virtual_source_memory_contract = {effects = []}, atlas.virtual_buffer_contract = {packs = [], reads = []},
    atlas.virtual_cfg_contract = {operations = []}
  } {
    %s = "atlas.start"() : () -> !atlas.state
  })mlir", &context);
  if (!module) { check(false, "empty stream parses"); return; }
  ctx.module = *module; ctx.generated = true; ctx.stream.emplace();
  diagnostics.clear();
  check(succeeded(verifyAtlasGeneratedBufferContract(ctx)), "an empty stream with no readers has no buffer obligation", diagnostics);
}
} // namespace

void atlas_test::runBufferContract(MLIRContext &context) {
  records(context); pack(context); mailboxReuse(context); joins(context); loops(context); strayStores(context); readback(context); emptyStream(context);
}
