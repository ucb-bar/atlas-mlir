#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "VerificationTestSupport.h"
#include <functional>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral reader = "buffer contract: VLOAD reads a word its writer did not write";
constexpr StringLiteral capture = "buffer contract: DMA store captures a word its VSTORE did not write";
constexpr StringLiteral interleave = "buffer contract: PACK relayout word differs from the selected row interleave";
constexpr StringLiteral stale = "buffer contract: PACK relayout word is not a copy of its current raw conversion store";
constexpr StringLiteral inconsistent = "buffer contract has malformed or inconsistent source records";
constexpr StringLiteral loopShape = "buffer contract: PACK helper branch does not close a PACK relayout loop";
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
DictionaryAttr with(DictionaryAttr record, StringRef name, Attribute value) {
  NamedAttrList fields(record);
  fields.set(name, value);
  return fields.getDictionary(record.getContext());
}

OwningOpRef<ModuleOp> lowered(MLIRContext &context) { return runPipeline(context, sourceText, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits"); }
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
  auto function = firstFunction(*source);
  auto cfg = (*module)->getAttrOfType<DictionaryAttr>(kAtlasCFGContract);
  auto rebuilt = buildAtlasBufferContract(function, cfg, tiles);
  check(succeeded(rebuilt) && *rebuilt == contract, "the builder is a function of live source and the two contracts");
  SmallVector<Attribute> operations(cfg.getAs<ArrayAttr>("operations").begin(), cfg.getAs<ArrayAttr>("operations").end());
  for (Attribute &a : operations) {
    auto op = cast<DictionaryAttr>(a);
    auto commands = op.getAs<DenseI32ArrayAttr>("tile_commands").asArrayRef();
    if (op.getAs<StringAttr>("name").getValue() == VirtualPackFP8Op::getOperationName())
      a = with(op, "tile_commands", Builder(&context).getDenseI32ArrayAttr({commands[1], commands[0]}));
  }
  diagnostics.clear();
  check(failed(buildAtlasBufferContract(function, with(cfg, "operations", Builder(&context).getArrayAttr(operations)), tiles)) && diagnosed("buffer contract cannot derive tile readers and PACK endpoints"), "PACK endpoints must be its raw VSTORE then its relayout VLOAD");
}

void pack(MLIRContext &context) {
  const FixedResourcePlacement &f = fixed();
  auto limit = [&](Operation &op) { auto a = dyn_cast<ALUImmOp>(op); return a && helper(op) && a.getDst() == f.packRowsReg; };
  auto backedge = [](Operation &op) { return isa<BranchOp>(op) && helper(op); };
  auto copyLoad = [&](Operation &op) { auto l = dyn_cast<ScalarLoadOp>(op); return l && helper(op) && l.getKind() == "lw"; };
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
      {"forward helper branch is not a counted copy", loopShape, [&](ModuleOp m) { set(find(m, backedge), "offset_bytes", 4); }},
      {"a copy loop body instruction of another source operation", loopShape, [&](ModuleOp m) { Operation *l = find(m, copyLoad); set(l, kAtlasTagCFGSource, i32(l, kAtlasTagCFGSource) + 1); }},
      {"interior backedge skips the first load on later rows", interleave, [&](ModuleOp m) { Operation *b = find(m, backedge); set(b, "offset_bytes", i32(b, "offset_bytes") + 2); }},
      {"retained scale code is checked against the conversion", "buffer contract: PACK conversion scale register does not hold its source scale code", [&](ModuleOp m) {
         auto contract = m->getAttrOfType<DictionaryAttr>(kAtlasBufferContract);
         auto record = with(cast<DictionaryAttr>(contract.getAs<ArrayAttr>("packs")[0]), "scale_code", Builder(&context).getI32IntegerAttr(126));
         m->setAttr(kAtlasBufferContract, with(contract, "packs", Builder(&context).getArrayAttr({record}))); }},
      {"missing contract fails classification", "resource-contract-v5 artifact requires atlas.virtual_buffer_contract", [&](ModuleOp m) { m->removeAttr(kAtlasBufferContract); }},
  };
  for (const Case &c : cases) {
    auto module = lowered(context);
    if (!module) { check(false, c.name); continue; }
    c.corrupt(*module);
    check(rejects(*module, c.diagnostic), c.name, diagnostics);
  }
  // A copy's store may issue anywhere between its load and the next load of its temporary: moving each
  // store up to its own load, the serial order lowering used to emit, keeps the interleave.
  auto module = lowered(context);
  if (!module) return;
  auto copyStore = [](Operation &op) { return isa<ScalarStoreOp>(op) && helper(op); };
  std::vector<Operation *> loads, stores;
  for (Operation &op : module->getBody()->getOperations()) {
    if (copyLoad(op)) loads.push_back(&op);
    if (copyStore(op)) stores.push_back(&op);
  }
  check(loads.size() == 8 && stores.size() == 8, "one row of eight copies");
  for (size_t n = 0; n + 1 < loads.size() && n < stores.size(); ++n) moveBefore(stores[n], loads[n + 1]);
  check(accepts(*module), "each store moved up to its own load keeps the interleave", diagnostics);
}

struct Stream : IssuedStream {
  OwningOpRef<ModuleOp> source;
  SmallVector<Attribute> tiles;
  explicit Stream(MLIRContext &c) : IssuedStream(c), source(parseSourceString<ModuleOp>("module { func.func @f() { return } }", &c)) {}
  int record(StringRef kind, uint32_t vmem, uint32_t bytes, ArrayRef<int32_t> after = {}) {
    tiles.push_back(b.getDictionaryAttr({i("id", tiles.size()), text("kind", kind), i("vmem_byte", vmem), i("bytes", bytes), b.getNamedAttr("after", b.getDenseI32ArrayAttr(after))}));
    return tiles.size() - 1;
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
  // Derives the contract from the records unless one is `retained`.
  bool verify(DictionaryAttr retained = {}) {
    halt();
    Builder &builder = b;
    auto cfg = builder.getDictionaryAttr({builder.getNamedAttr("operations", builder.getArrayAttr({}))});
    auto function = *source->getOps<func::FuncOp>().begin();
    FailureOr<DictionaryAttr> contract = retained ? FailureOr<DictionaryAttr>(retained) : buildAtlasBufferContract(function, cfg, builder.getArrayAttr(tiles));
    if (failed(contract)) return false;
    markGenerated(cfg, builder.getArrayAttr(tiles), builder.getDictionaryAttr({}), *contract);
    return accepts(*module);
  }
};

// Hand-built source records let the buffer proof's recurrence checks run
// independently of CFG authentication and the indexed-access range proof.
// The issued loop writes two words per iteration and reads its last word.
struct CountedStream : Stream {
  SmallVector<Attribute> values, blocks, edges, operations;
  explicit CountedStream(MLIRContext &context, bool splitExitDMA = false) : Stream(context) {
    auto array = [&](StringRef name, ArrayRef<int32_t> ids) { return b.getNamedAttr(name, b.getDenseI32ArrayAttr(ids)); };
    auto value = [&](StringRef def, int block, ArrayRef<int32_t> operands = {}) {
      return b.getDictionaryAttr({text("def", def), i("block", block), array("operands", operands)});
    };
    for (int n : {0, 1, 2, 4}) values.push_back(with(value("arith.constant", 0), "constant", b.getI32IntegerAttr(n)));
    values.push_back(value("argument", 1)); // counter
    values.push_back(value("argument", 1)); // word pointer
    values.push_back(with(value("arith.cmpi", 1, {4, 3}), "predicate", b.getI32IntegerAttr(6))); // ult
    values.push_back(value("arith.addi", 2, {4, 1}));
    values.push_back(value("arith.addi", 2, {5, 2}));
    values.push_back(value("atlas.virtual_slot_load", 3));
    auto edge = [&](int from, int to, ArrayRef<int32_t> incoming) {
      edges.push_back(b.getDictionaryAttr({i("from", from), i("to", to), array("incoming", incoming)}));
    };
    edge(0, 1, {0, 0}); edge(1, 2, {}); edge(1, 3, {}); edge(2, 1, {7, 8});
    auto sourceBlock = [&](int id, ArrayRef<int32_t> args, ArrayRef<int32_t> ops, ArrayRef<int32_t> exits, int condition = -1) {
      blocks.push_back(b.getDictionaryAttr({i("id", id), i("condition", condition), array("args", args), array("operations", ops), array("edges", exits)}));
    };
    sourceBlock(0, {}, {}, {0}); sourceBlock(1, {4, 5}, {0}, {1, 2}, 6);
    sourceBlock(2, {}, {1, 2, 3, 4}, {3}); sourceBlock(3, {}, {5}, {});
    operations.push_back(b.getDictionaryAttr({i("id", 0), text("name", "arith.cmpi")}));
    uint32_t slot = fixed().spillWord;
    auto access = [&](int id, bool store, int offset) {
      int command = record(store ? "slot_index_store" : "slot_load", slot * 4 + offset, store ? 28 : 4);
      operations.push_back(b.getDictionaryAttr({i("id", id), text("name", store ? "atlas.virtual_slot_store" : "atlas.virtual_slot_load"),
          text("kind", store ? "sw" : "lw"), i("slot_word", slot), i("slot_words", kAtlasRegisterWords), text("slot_kind", "raw"),
          i("byte_offset", offset), array("operands", store ? ArrayRef<int32_t>({0, 5}) : ArrayRef<int32_t>()),
          array("results", store ? ArrayRef<int32_t>() : ArrayRef<int32_t>({9})), array("tile_commands", {command})}));
    };
    access(1, true, 0); access(2, true, 4);
    operations.push_back(b.getDictionaryAttr({i("id", 3), text("name", "arith.addi")}));
    operations.push_back(b.getDictionaryAttr({i("id", 4), text("name", "arith.addi")}));
    access(5, false, 28);

    block = 0;
    constant(10, 0); constant(11, 0); constant(12, 4); constant(13, slot * 4);
    block = 1;
    size_t header = ops.size();
    Operation *exit = emit("atlas.branch", {text("kind", "bgeu"), i("lhs", 10), i("rhs", 12), i("offset_bytes", 0)});
    nop();
    block = 2;
    emit("atlas.alu_imm", {text("kind", "slli"), i("dst", 14), i("src", 11), i("immediate", 2)});
    emit("atlas.alu_reg", {text("kind", "add"), i("dst", 14), i("lhs", 13), i("rhs", 14)});
    for (int n : {0, 1}) emit("atlas.scalar_store", {text("kind", "sw"), i("src", 0), i("base", 14), i("offset", n * 4)}, n);
    emit("atlas.alu_imm", {text("kind", "addi"), i("dst", 10), i("src", 10), i("immediate", 1)});
    emit("atlas.alu_imm", {text("kind", "addi"), i("dst", 11), i("src", 11), i("immediate", 2)});
    Operation *back = jump(); nop(); aim(back, header);
    block = 3;
    size_t exitPC = ops.size();
    if (splitExitDMA) {
      int launch = record("dma_load", slot * 4, 1024);
      constant(4, slot);
      emit("atlas.dma", {text("direction", "load"), i("channel", 0), i("reg", 4), i("dram", 5), i("size", 9)}, launch);
      // A machine split inside this source block must not reassert the loop's
      // initialization after the DMA has invalidated it and before its WAIT.
      Operation *split = jump(); nop(); aim(split, ops.size());
    }
    emit("atlas.scalar_load", {text("kind", "lw"), i("dst", 15), i("base", 13), i("offset", 28)}, 2);
    aim(exit, exitPC);
  }
  void change(SmallVectorImpl<Attribute> &records, unsigned id, StringRef field, int value) {
    records[id] = with(cast<DictionaryAttr>(records[id]), field, b.getI32IntegerAttr(value));
  }
  void change(SmallVectorImpl<Attribute> &records, unsigned id, StringRef field, ArrayRef<int32_t> ids) {
    records[id] = with(cast<DictionaryAttr>(records[id]), field, b.getDenseI32ArrayAttr(ids));
  }
  bool verify() {
    halt();
    auto cfg = b.getDictionaryAttr({b.getNamedAttr("values", b.getArrayAttr(values)), b.getNamedAttr("blocks", b.getArrayAttr(blocks)),
        b.getNamedAttr("edges", b.getArrayAttr(edges)), b.getNamedAttr("operations", b.getArrayAttr(operations))});
    auto function = *source->getOps<func::FuncOp>().begin();
    auto contract = buildAtlasBufferContract(function, cfg, b.getArrayAttr(tiles));
    if (failed(contract)) return false;
    markGenerated(cfg, b.getArrayAttr(tiles), b.getDictionaryAttr({}), *contract);
    return accepts(*module);
  }
};

void countedInitializations(MLIRContext &context) {
  struct Case { StringRef name; bool accepted; std::function<void(CountedStream &)> mutate; };
  const std::vector<Case> cases = {
      {"a counted contiguous word loop initializes its exact exit footprint", true, [](CountedStream &) {}},
      {"commuted addition operands retain the counted proof", true, [](CountedStream &s) { s.change(s.values, 7, "operands", {1, 4}); s.change(s.values, 8, "operands", {2, 5}); }},
      {"a larger count may initialize more bytes within the slot", true, [](CountedStream &s) { s.change(s.values, 3, "constant", 5); }},
      {"a smaller count leaves the final word uninitialized", false, [](CountedStream &s) { s.change(s.values, 3, "constant", 3); }},
      {"a zero count establishes no initialization", false, [](CountedStream &s) { s.change(s.values, 3, "constant", 0); }},
      {"an oversized count cannot initialize beyond the slot", false, [](CountedStream &s) { s.change(s.values, 3, "constant", kAtlasRegisterWords); }},
      {"the counted proof requires an unsigned strict predicate", false, [](CountedStream &s) { s.change(s.values, 6, "predicate", 7); }},
      {"the counter must be seeded with zero", false, [](CountedStream &s) { s.change(s.edges, 0, "incoming", {1, 0}); }},
      {"the pointer must be seeded with zero", false, [](CountedStream &s) { s.change(s.edges, 0, "incoming", {0, 1}); }},
      {"the counter must advance by exactly one", false, [](CountedStream &s) { s.change(s.values, 7, "operands", {4, 2}); }},
      {"the pointer stride must match the store count", false, [](CountedStream &s) { s.change(s.values, 8, "operands", {5, 1}); }},
      {"the body must be the true successor and the exit the false successor", false, [](CountedStream &s) { s.change(s.blocks, 1, "edges", {2, 1}); }},
      {"duplicate store offsets establish no counted footprint", false, [](CountedStream &s) { s.change(s.operations, 2, "byte_offset", 0); s.change(s.tiles, 1, "vmem_byte", fixed().spillWord * 4); }},
      {"a gap between word stores establishes no counted footprint", false, [](CountedStream &s) { s.change(s.operations, 2, "byte_offset", 8); s.change(s.tiles, 1, "vmem_byte", fixed().spillWord * 4 + 8); }},
      {"partial stores establish no counted word footprint", false, [](CountedStream &s) {
         s.operations[2] = with(cast<DictionaryAttr>(s.operations[2]), "kind", s.b.getStringAttr("sh"));
         s.change(s.tiles, 1, "bytes", 26);
      }},
  };
  for (const Case &c : cases) {
    CountedStream s(context);
    c.mutate(s); diagnostics.clear();
    bool accepted = s.verify();
    check(c.accepted ? accepted : !accepted && diagnosed("buffer contract: scalar slot load reads an uninitialized byte"), c.name, diagnostics);
  }
  CountedStream split(context, true);
  diagnostics.clear();
  check(!split.verify() && diagnosed("buffer contract: scalar slot load reads an uninitialized byte"),
        "a machine split in the exit source block does not reinitialize a pending DMA's bytes", diagnostics);
}
constexpr uint32_t A = 0x8000; // a VMEM byte; DMA and vector bases hold word A / 4
// Must-join: a word survives a join (a skipped overwrite) or loop header (an overwriting body) only when every path keeps it.
void joins(MLIRContext &context) {
  for (bool loop : {false, true})
    for (bool overwrite : {false, true}) {
      Stream s(context);
      int wait = s.load(A);
      int read = s.record("vload", A, 1024, {wait});
      size_t target = s.ops.size();
      if (loop) {
        s.vector("atlas.vload", read, A);
        if (overwrite) s.load(A);
      }
      Operation *redirect = s.branch();
      s.nop();
      if (!loop) {
        if (overwrite) s.load(A); else s.nop();
        target = s.ops.size();
        s.vector("atlas.vload", read, A);
      }
      s.aim(redirect, target);
      StringRef name = loop ? (overwrite ? "a later iteration cannot read a word its body overwrote" : "a word preserved around the loop is read on every iteration")
                            : (overwrite ? "a word overwritten on one path does not survive the join" : "a word kept on both paths survives the join");
      check(overwrite ? !s.verify() && diagnosed(reader) : s.verify(), name, diagnostics);
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
  auto module = parseSourceString<ModuleOp>(emptyIssuedStream, &context);
  if (!module) { check(false, "empty stream parses"); return; }
  AtlasVerificationContext ctx;
  ctx.module = *module; ctx.generated = true; ctx.stream.emplace();
  diagnostics.clear();
  check(succeeded(verifyAtlasGeneratedBufferContract(ctx)), "an empty stream with no readers has no buffer obligation", diagnostics);
}
} // namespace

void atlas_test::runBufferContract(MLIRContext &context) {
  records(context); pack(context); countedInitializations(context); joins(context); strayStores(context); readback(context); emptyStream(context);
}
