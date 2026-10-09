#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
namespace {
unsigned checks = 0, failures = 0;
std::string diagnostics;
void check(bool ok, StringRef name) {
  ++checks;
  if (!ok) { ++failures; llvm::errs() << "FAIL: " << name << '\n'; }
}
bool diagnosed(StringRef message) { return StringRef(diagnostics).contains(message); }
constexpr StringLiteral notCompleted = "overlapping source predecessor has not completed in this visit";
constexpr StringLiteral incomplete = "source block exits before its required effects complete";
constexpr StringLiteral pending = "source block exits with a pending DRAM effect";
constexpr StringLiteral sourceText = R"mlir(module {
  func.func @effects() -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %input = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %a = arith.constant -1879048192 : i32
    %b = arith.constant -1879048192 : i32
    %size = arith.constant 1024 : i32
    %s2, %store = "atlas.virtual_dma_store_fp8"(%s1, %input, %a, %size) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s3 = "atlas.virtual_dma_wait"(%s2, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s4, %load = "atlas.virtual_dma_load_fp8"(%s3, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s5, %result = "atlas.virtual_dma_await_fp8"(%s4, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    return %s5 : !atlas.virtual_state
  }
})mlir";
constexpr StringLiteral loopText = R"mlir(module {
  func.func @effects() -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %input = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.br ^loop(%s1, %input : !atlas.virtual_state, !atlas.virtual_fp8)
  ^loop(%state : !atlas.virtual_state, %tile : !atlas.virtual_fp8):
    %a = arith.constant -1879048192 : i32
    %b = arith.constant -1879048192 : i32
    %size = arith.constant 1024 : i32
    %choose = arith.constant true
    %s2, %store = "atlas.virtual_dma_store_fp8"(%state, %tile, %a, %size) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s3 = "atlas.virtual_dma_wait"(%s2, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s4, %load = "atlas.virtual_dma_load_fp8"(%s3, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s5, %result = "atlas.virtual_dma_await_fp8"(%s4, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.cond_br %choose, ^loop(%s5, %tile : !atlas.virtual_state, !atlas.virtual_fp8), ^exit(%s5 : !atlas.virtual_state)
  ^exit(%final : !atlas.virtual_state):
    return %final : !atlas.virtual_state
  }
})mlir";
constexpr StringLiteral branchText = R"mlir(module {
  func.func @effects(%choose : i1) -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64,
    atlas.control_dram_base = 2466250752 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %input = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.cond_br %choose, ^left(%s1, %input : !atlas.virtual_state, !atlas.virtual_fp8), ^right(%s1 : !atlas.virtual_state)
  ^left(%state : !atlas.virtual_state, %tile : !atlas.virtual_fp8):
    %a = arith.constant -1879048192 : i32
    %b = arith.constant -1879048192 : i32
    %size = arith.constant 1024 : i32
    %s2, %store = "atlas.virtual_dma_store_fp8"(%state, %tile, %a, %size) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s3 = "atlas.virtual_dma_wait"(%s2, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s4, %load = "atlas.virtual_dma_load_fp8"(%s3, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s5, %result = "atlas.virtual_dma_await_fp8"(%s4, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.br ^join(%s5 : !atlas.virtual_state)
  ^right(%rstate : !atlas.virtual_state):
    %raddr = arith.constant -1879048192 : i32
    %rsize = arith.constant 1024 : i32
    %r1, %rload = "atlas.virtual_dma_load_fp8"(%rstate, %raddr, %rsize) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %r2, %rresult = "atlas.virtual_dma_await_fp8"(%r1, %rload) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.br ^join(%r2 : !atlas.virtual_state)
  ^join(%final : !atlas.virtual_state):
    return %final : !atlas.virtual_state
  }
})mlir";
// Literal source expectations and manually issued groups isolate this
// obligation. Final integration separately checks complete CFG/tile/lifecycle
// metadata and source value semantics before invoking this checker.
struct Fixture {
  MLIRContext &context;
  OpBuilder b;
  OwningOpRef<ModuleOp> source, issued;
  DictionaryAttr cfg, contract;
  ArrayAttr tiles;
  Value state;
  std::vector<Operation *> ops;
  int block = 0;
  explicit Fixture(MLIRContext &c, StringRef text) : context(c), b(&c), source(parseSourceString<ModuleOp>(text, &c)), issued(ModuleOp::create(b.getUnknownLoc())) {
    if (!source) return;
    auto function = *source->getOps<func::FuncOp>().begin();
    SmallVector<VirtualRegisterAssignment> registers;
    SmallVector<VirtualDMAAssignment> dma;
    unsigned scalar = 19, tensor = 12, transfer = 0;
    for (Block &body : function.getBody()) {
      auto add = [&](Value v) {
        if (v.getType().isInteger(1) || v.getType().isInteger(32)) registers.push_back({v, scalar++});
        else if (isa<VirtualFP8Type>(v.getType())) registers.push_back({v, tensor++});
        else if (isa<VirtualBF16Type>(v.getType())) { registers.push_back({v, tensor}); tensor += 2; }
      };
      for (Value v : body.getArguments()) add(v);
      for (Operation &op : body) {
        for (Value v : op.getResults()) add(v);
        if (isa<VirtualDMAStoreFP8Op, VirtualDMALoadFP8Op>(op)) {
          dma.push_back({op.getResult(1), {2 + transfer, 1, transfer, 131072 + 256 * transfer, 4, 5, 9}});
          ++transfer;
        }
      }
    }
    FixedResourcePlacement fixed{};
    fixed.inputWindowWords = fixed.outputWindowWords = 65536;
    fixed.outputWord = 65536;
    fixed.loadChannel = 1;
    fixed.storeChannel = 2;
    SmallVector<int32_t> arguments;
    for (unsigned n = 0; n < function.getNumArguments(); ++n) arguments.push_back(registers[n].reg);
    auto cfgResult = buildAtlasCFGContract(function, registers);
    SmallVector<VirtualRegisterAssignment> tensors;
    for (auto placement : registers)
      if (isa<VirtualBF16Type, VirtualFP8Type>(placement.value.getType())) tensors.push_back(placement);
    auto tileResult = buildAtlasTileContract(function, tensors, dma, fixed, arguments);
    if (failed(cfgResult) || failed(tileResult)) return;
    cfg = *cfgResult;
    tiles = *tileResult;
    auto effects = buildAtlasSourceMemoryEffectContract(function, cfg, tiles);
    if (failed(effects)) return;
    contract = *effects;
    (*issued)->setAttr("atlas.generated_from_virtual", b.getStringAttr("resource-contract-v4"));
    (*issued)->setAttr("atlas.timing_state", b.getStringAttr("untimed"));
    (*issued)->setAttr("atlas.virtual_dma_contract", b.getArrayAttr({}));
    (*issued)->setAttr("atlas.virtual_mxu_contract", b.getArrayAttr({}));
    (*issued)->setAttr("atlas.virtual_cfg_contract", cfg);
    (*issued)->setAttr("atlas.virtual_tile_contract", tiles);
    (*issued)->setAttr("atlas.virtual_source_memory_contract", contract);
    b.setInsertionPointToEnd(issued->getBody());
    OperationState start(b.getUnknownLoc(), "atlas.start"); start.addTypes(StateType::get(&c));
    state = b.create(start)->getResult(0);
  }
  NamedAttribute i(StringRef name, int64_t n) { return b.getNamedAttr(name, b.getI32IntegerAttr(n)); }
  NamedAttribute text(StringRef name, StringRef s) { return b.getNamedAttr(name, b.getStringAttr(s)); }
  Operation *emit(StringRef name, std::initializer_list<NamedAttribute> attributes, int command = -1, int edge = -1) {
    OperationState op(b.getUnknownLoc(), name); op.addOperands(state); op.addTypes(StateType::get(&context)); op.addAttributes(attributes);
    op.addAttribute("atlas.virtual_cfg_block", b.getI32IntegerAttr(block));
    if (command >= 0) op.addAttribute("atlas.virtual_tile_command", b.getI32IntegerAttr(command));
    if (edge >= 0) op.addAttribute("atlas.virtual_cfg_edge", b.getI32IntegerAttr(edge));
    Operation *created = b.create(op); state = created->getResult(0); ops.push_back(created); return created;
  }
  void constant(int reg, uint32_t value) {
    int32_t low = int32_t(value << 20) >> 20;
    uint32_t high = (value - uint32_t(low)) >> 12;
    if (high) emit("atlas.upper", {text("kind", "lui"), i("dst", reg), i("immediate", high)});
    if (low || !high) emit("atlas.alu_imm", {text("kind", "addi"), i("dst", reg), i("src", high ? reg : 0), i("immediate", low)});
  }
  void nop() { emit("atlas.alu_imm", {text("kind", "addi"), i("dst", 0), i("src", 0), i("immediate", 0)}); }
  void halt() { emit("atlas.trap", {text("kind", "ecall")}); }
  DictionaryAttr tile(unsigned effect, StringRef role) {
    auto e = cast<DictionaryAttr>(contract.getAs<ArrayAttr>("effects")[effect]);
    return cast<DictionaryAttr>(tiles[e.getAs<IntegerAttr>(role).getInt()]);
  }
  // A group is its launch (address setup, staging VSTOREs, DMA) then its completion (WAIT, staging VLOADs).
  void launch(unsigned effect) {
    auto t = tile(effect, "launch");
    bool store = t.getAs<StringAttr>("kind").getValue() == "dma_store";
    constant(4, uint32_t(t.getAs<IntegerAttr>("vmem_byte").getInt()) / 4);
    constant(5, uint32_t(t.getAs<IntegerAttr>("dram_byte").getInt()));
    constant(9, t.getAs<IntegerAttr>("bytes").getInt());
    if (store) for (int id : t.getAs<DenseI32ArrayAttr>("after").asArrayRef()) {
      auto v = cast<DictionaryAttr>(tiles[id]);
      emit("atlas.vstore", {i("src", v.getAs<IntegerAttr>("reg").getInt()), i("base", 4), i("offset", 0), text("format", "raw")}, id);
    }
    emit("atlas.dma", {text("direction", store ? "store" : "load"), i("reg", 4), i("dram", 5), i("size", 9), i("channel", t.getAs<IntegerAttr>("channel").getInt())}, t.getAs<IntegerAttr>("id").getInt());
  }
  void complete(unsigned effect) {
    auto t = tile(effect, "completion");
    int completion = t.getAs<IntegerAttr>("id").getInt();
    emit("atlas.dma_wait", {i("channel", t.getAs<IntegerAttr>("channel").getInt())}, completion);
    if (tile(effect, "launch").getAs<StringAttr>("kind").getValue() == "dma_load") for (Attribute a : tiles) {
      auto v = cast<DictionaryAttr>(a);
      if (v.getAs<StringAttr>("kind").getValue() != "vload" || v.getAs<DenseI32ArrayAttr>("after").asArrayRef() != ArrayRef<int32_t>({completion})) continue;
      emit("atlas.vload", {i("dst", v.getAs<IntegerAttr>("reg").getInt()), i("base", 4), i("offset", 0), text("format", "raw")}, v.getAs<IntegerAttr>("id").getInt());
    }
  }
  void group(unsigned effect) { launch(effect); complete(effect); }
  void release() { emit("atlas.csr", {text("kind", "rrw"), i("dst", 0), i("source", 10), i("address", 0xc10)}); }
  Operation *jump(int edge) { return emit("atlas.jump", {text("kind", "jal"), i("dst", 0), i("base", 0), i("offset", 0)}, -1, edge); }
  Operation *branch() { return emit("atlas.branch", {text("kind", "bne"), i("lhs", 18), i("rhs", 0), i("offset_bytes", 0)}); }
  void aim(Operation *redirect, Operation *target) {
    auto find = [&](Operation *op) { return std::find(ops.begin(), ops.end(), op) - ops.begin(); };
    redirect->setAttr(isa<BranchOp>(redirect) ? "offset_bytes" : "offset", b.getI32IntegerAttr(2 * (find(target) - find(redirect))));
  }
  bool verify() { diagnostics.clear(); return succeeded(verifyAtlasGeneratedSourceMemoryEffectContract(*issued)); }
  bool rejects(StringRef message) { return !verify() && diagnosed(message); }
};
int field(DictionaryAttr d, StringRef key) { return d.getAs<IntegerAttr>(key).getInt(); }
void straight(MLIRContext &context) {
  for (unsigned offset : {0u, 512u, 2048u}) {
    std::string source = sourceText.str();
    auto pos = source.find("%b = arith.constant -1879048192");
    source.replace(pos, std::string("%b = arith.constant -1879048192").size(), "%b = arith.constant " + std::to_string(int32_t(0x90000000u + offset)));
    for (bool reorder : {false, true}) {
      Fixture f(context, source); check(bool(f.contract), "literal source memory contract builds"); if (!f.contract) continue;
      auto effects = f.contract.getAs<ArrayAttr>("effects");
      check(effects.size() == 3, "literal implicit-input/explicit-store/explicit-load cardinality");
      auto store = cast<DictionaryAttr>(effects[1]), load = cast<DictionaryAttr>(effects[2]);
      check(field(store, "source") == 5 && field(store, "launch") == 4 && field(store, "completion") == 5, "literal store source identity and completion");
      check(field(load, "source") == 7 && uint32_t(field(load, "dram_byte")) == 0x90000000u + offset && field(load, "bytes") == 1024, "literal load span and source identity");
      auto pred = load.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef();
      check(offset < 1024 ? pred == ArrayRef<int32_t>({1}) : pred.empty(), "literal source-overlap predecessor set");
      f.group(0); if (reorder) { f.group(2); f.group(1); } else { f.group(1); f.group(2); } f.halt();
      check(reorder && offset < 1024 ? f.rejects(notCompleted) : f.verify(), offset >= 1024 ? "disjoint write/read groups may reorder" : reorder ? "completed overlapping groups on disjoint staging cannot reorder" : "source store completes before overlapping read issue");
    }
  }
  // A literal second fixture replaces the first write by a read. Equal host
  // spans then carry no source effect edge, even with completed groups swapped.
  std::string reads = sourceText.str();
  size_t begin = reads.find("    %s2, %store"); size_t end = reads.find("    %s4, %load");
  reads.replace(begin, end - begin, R"mlir(    %s2, %first = "atlas.virtual_dma_load_fp8"(%s1, %a, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s3, %first_result = "atlas.virtual_dma_await_fp8"(%s2, %first) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
)mlir");
  Fixture f(context, reads); check(bool(f.contract), "literal read/read source builds");
  if (f.contract) {
    auto last = cast<DictionaryAttr>(f.contract.getAs<ArrayAttr>("effects")[2]);
    check(last.getAs<DenseI32ArrayAttr>("predecessors").empty(), "same-span read/read has no ordering edge");
    f.group(0); f.group(2); f.group(1); f.halt(); check(f.verify(), "same-span completed read/read may reorder");
  }
}
void boundary(MLIRContext &context) {
  constexpr StringLiteral text = R"mlir(module {
    func.func @effects() -> !atlas.virtual_state attributes {
      atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2432696320 : i64
    } {
      %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
      %s1, %input = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
      %s2 = "atlas.virtual_output_bf16"(%s1, %input) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
      return %s2 : !atlas.virtual_state
    }
  })mlir";
  for (bool reorder : {false, true}) {
    Fixture f(context, text); check(bool(f.contract), "literal overlapping boundary I/O source builds"); if (!f.contract) continue;
    auto effects = f.contract.getAs<ArrayAttr>("effects");
    check(effects.size() == 4, "literal boundary BF16 half-effect count");
    auto first = cast<DictionaryAttr>(effects[2]), second = cast<DictionaryAttr>(effects[3]);
    check(field(first, "source") == 2 && field(first, "launch") == 7 && field(first, "completion") == 8 &&
          first.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef() == ArrayRef<int32_t>({0}) &&
          second.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef() == ArrayRef<int32_t>({1}),
          "boundary writes retain the literal corresponding prior source reads");
    if (reorder) { f.group(2); f.group(0); f.group(1); f.group(3); }
    else { f.group(0); f.group(1); f.group(2); f.group(3); }
    f.halt(); check(reorder ? f.rejects(notCompleted) : f.verify(), reorder ? "completed boundary write cannot precede its overlapping source read" : "serialized overlapping boundary I/O preserves source order");
  }
  Fixture f(context, sourceText); check(bool(f.contract), "edge-stripping source fixture builds");
  if (f.contract) {
    SmallVector<Attribute> effects(f.contract.getAs<ArrayAttr>("effects").begin(), f.contract.getAs<ArrayAttr>("effects").end());
    NamedAttrList fields(cast<DictionaryAttr>(effects[2])); fields.set("predecessors", f.b.getDenseI32ArrayAttr({}));
    effects[2] = fields.getDictionary(&context);
    (*f.issued)->setAttr("atlas.virtual_source_memory_contract", f.b.getDictionaryAttr({f.b.getNamedAttr("effects", f.b.getArrayAttr(effects))}));
    f.group(0); f.group(2); f.group(1); f.halt();
    check(f.rejects("source memory contract has malformed or inconsistent source records"), "stripping source predecessor metadata cannot bypass recomputed conflicts");
  }
}
void loops(MLIRContext &context) {
  for (bool stale : {false, true}) {
    Fixture f(context, loopText); check(bool(f.contract), "literal loop source builds"); if (!f.contract) continue;
    f.group(0); Operation *enter = f.jump(0); f.nop();
    f.block = 1; size_t header = f.ops.size(); f.group(1); size_t read = f.ops.size(); f.group(2);
    Operation *branch = f.branch(); f.nop(); Operation *exit = f.jump(2); f.nop();
    Operation *back = f.jump(1); f.nop(); f.block = 2; f.halt();
    f.aim(enter, f.ops[header]); f.aim(branch, back); f.aim(back, f.ops[stale ? read : header]); f.aim(exit, f.ops.back());
    check(stale ? f.rejects(notCompleted) : f.verify(), stale ? "previous loop visit completion cannot satisfy skipped current predecessor" : "each source loop visit independently completes conflicting predecessor");
  }
}
void branches(MLIRContext &context) {
  for (bool reorder : {false, true}) {
    Fixture f(context, branchText); check(bool(f.contract), "literal branch source builds"); if (!f.contract) continue;
    auto effects = f.contract.getAs<ArrayAttr>("effects");
    check(effects.size() == 5, "mailbox/input/left-store/left-read/right-read literal effect count");
    auto leftRead = cast<DictionaryAttr>(effects[3]), rightRead = cast<DictionaryAttr>(effects[4]);
    check(leftRead.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef() == ArrayRef<int32_t>({2}) && rightRead.getAs<DenseI32ArrayAttr>("predecessors").empty(), "predecessor belongs only to its source branch");
    f.group(0); f.group(1); Operation *choose = f.branch(); f.nop(); Operation *left = f.jump(0); f.nop();
    f.block = 1; size_t leftBody = f.ops.size();
    if (reorder) { f.group(3); f.group(2); } else { f.group(2); f.group(3); }
    Operation *leftExit = f.jump(2); f.nop();
    f.block = 0; Operation *right = f.jump(1); f.nop(); f.block = 2; size_t rightBody = f.ops.size(); f.group(4); Operation *rightExit = f.jump(3); f.nop();
    f.block = 3; f.halt();
    f.aim(choose, right); f.aim(left, f.ops[leftBody]); f.aim(right, f.ops[rightBody]); f.aim(leftExit, f.ops.back()); f.aim(rightExit, f.ops.back());
    check(reorder ? f.rejects(notCompleted) : f.verify(), reorder ? "branch-local completed transfers cannot reverse their source overlap" : "untaken branch predecessor is not imposed on the other branch");
  }
}
// Each issued stream isolates one lifecycle rejection; later failures in the same stream are not reached.
void lifecycle(MLIRContext &context) {
  {
    Fixture f(context, sourceText); check(bool(f.contract), "missing-completion fixture builds");
    if (f.contract) { f.group(0); f.group(1); f.halt(); check(f.rejects("source memory contract requires one issued launch and completion per effect"), "every effect issues exactly one launch and completion"); }
  }
  {
    Fixture f(context, sourceText); check(bool(f.contract), "early-completion fixture builds");
    if (f.contract) { f.group(0); f.complete(1); f.launch(1); f.group(2); f.halt(); check(f.rejects("completion has no matching current launch"), "completion before its launch is rejected"); }
  }
  {
    Fixture f(context, sourceText); check(bool(f.contract), "issued-loop fixture builds");
    if (f.contract) {
      f.group(0); size_t body = f.ops.size(); f.group(1); Operation *again = f.branch(); f.nop(); f.group(2); f.halt();
      f.aim(again, f.ops[body]);
      check(f.rejects("effect launched twice within one source visit"), "an untagged issued loop cannot relaunch within one source visit");
    }
  }
  {
    Fixture f(context, branchText); check(bool(f.contract), "shared-channel fixture builds");
    if (f.contract) {
      check(f.tile(0, "launch").getAs<IntegerAttr>("channel") == f.tile(1, "launch").getAs<IntegerAttr>("channel"), "mailbox and input share their load channel");
      f.launch(0); f.launch(1); f.complete(0); f.complete(1); f.group(2); f.group(3); f.group(4); f.halt();
      check(f.rejects("effect channel is not idle"), "a launch cannot reuse a channel with a pending effect");
    }
  }
  {
    Fixture f(context, loopText); check(bool(f.contract), "wrong-block fixture builds");
    if (f.contract) {
      f.group(0); f.group(1); Operation *enter = f.jump(0); f.nop();
      f.block = 1; size_t header = f.ops.size(); f.group(2); Operation *branch = f.branch(); f.nop(); Operation *exit = f.jump(2); f.nop();
      Operation *back = f.jump(1); f.nop(); f.block = 2; f.halt();
      f.aim(enter, f.ops[header]); f.aim(branch, back); f.aim(back, f.ops[header]); f.aim(exit, f.ops.back());
      check(f.rejects("launch belongs to a different source-block visit"), "a launch issued in another source block is rejected");
    }
  }
}
// The store is predecessor 1 of the overlapping load (effect 2); both paths of an untagged branch join before the load.
void joins(MLIRContext &context) {
  for (bool waitBothPaths : {true, false}) {
    Fixture f(context, sourceText); check(bool(f.contract), "must-join fixture builds"); if (!f.contract) continue;
    f.group(0); f.launch(1); Operation *skip = f.branch(); f.nop();
    if (waitBothPaths) f.nop(); else f.complete(1);
    size_t join = f.ops.size();
    if (waitBothPaths) f.complete(1);
    f.group(2); f.halt(); f.aim(skip, f.ops[join]);
    check(waitBothPaths ? f.verify() : f.rejects(notCompleted), waitBothPaths ? "a predecessor in flight on both paths completes at the join" : "completion on only one untagged path does not satisfy the join");
  }
}
// Exits drain pending launches first, then require every effect of the visit.
void exits(MLIRContext &context) {
  for (bool launched : {true, false}) {
    StringRef expected = launched ? pending : incomplete;
    {
      Fixture f(context, loopText); check(bool(f.contract), "edge-drain fixture builds");
      if (f.contract) {
        if (launched) f.launch(0);
        Operation *enter = f.jump(0); f.nop();
        if (!launched) f.group(0); else f.complete(0);
        f.block = 1; size_t header = f.ops.size(); f.group(1); f.group(2); Operation *branch = f.branch(); f.nop(); Operation *exit = f.jump(2); f.nop();
        Operation *back = f.jump(1); f.nop(); f.block = 2; f.halt();
        f.aim(enter, f.ops[header]); f.aim(branch, back); f.aim(back, f.ops[header]); f.aim(exit, f.ops.back());
        check(f.rejects(expected), launched ? "a tagged source edge cannot leave an effect pending" : "a tagged source edge requires its block's effects");
      }
    }
    {
      Fixture f(context, sourceText); check(bool(f.contract), "halt-drain fixture builds");
      if (f.contract) {
        f.group(0); f.group(1); if (launched) f.launch(2); f.halt();
        if (launched) f.complete(2); else f.group(2);
        check(f.rejects(expected), launched ? "a halt cannot leave an effect pending" : "a halt requires its block's effects");
      }
    }
    {
      Fixture f(context, sourceText); check(bool(f.contract), "fall-off fixture builds");
      if (f.contract) {
        f.group(0); f.group(1); if (launched) f.launch(2);
        Operation *skip = f.jump(-1); f.nop();
        if (launched) f.complete(2); else f.group(2);
        f.nop(); f.aim(skip, f.ops.back());
        check(f.rejects(expected), launched ? "falling off the stream cannot leave an effect pending" : "falling off the stream requires its block's effects");
      }
    }
  }
  Fixture f(context, sourceText); check(bool(f.contract), "release-drain fixture builds");
  if (f.contract) {
    f.group(0); f.group(1); f.launch(2); f.release(); f.complete(2); f.halt();
    check(f.rejects(pending), "the completion CSR write cannot leave an effect pending");
  }
}
// Swapping the store/wait tiles (3-5) with the load/await tiles (6-8) in both contracts keeps them mutually consistent but inverts source order.
void renumbered(MLIRContext &context) {
  Fixture f(context, sourceText); check(bool(f.contract), "renumbering fixture builds"); if (!f.contract) return;
  auto map = [](int32_t id) { return id >= 3 && id < 6 ? id + 3 : id >= 6 && id < 9 ? id - 3 : id; };
  auto remap = [&](DictionaryAttr d, StringRef key) {
    SmallVector<int32_t> ids;
    for (int32_t id : d.getAs<DenseI32ArrayAttr>(key).asArrayRef()) ids.push_back(map(id));
    NamedAttrList fields(d); fields.set(key, f.b.getDenseI32ArrayAttr(ids)); return fields;
  };
  SmallVector<Attribute> tiles(f.tiles.size());
  for (Attribute a : f.tiles) {
    auto fields = remap(cast<DictionaryAttr>(a), "after");
    int32_t id = map(field(cast<DictionaryAttr>(a), "id"));
    fields.set("id", f.b.getI32IntegerAttr(id)); tiles[id] = fields.getDictionary(&context);
  }
  SmallVector<Attribute> operations;
  for (Attribute a : f.cfg.getAs<ArrayAttr>("operations")) operations.push_back(remap(cast<DictionaryAttr>(a), "tile_commands").getDictionary(&context));
  NamedAttrList cfg(f.cfg); cfg.set("operations", f.b.getArrayAttr(operations));
  diagnostics.clear();
  auto built = buildAtlasSourceMemoryEffectContract(*f.source->getOps<func::FuncOp>().begin(), cfg.getDictionary(&context), f.b.getArrayAttr(tiles));
  check(failed(built) && diagnosed("source memory contract cannot derive block-local source spans and completions"), "consistently renumbered tile ids cannot invert source order");
}
void emptyStream(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module attributes {
    atlas.generated_from_virtual = "resource-contract-v4", atlas.timing_state = "untimed",
    atlas.virtual_dma_contract = [], atlas.virtual_mxu_contract = [], atlas.virtual_tile_contract = [],
    atlas.virtual_source_memory_contract = {effects = []},
    atlas.virtual_cfg_contract = {
      values = [], operations = [], edges = [],
      blocks = [{id = 0 : i32, condition = -1 : i32, args = array<i32>,
                 live_in = array<i32>, operations = array<i32>, edges = array<i32>}]
    }
  } {
    %s = "atlas.start"() : () -> !atlas.state
  })mlir", &context);
  check(bool(module), "empty issued stream fixture parses"); if (!module) return;
  diagnostics.clear();
  check(failed(verifyAtlasGeneratedSourceMemoryEffectContract(*module)) && diagnosed("at least one encoded instruction"), "the module entry rejects an empty stream while decoding");
  // A caller-built context can still carry an empty decoded stream; the checker guards it itself.
  AtlasVerificationContext ctx; ctx.module = *module; ctx.generated = true; ctx.stream.emplace();
  diagnostics.clear();
  check(failed(verifyAtlasGeneratedSourceMemoryEffectContract(ctx)) && diagnosed("source memory contract requires a nonempty issued instruction stream"), "an empty decoded stream is rejected before indexing PC zero");
}
void gate(MLIRContext &context) {
  Fixture f(context, sourceText); check(bool(f.contract), "gate fixture builds"); if (!f.contract) return;
  f.group(0); f.group(1); f.group(2); f.halt();
  check(f.verify(), "complete source order verifies");
  (*f.issued)->setAttr("atlas.generated_from_virtual", f.b.getStringAttr("resource-contract-v3"));
  check(f.rejects("unsupported Atlas virtual-to-machine artifact marker"), "the previous marker is unsupported even with the contract");
  (*f.issued)->setAttr("atlas.generated_from_virtual", f.b.getStringAttr("resource-contract-v4"));
  (*f.issued)->removeAttr("atlas.virtual_source_memory_contract");
  check(f.rejects("artifact requires atlas.virtual_source_memory_contract"), "generated artifacts require the contract");
}
} // namespace
int main() {
  MLIRContext context; context.getOrLoadDialect<AtlasDialect>(); context.getOrLoadDialect<arith::ArithDialect>(); context.getOrLoadDialect<cf::ControlFlowDialect>(); context.getOrLoadDialect<func::FuncDialect>();
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &d) { diagnostics += d.str() + '\n'; return success(); });
  straight(context); boundary(context); loops(context); branches(context); lifecycle(context); joins(context); exits(context); renumbered(context); emptyStream(context); gate(context);
  llvm::outs() << checks << " source-memory-effect checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
