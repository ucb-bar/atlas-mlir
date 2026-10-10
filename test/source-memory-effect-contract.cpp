#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasVerificationContext.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;
namespace {
constexpr StringLiteral notCompleted = "overlapping source predecessor has not completed in this visit";
constexpr StringLiteral incomplete = "source block exits before its required effects complete";
constexpr StringLiteral pending = "source block exits with a pending DRAM effect";
constexpr StringLiteral state = "!atlas.virtual_state", fp8 = "!atlas.virtual_fp8";

// An FP8 input, then `body`; `extraAttributes` extends the DRAM bases.
std::string program(StringRef arguments, StringRef extraAttributes, StringRef body) {
  return ("module {\n  func.func @effects(" + arguments + ") -> !atlas.virtual_state attributes {\n"
          "    atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64" + extraAttributes + "\n  } {\n"
          "    %s0 = \"atlas.virtual_start\"() : () -> !atlas.virtual_state\n"
          "    %s1, %input = \"atlas.virtual_input_fp8\"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n" +
          body + "  }\n}").str();
}

// Stores `tile` then reloads the same source span: the store is the load's source predecessor.
std::string storeThenLoad(StringRef in, StringRef tile) {
  return R"mlir(    %a = arith.constant -1879048192 : i32
    %b = arith.constant -1879048192 : i32
    %size = arith.constant 1024 : i32
    %s2, %store = "atlas.virtual_dma_store_fp8"()mlir" + in.str() + ", " + tile.str() + R"mlir(, %a, %size) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s3 = "atlas.virtual_dma_wait"(%s2, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s4, %load = "atlas.virtual_dma_load_fp8"(%s3, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s5, %result = "atlas.virtual_dma_await_fp8"(%s4, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
)mlir";
}

std::string blockHeader(StringRef name, StringRef arguments) { return ("  ^" + name + "(" + arguments + "):\n").str(); }

const std::string sourceText = program("", "", storeThenLoad("%s1", "%input") + "    return %s5 : !atlas.virtual_state\n");
const std::string loopText = program("", "",
    "    cf.br ^loop(%s1, %input : !atlas.virtual_state, !atlas.virtual_fp8)\n" + blockHeader("loop", "%state : !atlas.virtual_state, %tile : !atlas.virtual_fp8") +
    storeThenLoad("%state", "%tile") + "    %choose = arith.constant true\n"
    "    cf.cond_br %choose, ^loop(%s5, %tile : !atlas.virtual_state, !atlas.virtual_fp8), ^exit(%s5 : !atlas.virtual_state)\n" +
    blockHeader("exit", "%final : !atlas.virtual_state") + "    return %final : !atlas.virtual_state\n");
const std::string branchText = program("%choose : i1", ",\n    atlas.control_dram_base = 2466250752 : i64",
    "    cf.cond_br %choose, ^left(%s1, %input : !atlas.virtual_state, !atlas.virtual_fp8), ^right(%s1 : !atlas.virtual_state)\n" +
    blockHeader("left", "%state : !atlas.virtual_state, %tile : !atlas.virtual_fp8") + storeThenLoad("%state", "%tile") +
    "    cf.br ^join(%s5 : !atlas.virtual_state)\n" + blockHeader("right", "%rstate : !atlas.virtual_state") + R"mlir(    %raddr = arith.constant -1879048192 : i32
    %rsize = arith.constant 1024 : i32
    %r1, %rload = "atlas.virtual_dma_load_fp8"(%rstate, %raddr, %rsize) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %r2, %rresult = "atlas.virtual_dma_await_fp8"(%r1, %rload) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    cf.br ^join(%r2 : !atlas.virtual_state)
)mlir" + blockHeader("join", "%final : !atlas.virtual_state") + "    return %final : !atlas.virtual_state\n");

// Literal source expectations and manually issued groups isolate this obligation; final integration separately
// checks complete CFG/tile/lifecycle metadata and source value semantics before invoking this checker.
struct Fixture : IssuedStream {
  OwningOpRef<ModuleOp> source;
  DictionaryAttr cfg, contract;
  ArrayAttr tiles;
  Fixture(MLIRContext &c, StringRef text, StringRef name) : IssuedStream(c), source(parseSourceString<ModuleOp>(text, &c)) {
    block = 0;
    build();
    check(bool(contract), name);
  }
  explicit operator bool() const { return bool(contract); }
  void build() {
    if (!source) return;
    auto function = firstFunction(*source);
    SmallVector<VirtualRegisterAssignment> registers;
    SmallVector<VirtualDMAAssignment> dma;
    unsigned scalar = 19, tensor = 12, transfer = 0;
    for (Block &body : function.getBody()) {
      auto add = [&](Value v) {
        if (v.getType().isInteger(1) || v.getType().isInteger(32)) registers.push_back({v, scalar++});
        else if (isa<VirtualFP8Type>(v.getType())) registers.push_back({v, tensor++});
        else if (isa<VirtualBF16Type>(v.getType())) { registers.push_back({v, tensor}); tensor += 2; }
      };
      llvm::for_each(body.getArguments(), add);
      for (Operation &op : body) {
        llvm::for_each(op.getResults(), add);
        if (isa<VirtualDMAStoreFP8Op, VirtualDMALoadFP8Op>(op)) {
          dma.push_back({op.getResult(1), {2 + transfer, 1, transfer, 131072 + 256 * transfer, 4, 5, 9}});
          ++transfer;
        }
      }
    }
    FixedResourcePlacement fixed{};
    fixed.inputWindowWords = fixed.outputWindowWords = fixed.outputWord = 65536;
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
    (*module)->setAttr(kAtlasGeneratedMarker, b.getStringAttr(kAtlasGeneratedVersion));
    (*module)->setAttr(kAtlasTimingState, b.getStringAttr("untimed"));
    for (StringRef name : {kAtlasDMAContract, kAtlasMXUContract}) (*module)->setAttr(name, b.getArrayAttr({}));
    (*module)->setAttr(kAtlasCFGContract, cfg);
    (*module)->setAttr(kAtlasTileContract, tiles);
    (*module)->setAttr(kAtlasSourceMemoryContract, contract);
    (*module)->setAttr(kAtlasBufferContract, b.getDictionaryAttr({}));
  }
  DictionaryAttr tile(unsigned effect, StringRef role) {
    auto e = cast<DictionaryAttr>(contract.getAs<ArrayAttr>("effects")[effect]);
    return cast<DictionaryAttr>(tiles[e.getAs<IntegerAttr>(role).getInt()]);
  }
  // A group is its launch (address setup, staging VSTOREs, DMA) then its completion (WAIT, staging VLOADs).
  void launch(unsigned effect) {
    auto t = tile(effect, "launch");
    bool store = t.getAs<StringAttr>("kind").getValue() == "dma_store";
    constant(4, uint32_t(integer(t, "vmem_byte")) / 4);
    constant(5, uint32_t(integer(t, "dram_byte")));
    constant(9, integer(t, "bytes"));
    if (store) for (int id : t.getAs<DenseI32ArrayAttr>("after").asArrayRef())
      emit("atlas.vstore", {i("src", integer(cast<DictionaryAttr>(tiles[id]), "reg")), i("base", 4), i("offset", 0), text("format", "raw")}, id);
    emit("atlas.dma", {text("direction", store ? "store" : "load"), i("reg", 4), i("dram", 5), i("size", 9), i("channel", integer(t, "channel"))}, integer(t, "id"));
  }
  void complete(unsigned effect) {
    auto t = tile(effect, "completion");
    int completion = integer(t, "id");
    emit("atlas.dma_wait", {i("channel", integer(t, "channel"))}, completion);
    if (tile(effect, "launch").getAs<StringAttr>("kind").getValue() == "dma_load") for (Attribute a : tiles) {
      auto v = cast<DictionaryAttr>(a);
      if (v.getAs<StringAttr>("kind").getValue() != "vload" || v.getAs<DenseI32ArrayAttr>("after").asArrayRef() != ArrayRef<int32_t>({completion})) continue;
      emit("atlas.vload", {i("dst", integer(v, "reg")), i("base", 4), i("offset", 0), text("format", "raw")}, integer(v, "id"));
    }
  }
  void group(unsigned effect) { launch(effect); complete(effect); }
  void release() { emit("atlas.csr", {text("kind", "rrw"), i("dst", 0), i("source", 10), i("address", 0xc10)}); }
  // Block 0 runs `entry` and jumps over `skipped` to the block-1 loop header running `body`; the backedge targets
  // `back` (default: the header).
  void loop(function_ref<void()> entry, function_ref<void()> body, function_ref<size_t(size_t)> back = {}, function_ref<void()> skipped = {}) {
    entry();
    Operation *enter = jump(0); nop();
    if (skipped) skipped();
    block = 1; size_t header = ops.size(); body();
    Operation *exitBranch = branch(); nop(); Operation *exit = jump(2); nop();
    Operation *backedge = jump(1); nop(); block = 2; halt();
    aim(enter, header); aim(exitBranch, backedge); aim(backedge, back ? back(header) : header); aim(exit, ops.back());
  }
  bool verify() { diagnostics.clear(); return succeeded(verifyAtlasGeneratedSourceMemoryEffectContract(*module)); }
  bool rejects(StringRef message) { return !verify() && diagnosed(message); }
};

void straight(MLIRContext &context) {
  for (unsigned offset : {0u, 512u, 2048u}) {
    std::string source = replace(sourceText, "%b = arith.constant -1879048192", "%b = arith.constant " + std::to_string(int32_t(0x90000000u + offset)));
    for (bool reorder : {false, true}) {
      Fixture f(context, source, "literal source memory contract builds"); if (!f) continue;
      auto effects = f.contract.getAs<ArrayAttr>("effects");
      check(effects.size() == 3, "literal implicit-input/explicit-store/explicit-load cardinality");
      auto store = cast<DictionaryAttr>(effects[1]), load = cast<DictionaryAttr>(effects[2]);
      check(integer(store, "source") == 5 && integer(store, "launch") == 4 && integer(store, "completion") == 5, "literal store source identity and completion");
      check(integer(load, "source") == 7 && uint32_t(integer(load, "dram_byte")) == 0x90000000u + offset && integer(load, "bytes") == 1024, "literal load span and source identity");
      auto pred = load.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef();
      check(offset < 1024 ? pred == ArrayRef<int32_t>({1}) : pred.empty(), "literal source-overlap predecessor set");
      f.group(0); if (reorder) { f.group(2); f.group(1); } else { f.group(1); f.group(2); } f.halt();
      check(reorder && offset < 1024 ? f.rejects(notCompleted) : f.verify(), offset >= 1024 ? "disjoint write/read groups may reorder" : reorder ? "completed overlapping groups on disjoint staging cannot reorder" : "source store completes before overlapping read issue");
    }
  }
  // Replacing the first write by a read: equal host spans carry no source effect edge, even with completed groups swapped.
  std::string reads = sourceText;
  size_t begin = reads.find("    %s2, %store"), end = reads.find("    %s4, %load");
  reads.replace(begin, end - begin, R"mlir(    %s2, %first = "atlas.virtual_dma_load_fp8"(%s1, %a, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s3, %first_result = "atlas.virtual_dma_await_fp8"(%s2, %first) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
)mlir");
  if (Fixture f(context, reads, "literal read/read source builds"); f) {
    check(cast<DictionaryAttr>(f.contract.getAs<ArrayAttr>("effects")[2]).getAs<DenseI32ArrayAttr>("predecessors").empty(), "same-span read/read has no ordering edge");
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
    Fixture f(context, text, "literal overlapping boundary I/O source builds"); if (!f) continue;
    auto effects = f.contract.getAs<ArrayAttr>("effects");
    check(effects.size() == 4, "literal boundary BF16 half-effect count");
    auto first = cast<DictionaryAttr>(effects[2]), second = cast<DictionaryAttr>(effects[3]);
    check(integer(first, "source") == 2 && integer(first, "launch") == 7 && integer(first, "completion") == 8 &&
          first.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef() == ArrayRef<int32_t>({0}) &&
          second.getAs<DenseI32ArrayAttr>("predecessors").asArrayRef() == ArrayRef<int32_t>({1}),
          "boundary writes retain the literal corresponding prior source reads");
    const unsigned order[2][4] = {{0, 1, 2, 3}, {2, 0, 1, 3}};
    for (unsigned effect : order[reorder]) f.group(effect);
    f.halt(); check(reorder ? f.rejects(notCompleted) : f.verify(), reorder ? "completed boundary write cannot precede its overlapping source read" : "serialized overlapping boundary I/O preserves source order");
  }
}
void loops(MLIRContext &context) {
  for (bool stale : {false, true}) {
    Fixture f(context, loopText, "literal loop source builds"); if (!f) continue;
    size_t read = 0;
    f.loop([&] { f.group(0); }, [&] { f.group(1); read = f.ops.size(); f.group(2); }, [&](size_t header) { return stale ? read : header; });
    check(stale ? f.rejects(notCompleted) : f.verify(), stale ? "previous loop visit completion cannot satisfy skipped current predecessor" : "each source loop visit independently completes conflicting predecessor");
  }
}
void branches(MLIRContext &context) {
  for (bool reorder : {false, true}) {
    Fixture f(context, branchText, "literal branch source builds"); if (!f) continue;
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
    f.aim(choose, right); f.aim(left, leftBody); f.aim(right, rightBody); f.aim(leftExit, f.ops.back()); f.aim(rightExit, f.ops.back());
    check(reorder ? f.rejects(notCompleted) : f.verify(), reorder ? "branch-local completed transfers cannot reverse their source overlap" : "untaken branch predecessor is not imposed on the other branch");
  }
}
// Each issued stream isolates one lifecycle rejection; later failures in the same stream are not reached.
void lifecycle(MLIRContext &context) {
  if (Fixture f(context, sourceText, "missing-completion fixture builds"); f) {
    f.group(0); f.group(1); f.halt();
    check(f.rejects("source memory contract requires one issued launch and completion per effect"), "every effect issues exactly one launch and completion");
  }
  if (Fixture f(context, sourceText, "early-completion fixture builds"); f) {
    f.group(0); f.complete(1); f.launch(1); f.group(2); f.halt();
    check(f.rejects("completion has no matching current launch"), "completion before its launch is rejected");
  }
  if (Fixture f(context, sourceText, "issued-loop fixture builds"); f) {
    f.group(0); size_t body = f.ops.size(); f.group(1); Operation *again = f.branch(); f.nop(); f.group(2); f.halt();
    f.aim(again, body);
    check(f.rejects("effect launched twice within one source visit"), "an untagged issued loop cannot relaunch within one source visit");
  }
  if (Fixture f(context, branchText, "shared-channel fixture builds"); f) {
    check(f.tile(0, "launch").getAs<IntegerAttr>("channel") == f.tile(1, "launch").getAs<IntegerAttr>("channel"), "mailbox and input share their load channel");
    f.launch(0); f.launch(1); f.complete(0); f.complete(1); f.group(2); f.group(3); f.group(4); f.halt();
    check(f.rejects("effect channel is not idle"), "a launch cannot reuse a channel with a pending effect");
  }
  if (Fixture f(context, loopText, "wrong-block fixture builds"); f) {
    f.loop([&] { f.group(0); f.group(1); }, [&] { f.group(2); });
    check(f.rejects("launch belongs to a different source-block visit"), "a launch issued in another source block is rejected");
  }
}
// The store is predecessor 1 of the overlapping load (effect 2); both paths of an untagged branch join before the load.
void joins(MLIRContext &context) {
  for (bool waitBothPaths : {true, false}) {
    Fixture f(context, sourceText, "must-join fixture builds"); if (!f) continue;
    f.group(0); f.launch(1); Operation *skip = f.branch(); f.nop();
    if (waitBothPaths) f.nop(); else f.complete(1);
    size_t join = f.ops.size();
    if (waitBothPaths) f.complete(1);
    f.group(2); f.halt(); f.aim(skip, join);
    check(waitBothPaths ? f.verify() : f.rejects(notCompleted), waitBothPaths ? "a predecessor in flight on both paths completes at the join" : "completion on only one untagged path does not satisfy the join");
  }
}
// Exits drain pending launches first, then require every effect of the visit.
void exits(MLIRContext &context) {
  for (bool launched : {true, false}) {
    StringRef expected = launched ? pending : incomplete;
    if (Fixture f(context, loopText, "edge-drain fixture builds"); f) {
      // The entry effect is still pending (or not yet issued) when the tagged entry edge leaves block 0.
      f.loop([&] { if (launched) f.launch(0); }, [&] { f.group(1); f.group(2); }, {}, [&] { if (launched) f.complete(0); else f.group(0); });
      check(f.rejects(expected), launched ? "a tagged source edge cannot leave an effect pending" : "a tagged source edge requires its block's effects");
    }
    if (Fixture f(context, sourceText, "halt-drain fixture builds"); f) {
      f.group(0); f.group(1); if (launched) f.launch(2); f.halt();
      if (launched) f.complete(2); else f.group(2);
      check(f.rejects(expected), launched ? "a halt cannot leave an effect pending" : "a halt requires its block's effects");
    }
    if (Fixture f(context, sourceText, "fall-off fixture builds"); f) {
      f.group(0); f.group(1); if (launched) f.launch(2);
      Operation *skip = f.jump(); f.nop();
      if (launched) f.complete(2); else f.group(2);
      f.nop(); f.aim(skip, f.ops.back());
      check(f.rejects(expected), launched ? "falling off the stream cannot leave an effect pending" : "falling off the stream requires its block's effects");
    }
  }
  if (Fixture f(context, sourceText, "release-drain fixture builds"); f) {
    f.group(0); f.group(1); f.launch(2); f.release(); f.complete(2); f.halt();
    check(f.rejects(pending), "the completion CSR write cannot leave an effect pending");
  }
}
// Swapping the store/wait tiles (3-5) with the load/await tiles (6-8) in both contracts keeps them mutually consistent but inverts source order.
void renumbered(MLIRContext &context) {
  Fixture f(context, sourceText, "renumbering fixture builds"); if (!f) return;
  auto map = [](int32_t id) { return id >= 3 && id < 6 ? id + 3 : id >= 6 && id < 9 ? id - 3 : id; };
  auto remap = [&](DictionaryAttr d, StringRef key) {
    SmallVector<int32_t> ids;
    for (int32_t id : d.getAs<DenseI32ArrayAttr>(key).asArrayRef()) ids.push_back(map(id));
    NamedAttrList fields(d); fields.set(key, f.b.getDenseI32ArrayAttr(ids)); return fields;
  };
  SmallVector<Attribute> tiles(f.tiles.size());
  for (Attribute a : f.tiles) {
    auto fields = remap(cast<DictionaryAttr>(a), "after");
    int32_t id = map(integer(cast<DictionaryAttr>(a), "id"));
    fields.set("id", f.b.getI32IntegerAttr(id)); tiles[id] = fields.getDictionary(&context);
  }
  SmallVector<Attribute> operations;
  for (Attribute a : f.cfg.getAs<ArrayAttr>("operations")) operations.push_back(remap(cast<DictionaryAttr>(a), "tile_commands").getDictionary(&context));
  NamedAttrList cfg(f.cfg); cfg.set("operations", f.b.getArrayAttr(operations));
  diagnostics.clear();
  auto built = buildAtlasSourceMemoryEffectContract(firstFunction(*f.source), cfg.getDictionary(&context), f.b.getArrayAttr(tiles));
  check(failed(built) && diagnosed("source memory contract cannot derive block-local source spans and completions"), "consistently renumbered tile ids cannot invert source order");
}
void emptyStream(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(emptyIssuedStream, &context);
  check(bool(module), "empty issued stream fixture parses"); if (!module) return;
  diagnostics.clear();
  check(failed(verifyAtlasGeneratedSourceMemoryEffectContract(*module)) && diagnosed("at least one encoded instruction"), "the module entry rejects an empty stream while decoding");
  // A caller-built context can still carry an empty decoded stream; the checker guards it itself.
  AtlasVerificationContext ctx; ctx.module = *module; ctx.generated = true; ctx.stream.emplace();
  diagnostics.clear();
  check(failed(verifyAtlasGeneratedSourceMemoryEffectContract(ctx)) && diagnosed("source memory contract requires a nonempty issued instruction stream"), "an empty decoded stream is rejected before indexing PC zero");
}
} // namespace
void atlas_test::runSourceMemoryContract(MLIRContext &context) {
  straight(context); boundary(context); loops(context); branches(context); lifecycle(context); joins(context); exits(context); renumbered(context); emptyStream(context);
}
