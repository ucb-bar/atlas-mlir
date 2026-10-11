#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "VerificationTestSupport.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral source = R"mlir(module {
  func.func @cfg(%choose: i1) -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64, atlas.control_dram_base = 2415927296 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %tile = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %sum = arith.addi %zero, %one : i32
    %more = arith.cmpi slt, %zero, %sum : i32
    cf.cond_br %choose, ^left(%s1, %tile : !atlas.virtual_state, !atlas.virtual_bf16), ^right(%s1, %tile : !atlas.virtual_state, !atlas.virtual_bf16)
  ^left(%ls: !atlas.virtual_state, %lt: !atlas.virtual_bf16):
    cf.br ^join(%ls, %lt, %sum : !atlas.virtual_state, !atlas.virtual_bf16, i32)
  ^right(%rs: !atlas.virtual_state, %rt: !atlas.virtual_bf16):
    cf.br ^join(%rs, %rt, %one : !atlas.virtual_state, !atlas.virtual_bf16, i32)
  ^join(%js: !atlas.virtual_state, %jt: !atlas.virtual_bf16, %index: i32):
    %s2 = "atlas.virtual_output_bf16"(%js, %jt) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
})mlir";

ArrayRef<int32_t> ids(DictionaryAttr record, StringRef name) { return record.getAs<DenseI32ArrayAttr>(name).asArrayRef(); }

// Literal source identities and intentionally nonpreferred registers form the oracle; no allocator or lowering is used.
void run(MLIRContext &context) {
  auto module = parse(context, source, "typed source parses independently");
  if (!module)
    return;
  auto function = firstFunction(*module);
  constexpr unsigned selected[] = {19, 40, 26, 21, 22, 23, 42, 44, 46, 24};
  SmallVector<VirtualRegisterAssignment> assignments;
  for (Block &block : function.getBody()) {
    auto add = [&](Value v) {
      if (v.getType().isInteger(1) || v.getType().isInteger(32) || isa<VirtualBF16Type>(v.getType()))
        assignments.push_back({v, selected[assignments.size()]});
    };
    llvm::for_each(block.getArguments(), add);
    for (Operation &op : block)
      llvm::for_each(op.getResults(), add);
  }
  auto contract = buildAtlasCFGContract(function, assignments);
  check(succeeded(contract), "source contract builds from explicit claims");
  if (failed(contract))
    return;
  auto values = contract->getAs<ArrayAttr>("values"), blocks = contract->getAs<ArrayAttr>("blocks"), edges = contract->getAs<ArrayAttr>("edges");
  check(values.size() == 10 && blocks.size() == 4 && edges.size() == 4, "literal source cardinalities");
  for (unsigned n = 0; n < values.size(); ++n) {
    auto r = cast<DictionaryAttr>(values[n]);
    check(integer(r, "id") == int32_t(n) && integer(r, "reg") == int32_t(selected[n]), "stable value identity and supplied placement");
  }
  auto sum = cast<DictionaryAttr>(values[4]), cmp = cast<DictionaryAttr>(values[5]);
  check(sum.getAs<StringAttr>("def").getValue() == "arith.addi" && ids(sum, "operands") == ArrayRef<int32_t>({2, 3}), "source add operands");
  check(integer(cmp, "predicate") == int32_t(arith::CmpIPredicate::slt) && ids(cmp, "operands") == ArrayRef<int32_t>({2, 4}),
        "source compare predicate and operands");
  auto entry = cast<DictionaryAttr>(blocks[0]);
  check(integer(entry, "condition") == 0 && ids(entry, "edges") == ArrayRef<int32_t>({0, 1}), "true and false source edges retain order");
  auto left = cast<DictionaryAttr>(edges[2]), right = cast<DictionaryAttr>(edges[3]);
  check(integer(left, "from") == 1 && integer(left, "to") == 3 && ids(left, "incoming") == ArrayRef<int32_t>({6, 4}), "left simultaneous incoming values");
  check(integer(right, "from") == 2 && integer(right, "to") == 3 && ids(right, "incoming") == ArrayRef<int32_t>({7, 3}), "right simultaneous incoming values");
  std::reverse(assignments.begin(), assignments.end());
  auto reordered = buildAtlasCFGContract(function, assignments);
  check(succeeded(reordered) && *reordered == *contract, "placement vector order is not the oracle");
  assignments.pop_back();
  check(failed(buildAtlasCFGContract(function, assignments)), "missing source claim fails");
  assignments.push_back(assignments.front());
  check(failed(buildAtlasCFGContract(function, assignments)), "duplicate claim fails");
}

void delaySlot(MLIRContext &context) {
  auto module = runPipeline(context, source, "lower-atlas-virtual-to-machine");
  check(module && succeeded(verifyAtlasGeneratedCFGContract(*module)), "standalone CFG checker accepts fixture", diagnostics);
  if (!module)
    return;
  Operation *branch = nullptr;
  module->walk([&](BranchOp op) { branch = branch ? branch : op; });
  check(branch, "local delay-slot fixture contains branch");
  if (!branch)
    return;
  cast<ALUImmOp>(branch->getNextNode())->setAttr("dst", Builder(&context).getI32IntegerAttr(9));
  check(failed(verifyAtlasGeneratedCFGContract(*module)), "standalone CFG checker rejects semantic delay-slot write");
}

// Untagged physical writers go just before the second output half: killing only the first half of a pair would
// leave the second store's old identity intact and incorrectly accept it.
void rawTensorWrites(MLIRContext &context) {
  struct Writer { StringRef name; bool pair; StringRef fields; };
  const Writer writers[] = {
      {"atlas.vli", true, "mode = \"all\", immediate = 1 : i32"},
      {"atlas.vli", true, "mode = \"row\", immediate = 1 : i32"},
      {"atlas.vli", false, "mode = \"col\", immediate = 1 : i32"},
      {"atlas.vli", false, "mode = \"one\", immediate = 1 : i32"},
      {"atlas.vpu_reduce", true, "kind = \"row_sum\", src = 32 : i32"},
      {"atlas.xlu_transpose", false, "src = 32 : i32"},
      {"atlas.vpu_pack", true, "direction = \"fp8_to_bf16\", src = 32 : i32, scale_reg = 1 : i32"},
      {"atlas.vpu_pack", false, "direction = \"bf16_to_fp8\", src = 32 : i32, scale_reg = 1 : i32"},
      {"atlas.vload", false, "format = \"raw\", base = 4 : i32, offset = 0 : i32"},
      {"atlas.vpu_binary", true, "kind = \"add\", lhs = 32 : i32, rhs = 34 : i32"},
      {"atlas.vpu_unary", true, "kind = \"relu\", src = 32 : i32"},
      {"atlas.mxu_pop", true, "format = \"bf16\", unit = 0 : i32, slot = 0 : i32, scale_reg = 0 : i32"},
      {"atlas.mxu_pop", false, "format = \"fp8\", unit = 0 : i32, slot = 0 : i32, scale_reg = 1 : i32"}};
  for (const Writer &w : writers) {
    auto module = runPipeline(context, source, "lower-atlas-virtual-to-machine");
    VStoreOp store;
    if (module)
      module->walk([&](VStoreOp candidate) { store = candidate.getSrc() % 2 ? candidate : store; });
    if (!store) {
      check(false, "raw tensor fixture lowers to a second output half");
      return;
    }
    std::string fields = "{dst = " + std::to_string(store.getSrc() - w.pair) + " : i32, " + w.fields.str() + "}";
    OperationState state(store.getLoc(), w.name);
    state.addOperands(store.getState());
    state.addTypes(store.getState().getType());
    state.addAttributes(cast<DictionaryAttr>(parseAttribute(fields, &context)).getValue());
    state.addAttribute(kAtlasTagCFGBlock, store->getAttr(kAtlasTagCFGBlock));
    store->setOperand(0, OpBuilder(store).create(state)->getResult(0));
    std::string label = w.name.str() + " " + fields;
    check(succeeded(verify(*module)), "valid inserted tensor writer: " + label);
    check(failed(verifyAtlasGeneratedCFGContract(*module)), "actual origin killed by tensor writer: " + label);
  }
}

void emptyStream(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(emptyIssuedStream, &context);
  check(module && failed(verifyAtlasGeneratedCFGContract(*module)), "empty issued stream fails without indexing PC zero");
}

void bindings(MLIRContext &context) {
  auto module = parse(context, R"mlir(module {
    func.func @bindings(%choose: i1) -> !atlas.virtual_state {
      %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
      %s1, %seed = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
      %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
      %addr = arith.constant -1879048192 : i32
      %size = arith.constant 1024 : i32
      %size_bf16 = arith.constant 2048 : i32
      %s2, %load = "atlas.virtual_dma_load_fp8"(%s1, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
      %s3, %x = "atlas.virtual_dma_await_fp8"(%s2, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
      %legacy = "atlas.virtual_mxu_matmul"(%x, %packed) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
      %s4, %weight = "atlas.virtual_mxu_load_weight"(%s3, %packed) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
      %s5, %acc = "atlas.virtual_mxu_reset"(%s4, %x, %weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
      %s6, %y = "atlas.virtual_mxu_readout_bf16"(%s5, %acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
      %s7 = "atlas.virtual_output_bf16"(%s6, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
      %s8, %store = "atlas.virtual_dma_store_bf16"(%s7, %legacy, %addr, %size_bf16) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
      %s9 = "atlas.virtual_dma_wait"(%s8, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
      return %s9 : !atlas.virtual_state
    }
  })mlir", "binding fixture typed source");
  if (!module)
    return;
  auto function = firstFunction(*module);
  SmallVector<VirtualRegisterAssignment> claims;
  unsigned scalarReg = 19, tensorReg = 32;
  auto add = [&](Value v) {
    Type t = v.getType();
    if (t.isInteger(1) || t.isInteger(32)) {
      claims.push_back({v, scalarReg++});
    } else if (isa<VirtualBF16Type, VirtualFP8Type>(t)) {
      bool bf16 = isa<VirtualBF16Type>(t);
      tensorReg += bf16 && tensorReg % 2;
      claims.push_back({v, tensorReg});
      tensorReg += bf16 ? 2 : 1;
    }
  };
  llvm::for_each(function.getArguments(), add);
  for (Operation &op : function.getBody().front())
    llvm::for_each(op.getResults(), add);
  auto contract = buildAtlasCFGContract(function, claims);
  check(succeeded(contract), "source binding contract builds");
  if (failed(contract))
    return;
  const std::tuple<StringRef, StringRef, SmallVector<int32_t>> expected[] = {
      {"atlas.virtual_input_bf16", "tile_commands", {3, 4, 5, 6, 7, 8}},
      {"atlas.virtual_pack_fp8", "tile_commands", {9, 10}},
      {"atlas.virtual_dma_load_fp8", "tile_commands", {11}},
      {"atlas.virtual_dma_await_fp8", "tile_commands", {12, 13}},
      {"atlas.virtual_output_bf16", "tile_commands", {14, 15, 16, 17, 18, 19}},
      {"atlas.virtual_dma_store_bf16", "tile_commands", {20, 21, 22}},
      {"atlas.virtual_dma_wait", "tile_commands", {23}},
      {"atlas.virtual_mxu_matmul", "mxu_commands", {0, 1, 2}},
      {"atlas.virtual_mxu_load_weight", "mxu_commands", {3}},
      {"atlas.virtual_mxu_reset", "mxu_commands", {4}},
      {"atlas.virtual_mxu_readout_bf16", "mxu_commands", {5}}};
  for (const auto &[name, fieldName, commands] : expected) {
    bool found = false;
    for (Attribute attr : contract->getAs<ArrayAttr>("operations")) {
      auto r = cast<DictionaryAttr>(attr);
      if (r.getAs<StringAttr>("name").getValue() != name)
        continue;
      found = true;
      check(ids(r, fieldName) == ArrayRef<int32_t>(commands), "literal source-to-command binding");
    }
    check(found, "source binding operation exists");
  }
}
} // namespace

void atlas_test::runCFGContract(MLIRContext &context) {
  run(context);
  bindings(context);
  delaySlot(context);
  rawTensorWrites(context);
  emptyStream(context);
}
