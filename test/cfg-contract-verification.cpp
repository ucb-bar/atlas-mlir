#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "VerificationTestSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <algorithm>
using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;
namespace {
constexpr StringLiteral source = R"mlir(module {
  func.func @cfg(%choose: i1) -> !atlas.virtual_state {
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
void run(MLIRContext &context) {
  auto module = parse(context,source,"typed source parses independently");
  if (!module) return;
  auto function = *module->getOps<func::FuncOp>().begin();
  // Literal source identities and intentionally nonpreferred registers form
  // the oracle. No allocator, lowering pass, or planned instructions are used.
  constexpr unsigned selected[] = {19,40,26,21,22,23,42,44,46,24};
  SmallVector<VirtualRegisterAssignment> assignments;
  for (Block &block : function.getBody()) {
    auto add = [&](Value v) {
      Type t = v.getType();
      if (t.isInteger(1) || t.isInteger(32) || isa<VirtualBF16Type>(t))
        assignments.push_back({v,selected[assignments.size()]});
    };
    for (Value v : block.getArguments()) add(v);
    for (Operation &op : block) for (Value v : op.getResults()) add(v);
  }
  auto contract = buildAtlasCFGContract(function,assignments);
  check(succeeded(contract),"source contract builds from explicit claims");
  if (failed(contract)) return;
  auto values = contract->getAs<ArrayAttr>("values");
  auto blocks = contract->getAs<ArrayAttr>("blocks");
  auto edges = contract->getAs<ArrayAttr>("edges");
  check(values.size() == 10 && blocks.size() == 4 && edges.size() == 4,"literal source cardinalities");
  for (unsigned n = 0; n < values.size(); ++n) {
    auto r = cast<DictionaryAttr>(values[n]);
    check(integer(r,"id") == int32_t(n) && integer(r,"reg") == int32_t(selected[n]),"stable value identity and supplied placement");
  }
  auto sum = cast<DictionaryAttr>(values[4]);
  auto cmp = cast<DictionaryAttr>(values[5]);
  check(sum.getAs<StringAttr>("def").getValue() == "arith.addi" && sum.getAs<DenseI32ArrayAttr>("operands").asArrayRef() == ArrayRef<int32_t>({2,3}),"source add operands");
  check(integer(cmp,"predicate") == int32_t(arith::CmpIPredicate::slt) && cmp.getAs<DenseI32ArrayAttr>("operands").asArrayRef() == ArrayRef<int32_t>({2,4}),"source compare predicate and operands");
  auto entry = cast<DictionaryAttr>(blocks[0]);
  check(integer(entry,"condition") == 0 && entry.getAs<DenseI32ArrayAttr>("edges").asArrayRef() == ArrayRef<int32_t>({0,1}),"true and false source edges retain order");
  auto left = cast<DictionaryAttr>(edges[2]), right = cast<DictionaryAttr>(edges[3]);
  check(integer(left,"from") == 1 && integer(left,"to") == 3 && left.getAs<DenseI32ArrayAttr>("incoming").asArrayRef() == ArrayRef<int32_t>({6,4}),"left simultaneous incoming values");
  check(integer(right,"from") == 2 && integer(right,"to") == 3 && right.getAs<DenseI32ArrayAttr>("incoming").asArrayRef() == ArrayRef<int32_t>({7,3}),"right simultaneous incoming values");
  std::reverse(assignments.begin(),assignments.end());
  auto reordered = buildAtlasCFGContract(function,assignments);
  check(succeeded(reordered) && *reordered == *contract,"placement vector order is not the oracle");
  assignments.pop_back();
  check(failed(buildAtlasCFGContract(function,assignments)),"missing source claim fails");
  assignments.push_back(assignments.front());
  check(failed(buildAtlasCFGContract(function,assignments)),"duplicate claim fails");
}

void delaySlot(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(source,&context);
  check(bool(module),"local delay-slot source parses");
  if (!module) return;
  auto function = *module->getOps<func::FuncOp>().begin();
  Builder b(&context);
  function->setAttr("atlas.input_dram_base",b.getI64IntegerAttr(2415919104));
  function->setAttr("atlas.output_dram_base",b.getI64IntegerAttr(2415923200));
  function->setAttr("atlas.control_dram_base",b.getI64IntegerAttr(2415927296));
  PassManager passes(&context);
  check(succeeded(parsePassPipeline("lower-atlas-virtual-to-machine",passes)),"local lowering pipeline parses");
  if (failed(passes.run(*module))) { check(false,"local delay-slot fixture lowers"); return; }
  check(succeeded(verifyAtlasGeneratedCFGContract(*module)),"standalone CFG checker accepts fixture");
  for (Operation &op : module->getBody()->getOperations()) {
    if (!isa<BranchOp>(op)) continue;
    auto slot = cast<ALUImmOp>(op.getNextNode());
    slot->setAttr("dst",b.getI32IntegerAttr(9));
    check(failed(verifyAtlasGeneratedCFGContract(*module)),"standalone CFG checker rejects semantic delay-slot write");
    return;
  }
  check(false,"local delay-slot fixture contains branch");
}

void rawTensorWrites(MLIRContext &context) {
  // Insert valid physical operations without source producer tags immediately
  // before the second output half. Killing only the first half of a pair would
  // leave the second store's old identity intact and incorrectly accept it.
  for (StringRef kind : {"vli.all", "vli.row", "vli.col", "vli.one",
                         "reduce", "transpose", "unpack", "pack",
                         "load", "binary", "unary", "pop.bf16", "pop.fp8"}) {
    auto module = parseSourceString<ModuleOp>(source,&context);
    if (!module) { check(false,"raw tensor fixture parses"); return; }
    auto function = *module->getOps<func::FuncOp>().begin();
    Builder b(&context);
    function->setAttr("atlas.input_dram_base",b.getI64IntegerAttr(2415919104));
    function->setAttr("atlas.output_dram_base",b.getI64IntegerAttr(2415923200));
    function->setAttr("atlas.control_dram_base",b.getI64IntegerAttr(2415927296));
    PassManager passes(&context);
    if (failed(parsePassPipeline("lower-atlas-virtual-to-machine",passes)) ||
        failed(passes.run(*module))) {
      check(false,"raw tensor fixture lowers"); return;
    }
    VStoreOp store;
    for (Operation &op : module->getBody()->getOperations())
      if (auto candidate = dyn_cast<VStoreOp>(op))
        if (candidate.getSrc() % 2) store = candidate;
    if (!store) { check(false,"raw tensor fixture has second output half"); return; }
    unsigned single = store.getSrc(), pair = single - 1;
    SmallVector<NamedAttribute> fields;
    auto integer = [&](StringRef name, unsigned value) {
      fields.push_back(b.getNamedAttr(name,b.getI32IntegerAttr(value)));
    };
    auto text = [&](StringRef name, StringRef value) {
      fields.push_back(b.getNamedAttr(name,b.getStringAttr(value)));
    };
    StringRef name;
    if (kind.starts_with("vli.")) {
      name = "atlas.vli";
      StringRef mode = kind.drop_front(4);
      text("mode",mode); integer("dst",mode == "all" || mode == "row" ? pair : single);
      integer("immediate",1);
    } else if (kind == "reduce") {
      name = "atlas.vpu_reduce"; text("kind","row_sum"); integer("dst",pair); integer("src",32);
    } else if (kind == "transpose") {
      name = "atlas.xlu_transpose"; integer("dst",single); integer("src",32);
    } else if (kind == "unpack" || kind == "pack") {
      name = "atlas.vpu_pack"; text("direction",kind == "unpack" ? "fp8_to_bf16" : "bf16_to_fp8");
      integer("dst",kind == "unpack" ? pair : single); integer("src",32); integer("scale_reg",1);
    } else if (kind == "load") {
      name = "atlas.vload"; text("format","raw"); integer("dst",single); integer("base",4); integer("offset",0);
    } else if (kind == "binary") {
      name = "atlas.vpu_binary"; text("kind","add"); integer("dst",pair); integer("lhs",32); integer("rhs",34);
    } else if (kind == "unary") {
      name = "atlas.vpu_unary"; text("kind","relu"); integer("dst",pair); integer("src",32);
    } else {
      name = "atlas.mxu_pop"; text("format",kind == "pop.bf16" ? "bf16" : "fp8");
      integer("dst",kind == "pop.bf16" ? pair : single); integer("unit",0); integer("slot",0);
      integer("scale_reg",kind == "pop.bf16" ? 0 : 1);
    }
    fields.push_back(b.getNamedAttr("atlas.virtual_cfg_block",store->getAttr("atlas.virtual_cfg_block")));
    OpBuilder insertion(store);
    OperationState state(store.getLoc(),name);
    state.addOperands(store.getState()); state.addTypes(store.getState().getType()); state.addAttributes(fields);
    Operation *write = insertion.create(state);
    store->setOperand(0,write->getResult(0));
    check(succeeded(verify(*module)),std::string("valid inserted tensor writer: ") + kind.str());
    check(failed(verifyAtlasGeneratedCFGContract(*module)),std::string("actual origin killed by tensor writer: ") + kind.str());
  }
}

void emptyIssuedStream(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module attributes {
    atlas.generated_from_virtual = "resource-contract-v5", atlas.timing_state = "untimed",
    atlas.virtual_dma_contract = [], atlas.virtual_mxu_contract = [], atlas.virtual_tile_contract = [],
    atlas.virtual_source_memory_contract = {effects = []}, atlas.virtual_buffer_contract = {packs = [], reads = []},
    atlas.virtual_cfg_contract = {
      values = [], operations = [], edges = [],
      blocks = [{id = 0 : i32, condition = -1 : i32, args = array<i32>,
                 live_in = array<i32>, operations = array<i32>, edges = array<i32>}]
    }
  } {
    %s = "atlas.start"() : () -> !atlas.state
  })mlir",&context);
  check(bool(module),"empty issued stream fixture parses");
  if (module)
    check(failed(verifyAtlasGeneratedCFGContract(*module)),"empty issued stream fails without indexing PC zero");
}
void bindings(MLIRContext &context) {
  constexpr StringLiteral text = R"mlir(module {
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
  })mlir";
  auto module = parse(context,text,"binding fixture typed source");
  if (!module) return;
  auto function = *module->getOps<func::FuncOp>().begin();
  SmallVector<VirtualRegisterAssignment> claims;
  unsigned scalarReg = 19, tensorReg = 32;
  auto add = [&](Value v) {
    Type t = v.getType();
    if (t.isInteger(1) || t.isInteger(32)) claims.push_back({v,scalarReg++});
    else if (isa<VirtualBF16Type,VirtualFP8Type>(t)) {
      if (isa<VirtualBF16Type>(t) && tensorReg % 2) ++tensorReg;
      claims.push_back({v,tensorReg}); tensorReg += isa<VirtualBF16Type>(t) ? 2 : 1;
    }
  };
  for (Value v : function.getArguments()) add(v);
  for (Operation &op : function.getBody().front()) for (Value v : op.getResults()) add(v);
  auto contract = buildAtlasCFGContract(function,claims);
  check(succeeded(contract),"source binding contract builds");
  if (failed(contract)) return;
  auto records = contract->getAs<ArrayAttr>("operations");
  auto association = [&](StringRef name, StringRef fieldName, ArrayRef<int32_t> expected) {
    bool found = false;
    for (Attribute attr : records) {
      auto r = cast<DictionaryAttr>(attr);
      if (r.getAs<StringAttr>("name").getValue() != name) continue;
      found = true;
      check(r.getAs<DenseI32ArrayAttr>(fieldName).asArrayRef() == expected,"literal source-to-command binding");
    }
    check(found,"source binding operation exists");
  };
  association("atlas.virtual_input_bf16","tile_commands",{3,4,5,6,7,8});
  association("atlas.virtual_pack_fp8","tile_commands",{9,10});
  association("atlas.virtual_dma_load_fp8","tile_commands",{11});
  association("atlas.virtual_dma_await_fp8","tile_commands",{12,13});
  association("atlas.virtual_output_bf16","tile_commands",{14,15,16,17,18,19});
  association("atlas.virtual_dma_store_bf16","tile_commands",{20,21,22});
  association("atlas.virtual_dma_wait","tile_commands",{23});
  association("atlas.virtual_mxu_matmul","mxu_commands",{0,1,2});
  association("atlas.virtual_mxu_load_weight","mxu_commands",{3});
  association("atlas.virtual_mxu_reset","mxu_commands",{4});
  association("atlas.virtual_mxu_readout_bf16","mxu_commands",{5});
}
}
void atlas_test::runCFGContract(MLIRContext &context) {
  run(context);
  bindings(context);
  delaySlot(context);
  rawTensorWrites(context);
  emptyIssuedStream(context);
}
