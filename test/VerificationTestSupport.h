// Shared checks for the C++ verification suites. One executable runs the suite named by its argument; each suite
// gets a fresh context whose diagnostics (with notes) accumulate in `diagnostics` instead of being printed. Typed-SSA
// fixtures claim resource placements by hand, never through the allocator.
#ifndef ATLAS_TEST_VERIFICATIONTESTSUPPORT_H
#define ATLAS_TEST_VERIFICATIONTESTSUPPORT_H

#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace atlas_test {
using namespace mlir;

inline unsigned checks = 0, failures = 0;
inline std::string diagnostics;

inline void check(bool condition, llvm::StringRef name, llvm::StringRef detail = {}) {
  ++checks;
  if (condition)
    return;
  ++failures;
  llvm::errs() << "FAIL: " << name << '\n';
  if (!detail.empty())
    llvm::errs() << detail << '\n';
}

inline bool diagnosed(llvm::StringRef message) { return llvm::StringRef(diagnostics).contains(message); }

// Runs `verifier` (returning LogicalResult); a valid case emits no diagnostics, and every fragment must appear.
template <typename Verifier>
void expectVerified(llvm::StringRef name, bool valid, llvm::ArrayRef<llvm::StringRef> fragments, Verifier &&verifier) {
  diagnostics.clear();
  bool accepted = succeeded(verifier());
  check(accepted == valid, name, diagnostics);
  if (valid)
    check(diagnostics.empty(), name, diagnostics);
  for (llvm::StringRef fragment : fragments)
    check(diagnosed(fragment), name, "missing diagnostic fragment '" + fragment.str() + "':\n" + diagnostics);
}

inline int64_t integer(DictionaryAttr record, llvm::StringRef name) { return record.getAs<IntegerAttr>(name).getInt(); }

// A signless i32 field equal to `expected` modulo 2^32.
inline void field(DictionaryAttr record, llvm::StringRef name, int64_t expected) {
  auto value = record.getAs<IntegerAttr>(name);
  check(value && value.getType().isSignlessInteger(32) && value.getValue().getZExtValue() == uint32_t(expected), name);
}

inline std::string replace(std::string text, llvm::StringRef from, llvm::StringRef to) {
  size_t position = text.find(from.str());
  check(position != std::string::npos, "fixture contains " + from.str());
  if (position != std::string::npos)
    text.replace(position, from.size(), to.str());
  return text;
}

// Parses typed source that must verify; null otherwise.
inline OwningOpRef<ModuleOp> parse(MLIRContext &context, llvm::StringRef text, llvm::StringRef name) {
  auto module = parseSourceString<ModuleOp>(text, &context);
  bool valid = module && succeeded(verify(*module));
  check(valid, name);
  return valid ? std::move(module) : OwningOpRef<ModuleOp>();
}

inline func::FuncOp firstFunction(ModuleOp module) { return *module.getOps<func::FuncOp>().begin(); }

template <typename... HandleTypes> struct HandleFixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<Value> handles;
  HandleFixture(MLIRContext &context, llvm::StringRef source, llvm::StringRef name = "fixture is typed SSA")
      : module(parse(context, source, name)) {
    if (!module)
      return;
    function = firstFunction(*module);
    function.walk([&](Operation *op) {
      for (Value result : op->getResults())
        if (isa<HandleTypes...>(result.getType()))
          handles.push_back(result);
    });
  }
};

// Applies `mutate` to a copy of `base` and checks it with the suite's `expect`; the case is valid exactly when no
// diagnostic fragment is expected.
template <typename Fixture, typename Assignments, typename Mutate>
void mutated(Fixture &fixture, const Assignments &base, llvm::StringRef name, Mutate mutate,
             llvm::ArrayRef<llvm::StringRef> fragments = {}) {
  auto changed = base;
  mutate(changed);
  expect(fixture, name, changed, fragments.empty(), fragments);
}

// Generates typed MXU source only, threading one virtual state through its effects.
struct Source {
  static constexpr llvm::StringLiteral state = "!atlas.virtual_state", fp8 = "!atlas.virtual_fp8", bf16 = "!atlas.virtual_bf16";
  std::string body;
  unsigned next = 2;
  std::string current = "%s2";

  explicit Source(unsigned scaleCode = 127)
      : body(R"mlir(module { func.func @test() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %seed = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %scale = "atlas.virtual_scale_constant"() {code = )mlir" + std::to_string(scaleCode) + R"mlir( : i32} : () -> !atlas.virtual_scale
)mlir") {}
  static std::string type(llvm::StringRef bank, unsigned unit) { return "!atlas.virtual_mxu_" + bank.str() + "<" + std::to_string(unit) + ">"; }
  std::string effect(llvm::StringRef name, std::string args, std::string types, std::string resultType, std::string attributes = "") {
    unsigned index = ++next;
    std::string result = "%h" + std::to_string(index), after = "%s" + std::to_string(index);
    body += "    " + after + ", " + result + " = \"atlas.virtual_" + name.str() + "\"(" + current + ", " + args + ") " + attributes;
    body += " : (" + state.str() + ", " + types + ") -> (" + state.str() + ", " + resultType + ")\n";
    current = after;
    return result;
  }
  std::string unitAttr(unsigned unit) { return "{unit = " + std::to_string(unit) + " : i32}"; }
  std::string weight(unsigned unit) { return effect("mxu_load_weight", "%x", fp8.str(), type("weight", unit), unitAttr(unit)); }
  std::string seed(unsigned unit, bool narrow = false) {
    return effect(narrow ? "mxu_load_acc_fp8" : "mxu_load_acc_bf16", narrow ? "%x" : "%seed", narrow ? fp8.str() : bf16.str(), type("acc", unit), unitAttr(unit));
  }
  std::string reset(std::string weight, unsigned unit) {
    return effect("mxu_reset", "%x, " + weight, fp8.str() + ", " + type("weight", unit), type("acc", unit));
  }
  std::string accumulate(std::string weight, std::string acc, unsigned unit) {
    return effect("mxu_accumulate", "%x, " + weight + ", " + acc, fp8.str() + ", " + type("weight", unit) + ", " + type("acc", unit), type("acc", unit));
  }
  void readout(std::string acc, unsigned unit, bool narrow = false) {
    effect(narrow ? "mxu_readout_fp8" : "mxu_readout_bf16", acc + (narrow ? ", %scale" : ""), type("acc", unit) + (narrow ? ", !atlas.virtual_scale" : ""), narrow ? fp8.str() : bf16.str());
  }
  void legacy(unsigned unit) {
    body += "    %legacy" + std::to_string(++next) + " = \"atlas.virtual_mxu_matmul\"(%x, %x) " + unitAttr(unit) + " : (" + fp8.str() + ", " + fp8.str() + ") -> " + bf16.str() + "\n";
  }
  void edge(llvm::StringRef block, llvm::StringRef argument) {
    body += "    cf.br ^" + block.str() + "(" + current + " : !atlas.virtual_state)\n  ^" + block.str() + "(" + argument.str() + ": !atlas.virtual_state):\n";
    current = argument.str();
  }
  std::string finish() const {
    return body + "    %out = \"atlas.virtual_output_bf16\"(" + current + ", %seed) {index = 0 : i32} : (" + state.str() + ", " + bf16.str() + ") -> " + state.str() + "\n    return %out : " + state.str() + "\n} }";
  }
};

// Parses `text` and runs `pipeline`; null on any failure.
inline OwningOpRef<ModuleOp> runPipeline(MLIRContext &context, llvm::StringRef text, llvm::StringRef pipeline) {
  auto module = parseSourceString<ModuleOp>(text, &context);
  PassManager pm = PassManager::on<ModuleOp>(&context);
  if (!module || failed(parsePassPipeline(pipeline, pm)) || failed(pm.run(*module)))
    return {};
  return module;
}

// A generated module whose issued stream holds no instruction.
constexpr llvm::StringLiteral emptyIssuedStream = R"mlir(module attributes {
    atlas.generated_from_virtual = "resource-contract-v5", atlas.timing_state = "untimed",
    atlas.virtual_dma_contract = [], atlas.virtual_mxu_contract = [], atlas.virtual_tile_contract = [],
    atlas.virtual_source_memory_contract = {effects = []}, atlas.virtual_buffer_contract = {packs = [], reads = []},
    atlas.virtual_cfg_contract = {values = [], operations = [], edges = [], blocks = [{id = 0 : i32, condition = -1 : i32,
      args = array<i32>, live_in = array<i32>, operations = array<i32>, edges = array<i32>}]}
  } {
    %s = "atlas.start"() : () -> !atlas.state
  })mlir";

// A hand-issued physical stream; operations carry the CFG block tag when `block` is nonnegative.
struct IssuedStream {
  MLIRContext &context;
  OpBuilder b;
  OwningOpRef<ModuleOp> module;
  Value state;
  std::vector<Operation *> ops;
  int block = -1;
  explicit IssuedStream(MLIRContext &c) : context(c), b(&c), module(ModuleOp::create(b.getUnknownLoc())) {
    c.getOrLoadDialect<atlas::AtlasDialect>();
    b.setInsertionPointToEnd(module->getBody());
    OperationState start(b.getUnknownLoc(), "atlas.start");
    start.addTypes(atlas::StateType::get(&c));
    state = b.create(start)->getResult(0);
  }
  NamedAttribute i(llvm::StringRef name, int64_t n) { return b.getNamedAttr(name, b.getI32IntegerAttr(n)); }
  NamedAttribute text(llvm::StringRef name, llvm::StringRef s) { return b.getNamedAttr(name, b.getStringAttr(s)); }
  Operation *emit(llvm::StringRef name, std::initializer_list<NamedAttribute> attributes, int command = -1, int edge = -1) {
    OperationState op(b.getUnknownLoc(), name);
    op.addOperands(state);
    op.addTypes(atlas::StateType::get(&context));
    op.addAttributes(attributes);
    auto tag = [&](llvm::StringRef name, int value) {
      if (value >= 0)
        op.addAttribute(name, b.getI32IntegerAttr(value));
    };
    tag(atlas::kAtlasTagCFGBlock, block);
    tag(atlas::kAtlasTagTileCommand, command);
    tag(atlas::kAtlasTagCFGEdge, edge);
    ops.push_back(b.create(op));
    state = ops.back()->getResult(0);
    return ops.back();
  }
  // LUI then ADDI, omitting a zero half.
  void constant(int reg, uint32_t value) {
    int32_t low = int32_t(value << 20) >> 20;
    uint32_t high = (value - uint32_t(low)) >> 12;
    if (high)
      emit("atlas.upper", {text("kind", "lui"), i("dst", reg), i("immediate", high)});
    if (low || !high)
      emit("atlas.alu_imm", {text("kind", "addi"), i("dst", reg), i("src", high ? reg : 0), i("immediate", low)});
  }
  void nop() { emit("atlas.alu_imm", {text("kind", "addi"), i("dst", 0), i("src", 0), i("immediate", 0)}); }
  void markGenerated(DictionaryAttr cfg, ArrayAttr tiles, DictionaryAttr sourceMemory, DictionaryAttr buffer) {
    (*module)->setAttr(atlas::kAtlasGeneratedMarker, b.getStringAttr(atlas::kAtlasGeneratedVersion));
    (*module)->setAttr(atlas::kAtlasTimingState, b.getStringAttr("untimed"));
    for (llvm::StringRef name : {atlas::kAtlasDMAContract, atlas::kAtlasMXUContract}) (*module)->setAttr(name, b.getArrayAttr({}));
    (*module)->setAttr(atlas::kAtlasCFGContract, cfg);
    (*module)->setAttr(atlas::kAtlasTileContract, tiles);
    (*module)->setAttr(atlas::kAtlasSourceMemoryContract, sourceMemory);
    (*module)->setAttr(atlas::kAtlasBufferContract, buffer);
  }
  void halt() { emit("atlas.trap", {text("kind", "ecall")}); }
  Operation *jump(int edge = -1) { return emit("atlas.jump", {text("kind", "jal"), i("dst", 0), i("base", 0), i("offset", 0)}, -1, edge); }
  Operation *branch() { return emit("atlas.branch", {text("kind", "bne"), i("lhs", 18), i("rhs", 0), i("offset_bytes", 0)}); }
  // Points a jump or branch at ops[target].
  void aim(Operation *redirect, size_t target) {
    int64_t offset = 2 * (int64_t(target) - (llvm::find(ops, redirect) - ops.begin()));
    redirect->setAttr(isa<atlas::BranchOp>(redirect) ? "offset_bytes" : "offset", b.getI32IntegerAttr(offset));
  }
  void aim(Operation *redirect, Operation *target) { aim(redirect, llvm::find(ops, target) - ops.begin()); }
};

using Suite = void (*)(MLIRContext &);
void runBufferContract(MLIRContext &);
void runCFGContract(MLIRContext &);
void runDMAAllocation(MLIRContext &);
void runDMAHelper(MLIRContext &);
void runMXUAllocation(MLIRContext &);
void runMXUContract(MLIRContext &);
void runSourceMemoryContract(MLIRContext &);
void runTileContract(MLIRContext &);
void runTimingProvider(MLIRContext &);
} // namespace atlas_test

#endif
