#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "Atlas/AtlasTypes.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;

namespace {
unsigned failures = 0;
unsigned checks = 0;

void check(bool condition, StringRef name, StringRef detail = {}) {
  ++checks;
  if (condition)
    return;
  ++failures;
  llvm::errs() << "FAIL: " << name << '\n';
  if (!detail.empty())
    llvm::errs() << detail << '\n';
}

using Assignments = SmallVector<VirtualRegisterAssignment>;

// Fixtures start with the production allocator's answer. Tests change only
// copies of its public placements, never source IR or allocator internals.
struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  VirtualAllocationPlan plan;
  Assignments assignments;
  SmallVector<int32_t> abi;

  bool initialize(MLIRContext &context, StringRef source, StringRef name) {
    module = parseSourceString<ModuleOp>(source, &context);
    check(bool(module), name, "fixture failed to parse");
    if (!module)
      return false;
    bool valid = succeeded(verify(*module)) &&
                 succeeded(verifyAtlasVirtualModule(*module));
    check(valid, name, "fixture is not a verified virtual CFG");
    if (!valid)
      return false;
    function = *module->getOps<func::FuncOp>().begin();
    valid = succeeded(plan.allocate(function));
    check(valid, name, "initial allocation failed");
    if (!valid)
      return false;
    auto append = [&](Value value) {
      Type type = value.getType();
      if (isa<VirtualBF16Type>(type))
        assignments.push_back({value, plan.tile(value)});
      else if (isa<VirtualFP8Type>(type))
        assignments.push_back({value, plan.fp8(value)});
      else if (type.isInteger(1) || type.isInteger(32))
        assignments.push_back({value, plan.scalar(value)});
    };
    for (Block &block : function.getBody()) {
      for (BlockArgument argument : block.getArguments())
        append(argument);
      for (Operation &op : block)
        for (Value result : op.getResults())
          append(result);
    }
    abi.append(plan.scalarArguments().begin(), plan.scalarArguments().end());
    return true;
  }

  Value result(StringRef operation, unsigned resultIndex = 0,
               unsigned occurrence = 0) {
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (op.getName().getStringRef() == operation) {
          if (occurrence-- == 0)
            return op.getResult(resultIndex);
        }
    llvm::report_fatal_error("test fixture result not found");
  }

  Block &block(unsigned index) {
    auto it = function.getBody().begin();
    std::advance(it, index);
    return *it;
  }
};

unsigned regFor(ArrayRef<VirtualRegisterAssignment> assignments, Value value) {
  for (const auto &assignment : assignments)
    if (assignment.value == value)
      return assignment.reg;
  llvm::report_fatal_error("test fixture assignment not found");
}

void setReg(Assignments &assignments, Value value, unsigned reg) {
  for (auto &assignment : assignments)
    if (assignment.value == value) {
      assignment.reg = reg;
      return;
    }
  llvm::report_fatal_error("test fixture assignment not found");
}

void expect(Fixture &fixture, StringRef name, const Assignments &assignments,
            ArrayRef<int32_t> abi, const FixedResourcePlacement &fixed,
            bool valid, ArrayRef<StringRef> diagnosticParts = {}) {
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  ScopedDiagnosticHandler handler(fixture.function.getContext(),
                                 [&](Diagnostic &diagnostic) {
    diagnostic.print(stream);
    stream << '\n';
    for (const Diagnostic &note : diagnostic.getNotes()) {
      note.print(stream);
      stream << '\n';
    }
    return success();
  });
  bool accepted = succeeded(verifyAtlasRegisterAllocation(
      fixture.function, assignments, fixed, abi));
  stream.flush();
  check(accepted == valid, name, diagnostics);
  if (!valid)
    for (StringRef part : diagnosticParts)
      check(StringRef(diagnostics).contains(part), name,
            "missing diagnostic fragment '" + part.str() + "':\n" + diagnostics);
  else
    check(diagnostics.empty(), name, diagnostics);
}

void expectValid(Fixture &fixture, StringRef name,
                 const Assignments &assignments) {
  expect(fixture, name, assignments, fixture.abi, fixture.plan.fixed(), true);
}

void expectInvalid(Fixture &fixture, StringRef name,
                   const Assignments &assignments,
                   ArrayRef<StringRef> diagnosticParts) {
  expect(fixture, name, assignments, fixture.abi, fixture.plan.fixed(), false,
         diagnosticParts);
}

constexpr StringLiteral straightLine = R"mlir(
module {
  func.func @straight() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %sum = "atlas.virtual_vpu_binary"(%a, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %s3 = "atlas.virtual_output_bf16"(%s2, %sum) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s4, %c = "atlas.virtual_input_bf16"(%s3) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %d = "atlas.virtual_vpu_unary"(%c) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %s5 = "atlas.virtual_output_bf16"(%s4, %d) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s5 : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral scalars = R"mlir(
module {
  func.func @controls(%a: i32, %b: i32) -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %c = arith.constant 1 : i32
    %d = arith.addi %c, %c : i32
    %s1, %tile = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2 = "atlas.virtual_output_bf16"(%s1, %tile) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral packing = R"mlir(
module {
  func.func @packing() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %c = arith.constant 1 : i32
    %s1, %tile = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %p = "atlas.virtual_pack_fp8"(%tile) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %s2 = "atlas.virtual_output_bf16"(%s1, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral mixed = R"mlir(
module {
  func.func @mixed() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %p = "atlas.virtual_input_fp8"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %y = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %s3 = "atlas.virtual_output_bf16"(%s2, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %b = "atlas.virtual_vpu_unary"(%a) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %s4 = "atlas.virtual_output_bf16"(%s3, %b) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s5, %q = "atlas.virtual_input_fp8"(%s4) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %z = "atlas.virtual_mxu_matmul"(%q, %q) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %s6 = "atlas.virtual_output_bf16"(%s5, %z) {index = 2 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s6 : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral liveThrough = R"mlir(
module {
  func.func @live_through() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    cf.br ^next(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
  ^next(%state: !atlas.virtual_state, %arg: !atlas.virtual_bf16):
    %dead = "atlas.virtual_vpu_unary"(%arg) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum = "atlas.virtual_vpu_binary"(%a, %arg) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %out = "atlas.virtual_output_bf16"(%state, %sum) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %out : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral unusedEdgeArgument = R"mlir(
module {
  func.func @unused_edge_argument() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    cf.br ^next(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
  ^next(%state: !atlas.virtual_state, %unused: !atlas.virtual_bf16):
    %result = "atlas.virtual_vpu_unary"(%a) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %out = "atlas.virtual_output_bf16"(%state, %result) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %out : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral swapLoop = R"mlir(
module {
  func.func @swap_loop() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a0 = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %b0 = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    cf.br ^loop(%s2, %a0, %b0, %zero : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)
  ^loop(%state: !atlas.virtual_state, %a: !atlas.virtual_bf16, %b: !atlas.virtual_bf16, %i: i32):
    %more = arith.cmpi slt, %i, %one : i32
    %next = arith.addi %i, %one : i32
    cf.cond_br %more, ^loop(%state, %b, %a, %next : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32), ^exit(%state, %a : !atlas.virtual_state, !atlas.virtual_bf16)
  ^exit(%out_state: !atlas.virtual_state, %selected: !atlas.virtual_bf16):
    %out = "atlas.virtual_output_bf16"(%out_state, %selected) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %out : !atlas.virtual_state
  }
})mlir";

void testTensorAndCoverage(MLIRContext &context) {
  Fixture fixture, foreign;
  if (!fixture.initialize(context, straightLine, "straight fixture") ||
      !foreign.initialize(context, straightLine, "foreign fixture"))
    return;
  expectValid(fixture, "allocator baseline", fixture.assignments);
  Value a = fixture.result("atlas.virtual_input_bf16", 1);
  Value b = fixture.result("atlas.virtual_input_bf16", 1, 1);
  Value c = fixture.result("atlas.virtual_input_bf16", 1, 2);
  Value d = fixture.result("atlas.virtual_vpu_unary");
  Assignments reused = fixture.assignments;
  setReg(reused, c, regFor(reused, a));
  setReg(reused, d, 60);
  expectValid(fixture, "BF16 reuse after death", reused);

  auto candidate = fixture.assignments;
  setReg(candidate, b, regFor(candidate, a));
  expectInvalid(fixture, "overlapping BF16 pairs", candidate,
                {"register allocation overlap", "tensor", "first value",
                 "conflicting value", "%"});
  candidate = fixture.assignments;
  setReg(candidate, d, regFor(candidate, c));
  expectInvalid(fixture, "last-use operand/result alias", candidate,
                {"register allocation overlap", "operation results and operands", "%"});
  for (auto test : {std::make_pair(1u, "even tensor register"),
                    std::make_pair(64u, "outside tensor register bank"),
                    std::make_pair(62u, "reserved tensor temporary")}) {
    candidate = fixture.assignments;
    setReg(candidate, a, test.first);
    expectInvalid(fixture, "invalid BF16 placement", candidate,
                  {test.second, "tensor", "%"});
  }
  candidate = fixture.assignments;
  llvm::erase_if(candidate, [&](const auto &entry) { return entry.value == a; });
  expectInvalid(fixture, "missing assignment", candidate,
                {"missing register assignment", "%"});
  candidate = fixture.assignments;
  candidate.push_back({a, regFor(candidate, a)});
  expectInvalid(fixture, "duplicate assignment", candidate,
                {"duplicate register assignment", "%"});
  candidate = fixture.assignments;
  candidate.push_back({foreign.result("atlas.virtual_input_bf16", 1), 0});
  expectInvalid(fixture, "foreign assignment", candidate,
                {"foreign, stale, or untracked value"});
  candidate = fixture.assignments;
  candidate.push_back({fixture.result("atlas.virtual_start"), 0});
  expectInvalid(fixture, "state is not a register value", candidate,
                {"unsupported value type", "%"});
}

void testScalars(MLIRContext &context) {
  Fixture fixture, pack;
  if (!fixture.initialize(context, scalars, "scalar fixture") ||
      !pack.initialize(context, packing, "packing fixture"))
    return;
  expectValid(fixture, "scalar baseline", fixture.assignments);
  expectValid(pack, "pack baseline", pack.assignments);
  Value c = fixture.result("arith.constant");
  for (unsigned reg : {0u, 4u, 27u, 28u}) {
    auto candidate = fixture.assignments;
    setReg(candidate, c, reg);
    std::string registerName = "x" + std::to_string(reg);
    expectInvalid(fixture, "scalar scratch reservation", candidate,
                  {"outside scalar allocation pool", registerName, "%"});
  }
  auto candidate = fixture.assignments;
  setReg(candidate, c, 32);
  expectInvalid(fixture, "scalar outside bank", candidate,
                {"outside scalar allocation pool", "x32", "%"});
  candidate = fixture.assignments;
  setReg(candidate, fixture.result("arith.addi"), regFor(candidate, c));
  expectInvalid(fixture, "scalar operand/result alias", candidate,
                {"register allocation overlap", "operation results and operands", "x", "%"});

  auto fixed = fixture.plan.fixed();
  fixed.scaleReg = regFor(fixture.assignments, c);
  expect(fixture, "scale register bank is separate", fixture.assignments,
         fixture.abi, fixed, true);
  for (unsigned reg = 10; reg <= 17; ++reg) {
    candidate = pack.assignments;
    setReg(candidate, pack.result("arith.constant"), reg);
    std::string registerName = "x" + std::to_string(reg);
    expectInvalid(pack, "pack scalar reservation", candidate,
                  {"reserved scalar register", registerName, "%"});
  }

  candidate = fixture.assignments;
  Value first = fixture.function.getArgument(0);
  Value second = fixture.function.getArgument(1);
  setReg(candidate, second, regFor(candidate, first));
  auto abi = fixture.abi;
  abi[1] = abi[0];
  expect(fixture, "dead entry arguments overlap at staging", candidate, abi,
         fixture.plan.fixed(), false,
         {"register allocation overlap", "entry argument staging", "x", "%"});
  abi = fixture.abi;
  abi[0] = abi[0] == 26 ? 25 : 26;
  expect(fixture, "ABI placement mismatch", fixture.assignments, abi,
         fixture.plan.fixed(), false, {"scalar argument ABI", "does not match"});
  abi = fixture.abi;
  abi.pop_back();
  expect(fixture, "ABI missing argument", fixture.assignments, abi,
         fixture.plan.fixed(), false, {"scalar argument ABI"});
}

void testMixedTensorBanks(MLIRContext &context) {
  Fixture fixture;
  if (!fixture.initialize(context, mixed, "mixed fixture"))
    return;
  expectValid(fixture, "mixed baseline", fixture.assignments);
  Value a = fixture.result("atlas.virtual_input_bf16", 1);
  Value p = fixture.result("atlas.virtual_input_fp8", 1);
  Value q = fixture.result("atlas.virtual_input_fp8", 1, 1);
  auto candidate = fixture.assignments;
  setReg(candidate, p, regFor(candidate, a) + 1);
  expectInvalid(fixture, "FP8 aliases BF16 upper half", candidate,
                {"register allocation overlap", "tensor", "%"});
  candidate = fixture.assignments;
  setReg(candidate, a, 32);
  setReg(candidate, fixture.result("atlas.virtual_vpu_unary"), 34);
  setReg(candidate, fixture.result("atlas.virtual_mxu_matmul"), 36);
  setReg(candidate, fixture.result("atlas.virtual_mxu_matmul", 0, 1), 38);
  setReg(candidate, p, 0);
  setReg(candidate, q, 33);
  expectValid(fixture, "FP8 reuses dead BF16 upper half", candidate);
  setReg(candidate, q, 64);
  expectInvalid(fixture, "FP8 outside bank", candidate,
                {"outside tensor register bank", "tensor", "%"});
}

void testCFG(MLIRContext &context) {
  Fixture captured, unused, loop;
  if (!captured.initialize(context, liveThrough, "live-through fixture") ||
      !unused.initialize(context, unusedEdgeArgument, "unused edge fixture") ||
      !loop.initialize(context, swapLoop, "swap-loop fixture"))
    return;
  expectValid(captured, "live-through baseline", captured.assignments);
  expectValid(unused, "unused edge argument baseline", unused.assignments);
  expectValid(loop, "loop baseline", loop.assignments);
  Value a = captured.result("atlas.virtual_input_bf16", 1);
  auto candidate = captured.assignments;
  setReg(candidate, captured.block(1).getArgument(1), regFor(candidate, a));
  expectInvalid(captured, "edge destination clobbers live-through", candidate,
                {"register allocation overlap",
                 "edge destinations and live-through values", "tensor", "%"});
  candidate = captured.assignments;
  setReg(candidate, captured.result("atlas.virtual_vpu_unary"),
         regFor(candidate, a));
  expectInvalid(captured, "dead result clobbers live-through", candidate,
                {"register allocation overlap",
                 "operation results and live-after values", "tensor", "%"});
  candidate = unused.assignments;
  setReg(candidate, unused.block(1).getArgument(1),
         regFor(candidate, unused.result("atlas.virtual_input_bf16", 1)));
  expectInvalid(unused, "unused edge destination clobbers live-through", candidate,
                {"register allocation overlap",
                 "edge destinations and live-through values", "tensor", "%"});

  // Entry and backedge sources may reuse destination registers. The backedge
  // exchanges these two registers and therefore requires a parallel-copy cycle.
  candidate = loop.assignments;
  setReg(candidate, loop.result("atlas.virtual_input_bf16", 1), 0);
  setReg(candidate, loop.result("atlas.virtual_input_bf16", 1, 1), 2);
  setReg(candidate, loop.block(1).getArgument(1), 0);
  setReg(candidate, loop.block(1).getArgument(2), 2);
  setReg(candidate, loop.block(2).getArgument(1), 0);
  expectValid(loop, "edge-copy swap and last-use source reuse", candidate);
  setReg(candidate, loop.block(1).getArgument(2), 0);
  expectInvalid(loop, "loop-carried values interfere", candidate,
                {"register allocation overlap", "tensor", "%"});
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect>();
  MLIRContext context(registry);
  testTensorAndCoverage(context);
  testScalars(context);
  testMixedTensorBanks(context);
  testCFG(context);
  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
