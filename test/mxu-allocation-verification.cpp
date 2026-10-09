#include "VerificationTestSupport.h"
#include "Atlas/AtlasMXUAllocationVerification.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
using Assignments = SmallVector<VirtualMXUAssignment>;
constexpr StringLiteral state = "!atlas.virtual_state";
constexpr StringLiteral fp8 = "!atlas.virtual_fp8";
constexpr StringLiteral bf16 = "!atlas.virtual_bf16";


std::string type(StringRef bank, unsigned unit) {
  return "!atlas.virtual_mxu_" + bank.str() + "<" + std::to_string(unit) + ">";
}

// Generates typed source only. Placements below are manually claimed; neither
// the allocator nor its interference/lifetime summaries supply expected maps.
struct Source {
  std::string body = R"mlir(module { func.func @test() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %seed = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %scale = "atlas.virtual_scale_constant"() {code = 127 : i32} : () -> !atlas.virtual_scale
)mlir";
  unsigned next = 2;
  std::string current = "%s2";

  std::string effect(StringRef name, std::string args, std::string types,
                     std::string resultType, std::string attributes = "") {
    unsigned index = ++next;
    std::string result = "%h" + std::to_string(index);
    std::string after = "%s" + std::to_string(index);
    body += "    " + after + ", " + result + " = \"atlas.virtual_" + name.str() + "\"(" + current + ", " + args + ") " + attributes;
    body += " : (" + state.str() + ", " + types + ") -> (" + state.str() + ", " + resultType + ")\n";
    current = after;
    return result;
  }
  std::string weight(unsigned unit) {
    return effect("mxu_load_weight", "%x", fp8.str(), type("weight", unit), "{unit = " + std::to_string(unit) + " : i32}");
  }
  std::string seed(unsigned unit, bool narrow = false) {
    return effect(narrow ? "mxu_load_acc_fp8" : "mxu_load_acc_bf16", narrow ? "%x" : "%seed", narrow ? fp8.str() : bf16.str(), type("acc", unit), "{unit = " + std::to_string(unit) + " : i32}");
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
    body += "    %legacy" + std::to_string(++next) + " = \"atlas.virtual_mxu_matmul\"(%x, %x) {unit = " + std::to_string(unit) + " : i32} : (" + fp8.str() + ", " + fp8.str() + ") -> " + bf16.str() + "\n";
  }
  std::string finish() const {
    return body + "    %out = \"atlas.virtual_output_bf16\"(" + current + ", %seed) {index = 0 : i32} : (" + state.str() + ", " + bf16.str() + ") -> " + state.str() + "\n    return %out : " + state.str() + "\n} }";
  }
};

struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<Value> handles;
  bool initialize(MLIRContext &context, StringRef source) {
    module = parseSourceString<ModuleOp>(source, &context);
    check(bool(module), "fixture parses");
    if (!module)
      return false;
    bool valid = succeeded(verify(*module));
    check(valid, "fixture is typed SSA");
    if (!valid)
      return false;
    function = *module->getOps<func::FuncOp>().begin();
    function.walk([&](Operation *op) {
      for (Value result : op->getResults())
        if (isa<VirtualMXUWeightType, VirtualMXUAccType>(result.getType()))
          handles.push_back(result);
    });
    return true;
  }
};

void expect(Fixture &fixture, StringRef name, const Assignments &assignments,
            bool valid, ArrayRef<StringRef> fragments = {}, unsigned legacyWeight = 0,
            unsigned legacyAcc = 0) {
  FixedResourcePlacement fixed{};
  fixed.mxuWeightSlot = legacyWeight;
  fixed.mxuAccSlot = legacyAcc;
  expectVerified(name, valid, fragments, [&] { return verifyAtlasMXUAllocation(fixture.function, assignments, fixed); });
}

void assignmentAndChainTests(MLIRContext &context, unsigned unit) {
  Source source;
  auto w0 = source.weight(unit), w1 = source.weight(unit);
  auto a0 = source.reset(w0, unit), a1 = source.seed(unit, true);
  auto next = source.accumulate(w1, a1, unit);
  source.readout(next, unit, true);
  source.readout(a0, unit);
  Fixture fixture, foreign;
  if (!fixture.initialize(context, source.finish()) || !foreign.initialize(context, source.finish()))
    return;
  const auto &h = fixture.handles;
  Assignments base = {{h[0], {unit, 1}}, {h[1], {unit, 0}}, {h[2], {unit, 0}}, {h[3], {unit, 1}}, {h[4], {unit, 1}}};
  expect(fixture, "nonpreferred slots and independent weight/accumulator banks", base, true);
  auto changed = base;
  std::reverse(changed.begin(), changed.end());
  expect(fixture, "assignment order is irrelevant", changed, true);
  changed = base;
  changed.pop_back();
  expect(fixture, "missing successor version", changed, false, {"missing MXU assignment"});
  changed = base;
  changed.push_back(base[0]);
  expect(fixture, "duplicate placement", changed, false, {"duplicate MXU assignment"});
  changed = base;
  changed[0].handle = Value{};
  expect(fixture, "null handle", changed, false, {"foreign, stale, or untracked"});
  changed[0].handle = foreign.handles[0];
  expect(fixture, "foreign handle", changed, false, {"foreign, stale, or untracked"});
  changed[0].handle = fixture.function.getBody().front().front().getResult(0);
  expect(fixture, "known state is not an MXU handle", changed, false, {"source handle result"});
  changed = base;
  changed[0].placement.unit = 1 - unit;
  expect(fixture, "wrong unit", changed, false, {"unit must match"});
  changed[0].placement.unit = 2;
  expect(fixture, "unit bound", changed, false, {"unit must match"});
  changed = base;
  changed[0].placement.slot = 2;
  expect(fixture, "slot bound", changed, false, {"slot is outside"});
  changed = base;
  changed[1].placement.slot = 1;
  expect(fixture, "second live weight aliases first", changed, false, {"overwrites a logically live slot", "weight slot 1", "current owner"});
  changed = base;
  changed[3].placement.slot = changed[4].placement.slot = 0;
  expect(fixture, "seed clobbers live reset accumulator", changed, false, {"overwrites a logically live slot", "accumulator slot 0"});
  changed = base;
  changed[4].placement.slot = 0;
  expect(fixture, "continuation cannot move to another slot", changed, false, {"continuation must retain"});
}

void reuseAndDeadTests(MLIRContext &context) {
  Source source;
  auto w = source.weight(0);
  auto acc = source.reset(w, 0);
  source.weight(0); // Dead load after the last use may reuse the weight slot.
  source.readout(acc, 0);
  auto seed = source.seed(0);
  source.readout(seed, 0, true);
  auto other = source.seed(1, true);
  source.readout(other, 1);
  Fixture fixture;
  if (fixture.initialize(context, source.finish())) {
    const auto &h = fixture.handles;
    Assignments claimed = {{h[0], {0, 1}}, {h[1], {0, 1}}, {h[2], {0, 1}}, {h[3], {0, 1}}, {h[4], {1, 1}}};
    expect(fixture, "reuse after last weight use and readout, both units", claimed, true);
    auto missing = claimed;
    missing.erase(missing.begin() + 2);
    expect(fixture, "dead weight still needs a placement", missing, false, {"missing MXU assignment"});
  }
  Source dead;
  auto live = dead.weight(0);
  dead.weight(0); // No uses does not excuse overwriting the still-live weight.
  dead.readout(dead.reset(live, 0), 0);
  Fixture overwrite;
  if (overwrite.initialize(context, dead.finish())) {
    const auto &h = overwrite.handles;
    expect(overwrite, "dead load cannot overwrite a live weight", {{h[0], {0, 0}}, {h[1], {0, 0}}, {h[2], {0, 0}}}, false, {"overwrites a logically live slot", "weight slot 0"});
    expect(overwrite, "dead load may use the other weight slot", {{h[0], {0, 0}}, {h[1], {0, 1}}, {h[2], {0, 0}}}, true);
  }
  Source shared;
  auto retained = shared.weight(0);
  auto first = shared.reset(retained, 0);
  shared.weight(0);
  shared.readout(shared.accumulate(retained, first, 0), 0);
  Fixture futureUse;
  if (futureUse.initialize(context, shared.finish())) {
    const auto &h = futureUse.handles;
    Assignments claimed = {{h[0], {0, 0}}, {h[1], {0, 1}}, {h[2], {0, 0}}, {h[3], {0, 1}}};
    expect(futureUse, "first consumer does not release a weight with future uses", claimed, false, {"overwrites a logically live slot"});
    claimed[2].placement.slot = 1;
    expect(futureUse, "shared weight remains usable through its last consumer", claimed, true);
  }
  Source both;
  auto a0 = both.seed(0), a1 = both.seed(1);
  both.readout(a1, 1);
  both.readout(a0, 0);
  Fixture units;
  if (units.initialize(context, both.finish()))
    expect(units, "same slot index is independent across units", {{units.handles[0], {0, 1}}, {units.handles[1], {1, 1}}}, true);
}

void legacyTests(MLIRContext &context) {
  Source source;
  auto w = source.weight(0);
  auto acc = source.seed(0);
  source.legacy(0);
  source.readout(source.accumulate(w, acc, 0), 0);
  Fixture fixture;
  if (!fixture.initialize(context, source.finish()))
    return;
  const auto &h = fixture.handles;
  Assignments claimed = {{h[0], {0, 1}}, {h[1], {0, 1}}, {h[2], {0, 1}}};
  // The pinned virtual admission is more conservative than this footprint:
  // it disallows any live handle in the unit during legacy matmul. Here we
  // directly test the actual fixed slots used by the lowering, not admission.
  expect(fixture, "legacy fixed zero slots do not touch live slot one", claimed, true);
  expect(fixture, "legacy fixed weight collision", claimed, false, {"legacy MXU matmul overwrites", "weight slot 1"}, 1, 0);
  expect(fixture, "legacy fixed accumulator collision", claimed, false, {"legacy MXU matmul overwrites", "accumulator slot 1"}, 0, 1);
  expect(fixture, "legacy weight bound", claimed, false, {"legacy MXU fixed unit/slots"}, 2, 0);
  expect(fixture, "legacy accumulator bound", claimed, false, {"legacy MXU fixed unit/slots"}, 0, 2);
  Source free;
  free.legacy(1);
  Fixture empty;
  if (empty.initialize(context, free.finish()))
    expect(empty, "legacy needs no explicit handle assignments", {}, true, {}, 1, 1);
}

std::string replace(std::string source, StringRef before, StringRef after) {
  size_t position = source.find(before.str());
  check(position != std::string::npos, "fixture source mutation exists");
  if (position != std::string::npos)
    source.replace(position, before.size(), after.str());
  return source;
}

void lifetimeAndCFGTests(MLIRContext &context) {
  Source source;
  auto w = source.weight(0), acc = source.seed(0);
  auto next = source.accumulate(w, acc, 0);
  std::string readoutState = source.current;
  source.readout(next, 0);
  Fixture stale;
  std::string text = replace(source.finish(), "(" + readoutState + ", " + next + ")", "(" + readoutState + ", " + acc + ")");
  if (stale.initialize(context, text))
    expect(stale, "stale accumulator version readout", {{stale.handles[0], {0, 0}}, {stale.handles[1], {0, 0}}, {stale.handles[2], {0, 0}}}, false, {"readout requires the current accumulator version"});
  Source unclosed;
  unclosed.seed(0);
  Fixture missing;
  if (missing.initialize(context, unclosed.finish()))
    expect(missing, "accumulator must be read out before block exit", {{missing.handles[0], {0, 0}}}, false, {"ownership remains live at block exit"});
  Source loop;
  loop.body += "    cf.br ^loop(%s2 : !atlas.virtual_state)\n  ^loop(%edge: !atlas.virtual_state):\n";
  loop.current = "%edge";
  auto loopWeight = loop.weight(0);
  loop.readout(loop.reset(loopWeight, 0), 0);
  loop.body += "    %again = arith.constant false\n    cf.cond_br %again, ^loop(" + loop.current + " : !atlas.virtual_state), ^exit(" + loop.current + " : !atlas.virtual_state)\n  ^exit(%done: !atlas.virtual_state):\n";
  loop.current = "%done";
  Fixture complete;
  if (complete.initialize(context, loop.finish()))
    expect(complete, "loop-local claims complete before every backedge", {{complete.handles[0], {0, 1}}, {complete.handles[1], {0, 1}}}, true);
  Source crossing;
  auto captured = crossing.weight(0);
  crossing.body += "    cf.br ^next(" + crossing.current + " : !atlas.virtual_state)\n  ^next(%edge: !atlas.virtual_state):\n";
  crossing.current = "%edge";
  crossing.readout(crossing.reset(captured, 0), 0);
  Fixture foreignBlock;
  if (foreignBlock.initialize(context, crossing.finish()))
    expect(foreignBlock, "weight capture across blocks is unsupported", {{foreignBlock.handles[0], {0, 0}}, {foreignBlock.handles[1], {0, 0}}}, false, {"cannot cross CFG blocks"});
}
} // namespace

void atlas_test::runMXUAllocation(MLIRContext &context) {
  assignmentAndChainTests(context, 0);
  assignmentAndChainTests(context, 1);
  reuseAndDeadTests(context);
  legacyTests(context);
  lifetimeAndCFGTests(context);
}
