#include "Atlas/AtlasMXUAllocationVerification.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
using Assignments = SmallVector<VirtualMXUAssignment>;
constexpr StringLiteral state = "!atlas.virtual_state", fp8 = "!atlas.virtual_fp8", bf16 = "!atlas.virtual_bf16";
constexpr StringLiteral liveSlot = "overwrites a logically live slot";

std::string type(StringRef bank, unsigned unit) { return "!atlas.virtual_mxu_" + bank.str() + "<" + std::to_string(unit) + ">"; }

// Generates typed source only; placements are claimed by hand, never by the allocator or its summaries.
struct Source {
  std::string body = R"mlir(module { func.func @test() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %seed = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %scale = "atlas.virtual_scale_constant"() {code = 127 : i32} : () -> !atlas.virtual_scale
)mlir";
  unsigned next = 2;
  std::string current = "%s2";

  std::string effect(StringRef name, std::string args, std::string types, std::string resultType, std::string attributes = "") {
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
  void edge(StringRef block, StringRef argument) {
    body += "    cf.br ^" + block.str() + "(" + current + " : !atlas.virtual_state)\n  ^" + block.str() + "(" + argument.str() + ": !atlas.virtual_state):\n";
    current = argument.str();
  }
  std::string finish() const {
    return body + "    %out = \"atlas.virtual_output_bf16\"(" + current + ", %seed) {index = 0 : i32} : (" + state.str() + ", " + bf16.str() + ") -> " + state.str() + "\n    return %out : " + state.str() + "\n} }";
  }
};

struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<Value> handles;
  Fixture(MLIRContext &context, StringRef source) : module(parse(context, source, "fixture is typed SSA")) {
    if (!module)
      return;
    function = firstFunction(*module);
    function.walk([&](Operation *op) {
      for (Value result : op->getResults())
        if (isa<VirtualMXUWeightType, VirtualMXUAccType>(result.getType()))
          handles.push_back(result);
    });
  }
};

void expect(Fixture &fixture, StringRef name, const Assignments &assignments, bool valid, ArrayRef<StringRef> fragments = {},
            unsigned legacyWeight = 0, unsigned legacyAcc = 0) {
  FixedResourcePlacement fixed{};
  fixed.mxuWeightSlot = legacyWeight;
  fixed.mxuAccSlot = legacyAcc;
  expectVerified(name, valid, fragments, [&] { return verifyAtlasMXUAllocation(fixture.function, assignments, fixed); });
}

// Every handle claimed in `slots` order on `unit`.
Assignments claimed(Fixture &fixture, ArrayRef<unsigned> slots, ArrayRef<unsigned> units = {}) {
  Assignments result;
  for (auto [index, slot] : llvm::enumerate(slots))
    result.push_back({fixture.handles[index], {units.empty() ? 0 : units[index], slot}});
  return result;
}

void assignmentAndChainTests(MLIRContext &context, unsigned unit) {
  Source source;
  auto w0 = source.weight(unit), w1 = source.weight(unit);
  auto a0 = source.reset(w0, unit), a1 = source.seed(unit, true);
  source.readout(source.accumulate(w1, a1, unit), unit, true);
  source.readout(a0, unit);
  Fixture f(context, source.finish()), foreign(context, source.finish());
  if (!f.module || !foreign.module)
    return;
  const auto base = claimed(f, {1, 0, 0, 1, 1}, {unit, unit, unit, unit, unit});
  auto mutated = [&](StringRef name, function_ref<void(Assignments &)> mutate, ArrayRef<StringRef> fragments = {}) {
    auto changed = base;
    mutate(changed);
    expect(f, name, changed, fragments.empty(), fragments);
  };
  constexpr StringLiteral untracked = "foreign, stale, or untracked", unitMismatch = "unit must match";
  mutated("nonpreferred slots and independent weight/accumulator banks", [](Assignments &) {});
  mutated("assignment order is irrelevant", [](Assignments &a) { std::reverse(a.begin(), a.end()); });
  mutated("missing successor version", [](Assignments &a) { a.pop_back(); }, {"missing MXU assignment"});
  mutated("duplicate placement", [](Assignments &a) { a.push_back(a[0]); }, {"duplicate MXU assignment"});
  mutated("null handle", [](Assignments &a) { a[0].handle = Value{}; }, {untracked});
  mutated("foreign handle", [&](Assignments &a) { a[0].handle = foreign.handles[0]; }, {untracked});
  mutated("known state is not an MXU handle", [&](Assignments &a) { a[0].handle = f.function.getBody().front().front().getResult(0); },
          {"source handle result"});
  mutated("wrong unit", [&](Assignments &a) { a[0].placement.unit = 1 - unit; }, {unitMismatch});
  mutated("unit bound", [](Assignments &a) { a[0].placement.unit = 2; }, {unitMismatch});
  mutated("slot bound", [](Assignments &a) { a[0].placement.slot = 2; }, {"slot is outside"});
  mutated("second live weight aliases first", [](Assignments &a) { a[1].placement.slot = 1; }, {liveSlot, "weight slot 1", "current owner"});
  mutated("seed clobbers live reset accumulator", [](Assignments &a) { a[3].placement.slot = a[4].placement.slot = 0; },
          {liveSlot, "accumulator slot 0"});
  mutated("continuation cannot move to another slot", [](Assignments &a) { a[4].placement.slot = 0; }, {"continuation must retain"});
}

void reuseAndDeadTests(MLIRContext &context) {
  Source source;
  auto acc = source.reset(source.weight(0), 0);
  source.weight(0); // Dead load after the last use may reuse the weight slot.
  source.readout(acc, 0);
  source.readout(source.seed(0), 0, true);
  source.readout(source.seed(1, true), 1);
  if (Fixture f(context, source.finish()); f.module) {
    auto all = claimed(f, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 1});
    expect(f, "reuse after last weight use and readout, both units", all, true);
    all.erase(all.begin() + 2);
    expect(f, "dead weight still needs a placement", all, false, {"missing MXU assignment"});
  }
  Source dead;
  auto live = dead.weight(0);
  dead.weight(0); // No uses does not excuse overwriting the still-live weight.
  dead.readout(dead.reset(live, 0), 0);
  if (Fixture f(context, dead.finish()); f.module) {
    expect(f, "dead load cannot overwrite a live weight", claimed(f, {0, 0, 0}), false, {liveSlot, "weight slot 0"});
    expect(f, "dead load may use the other weight slot", claimed(f, {0, 1, 0}), true);
  }
  Source shared;
  auto retained = shared.weight(0);
  auto first = shared.reset(retained, 0);
  shared.weight(0);
  shared.readout(shared.accumulate(retained, first, 0), 0);
  if (Fixture f(context, shared.finish()); f.module) {
    expect(f, "first consumer does not release a weight with future uses", claimed(f, {0, 1, 0, 1}), false, {liveSlot});
    expect(f, "shared weight remains usable through its last consumer", claimed(f, {0, 1, 1, 1}), true);
  }
  Source both;
  auto a0 = both.seed(0), a1 = both.seed(1);
  both.readout(a1, 1);
  both.readout(a0, 0);
  if (Fixture f(context, both.finish()); f.module)
    expect(f, "same slot index is independent across units", claimed(f, {1, 1}, {0, 1}), true);
}

void legacyTests(MLIRContext &context) {
  Source source;
  auto w = source.weight(0);
  auto acc = source.seed(0);
  source.legacy(0);
  source.readout(source.accumulate(w, acc, 0), 0);
  if (Fixture f(context, source.finish()); f.module) {
    auto all = claimed(f, {1, 1, 1});
    // Virtual admission is stricter (no live handle in the unit during legacy matmul); this checks only the fixed slots.
    expect(f, "legacy fixed zero slots do not touch live slot one", all, true);
    expect(f, "legacy fixed weight collision", all, false, {"legacy MXU matmul overwrites", "weight slot 1"}, 1, 0);
    expect(f, "legacy fixed accumulator collision", all, false, {"legacy MXU matmul overwrites", "accumulator slot 1"}, 0, 1);
    expect(f, "legacy weight bound", all, false, {"legacy MXU fixed unit/slots"}, 2, 0);
    expect(f, "legacy accumulator bound", all, false, {"legacy MXU fixed unit/slots"}, 0, 2);
  }
  Source free;
  free.legacy(1);
  if (Fixture f(context, free.finish()); f.module)
    expect(f, "legacy needs no explicit handle assignments", {}, true, {}, 1, 1);
}

void lifetimeAndCFGTests(MLIRContext &context) {
  Source source;
  auto w = source.weight(0), acc = source.seed(0);
  auto next = source.accumulate(w, acc, 0);
  std::string readoutState = source.current;
  source.readout(next, 0);
  if (Fixture f(context, replace(source.finish(), "(" + readoutState + ", " + next + ")", "(" + readoutState + ", " + acc + ")")); f.module)
    expect(f, "stale accumulator version readout", claimed(f, {0, 0, 0}), false, {"readout requires the current accumulator version"});
  Source unclosed;
  unclosed.seed(0);
  if (Fixture f(context, unclosed.finish()); f.module)
    expect(f, "accumulator must be read out before block exit", claimed(f, {0}), false, {"ownership remains live at block exit"});
  Source loop;
  loop.edge("loop", "%edge");
  loop.readout(loop.reset(loop.weight(0), 0), 0);
  loop.body += "    %again = arith.constant false\n    cf.cond_br %again, ^loop(" + loop.current + " : !atlas.virtual_state), ^exit(" + loop.current + " : !atlas.virtual_state)\n  ^exit(%done: !atlas.virtual_state):\n";
  loop.current = "%done";
  if (Fixture f(context, loop.finish()); f.module)
    expect(f, "loop-local claims complete before every backedge", claimed(f, {1, 1}), true);
  Source crossing;
  auto captured = crossing.weight(0);
  crossing.edge("next", "%edge");
  crossing.readout(crossing.reset(captured, 0), 0);
  if (Fixture f(context, crossing.finish()); f.module)
    expect(f, "weight capture across blocks is unsupported", claimed(f, {0, 0}), false, {"cannot cross CFG blocks"});
}
} // namespace

void atlas_test::runMXUAllocation(MLIRContext &context) {
  assignmentAndChainTests(context, 0);
  assignmentAndChainTests(context, 1);
  reuseAndDeadTests(context);
  legacyTests(context);
  lifetimeAndCFGTests(context);
}
