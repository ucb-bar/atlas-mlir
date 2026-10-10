#include "VerificationTestSupport.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;
using namespace mlir::atlas::timing;

namespace {

struct OperationSpec { StringRef name, fields; };
const OperationSpec a = {"alu_imm", "kind = \"addi\", dst = 1 : i32, src = 0 : i32, immediate = 0 : i32"};
const OperationSpec b = {"alu_imm", "kind = \"addi\", dst = 2 : i32, src = 0 : i32, immediate = 0 : i32"};
const OperationSpec nop = {"alu_imm", "kind = \"addi\", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32"};
const OperationSpec delay = {"delay", "cycles = 7 : i32"};
const OperationSpec dma = {"dma", "direction = \"load\", channel = 0 : i32, reg = 0 : i32, dram = 0 : i32, size = 0 : i32"};
const OperationSpec wait = {"dma_wait", "channel = 0 : i32"};
const OperationSpec halt = {"trap", "kind = \"ecall\""};

bool accepts(MLIRContext &context, ArrayRef<OperationSpec> specs,
             const TimingProvider &provider) {
  std::string source = "module { %s0 = \"atlas.start\"() : () -> !atlas.state\n";
  for (auto [index, op] : llvm::enumerate(specs)) {
    source += "%s" + std::to_string(index + 1) + " = \"atlas." + op.name.str() +
              "\"(%s" + std::to_string(index) + ") {" + op.fields.str() +
              "} : (!atlas.state) -> !atlas.state\n";
  }
  source += "}";
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!module)
    return false;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  return succeeded(stream) && succeeded(verifyAtlasTimedStream(*stream, provider));
}

// Deliberately synthetic policies: these tests establish policy isolation, not
// a claim about any hardware. Capacity and WAIT behavior live only in this state.
class ToyReservations final : public TimingReservations {
public:
  ToyReservations(int capacity, bool extend, bool missing = false)
      : capacity(capacity), extend(extend), missing(missing) {}
  TimingRuleResult<std::string>
  conflict(const Instr &, const Footprint &f, int cycle) const override {
    if (missing) return {{}, "toy reservation resource has no coverage"};
    if (f.holds.empty()) return {};
    unsigned live = std::count_if(until.begin(), until.end(),
                                 [&](int end) { return end > cycle; });
    return {live >= unsigned(capacity) ? "toy capacity occupied" : "", {}};
  }
  std::string reserve(const Instr &, const Footprint &f, int cycle) override {
    for (const Hold &h : f.holds)
      until.push_back(cycle + h.to + 1);
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    if (extend)
      until.push_back(cycle + 3);
    return {};
  }
private:
  int capacity;
  bool extend, missing;
  std::vector<int> until;
};

TimingProvider toy(int capacity = 1, bool extend = false) {
  TimingProvider p;
  p.id = "synthetic-test-policy";
  p.validateScope = [](const AtlasStream &) { return std::string{}; };
  p.footprint = [](const Instr &, const RegValues &) { return Footprint{}; };
  p.dependence = [](const Instr &, const Footprint &, const Instr &, const Footprint &) {
    return TimingRuleResult<Dependence>{};
  };
  p.dmaConflict = [](const Footprint &, const Footprint &) {
    return TimingRuleResult<DMACompletionConflict>{};
  };
  p.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  p.createReservations = [=] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<ToyReservations>(capacity, extend);
    return result;
  };
  return p;
}

Footprint occupied(const Instr &in, const RegValues &) {
  Footprint f;
  if (in.op->opClass == OpClass::Alu && in.rd != 0) {
    f.holds.push_back({Unit::VloadPath, 0, 0, 2});
    f.doneAge = 2;
  }
  return f;
}

void testPolicies(MLIRContext &context) {
  const OperationSpec shortStream[] = {a, b, halt};
  auto p = toy();
  check(accepts(context, shortStream, p), "explicit no-dependence policy accepts adjacent instructions");
  p.dependence = [](const Instr &first, const Footprint &, const Instr &second, const Footprint &) {
    Dependence d;
    if (first.rd == 1 && second.rd == 2) {
      d.distance = 7;
      d.reason = "toy dependence";
    }
    return TimingRuleResult<Dependence>{d, {}};
  };
  check(!accepts(context, shortStream, p), "supplied pair distance rejects identical footprints");
  const OperationSpec spaced[] = {a, delay, nop, b, halt};
  check(accepts(context, spaced, p), "supplied pair distance accepts sufficient spacing");
  p.issueGap = [](const Instr &) { return TimingRuleResult<int>{1, {}}; };
  check(!accepts(context, spaced, p), "supplied issue gap controls explicit delays");

  const OperationSpec resources[] = {a, b, nop, nop, nop, halt};
  p = toy(1);
  p.footprint = occupied;
  check(!accepts(context, resources, p), "capacity one rejects overlapping holds");
  p = toy(2);
  p.footprint = occupied;
  check(accepts(context, resources, p), "capacity two accepts identical overlapping holds");
  auto model = npuModelTimingProvider();
  model.footprint = occupied;
  check(!accepts(context, resources, model), "model capacity does not leak into custom capacity two");
  p = toy(0);
  p.footprint = occupied;
  const OperationSpec isolated[] = {a, nop, nop, nop, halt};
  check(!accepts(context, isolated, p), "custom reservation can reject model-admissible hold");
  check(accepts(context, isolated, model), "model adapter preserves its existing reservation rule");

  const OperationSpec waits[] = {dma, wait, b, nop, nop, nop, halt};
  p = toy(1, false);
  p.footprint = occupied;
  check(accepts(context, waits, p), "provider may release reservations at WAIT");
  p = toy(1, true);
  p.footprint = occupied;
  check(!accepts(context, waits, p), "provider WAIT extension changes acceptance");
  const OperationSpec pending[] = {dma, b, wait, halt};
  p = toy();
  check(accepts(context, pending, p), "covered DMA disjointness policy accepts nonconflicting work");
  p.dmaConflict = [](const Footprint &, const Footprint &) {
    return TimingRuleResult<DMACompletionConflict>{{true, EdgeKind::RAW}, {}};
  };
  check(!accepts(context, pending, p), "supplied DMA conflict rule changes acceptance");

  const OperationSpec publish = {"csr", "kind = \"rrw\", address = 3088 : i32, dst = 0 : i32, source = 0 : i32"};
  const OperationSpec premature[] = {a, publish, nop, nop, halt};
  const OperationSpec complete[] = {a, nop, nop, publish, halt};
  p = toy(2);
  p.footprint = occupied;
  check(!accepts(context, premature, p), "publication waits for supplied completion even without a pair dependence");
  check(accepts(context, complete, p), "publication accepts completed work under supplied policy");

  unsigned factories = 0;
  p = toy();
  p.createReservations = [&] {
    ++factories;
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<ToyReservations>(1, false);
    return result;
  };
  const OperationSpec blocks[] = {
      {"jump", "kind = \"jal\", dst = 0 : i32, base = 0 : i32, offset = 4 : i32"}, nop, a, halt};
  check(accepts(context, blocks, p) && factories == 2,
        "each emitted block receives a fresh provider reservation state");
}

void testMissingCoverage(MLIRContext &context) {
  const OperationSpec stream[] = {a, b, halt};
  auto p = toy();
  for (unsigned field = 0; field < 7; ++field) {
    p = toy();
    switch (field) {
    case 0: p.id.clear(); break;
    case 1: p.validateScope = {}; break;
    case 2: p.footprint = {}; break;
    case 3: p.dependence = {}; break;
    case 4: p.dmaConflict = {}; break;
    case 5: p.issueGap = {}; break;
    case 6: p.createReservations = {}; break;
    }
    check(!accepts(context, stream, p), "missing provider field fails without fallback");
  }
  p = toy();
  p.createReservations = [] { return TimingRuleResult<std::unique_ptr<TimingReservations>>{}; };
  check(!accepts(context, stream, p), "null reservation state fails closed");
  p = toy();
  p.createReservations = [] {
    return TimingRuleResult<std::unique_ptr<TimingReservations>>{{}, "uncovered reservation policy"};
  };
  check(!accepts(context, stream, p), "factory coverage error fails closed");
  p = toy();
  p.dependence = [](const Instr &, const Footprint &, const Instr &, const Footprint &) {
    return TimingRuleResult<Dependence>{{}, "uncovered toy instruction pair"};
  };
  check(!accepts(context, stream, p), "unknown pair is distinct from zero spacing");
  p = toy();
  p.createReservations = [] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<ToyReservations>(1, false, true);
    return result;
  };
  check(!accepts(context, stream, p), "unknown reservation resource is distinct from no conflict");
  p = toy();
  p.issueGap = [](const Instr &) { return TimingRuleResult<int>{0, {}}; };
  check(!accepts(context, stream, p), "invalid issue gap fails closed");
  p = toy();
  p.issueGap = [](const Instr &) { return TimingRuleResult<int>{0, "uncovered issue gap"}; };
  check(!accepts(context, stream, p), "missing issue gap coverage fails closed");
  p = toy();
  p.footprint = [](const Instr &, const RegValues &) {
    Footprint f;
    f.error = "unknown toy operand instance";
    return f;
  };
  check(!accepts(context, stream, p), "unknown footprint instance fails closed");
  p = toy();
  unsigned footprintCalls = 0;
  p.validateScope = [](const AtlasStream &) { return "outside toy VLS scope"; };
  p.footprint = [&](const Instr &, const RegValues &) { ++footprintCalls; return Footprint{}; };
  check(!accepts(context, stream, p) && footprintCalls == 0,
        "program scope is checked before timing rules");
  check(!lookupTimingProvider("unknown-policy").error.empty(), "unknown selection does not fall back");
  check(!lookupTimingProvider("atlas.vls.conservative.v1").error.empty(),
        "footprint-only CIRCT evidence does not borrow model policy");
}

void testBoundedScope(MLIRContext &context) {
  auto p = toy();
  p.id = "synthetic-vls-subset";
  p.validateScope = [](const AtlasStream &s) {
    for (const Instr &in : s.instrs) {
      const std::string &name = in.op->name;
      if (name != "addi" && name != "lui" && name != "vload" &&
          name != "vstore" && name != "delay" && name != "csrrw" && name != "ecall")
        return std::string("operation outside synthetic VLS subset");
    }
    return std::string{};
  };
  p.footprint = [](const Instr &in, const RegValues &regs) {
    Footprint f;
    if ((in.op->opClass == OpClass::VLoad || in.op->opClass == OpClass::VStore) && !regs[in.rs1])
      f.error = "unknown base outside synthetic VLS subset";
    return f;
  };
  const OperationSpec vector = {"vload", "dst = 0 : i32, base = 1 : i32, offset = 0 : i32, format = \"raw\""};
  const OperationSpec known[] = {a, vector, halt};
  const OperationSpec unknown[] = {vector, halt};
  check(accepts(context, known, p), "covered straight-line VLS subset accepts");
  check(!accepts(context, unknown, p), "VLS subset rejects unknown bases");
  const OperationSpec transfer[] = {dma, wait, halt};
  check(!accepts(context, transfer, p), "VLS subset rejects DMA without model fallback");
  const OperationSpec mxu[] = {{"mxu_push", "kind = \"weight_fp8\", unit = 0 : i32, src = 0 : i32, slot = 0 : i32"}, halt};
  check(!accepts(context, mxu, p), "VLS subset rejects MXU without model fallback");
  const OperationSpec vpu[] = {{"vpu_unary", "kind = \"mov\", dst = 0 : i32, src = 0 : i32"}, halt};
  check(!accepts(context, vpu, p), "VLS subset rejects VPU without model fallback");
  const OperationSpec branch[] = {{"branch", "kind = \"beq\", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 4 : i32"}, nop, halt};
  check(!accepts(context, branch, p), "VLS subset rejects CFG without model fallback");
}

void testRetainedProvider(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {
      atlas.timing_state = "timed",
      atlas.timing_provider = "synthetic-test-policy"
    } {
      %s0 = "atlas.start"() : () -> !atlas.state
      %s1 = "atlas.trap"(%s0) {kind = "ecall"} : (!atlas.state) -> !atlas.state
    }
  )mlir", &context);
  check(bool(module), "retained custom policy module parses");
  if (!module)
    return;
  auto p = toy();
  check(succeeded(verifyAtlasTiming(*module, p)),
        "retained custom policy accepts complete supplied rules");
  auto mismatch = p;
  mismatch.id = "different-synthetic-policy";
  check(failed(verifyAtlasTiming(*module, mismatch)),
        "retained custom policy rejects supplied identity mismatch");
  check(failed(verifyAtlasTiming(*module)),
        "retained custom policy requires supplied rules outside the registry");
  p.dependence = {};
  check(failed(verifyAtlasTiming(*module, p)),
        "retained custom policy rejects incomplete supplied coverage");
}
const char *kStrict = "synthetic-strict-spacing-policy";

// The model's rules with every positive pair distance lengthened by four.
TimingRuleResult<TimingProvider> strictProvider(ModuleOp) {
  auto p = npuModelTimingProvider();
  p.id = kStrict;
  auto model = p.dependence;
  p.dependence = [model](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    auto rule = model(a, fa, b, fb);
    if (rule.value.distance > 0)
      rule.value.distance += 4;
    return rule;
  };
  return {p, {}};
}

const char *kCopy = R"mlir(
  module {
    %s0 = "atlas.start"() : () -> !atlas.state
    %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
    %s2 = "atlas.vload"(%s1) {dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
    %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
    %s4 = "atlas.vstore"(%s3) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
    %s5 = "atlas.trap"(%s4) {kind = "ecall"} : (!atlas.state) -> !atlas.state
  }
)mlir";

OwningOpRef<ModuleOp> rewrite(MLIRContext &context, StringRef source, StringRef pipeline) {
  auto module = parseSourceString<ModuleOp>(source, &context);
  PassManager pm = PassManager::on<ModuleOp>(&context);
  if (!module || failed(parsePassPipeline(pipeline, pm)) || failed(pm.run(*module)))
    return {};
  return module;
}

int64_t delayCycles(ModuleOp module) {
  int64_t total = 0;
  module.walk([&](DelayOp delay) { total += delay.getCycles() + 1; });
  return total;
}

std::string print(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

StringRef retainedProvider(ModuleOp module) {
  auto id = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
  return id ? id.getValue() : StringRef();
}

void testRegistry(MLIRContext &context) {
  auto model = lookupTimingProvider(kNpuModelTimingProviderId.str());
  check(model.error.empty() && model.value.id == kNpuModelTimingProviderId, "registry is seeded with the npu-model provider");
  auto unknown = lookupTimingProvider("unknown-policy");
  check(StringRef(unknown.error).contains("unknown Atlas timing provider"), "unknown id fails explicitly", unknown.error);
  auto footprintOnly = lookupTimingProvider("atlas.vls.conservative.v1");
  check(StringRef(footprintOnly.error).contains("footprint-only"), "unregistered footprint-only id keeps its rejection", footprintOnly.error);
  auto complete = [](ModuleOp) { return TimingRuleResult<TimingProvider>{npuModelTimingProvider(), {}}; };
  check(!registerAtlasTimingProvider(kNpuModelTimingProviderId.str(), complete).empty(), "duplicate registration fails");
  check(!registerAtlasTimingProvider("", complete).empty(), "empty registration id fails");
  check(!registerAtlasTimingProvider("synthetic-null-factory", {}).empty(), "null factory fails");
  check(registerAtlasTimingProvider("synthetic-renamed-policy", complete).empty() &&
            !lookupTimingProvider("synthetic-renamed-policy").error.empty(),
        "factory must return the registered identity");
  check(registerAtlasTimingProvider("synthetic-incomplete-policy", [](ModuleOp) {
          auto p = npuModelTimingProvider();
          p.id = "synthetic-incomplete-policy";
          p.issueGap = {};
          return TimingRuleResult<TimingProvider>{p, {}};
        }).empty() && !lookupTimingProvider("synthetic-incomplete-policy").error.empty(),
        "registered incomplete policy is not returned");
  check(registerAtlasTimingProvider(kStrict, strictProvider).empty(), "synthetic strict policy registers");
  check(lookupTimingProvider(kStrict).error.empty(), "registered id is found");

  registerScheduleAtlasStreamPass();
  registerVerifyAtlasTimingPass();
  auto byModel = rewrite(context, kCopy, "insert-atlas-delays");
  auto byDefaultName = rewrite(context, kCopy, std::string("insert-atlas-delays{provider=") + kNpuModelTimingProviderId.str() + "}");
  auto byStrict = rewrite(context, kCopy, std::string("insert-atlas-delays{provider=") + kStrict + "}");
  check(byModel && byDefaultName && byStrict, "delay insertion accepts each registered provider", diagnostics);
  if (!byModel || !byDefaultName || !byStrict)
    return;
  check(print(*byModel) == print(*byDefaultName), "naming the default provider changes nothing");
  check(retainedProvider(*byModel) == kNpuModelTimingProviderId && retainedProvider(*byStrict) == kStrict,
        "delay insertion stamps the provider it used");
  check(delayCycles(*byStrict) > delayCycles(*byModel), "stricter selected spacing emits longer delays");
  check(succeeded(verifyAtlasTiming(*byStrict)), "registry dispatch verifies the strict result", diagnostics);
  auto strict = strictProvider({}).value;
  auto modelStream = readAtlasStream(*byModel, AtlasStreamReadMode::Verification);
  diagnostics.clear();
  check(succeeded(modelStream) && failed(verifyAtlasTimedStream(*modelStream, strict)) && diagnosed("insufficient issue spacing"),
        "model spacing fails the strict provider's verification", diagnostics);

  diagnostics.clear();
  check(failed(verifyAtlasTiming(*byStrict, npuModelTimingProvider())) && diagnosed("disagrees with retained provider identity"),
        "module stamped with strict rejects the model", diagnostics);
  for (StringRef pass : {"verify-atlas-timing", "insert-atlas-delays", "schedule-atlas-stream"}) {
    diagnostics.clear();
    std::string pipeline = pass.str() + "{provider=" + kStrict + "}";
    check(!rewrite(context, print(*byModel), pipeline) && diagnosed("disagrees with retained provider identity"),
          "module stamped with the model rejects a different selection", pipeline + "\n" + diagnostics);
  }
  diagnostics.clear();
  check(!rewrite(context, kCopy, "insert-atlas-delays{provider=unknown-policy}") && diagnosed("unknown Atlas timing provider"),
        "unknown selection fails at the producer", diagnostics);

  // The list scheduler spaces work with the model graph, so its timed output
  // must pass the selected provider; reordering then inserting delays does.
  diagnostics.clear();
  check(!rewrite(context, kCopy, std::string("schedule-atlas-stream{provider=") + kStrict + "}") && diagnosed("insufficient issue spacing"),
        "scheduler output is checked under a non-model provider", diagnostics);
  auto reordered = rewrite(context, kCopy, std::string("schedule-atlas-stream{insert-delays=false provider=") + kStrict + "},insert-atlas-delays{provider=" + kStrict + "},verify-atlas-timing");
  check(reordered && retainedProvider(*reordered) == kStrict, "reorder then strict delay insertion verifies", diagnostics);

  // A module-scoped policy (as RTL evidence selection would supply) replaces a
  // footprint-only rejection once registered.
  check(registerAtlasTimingProvider("atlas.vls.conservative.v1", [](ModuleOp module) -> TimingRuleResult<TimingProvider> {
          if (!module || !module->hasAttr("atlas.rtl_evidence"))
            return {{}, "synthetic evidence policy requires selected evidence"};
          auto p = npuModelTimingProvider();
          p.id = "atlas.vls.conservative.v1";
          return {p, {}};
        }).empty(), "a footprint-only id can be registered by a complete policy");
  auto withoutEvidence = lookupTimingProvider("atlas.vls.conservative.v1");
  check(StringRef(withoutEvidence.error).contains("requires selected evidence"), "module-scoped factory sees no module", withoutEvidence.error);
  (*byModel)->setAttr("atlas.rtl_evidence", UnitAttr::get(&context));
  check(lookupTimingProvider("atlas.vls.conservative.v1", *byModel).error.empty(), "module-scoped factory reads its module");
}
} // namespace

void atlas_test::runTimingProvider(MLIRContext &context) {
  testPolicies(context);
  testMissingCoverage(context);
  testBoundedScope(context);
  testRetainedProvider(context);
  testRegistry(context);
}
