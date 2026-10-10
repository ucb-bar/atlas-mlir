#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasStream.h"
#include "VerificationTestSupport.h"
#include "llvm/Support/MemoryBuffer.h"
#include <functional>

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

bool accepts(MLIRContext &context, ArrayRef<OperationSpec> specs, const TimingProvider &provider) {
  std::string source = "module { %s0 = \"atlas.start\"() : () -> !atlas.state\n";
  for (auto [index, op] : llvm::enumerate(specs))
    source += "%s" + std::to_string(index + 1) + " = \"atlas." + op.name.str() + "\"(%s" + std::to_string(index) + ") {" +
              op.fields.str() + "} : (!atlas.state) -> !atlas.state\n";
  auto module = parseSourceString<ModuleOp>(source + "}", &context);
  if (!module)
    return false;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  return succeeded(stream) && succeeded(verifyAtlasTimedStream(*stream, provider));
}

// Deliberately synthetic policies establish policy isolation, not a claim about any hardware.
class ToyReservations final : public TimingReservations {
public:
  ToyReservations(int capacity, bool extend, bool missing = false) : capacity(capacity), extend(extend), missing(missing) {}
  TimingRuleResult<std::string> conflict(const Instr &, const Footprint &f, int cycle) const override {
    if (missing)
      return {{}, "toy reservation resource has no coverage"};
    if (f.holds.empty())
      return {};
    unsigned live = llvm::count_if(until, [&](int end) { return end > cycle; });
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

using Reservations = TimingRuleResult<std::unique_ptr<TimingReservations>>;
std::function<Reservations()> reservations(int capacity, bool extend = false, bool missing = false) {
  return [=] { return Reservations{std::make_unique<ToyReservations>(capacity, extend, missing), {}}; };
}

TimingProvider toy(int capacity = 1, bool extend = false) {
  TimingProvider p;
  p.id = "synthetic-test-policy";
  p.validateScope = [](const AtlasStream &) { return std::string{}; };
  p.footprint = [](const Instr &, const RegValues &) { return Footprint{}; };
  p.dependence = [](const Instr &, const Footprint &, const Instr &, const Footprint &) { return TimingRuleResult<Dependence>{}; };
  p.dmaConflict = [](const Footprint &, const Footprint &) { return TimingRuleResult<DMACompletionConflict>{}; };
  p.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  p.createReservations = reservations(capacity, extend);
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

TimingProvider occupying(int capacity, bool extend = false) {
  auto p = toy(capacity, extend);
  p.footprint = occupied;
  return p;
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
  check(!accepts(context, resources, occupying(1)), "capacity one rejects overlapping holds");
  check(accepts(context, resources, occupying(2)), "capacity two accepts identical overlapping holds");
  auto model = npuModelTimingProvider();
  model.footprint = occupied;
  check(!accepts(context, resources, model), "model capacity does not leak into custom capacity two");
  const OperationSpec isolated[] = {a, nop, nop, nop, halt};
  check(!accepts(context, isolated, occupying(0)), "custom reservation can reject model-admissible hold");
  check(accepts(context, isolated, model), "model adapter preserves its existing reservation rule");

  const OperationSpec waits[] = {dma, wait, b, nop, nop, nop, halt};
  check(accepts(context, waits, occupying(1, false)), "provider may release reservations at WAIT");
  check(!accepts(context, waits, occupying(1, true)), "provider WAIT extension changes acceptance");
  const OperationSpec pending[] = {dma, b, wait, halt};
  p = toy();
  check(accepts(context, pending, p), "covered DMA disjointness policy accepts nonconflicting work");
  p.dmaConflict = [](const Footprint &, const Footprint &) { return TimingRuleResult<DMACompletionConflict>{{true, EdgeKind::RAW}, {}}; };
  check(!accepts(context, pending, p), "supplied DMA conflict rule changes acceptance");

  const OperationSpec publish = {"csr", "kind = \"rrw\", address = 3088 : i32, dst = 0 : i32, source = 0 : i32"};
  const OperationSpec premature[] = {a, publish, nop, nop, halt}, complete[] = {a, nop, nop, publish, halt};
  check(!accepts(context, premature, occupying(2)), "publication waits for supplied completion even without a pair dependence");
  check(accepts(context, complete, occupying(2)), "publication accepts completed work under supplied policy");

  unsigned factories = 0;
  p = toy();
  p.createReservations = [&] { ++factories; return reservations(1)(); };
  const OperationSpec blocks[] = {{"jump", "kind = \"jal\", dst = 0 : i32, base = 0 : i32, offset = 4 : i32"}, nop, a, halt};
  check(accepts(context, blocks, p) && factories == 2, "each emitted block receives a fresh provider reservation state");
}

void testMissingCoverage(MLIRContext &context) {
  const OperationSpec stream[] = {a, b, halt};
  using Mutation = std::function<void(TimingProvider &)>;
  const std::pair<StringRef, Mutation> cases[] = {
      {"missing provider id fails without fallback", [](TimingProvider &p) { p.id.clear(); }},
      {"missing scope rule fails without fallback", [](TimingProvider &p) { p.validateScope = {}; }},
      {"missing footprint rule fails without fallback", [](TimingProvider &p) { p.footprint = {}; }},
      {"missing dependence rule fails without fallback", [](TimingProvider &p) { p.dependence = {}; }},
      {"missing DMA conflict rule fails without fallback", [](TimingProvider &p) { p.dmaConflict = {}; }},
      {"missing issue gap rule fails without fallback", [](TimingProvider &p) { p.issueGap = {}; }},
      {"missing reservation factory fails without fallback", [](TimingProvider &p) { p.createReservations = {}; }},
      {"null reservation state fails closed", [](TimingProvider &p) { p.createReservations = [] { return Reservations{}; }; }},
      {"factory coverage error fails closed",
       [](TimingProvider &p) { p.createReservations = [] { return Reservations{{}, "uncovered reservation policy"}; }; }},
      {"unknown pair is distinct from zero spacing", [](TimingProvider &p) {
         p.dependence = [](const Instr &, const Footprint &, const Instr &, const Footprint &) { return TimingRuleResult<Dependence>{{}, "uncovered toy instruction pair"}; };
       }},
      {"unknown reservation resource is distinct from no conflict", [](TimingProvider &p) { p.createReservations = reservations(1, false, true); }},
      {"invalid issue gap fails closed", [](TimingProvider &p) { p.issueGap = [](const Instr &) { return TimingRuleResult<int>{0, {}}; }; }},
      {"missing issue gap coverage fails closed",
       [](TimingProvider &p) { p.issueGap = [](const Instr &) { return TimingRuleResult<int>{0, "uncovered issue gap"}; }; }},
      {"unknown footprint instance fails closed", [](TimingProvider &p) {
         p.footprint = [](const Instr &, const RegValues &) { Footprint f; f.error = "unknown toy operand instance"; return f; };
       }}};
  for (const auto &[name, mutate] : cases) {
    auto p = toy();
    mutate(p);
    check(!accepts(context, stream, p), name);
  }
  auto p = toy();
  unsigned footprintCalls = 0;
  p.validateScope = [](const AtlasStream &) { return "outside toy VLS scope"; };
  p.footprint = [&](const Instr &, const RegValues &) { ++footprintCalls; return Footprint{}; };
  check(!accepts(context, stream, p) && footprintCalls == 0, "program scope is checked before timing rules");
}

void testBoundedScope(MLIRContext &context) {
  auto p = toy();
  p.id = "synthetic-vls-subset";
  p.validateScope = [](const AtlasStream &s) {
    for (const Instr &in : s.instrs)
      if (!llvm::is_contained({"addi", "lui", "vload", "vstore", "delay", "csrrw", "ecall"}, in.op->name))
        return std::string("operation outside synthetic VLS subset");
    return std::string{};
  };
  p.footprint = [](const Instr &in, const RegValues &regs) {
    Footprint f;
    if ((in.op->opClass == OpClass::VLoad || in.op->opClass == OpClass::VStore) && !regs[in.rs1])
      f.error = "unknown base outside synthetic VLS subset";
    return f;
  };
  const OperationSpec vector = {"vload", "dst = 0 : i32, base = 1 : i32, offset = 0 : i32, format = \"raw\""};
  const OperationSpec known[] = {a, vector, halt}, unknown[] = {vector, halt}, transfer[] = {dma, wait, halt};
  const OperationSpec mxu[] = {{"mxu_push", "kind = \"weight_fp8\", unit = 0 : i32, src = 0 : i32, slot = 0 : i32"}, halt};
  const OperationSpec vpu[] = {{"vpu_unary", "kind = \"mov\", dst = 0 : i32, src = 0 : i32"}, halt};
  const OperationSpec branch[] = {{"branch", "kind = \"beq\", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 4 : i32"}, nop, halt};
  check(accepts(context, known, p), "covered straight-line VLS subset accepts");
  check(!accepts(context, unknown, p), "VLS subset rejects unknown bases");
  check(!accepts(context, transfer, p), "VLS subset rejects DMA without model fallback");
  check(!accepts(context, mxu, p), "VLS subset rejects MXU without model fallback");
  check(!accepts(context, vpu, p), "VLS subset rejects VPU without model fallback");
  check(!accepts(context, branch, p), "VLS subset rejects CFG without model fallback");
}

void testRetainedProvider(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {atlas.timing_state = "timed", atlas.timing_provider = "synthetic-test-policy"} {
      %s0 = "atlas.start"() : () -> !atlas.state
      %s1 = "atlas.trap"(%s0) {kind = "ecall"} : (!atlas.state) -> !atlas.state
    })mlir", &context);
  check(bool(module), "retained custom policy module parses");
  if (!module)
    return;
  auto p = toy();
  check(succeeded(verifyAtlasTiming(*module, p)), "retained custom policy accepts complete supplied rules");
  auto mismatch = p;
  mismatch.id = "different-synthetic-policy";
  check(failed(verifyAtlasTiming(*module, mismatch)), "retained custom policy rejects supplied identity mismatch");
  check(failed(verifyAtlasTiming(*module)), "retained custom policy requires supplied rules outside the registry");
  p.dependence = {};
  check(failed(verifyAtlasTiming(*module, p)), "retained custom policy rejects incomplete supplied coverage");
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

const char *kCopy = R"mlir(module {
    %s0 = "atlas.start"() : () -> !atlas.state
    %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
    %s2 = "atlas.vload"(%s1) {dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
    %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
    %s4 = "atlas.vstore"(%s3) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
    %s5 = "atlas.trap"(%s4) {kind = "ecall"} : (!atlas.state) -> !atlas.state
  })mlir";

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
  std::string strictOption = std::string("{provider=") + kStrict + "}";
  auto byModel = runPipeline(context, kCopy, "insert-atlas-delays");
  auto byDefaultName = runPipeline(context, kCopy, "insert-atlas-delays{provider=" + kNpuModelTimingProviderId.str() + "}");
  auto byStrict = runPipeline(context, kCopy, "insert-atlas-delays" + strictOption);
  check(byModel && byDefaultName && byStrict, "delay insertion accepts each registered provider", diagnostics);
  if (!byModel || !byDefaultName || !byStrict)
    return;
  check(print(*byModel) == print(*byDefaultName), "naming the default provider changes nothing");
  check(retainedProvider(*byModel) == kNpuModelTimingProviderId && retainedProvider(*byStrict) == kStrict,
        "delay insertion stamps the provider it used");
  check(delayCycles(*byStrict) > delayCycles(*byModel), "stricter selected spacing emits longer delays");
  check(succeeded(verifyAtlasTiming(*byStrict)), "registry dispatch verifies the strict result", diagnostics);
  auto modelStream = readAtlasStream(*byModel, AtlasStreamReadMode::Verification);
  diagnostics.clear();
  check(succeeded(modelStream) && failed(verifyAtlasTimedStream(*modelStream, strictProvider({}).value)) && diagnosed("insufficient issue spacing"),
        "model spacing fails the strict provider's verification", diagnostics);
  diagnostics.clear();
  check(failed(verifyAtlasTiming(*byStrict, npuModelTimingProvider())) && diagnosed("disagrees with retained provider identity"),
        "module stamped with strict rejects the model", diagnostics);
  for (StringRef pass : {"verify-atlas-timing", "insert-atlas-delays", "schedule-atlas-stream"}) {
    diagnostics.clear();
    std::string pipeline = pass.str() + strictOption;
    check(!runPipeline(context, print(*byModel), pipeline) && diagnosed("disagrees with retained provider identity"),
          "module stamped with the model rejects a different selection", pipeline + "\n" + diagnostics);
  }
  diagnostics.clear();
  check(!runPipeline(context, kCopy, "insert-atlas-delays{provider=unknown-policy}") && diagnosed("unknown Atlas timing provider"),
        "unknown selection fails at the producer", diagnostics);

  // The list scheduler spaces work with the model graph, so its output must pass the selected provider.
  diagnostics.clear();
  check(!runPipeline(context, kCopy, "schedule-atlas-stream" + strictOption) && diagnosed("insufficient issue spacing"),
        "scheduler output is checked under a non-model provider", diagnostics);
  auto reordered = runPipeline(context, kCopy, std::string("schedule-atlas-stream{insert-delays=false provider=") + kStrict +
                                                   "},insert-atlas-delays" + strictOption + ",verify-atlas-timing");
  check(reordered && retainedProvider(*reordered) == kStrict, "reorder then strict delay insertion verifies", diagnostics);

  // A module-scoped policy (as RTL evidence selection would supply) replaces a footprint-only rejection once registered.
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

// A real virtual program lowered in-process: each verification boundary decodes and encodes the artifact once.
void testVerificationContext(MLIRContext &context) {
  auto fixture = llvm::MemoryBuffer::getFile(ATLAS_GENERATED_FIXTURE);
  auto module = fixture ? runPipeline(context, (*fixture)->getBuffer(), "lower-atlas-virtual-to-machine,insert-atlas-delays")
                        : OwningOpRef<ModuleOp>();
  check(bool(module), "virtual fixture lowers and is timed", diagnostics);
  if (!module)
    return;
  auto kind = classifyAtlasGeneratedArtifact(*module);
  check(succeeded(kind) && *kind == AtlasArtifactKind::Generated, "lowering output classifies as generated");
  SmallVector<uint32_t> expected, words;
  check(succeeded(encodeAtlasWords(*module, expected)) && !expected.empty(), "generated artifact encodes");
  auto onceEach = [](StringRef name, function_ref<LogicalResult()> verifier) {
    unsigned decodes = atlasStreamDecodeCount(), encodes = atlasWordEncodeCount();
    check(succeeded(verifier()), name);
    check(atlasStreamDecodeCount() - decodes == 1 && atlasWordEncodeCount() - encodes == 1, (name + " decodes and encodes once").str());
  };
  onceEach("generated schedule", [&] { return verifyAtlasGeneratedSchedule(*module); });
  onceEach("verification boundary", [&] { return verifyAtlasArtifact(*module, /*llvmBlock=*/true, words); });
  check(words == expected, "boundary returns the single encoding");
}
} // namespace

void atlas_test::runTimingProvider(MLIRContext &context) {
  testPolicies(context);
  testMissingCoverage(context);
  testBoundedScope(context);
  testRetainedProvider(context);
  testRegistry(context);
  testVerificationContext(context);
}
