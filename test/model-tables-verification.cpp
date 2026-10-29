#include "Atlas/AtlasModelTables.h"
#include "VerificationTestSupport.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdlib>

using namespace mlir::atlas::timing;
using namespace atlas_test;

namespace {
std::string jsonText(const llvm::json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  return text;
}

bool sameAccesses(const Footprint &a, const Footprint &b) {
  if (a.accesses.size() != b.accesses.size()) return false;
  for (size_t i = 0; i < a.accesses.size(); ++i) {
    const auto &x = a.accesses[i], &y = b.accesses[i];
    if (x.res != y.res || x.write != y.write || x.first != y.first ||
        x.count != y.count || x.age != y.age || x.step != y.step ||
        x.anywhere != y.anywhere || x.atCompletion != y.atCompletion)
      return false;
  }
  return true;
}

bool sameTiming(const Footprint &a, const Footprint &b) {
  if (!sameAccesses(a, b) || a.readRelease != b.readRelease ||
      a.writeRelease != b.writeRelease || a.doneAge != b.doneAge ||
      a.vpuLive != b.vpuLive || a.holds.size() != b.holds.size())
    return false;
  for (size_t i = 0; i < a.holds.size(); ++i) {
    const auto &x = a.holds[i], &y = b.holds[i];
    if (x.unit != y.unit || x.index != y.index || x.from != y.from ||
        x.to != y.to || x.alt != y.alt)
      return false;
  }
  return true;
}

void sharedRules(const ModelTimingTables &facts) {
  const auto &tables = *facts.timing;
  auto provider = modelTimingProvider(std::make_shared<ModelTimingTables>(facts));
  for (const char *name : {"vmov", "vadd.bf16", "vredsum.row.bf16",
                           "vredmax.bf16", "vpack.bf16.fp8", "vload",
                           "vstore", "vtrpose.xlu"}) {
    Instr first{findOp(name), 4, 0, 2}, second{findOp(name), 12, 8, 10};
    if (!first.op->twoInput && first.op->opClass != OpClass::VpuPack) {
      first.rs2 = second.rs2 = 0;
    }
    auto a = footprintOf(first, unknownRegs());
    auto b = provider.footprint(first, unknownRegs());
    check(a.error.empty() && b.error.empty(), "both rules cover the operation", name);
    check(sameAccesses(a, b), "equal extracted events preserve access timing", name);
    check(a.doneAge == b.doneAge, "equal first-free ages preserve completion", name);
    auto last = footprintOf(second, unknownRegs());
    auto modelLast = provider.footprint(second, unknownRegs());
    check(dependence(first, a, second, last).distance ==
              dependence(first, b, second, modelLast, tables).distance,
          "equal extracted values preserve dependence", name);
  }
  Instr move{findOp("vmov"), 4, 0}, reciprocal{findOp("vrecip.bf16"), 12, 8};
  auto a = footprintOf(move, unknownRegs());
  auto b = footprintOf(reciprocal, unknownRegs());
  auto modelA = provider.footprint(move, unknownRegs());
  auto modelB = provider.footprint(reciprocal, unknownRegs());
  ReservationTable old, bound(facts.timing);
  old.reserve(move, a, 0);
  bound.reserve(move, modelA, 0);
  for (int gap = 1; gap < 140; ++gap)
    check(old.conflict(reciprocal, b, gap).empty() ==
              bound.conflict(reciprocal, modelB, gap).empty(),
          "equal values preserve per-cycle reservation admission");
  for (const char *name : {"addi", "delay", "jal"}) {
    Instr in{findOp(name), 1, 2, 3, 7};
    check(naturalGap(in) == provider.issueGap(in).value,
          "equal frontend values preserve issue gaps", name);
  }
}

void operationValues(const TimingTables &tables) {
  for (const auto &[name, timing] : tables.operations) {
    const OpInfo *op = findOp(name);
    Instr in{op, 4, 0, 2};
    if (op->mxu >= 0)
      in = {op, 0, 2, 0};
    auto original = footprintOf(in, unknownRegs(), tables);
    check(original.error.empty(), "operation has complete timing", name);
    for (const auto &[key, event] : timing.events) {
      auto changed = tables;
      auto &value = changed.operations.at(name).events.at(key);
      value.firstAge += 200;
      value.lastAge += 200;
      auto mutated = footprintOf(in, unknownRegs(), changed);
      check(mutated.error.empty() && !sameTiming(original, mutated),
            "extracted event changes timing", name + ": " + key);
    }
    for (const auto &[key, release] : timing.releases) {
      auto changed = tables;
      changed.operations.at(name).releases.at(key).to += 200;
      auto mutated = footprintOf(in, unknownRegs(), changed);
      check(mutated.error.empty() && !sameTiming(original, mutated),
            "extracted release changes timing", name + ": " + key);
    }
    auto changed = tables;
    changed.operations.at(name).firstFreeAge += 200;
    check(!sameTiming(original, footprintOf(in, unknownRegs(), changed)),
          "extracted completion age changes timing", name);
  }
}

void resourceValues(const ModelTimingTables &facts) {
  auto provider = modelTimingProvider(std::make_shared<ModelTimingTables>(facts));
  const auto &tables = *facts.timing;
  auto changed = std::make_shared<TimingTables>(tables);
  changed->ordinaryIssueGap += 2;
  check(naturalGap({findOp("addi")}, *changed) == tables.ordinaryIssueGap + 2,
        "frontend issue spacing comes from tables");
  changed->mxu[0].computeCapacity += 1;
  check(unitCapacity(Unit::MxuCompute, 0, *changed) ==
            unitCapacity(Unit::MxuCompute, 0, tables) + 1,
        "MXU tracker capacity comes from tables");
  for (int engine = 0; engine < 2; ++engine) {
    Instr compute{findOp("vmatmul.mxu" + std::to_string(engine)), 0, 0, 0};
    auto footprint = provider.footprint(compute, unknownRegs());
    std::vector<Access> weights;
    for (const auto &access : footprint.accesses)
      if (access.res == Res::Weight && !access.write) weights.push_back(access);
    check(weights.size() == 2, "both extracted weight endpoints are retained");
    if (weights.size() == 2) {
      check(weights[0].age == tables.mxu[engine].weightFirst.firstAge &&
                weights[1].age == tables.mxu[engine].weightLast.firstAge &&
                weights[0].step == tables.mxu[engine].weightFirst.step &&
                weights[1].step == tables.mxu[engine].weightLast.step,
            "weight endpoint timing comes from the extracted row windows");
    }
  }
}

void frontendValues(const TimingTables &tables) {
  auto changed = tables;
  Instr add{findOp("addi"), 1, 2, 0, 3};
  auto original = footprintOf(add, unknownRegs(), tables);
  changed.scalarReadAge += 3;
  check(!sameAccesses(original, footprintOf(add, unknownRegs(), changed)),
        "scalar source age comes from tables");
  changed = tables;
  changed.scalarWriteAge += 3;
  check(!sameAccesses(original, footprintOf(add, unknownRegs(), changed)),
        "scalar result age comes from tables");
  Instr branch{findOp("beq"), 0, 1, 2};
  changed = tables;
  changed.controlReadAge += 3;
  check(!sameAccesses(footprintOf(branch, unknownRegs(), tables),
                     footprintOf(branch, unknownRegs(), changed)),
        "control operand age comes from tables");
  Instr delay{findOp("delay"), 0, 0, 0, 3};
  changed = tables;
  changed.delayBaseGap += 2;
  check(naturalGap(delay, changed) == naturalGap(delay, tables) + 2,
        "delay base gap comes from tables");
  changed = tables;
  changed.delayImmediateMask = 1;
  check(naturalGap(delay, changed) == tables.delayBaseGap + 1,
        "delay immediate mask comes from tables");
  Instr load{findOp("lw"), 1, 2};
  changed = tables;
  changed.scalarLoadGap += 3;
  check(!sameTiming(footprintOf(load, unknownRegs(), tables),
                   footprintOf(load, unknownRegs(), changed)),
        "scalar load response gap comes from tables");
}

void reservationValues(const TimingTables &tables) {
  Instr move{findOp("vmov"), 4, 0};
  Instr reciprocal{findOp("vrecip.bf16"), 12, 8};
  auto a = footprintOf(move, unknownRegs(), tables);
  auto b = footprintOf(reciprocal, unknownRegs(), tables);
  auto conflict = [&](std::shared_ptr<const TimingTables> values) {
    ReservationTable table(std::move(values));
    table.reserve(move, a, 0);
    return !table.conflict(reciprocal, b, 1).empty();
  };
  check(!conflict(std::make_shared<TimingTables>(tables)),
        "independent VPU operations overlap");
  auto changed = std::make_shared<TimingTables>(tables);
  changed->vpuIssueSlots = 1;
  check(conflict(changed), "VPU slot capacity controls admission");
  *changed = tables;
  changed->vpuDouble.push_back(move.op->name);
  check(conflict(changed), "VPU double-slot set controls admission");
  *changed = tables;
  changed->vpuLogicGroups.push_back({move.op->name, reciprocal.op->name});
  check(conflict(changed), "VPU logic groups control admission");
  for (int engine = 0; engine < 2; ++engine) {
    Instr compute{findOp("vmatmul.mxu" + std::to_string(engine)), 0, 0, 0};
    auto footprint = footprintOf(compute, unknownRegs(), tables);
    ReservationTable original(std::make_shared<TimingTables>(tables));
    original.reserve(compute, footprint, 0);
    check(original.conflict(compute, footprint, 32).empty(),
          "MXU tracker admits another compute after the port sweep");
    *changed = tables;
    changed->mxu[engine].computeCapacity = 1;
    ReservationTable reduced(changed);
    reduced.reserve(compute, footprint, 0);
    check(!reduced.conflict(compute, footprint, 32).empty(),
          "MXU tracker capacity controls admission");
  }
}

void dmaValues(const ModelTimingTables &facts) {
  auto provider = modelTimingProvider(std::make_shared<ModelTimingTables>(facts));
  for (bool store : {false, true}) {
    auto estimate = provider.dmaEstimate(1024, store);
    int startup = store ? facts.dma.storeStartup : facts.dma.loadStartup;
    check(estimate.error.empty() &&
              estimate.value == startup + (1024 / facts.dma.beatBytes) * facts.dma.perBeatBound,
          "DMA estimate uses the extracted bound");
    check(estimate.value != dmaTransferCycles(1024, store),
          "model DMA estimate differs from the calibrated baseline");
  }
}

void loaderRefusal(llvm::StringRef bytes) {
  auto document = llvm::json::parse(bytes);
  check(bool(document), "facts are JSON");
  if (!document) { llvm::consumeError(document.takeError()); return; }
  auto *root = document->getAsObject();
  auto *architecture = root->getObject("architecture");
  if (architecture) {
    (*architecture)["mreg_count"] = 63;
    auto bad = parseModelTimingTables(jsonText(*document));
    check(!bad, "unsupported geometry refuses");
    if (!bad) llvm::consumeError(bad.takeError());
  }
  document = llvm::json::parse(bytes);
  if (!document) { llvm::consumeError(document.takeError()); return; }
  auto *instructions = document->getAsObject()->getObject("frontend_timing")->getObject("instructions");
  auto *add = instructions->getObject("addi");
  (*add)["reads"] = llvm::json::Array{};
  auto roles = parseModelTimingTables(jsonText(*document));
  check(!roles, "changed scalar operand roles refuse");
  if (!roles) llvm::consumeError(roles.takeError());
  auto malformed = parseModelTimingTables("{}");
  check(!malformed, "missing timing tables refuse");
  if (!malformed) llvm::consumeError(malformed.takeError());
  auto truncated = parseModelTimingTables(bytes.take_front(bytes.size() / 2));
  check(!truncated, "truncated facts refuse");
  if (!truncated) llvm::consumeError(truncated.takeError());
}
} // namespace

void atlas_test::runModelTables(MLIRContext &) {
  const char *directory = std::getenv("ATLAS_TEST_MODEL_FACTS");
  if (!directory) {
    llvm::outs() << "Set ATLAS_TEST_MODEL_FACTS to test regenerated timing tables\n";
    return;
  }
  std::string path = std::string(directory) + "/model-timing.json";
  auto file = llvm::MemoryBuffer::getFile(path);
  check(bool(file), "regenerated facts are readable");
  if (!file) return;
  auto facts = parseModelTimingTables((*file)->getBuffer());
  if (!facts) {
    check(false, "regenerated facts load", llvm::toString(facts.takeError()));
    return;
  }
  sharedRules(*facts);
  operationValues(*facts->timing);
  resourceValues(*facts);
  frontendValues(*facts->timing);
  reservationValues(*facts->timing);
  dmaValues(*facts);
  loaderRefusal((*file)->getBuffer());
  auto wrongHash = loadModelTimingTables(path, std::string(64, '0'));
  check(!wrongHash, "selected facts hash mismatch refuses");
  if (!wrongHash) llvm::consumeError(wrongHash.takeError());
}
