#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasModelVocabulary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir::atlas;
using namespace mlir::atlas::timing;

int testEngineAdmission() {
  const char *directory = std::getenv("ATLAS_TEST_MODEL_FACTS");
  if (!directory) return 0;
  std::string path = std::string(directory) + "/model-timing.json";
  auto bytes = llvm::MemoryBuffer::getFile(path);
  if (!bytes) { llvm::errs() << "FAIL: engine admission facts unavailable\n"; return 1; }
  auto loaded = parseModelTimingFacts((*bytes)->getBuffer(), modelSha256((*bytes)->getBuffer()), true);
  if (!loaded) { llvm::errs() << llvm::toString(loaded.takeError()) << '\n'; return 1; }
  auto facts = std::make_shared<ModelTimingFacts>(std::move(*loaded));
  auto pairs = llvm::MemoryBuffer::getFile(std::string(directory) + "/model-overlap.json");
  if (!pairs) { llvm::errs() << "FAIL: engine admission overlap contract unavailable\n"; return 1; }
  auto overlap = parseModelOverlapFacts((*pairs)->getBuffer(), modelSha256((*pairs)->getBuffer()), *facts);
  if (!overlap) { llvm::errs() << llvm::toString(overlap.takeError()) << '\n'; return 1; }
  auto provider = modelTimingProvider(facts, std::make_shared<ModelOverlapFacts>(std::move(*overlap)), true);
  int failures = 0;
  auto check = [&](bool okay, const char *what) {
    if (!okay) { ++failures; llvm::errs() << "FAIL: " << what << '\n'; }
  };
  // The explicit test parser lets this fixture reach cross-record checks after
  // rebinding per-operation bytes; production sealing rejects stale inputs first.
  auto mismatch = [&](bool capacity) {
    auto changed = llvm::json::parse((*bytes)->getBuffer());
    if (!changed) { llvm::consumeError(changed.takeError()); check(false, "parse facts for metadata mismatch fixture"); return; }
    changed->getAsObject()->getObject("dma_timing")->getObject("validation")->erase("payload_sha256");
    auto *engine = changed->getAsObject()->getObject("engine_admission");
    if (capacity) (*engine->getObject("vpu"))["capacity"] = 3;
    else (*engine->getObject("source_contract")->getObject("source_sha256"))["vpu.py"] = std::string(64, 'b');
    auto changedBytes = canonicalJson(*changed);
    auto selected = parseModelTimingFactsForTesting(changedBytes, modelSha256(changedBytes), true);
    check(bool(selected), "cross-record fixture reaches overlap parser");
    if (!selected) { llvm::errs() << llvm::toString(selected.takeError()) << '\n'; return; }
    auto pairDocument = llvm::json::parse((*pairs)->getBuffer());
    if (!pairDocument) { llvm::consumeError(pairDocument.takeError()); check(false, "parse overlap for metadata mismatch fixture"); return; }
    (*pairDocument->getAsObject())["per_op_sha256"] = selected->factsSha256();
    auto pairBytes = canonicalJson(*pairDocument);
    auto result = parseModelOverlapFacts(pairBytes, modelSha256(pairBytes), *selected);
    check(!result, "overlap refuses contradictory engine metadata after independent hash rebinding");
    if (!result) {
      auto diagnostic = llvm::toString(result.takeError());
      check(diagnostic.find(capacity ? "capacity differs" : "source differs") != std::string::npos,
            "cross-record refusal identifies authoritative capacity or source mismatch");
    }
  };
  mismatch(true);
  mismatch(false);
  check(facts->engineAdmission() != nullptr, "production evidence carries source engine admission");
  auto instr = [](const char *name, int rd, int rs1, int rs2 = 0) { return Instr{findOp(name), rd, rs1, rs2}; };
  auto relu = instr("vrelu.bf16", 4, 0), exp = instr("vexp2.bf16", 12, 8), sine = instr("vsin.bf16", 20, 16);
  auto transpose = instr("vtrpose.xlu", 22, 20);
  RegValues regs = unknownRegs();
  regs[2] = 0; regs[3] = kVmemBankBytes / 4; regs[6] = kVmemBankBytes;
  auto conflict = [&](const std::vector<std::pair<Instr, int>> &commands) {
    auto reservations = provider.createReservations().value;
    std::string answer;
    for (const auto &[command, cycle] : commands) {
      auto f = provider.footprint(command, regs);
      check(f.error.empty(), "qualified admission operand resolves");
      if (!f.error.empty()) { llvm::errs() << f.error << '\n'; return f.error; }
      answer = reservations->reserve(command, f, cycle);
      if (!answer.empty()) break;
    }
    return answer;
  };
  {
    auto operandBytes = llvm::MemoryBuffer::getFile(std::string(directory) + "/overlap-validation/standalone-operands-report.json");
    if (!operandBytes) { check(false, "source-class consumer coverage needs frozen standalone descriptors"); return failures; }
    auto report = llvm::json::parse((*operandBytes)->getBuffer());
    if (!report) { llvm::consumeError(report.takeError()); check(false, "parse standalone descriptor report"); return failures; }
    const auto *object = report->getAsObject();
    const auto *descriptors = object ? object->getArray("descriptors") : nullptr;
    check(object && object->getBoolean("passed") == true && object->getBoolean("static_input_preservation") == true && descriptors,
          "consumer checks use source-qualified input-preserving classes");
    if (!descriptors) return failures;
    int checked = 0;
    for (const auto &value : *descriptors) {
      const auto *descriptor = value.getAsObject();
      auto operation = descriptor ? descriptor->getString("operation") : std::nullopt;
      const auto *operands = descriptor ? descriptor->getObject("operands") : nullptr;
      const OpInfo *op = nullptr;
      if (operation == "xlu.transpose") op = findOp("vtrpose.xlu");
      else if (operation)
        for (const auto &[mnemonic, name] : model_schema::vpuBlocks())
          if (*operation == name) op = findOp(mnemonic);
      check(op && operands && descriptor->getBoolean("input_rows_preserved") == true,
            "standalone descriptor maps to reviewed compiler operands");
      if (!op || !operands) continue;
      Instr command{op, int(operands->getInteger("rd").value_or(0)),
          int(operands->getInteger("rs1").value_or(0)), int(operands->getInteger("rs2").value_or(0))};
      auto result = conflict({{command, 0}});
      check(result.empty(), "every frozen source-legal internal class resolves and reserves in the consumer");
      if (!result.empty()) llvm::errs() << *operation << ' ' << canonicalJson(value) << '\n';
      ++checked;
    }
    check(object->getInteger("alias_class_count") == checked, "consumer covers every frozen internal operand class");
  }
  check(!conflict({{relu, 0}, {transpose, 1}}).empty(), "omitted VPU then XLU pair remains serialized");
  check(!conflict({{transpose, 0}, {relu, 1}}).empty(), "omitted XLU then VPU pair remains serialized");
  check(conflict({{relu, 0}, {transpose, provider.footprint(relu, regs).doneAge + 1}}).empty(),
        "XLU remains legal after VPU drain");
  check(conflict({{transpose, 0}, {relu, provider.footprint(transpose, regs).doneAge + 1}}).empty(),
        "VPU remains legal after XLU drain");
  check(!conflict({{relu, 0}, {exp, 0}}).empty(), "one frontend issue per edge");
  check(conflict({{relu, 0}, {exp, 1}}).empty(), "independent VPU logic uses qualified pair rules");
  check(!conflict({{relu, 0}, {exp, 1}, {transpose, 2}}).empty(), "mixed VPU/XLU live chains retain conservative support boundary");
  {
    auto noPairs = modelTimingProvider(facts).createReservations().value;
    check(noPairs->reserve(relu, facts->resolve(relu, regs), 0).empty(), "isolated VPU operation needs no overlap waiver");
    check(!noPairs->conflict(exp, facts->resolve(exp, regs), 1).value.empty(),
          "engine metadata cannot bypass missing overlap rules");
  }
  check(!conflict({{relu, 0}, {exp, 1}, {sine, 2}}).empty(), "third live VPU operation exceeds source slot capacity");
  int final = facts->block("vpu.relu")->firstFreeAge - 1;
  check(conflict({{relu, 0}, {exp, 1}, {sine, final}}).empty(), "source VPU slot available on final write edge");
  check(!conflict({{exp, 1}, {sine, 2}, {relu, 0}}).empty(), "reverse insertion rechecks later VPU prefixes");
  auto repeat = instr("vrelu.bf16", 12, 8);
  check(!conflict({{relu, 0}, {repeat, final-1}}).empty(), "same VPU logic remains occupied before final edge");
  check(conflict({{relu, 0}, {repeat, final}}).empty(), "same VPU logic reuses final edge slot with disjoint registers");
  auto doubleInput = instr("vadd.bf16", 4, 0, 2);
  check(!conflict({{doubleInput, 0}, {exp, 1}}).empty(), "double-input source guard survives generic admission");
  check(!conflict({{relu, 0}, {instr("vtrpose.xlu", 12, 4), 1}}).empty(), "logical MREG write reservations remain enforced");
  check(!conflict({{transpose, 0}, {instr("vrelu.bf16", 12, 52), 1}}).empty(), "cross-engine physical read bank collision remains enforced");
  check(!conflict({{instr("vredmax.bf16", 4, 0), 0}, {instr("vrelu.bf16", 12, 0), 64}}).empty(),
        "source pair rules retain suffix-safe support even for an isolated legal coalescing gap");
  check(!conflict({{instr("vredmax.bf16", 4, 0), 0}, {instr("vrelu.bf16", 12, 0), 96}}).empty(),
        "omitted shared-read pair remains serialized at the former suffix boundary");
  auto reduction = instr("vredmax.bf16", 4, 0);
  check(conflict({{reduction, 0},
                  {instr("vrelu.bf16", 12, 0), provider.footprint(reduction, regs).doneAge + 1}}).empty(),
        "shared-read operands remain legal after the reduction drains");
  check(conflict({{instr("vadd.bf16", 4, 0, 0), 0}}).empty(),
        "identical same-row duplicate sources use native VPU read coalescing");
  auto vload = instr("vload", 4, 2), vstore = instr("vstore", 8, 3);
  check(!conflict({{vload, 0}, {vstore, 0}}).empty(), "independent LSU paths retain one frontend issue per edge");
  check(conflict({{vload, 0}, {vstore, 1}}).empty(), "VLOAD and VSTORE have independent source paths");
  check(!conflict({{vload, 0}, {instr("vstore", 8, 2), 1}}).empty(), "independent VLS paths retain same-bank VMEM conflict");
  auto lw = instr("lw", 4, 2), lwOther = instr("lw", 5, 6);
  check(!conflict({{lw, 0}, {lwOther, 2}}).empty(), "scalar load response busy predicate");
  check(conflict({{lw, 0}, {lwOther, 3}}).empty(), "scalar load pipeline initiates at response edge");
  check(conflict({{vload, 0}, {lwOther, 1}}).empty(), "scalar LSU and vector LSU paths are independent");
  {
    auto reservations = provider.createReservations().value;
    check(reservations->reserve(relu, provider.footprint(relu, regs), 0).empty(), "wait regression first issue");
    reservations->onWait(Instr{}, 1);
    check(!reservations->conflict(exp, provider.footprint(exp, regs), 2).value.empty(), "wait invalidates live VPU admission");
    check(reservations->conflict(exp, provider.footprint(exp, regs), final+1).value.empty(), "drained epoch restarts after wait");
  }
  return failures;
}
