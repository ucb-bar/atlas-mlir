#include "Atlas/AtlasModelTiming.h"
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
  auto provider = modelTimingProvider(facts);
  int failures = 0;
  auto check = [&](bool okay, const char *what) {
    if (!okay) { ++failures; llvm::errs() << "FAIL: " << what << '\n'; }
  };
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
      if (!f.error.empty()) return f.error;
      answer = reservations->reserve(command, f, cycle);
      if (!answer.empty()) break;
    }
    return answer;
  };
  check(conflict({{relu, 0}, {transpose, 1}}).empty(), "VPU then XLU uses independent engines");
  check(conflict({{transpose, 0}, {relu, 1}}).empty(), "XLU then VPU uses independent engines");
  check(!conflict({{relu, 0}, {exp, 0}}).empty(), "one frontend issue per edge");
  check(conflict({{relu, 0}, {exp, 1}}).empty(), "independent VPU logic initiates before drain without pair catalog");
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
  check(conflict({{instr("vredmax.bf16", 4, 0), 0}, {instr("vrelu.bf16", 12, 0), 64}}).empty(),
        "same-register same-row VPU reads use native coalescing");
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
