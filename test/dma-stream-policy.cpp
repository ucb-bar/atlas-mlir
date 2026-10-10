// checkAtlasStream's dma=wait policy, with legacy footprints rewritten to it.
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

static Footprint captured(const Instr &in, const RegValues &regs) {
  Footprint f = footprintOf(in, regs);
  if (in.op->opClass == OpClass::DmaConfig) {
    f.dmaAsync = false;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      access.atCompletion = false;
  } else if (f.dmaAsync) {
    f.exclusiveVmemUntilWait = true;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      if (access.res == Res::XReg || access.res == Res::DmaBase)
        access.atCompletion = false;
  }
  return f;
}

static std::string program(const std::vector<std::string> &commands) {
  std::string text = "module {\n%s0 = \"atlas.start\"() : () -> !atlas.state\n";
  for (size_t i = 1; i <= commands.size(); ++i) {
    const std::string &command = commands[i - 1];
    size_t fields = command.find(' ');
    text += "%s" + std::to_string(i) + " = \"atlas." + command.substr(0, fields) +
            "\"(%s" + std::to_string(i - 1) + ") {" + command.substr(fields + 1) +
            "} : (!atlas.state) -> !atlas.state\n";
  }
  return text + "}\n";
}

int main() {
  MLIRContext context;
  context.getOrLoadDialect<AtlasDialect>();
  unsigned checks = 0, failures = 0;
  const std::vector<std::string> setup = {
      "alu_imm kind = \"addi\", dst = 6 : i32, src = 0 : i32, immediate = 256 : i32",
      "alu_imm kind = \"addi\", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32",
      "alu_imm kind = \"addi\", dst = 2 : i32, src = 0 : i32, immediate = 128 : i32",
      "upper kind = \"lui\", dst = 1 : i32, immediate = 589824 : i32",
      "alu_imm kind = \"addi\", dst = 5 : i32, src = 0 : i32, immediate = 0 : i32"};
  const std::string load0 = "dma direction = \"load\", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32";
  const std::string load1 = "dma direction = \"load\", channel = 1 : i32, reg = 8 : i32, dram = 1 : i32, size = 2 : i32";
  const std::string config = "dma_config channel = 7 : i32, base_reg = 5 : i32";
  const std::string wait0 = "dma_wait channel = 0 : i32";
  const std::string wait1 = "dma_wait channel = 1 : i32";
  const std::string halt = "trap kind = \"ecall\"";
  const std::string vload = "vload dst = 4 : i32, base = 8 : i32, offset = 0 : i32, format = \"raw\"";
  const std::string marker = "csr kind = \"rrw\", dst = 0 : i32, source = 5 : i32, address = 3088 : i32";
  auto test = [&](StringRef name, std::vector<std::string> body, bool selected,
                  bool expected, StringRef error = {}) {
    ++checks;
    std::vector<std::string> commands = setup;
    commands.insert(commands.end(), body.begin(), body.end());
    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      llvm::raw_string_ostream os(diagnostics);
      diagnostic.print(os);
      return success();
    });
    auto module = parseSourceString<ModuleOp>(program(commands), &context);
    FailureOr<AtlasStream> stream =
        module ? readAtlasStream(*module) : FailureOr<AtlasStream>(failure());
    bool accepted = false;
    if (succeeded(stream)) {
      for (Instr &in : stream->instrs)
        in.release = selected && in.op->opClass == OpClass::Csr;
      accepted = succeeded(checkAtlasStream(
          *stream, selected ? TargetTiming{captured} : TargetTiming{}));
    }
    if (failed(stream) || accepted != expected ||
        (!error.empty() && diagnostics.find(error.str()) == std::string::npos)) {
      ++failures;
      llvm::errs() << "FAIL " << name << ": accepted=" << accepted << '\n'
                   << diagnostics;
    }
  };
  test("selected empty wait", {wait0, halt}, true, false, "no pending transfer");
  test("selected config is synchronous", {config, halt}, true, true);
  test("selected config creates no transfer", {config, wait0, halt}, true,
       false, "no pending transfer");
  test("selected wrong channel", {load0, wait1, wait0, halt}, true, false,
       "no pending transfer on channel 1");
  test("selected stale wait", {load0, wait0, wait0, halt}, true, false,
       "no pending transfer");
  test("selected matching waits", {load0, wait0, load1, wait1, halt}, true,
       true);
  test("selected captured operand reuse",
       {load0,
        "alu_imm kind = \"addi\", dst = 1 : i32, src = 0 : i32, immediate = 17 : i32",
        config, wait0, halt},
       true, true);
  test("selected simultaneous launches", {load0, load1, wait0, wait1, halt},
       true, false, "may still be in flight");
  test("selected pending VMEM exclusion", {load0, vload, wait0, halt}, true,
       false, "may still be in flight");
  test("selected pending publication", {load0, marker, wait0, halt}, true,
       false, "completion publication requires");
  test("selected missing wait", {load0, halt}, true, false,
       "halts while DMA channel");
  test("legacy empty waits unchanged", {wait0, wait1, halt}, false, true);
  test("legacy config completion unchanged",
       {"dma_config channel = 0 : i32, base_reg = 5 : i32", wait0, halt}, false,
       true);
  llvm::outs() << checks << " DMA stream policy checks, " << failures
               << " failures\n";
  return failures ? 1 : 0;
}
