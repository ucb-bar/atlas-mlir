#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
unsigned checks = 0, failures = 0;
void check(bool condition, StringRef label) {
  ++checks;
  if (!condition) {
    ++failures;
    llvm::errs() << "FAIL: " << label << '\n';
  }
}

// Hand-written (no marker, contracts or tags) and timed: only the timing check
// reads the stream.
constexpr StringLiteral handTimed = R"mlir(module attributes {
  atlas.timing_state = "timed", atlas.timing_provider = "npu-model-rtl-match-v1"
} {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.mxu_push"(%s0) {unit = 1 : i32, kind = "weight_fp8", src = 11 : i32, slot = 1 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.delay"(%s1) {cycles = 256 : i32, atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.trap"(%s3) {kind = "ecall"} : (!atlas.state) -> !atlas.state
})mlir";

// Hand-written and untimed: no check reads the stream, so decoding, which
// rejects every JALR, never runs.
constexpr StringLiteral handJalr = R"mlir(module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.upper"(%s0) {kind = "lui", dst = 4 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_reg"(%s1) {kind = "add", dst = 5 : i32, lhs = 4 : i32, rhs = 4 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.jump"(%s2) {kind = "jalr", dst = 0 : i32, base = 31 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
})mlir";

struct Counts {
  unsigned decodes, encodes;
};

template <typename Fn>
Counts counted(Fn &&fn) {
  unsigned decodes = atlasStreamDecodeCount(), encodes = atlasWordEncodeCount();
  fn();
  return {atlasStreamDecodeCount() - decodes, atlasWordEncodeCount() - encodes};
}

// Lowers a real virtual program in-process to a timed generated artifact.
void testGenerated(MLIRContext &context) {
  auto module = parseSourceFile<ModuleOp>(ATLAS_GENERATED_FIXTURE, ParserConfig(&context));
  check(bool(module), "virtual fixture parses");
  if (!module)
    return;
  auto pipeline = parsePassPipeline(
      "builtin.module(lower-atlas-virtual-to-machine,insert-atlas-delays)");
  check(succeeded(pipeline), "lowering pipeline parses");
  if (failed(pipeline))
    return;
  auto pm = PassManager::on<ModuleOp>(&context);
  static_cast<OpPassManager &>(pm) = *pipeline;
  bool lowered = succeeded(pm.run(*module));
  check(lowered, "virtual fixture lowers and is timed");
  if (!lowered)
    return;
  auto kind = classifyAtlasGeneratedArtifact(*module);
  check(succeeded(kind) && *kind == AtlasArtifactKind::Generated, "lowering output classifies as generated");
  auto state = (*module)->getAttrOfType<StringAttr>(kAtlasTimingState);
  check(state && state.getValue() == "timed", "lowering output is timed");
  llvm::SmallVector<uint32_t> expected;
  check(succeeded(encodeAtlasWords(*module, expected)) && !expected.empty(), "generated artifact encodes");
  bool ok = false;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasGeneratedSchedule(*module)); });
  check(ok, "generated schedule accepts the lowered artifact");
  check(counts.decodes == 1 && counts.encodes == 1, "generated schedule decodes and encodes once");
  llvm::SmallVector<uint32_t> words;
  counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/true, words)); });
  check(ok && words == expected, "boundary returns the single encoding");
  check(counts.decodes == 1 && counts.encodes == 1, "boundary decodes and encodes once");
}

void testHandTimed(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(handTimed, &context);
  check(bool(module), "hand-written timed fixture parses");
  if (!module)
    return;
  auto kind = classifyAtlasGeneratedArtifact(*module);
  check(succeeded(kind) && *kind == AtlasArtifactKind::HandWritten, "timed fixture classifies as hand-written");
  bool ok = false;
  llvm::SmallVector<uint32_t> words;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/true, words)); });
  check(ok && words.size() == 4, "boundary accepts and encodes hand-written timed stream");
  check(counts.decodes == 1 && counts.encodes == 1, "boundary decodes hand-written timed stream once");
}

void testHandJalr(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(handJalr, &context);
  check(bool(module), "hand-written JALR fixture parses");
  if (!module)
    return;
  auto kind = classifyAtlasGeneratedArtifact(*module);
  check(succeeded(kind) && *kind == AtlasArtifactKind::HandWritten, "JALR fixture classifies as hand-written");
  bool ok = false;
  llvm::SmallVector<uint32_t> words;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/false, words)); });
  check(ok && words.size() == 4, "boundary accepts and encodes hand-written untimed JALR stream");
  check(counts.decodes == 0 && counts.encodes == 1, "boundary encodes untimed JALR stream once without decoding");
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect>();
  registerLowerAtlasVirtualToMachinePass();
  registerInsertAtlasDelaysPass();
  MLIRContext context(registry);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &diagnostic) {
    llvm::errs() << diagnostic << '\n';
    return success();
  });
  testGenerated(context);
  testHandTimed(context);
  testHandJalr(context);
  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures != 0;
}
