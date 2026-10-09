#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
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

// MXU-tagged and timed: the MXU contract and timing checks both read the stream.
constexpr StringLiteral timedTagged = R"mlir(module attributes {
  atlas.generated_from_virtual = "resource-contract-v1", atlas.virtual_dma_contract = [],
  atlas.virtual_mxu_contract = [{id = 0 : i32, block = 0 : i32, kind = "weight_fp8", unit = 1 : i32, reg = 11 : i32, slot = 1 : i32, weight_slot = -1 : i32, weight = -1 : i32, previous = -1 : i32, scale_reg = -1 : i32, scale = -1 : i32}],
  atlas.timing_state = "timed", atlas.timing_provider = "npu-model-rtl-match-v1"
} {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.mxu_push"(%s0) {unit = 1 : i32, kind = "weight_fp8", src = 11 : i32, slot = 1 : i32, atlas.virtual_mxu_command = 0 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.delay"(%s1) {cycles = 256 : i32, atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.trap"(%s3) {kind = "ecall"} : (!atlas.state) -> !atlas.state
})mlir";

// Legacy generated stream without DMA: decoding would reject its JALR.
constexpr StringLiteral legacyJalr = R"mlir(module attributes {atlas.generated_from_virtual} {
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

void testTimedTagged(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(timedTagged, &context);
  check(bool(module), "timed tagged fixture parses");
  if (!module)
    return;
  bool ok = false;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasGeneratedSchedule(*module)); });
  check(ok, "generated schedule accepts timed tagged fixture");
  check(counts.decodes == 1 && counts.encodes == 1, "generated schedule decodes and encodes once");
  llvm::SmallVector<uint32_t> words;
  counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/true, words)); });
  check(ok && words.size() == 4, "boundary returns the single encoding");
  check(counts.decodes == 1 && counts.encodes == 1, "boundary decodes and encodes once");
}

void testLegacyJalr(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(legacyJalr, &context);
  check(bool(module), "legacy JALR fixture parses");
  if (!module)
    return;
  bool ok = false;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasGeneratedSchedule(*module)); });
  check(ok, "legacy JALR artifact without DMA is accepted");
  check(counts.decodes == 0 && counts.encodes == 0, "legacy JALR artifact is never decoded");
  llvm::SmallVector<uint32_t> words;
  counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/false, words)); });
  check(ok && words.size() == 4, "boundary accepts and encodes legacy JALR artifact");
  check(counts.decodes == 0 && counts.encodes == 1, "boundary encodes legacy JALR artifact once without decoding");
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &diagnostic) {
    llvm::errs() << diagnostic << '\n';
    return success();
  });
  testTimedTagged(context);
  testLegacyJalr(context);
  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures != 0;
}
