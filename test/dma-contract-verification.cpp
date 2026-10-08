#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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

constexpr StringLiteral source = R"mlir(
module {
  func.func @transfers() -> !atlas.virtual_state {
    %base = arith.constant -2147483648 : i32
    %offset = arith.constant 1024 : i32
    %addr = arith.addi %base, %offset : i32
    %size = arith.addi %offset, %offset : i32
    %dst = arith.constant -2147475456 : i32
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %load = "atlas.virtual_dma_load_bf16"(%s0, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %s2, %tile = "atlas.virtual_dma_await_bf16"(%s1, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s3, %store = "atlas.virtual_dma_store_bf16"(%s2, %tile, %dst, %size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s4 = "atlas.virtual_dma_wait"(%s3, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %s4 : !atlas.virtual_state
  }
})mlir";

void field(DictionaryAttr record, StringRef name, uint32_t expected) {
  auto value = record.getAs<IntegerAttr>(name);
  check(value && value.getType().isSignlessInteger(32) && value.getValue().getZExtValue() == expected, name);
}

void testSourceContract(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(source, &context);
  check(module && succeeded(verify(*module)), "valid typed source");
  if (!module)
    return;
  auto function = *module->getOps<func::FuncOp>().begin();
  SmallVector<VirtualDMAAssignment> assignments;
  function.walk([&](Operation *op) {
    if (isa<VirtualDMALoadBF16Op>(op))
      assignments.push_back({op->getResult(1), {3, 2, 41, 0x2000, 9, 7, 4}});
    if (isa<VirtualDMAStoreBF16Op>(op))
      assignments.push_back({op->getResult(1), {3, 2, 7, 0x2000, 9, 7, 4}});
  });
  auto contract = buildAtlasDMAContract(function, assignments);
  check(succeeded(contract) && contract->size() == 2, "two source records");
  if (failed(contract) || contract->size() != 2)
    return;
  for (unsigned index = 0; index < 2; ++index) {
    auto record = cast<DictionaryAttr>((*contract)[index]);
    field(record, "id", index == 0 ? 7 : 41);
    field(record, "dram_byte", index == 0 ? 0x80002000 : 0x80000400);
    field(record, "size_bytes", 2048);
    field(record, "channel", 3);
    field(record, "staging_word", 0x2000);
    field(record, "staging_reg", 9);
    field(record, "dram_reg", 7);
    field(record, "size_reg", 4);
    auto direction = record.getAs<StringAttr>("direction");
    check(direction && direction.getValue() == (index == 0 ? "store" : "load"), "source direction");
  }
  std::reverse(assignments.begin(), assignments.end());
  auto reordered = buildAtlasDMAContract(function, assignments);
  check(succeeded(reordered) && *reordered == *contract, "assignment order does not change source contract");

  auto base = cast<arith::ConstantOp>(function.getBody().front().front());
  base.setValueAttr(IntegerAttr::get(IntegerType::get(&context, 32), 0xfffffc00));
  auto wrapped = buildAtlasDMAContract(function, assignments);
  // 0xfffffc00 + 1024 wraps to zero, outside the selected DRAM address domain.
  check(failed(wrapped), "wrapping source address cannot masquerade as selected DRAM");
  base.setValueAttr(IntegerAttr::get(IntegerType::get(&context, 32), 0xfffffc00u - 0x80000000u));
  auto changed = buildAtlasDMAContract(function, assignments);
  check(succeeded(changed), "wrapping i32 addition reaches selected DRAM");
  if (succeeded(changed))
    field(cast<DictionaryAttr>((*changed)[1]), "dram_byte", 0x80000000);

  assignments.pop_back();
  check(failed(buildAtlasDMAContract(function, assignments)), "missing source assignment fails");
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect, arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) { return success(); });
  testSourceContract(context);
  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures != 0;
}
