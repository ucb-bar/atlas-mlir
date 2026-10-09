#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "VerificationTestSupport.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
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
} // namespace

// Source records are ordered by transfer id, not by assignment order; Python checks lowered field values.
void atlas_test::runDMAContract(MLIRContext &context) {
  auto module = parse(context, source, "valid typed source");
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
  for (unsigned index = 0; index < 2; ++index)
    field(cast<DictionaryAttr>((*contract)[index]), "id", index == 0 ? 7 : 41);
  std::reverse(assignments.begin(), assignments.end());
  auto reordered = buildAtlasDMAContract(function, assignments);
  check(succeeded(reordered) && *reordered == *contract, "assignment order does not change source contract");

  auto base = cast<arith::ConstantOp>(function.getBody().front().front());
  base.setValueAttr(IntegerAttr::get(IntegerType::get(&context, 32), 0xfffffc00));
  // 0xfffffc00 + 1024 wraps to zero, outside the selected DRAM address domain.
  check(failed(buildAtlasDMAContract(function, assignments)), "wrapping source address cannot masquerade as selected DRAM");
  base.setValueAttr(IntegerAttr::get(IntegerType::get(&context, 32), 0xfffffc00u - 0x80000000u));
  auto changed = buildAtlasDMAContract(function, assignments);
  check(succeeded(changed), "wrapping i32 addition reaches selected DRAM");
  if (succeeded(changed))
    field(cast<DictionaryAttr>((*changed)[1]), "dram_byte", 0x80000000);
  assignments.pop_back();
  check(failed(buildAtlasDMAContract(function, assignments)), "missing source assignment fails");
}
