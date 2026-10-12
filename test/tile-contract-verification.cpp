#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral source = R"mlir(module {
  func.func @tiles(%control: i1, %count: i32) -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415951872 : i64,
    atlas.control_dram_base = 2415984640 : i64
  } {
    %base = arith.constant -2147483648 : i32
    %half = arith.constant 1024 : i32
    %addr = arith.addi %base, %half : i32
    %size = arith.addi %half, %half : i32
    %dst = arith.constant -2147475456 : i32
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %seed = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s3, %pending = "atlas.virtual_dma_load_bf16"(%s2, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %s4, %loaded = "atlas.virtual_dma_await_bf16"(%s3, %pending) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s5, %store = "atlas.virtual_dma_store_bf16"(%s4, %loaded, %dst, %size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s6 = "atlas.virtual_dma_wait"(%s5, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s7 = "atlas.virtual_output_bf16"(%s6, %seed) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s7 : !atlas.virtual_state
  }
})mlir";

// Explicit transfers get ids 17 (load) and 41 (store) in nonpreferred placements.
SmallVector<VirtualDMAAssignment> transfersOf(func::FuncOp function) {
  SmallVector<VirtualDMAAssignment> transfers;
  function.walk([&](Operation *op) {
    if (isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(op))
      transfers.push_back({op->getResult(1), {7, 2, isa<VirtualDMALoadBF16Op>(op) ? 17u : 41u, 0x20000, 9, 7, 4}});
  });
  return transfers;
}

struct Expected {
  StringRef kind;
  int reg;
  uint32_t vmem, dram;
  int bytes, channel = -1, transfer = -1;
  SmallVector<int32_t> after;
};

void testSourceContract(MLIRContext &context, bool multipleBlocks) {
  std::string text = source.str();
  if (multipleBlocks) {
    text = replace(text, "    %s7", "    cf.br ^next(%s6 : !atlas.virtual_state)\n  ^next(%next_state: !atlas.virtual_state):\n    %s7");
    text = replace(text, "(%s6, %seed)", "(%next_state, %seed)");
  }
  auto module = parse(context, text, "typed tile source parses");
  if (!module)
    return;
  auto function = firstFunction(*module);
  SmallVector<VirtualRegisterAssignment> registers;
  function.walk([&](Operation *op) {
    for (Value value : op->getResults()) {
      if (isa<VirtualFP8Type>(value.getType()))
        registers.push_back({value, isa<VirtualPackFP8Op>(op) ? 13u : 11u});
      if (isa<VirtualBF16Type>(value.getType()))
        registers.push_back({value, isa<VirtualInputBF16Op>(op) ? 40u : 44u});
    }
  });
  auto transfers = transfersOf(function);
  FixedResourcePlacement fixed{};
  fixed.inputWord = 256; fixed.outputWord = 66048; fixed.inputWindowWords = 1536; fixed.outputWindowWords = 1024;
  fixed.mailboxWord = 1024; fixed.packWord = 33024; fixed.packRelayoutWord = 33280; fixed.loadChannel = 3; fixed.storeChannel = 5;
  const int32_t scalarRegs[] = {19, 21};
  // Literal source facts and nonpreferred placements are the oracle; one line per issued group.
  const Expected expected[] = {
      {"dma_load", -1, 0x1000, 0x90010000, 1024, 3}, {"dma_wait", -1, 0, 0, 0, 3, -1, {0}},
      {"mailbox_load", 19, 0x1000, 0, 4, -1, -1, {1}}, {"mailbox_load", 21, 0x1004, 0, 4, -1, -1, {1}},
      {"dma_load", -1, 0x1400, 0x90001000, 1024, 3}, {"dma_wait", -1, 0, 0, 0, 3, -1, {4}}, {"vload", 11, 0x1400, 0, 1024, -1, -1, {5}},
      {"dma_load", -1, 0xc00, 0x90000800, 1024, 3}, {"dma_wait", -1, 0, 0, 0, 3, -1, {7}}, {"vload", 40, 0xc00, 0, 1024, -1, -1, {8}},
      {"dma_load", -1, 0x1000, 0x90000c00, 1024, 3}, {"dma_wait", -1, 0, 0, 0, 3, -1, {10}}, {"vload", 41, 0x1000, 0, 1024, -1, -1, {11}},
      {"vstore", 13, 0x20400, 0, 1024}, {"vload", 13, 0x20800, 0, 1024, -1, -1, {13}},
      {"dma_load", -1, 0x80000, 0x80000400, 2048, 7, 17}, {"dma_wait", -1, 0, 0, 0, 7, 17, {15}},
      {"vload", 44, 0x80000, 0, 1024, -1, 17, {16}}, {"vload", 45, 0x80400, 0, 1024, -1, 17, {16}},
      {"vstore", 44, 0x80000, 0, 1024, -1, 41}, {"vstore", 45, 0x80400, 0, 1024, -1, 41},
      {"dma_store", -1, 0x80000, 0x80002000, 2048, 7, 41, {19, 20}}, {"dma_wait", -1, 0, 0, 0, 7, 41, {21}},
      {"vstore", 40, 0x41000, 0, 1024}, {"dma_store", -1, 0x41000, 0x90008800, 1024, 5, -1, {23}}, {"dma_wait", -1, 0, 0, 0, 5, -1, {24}},
      {"vstore", 41, 0x41400, 0, 1024}, {"dma_store", -1, 0x41400, 0x90008c00, 1024, 5, -1, {26}}, {"dma_wait", -1, 0, 0, 0, 5, -1, {27}}};
  auto contract = buildAtlasTileContract(function, registers, transfers, fixed, scalarRegs);
  check(succeeded(contract) && contract->size() == std::size(expected), "source expansion includes boundary, explicit, mailbox and PACK endpoints");
  if (failed(contract) || contract->size() != std::size(expected))
    return;
  for (unsigned id = 0; id < std::size(expected); ++id) {
    auto record = cast<DictionaryAttr>((*contract)[id]);
    const Expected &e = expected[id];
    check(record.size() == 9, "closed nine-field schema");
    for (auto [name, value] : {std::pair<StringRef, int64_t>{"id", id}, {"reg", e.reg}, {"vmem_byte", e.vmem}, {"dram_byte", e.dram},
                               {"bytes", e.bytes}, {"channel", e.channel}, {"transfer", e.transfer}})
      field(record, name, value);
    auto kind = record.getAs<StringAttr>("kind");
    check(kind && kind.getValue() == e.kind, "source-derived transfer kind");
    auto after = record.getAs<DenseI32ArrayAttr>("after");
    check(after && after.asArrayRef() == ArrayRef<int32_t>(e.after), "source-derived predecessor identities");
  }
  std::reverse(registers.begin(), registers.end());
  std::reverse(transfers.begin(), transfers.end());
  auto reordered = buildAtlasTileContract(function, registers, transfers, fixed, scalarRegs);
  check(succeeded(reordered) && *reordered == *contract, "source order is independent of assignment vector order");

  auto fails = [&](StringRef name, ArrayRef<VirtualRegisterAssignment> r, ArrayRef<VirtualDMAAssignment> t,
                   const FixedResourcePlacement &f, ArrayRef<int32_t> arguments) {
    check(failed(buildAtlasTileContract(function, r, t, f, arguments)), name);
  };
  for (uint32_t words : {0u, 1024u}) {
    auto invalid = fixed;
    invalid.inputWindowWords = words;
    fails("input index requires a nonempty sufficient VMEM window", registers, transfers, invalid, scalarRegs);
  }
  for (uint32_t words : {0u, 512u}) {
    auto invalid = fixed;
    invalid.outputWindowWords = words;
    fails("output index requires a nonempty sufficient VMEM window", registers, transfers, invalid, scalarRegs);
  }
  auto foreign = parseSourceString<ModuleOp>(source, &context);
  Value foreignTensor;
  firstFunction(*foreign).walk([&](VirtualInputFP8Op op) { foreignTensor = op.getValue(); });
  for (auto [name, claim] : {std::pair<StringRef, Value>{"null tensor claim fails safely", Value{}},
                             {"foreign source tensor claim fails", foreignTensor},
                             {"scalar cannot claim a tensor placement", function.getBody().front().front().getResult(0)}}) {
    auto claimed = registers;
    claimed.push_back({claim, 11});
    fails(name, claimed, transfers, fixed, scalarRegs);
  }
  fails("missing mailbox argument placements fail", registers, transfers, fixed, {});
  fails("missing explicit transfer placement fails", registers, ArrayRef(transfers).drop_back(), fixed, scalarRegs);
  fails("missing source tensor placement fails", {}, transfers, fixed, scalarRegs);
}

// Source DMA records are ordered by transfer id, not by assignment order; Python checks lowered field values.
void testDMAContract(MLIRContext &context) {
  auto module = parse(context, source, "typed DMA source parses");
  if (!module)
    return;
  auto function = firstFunction(*module);
  auto assignments = transfersOf(function);
  std::reverse(assignments.begin(), assignments.end());
  auto contract = buildAtlasDMAContract(function, assignments);
  check(succeeded(contract) && contract->size() == 2, "two source records");
  if (failed(contract) || contract->size() != 2)
    return;
  field(cast<DictionaryAttr>((*contract)[0]), "id", 17);
  field(cast<DictionaryAttr>((*contract)[1]), "id", 41);
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
    field(cast<DictionaryAttr>((*changed)[0]), "dram_byte", 0x80000000);
  assignments.pop_back();
  check(failed(buildAtlasDMAContract(function, assignments)), "missing source assignment fails");
}
} // namespace

void atlas_test::runTileContract(MLIRContext &context) {
  testSourceContract(context, false);
  testSourceContract(context, true);
  testDMAContract(context);
}
