#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>
#include <string>

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

// Source SSA and deliberately nonpreferred placements define the oracle.
// No allocator, lifetime summary, or emitted command is consulted.
std::string source(unsigned unit) {
  std::string u = std::to_string(unit);
  std::string weight = "!atlas.virtual_mxu_weight<" + u + ">";
  std::string acc = "!atlas.virtual_mxu_acc<" + u + ">";
  std::string body = R"mlir(module { func.func @test() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %seed = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %scale = "atlas.virtual_scale_constant"() {code = 173 : i32} : () -> !atlas.virtual_scale
)mlir";
  unsigned next = 2;
  std::string current = "%s2";
  auto effect = [&](StringRef name, std::string operands, std::string types, std::string result, std::string resultType, bool attribute = false) {
    std::string after = "%s" + std::to_string(++next);
    body += "    " + after + ", %" + result + " = \"atlas.virtual_mxu_" + name.str() + "\"(" + current + ", " + operands + ")";
    if (attribute)
      body += " {unit = " + u + " : i32}";
    body += " : (!atlas.virtual_state, " + types + ") -> (!atlas.virtual_state, " + resultType + ")\n";
    current = after;
  };
  effect("load_weight", "%x", "!atlas.virtual_fp8", "w", weight, true);
  effect("load_acc_bf16", "%seed", "!atlas.virtual_bf16", "b", acc, true);
  effect("readout_bf16", "%b", acc, "yb", "!atlas.virtual_bf16");
  effect("load_acc_fp8", "%x", "!atlas.virtual_fp8", "a0", acc, true);
  effect("accumulate", "%x, %w, %a0", "!atlas.virtual_fp8, " + weight + ", " + acc, "a1", acc);
  effect("accumulate", "%x, %w, %a1", "!atlas.virtual_fp8, " + weight + ", " + acc, "a2", acc);
  effect("readout_fp8", "%a2, %scale", acc + ", !atlas.virtual_scale", "yf", "!atlas.virtual_fp8");
  effect("reset", "%x, %w", "!atlas.virtual_fp8, " + weight, "r", acc);
  effect("readout_bf16", "%r", acc, "yr", "!atlas.virtual_bf16");
  body += "    %legacy = \"atlas.virtual_mxu_matmul\"(%x, %x) {unit = " + u + " : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16\n";
  return body + "    return " + current + " : !atlas.virtual_state\n} }";
}

struct Expected {
  StringRef kind;
  int reg, slot, weightSlot = -1, weight = -1, previous = -1, scaleReg = -1, scale = -1;
};

void field(DictionaryAttr record, StringRef name, int expected) {
  auto value = record.getAs<IntegerAttr>(name);
  check(value && value.getType().isSignlessInteger(32) && value.getInt() == expected, name);
}

void testSourceContract(MLIRContext &context, unsigned unit) {
  auto module = parseSourceString<ModuleOp>(source(unit), &context);
  check(module && succeeded(verify(*module)), "typed source includes every MXU expansion family");
  if (!module)
    return;
  auto function = *module->getOps<func::FuncOp>().begin();
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualMXUAssignment> handles;
  function.walk([&](Operation *op) {
    for (Value value : op->getResults()) {
      if (isa<VirtualFP8Type>(value.getType()))
        registers.push_back({value, isa<VirtualMXUReadoutFP8Op>(op) ? 13u : 11u});
      if (isa<VirtualBF16Type>(value.getType()))
        registers.push_back({value, isa<VirtualInputBF16Op>(op) ? 40u : isa<VirtualMXUMatmulOp>(op) ? 48u : 44u});
      if (isa<VirtualMXUWeightType, VirtualMXUAccType>(value.getType()))
        handles.push_back({value, {unit, 1}});
    }
  });
  FixedResourcePlacement fixed{};
  fixed.scaleReg = 7;
  fixed.mxuWeightSlot = 0;
  fixed.mxuAccSlot = 0;
  const Expected expected[] = {
      {"weight_fp8", 11, 1}, {"acc_bf16", 40, 1},
      {"pop_bf16", 44, 1, -1, -1, 1, 0}, {"acc_fp8", 11, 1},
      {"accumulate", 11, 1, 1, 0, 3}, {"accumulate", 11, 1, 1, 0, 4},
      {"pop_fp8", 13, 1, -1, -1, 5, 7, 173}, {"reset", 11, 1, 1, 0},
      {"pop_bf16", 44, 1, -1, -1, 7, 0}, {"weight_fp8", 11, 0},
      {"reset", 11, 0, 0, 9}, {"pop_bf16", 48, 0, -1, -1, 10, 0}};
  auto contract = buildAtlasMXUContract(function, registers, handles, fixed);
  check(succeeded(contract) && contract->size() == std::size(expected), "source traversal expands legacy into three records");
  if (failed(contract) || contract->size() != std::size(expected))
    return;
  for (unsigned id = 0; id < std::size(expected); ++id) {
    auto record = cast<DictionaryAttr>((*contract)[id]);
    const Expected &e = expected[id];
    check(record.size() == 11, "closed eleven-field schema");
    field(record, "id", id);
    field(record, "block", 0);
    field(record, "unit", unit);
    field(record, "reg", e.reg);
    field(record, "slot", e.slot);
    field(record, "weight_slot", e.weightSlot);
    field(record, "weight", e.weight);
    field(record, "previous", e.previous);
    field(record, "scale_reg", e.scaleReg);
    field(record, "scale", e.scale);
    auto kind = record.getAs<StringAttr>("kind");
    check(kind && kind.getValue() == e.kind, "source-derived command kind");
  }
  std::reverse(registers.begin(), registers.end());
  std::reverse(handles.begin(), handles.end());
  auto reordered = buildAtlasMXUContract(function, registers, handles, fixed);
  check(succeeded(reordered) && *reordered == *contract, "assignment order does not define source command identity");
  function.walk([&](VirtualScaleConstantOp op) { op->setAttr("code", IntegerAttr::get(IntegerType::get(&context, 32), 0)); });
  auto zeroScale = buildAtlasMXUContract(function, registers, handles, fixed);
  check(succeeded(zeroScale), "scale code zero remains a source fact");
  if (succeeded(zeroScale))
    field(cast<DictionaryAttr>((*zeroScale)[6]), "scale", 0);
  auto saved = handles.pop_back_val();
  check(failed(buildAtlasMXUContract(function, registers, handles, fixed)), "missing source handle placement fails");
  handles.push_back(saved);
  registers.clear();
  check(failed(buildAtlasMXUContract(function, registers, handles, fixed)), "missing source tensor placement fails");
}

void testStreamMetadata(MLIRContext &context) {
  // A complete generated envelope: the pushed register needs a tile-checked
  // origin, because the stream rewrite rechecks every contract.
  constexpr StringLiteral source = R"mlir(module attributes {
    atlas.generated_from_virtual = "resource-contract-v4", atlas.timing_state = "timed",
    atlas.timing_provider = "npu-model-rtl-match-v1", atlas.virtual_dma_contract = [],
    atlas.virtual_mxu_contract = [{id = 0 : i32, block = 0 : i32, kind = "weight_fp8", unit = 1 : i32, reg = 11 : i32, slot = 1 : i32, weight_slot = -1 : i32, weight = -1 : i32, previous = -1 : i32, scale_reg = -1 : i32, scale = -1 : i32}],
    atlas.virtual_tile_contract = [
      {id = 0 : i32, kind = "dma_load", reg = -1 : i32, vmem_byte = 1310720 : i32, dram_byte = -1870659584 : i32, bytes = 1024 : i32, channel = 0 : i32, transfer = -1 : i32, after = array<i32>},
      {id = 1 : i32, kind = "dma_wait", reg = -1 : i32, vmem_byte = 0 : i32, dram_byte = 0 : i32, bytes = 0 : i32, channel = 0 : i32, transfer = -1 : i32, after = array<i32: 0>},
      {id = 2 : i32, kind = "vload", reg = 11 : i32, vmem_byte = 1310720 : i32, dram_byte = 0 : i32, bytes = 1024 : i32, channel = -1 : i32, transfer = -1 : i32, after = array<i32: 1>}],
    atlas.virtual_source_memory_contract = {effects = [{id = 0 : i32, source = 0 : i32, block = 0 : i32, launch = 0 : i32, completion = 1 : i32, dram_byte = -1870659584 : i32, bytes = 1024 : i32, write = false, predecessors = array<i32>}]},
    atlas.virtual_cfg_contract = {
      values = [{id = 0 : i32, block = 0 : i32, reg = 11 : i32, type = "fp8", operands = array<i32>, def = "atlas.virtual_dma_await_fp8"}],
      blocks = [{id = 0 : i32, condition = -1 : i32, args = array<i32>, live_in = array<i32>, operations = array<i32: 0, 1, 2>, edges = array<i32>}],
      edges = [],
      operations = [
        {id = 0 : i32, block = 0 : i32, name = "atlas.virtual_dma_load_fp8", operands = array<i32>, results = array<i32>, tile_commands = array<i32: 0>, mxu_commands = array<i32>},
        {id = 1 : i32, block = 0 : i32, name = "atlas.virtual_dma_await_fp8", operands = array<i32>, results = array<i32: 0>, tile_commands = array<i32: 1, 2>, mxu_commands = array<i32>},
        {id = 2 : i32, block = 0 : i32, name = "atlas.virtual_mxu_load_weight", operands = array<i32: 0>, results = array<i32>, tile_commands = array<i32>, mxu_commands = array<i32: 0>}]}
  } {
    %s0 = "atlas.start"() : () -> !atlas.state
    %s1 = "atlas.dma_config"(%s0) {channel = 0 : i32, base_reg = 0 : i32, atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 29 : i32, src = 0 : i32, immediate = 1024 : i32, atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s3 = "atlas.upper"(%s2) {kind = "lui", dst = 31 : i32, immediate = 591872 : i32, atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s4 = "atlas.upper"(%s3) {kind = "lui", dst = 30 : i32, immediate = 80 : i32, atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s5 = "atlas.dma"(%s4) {direction = "load", channel = 0 : i32, reg = 30 : i32, dram = 31 : i32, size = 29 : i32, atlas.virtual_tile_command = 0 : i32, atlas.virtual_cfg_block = 0 : i32, atlas.virtual_cfg_source = 0 : i32, atlas.virtual_cfg_operation = 0 : i32} : (!atlas.state) -> !atlas.state
    %s6 = "atlas.dma_wait"(%s5) {channel = 0 : i32, atlas.virtual_tile_command = 1 : i32, atlas.virtual_cfg_block = 0 : i32, atlas.virtual_cfg_source = 1 : i32} : (!atlas.state) -> !atlas.state
    %s7 = "atlas.vload"(%s6) {dst = 11 : i32, base = 30 : i32, offset = 0 : i32, format = "raw", atlas.virtual_tile_command = 2 : i32, atlas.virtual_cfg_block = 0 : i32, atlas.virtual_tensor_result = 0 : i32, atlas.virtual_cfg_source = 1 : i32, atlas.virtual_cfg_operation = 1 : i32} : (!atlas.state) -> !atlas.state
    %s8 = "atlas.delay"(%s7) {cycles = 256 : i32, atlas.delay_reason = "tensor_completion", atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s9 = "atlas.mxu_push"(%s8) {unit = 1 : i32, kind = "weight_fp8", src = 11 : i32, slot = 1 : i32, atlas.virtual_mxu_command = 0 : i32, atlas.virtual_cfg_block = 0 : i32, atlas.virtual_cfg_source = 2 : i32, atlas.virtual_cfg_operation = 2 : i32} : (!atlas.state) -> !atlas.state
    %s10 = "atlas.delay"(%s9) {cycles = 256 : i32, atlas.delay_reason = "mxu_weight_completion", atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32, atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
    %s12 = "atlas.trap"(%s11) {kind = "ecall", atlas.virtual_cfg_block = 0 : i32} : (!atlas.state) -> !atlas.state
  })mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  check(module && succeeded(verify(*module)), "stream fixture parses");
  if (!module)
    return;
  Attribute contract = (*module)->getAttr("atlas.virtual_mxu_contract");
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream), "verification stream retains emitted delays");
  if (failed(stream))
    return;
  SmallVector<size_t> order(stream->ops.size());
  std::iota(order.begin(), order.end(), 0);
  SmallVector<DelayInsertion> before(stream->ops.size());
  check(succeeded(writeAtlasStream(*module, *stream, order, before)), "identity stream rewrite succeeds");
  check((*module)->getAttr("atlas.virtual_mxu_contract") == contract, "stream rewrite preserves source contract");
  auto push = *module->getOps<MXUPushOp>().begin();
  auto tag = push->getAttrOfType<IntegerAttr>("atlas.virtual_mxu_command");
  check(tag && tag.getType().isSignlessInteger(32) && tag.getInt() == 0, "stream rewrite preserves command identity");
  check(succeeded(verifyAtlasGeneratedMXUContract(*module)), "rewritten stream still satisfies source contract");
}

void testSourceBlocks(MLIRContext &context) {
  std::string text = source(1);
  text.insert(text.find("    %legacy"), "    cf.br ^next(%s11 : !atlas.virtual_state)\n  ^next(%next_state: !atlas.virtual_state):\n");
  text.replace(text.find("return %s11"), std::string("return %s11").size(), "return %next_state");
  auto module = parseSourceString<ModuleOp>(text, &context);
  check(module && succeeded(verify(*module)), "two source blocks with block-local resident handles");
  if (!module)
    return;
  auto function = *module->getOps<func::FuncOp>().begin();
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualMXUAssignment> handles;
  function.walk([&](Operation *op) {
    for (Value value : op->getResults()) {
      if (isa<VirtualFP8Type>(value.getType()))
        registers.push_back({value, isa<VirtualMXUReadoutFP8Op>(op) ? 13u : 11u});
      if (isa<VirtualBF16Type>(value.getType()))
        registers.push_back({value, 40});
      if (isa<VirtualMXUWeightType, VirtualMXUAccType>(value.getType()))
        handles.push_back({value, {1, 1}});
    }
  });
  FixedResourcePlacement fixed{};
  fixed.scaleReg = 3;
  auto contract = buildAtlasMXUContract(function, registers, handles, fixed);
  check(succeeded(contract) && contract->size() == 12, "global command ids span source blocks");
  if (failed(contract) || contract->size() != 12)
    return;
  for (unsigned id = 0; id < 12; ++id) {
    auto record = cast<DictionaryAttr>((*contract)[id]);
    field(record, "id", id);
    field(record, "block", id < 9 ? 0 : 1);
  }
  field(cast<DictionaryAttr>((*contract)[10]), "weight", 9);
  field(cast<DictionaryAttr>((*contract)[11]), "previous", 10);
}

void testScaleFacts() {
  auto instruction = [](const char *name, int rd, long long imm = 0) {
    timing::Instr in;
    in.op = timing::findOp(name);
    in.rd = rd;
    in.imm = imm;
    return in;
  };
  timing::RegValues regs{};
  check(!regs[0], "e0 starts unknown");
  applyAtlasScaleRegister(instruction("seli", 0, 255), regs);
  check(regs[0] == 255, "e0 accepts SELI writes");
  applyAtlasScaleRegister(instruction("seli", 3, 129), regs);
  for (const char *name : {"addi", "lw"})
    for (int rd : {0, 3})
      applyAtlasScaleRegister(instruction(name, rd), regs);
  check(regs[0] == 255 && regs[3] == 129, "XRF writes preserve independent ERF contents");
  applyAtlasScaleRegister(instruction("seld", 0), regs);
  check(!regs[0] && regs[3] == 129, "SELD invalidates exactly the selected ERF register");

  AtlasStream stream;
  stream.ops.resize(3);
  stream.instrs = {instruction("seli", 3, 129), instruction("seli", 3, 128), instruction("addi", 0)};
  stream.starts = {0, 1, 2};
  stream.succs = {{2}, {2}, {}};
  auto facts = atlasScaleRegisterEntries(stream);
  check(facts[2][3] == 129 && !facts[1][3], "unreachable conflicting predecessor does not weaken a known scale");
  for (int reg : {0, 3}) {
    for (int code : {129, 128}) {
      stream.ops.resize(4);
      stream.instrs = {instruction("seli", reg, 129), instruction("addi", 0), instruction("seli", reg, code), instruction("addi", 0)};
      stream.starts = {0, 1, 2, 3};
      stream.succs = {{1}, {2, 3}, {1}, {}};
      facts = atlasScaleRegisterEntries(stream);
      check(code == 129 ? facts[1][reg] == 129 && facts[3][reg] == 129 : !facts[1][reg] && !facts[3][reg], "loop scales require agreement including e0");
    }
  }
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect>();
  MLIRContext context(registry);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) { return success(); });
  for (unsigned unit : {0u, 1u})
    testSourceContract(context, unit);
  testStreamMetadata(context);
  testSourceBlocks(context);
  testScaleFacts();
  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures != 0;
}
