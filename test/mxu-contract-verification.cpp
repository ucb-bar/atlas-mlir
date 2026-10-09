#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTypes.h"
#include "VerificationTestSupport.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
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

void testSourceContract(MLIRContext &context, unsigned unit) {
  auto module = parse(context, source(unit), "typed source includes every MXU expansion family");
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

void testSourceBlocks(MLIRContext &context) {
  std::string text = source(1);
  text.insert(text.find("    %legacy"), "    cf.br ^next(%s11 : !atlas.virtual_state)\n  ^next(%next_state: !atlas.virtual_state):\n");
  text.replace(text.find("return %s11"), std::string("return %s11").size(), "return %next_state");
  auto module = parse(context, text, "two source blocks with block-local resident handles");
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

// Python covers scale clobbers, SELD, joins and loop fixed points; these two facts are not reachable from streams.
void testScaleFacts() {
  auto instruction = [](const char *name, int rd, long long imm = 0) {
    timing::Instr in;
    in.op = timing::findOp(name);
    in.rd = rd;
    in.imm = imm;
    return in;
  };
  timing::RegValues regs{};
  applyAtlasScaleRegister(instruction("seli", 0, 255), regs);
  applyAtlasScaleRegister(instruction("seli", 3, 129), regs);
  for (const char *name : {"addi", "lw"})
    for (int rd : {0, 3})
      applyAtlasScaleRegister(instruction(name, rd), regs);
  check(regs[0] == 255 && regs[3] == 129, "XRF writes preserve independent ERF contents");

  AtlasStream stream;
  stream.ops.resize(3);
  stream.instrs = {instruction("seli", 3, 129), instruction("seli", 3, 128), instruction("addi", 0)};
  stream.starts = {0, 1, 2};
  stream.succs = {{2}, {2}, {}};
  auto facts = atlasScaleRegisterEntries(stream);
  check(facts[2][3] == 129 && !facts[1][3], "unreachable conflicting predecessor does not weaken a known scale");
}
} // namespace

void atlas_test::runMXUContract(MLIRContext &context) {
  for (unsigned unit : {0u, 1u})
    testSourceContract(context, unit);
  testSourceBlocks(context);
  testScaleFacts();
}
