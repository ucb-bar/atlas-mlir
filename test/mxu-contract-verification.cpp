#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasStream.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
// Source SSA and deliberately nonpreferred placements define the oracle.
Source families(unsigned unit) {
  Source source(173);
  auto weight = source.weight(unit);
  source.readout(source.seed(unit), unit);
  source.readout(source.accumulate(weight, source.accumulate(weight, source.seed(unit, true), unit), unit), unit, true);
  source.readout(source.reset(weight, unit), unit);
  return source;
}

void claim(func::FuncOp function, unsigned unit, SmallVector<VirtualRegisterAssignment> &registers,
           SmallVector<VirtualMXUAssignment> &handles) {
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
}

struct Expected {
  StringRef kind;
  int reg, slot, weightSlot = -1, weight = -1, previous = -1, scaleReg = -1, scale = -1;
};

void testSourceContract(MLIRContext &context, unsigned unit) {
  Source source = families(unit);
  source.legacy(unit);
  auto module = parse(context, source.finish(), "typed source includes every MXU expansion family");
  if (!module)
    return;
  auto function = firstFunction(*module);
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualMXUAssignment> handles;
  claim(function, unit, registers, handles);
  FixedResourcePlacement fixed{};
  fixed.scaleReg = 7;
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
    for (auto [name, value] : {std::pair<StringRef, int64_t>{"id", id}, {"block", 0}, {"unit", unit}, {"reg", e.reg}, {"slot", e.slot},
                               {"weight_slot", e.weightSlot}, {"weight", e.weight}, {"previous", e.previous}, {"scale_reg", e.scaleReg}, {"scale", e.scale}})
      field(record, name, value);
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
  check(failed(buildAtlasMXUContract(function, registers, ArrayRef(handles).drop_back(), fixed)), "missing source handle placement fails");
  check(failed(buildAtlasMXUContract(function, {}, handles, fixed)), "missing source tensor placement fails");
}

void testSourceBlocks(MLIRContext &context) {
  Source source = families(1);
  source.edge("next", "%next_state");
  source.legacy(1);
  auto module = parse(context, source.finish(), "two source blocks with block-local resident handles");
  if (!module)
    return;
  auto function = firstFunction(*module);
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualMXUAssignment> handles;
  claim(function, 1, registers, handles);
  auto contract = buildAtlasMXUContract(function, registers, handles, FixedResourcePlacement{});
  check(succeeded(contract) && contract->size() == 12, "global command ids span source blocks");
  if (failed(contract) || contract->size() != 12)
    return;
  for (unsigned id = 0; id < 12; ++id) {
    field(cast<DictionaryAttr>((*contract)[id]), "id", id);
    field(cast<DictionaryAttr>((*contract)[id]), "block", id < 9 ? 0 : 1);
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
