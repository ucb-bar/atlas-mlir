// Print built-in footprints with fixed legal operands for test_rtl_extract_compiler.
#include "Atlas/AtlasTiming.h"
#include <iostream>
#include <string>

using namespace mlir::atlas::timing;

static const char *resourceName(Res r) {
  static const char *names[] = {
      "XReg", "EReg", "MReg", "Acc", "Weight", "Vmem", "DmaBase", "Dram"};
  unsigned index = static_cast<unsigned>(r);
  return index < sizeof(names) / sizeof(*names) ? names[index] : "invalid";
}

// Unit enumerator names, in declaration order.
static const char *unitId(Unit u) {
  static const char *names[] = {
      "ScalarLoad", "ScalarWriteback", "VloadPath", "VstorePath", "VmemBank",
      "Xlu", "MxuPort", "MxuCompute", "MxuAccRead", "MxuAccWrite",
      "MxuWeightStream", "MxuAccStream", "Vpu"};
  return names[static_cast<int>(u)];
}

static const char *boolean(bool b) { return b ? "true" : "false"; }

static int operandValue(char kind, char field) {
  switch (kind) {
  case 'm': return field == 'd' ? 4 : field == '1' ? 8 : 12;
  case 'a': case 'w': return field == 'd' ? 1 : 0;
  case 'x': return field == 'd' ? 5 : field == '1' ? 8 : 9;
  case 'e': return field == 'd' ? 2 : 3;
  }
  return 0;
}

static Instr instruction(const OpInfo *op) {
  Instr in;
  in.op = op;
  std::string token;
  for (char c : op->operands + " ") {
    if (c != ' ') { token += c; continue; }
    if (token == "@") in.rs1 = 8;
    else if (token.size() == 2 && token[1] == 'd') in.rd = operandValue(token[0], 'd');
    else if (token.size() == 2 && token[1] == '1') in.rs1 = operandValue(token[0], '1');
    else if (token.size() == 2 && token[1] == '2') in.rs2 = operandValue(token[0], '2');
    token.clear();
  }
  return in;
}

static void footprint(const Footprint &f) {
  std::cout << "{\"error\":\"" << f.error << "\",\"accesses\":[";
  for (size_t i = 0; i < f.accesses.size(); i++) {
    const Access &a = f.accesses[i];
    std::cout << (i ? "," : "") << "{\"resource\":\"" << resourceName(a.res)
              << "\",\"write\":" << boolean(a.write) << ",\"first\":" << a.first
              << ",\"count\":" << a.count << ",\"age\":" << a.age
              << ",\"step\":" << a.step << ",\"anywhere\":" << boolean(a.anywhere)
              << ",\"at_completion\":" << boolean(a.atCompletion) << '}';
  }
  std::cout << "],\"holds\":[";
  for (size_t i = 0; i < f.holds.size(); i++) {
    const Hold &h = f.holds[i];
    std::cout << (i ? "," : "") << "{\"unit\":\"" << unitId(h.unit)
              << "\",\"index\":" << h.index << ",\"from\":" << h.from
              << ",\"to\":" << h.to << ",\"alt\":" << h.alt << '}';
  }
  std::cout << "],\"read_release\":" << f.readRelease
            << ",\"write_release\":" << f.writeRelease
            << ",\"vpu_live\":" << f.vpuLive << ",\"done_age\":" << f.doneAge << '}';
}

int main(int argc, char **argv) {
  std::cout << "{\"schema\":\"atlas.compiler_timing_facts_probe.v0\",\"instances\":[";
  for (int i = 1; i < argc; i++) {
    const OpInfo *op = findOp(argv[i]);
    if (!op) {
      std::cerr << "unknown mnemonic: " << argv[i] << '\n';
      return 2;
    }
    Instr in = instruction(op);
    RegValues regs = unknownRegs();
    regs[8] = 0;
    std::cout << (i > 1 ? "," : "") << "{\"operation\":\"" << op->name
              << "\",\"rd\":" << in.rd << ",\"rs1\":" << in.rs1
              << ",\"rs2\":" << in.rs2 << ",\"footprint\":";
    footprint(footprintOf(in, regs));
    std::cout << '}';
  }
  std::cout << "]}\n";
}
