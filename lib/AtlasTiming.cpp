#include "Atlas/AtlasTiming.h"
#include "Atlas/AtlasTimingProvider.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <limits>

using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

// Per-unit timing in cycles after issue. defUseLatency is the earliest issue gap for a consumer of the whole result;
// a CycleWindow is an inclusive [from, to] reservation of a unit, port or bank.
constexpr int kDefaultUnitCapacity = 1;

struct CycleWindow {
  int from, to;
};
struct RowStream {
  int firstAge;
  int secondAge = 0;
  int step = 1;
};

// ALU and SELI write the scalar port in the issue cycle.
struct ScalarTiming {
  int defUseLatency;
  CycleWindow resourceOccupancy;
};
constexpr ScalarTiming kScalarTiming = {1, {0, 0}};

// Scalar load and store; only loads use the load path and the scalar write port.
struct ScalarLsuTiming {
  int sourceReadAge, memoryAccessAge;
  CycleWindow bank;
  int loadDefUseLatency;
  CycleWindow loadPath, loadWriteback;
};
constexpr ScalarLsuTiming kScalarLsuTiming = {0, 1, {1, 1}, 4, {0, 2}, {3, 3}};

// VLOAD and VSTORE. resultWriteAge is the first destination row.
struct VloadTiming {
  int memoryAccessAge, resultWriteAge, defUseLatency;
  CycleWindow path, bank;
};
struct VstoreTiming {
  int sourceReadAge, memoryAccessAge, sourceReadRelease;
  CycleWindow path, bank;
};
constexpr VloadTiming kVloadTiming = {1, 3, 35, {0, 34}, {1, 32}};
constexpr VstoreTiming kVstoreTiming = {1, 3, 34, {0, 34}, {3, 34}};

// MXU0 is the systolic array, MXU1 the inner-product trees. A matmul writes accumulator rows from accumulatorRow0 - 1
// for 32 cycles and holds its in-flight slot until the last of those writes. weightOverwriteAfterCompute is the
// sequencer's earliest weight push into a slot that a matmul is still reading.
struct MxuTiming {
  struct {
    int accumulatorRow0, poppedMreg;
  } defUseLatency;
  int weightReadAge, sourceReadRelease, computeCapacity, weightOverwriteAfterCompute;
  struct {
    CycleWindow sourcePorts, resultPorts, pushStream, accumulatorRead;
  } resourceOccupancy;
};
constexpr MxuTiming kMxuTiming[] = {
    // {accumulatorRow0, poppedMreg}, weightReadAge, sourceReadRelease, computeCapacity, weightOverwriteAfterCompute,
    // {sourcePorts, resultPorts, pushStream, accumulatorRead}
    {{64, 33}, 1, 32, 3, 63, {{0, 31}, {0, 31}, {1, 32}, {0, 31}}}, // MXU0
    {{4, 33}, 1, 32, 2, 32, {{0, 31}, {0, 31}, {1, 32}, {0, 31}}},  // MXU1
};
constexpr int kMxuSourceReadAge = 0, kMxuPushResultAge = 1, kMxuPopResultAge = 1;
// MXU1's sequencer also blocks a weight or accumulator slot while a push into it is still writing.
constexpr int kMxu1WeightUseAfterPush = 32, kMxu1AccumulatorUseAfterPush = 33;

// VPU. An instruction holds its issue slot through its last destination write, one cycle before defUseLatency
// (npu_model vpu.py issue_cycles).
struct VpuTiming {
  RowStream sourceReads, resultWrites;
  int sourceReadRelease, defUseLatency;
};
constexpr VpuTiming kVpuElementwiseTiming = {{0, 32}, {2, 34}, 63, 66};
constexpr VpuTiming kVpuPackTiming = {{0, 32}, {3, 0, 2}, 63, 66};
constexpr VpuTiming kVpuUnpackTiming = {{0}, {3, 35}, 31, 67};
constexpr VpuTiming kVpuRowSumTiming = {{0, 0}, {7, 7}, 31, 39};
constexpr VpuTiming kVpuRowMinMaxTiming = {{0, 0}, {2, 2}, 31, 34};
constexpr VpuTiming kVpuColReduceTiming = {{0, 32}, {66, 98}, 127, 130};
constexpr VpuTiming kVpuLoadImmPairTiming = {{0}, {1, 33}, 0, 65};
constexpr VpuTiming kVpuLoadImmSingleTiming = {{0}, {1}, 0, 33};
constexpr int kVpuColReducePassInterval = 64;
constexpr unsigned kVpuIssueSlots = 2;

// XLU transpose: 32-row fill, transpose, 32-row result stream.
struct XluTiming {
  int sourceReadAge, resultWriteAge, sourceReadRelease, defUseLatency;
  CycleWindow resourceOccupancy;
};
constexpr XluTiming kXluTiming = {1, 34, 33, 66, {0, 65}};

// DMA cost calibrated on npu_model 337e9f2, not an RTL latency. For 1 KiB the formula gives 781 cycles to load and
// 724 to store; npu_model measured 781 and 723.
struct DmaTiming {
  long long coreMHz, readPeakMBps, writePeakMBps, startCycles, beatBytes;
};
constexpr DmaTiming kDmaTiming = {500, 682, 738, 30, 32};

} // namespace

static std::vector<OpInfo> buildTable() {
  std::vector<OpInfo> t;
  auto add = [&](const std::string &name, const char *operands, OpClass c,
                 Engine e) { t.push_back(OpInfo{name, operands, c, e}); };
  for (const char *n :
       {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"})
    add(n, "xd x1 x2", OpClass::Alu, Engine::Scalar);
  for (const char *n : {"addi", "slti", "sltiu", "xori", "ori", "andi", "slli",
                        "srli", "srai"})
    add(n, "xd x1 i", OpClass::Alu, Engine::Scalar);
  add("lui", "xd i", OpClass::Alu, Engine::Scalar);
  add("auipc", "xd i", OpClass::Alu, Engine::Scalar);
  for (const char *n :
       {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"})
    add(n, "xd x1 i", OpClass::Csr, Engine::Scalar);
  for (const char *n : {"beq", "bne", "blt", "bge", "bltu", "bgeu"})
    add(n, "x1 x2 t", OpClass::Branch, Engine::Scalar);
  add("jal", "xd t", OpClass::Jump, Engine::Scalar);
  add("jalr", "xd x1 i", OpClass::Jump, Engine::Scalar);
  add("delay", "i", OpClass::Delay, Engine::Scalar);
  add("ecall", "", OpClass::Halt, Engine::Scalar);
  add("ebreak", "", OpClass::Halt, Engine::Scalar);
  add("fence", "", OpClass::Fence, Engine::Scalar);
  add("seli", "ed i", OpClass::ScaleImm, Engine::Scalar);

  for (const char *n : {"lb", "lh", "lw", "lbu", "lhu"})
    add(n, "xd @", OpClass::ScalarLoad, Engine::Lsu);
  for (const char *n : {"sb", "sh", "sw"})
    add(n, "x2 @", OpClass::ScalarStore, Engine::Lsu);
  add("seld", "ed @", OpClass::ScaleLoad, Engine::Lsu);
  add("vload", "md @", OpClass::VLoad, Engine::Lsu);
  add("vstore", "md @", OpClass::VStore, Engine::Lsu);

  for (int m = 0; m < 2; m++) {
    Engine e = m == 0 ? Engine::Mxu0 : Engine::Mxu1;
    std::string s = ".mxu" + std::to_string(m);
    add("vmatpush.weight" + s, "wd m1", OpClass::WeightPush, e);
    add("vmatpush.acc.fp8" + s, "ad m1", OpClass::AccPushFp8, e);
    add("vmatpush.acc.bf16" + s, "ad m1", OpClass::AccPushBf16, e);
    add("vmatpop.fp8.acc" + s, "md a2 e1", OpClass::PopFp8, e);
    add("vmatpop.bf16.acc" + s, "md a2", OpClass::PopBf16, e);
    add("vmatmul" + s, "ad m1 w2", OpClass::MatMul, e);
    add("vmatmul.acc" + s, "ad m1 w2", OpClass::MatMulAcc, e);
    for (int i = static_cast<int>(t.size()) - 7;
         i < static_cast<int>(t.size()); i++)
      t[i].mxu = m;
  }

  for (const char *n : {"vadd.bf16", "vsub.bf16", "vmul.bf16",
                        "vminimum.bf16", "vmaximum.bf16"}) {
    add(n, "md m1 m2", OpClass::VpuElementwise, Engine::Vpu);
    t.back().twoInput = true;
  }
  for (const char *n : {"vmov", "vrecip.bf16", "vexp.bf16", "vexp2.bf16",
                        "vrelu.bf16", "vsin.bf16", "vcos.bf16", "vtanh.bf16",
                        "vlog2.bf16", "vsqrt.bf16", "vsquare.bf16",
                        "vcube.bf16"})
    add(n, "md m1", OpClass::VpuElementwise, Engine::Vpu);
  for (const char *n :
       {"vredsum.row.bf16", "vredmin.row.bf16", "vredmax.row.bf16"})
    add(n, "md m1", OpClass::VpuRowReduce, Engine::Vpu);
  for (const char *n : {"vredsum.bf16", "vredmin.bf16", "vredmax.bf16"})
    add(n, "md m1", OpClass::VpuColReduce, Engine::Vpu);
  add("vpack.bf16.fp8", "md m2 e1", OpClass::VpuPack, Engine::Vpu);
  add("vunpack.fp8.bf16", "md m2 e1", OpClass::VpuUnpack, Engine::Vpu);
  for (const char *n : {"vli.all", "vli.row"})
    add(n, "md i", OpClass::VpuLoadImmPair, Engine::Vpu);
  for (const char *n : {"vli.col", "vli.one"})
    add(n, "md i", OpClass::VpuLoadImmSingle, Engine::Vpu);
  add("vtrpose.xlu", "md m1", OpClass::Transpose, Engine::Xlu);

  for (int ch = 0; ch < kAtlasDMAChannels; ch++) {
    std::string s = ".ch" + std::to_string(ch);
    add("dma.load" + s, "xd x1 x2", OpClass::DmaLoad, Engine::Dma);
    add("dma.store" + s, "xd x1 x2", OpClass::DmaStore, Engine::Dma);
    add("dma.config" + s, "x1", OpClass::DmaConfig, Engine::Dma);
    add("dma.wait" + s, "", OpClass::DmaWait, Engine::Dma);
    for (int i = static_cast<int>(t.size()) - 4;
         i < static_cast<int>(t.size()); i++)
      t[i].channel = ch;
  }
  return t;
}

const std::vector<OpInfo> &mlir::atlas::timing::machineOperations() {
  static const std::vector<OpInfo> table = buildTable();
  return table;
}

const OpInfo *mlir::atlas::timing::findOp(const std::string &name) {
  static const std::map<std::string, const OpInfo *> byName = [] {
    std::map<std::string, const OpInfo *> m;
    for (const OpInfo &op : machineOperations())
      m[op.name] = &op;
    return m;
  }();
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : it->second;
}

std::string mlir::atlas::timing::vpuMnemonic(const std::string &kind) {
  if (kind == "mov")
    return "vmov";
  if (kind == "min")
    return "vminimum.bf16";
  if (kind == "max")
    return "vmaximum.bf16";
  // row_max -> vredmax.row.bf16, col_sum -> vredsum.bf16.
  if (kind.rfind("row_", 0) == 0)
    return "vred" + kind.substr(4) + ".row.bf16";
  if (kind.rfind("col_", 0) == 0)
    return "vred" + kind.substr(4) + ".bf16";
  return "v" + kind + ".bf16";
}

bool mlir::atlas::timing::isControlFlow(const OpInfo &op) {
  return op.opClass == OpClass::Branch || op.opClass == OpClass::Jump;
}

long long mlir::atlas::timing::signExtend(long long value, int bits) {
  value &= (1LL << bits) - 1;
  if (value & (1LL << (bits - 1)))
    value -= 1LL << bits;
  return value;
}

static std::vector<std::string> tokenize(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur += c;
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

bool mlir::atlas::timing::hasOperand(const OpInfo &op,
                                     const std::string &token) {
  std::vector<std::string> spec = tokenize(op.operands);
  return std::find(spec.begin(), spec.end(), token) != spec.end();
}

int mlir::atlas::timing::naturalGap(const Instr &in,
                                           const TimingTables &tables) {
  return in.op->opClass == OpClass::Delay
             ? tables.delayBaseGap + int(in.imm & tables.delayImmediateMask)
             : tables.ordinaryIssueGap;
}

int mlir::atlas::timing::naturalGap(const Instr &in) {
  if (in.op->opClass != OpClass::Delay)
    return 1;
  return 1 + static_cast<int>(in.imm & 0xFFF);
}

RegValues mlir::atlas::timing::unknownRegs() {
  RegValues r;
  r[0] = 0;
  return r;
}

// RV32 semantics of npu_model's isa_definition.py, masked to 32 bits.
std::optional<uint32_t> mlir::atlas::timing::aluResult(const Instr &in,
                                                       const RegValues &regs) {
  const std::string &n = in.op->name;
  if (n == "lui")
    return static_cast<uint32_t>((in.imm & 0xFFFFF) << 12);
  if (!regs[in.rs1])
    return std::nullopt;
  uint32_t a = *regs[in.rs1];
  int32_t sa = static_cast<int32_t>(a);
  if (hasOperand(*in.op, "i")) {
    uint32_t imm = static_cast<uint32_t>(signExtend(in.imm, 12));
    int shamt = static_cast<int>(in.imm & 0x1F);
    return llvm::StringSwitch<std::optional<uint32_t>>(n)
        .Case("addi", a + imm)
        .Case("slti", sa < static_cast<int32_t>(imm) ? 1u : 0u)
        .Case("sltiu", a < imm ? 1u : 0u)
        .Case("xori", a ^ imm)
        .Case("ori", a | imm)
        .Case("andi", a & imm)
        .Case("slli", a << shamt)
        .Case("srli", a >> shamt)
        .Case("srai", static_cast<uint32_t>(sa >> shamt))
        .Default(std::nullopt);
  }
  if (!regs[in.rs2])
    return std::nullopt;
  uint32_t b = *regs[in.rs2];
  int32_t sb = static_cast<int32_t>(b);
  return llvm::StringSwitch<std::optional<uint32_t>>(n)
      .Case("add", a + b)
      .Case("sub", a - b)
      .Case("sll", a << (b & 0x1F))
      .Case("slt", sa < sb ? 1u : 0u)
      .Case("sltu", a < b ? 1u : 0u)
      .Case("xor", a ^ b)
      .Case("srl", a >> (b & 0x1F))
      .Case("sra", static_cast<uint32_t>(sa >> (b & 0x1F)))
      .Case("or", a | b)
      .Case("and", a & b)
      .Default(std::nullopt);
}

void mlir::atlas::timing::applyScalar(const Instr &in, RegValues &regs) {
  OpClass c = in.op->opClass;
  bool writesX = c == OpClass::Alu || c == OpClass::Csr ||
                 c == OpClass::Jump || c == OpClass::ScalarLoad;
  if (!writesX || in.rd == 0)
    return;
  regs[in.rd] = c == OpClass::Alu ? aluResult(in, regs) : std::nullopt;
}

int mlir::atlas::timing::dmaTransferCycles(long long bytes, bool store) {
  const DmaTiming &t = kDmaTiming;
  long long beats = std::max(1LL, (bytes + t.beatBytes - 1) / t.beatBytes);
  long long peak = store ? t.writePeakMBps : t.readPeakMBps;
  return static_cast<int>(t.startCycles + (beats * t.beatBytes * t.coreMHz + peak - 1) / peak);
}

int mlir::atlas::timing::unitCapacity(Unit u, int index,
                                        const TimingTables &tables) {
  return u == Unit::MxuCompute ? tables.mxu[index == 0 ? 0 : 1].computeCapacity
                               : kDefaultUnitCapacity;
}

int mlir::atlas::timing::unitCapacity(Unit u, int index) {
  if (u == Unit::MxuCompute)
    return kMxuTiming[index == 0 ? 0 : 1].computeCapacity;
  return kDefaultUnitCapacity;
}

const char *mlir::atlas::timing::unitName(Unit u) {
  static const char *names[] = {
      "scalar load path", "scalar write port", "VLOAD path", "VSTORE path",
      "VMEM bank", "XLU", "MXU port", "MXU in-flight matmuls",
      "accumulator read", "accumulator write", "weight push stream",
      "accumulator push stream"};
  return names[static_cast<int>(u)];
}

const char *mlir::atlas::timing::edgeKindName(EdgeKind k) {
  static const char *names[] = {"RAW", "WAR", "WAW", "rule", "order"};
  return names[static_cast<int>(k)];
}

bool mlir::atlas::timing::isBarrier(const Instr &in) {
  OpClass c = in.op->opClass;
  return c == OpClass::Csr || c == OpClass::Fence;
}

bool mlir::atlas::timing::vpuUsesBothSlots(const OpInfo &op,
                                            const TimingTables &tables) {
  return std::find(tables.vpuDouble.begin(), tables.vpuDouble.end(), op.name) !=
         tables.vpuDouble.end();
}

bool mlir::atlas::timing::vpuCanOverlap(const OpInfo &a, const OpInfo &b,
                                         const TimingTables &tables) {
  if (vpuUsesBothSlots(a, tables) || vpuUsesBothSlots(b, tables) ||
      a.name == b.name)
    return false;
  for (const auto &group : tables.vpuLogicGroups)
    if (std::find(group.begin(), group.end(), a.name) != group.end() &&
        std::find(group.begin(), group.end(), b.name) != group.end())
      return false;
  return true;
}

bool mlir::atlas::timing::vpuUsesBothSlots(const OpInfo &op) {
  return op.twoInput || op.opClass == OpClass::VpuRowReduce;
}

// Lane-logic groups from npu_model's vpu.py; members cannot overlap.
static int vpuLogicGroup(const std::string &name) {
  static const std::vector<std::vector<std::string>> groups = {
      {"vadd.bf16", "vsub.bf16", "vredsum.row.bf16"},
      {"vexp.bf16", "vexp2.bf16"},
      {"vsin.bf16", "vcos.bf16"},
      {"vsquare.bf16", "vcube.bf16"},
      {"vmaximum.bf16", "vredmax.bf16"},
      {"vminimum.bf16", "vredmin.bf16"},
      {"vli.all", "vli.row", "vli.col", "vli.one"},
  };
  for (size_t g = 0; g < groups.size(); g++)
    for (const std::string &n : groups[g])
      if (n == name)
        return static_cast<int>(g);
  return -1;
}

bool mlir::atlas::timing::vpuCanOverlap(const OpInfo &a, const OpInfo &b) {
  if (vpuUsesBothSlots(a) || vpuUsesBothSlots(b))
    return false;
  if (a.name == b.name)
    return false;
  int ga = vpuLogicGroup(a.name);
  return ga < 0 || ga != vpuLogicGroup(b.name);
}

namespace {

struct Builder {
  Footprint f;
  const TimingTables *tables = nullptr;
  const OperationTiming *timing = nullptr;
  const Instr &instruction;

  Builder(const Instr &in, const TimingTables *tables)
      : tables(tables), instruction(in) {
    if (!tables)
      return;
    auto found = tables->operations.find(in.op->name);
    if (found != tables->operations.end())
      timing = &found->second;
    else if (in.op->engine != Engine::Scalar && in.op->engine != Engine::Dma)
      f.error = in.op->name + ": missing timing operation";
  }
  int eventCount(const char *key) const {
    switch (instruction.op->opClass) {
    case OpClass::VpuElementwise: return 64;
    case OpClass::VpuPack: return key[0] == 'r' ? 64 : 32;
    case OpClass::VpuUnpack: return key[0] == 'r' ? 32 : 64;
    case OpClass::VpuColReduce: return key[0] == 'r' ? 128 : 64;
    case OpClass::VpuLoadImmPair: return 64;
    case OpClass::ScalarLoad:
    case OpClass::ScaleLoad:
    case OpClass::ScalarStore: return 1;
    default: return 32;
    }
  }
  const TimingEvent *event(const char *key) {
    if (!tables)
      return nullptr;
    if (timing) {
      auto found = timing->events.find(key);
      if (found != timing->events.end()) {
        const TimingEvent &value = found->second;
        if (value.count != eventCount(key) || value.step <= 0 ||
            value.lastAge != value.firstAge + (value.count - 1) * value.step) {
          f.error = instruction.op->name + ": invalid timing event " + key;
          return nullptr;
        }
        return &value;
      }
    }
    f.error = instruction.op->name + ": missing timing event " + key;
    return nullptr;
  }
  int age(const char *key, int fallback, int offset = 0) {
    if (!tables)
      return fallback;
    auto value = event(key);
    return value ? value->firstAge + offset * value->step : 0;
  }
  int readAge() const {
    return tables ? (isControlFlow(*instruction.op) ? tables->controlReadAge
                                                  : tables->scalarReadAge)
                  : 0;
  }
  TimingWindow release(const char *key) {
    if (timing) {
      auto found = timing->releases.find(key);
      if (found != timing->releases.end())
        return found->second;
    }
    f.error = instruction.op->name + ": missing timing release " + key;
    return {};
  }
  void stream(Access &access, const char *key, int offset) {
    if (!tables)
      return;
    if (auto value = event(key)) {
      access.age = value->firstAge + offset * value->step;
      access.step = value->step;
    }
  }
  const MxuRules &mxuRules() const {
    return tables->mxu[instruction.op->mxu];
  }
  void mxuPorts(std::vector<int> ports, bool single = false) {
    int mxu = instruction.op->mxu;
    TimingWindow window{0, 31};
    if (tables) {
      const auto &rules = mxuRules();
      switch (instruction.op->opClass) {
      case OpClass::MatMul:
      case OpClass::MatMulAcc: ports = rules.computePorts; break;
      case OpClass::AccPushBf16: ports = rules.pushBf16Ports; break;
      case OpClass::PopFp8: ports = rules.popFp8Ports; break;
      case OpClass::PopBf16: ports = rules.popBf16Ports; break;
      default: ports = rules.singlePushPriority; break;
      }
      bool pop = instruction.op->opClass == OpClass::PopFp8 ||
                 instruction.op->opClass == OpClass::PopBf16;
      window = pop ? rules.resultPorts : rules.sourcePorts;
    }
    if (single) {
      if (ports.empty()) {
        f.error = "missing MXU single-push ports";
        return;
      }
      f.holds.push_back({Unit::MxuPort, mxu * 4 + ports.front(),
                         window.from, window.to,
                         ports.size() > 1 ? mxu * 4 + ports[1] : -1});
    } else {
      for (int port : ports)
        f.holds.push_back({Unit::MxuPort, mxu * 4 + port,
                           window.from, window.to});
    }
  }

  void x(int reg, bool write, int age, bool atCompletion = false) {
    if (reg == 0)
      return;
    Access a{Res::XReg, write, reg, 1, age, 1};
    a.atCompletion = atCompletion;
    f.accesses.push_back(a);
  }
  void e(int reg, bool write, int age) {
    f.accesses.push_back({Res::EReg, write, reg, 1, age, 1});
  }
  void mrows(int reg, bool write, int age, int step = 1,
             const char *key = "", int offset = 0) {
    if (reg < 0 || reg > 63) {
      f.error = "register pair runs past m63";
      return;
    }
    Access access{Res::MReg, write, reg * 32, 32, age, step};
    stream(access, key, offset);
    f.accesses.push_back(access);
  }
  void mxuRows(Res res, int mxu, int index, bool write, int age,
               const char *key) {
    Access access{res, write, (mxu * 2 + index) * 32, 32, age, 1};
    stream(access, key, 0);
    f.accesses.push_back(access);
  }
  void hold(Unit u, int index, int from, int to, int alt = -1) {
    if (tables) {
      TimingWindow window{from, to};
      switch (u) {
      case Unit::ScalarLoad:
        window = {release("unit").from, tables->scalarLoadGap - 1}; break;
      case Unit::ScalarWriteback:
        if (instruction.op->opClass == OpClass::ScalarLoad ||
            instruction.op->opClass == OpClass::ScaleLoad) {
          int write = age(instruction.op->opClass == OpClass::ScaleLoad
                              ? "scale_write" : "load_write", 0);
          window = {write, write};
        } else
          window = {tables->scalarWriteAge, tables->scalarWriteAge};
        break;
      case Unit::VloadPath:
      case Unit::VstorePath:
      case Unit::Xlu: window = release("unit"); break;
      case Unit::VmemBank:
        window = release(instruction.op->opClass == OpClass::ScalarStore ||
                                 instruction.op->opClass == OpClass::VStore
                             ? "vmem_write" : "vmem_read");
        break;
      case Unit::MxuCompute: window = release("unit"); break;
      case Unit::MxuWeightStream:
      case Unit::MxuAccStream:
      case Unit::MxuAccRead: {
        const char *key = u == Unit::MxuWeightStream ? "weight_write" :
                          u == Unit::MxuAccStream ? "acc_load" :
                          instruction.op->opClass == OpClass::MatMulAcc
                              ? "acc_read" : "acc_store";
        auto value = event(key);
        window = value ? TimingWindow{value->firstAge, value->lastAge}
                       : TimingWindow{};
        break;
      }
      case Unit::MxuAccWrite: {
        auto value = event(instruction.op->opClass == OpClass::MatMul ||
                                   instruction.op->opClass == OpClass::MatMulAcc
                               ? "acc_write" : "acc_load");
        window = value ? TimingWindow{value->firstAge, value->lastAge}
                       : TimingWindow{};
        break;
      }
      case Unit::MxuPort: break;
      }
      from = window.from;
      to = window.to;
    }
    f.holds.push_back({u, index, from, to, alt});
  }
  void hold(Unit u, int index, CycleWindow w, int alt = -1) { hold(u, index, w.from, w.to, alt); }
  void needEven(int reg, const char *what) {
    if (reg & 1)
      f.error = std::string(what) + " must name an even register (m" +
                std::to_string(reg) + ")";
  }

  void vmem(std::optional<long long> byteAddr, std::optional<long long> bytes,
            bool write, int age, int step, bool atCompletion = false,
            const char *key = "") {
    if (bytes && *bytes <= 0)
      return;
    Access a{Res::Vmem, write, 0, 1, age, step};
    a.atCompletion = atCompletion;
    if (!atCompletion)
      stream(a, key, 0);
    if (!byteAddr || !bytes) {
      a.anywhere = true;
      a.count = bytes ? static_cast<int>((*bytes + kLineBytes - 1) / kLineBytes)
                       : 1;
    } else {
      long long firstLine = *byteAddr / kLineBytes;
      long long lastLine = (*byteAddr + *bytes - 1) / kLineBytes;
      a.first = static_cast<int>(firstLine);
      a.count = static_cast<int>(lastLine - firstLine + 1);
      if (*byteAddr + *bytes > kAtlasVmemBytes)
        f.error = "VMEM access past the end of VMEM";
    }
    f.accesses.push_back(a);
  }
  void bankHold(std::optional<long long> byteAddr, int from, int to) {
    if (byteAddr) {
      hold(Unit::VmemBank, static_cast<int>(*byteAddr / kVmemBankBytes), from,
           to);
    } else {
      for (int b = 0; b < kVmemBanks; b++)
        hold(Unit::VmemBank, b, from, to);
    }
  }
  void bankHold(std::optional<long long> byteAddr, CycleWindow w) { bankHold(byteAddr, w.from, w.to); }
};

std::optional<long long> reg(const RegValues &regs, int r) {
  if (!regs[r])
    return std::nullopt;
  return static_cast<long long>(*regs[r]);
}

// ScalarCore keeps only the VMEM bits of a scalar byte address.
std::optional<long long> scalarAddress(const Instr &in, const RegValues &regs) {
  auto base = reg(regs, in.rs1);
  if (!base)
    return std::nullopt;
  return (*base + signExtend(in.imm, 12)) & 0x1FFFFF;
}

// VLS bases are word addresses; the immediate counts 32 words and the LSU
// drops the three low word bits to get a 32-byte line.
std::optional<long long> vectorAddress(const Instr &in, const RegValues &regs) {
  auto base = reg(regs, in.rs1);
  if (!base)
    return std::nullopt;
  long long alu = *base + signExtend(in.imm, 12) * 32;
  return ((alu >> 3) & 0xFFFF) * kLineBytes;
}

void scalarRegs(Builder &b, const Instr &in) {
  const OpInfo &op = *in.op;
  // csrr*i encodes an immediate in rs1.
  bool immediateCsr = op.opClass == OpClass::Csr && op.name.back() == 'i';
  if (hasOperand(op, "x1") && !immediateCsr)
    b.x(in.rs1, false, b.readAge());
  if (hasOperand(op, "x2"))
    b.x(in.rs2, false, b.readAge());
  if (hasOperand(op, "xd")) {
    b.x(in.rd, true, b.tables ? b.tables->scalarWriteAge
                             : kScalarTiming.defUseLatency - 1);
    if (in.rd != 0)
      b.hold(Unit::ScalarWriteback, 0, kScalarTiming.resourceOccupancy);
  }
}

} // namespace

ScalarAccess mlir::atlas::timing::scalarAccess(const Instr &in, const RegValues &regs) {
  bool store = in.op->opClass == OpClass::ScalarStore;
  int bytes = !store ? 4 : in.op->name == "sb" ? 1 : in.op->name == "sh" ? 2 : 4;
  std::optional<long long> first;
  if (auto address = scalarAddress(in, regs))
    first = *address & ~static_cast<long long>(bytes - 1);
  return {first, bytes, store};
}

Footprint mlir::atlas::timing::footprintOf(const Instr &in,
                                           const RegValues &regs,
                                           const TimingTables *tables) {
  ::Builder b(in, tables);
  const OpInfo &op = *in.op;
  if (in.release && op.opClass != OpClass::Csr) {
    b.f.error = "atlas.complete is only valid on a CSR instruction";
    return b.f;
  }
  int mxu = op.mxu;
  const MxuTiming &mt = kMxuTiming[mxu == 0 ? 0 : 1];
  const auto &occ = mt.resourceOccupancy;

  switch (op.opClass) {
  case OpClass::Alu:
  case OpClass::Csr:
  case OpClass::Branch:
  case OpClass::Jump:
    scalarRegs(b, in);
    break;
  case OpClass::Delay:
  case OpClass::Halt:
  case OpClass::Fence:
  case OpClass::DmaWait:
    break;
  case OpClass::ScaleImm:
    b.e(in.rd, true, tables ? tables->scalarWriteAge
                           : kScalarTiming.defUseLatency - 1);
    b.hold(Unit::ScalarWriteback, 0, kScalarTiming.resourceOccupancy);
    break;

  case OpClass::ScalarLoad:
  case OpClass::ScaleLoad: {
    const ScalarLsuTiming &t = kScalarLsuTiming;
    ScalarAccess access = scalarAccess(in, regs);
    b.x(in.rs1, false, b.readAge());
    b.vmem(access.first, access.bytes, false, t.memoryAccessAge, 1, false,
           "vmem_read");
    if (op.opClass == OpClass::ScaleLoad)
      b.e(in.rd, true, b.age("scale_write", t.loadDefUseLatency - 1));
    else
      b.x(in.rd, true, b.age("load_write", t.loadDefUseLatency - 1));
    b.hold(Unit::ScalarLoad, 0, t.loadPath);
    b.hold(Unit::ScalarWriteback, 0, t.loadWriteback);
    b.bankHold(access.first, t.bank);
    break;
  }
  case OpClass::ScalarStore: {
    const ScalarLsuTiming &t = kScalarLsuTiming;
    ScalarAccess access = scalarAccess(in, regs);
    b.x(in.rs1, false, b.readAge());
    b.x(in.rs2, false, b.readAge());
    b.vmem(access.first, access.bytes, true, t.memoryAccessAge, 1, false,
           "vmem_write");
    b.bankHold(access.first, t.bank);
    break;
  }
  case OpClass::VLoad:
  case OpClass::VStore: {
    bool load = op.opClass == OpClass::VLoad;
    auto addr = vectorAddress(in, regs);
    if (addr && (*addr % kAtlasRegisterBytes != 0 ||
                 *addr / kVmemBankBytes != (*addr + kAtlasRegisterBytes - 1) / kVmemBankBytes))
      b.f.error =
          "vload/vstore address must be 1 KiB aligned inside one VMEM bank";
    b.x(in.rs1, false, b.readAge());
    if (load) {
      const VloadTiming &t = kVloadTiming;
      b.vmem(addr, kAtlasRegisterBytes, false, t.memoryAccessAge, 1, false,
             "vmem_read");
      b.mrows(in.rd, true, t.resultWriteAge, 1, "mreg_write");
      b.f.mregWrites = {in.rd};
      b.f.writeRelease = t.defUseLatency - 1;
      b.f.writeDuringRead = true;
      b.hold(Unit::VloadPath, 0, t.path);
      b.bankHold(addr, t.bank);
    } else {
      const VstoreTiming &t = kVstoreTiming;
      b.mrows(in.rd, false, t.sourceReadAge, 1, "mreg_read");
      b.vmem(addr, kAtlasRegisterBytes, true, t.memoryAccessAge, 1, false,
             "vmem_write");
      b.f.mregReads = {in.rd};
      b.f.readRelease = b.f.writeRelease = t.sourceReadRelease;
      b.hold(Unit::VstorePath, 0, t.path);
      b.bankHold(addr, t.bank);
    }
    break;
  }

  case OpClass::WeightPush:
    b.mrows(in.rs1, false, kMxuSourceReadAge, 1, "mreg_read1");
    b.mxuRows(Res::Weight, mxu, in.rd, true, kMxuPushResultAge,
              "weight_write");
    b.f.mregReads = {in.rs1};
    b.f.readRelease = b.f.writeRelease = mt.sourceReadRelease;
    b.mxuPorts({1, 0}, true);
    b.hold(Unit::MxuWeightStream, mxu, occ.pushStream);
    break;
  case OpClass::AccPushFp8:
  case OpClass::AccPushBf16: {
    bool pair = op.opClass == OpClass::AccPushBf16;
    b.mrows(in.rs1, false, kMxuSourceReadAge, 1,
            pair ? "mreg_read0" : "mreg_read1");
    if (pair)
      b.mrows(in.rs1 + 1, false, kMxuSourceReadAge, 1, "mreg_read1");
    b.mxuRows(Res::Acc, mxu, in.rd, true, kMxuPushResultAge, "acc_load");
    b.f.mregReads = pair ? std::vector<int>{in.rs1, in.rs1 + 1}
                         : std::vector<int>{in.rs1};
    b.f.readRelease = b.f.writeRelease = mt.sourceReadRelease;
    if (pair) {
      b.mxuPorts({0, 1});
    } else {
      b.mxuPorts({1, 0}, true);
    }
    b.hold(Unit::MxuAccStream, mxu, occ.pushStream);
    b.hold(Unit::MxuAccWrite, mxu * 2 + in.rd, occ.pushStream);
    break;
  }
  case OpClass::PopFp8:
  case OpClass::PopBf16: {
    bool pair = op.opClass == OpClass::PopBf16;
    b.mxuRows(Res::Acc, mxu, in.rs2, false, kMxuSourceReadAge, "acc_store");
    if (!pair)
      b.e(in.rs1, false, b.readAge());
    b.mrows(in.rd, true, kMxuPopResultAge, 1, "mreg_write0");
    if (pair)
      b.mrows(in.rd + 1, true, kMxuPopResultAge, 1, "mreg_write1");
    b.f.mregWrites = pair ? std::vector<int>{in.rd, in.rd + 1}
                          : std::vector<int>{in.rd};
    b.f.readRelease = mt.sourceReadRelease;
    b.f.writeRelease = mt.defUseLatency.poppedMreg - 1;
    b.mxuPorts(pair ? std::vector<int>{2, 3} : std::vector<int>{2});
    b.hold(Unit::MxuAccRead, mxu * 2 + in.rs2, occ.accumulatorRead);
    break;
  }

  case OpClass::MatMul:
  case OpClass::MatMulAcc: {
    CycleWindow accumulatorWrite = {mt.defUseLatency.accumulatorRow0 - 1, mt.defUseLatency.accumulatorRow0 + 30};
    b.mrows(in.rs1, false, kMxuSourceReadAge, 1, "mreg_read0");
    if (tables) {
      const auto &rules = tables->mxu[mxu];
      for (const TimingEvent &event : {rules.weightFirst, rules.weightLast})
        b.f.accesses.push_back({Res::Weight, false, (mxu * 2 + in.rs2) * 32,
                                32, event.firstAge, event.step});
    } else
      b.mxuRows(Res::Weight, mxu, in.rs2, false, mt.weightReadAge, "");
    if (op.opClass == OpClass::MatMulAcc)
      b.mxuRows(Res::Acc, mxu, in.rd, false, kMxuSourceReadAge, "acc_read");
    b.mxuRows(Res::Acc, mxu, in.rd, true, accumulatorWrite.from, "acc_write");
    b.f.mregReads = {in.rs1};
    b.f.readRelease = b.f.writeRelease = mt.sourceReadRelease;
    b.mxuPorts({0});
    b.hold(Unit::MxuCompute, mxu, 0, accumulatorWrite.to);
    if (op.opClass == OpClass::MatMulAcc)
      b.hold(Unit::MxuAccRead, mxu * 2 + in.rd, occ.accumulatorRead);
    b.hold(Unit::MxuAccWrite, mxu * 2 + in.rd, accumulatorWrite);
    break;
  }

  case OpClass::VpuElementwise: {
    const VpuTiming &t = kVpuElementwiseTiming;
    std::vector<int> sources = {in.rs1};
    if (op.twoInput)
      sources.push_back(in.rs2);
    for (size_t source = 0; source < sources.size(); ++source) {
      int s = sources[source];
      const char *key = source == 0 ? "read0" : "read1";
      b.needEven(s, "BF16 source");
      b.mrows(s, false, t.sourceReads.firstAge, 1, key);
      b.mrows(s + 1, false, t.sourceReads.secondAge, 1, key, 32);
      b.f.mregReads.push_back(s);
      b.f.mregReads.push_back(s + 1);
    }
    b.needEven(in.rd, "BF16 destination");
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.mrows(in.rd + 1, true, t.resultWrites.secondAge, 1, "write0", 32);
    b.f.mregWrites = {in.rd, in.rd + 1};
    b.f.readRelease = t.sourceReadRelease;
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }

  case OpClass::VpuPack: {
    const VpuTiming &t = kVpuPackTiming;
    b.needEven(in.rs2, "BF16 source");
    b.mrows(in.rs2, false, t.sourceReads.firstAge, 1, "read0");
    b.mrows(in.rs2 + 1, false, t.sourceReads.secondAge, 1, "read0", 32);
    b.e(in.rs1, false, b.readAge());
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.f.mregReads = {in.rs2, in.rs2 + 1};
    b.f.mregWrites = {in.rd};
    b.f.readRelease = t.sourceReadRelease;
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::VpuUnpack: {
    const VpuTiming &t = kVpuUnpackTiming;
    b.needEven(in.rd, "BF16 destination");
    b.mrows(in.rs2, false, t.sourceReads.firstAge, 1, "read0");
    b.e(in.rs1, false, b.readAge());
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.mrows(in.rd + 1, true, t.resultWrites.secondAge, 1, "write0", 32);
    b.f.mregReads = {in.rs2};
    b.f.mregWrites = {in.rd, in.rd + 1};
    b.f.readRelease = t.sourceReadRelease;
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::VpuRowReduce: {
    const VpuTiming &t = op.name == "vredsum.row.bf16" ? kVpuRowSumTiming : kVpuRowMinMaxTiming;
    b.needEven(in.rs1, "BF16 source");
    b.needEven(in.rd, "BF16 destination");
    b.mrows(in.rs1, false, t.sourceReads.firstAge, 1, "read0");
    b.mrows(in.rs1 + 1, false, t.sourceReads.secondAge, 1, "read1");
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.mrows(in.rd + 1, true, t.resultWrites.secondAge, 1, "write1", 0);
    b.f.mregReads = {in.rs1, in.rs1 + 1};
    b.f.mregWrites = {in.rd, in.rd + 1};
    b.f.readRelease = t.sourceReadRelease;
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::VpuColReduce: {
    const VpuTiming &t = kVpuColReduceTiming;
    b.needEven(in.rs1, "BF16 source");
    b.needEven(in.rd, "BF16 destination");
    for (int pass = 0; pass < 2; pass++) {
      b.mrows(in.rs1, false,
              pass * kVpuColReducePassInterval + t.sourceReads.firstAge,
              1, "read0", pass * 64);
      b.mrows(in.rs1 + 1, false,
              pass * kVpuColReducePassInterval + t.sourceReads.secondAge,
              1, "read0", pass * 64 + 32);
    }
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.mrows(in.rd + 1, true, t.resultWrites.secondAge, 1, "write0", 32);
    b.f.mregReads = {in.rs1, in.rs1 + 1};
    b.f.mregWrites = {in.rd, in.rd + 1};
    b.f.readRelease = t.sourceReadRelease;
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::VpuLoadImmPair: {
    const VpuTiming &t = kVpuLoadImmPairTiming;
    b.needEven(in.rd, "BF16 destination");
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.mrows(in.rd + 1, true, t.resultWrites.secondAge, 1, "write0", 32);
    b.f.mregWrites = {in.rd, in.rd + 1};
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::VpuLoadImmSingle: {
    const VpuTiming &t = kVpuLoadImmSingleTiming;
    b.mrows(in.rd, true, t.resultWrites.firstAge, t.resultWrites.step,
            "write0");
    b.f.mregWrites = {in.rd};
    b.f.writeRelease = t.defUseLatency - 1;
    break;
  }
  case OpClass::Transpose:
    b.mrows(in.rs1, false, kXluTiming.sourceReadAge, 1, "read");
    b.mrows(in.rd, true, kXluTiming.resultWriteAge, 1, "write");
    b.f.mregReads = {in.rs1};
    b.f.mregWrites = {in.rd};
    b.f.readRelease = kXluTiming.sourceReadRelease;
    b.f.writeRelease = kXluTiming.defUseLatency - 1;
    b.hold(Unit::Xlu, 0, kXluTiming.resourceOccupancy);
    break;
  case OpClass::DmaLoad:
  case OpClass::DmaStore: {
    // ScalarCore captures XRF operands and the shared upper address at launch;
    // VMEM contents remain owned by the transfer until its matching wait.
    bool load = op.opClass == OpClass::DmaLoad;
    int vmemReg = load ? in.rd : in.rs1;
    // AtlasCore.scala takes the DMA VMEM address in words, as for VLS bases.
    auto addr = reg(regs, vmemReg);
    if (addr)
      *addr *= 4;
    auto bytes = reg(regs, in.rs2);
    b.x(in.rd, false, b.readAge());
    b.x(in.rs1, false, b.readAge());
    b.x(in.rs2, false, b.readAge());
    Access base{Res::DmaBase, false, 0, 1, b.readAge(), 1};
    b.f.accesses.push_back(base);
    b.vmem(addr, bytes, load, 0, 0, true);
    if (addr && (*addr % 32 != 0))
      b.f.error = "DMA VMEM address must be 32-byte aligned";
    if (tables) {
      if (!bytes) {
        b.f.error = "model DMA requires a known transfer size";
      } else if (!tables->dmaEstimate) {
        b.f.error = "model timing lacks DMA estimates";
      } else {
        auto estimate = tables->dmaEstimate(*bytes, !load);
        if (!estimate.error.empty())
          b.f.error = estimate.error;
        else
          b.f.dmaCycles = estimate.value;
      }
    } else {
      b.f.dmaCycles = dmaTransferCycles(bytes ? *bytes : kAtlasRegisterBytes,
                                        !load);
    }
    break;
  }
  case OpClass::DmaConfig: {
    b.x(in.rs1, false, b.readAge());
    Access base{Res::DmaBase, true, 0, 1,
                tables ? tables->scalarWriteAge : 0, 1};
    b.f.accesses.push_back(base);
    break;
  }
  }

  Footprint &f = b.f;
  if (b.timing) {
    f.readRelease = f.writeRelease = 0;
    if (!f.mregReads.empty())
      f.readRelease = b.release("mreg_read").to;
    if (!f.mregWrites.empty())
      f.writeRelease = b.release("mreg_write").to;
    if (op.opClass == OpClass::VStore)
      f.writeRelease = b.release("vmem_write").to;
    for (const Access &access : f.accesses)
      if (access.write && (access.res == Res::Acc || access.res == Res::Weight))
        f.writeRelease = std::max(f.writeRelease, access.lastAge());
  }
  if (op.engine == Engine::Vpu)
    f.vpuLive = b.timing ? b.release("unit").to : f.writeRelease;
  f.doneAge = std::max(f.readRelease, f.writeRelease);
  if (b.timing)
    f.doneAge = std::max({f.doneAge, b.timing->firstFreeAge - 1,
                          b.release("unit").to});
  if (b.timing)
    for (const auto &[key, event] : b.timing->events)
      f.doneAge = std::max(f.doneAge, event.lastAge);
  for (const Access &a : f.accesses)
    if (!a.atCompletion)
      f.doneAge = std::max(f.doneAge, a.lastAge());
  for (const Hold &h : f.holds)
    f.doneAge = std::max(f.doneAge, h.to);
  return f;
}

Footprint mlir::atlas::timing::footprintOf(const Instr &in,
                                        const RegValues &regs,
                                        const TimingTables &tables) {
  return footprintOf(in, regs, &tables);
}

static std::string elementName(Res res, int element) {
  switch (res) {
  case Res::XReg:
    return "x" + std::to_string(element);
  case Res::EReg:
    return "e" + std::to_string(element);
  case Res::MReg:
    return "m" + std::to_string(element / 32);
  case Res::Acc:
    return "mxu" + std::to_string(element / 64) + ".acc" +
           std::to_string(element / 32 % 2);
  case Res::Weight:
    return "mxu" + std::to_string(element / 64) + ".w" +
           std::to_string(element / 32 % 2);
  case Res::Vmem: {
    char buf[32];
    snprintf(buf, sizeof buf, "VMEM 0x%x", element * kLineBytes);
    return buf;
  }
  case Res::DmaBase:
    return "dma.base";
  }
  return "?";
}

// INT_MIN when the two accesses cannot touch the same element.
static int dataDistance(const Access &x, const Access &y, int &element) {
  auto ageY = [&](long long e) {
    return y.atCompletion ? 0LL : y.age + (e - y.first) * y.step;
  };
  auto ageX = [&](long long e) { return x.age + (e - x.first) * x.step; };
  if (x.anywhere || y.anywhere) {
    element = x.anywhere ? y.first : x.first;
    int minY = y.atCompletion ? 0 : y.age;
    return x.lastAge() - minY + 1;
  }
  long long lo = std::max(x.first, y.first);
  long long hi = std::min(x.first + x.count - 1, y.first + y.count - 1);
  if (lo > hi)
    return INT_MIN;
  element = static_cast<int>(lo);
  return static_cast<int>(std::max(ageX(lo) - ageY(lo), ageX(hi) - ageY(hi))) +
         1;
}

static bool contains(const std::vector<int> &v, int x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

Dependence mlir::atlas::timing::dependence(const Instr &a, const Footprint &fa,
                                         const Instr &b, const Footprint &fb,
                                         const TimingTables *tables) {
  Dependence best;
  auto consider = [&](int d, EdgeKind kind, const std::string &why) {
    if (d > best.distance)
      best = {d, kind, why};
  };
  int issueGap = tables ? tables->ordinaryIssueGap : 1;
  const OpInfo &A = *a.op;
  const OpInfo &B = *b.op;

  if (isBarrier(a))
    consider(issueGap, EdgeKind::Order, A.name + " is a barrier");
  if (isBarrier(b))
    consider(issueGap, EdgeKind::Order, B.name + " is a barrier");
  if (b.release)
    consider(fa.doneAge + 1, EdgeKind::Order,
             "atlas.complete waits for prior fixed-latency work to complete");

  if (A.engine == Engine::Dma && B.engine == Engine::Dma) {
    bool aWait = A.opClass == OpClass::DmaWait,
         bWait = B.opClass == OpClass::DmaWait;
    if (!aWait && !bWait)
      consider(issueGap, EdgeKind::Order, "DMA queue order");
    else if (A.channel == B.channel)
      consider(issueGap, EdgeKind::Order, "DMA channel " + std::to_string(A.channel));
  }

  for (const Access &x : fa.accesses) {
    if (x.atCompletion)
      continue;
    for (const Access &y : fb.accesses) {
      if (x.res != y.res || (!x.write && !y.write))
        continue;
      int element = 0;
      int d = dataDistance(x, y, element);
      if (d == INT_MIN)
        continue;
      EdgeKind kind = x.write && y.write ? EdgeKind::WAW
                      : x.write          ? EdgeKind::RAW
                                         : EdgeKind::WAR;
      consider(std::max(d, issueGap), kind,
               std::string(edgeKindName(kind)) + " on " +
                   elementName(x.res, element));
    }
  }

  for (int r : fa.mregWrites) {
    if (contains(fb.mregReads, r))
      consider(fa.writeRelease + 1, EdgeKind::RAW,
               "m" + std::to_string(r) + " reserved for writing until age " +
                   std::to_string(fa.writeRelease));
    if (contains(fb.mregWrites, r))
      consider(fa.writeRelease + 1, EdgeKind::WAW,
               "m" + std::to_string(r) + " reserved for writing until age " +
                   std::to_string(fa.writeRelease));
  }
  if (!fb.writeDuringRead)
    for (int r : fa.mregReads)
      if (contains(fb.mregWrites, r))
        consider(fa.readRelease + 1, EdgeKind::WAR,
                 "m" + std::to_string(r) + " reserved for reading until age " +
                     std::to_string(fa.readRelease));

  // npu_model mxu.py sequencer rules.
  if (A.mxu >= 0 && A.mxu == B.mxu) {
    int m = A.mxu;
    const MxuTiming &mt = kMxuTiming[m];
    std::string mx = "MXU" + std::to_string(m) + ": ";
    auto isCompute = [](const OpInfo &o) {
      return o.opClass == OpClass::MatMul || o.opClass == OpClass::MatMulAcc;
    };
    auto isAccPush = [](const OpInfo &o) {
      return o.opClass == OpClass::AccPushFp8 ||
             o.opClass == OpClass::AccPushBf16;
    };
    auto accOf = [&](const Instr &in) {
      const OpInfo &o = *in.op;
      if (isCompute(o) || isAccPush(o))
        return in.rd;
      if (o.opClass == OpClass::PopBf16 || o.opClass == OpClass::PopFp8)
        return in.rs2;
      return -1;
    };
    auto slotOf = [&](const Instr &in) {
      if (isCompute(*in.op))
        return in.rs2;
      if (in.op->opClass == OpClass::WeightPush)
        return in.rd;
      return -1;
    };
    bool bWeight = B.opClass == OpClass::WeightPush;
    if (tables && isAccPush(A) && accOf(a) == accOf(b) &&
        (B.opClass == OpClass::PopFp8 || B.opClass == OpClass::PopBf16 ||
         B.opClass == OpClass::MatMulAcc))
      consider(tables->mxu[m].accumulatorForwardGap, EdgeKind::Rule,
               mx + "accumulator push forwarding");
    if (isCompute(A) && !bWeight && accOf(a) == accOf(b))
      consider(tables ? [&] {
                 for (const Access &access : fa.accesses)
                   if (access.res == Res::Acc && access.write)
                     return access.age + 1;
                 return 0;
               }() : mt.defUseLatency.accumulatorRow0, EdgeKind::Rule,
               mx + "wait until row 0 of the matmul on acc" +
                   std::to_string(accOf(a)) + " is written");
    if (m == 0 && isCompute(A) && bWeight && slotOf(a) == slotOf(b))
      consider(tables ? tables->mxu[m].weightOverwriteAfterCompute
                      : mt.weightOverwriteAfterCompute, EdgeKind::Rule,
               mx + "weight slot still feeding the systolic array");
    if (m == 1) {
      bool aWeight = A.opClass == OpClass::WeightPush;
      if (aWeight && (bWeight || isCompute(B)) && slotOf(a) == slotOf(b))
        consider(tables ? tables->mxu[m].weightUseAfterPush
                        : kMxu1WeightUseAfterPush, EdgeKind::Rule,
                 mx + "weight push still active on w" +
                     std::to_string(slotOf(a)));
      if (isCompute(A) && bWeight && slotOf(a) == slotOf(b))
        consider(tables ? tables->mxu[m].weightOverwriteAfterCompute
                      : mt.weightOverwriteAfterCompute, EdgeKind::Rule,
                 mx + "matmul still reading w" + std::to_string(slotOf(a)));
      if (isAccPush(A) && (isCompute(B) || B.opClass == OpClass::AccPushFp8) &&
          accOf(a) == accOf(b))
        consider(tables ? tables->mxu[m].accumulatorUseAfterPush
                        : kMxu1AccumulatorUseAfterPush, EdgeKind::Rule,
                 mx + "accumulator push still active on acc" +
                     std::to_string(accOf(a)));
    }
  }
  return best;
}

Dependence mlir::atlas::timing::dependence(const Instr &a, const Footprint &fa,
                                         const Instr &b, const Footprint &fb,
                                         const TimingTables &tables) {
  return dependence(a, fa, b, fb, &tables);
}

static bool overlaps(const Access &x, const Access &y) {
  if (x.res != y.res)
    return false;
  if (x.anywhere || y.anywhere)
    return true;
  return x.first < y.first + y.count && y.first < x.first + x.count;
}

bool mlir::atlas::timing::conflictsAtCompletion(const Footprint &dma,
                                                const Footprint &f,
                                                EdgeKind &kind) {
  for (const Access &x : dma.accesses) {
    if (!x.atCompletion)
      continue;
    for (const Access &y : f.accesses) {
      // The model forbids queued VMEM overlap, even between stores.
      bool queuedVmem = x.res == Res::Vmem && y.atCompletion;
      if (!overlaps(x, y) || (!x.write && !y.write && !queuedVmem))
        continue;
      if (x.res == Res::DmaBase && y.atCompletion)
        continue;
      kind = x.write && y.write ? EdgeKind::WAW
             : x.write          ? EdgeKind::RAW
             : y.write          ? EdgeKind::WAR
                                : EdgeKind::Order;
      return true;
    }
  }
  return false;
}

namespace {
// Adds an edge, or raises the distance of an existing edge between the nodes.
struct EdgeSet {
  DepGraph &g;
  std::map<std::pair<int, int>, int> index;

  void add(int from, int to, int distance, EdgeKind kind,
           const std::string &reason) {
    auto it = index.find({from, to});
    if (it == index.end()) {
      index[{from, to}] = static_cast<int>(g.edges.size());
      g.edges.push_back({from, to, distance, kind, reason});
    } else if (distance > g.edges[it->second].distance) {
      g.edges[it->second] = {from, to, distance, kind, reason};
    }
  }
};
} // namespace

TimingRuleResult<DepGraph> mlir::atlas::timing::buildGraph(const std::vector<Instr> &instrs,
                                         const RegValues &entry,
                                         const TimingProvider &p,
                                         const IncomingDma *incomingDma) {
  std::string missing = validateTimingProvider(p);
  if (!missing.empty()) return {{}, missing};
  int issueGap = p.frontendSpacing.ordinaryGap;
  DepGraph g;
  g.nodes = instrs;
  int n = static_cast<int>(instrs.size());
  RegValues regs = entry;
  for (const Instr &in : instrs) {
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty()) return {{}, f.error};
    if (f.doneAge < 0 || f.doneAge == std::numeric_limits<int>::max())
      return {{}, "timing provider completion age exceeds positive cycle domain"};
    g.footprints.push_back(std::move(f));
    applyScalar(in, regs);
  }

  EdgeSet edges{g, {}};
  for (int b = 0; b < n; b++)
    for (int a = 0; a < b; a++) {
      auto d = p.dependence(g.nodes[a], g.footprints[a], g.nodes[b], g.footprints[b]);
      if (!d.error.empty()) return {{}, d.error};
      if (d.value.distance < 0)
        return {{}, "timing provider returned a negative dependence distance"};
      if (d.value.distance > 0)
        edges.add(a, b, d.value.distance, d.value.kind, d.value.reason);
    }

  // A DMA's completion is known only once its dma.wait issues, so later
  // conflicting accesses wait for that dma.wait.
  for (int d = 0; d < n; d++) {
    const OpInfo &op = *g.nodes[d].op;
    if (op.opClass != OpClass::DmaLoad && op.opClass != OpClass::DmaStore)
      continue;
    int wait = -1;
    for (int k = d + 1; k < n && wait < 0; k++)
      if (g.nodes[k].op->opClass == OpClass::DmaWait &&
          g.nodes[k].op->channel == op.channel)
        wait = k;
    for (int k = d + 1; k < n; k++) {
      if (k == wait)
        continue;
      auto conflict = p.dmaConflict(g.footprints[d], g.footprints[k]);
      if (!conflict.error.empty()) return {{}, conflict.error};
      if (!conflict.value.conflict)
        continue;
      EdgeKind kind = conflict.value.kind;
      if (wait >= 0 && wait < k)
        edges.add(wait, k, issueGap, kind,
                  std::string(edgeKindName(kind)) + " with " + op.name +
                      " (done once the wait issues)");
      else
        edges.add(d, k, issueGap, EdgeKind::Order,
                  op.name + " may still be in flight (no dma.wait in between)");
    }
  }

  // A wait for a transfer from an earlier block guards conflicting accesses
  // and channel reuse.
  for (int w = 0; w < n; w++) {
    const OpInfo &op = *g.nodes[w].op;
    if (op.opClass != OpClass::DmaWait)
      continue;
    bool local = false;
    for (int d = 0; d < w; d++)
      if ((g.nodes[d].op->opClass == OpClass::DmaLoad ||
           g.nodes[d].op->opClass == OpClass::DmaStore) &&
          g.nodes[d].op->channel == op.channel)
        local = true;
    if (local)
      continue;
    for (int k = w + 1; k < n; k++) {
      bool guarded = g.nodes[k].op->opClass == OpClass::DmaLoad ||
                     g.nodes[k].op->opClass == OpClass::DmaStore;
      if (incomingDma) {
        guarded = guarded && g.nodes[k].op->channel == op.channel;
        for (const Footprint &dma : (*incomingDma)[op.channel]) {
          auto conflict = p.dmaConflict(dma, g.footprints[k]);
          if (!conflict.error.empty()) return {{}, conflict.error};
          guarded |= conflict.value.conflict;
        }
      } else {
        // The transfer latched its registers at launch, so only the VMEM it
        // moves at completion may conflict.
        for (const Access &a : g.footprints[k].accesses)
          if (a.res == Res::Vmem)
            guarded = true;
      }
      if (guarded)
        edges.add(w, k, issueGap, EdgeKind::Order,
                  op.name + " guards a transfer started in an earlier block");
    }
  }

  g.in.assign(n, {});
  g.out.assign(n, {});
  for (int e = 0; e < static_cast<int>(g.edges.size()); e++) {
    g.out[g.edges[e].from].push_back(e);
    g.in[g.edges[e].to].push_back(e);
  }
  return {std::move(g), {}};
}

TimingRuleResult<std::vector<int64_t>> mlir::atlas::timing::criticalHeights(
    const DepGraph &g, const TimingProvider &p) {
  int n = static_cast<int>(g.nodes.size());
  std::vector<int64_t> height(n, 0);
  for (int i = n - 1; i >= 0; i--) {
    height[i] = int64_t(g.footprints[i].doneAge) + 1;
    for (int e : g.out[i]) {
      const Edge &ed = g.edges[e];
      int64_t d = ed.distance;
      // A dma.wait holds the frontend until the transfer completes.
      const Instr &to = g.nodes[ed.to];
      if (to.op->opClass == OpClass::DmaWait &&
          to.op->channel == g.nodes[i].op->channel) {
        if (p.dmaProposal) {
          auto proposal = p.dmaProposal(g.footprints[i].dmaCycles, 0, std::nullopt);
          if (!proposal.error.empty()) return {{}, proposal.error};
          d = std::max(d, int64_t(proposal.value.waitIssue));
        } else if (g.footprints[i].dmaCycles > 0)
          return {{}, "timing provider lacks DMA proposal policy"};
      }
      height[i] = std::max(height[i], d + height[ed.to]);
    }
  }
  return {std::move(height), {}};
}

static int unitKey(Unit u, int index) {
  return static_cast<int>(u) * 64 + index;
}

bool ReservationTable::unitFree(Unit u, int index, int from, int to) const {
  auto it = units_.find(unitKey(u, index));
  if (it == units_.end())
    return true;
  int cap = tables_ ? unitCapacity(u, index, *tables_)
                    : unitCapacity(u, index);
  for (auto c = it->second.lower_bound(from);
       c != it->second.end() && c->first <= to; ++c)
    if (c->second >= cap)
      return false;
  return true;
}

int ReservationTable::chooseIndex(const Hold &h, int cycle) const {
  if (unitFree(h.unit, h.index, cycle + h.from, cycle + h.to))
    return h.index;
  if (h.alt >= 0 && unitFree(h.unit, h.alt, cycle + h.from, cycle + h.to))
    return h.alt;
  return -1;
}

std::vector<ReservationTable::PortRequest>
ReservationTable::portRequests(const Instr &in, const Footprint &f,
                               int cycle) const {
  std::vector<PortRequest> out;
  for (const Access &a : f.accesses) {
    if (a.res != Res::MReg)
      continue;
    for (int i = 0; i < a.count; i++) {
      int element = a.first + i;
      int reg = element / 32, row = element % 32;
      bool shareable = !a.write && in.op->engine == Engine::Vpu;
      out.push_back({(reg % 32) * 2 + (a.write ? 1 : 0),
                     cycle + a.age + i * a.step, reg, row, shareable});
    }
  }
  return out;
}

std::string ReservationTable::conflict(const Instr &in, const Footprint &f,
                                       int cycle) const {
  for (const Hold &h : f.holds)
    if (chooseIndex(h, cycle) < 0)
      return std::string(unitName(h.unit)) + " " + std::to_string(h.index) +
             " busy";

  if (f.vpuLive > 0) {
    auto it = vpu_.find(cycle);
    if (it != vpu_.end()) {
      if (it->second.size() >=
          (tables_ ? tables_->vpuIssueSlots : kVpuIssueSlots))
        return "both VPU slots busy";
      for (const OpInfo *other : it->second)
        if (!(tables_ ? vpuCanOverlap(*other, *in.op, *tables_)
                      : vpuCanOverlap(*other, *in.op)))
          return "VPU busy with " + other->name;
    }
  }

  std::vector<PortRequest> requests = portRequests(in, f, cycle);
  for (size_t i = 0; i < requests.size(); i++) {
    const PortRequest &p = requests[i];
    int bank = p.key / 2;
    auto shares = [&](const PortUse &u) {
      return u.shareable && p.shareable && u.reg == p.reg && u.row == p.row;
    };
    auto same = ports_.find(p.key);
    if (same != ports_.end()) {
      auto u = same->second.find(p.cycle);
      if (u != same->second.end() && !shares(u->second))
        return "MREG bank " + std::to_string(bank) + " port busy (m" +
               std::to_string(p.reg) + ")";
    }
    auto other = ports_.find(p.key ^ 1);
    if (other != ports_.end()) {
      auto u = other->second.find(p.cycle);
      if (u != other->second.end() &&
          (u->second.reg < 0 || u->second.reg == p.reg) &&
          (u->second.row < 0 || u->second.row == p.row))
        return "same-row read/write on m" + std::to_string(p.reg);
    }
    for (size_t j = 0; j < i; j++) {
      const PortRequest &q = requests[j];
      if (q.cycle != p.cycle)
        continue;
      if (q.key == p.key &&
          !(q.shareable && p.shareable && q.reg == p.reg && q.row == p.row))
        return "instruction needs MREG bank " + std::to_string(bank) +
               " twice in one cycle";
      if ((q.key ^ 1) == p.key && q.reg == p.reg && q.row == p.row)
        return "instruction reads and writes the same MREG row in one cycle";
    }
  }
  return "";
}

void ReservationTable::reserve(const Instr &in, const Footprint &f,
                               int cycle) {
  for (const Hold &h : f.holds) {
    int index = chooseIndex(h, cycle);
    if (index < 0)
      index = h.index;
    int key = unitKey(h.unit, index);
    for (int c = cycle + h.from; c <= cycle + h.to; c++)
      units_[key][c]++;
    unitWindows_.push_back({key, cycle + h.from, cycle + h.to});
  }
  for (int age = 0; age < f.vpuLive; age++)
    vpu_[cycle + age].push_back(in.op);

  std::map<int, PortWindow> windows;
  for (const PortRequest &p : portRequests(in, f, cycle)) {
    ports_[p.key][p.cycle] = {p.reg, p.row, p.shareable};
    auto it = windows.find(p.key);
    if (it == windows.end()) {
      windows[p.key] = {p.key, p.cycle, p.cycle};
    } else {
      it->second.from = std::min(it->second.from, p.cycle);
      it->second.to = std::max(it->second.to, p.cycle);
    }
  }
  for (auto &[key, w] : windows)
    portWindows_.push_back(w);
}

void ReservationTable::extendForWait(int cycle) {
  for (UnitWindow &w : unitWindows_) {
    if (w.to < cycle || w.from <= cycle)
      continue;
    for (int c = cycle; c < w.from; c++)
      units_[w.key][c]++;
    w.from = cycle;
  }
  for (PortWindow &w : portWindows_) {
    if (w.to < cycle)
      continue;
    for (int c = cycle; c <= w.to; c++)
      ports_[w.key][c] = PortUse{};
    w.from = std::min(w.from, cycle);
  }
}
