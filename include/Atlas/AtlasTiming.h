#ifndef ATLAS_TIMING_H
#define ATLAS_TIMING_H

// npu_model rtl-match timing rules, ported from atlas-compiler-experiments
// 3ae2b5d (src/core) with its names. DMA VMEM addresses count words, as in
// Atlas RTL. An age counts cycles after an instruction issues.

#include "Atlas/AtlasTarget.h"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::atlas::timing {

enum class Engine { Scalar, Lsu, Mxu0, Mxu1, Vpu, Xlu, Dma };

enum class OpClass {
  Alu, Csr, Branch, Jump, Delay, Halt, Fence,
  ScalarLoad, ScalarStore, ScaleLoad, ScaleImm,
  VLoad, VStore,
  WeightPush, AccPushFp8, AccPushBf16, PopFp8, PopBf16, MatMul, MatMulAcc,
  VpuElementwise, VpuPack, VpuUnpack, VpuRowReduce, VpuColReduce,
  VpuLoadImmPair, VpuLoadImmSingle,
  Transpose,
  DmaLoad, DmaStore, DmaConfig, DmaWait,
};

// `operands` lists assembly operand kinds, such as "xd x1 i" for addi.
struct OpInfo {
  std::string name;
  std::string operands;
  OpClass opClass;
  Engine engine;
  int mxu = -1;
  int channel = -1;
  bool twoInput = false;
};

const OpInfo *findOp(const std::string &name);
// The BF16 VPU instruction for an atlas.vpu_unary, vpu_binary or vpu_reduce
// kind, such as "exp2", "max" or "row_sum".
std::string vpuMnemonic(const std::string &kind);
bool hasOperand(const OpInfo &op, const std::string &token);
bool isControlFlow(const OpInfo &op);

struct Instr {
  const OpInfo *op = nullptr;
  int rd = 0, rs1 = 0, rs2 = 0;
  long long imm = 0;
  bool release = false;
};

// Compiler-owned proof from a decided 32-iteration PACK replay. Each address
// is the unrounded effective byte address at the same iteration ordinal.
class InstructionTimingContext {
public:
  bool matches(const Instr &in) const;
  bool sameRegion(const InstructionTimingContext &other) const { return region == other.region; }
  const std::array<uint32_t, 32> &scalarAddresses() const { return addresses; }

private:
  InstructionTimingContext(Instr instruction, std::shared_ptr<const uint8_t> region, std::array<uint32_t, 32> addresses)
      : instruction(instruction), region(std::move(region)), addresses(addresses) {}
  Instr instruction;
  std::shared_ptr<const uint8_t> region;
  std::array<uint32_t, 32> addresses;
  friend struct AtlasPackLoopTimingProofBuilder;
};
using TimingContext = std::shared_ptr<const InstructionTimingContext>;

long long signExtend(long long value, int bits);
int naturalGap(const Instr &in);

using RegValues = std::array<std::optional<uint32_t>, 32>;
RegValues unknownRegs();
std::optional<uint32_t> aluResult(const Instr &in, const RegValues &regs);
void applyScalar(const Instr &in, RegValues &regs);
// The VMEM bytes a scalar load or store touches: a load reads its whole
// aligned word, a store writes its naturally aligned bytes. `first` is known
// only when the base register is.
struct ScalarAccess {
  std::optional<long long> first;
  int bytes;
  bool write;
};
ScalarAccess scalarAccess(const Instr &in, const RegValues &regs);

const int kVmemBankBytes = 256 * 1024;
const int kVmemBanks = kAtlasVmemBytes / kVmemBankBytes;
const int kLineBytes = 32;

enum class Res { XReg, EReg, MReg, Acc, Weight, Vmem, DmaBase };

// Elements are register numbers, reg*32+row for MReg, (mxu*2+slot)*32+row for
// Acc and Weight, and 32-byte lines for Vmem; element i is touched at
// age + i * step.
struct Access {
  Res res;
  bool write;
  int first, count;
  int age, step;
  bool anywhere = false;
  bool atCompletion = false;
  int lastAge() const { return age + (count - 1) * step; }
};

enum class Unit {
  ScalarLoad,
  ScalarWriteback,
  VloadPath,
  VstorePath,
  VmemBank,
  Xlu,
  MxuPort, // mxu*4 + port: 0 and 1 read, 2 and 3 write
  MxuCompute,
  MxuAccRead,
  MxuAccWrite,
  MxuWeightStream,
  MxuAccStream,
};

struct Hold {
  Unit unit;
  int index;
  int from, to;
  int alt = -1;
};

struct Footprint {
  std::vector<Access> accesses;
  std::vector<Hold> holds;
  std::vector<int> mregReads, mregWrites;
  int readRelease = 0, writeRelease = 0;
  bool writeDuringRead = false;
  int vpuLive = 0;
  int doneAge = 0;
  int dmaCycles = 0;
  std::string error;
};

Footprint footprintOf(const Instr &in, const RegValues &regs);
// Supplied timing rules; the built-in footprintOf provider is the unqualified
// npu-model rtl-match model, not an independently qualified RTL contract.
using FootprintResolver = std::function<Footprint(const Instr &, const RegValues &)>;

enum class EdgeKind { RAW, WAR, WAW, Rule, Order };
const char *edgeKindName(EdgeKind k);

struct Dependence {
  int distance = 0;
  EdgeKind kind = EdgeKind::Order;
  std::string reason;
};

// Cycles b must issue after a, which precedes it; DMA completion is excluded.
Dependence dependence(const Instr &a, const Footprint &fa, const Instr &b,
                      const Footprint &fb);
bool conflictsAtCompletion(const Footprint &dma, const Footprint &other,
                           EdgeKind &kind);

// b issues at least `distance` cycles after a.
struct Edge {
  int from, to;
  int distance;
  EdgeKind kind;
  std::string reason;
};

// Nodes are one block's instructions in program order, so edges point forward.
struct DepGraph {
  std::vector<Instr> nodes;
  std::vector<Footprint> footprints;
  std::vector<Edge> edges;
  std::vector<std::vector<int>> in, out;
};

// Pending transfers per channel at block entry; null keeps broad barriers.
using IncomingDma = std::array<std::vector<Footprint>, kAtlasDMAChannels>;

bool isBarrier(const Instr &in);
bool vpuCanOverlap(const OpInfo &a, const OpInfo &b);
bool vpuUsesBothSlots(const OpInfo &op);
int unitCapacity(Unit u, int index);
const char *unitName(Unit u);
// Cycles from launching one DMA transfer on an idle engine to its completed
// wait, by npu_model main's DMA model: the beat-level engine against its
// default DRAM, the bandwidth-latency curves measured on VCS EE290SimConfig
// (ee290sim_vcs_probe.json). Beats stream at the curve's peak bandwidth after a
// fixed start; transfers queued behind one another pay only the beats.
int dmaTransferCycles(long long bytes, bool store);

class ReservationTable {
public:
  // Empty if `in` can issue at `cycle`, otherwise the reason it cannot.
  std::string conflict(const Instr &in, const Footprint &f, int cycle) const;
  void reserve(const Instr &in, const Footprint &f, int cycle);
  // A wait releases at an unknown cycle, so reservations extend back to it.
  void extendForWait(int cycle);

private:
  struct PortUse {
    int reg = -1, row = -1;
    bool shareable = false;
  };
  struct UnitWindow { int key, from, to; };
  struct PortWindow { int key, from, to; };
  struct PortRequest { int key, cycle, reg, row; bool shareable; };

  std::map<int, std::map<int, int>> units_;
  std::map<int, std::map<int, PortUse>> ports_;
  std::map<int, std::vector<const OpInfo *>> vpu_;
  std::vector<UnitWindow> unitWindows_;
  std::vector<PortWindow> portWindows_;

  std::vector<PortRequest> portRequests(const Instr &in, const Footprint &f,
                                        int cycle) const;
  bool unitFree(Unit u, int index, int from, int to) const;
  int chooseIndex(const Hold &h, int cycle) const;
};

} // namespace mlir::atlas::timing

#endif
