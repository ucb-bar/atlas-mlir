#ifndef ATLAS_TIMING_H
#define ATLAS_TIMING_H

// Timing model of the Atlas core, ported from atlas-compiler-experiments
// 3ae2b5d (src/core/asm, values, machine, reservations, and depgraph's DMA
// completion check). Function and field names match that source. Its rules
// and latencies follow npu_model's rtl-match branch, not the selected RTL
// revision. The one deliberate change: DMA VMEM addresses count 32-bit words,
// as in Atlas RTL, instead of bytes (see footprintOf). "Age" is the number of
// cycles since an instruction issued (age 0 is its issue cycle).

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::atlas::timing {

enum class Engine { Scalar, Lsu, Mxu0, Mxu1, Vpu, Xlu, Dma };

// Groups of instructions that share the same timing behavior.
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

// One npu_model opcode. `operands` lists its assembly operands, one token
// each: a register kind (x, e, m, w or a for acc) followed by its field
// (d = rd, 1 = rs1, 2 = rs2), or i (immediate), t (branch target), or
// @ (imm(x rs1)).
struct OpInfo {
  std::string name;
  std::string operands;
  OpClass opClass;
  Engine engine;
  int mxu = -1;          // 0 or 1 for MXU instructions
  int channel = -1;      // DMA channel for dma.* instructions
  bool twoInput = false; // VPU binary ops read two register pairs
};

const OpInfo *findOp(const std::string &name); // nullptr if unknown
bool hasOperand(const OpInfo &op, const std::string &token);
bool isControlFlow(const OpInfo &op); // branches and jumps (one delay slot)

// One instruction in npu_model's field layout: rd/rs1/rs2 hold x, e, m, w or
// acc register numbers depending on the opcode's operand list.
struct Instr {
  const OpInfo *op = nullptr;
  int rd = 0, rs1 = 0, rs2 = 0;
  long long imm = 0;
  bool release = false; // atlas.complete: finish prior work before this CSR
};

long long signExtend(long long value, int bits);
// Cycles until the next issue when nothing stalls (1, or N+1 for delay N).
int naturalGap(const Instr &in);

// Known scalar register values at one program point (nullopt = unknown).
using RegValues = std::array<std::optional<uint32_t>, 32>;
RegValues unknownRegs(); // x0 = 0, everything else unknown
std::optional<uint32_t> aluResult(const Instr &in, const RegValues &regs);
void applyScalar(const Instr &in, RegValues &regs);

const int kVmemBytes = 1536 * 1024;
const int kVmemBankBytes = 256 * 1024;
const int kVmemBanks = kVmemBytes / kVmemBankBytes;
const int kLineBytes = 32;

// Storage an instruction reads or writes.
enum class Res { XReg, EReg, MReg, Acc, Weight, Vmem, DmaBase };

// Elements [first, first + count) of one kind of storage. Element i is touched
// at age + i * step. Elements are: register numbers for XReg/EReg,
// reg*32+row for MReg, (mxu*2+index)*32+row for Acc/Weight, and 32-byte line
// numbers for Vmem.
struct Access {
  Res res;
  bool write;
  int first, count;
  int age, step;
  bool anywhere = false;     // address unknown: may touch any element
  bool atCompletion = false; // DMA: happens when the transfer finishes
  int lastAge() const { return age + (count - 1) * step; }
};

// Hardware structures that only one (or a few) instructions may use per cycle.
enum class Unit {
  ScalarLoad,      // scalar load command/response path
  ScalarWriteback, // scalar register write port shared by S1 and load responses
  VloadPath,
  VstorePath,
  VmemBank,        // index = VMEM bank (0..5), one LSU access per cycle
  Xlu,
  MxuPort,         // index = mxu*4 + port (0,1 read ports, 2,3 write ports)
  MxuCompute,      // index = mxu, in-flight matmuls (3 on MXU0, 2 on MXU1)
  MxuAccRead,      // index = mxu*2 + acc
  MxuAccWrite,     // index = mxu*2 + acc
  MxuWeightStream, // index = mxu, one weight push writing per cycle
  MxuAccStream,    // index = mxu, one accumulator push writing per cycle
};

// A unit held from age `from` to age `to` (inclusive). If `alt` >= 0 the
// instruction may use unit index `alt` instead when `index` is busy.
struct Hold {
  Unit unit;
  int index;
  int from, to;
  int alt = -1;
};

// Everything the timing rules need to know about one instruction.
struct Footprint {
  std::vector<Access> accesses;
  std::vector<Hold> holds;
  std::vector<int> mregReads, mregWrites; // logical MREG reservations at issue
  int readRelease = 0, writeRelease = 0;  // ages at which they are dropped
  bool writeDuringRead = false; // vload may write registers others still read
  int vpuLive = 0;   // VPU: occupies a VPU slot for ages 0 .. vpuLive-1
  int doneAge = 0;   // last age at which the instruction uses any resource
  int dmaCycles = 0; // DMA commands: expected transfer time (npu_model)
  std::string error; // set when the operands are illegal
};

Footprint footprintOf(const Instr &in, const RegValues &regs);

enum class EdgeKind { RAW, WAR, WAW, Rule, Order };
const char *edgeKindName(EdgeKind k);

// Why b must wait for a, and for how many cycles.
struct Dependence {
  int distance = 0; // b issues at least this many cycles after a; 0 = free
  EdgeKind kind = EdgeKind::Order;
  std::string reason;
};

// Minimum issue distance between a and b, where a comes first in program
// order. DMA accesses that happen at completion are not included.
Dependence dependence(const Instr &a, const Footprint &fa, const Instr &b,
                      const Footprint &fb);

// Does `other` conflict with what DMA command `dma` does at completion?
bool conflictsAtCompletion(const Footprint &dma, const Footprint &other,
                           EdgeKind &kind);

bool isBarrier(const Instr &in); // frontend barriers: nothing moves across
bool vpuCanOverlap(const OpInfo &a, const OpInfo &b);
bool vpuUsesBothSlots(const OpInfo &op);
int unitCapacity(Unit u, int index);
const char *unitName(Unit u);
int dmaTransferCycles(long long bytes); // npu_model's DMA engine estimate

// Cycle-by-cycle record of the hardware that issued instructions occupy:
// engine paths and ports (Hold), the 32 physical MREG banks (one read and one
// write port each; mN and m(N+32) share a bank), and the two VPU issue slots.
class ReservationTable {
public:
  // Returns "" if `in` can issue at `cycle`, otherwise why it cannot.
  std::string conflict(const Instr &in, const Footprint &f, int cycle) const;
  void reserve(const Instr &in, const Footprint &f, int cycle);
  // A dma.wait stalls for an unknown time, so every resource stays reserved
  // from the wait through its last use, and port rows become unknown.
  void extendForWait(int cycle);

private:
  struct PortUse {
    int reg = -1, row = -1; // -1: unknown (reserved by extendForWait)
    bool shareable = false; // VPU reads of the same register row share a port
  };
  struct UnitWindow { int key, from, to; };
  struct PortWindow { int key, from, to; };
  struct PortRequest { int key, cycle, reg, row; bool shareable; };

  std::map<int, std::map<int, int>> units_;      // unit key -> cycle -> users
  std::map<int, std::map<int, PortUse>> ports_;  // bank*2 + isWrite -> ...
  std::map<int, std::vector<const OpInfo *>> vpu_; // cycle -> VPU slot users
  std::vector<UnitWindow> unitWindows_;
  std::vector<PortWindow> portWindows_;

  std::vector<PortRequest> portRequests(const Instr &in, const Footprint &f,
                                        int cycle) const;
  bool unitFree(Unit u, int index, int from, int to) const;
  int chooseIndex(const Hold &h, int cycle) const; // -1 if busy
};

} // namespace mlir::atlas::timing

#endif
