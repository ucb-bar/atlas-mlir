// Pre-allocation list scheduling of virtual Atlas operations. The design,
// dependence model, and failure policy are in docs/virtual-scheduler-plan.md;
// section numbers below refer to it.

#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTiming.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <climits>
#include <optional>
#include <random>

using namespace mlir;
using namespace mlir::atlas;

namespace {

//===----------------------------------------------------------------------===//
// Register classes and caps (§7.1), as in VirtualAllocationPlan::colorValues
//===----------------------------------------------------------------------===//

enum RegClass { kBF16, kFP8, kScalar, kNumClasses };
using ClassCounts = std::array<int, kNumClasses>;

std::optional<RegClass> classOf(Type type) {
  if (isa<VirtualBF16Type>(type))
    return kBF16;
  if (isa<VirtualFP8Type>(type))
    return kFP8;
  if (type.isInteger(1) || type.isInteger(32))
    return kScalar;
  return std::nullopt;
}

const char *className(int k) {
  static const char *names[] = {"BF16", "FP8", "scalar"};
  return names[k];
}

struct Caps {
  ClassCounts limit;
  bool mixedFp8;
};

Caps capsFor(func::FuncOp function) {
  bool fp8 = false, pack = false;
  function.walk([&](Operation *op) {
    pack |= isa<VirtualPackFP8Op>(op);
    for (Value result : op->getResults())
      fp8 |= isa<VirtualFP8Type>(result.getType());
  });
  return {{fp8 ? 15 : 31, 32, pack ? 9 : 17}, fp8};
}

//===----------------------------------------------------------------------===//
// Implicit resources as pseudo-registers (§5.2, E2)
//===----------------------------------------------------------------------===//

enum Pseudo : unsigned {
  kDma,     // the single explicit-DMA pending slot
  kWeight0, // current weight, MXU0 and MXU1
  kWeight1,
  kAcc0, // accumulator occupancy, MXU0 and MXU1
  kAcc1,
  kPack,    // pack VMEM scratch and relayout registers
  kBarrier, // read by every operation; written by unknown operations
  kNumPseudo
};

constexpr uint32_t bit(unsigned r) { return 1u << r; }

struct Effects {
  uint32_t reads = 0, writes = 0;
};

unsigned unitOf(Type type) {
  if (auto weight = dyn_cast<VirtualMXUWeightType>(type))
    return weight.getUnit();
  return cast<VirtualMXUAccType>(type).getUnit();
}

// An explicit table: `Pure` is not trusted, because legacy virtual_mxu_matmul
// and virtual_pack_fp8 are declared Pure but have ordering obligations.
Effects effectsOf(Operation *op) {
  Effects e;
  if (isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(op)) {
    e.reads = bit(kDma);
  } else if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
                 VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op,
                 VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op,
                 VirtualDMAWaitOp>(op)) {
    e.writes = bit(kDma);
  } else if (isa<VirtualPackFP8Op>(op)) {
    e.reads = bit(kDma);
    e.writes = bit(kPack);
  } else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
    e.writes = bit(kWeight0 + unitOf(load.getWeight().getType()));
  } else if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
    unsigned unit = unitOf(reset.getAcc().getType());
    e.reads = bit(kWeight0 + unit);
    e.writes = bit(kAcc0 + unit);
  } else if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op)) {
    unsigned unit = unitOf(acc.getAcc().getType());
    e.reads = bit(kWeight0 + unit);
    e.writes = bit(kAcc0 + unit);
  } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
    e.writes = bit(kAcc0 + unitOf(op->getResult(1).getType()));
  } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
    e.writes = bit(kAcc0 + unitOf(op->getOperand(1).getType()));
  } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
    unsigned unit = matmul.getUnit();
    e.writes = bit(kWeight0 + unit) | bit(kAcc0 + unit);
  } else if (!isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualScaleConstantOp,
                  arith::ConstantOp, arith::AddIOp, arith::CmpIOp>(op)) {
    e.writes = bit(kNumPseudo) - 1; // unknown: a full barrier
  }
  e.reads |= bit(kBarrier);
  return e;
}

//===----------------------------------------------------------------------===//
// Placeholder registers for timing (§8.2)
//===----------------------------------------------------------------------===//

// Tiles get registers in the allocator's partition (FP8 m0-m31 and BF16 pairs
// m32-m62 when mixed, else BF16 pairs m0-m62), rotating so recently defined
// values rarely share a register. Results take a register when placed; a
// probe sees the next rotation register without committing it.
class Placeholders {
public:
  explicit Placeholders(bool mixedFp8) : mixed(mixedFp8) {}

  int lookup(Value value, bool commit, llvm::DenseMap<Value, int> &overlay,
             std::array<unsigned, kNumClasses> &probeCounters) {
    auto found = regs.find(value);
    if (found != regs.end())
      return found->second;
    auto pending = overlay.find(value);
    if (pending != overlay.end())
      return pending->second;
    std::optional<RegClass> k = classOf(value.getType());
    if (!k)
      return 0;
    unsigned &counter = commit ? counters[*k] : probeCounters[*k];
    int reg = pick(*k, counter++);
    (commit ? regs : overlay)[value] = reg;
    return reg;
  }

  std::array<unsigned, kNumClasses> counters{};

private:
  int pick(RegClass k, unsigned n) const {
    if (k == kBF16)
      return mixed ? 32 + 2 * static_cast<int>(n % 16)
                   : 2 * static_cast<int>(n % 32);
    if (k == kFP8)
      return static_cast<int>(n % 32);
    return 10 + static_cast<int>(n % 17);
  }

  bool mixed;
  llvm::DenseMap<Value, int> regs;
};

//===----------------------------------------------------------------------===//
// Machine templates (§4.4), mirroring AtlasVirtualToMachine.cpp without the
// fixed diagnostic delays
//===----------------------------------------------------------------------===//

struct Template {
  SmallVector<timing::Instr, 8> instrs;
  bool pack = false; // drain, run the relayout prefix, then `instrs`
};

const FixedResourcePlacement &fixedResources() {
  static const VirtualAllocationPlan plan;
  return plan.fixed();
}

std::string vpuName(StringRef kind, bool binary) {
  if (!binary && kind == "mov")
    return "vmov";
  if (binary && kind == "min")
    return "vminimum.bf16";
  if (binary && kind == "max")
    return "vmaximum.bf16";
  return "v" + kind.str() + ".bf16";
}

class TemplateBuilder {
public:
  TemplateBuilder(Placeholders &placeholders, bool commit, uint64_t inputBase,
                  uint64_t outputBase)
      : placeholders(placeholders), commit(commit), inputBase(inputBase),
        outputBase(outputBase), probeCounters(placeholders.counters) {}

  Template build(Operation *op);
  static Template packPrefix();

private:
  int reg(Value value) {
    return placeholders.lookup(value, commit, overlay, probeCounters);
  }
  void emit(const std::string &name, int rd = 0, int rs1 = 0, int rs2 = 0,
            long long imm = 0) {
    timing::Instr in;
    in.op = timing::findOp(name);
    if (!in.op)
      in.op = timing::findOp("vmov"); // an unmodeled VPU kind; timing only
    in.rd = rd;
    in.rs1 = rs1;
    in.rs2 = rs2;
    in.imm = imm;
    t.instrs.push_back(in);
  }
  void li(int dst, uint32_t value) {
    uint32_t upper = ((static_cast<uint64_t>(value) + 0x800) >> 12) & 0xfffff;
    int32_t lower = static_cast<int32_t>(value & 0xfff);
    if (lower >= 2048)
      lower -= 4096;
    if (upper)
      emit("lui", dst, 0, 0, upper);
    emit("addi", dst, upper ? dst : 0, 0, lower);
  }
  std::string ch(const char *kind, unsigned channel) {
    return std::string(kind) + ".ch" + std::to_string(channel);
  }
  std::string mxu(const char *kind, unsigned unit) {
    return std::string(kind) + ".mxu" + std::to_string(unit);
  }
  void inputHalf(int dst, uint64_t index, unsigned half);
  void outputHalf(int src, uint64_t index, unsigned half);
  void launch(Value dram, Value size, std::optional<int> src, unsigned halves);
  void complete(std::optional<int> dst, unsigned halves);
  void compare(arith::CmpIOp cmp);

  Placeholders &placeholders;
  bool commit;
  uint64_t inputBase, outputBase;
  llvm::DenseMap<Value, int> overlay;
  std::array<unsigned, kNumClasses> probeCounters;
  Template t;
};

void TemplateBuilder::inputHalf(int dst, uint64_t index, unsigned half) {
  const FixedResourcePlacement &f = fixedResources();
  li(f.inputBaseReg, f.inputWord + index * 512 + half * 256);
  li(f.inputDramReg, static_cast<uint32_t>(inputBase + index * 2048 +
                                           half * 1024));
  emit(ch("dma.load", f.loadChannel), f.inputBaseReg, f.inputDramReg,
       f.halfSizeReg);
  emit(ch("dma.wait", f.loadChannel));
  emit("vload", dst, f.inputBaseReg);
}

void TemplateBuilder::outputHalf(int src, uint64_t index, unsigned half) {
  const FixedResourcePlacement &f = fixedResources();
  li(f.outputBaseReg, f.outputWord + index * 512 + half * 256);
  emit("vstore", src, f.outputBaseReg);
  li(f.outputDramReg, static_cast<uint32_t>(outputBase + index * 2048 +
                                            half * 1024));
  emit(ch("dma.store", f.storeChannel), f.outputDramReg, f.outputBaseReg,
       f.halfSizeReg);
  emit(ch("dma.wait", f.storeChannel));
}

void TemplateBuilder::launch(Value dram, Value size, std::optional<int> src,
                             unsigned halves) {
  const FixedResourcePlacement &f = fixedResources();
  emit("addi", f.dmaDramReg, reg(dram), 0, 0);
  emit("addi", f.dmaSizeReg, reg(size), 0, 0);
  li(f.dmaBaseReg, f.stagingWord);
  if (src) {
    for (unsigned half = 0; half < halves; ++half) {
      if (half)
        li(f.dmaBaseReg, f.stagingWord + half * 256);
      emit("vstore", *src + half, f.dmaBaseReg);
    }
    if (halves > 1)
      li(f.dmaBaseReg, f.stagingWord);
    emit(ch("dma.store", f.storeChannel), f.dmaDramReg, f.dmaBaseReg,
         f.dmaSizeReg);
  } else {
    emit(ch("dma.load", f.loadChannel), f.dmaBaseReg, f.dmaDramReg,
         f.dmaSizeReg);
  }
}

void TemplateBuilder::complete(std::optional<int> dst, unsigned halves) {
  const FixedResourcePlacement &f = fixedResources();
  emit(ch("dma.wait", dst ? f.loadChannel : f.storeChannel));
  if (!dst)
    return;
  for (unsigned half = 0; half < halves; ++half) {
    if (half)
      li(f.dmaBaseReg, f.stagingWord + half * 256);
    emit("vload", *dst + half, f.dmaBaseReg);
  }
}

void TemplateBuilder::compare(arith::CmpIOp cmp) {
  int dst = reg(cmp.getResult()), lhs = reg(cmp.getLhs()),
      rhs = reg(cmp.getRhs());
  using P = arith::CmpIPredicate;
  P p = cmp.getPredicate();
  bool swap = p == P::sgt || p == P::sle || p == P::ugt || p == P::ule;
  bool invert = p == P::sle || p == P::sge || p == P::ule || p == P::uge;
  if (p == P::eq || p == P::ne) {
    emit("xor", dst, lhs, rhs);
    if (p == P::eq)
      emit("sltiu", dst, dst, 0, 1);
    else
      emit("sltu", dst, 0, dst);
    return;
  }
  bool isSigned = p == P::slt || p == P::sgt || p == P::sle || p == P::sge;
  emit(isSigned ? "slt" : "sltu", dst, swap ? rhs : lhs, swap ? lhs : rhs);
  if (invert)
    emit("xori", dst, dst, 0, 1);
}

Template TemplateBuilder::build(Operation *op) {
  const FixedResourcePlacement &f = fixedResources();
  t = Template();
  auto index = [](IntegerAttr attr) { return attr.getValue().getZExtValue(); };
  if (auto input = dyn_cast<VirtualInputBF16Op>(op)) {
    int dst = reg(input.getValue());
    for (unsigned half = 0; half < 2; ++half)
      inputHalf(dst + half, index(input.getIndexAttr()), half);
  } else if (auto input = dyn_cast<VirtualInputFP8Op>(op)) {
    inputHalf(reg(input.getValue()), index(input.getIndexAttr()), 0);
  } else if (auto output = dyn_cast<VirtualOutputBF16Op>(op)) {
    int src = reg(output.getValue());
    for (unsigned half = 0; half < 2; ++half)
      outputHalf(src + half, index(output.getIndexAttr()), half);
  } else if (auto load = dyn_cast<VirtualDMALoadFP8Op>(op)) {
    launch(load.getDramByte(), load.getSizeBytes(), std::nullopt, 1);
  } else if (auto load = dyn_cast<VirtualDMALoadBF16Op>(op)) {
    launch(load.getDramByte(), load.getSizeBytes(), std::nullopt, 2);
  } else if (auto store = dyn_cast<VirtualDMAStoreFP8Op>(op)) {
    launch(store.getDramByte(), store.getSizeBytes(), reg(store.getSrc()), 1);
  } else if (auto store = dyn_cast<VirtualDMAStoreBF16Op>(op)) {
    launch(store.getDramByte(), store.getSizeBytes(), reg(store.getSrc()), 2);
  } else if (auto await = dyn_cast<VirtualDMAAwaitFP8Op>(op)) {
    complete(reg(await.getValue()), 1);
  } else if (auto await = dyn_cast<VirtualDMAAwaitBF16Op>(op)) {
    complete(reg(await.getValue()), 2);
  } else if (isa<VirtualDMAWaitOp>(op)) {
    complete(std::nullopt, 0);
  } else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
    emit(mxu("vmatpush.weight", load.getUnit()), f.mxuWeightSlot,
         reg(load.getSrc()));
  } else if (auto load = dyn_cast<VirtualMXULoadAccFP8Op>(op)) {
    emit(mxu("vmatpush.acc.fp8", load.getUnit()), f.mxuAccSlot,
         reg(load.getSrc()));
  } else if (auto load = dyn_cast<VirtualMXULoadAccBF16Op>(op)) {
    emit(mxu("vmatpush.acc.bf16", load.getUnit()), f.mxuAccSlot,
         reg(load.getSrc()));
  } else if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
    emit(mxu("vmatmul", unitOf(reset.getAcc().getType())), f.mxuAccSlot,
         reg(reset.getActivation()), f.mxuWeightSlot);
  } else if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op)) {
    emit(mxu("vmatmul.acc", unitOf(acc.getAcc().getType())), f.mxuAccSlot,
         reg(acc.getActivation()), f.mxuWeightSlot);
  } else if (auto readout = dyn_cast<VirtualMXUReadoutBF16Op>(op)) {
    emit(mxu("vmatpop.bf16.acc", unitOf(readout.getAcc().getType())),
         reg(readout.getValue()), 0, f.mxuAccSlot);
  } else if (auto readout = dyn_cast<VirtualMXUReadoutFP8Op>(op)) {
    long long code = 127;
    if (auto scale = readout.getScale().getDefiningOp<VirtualScaleConstantOp>())
      code = scale.getCode();
    emit("seli", f.scaleReg, 0, 0, code);
    emit(mxu("vmatpop.fp8.acc", unitOf(readout.getAcc().getType())),
         reg(readout.getValue()), f.scaleReg, f.mxuAccSlot);
  } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
    unsigned unit = matmul.getUnit();
    emit(mxu("vmatpush.weight", unit), f.mxuWeightSlot, reg(matmul.getWeight()));
    emit(mxu("vmatmul", unit), f.mxuAccSlot, reg(matmul.getActivation()),
         f.mxuWeightSlot);
    emit(mxu("vmatpop.bf16.acc", unit), reg(matmul.getResult()), 0,
         f.mxuAccSlot);
  } else if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op)) {
    emit(vpuName(unary.getKind(), false), reg(unary.getDst()),
         reg(unary.getSrc()));
  } else if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op)) {
    emit(vpuName(binary.getKind(), true), reg(binary.getDst()),
         reg(binary.getLhs()), reg(binary.getRhs()));
  } else if (auto pack = dyn_cast<VirtualPackFP8Op>(op)) {
    reg(pack.getSrc());
    t.pack = true;
    li(f.inputBaseReg, f.packRelayoutWord);
    emit("vload", reg(pack.getResult()), f.inputBaseReg);
  } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    auto value = cast<IntegerAttr>(constant.getValue());
    li(reg(constant.getResult()),
       static_cast<uint32_t>(value.getValue().getSExtValue()));
  } else if (auto add = dyn_cast<arith::AddIOp>(op)) {
    emit("add", reg(add.getResult()), reg(add.getLhs()), reg(add.getRhs()));
  } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
    compare(cmp);
  }
  return std::move(t);
}

// The pack template up to its final VLOAD, with the 32-row relayout loop
// unrolled (AtlasVirtualToMachine.cpp lowerPack).
Template TemplateBuilder::packPrefix() {
  const FixedResourcePlacement &f = fixedResources();
  Placeholders none(true);
  TemplateBuilder b(none, true, 0, 0);
  b.emit("seli", f.scaleReg, 0, 0, 127);
  b.emit("vpack.bf16.fp8", 0, f.scaleReg, 32);
  b.li(f.outputBaseReg, f.packWord);
  b.emit("vstore", 0, f.outputBaseReg);
  b.li(f.packSourceRegs[0], f.packWord * 4);
  b.li(f.packSourceRegs[1], f.packWord * 4 + 512);
  b.li(f.packDestinationReg, f.packRelayoutWord * 4);
  b.li(f.packRowReg, 0);
  b.li(f.packRowsReg, 32);
  for (unsigned row = 0; row < 32; ++row) {
    for (unsigned word = 0; word < 4; ++word)
      for (unsigned half = 0; half < 2; ++half) {
        b.emit("lw", f.packTemporaryRegs[half], f.packSourceRegs[half], 0,
               4 * word);
        b.emit("sw", 0, f.packDestinationReg, f.packTemporaryRegs[half],
               4 * word + 16 * half);
      }
    for (unsigned r : {f.packSourceRegs[0], f.packSourceRegs[1]})
      b.emit("addi", r, r, 0, 16);
    b.emit("addi", f.packDestinationReg, f.packDestinationReg, 0, 32);
    b.emit("addi", f.packRowReg, f.packRowReg, 0, 1);
    b.emit("blt", 0, f.packRowReg, f.packRowsReg);
    b.emit("addi", 0, 0, 0, 0);
  }
  return std::move(b.t);
}

//===----------------------------------------------------------------------===//
// Timeline: order-preserving issue of templates through AtlasTiming (§8)
//===----------------------------------------------------------------------===//

const char *engineName(timing::Engine engine) {
  static const char *names[] = {"Scalar", "LSU", "MXU0", "MXU1",
                                "VPU",    "XLU", "DMA"};
  return names[static_cast<int>(engine)];
}

struct InstrSpan {
  std::string mnemonic, engine;
  int issue, end;
};

class Timeline {
public:
  Timeline() {
    const FixedResourcePlacement &f = fixedResources();
    regs = timing::unknownRegs();
    regs[f.halfSizeReg] = 1024; // the prologue's values
    regs[f.zeroReg] = 0;
    regs[f.oneReg] = 1;
  }

  // The cycle at which `t` would issue its first instruction.
  int probe(const Template &t) const {
    if (t.pack)
      return std::max(nextIssue, drain);
    if (t.instrs.empty())
      return nextIssue;
    const timing::Instr &first = t.instrs.front();
    return earliest(first, timing::footprintOf(first, regs));
  }

  // Issue `t` after everything placed so far. Returns {first issue, finish}.
  std::pair<int, int> place(const Template &t,
                            std::vector<InstrSpan> *spans = nullptr) {
    int first = nextIssue, finish = nextIssue;
    if (t.pack) {
      first = std::max(nextIssue, drain);
      int cycles = packPrefixCycles();
      // The relayout loop splits the stream into blocks, which drain.
      placed.clear();
      table = timing::ReservationTable();
      const FixedResourcePlacement &f = fixedResources();
      for (unsigned r : {f.outputBaseReg, f.packSourceRegs[0],
                         f.packSourceRegs[1], f.packDestinationReg,
                         f.packRowReg, f.packRowsReg, f.packTemporaryRegs[0],
                         f.packTemporaryRegs[1]})
        regs[r] = std::nullopt;
      nextIssue = drain = finish = first + cycles;
      if (spans)
        spans->push_back({"pack relayout loop", "LSU", first, finish});
    }
    for (size_t i = 0; i < t.instrs.size(); ++i) {
      auto [issue, end] = placeInstr(t.instrs[i], spans);
      if (i == 0 && !t.pack)
        first = issue;
      finish = std::max(finish, end);
    }
    return {first, finish};
  }

  int finish() const { return std::max(nextIssue, drain); }

  static int packPrefixCycles() {
    static const int cycles = [] {
      Timeline timeline;
      return timeline.place(TemplateBuilder::packPrefix()).second;
    }();
    return cycles;
  }

private:
  struct Placed {
    timing::Instr instr;
    timing::Footprint footprint;
    int issue;
  };

  int earliest(const timing::Instr &in, const timing::Footprint &f) const {
    int cycle = nextIssue;
    for (const Placed &p : placed) {
      // A dependence distance never exceeds the producer's doneAge + 1.
      if (p.issue + p.footprint.doneAge + 1 <= cycle)
        continue;
      timing::Dependence d = timing::dependence(p.instr, p.footprint, in, f);
      if (d.distance > 0)
        cycle = std::max(cycle, p.issue + d.distance);
    }
    if (in.op->opClass == timing::OpClass::DmaWait)
      cycle = std::max(cycle, channelRelease[in.op->channel]);
    for (int guard = 0; guard < 100000 && !table.conflict(in, f, cycle).empty();
         ++guard)
      ++cycle;
    return cycle;
  }

  std::pair<int, int> placeInstr(const timing::Instr &in,
                                 std::vector<InstrSpan> *spans) {
    timing::Footprint f = timing::footprintOf(in, regs);
    int cycle = earliest(in, f);
    table.reserve(in, f, cycle);
    int end = cycle + f.doneAge + 1;
    if (f.dmaCycles > 0 && in.op->opClass != timing::OpClass::DmaConfig) {
      // Transfers run one at a time, in issue order (AtlasScheduling.cpp).
      int latency = f.dmaCycles;
      dmaQueueEnd = std::max(cycle + latency - 1, dmaQueueEnd + latency);
      channelRelease[in.op->channel] = dmaQueueEnd + 2;
      end = channelRelease[in.op->channel];
    }
    placed.push_back({in, f, cycle});
    nextIssue = cycle + timing::naturalGap(in);
    drain = std::max(drain, end);
    timing::applyScalar(in, regs);
    if (spans)
      spans->push_back({in.op->name, engineName(in.op->engine), cycle, end});
    return {cycle, end};
  }

  std::vector<Placed> placed;
  timing::ReservationTable table;
  timing::RegValues regs;
  int nextIssue = 0, drain = 0, dmaQueueEnd = -1;
  std::array<int, 8> channelRelease{};
};

//===----------------------------------------------------------------------===//
// Per-block dependence graph (§5)
//===----------------------------------------------------------------------===//

struct BlockGraph {
  Block *block = nullptr;
  unsigned blockIndex = 0;
  SmallVector<Operation *> nodes; // free operations, source order
  std::vector<SmallVector<unsigned, 4>> preds, succs;
};

BlockGraph buildGraph(Block &block, unsigned blockIndex) {
  BlockGraph g;
  g.block = &block;
  g.blockIndex = blockIndex;
  for (Operation &op : block)
    if (!isa<VirtualStartOp>(op) && !op.hasTrait<OpTrait::IsTerminator>())
      g.nodes.push_back(&op);
  unsigned n = g.nodes.size();
  g.preds.assign(n, {});
  g.succs.assign(n, {});
  llvm::DenseMap<Operation *, unsigned> index;
  for (unsigned i = 0; i < n; ++i)
    index[g.nodes[i]] = i;
  llvm::DenseSet<std::pair<unsigned, unsigned>> seen;
  auto edge = [&](unsigned from, unsigned to) {
    if (from != to && seen.insert({from, to}).second) {
      g.succs[from].push_back(to);
      g.preds[to].push_back(from);
    }
  };
  // E1: data, excluding the state token.
  for (unsigned i = 0; i < n; ++i)
    for (Value operand : g.nodes[i]->getOperands()) {
      if (isa<VirtualStateType>(operand.getType()))
        continue;
      auto found = index.find(operand.getDefiningOp());
      if (operand.getDefiningOp() && found != index.end())
        edge(found->second, i);
    }
  // E2: read-after-write, write-after-read, write-after-write per register.
  std::vector<Effects> effects;
  for (Operation *op : g.nodes)
    effects.push_back(effectsOf(op));
  for (unsigned r = 0; r < kNumPseudo; ++r) {
    std::optional<unsigned> writer;
    SmallVector<unsigned> readers;
    for (unsigned i = 0; i < n; ++i) {
      if (effects[i].reads & bit(r)) {
        if (writer)
          edge(*writer, i);
        readers.push_back(i);
      }
      if (effects[i].writes & bit(r)) {
        if (writer)
          edge(*writer, i);
        for (unsigned reader : readers)
          edge(reader, i);
        readers.clear();
        writer = i;
      }
    }
  }
  return g;
}

//===----------------------------------------------------------------------===//
// Register pressure (§7)
//===----------------------------------------------------------------------===//

class Pressure {
public:
  Pressure(const BlockGraph &g, const Liveness &liveness) {
    Block *block = g.block;
    for (unsigned i = 0; i < g.nodes.size(); ++i) {
      SmallVector<Value, 4> uses;
      for (Value operand : g.nodes[i]->getOperands())
        if (classOf(operand.getType()) && !llvm::is_contained(uses, operand))
          uses.push_back(operand);
      for (Value use : uses)
        ++remaining[use];
      operands.push_back(uses);
      SmallVector<Value, 2> defs;
      for (Value result : g.nodes[i]->getResults())
        if (classOf(result.getType()))
          defs.push_back(result);
      results.push_back(defs);
    }
    // Exit set X(B): live-out plus every terminator operand (§6.2).
    llvm::DenseSet<Value> exitSet(liveness.getLiveOut(block).begin(),
                                  liveness.getLiveOut(block).end());
    for (Value operand : block->getTerminator()->getOperands())
      exitSet.insert(operand);
    for (Value value : exitSet)
      if (auto k = classOf(value.getType())) {
        pinned.insert(value);
        ++exitCounts[*k];
      }
    // Entry set N(B): live-in plus every block argument, used or not.
    llvm::DenseSet<Value> entrySet(liveness.getLiveIn(block).begin(),
                                   liveness.getLiveIn(block).end());
    for (BlockArgument arg : block->getArguments())
      entrySet.insert(arg);
    for (Value value : entrySet)
      if (auto k = classOf(value.getType())) {
        ++entryCounts[*k];
        if (remaining.lookup(value) > 0 || pinned.contains(value))
          live[*k].insert(value);
      }
    peak = entryCounts;
    for (int k = 0; k < kNumClasses; ++k)
      peak[k] = std::max(peak[k], exitCounts[k]);
  }

  ClassCounts at(unsigned node) const {
    ClassCounts counts;
    for (int k = 0; k < kNumClasses; ++k)
      counts[k] = live[k].size();
    for (Value result : results[node])
      if (!live[*classOf(result.getType())].contains(result))
        ++counts[*classOf(result.getType())];
    return counts;
  }

  bool fits(unsigned node, const ClassCounts &caps) const {
    ClassCounts counts = at(node);
    for (int k = 0; k < kNumClasses; ++k)
      if (counts[k] > caps[k])
        return false;
    return true;
  }

  // Net change in live values if `node` were placed now.
  int delta(unsigned node) const {
    int change = 0;
    for (Value result : results[node])
      if (remaining.lookup(result) > 0 || pinned.contains(result))
        ++change;
    for (Value operand : operands[node])
      if (remaining.lookup(operand) == 1 && !pinned.contains(operand))
        --change;
    return change;
  }

  void place(unsigned node) {
    ClassCounts counts = at(node);
    for (int k = 0; k < kNumClasses; ++k)
      peak[k] = std::max(peak[k], counts[k]);
    for (Value operand : operands[node])
      if (--remaining[operand] == 0 && !pinned.contains(operand))
        live[*classOf(operand.getType())].erase(operand);
    for (Value result : results[node])
      if (remaining.lookup(result) > 0 || pinned.contains(result))
        live[*classOf(result.getType())].insert(result);
  }

  ClassCounts current() const {
    ClassCounts counts;
    for (int k = 0; k < kNumClasses; ++k)
      counts[k] = live[k].size();
    return counts;
  }

  ClassCounts entryCounts{}, exitCounts{}, peak{};

private:
  std::array<llvm::DenseSet<Value>, kNumClasses> live;
  llvm::DenseMap<Value, int> remaining;
  llvm::DenseSet<Value> pinned;
  std::vector<SmallVector<Value, 4>> operands;
  std::vector<SmallVector<Value, 2>> results;
};

//===----------------------------------------------------------------------===//
// Scheduling (§9)
//===----------------------------------------------------------------------===//

enum class Mode { Latency, Pressure, Random };

struct Evaluation {
  int cycles = 0;
  std::vector<unsigned> order;
  std::vector<std::pair<int, int>> spans; // per node, in `order`
  std::vector<InstrSpan> instrs;
  std::vector<unsigned> instrNode;
  std::vector<ClassCounts> pressure; // live values after each node
  ClassCounts peak{};
};

struct ABI {
  uint64_t inputBase = 0x90000000ULL, outputBase = 0x90010000ULL;
};

class BlockScheduler {
public:
  BlockScheduler(const BlockGraph &g, const Liveness &liveness,
                 const Caps &caps, ABI abi)
      : g(g), liveness(liveness), caps(caps), abi(abi) {}

  // Simulate `order` and record its timing and pressure.
  Evaluation evaluate(ArrayRef<unsigned> order) const {
    Evaluation e;
    Placeholders placeholders(caps.mixedFp8);
    Timeline timeline;
    Pressure pressure(g, liveness);
    for (unsigned node : order) {
      TemplateBuilder builder(placeholders, true, abi.inputBase,
                              abi.outputBase);
      e.spans.push_back(
          timeline.place(builder.build(g.nodes[node]), &e.instrs));
      e.instrNode.resize(e.instrs.size(), node);
      pressure.place(node);
      e.pressure.push_back(pressure.current());
    }
    e.order.assign(order.begin(), order.end());
    e.cycles = timeline.finish();
    e.peak = pressure.peak;
    return e;
  }

  // Check the order-independent block boundaries (§6.2).
  std::optional<std::string> boundaryViolation() const {
    Pressure pressure(g, liveness);
    for (int k = 0; k < kNumClasses; ++k) {
      if (pressure.entryCounts[k] > caps.limit[k])
        return "entry " + std::string(className(k)) + " pressure " +
               std::to_string(pressure.entryCounts[k]) + " exceeds cap " +
               std::to_string(caps.limit[k]);
      if (pressure.exitCounts[k] > caps.limit[k])
        return "exit " + std::string(className(k)) + " pressure " +
               std::to_string(pressure.exitCounts[k]) + " exceeds cap " +
               std::to_string(caps.limit[k]);
    }
    return std::nullopt;
  }

  std::optional<std::vector<unsigned>> schedule(Mode mode, unsigned seed) {
    unsigned n = g.nodes.size();
    if (mode != Mode::Random && heights.empty())
      computeHeights();
    std::vector<unsigned> waiting(n);
    for (unsigned i = 0; i < n; ++i)
      waiting[i] = g.preds[i].size();
    std::vector<bool> placedNode(n, false);
    std::vector<unsigned> order;
    Placeholders placeholders(caps.mixedFp8);
    Timeline timeline;
    Pressure pressure(g, liveness);
    std::mt19937 rng(seed);
    while (order.size() < n) {
      SmallVector<unsigned> ready;
      for (unsigned i = 0; i < n; ++i)
        if (!placedNode[i] && waiting[i] == 0)
          ready.push_back(i);
      unsigned pick;
      if (mode == Mode::Random) {
        pick = ready[std::uniform_int_distribution<size_t>(
            0, ready.size() - 1)(rng)];
      } else {
        SmallVector<unsigned> fits;
        for (unsigned i : ready)
          if (pressure.fits(i, caps.limit))
            fits.push_back(i);
        if (fits.empty())
          return std::nullopt; // every ready operation breaks a cap
        std::optional<unsigned> best;
        if (mode == Mode::Pressure) {
          for (unsigned i : fits)
            if (!best || betterForPressure(i, *best, pressure))
              best = i;
        } else {
          SmallVector<int> starts;
          int earliest = INT_MAX;
          for (unsigned i : fits) {
            TemplateBuilder builder(placeholders, false, abi.inputBase,
                                    abi.outputBase);
            starts.push_back(timeline.probe(builder.build(g.nodes[i])));
            earliest = std::min(earliest, starts.back());
          }
          for (unsigned j = 0; j < fits.size(); ++j)
            if (starts[j] <= earliest &&
                (!best || betterForLatency(fits[j], *best, pressure)))
              best = fits[j];
        }
        pick = *best;
      }
      TemplateBuilder builder(placeholders, true, abi.inputBase,
                              abi.outputBase);
      timeline.place(builder.build(g.nodes[pick]));
      pressure.place(pick);
      placedNode[pick] = true;
      order.push_back(pick);
      for (unsigned s : g.succs[pick])
        --waiting[s];
    }
    return order;
  }

private:
  bool betterForLatency(unsigned a, unsigned b, const Pressure &p) const {
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    if (p.delta(a) != p.delta(b))
      return p.delta(a) < p.delta(b);
    return a < b;
  }

  bool betterForPressure(unsigned a, unsigned b, const Pressure &p) const {
    if (p.delta(a) != p.delta(b))
      return p.delta(a) < p.delta(b);
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    return a < b;
  }

  // Longest modeled path to the end of the block, including each node's own
  // completion. Edge latencies come from placing the two templates on a fresh
  // timeline with canonical placeholders, so they do not depend on order.
  void computeHeights() {
    unsigned n = g.nodes.size();
    heights.assign(n, 0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      Placeholders placeholders(caps.mixedFp8);
      Timeline alone;
      TemplateBuilder builder(placeholders, true, abi.inputBase,
                              abi.outputBase);
      auto [start, finish] = alone.place(builder.build(g.nodes[i]));
      int height = finish - start;
      for (unsigned s : g.succs[i]) {
        Placeholders pair(caps.mixedFp8);
        Timeline timeline;
        TemplateBuilder first(pair, true, abi.inputBase, abi.outputBase);
        int from = timeline.place(first.build(g.nodes[i])).first;
        TemplateBuilder second(pair, true, abi.inputBase, abi.outputBase);
        int latency = timeline.probe(second.build(g.nodes[s])) - from;
        height = std::max(height, latency + heights[s]);
      }
      heights[i] = height;
    }
  }

  const BlockGraph &g;
  const Liveness &liveness;
  const Caps &caps;
  ABI abi;
  std::vector<int> heights;
};

//===----------------------------------------------------------------------===//
// Applying an order (§5.4, §9.3) and the per-function transaction (§10)
//===----------------------------------------------------------------------===//

void applyOrder(const BlockGraph &g, ArrayRef<unsigned> order) {
  Operation *terminator = g.block->getTerminator();
  for (unsigned node : order)
    g.nodes[node]->moveBefore(terminator);
}

void rethread(Block &block) {
  Value state;
  if (block.isEntryBlock()) {
    // virtual_start is pinned first; its result starts the chain.
  } else {
    state = block.getArgument(0);
  }
  for (Operation &op : block) {
    if (isa<VirtualStartOp>(op)) {
      state = op.getResult(0);
      continue;
    }
    for (OpOperand &operand : op.getOpOperands())
      if (isa<VirtualStateType>(operand.get().getType()))
        operand.set(state);
    for (Value result : op.getResults())
      if (isa<VirtualStateType>(result.getType()))
        state = result;
  }
}

struct Snapshot {
  explicit Snapshot(func::FuncOp function) {
    for (Block &block : function.getBody()) {
      std::vector<Operation *> ops;
      for (Operation &op : block) {
        ops.push_back(&op);
        for (OpOperand &operand : op.getOpOperands())
          if (isa<VirtualStateType>(operand.get().getType()))
            operands.push_back({&op, operand.getOperandNumber(), operand.get()});
      }
      blocks.push_back({&block, std::move(ops)});
    }
  }

  void restore() {
    for (auto &[block, ops] : blocks) {
      Operation *terminator = block->getTerminator();
      for (Operation *op : ops)
        if (op != terminator)
          op->moveBefore(terminator);
    }
    for (auto &[op, index, value] : operands)
      op->setOperand(index, value);
  }

  std::vector<std::pair<Block *, std::vector<Operation *>>> blocks;
  std::vector<std::tuple<Operation *, unsigned, Value>> operands;
};

std::string labelOf(Operation *op) {
  std::string label = op->getName().getStringRef().str();
  StringRef name(label);
  name.consume_front("atlas.virtual_");
  name.consume_front("atlas.");
  std::string out = name.str();
  if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op))
    out += " " + unary.getKind().str();
  else if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op))
    out += " " + binary.getKind().str();
  else if (auto input = dyn_cast<VirtualInputBF16Op>(op))
    out += " #" + std::to_string(input.getIndex());
  else if (auto input = dyn_cast<VirtualInputFP8Op>(op))
    out += " #" + std::to_string(input.getIndex());
  else if (auto output = dyn_cast<VirtualOutputBF16Op>(op))
    out += " #" + std::to_string(output.getIndex());
  else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op))
    out += " u" + std::to_string(matmul.getUnit());
  else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    if (auto value = dyn_cast<IntegerAttr>(constant.getValue()))
      out += " " + std::to_string(value.getValue().getSExtValue());
  } else {
    SmallVector<Value> values(op->getOperands());
    llvm::append_range(values, op->getResults());
    for (Value v : values)
      if (isa<VirtualMXUWeightType, VirtualMXUAccType>(v.getType())) {
        out += " u" + std::to_string(unitOf(v.getType()));
        break;
      }
  }
  if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc()))
    out += " (L" + std::to_string(loc.getLine()) + ")";
  return out;
}

struct BlockRecord {
  unsigned index;
  std::vector<std::string> labels;
  std::optional<std::string> keptReason;
  Evaluation source, scheduled;
  Caps caps;
  std::string strategy = "source";
};

struct FunctionRecord {
  std::string name;
  std::vector<BlockRecord> blocks;
  std::string note;
};

// Run `fn` with its diagnostics captured instead of printed.
template <typename Fn>
bool quietly(MLIRContext *context, std::string &firstError, Fn &&fn) {
  ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error && firstError.empty())
      firstError = diag.str();
    return success();
  });
  return fn();
}

bool lowers(ModuleOp module, std::string &why) {
  OwningOpRef<ModuleOp> clone(module.clone());
  PassManager pm(module.getContext(), ModuleOp::getOperationName());
  llvm::raw_null_ostream discard;
  if (failed(parsePassPipeline("lower-atlas-virtual-to-machine", pm, discard))) {
    // The lowering is not registered: check allocation only (§10.4).
    return quietly(module.getContext(), why, [&] {
      for (auto function : clone->getOps<func::FuncOp>()) {
        VirtualAllocationPlan plan;
        if (failed(plan.allocate(function)) || failed(plan.verify()))
          return false;
      }
      return true;
    });
  }
  return quietly(module.getContext(), why,
                 [&] { return succeeded(pm.run(*clone)); });
}

void writeEvaluation(llvm::json::OStream &j, const Evaluation &e) {
  j.attribute("cycles", e.cycles);
  j.attributeArray("order", [&] {
    for (unsigned node : e.order)
      j.value(static_cast<int64_t>(node));
  });
  j.attributeArray("spans", [&] {
    for (size_t i = 0; i < e.order.size(); ++i)
      j.object([&] {
        j.attribute("node", static_cast<int64_t>(e.order[i]));
        j.attribute("start", e.spans[i].first);
        j.attribute("end", e.spans[i].second);
      });
  });
  j.attributeArray("instrs", [&] {
    for (size_t i = 0; i < e.instrs.size(); ++i)
      j.object([&] {
        j.attribute("node", static_cast<int64_t>(e.instrNode[i]));
        j.attribute("mnemonic", e.instrs[i].mnemonic);
        j.attribute("engine", e.instrs[i].engine);
        j.attribute("issue", e.instrs[i].issue);
        j.attribute("end", e.instrs[i].end);
      });
  });
  j.attributeArray("pressure", [&] {
    for (const ClassCounts &counts : e.pressure)
      j.array([&] {
        for (int count : counts)
          j.value(count);
      });
  });
  j.attributeArray("peak", [&] {
    for (int count : e.peak)
      j.value(count);
  });
}

void writeTrace(llvm::raw_ostream &os,
                const std::vector<FunctionRecord> &functions) {
  llvm::json::OStream j(os, 1);
  j.object([&] {
    j.attributeArray("functions", [&] {
      for (const FunctionRecord &f : functions)
        j.object([&] {
          j.attribute("name", f.name);
          j.attribute("note", f.note);
          j.attributeArray("blocks", [&] {
            for (const BlockRecord &b : f.blocks)
              j.object([&] {
                j.attribute("index", static_cast<int64_t>(b.index));
                j.attributeArray("caps", [&] {
                  for (int cap : b.caps.limit)
                    j.value(cap);
                });
                j.attribute("kept_source", b.keptReason.has_value());
                j.attribute("reason", b.keptReason.value_or(""));
                j.attribute("strategy", b.strategy);
                j.attributeArray("labels", [&] {
                  for (const std::string &label : b.labels)
                    j.value(label);
                });
                j.attributeObject("source",
                                  [&] { writeEvaluation(j, b.source); });
                j.attributeObject("scheduled",
                                  [&] { writeEvaluation(j, b.scheduled); });
              });
          });
        });
    });
  });
  os << "\n";
}

struct ScheduleAtlasVirtualPass
    : PassWrapper<ScheduleAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasVirtualPass)

  ScheduleAtlasVirtualPass() = default;
  ScheduleAtlasVirtualPass(const ScheduleAtlasVirtualPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "schedule-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Reorder each virtual Atlas block before register allocation";
  }

  Option<bool> report{*this, "report",
                      llvm::cl::desc("Remark modeled cycles and peak pressure "
                                     "for each block"),
                      llvm::cl::init(false)};
  Option<bool> strict{*this, "strict",
                      llvm::cl::desc("Fail instead of reverting when the "
                                     "scheduled order does not verify"),
                      llvm::cl::init(false)};
  Option<int> randomSeed{
      *this, "random-seed",
      llvm::cl::desc("Pick a random legal order (soundness testing); "
                     "negative disables"),
      llvm::cl::init(-1)};
  Option<std::string> traceFile{
      *this, "trace-file",
      llvm::cl::desc("Write the modeled source and scheduled timelines as "
                     "JSON"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    bool random = randomSeed >= 0;
    // The lowering accepts exactly one function (§10.4).
    bool trial = functions.size() == 1 && !random;
    std::string sourceWhy;
    bool sourceLowers = trial && lowers(module, sourceWhy);
    std::vector<FunctionRecord> records;
    for (func::FuncOp function : functions) {
      FunctionRecord record;
      record.name = function.getSymName().str();
      if (failed(scheduleFunction(module, function, random, trial,
                                  sourceLowers, record)))
        return signalPassFailure();
      records.push_back(std::move(record));
    }
    std::string path = traceFile;
    if (!path.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream os(path, error);
      if (error) {
        module.emitError("cannot write trace file ") << path << ": "
                                                     << error.message();
        return signalPassFailure();
      }
      writeTrace(os, records);
    }
  }

  LogicalResult scheduleFunction(ModuleOp module, func::FuncOp function,
                                 bool random, bool trial, bool sourceLowers,
                                 FunctionRecord &record) {
    Snapshot snapshot(function);
    ABI abi;
    if (auto attr = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base"))
      abi.inputBase = attr.getValue().getZExtValue();
    if (auto attr =
            function->getAttrOfType<IntegerAttr>("atlas.output_dram_base"))
      abi.outputBase = attr.getValue().getZExtValue();
    SmallVector<Mode> attempts =
        random ? SmallVector<Mode>{Mode::Random}
               : SmallVector<Mode>{Mode::Latency, Mode::Pressure};
    for (size_t attempt = 0; attempt < attempts.size(); ++attempt) {
      Mode mode = attempts[attempt];
      std::vector<BlockRecord> blocks;
      SmallVector<std::pair<Operation *, std::string>> remarks;
      bool changed = false;
      {
        Liveness liveness(function);
        Caps caps = capsFor(function);
        unsigned blockIndex = 0;
        for (Block &block : function.getBody()) {
          BlockGraph g = buildGraph(block, blockIndex);
          BlockScheduler scheduler(g, liveness, caps, abi);
          BlockRecord b{blockIndex, {}, std::nullopt, {}, {}, caps, "source"};
          for (Operation *op : g.nodes)
            b.labels.push_back(labelOf(op));
          std::vector<unsigned> source(g.nodes.size());
          for (unsigned i = 0; i < source.size(); ++i)
            source[i] = i;
          b.source = scheduler.evaluate(source);
          std::vector<unsigned> order = source;
          b.scheduled = b.source;
          if (g.nodes.empty()) {
            // Nothing to order.
          } else if (random) {
            order = *scheduler.schedule(Mode::Random,
                                        static_cast<unsigned>(randomSeed) +
                                            blockIndex);
            b.strategy = "random";
            b.scheduled = scheduler.evaluate(order);
          } else if (auto why = scheduler.boundaryViolation()) {
            b.keptReason = *why;
          } else {
            // Keep the fastest modeled candidate, never worse than a source
            // order that fits the caps; ties keep the source order.
            bool sourceFits = true;
            for (int k = 0; k < kNumClasses; ++k)
              sourceFits &= b.source.peak[k] <= caps.limit[k];
            std::optional<int> best;
            if (sourceFits) {
              best = b.source.cycles;
              b.strategy = "source";
            }
            SmallVector<std::pair<Mode, const char *>> modes;
            if (mode == Mode::Latency)
              modes.push_back({Mode::Latency, "latency"});
            modes.push_back({Mode::Pressure, "pressure"});
            for (auto [candidate, name] : modes) {
              auto found = scheduler.schedule(candidate, 0);
              if (!found)
                continue;
              Evaluation e = scheduler.evaluate(*found);
              if (!best || e.cycles < *best) {
                best = e.cycles;
                order = *found;
                b.scheduled = std::move(e);
                b.strategy = name;
              }
            }
            if (!best)
              b.keptReason = "no order fits the register caps";
          }
          Operation *where = block.getTerminator();
          if (b.keptReason)
            remarks.push_back(
                {where, "virtual schedule kept source order for block " +
                            std::to_string(blockIndex) + ": " +
                            *b.keptReason});
          if (report) {
            std::string text;
            llvm::raw_string_ostream os(text);
            os << "virtual schedule block " << blockIndex << ": modeled "
               << b.source.cycles << " cycles in source order, "
               << b.scheduled.cycles << " scheduled; peak pressure";
            for (int k = 0; k < kNumClasses; ++k)
              os << (k ? "," : "") << " " << className(k) << " "
                 << b.scheduled.peak[k] << "/" << caps.limit[k]
                 << " (source " << b.source.peak[k] << ")";
            remarks.push_back({where, text});
          }
          if (order != source) {
            applyOrder(g, order);
            changed = true;
          }
          blocks.push_back(std::move(b));
          ++blockIndex;
        }
      }
      for (Block &block : function.getBody())
        rethread(block);

      auto accept = [&](std::string note) {
        for (auto &[op, text] : remarks)
          op->emitRemark(text);
        record.blocks = std::move(blocks);
        record.note = std::move(note);
        return success();
      };
      if (!changed)
        return accept("");

      std::string why;
      if (!quietly(module.getContext(), why, [&] {
            return succeeded(verifyAtlasVirtualModule(module));
          })) {
        snapshot.restore();
        if (strict)
          return function.emitError("virtual schedule produced an invalid "
                                    "order for @")
                 << function.getSymName() << ": " << why;
        function.emitWarning("virtual schedule produced an invalid order for @")
            << function.getSymName() << "; reverted: " << why;
        return keepSource(record, std::move(blocks), "invalid order reverted");
      }
      if (trial && sourceLowers) {
        std::string lowerWhy;
        if (!lowers(module, lowerWhy)) {
          snapshot.restore();
          if (attempt + 1 < attempts.size())
            continue;
          function.emitRemark("virtual schedule reverted @")
              << function.getSymName() << ": lowering failed: " << lowerWhy;
          return keepSource(record, std::move(blocks), "lowering failed");
        }
      }
      return accept("");
    }
    return success();
  }

  LogicalResult keepSource(FunctionRecord &record,
                           std::vector<BlockRecord> blocks, std::string note) {
    for (BlockRecord &b : blocks) {
      b.scheduled = b.source;
      b.strategy = "source";
      if (!b.keptReason)
        b.keptReason = note;
    }
    record.blocks = std::move(blocks);
    record.note = std::move(note);
    return success();
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasVirtualPass() {
  PassRegistration<ScheduleAtlasVirtualPass>();
}
