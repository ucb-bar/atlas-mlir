#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTiming.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include <deque>
#include <set>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

int64_t signedValue(IntegerAttr attr) { return attr.getValue().getSExtValue(); }

FailureOr<Instr> toInstr(Operation *op) {
  Instr in;
  std::string name;
  auto mxu = [](uint32_t unit) { return ".mxu" + std::to_string(unit); };
  auto ch = [](uint32_t channel) { return ".ch" + std::to_string(channel); };
  if (auto x = dyn_cast<VLoadOp>(op)) {
    name = "vload";
    in.rd = x.getDst();
    in.rs1 = x.getBase();
    in.imm = signedValue(x.getOffsetAttr());
  } else if (auto x = dyn_cast<VStoreOp>(op)) {
    name = "vstore";
    in.rd = x.getSrc();
    in.rs1 = x.getBase();
    in.imm = signedValue(x.getOffsetAttr());
  } else if (auto x = dyn_cast<DMAOp>(op)) {
    // Register fields as in encodeMachineWord.
    bool store = x.getDirection() == "store";
    name = (store ? "dma.store" : "dma.load") + ch(x.getChannel());
    in.rd = store ? x.getDram() : x.getReg();
    in.rs1 = store ? x.getReg() : x.getDram();
    in.rs2 = x.getSize();
  } else if (auto x = dyn_cast<DMAWaitOp>(op)) {
    name = "dma.wait" + ch(x.getChannel());
  } else if (auto x = dyn_cast<DMAConfigOp>(op)) {
    name = "dma.config" + ch(x.getChannel());
    in.rs1 = x.getBaseReg();
  } else if (auto x = dyn_cast<MXUPushOp>(op)) {
    name = llvm::StringSwitch<std::string>(x.getKind())
               .Case("weight_fp8", "vmatpush.weight")
               .Case("acc_fp8", "vmatpush.acc.fp8")
               .Default("vmatpush.acc.bf16") +
           mxu(x.getUnit());
    in.rd = x.getSlot();
    in.rs1 = x.getSrc();
  } else if (auto x = dyn_cast<MXUMatmulOp>(op)) {
    name = (x.getAccumulate() ? "vmatmul.acc" : "vmatmul") + mxu(x.getUnit());
    in.rd = x.getAccSlot();
    in.rs1 = x.getSrc();
    in.rs2 = x.getWeightSlot();
  } else if (auto x = dyn_cast<MXUPopOp>(op)) {
    name = (x.getFormat() == "fp8" ? "vmatpop.fp8.acc" : "vmatpop.bf16.acc") +
           mxu(x.getUnit());
    in.rd = x.getDst();
    in.rs2 = x.getSlot();
    in.rs1 = x.getScaleReg();
  } else if (auto x = dyn_cast<VPUBinaryOp>(op)) {
    name = llvm::StringSwitch<std::string>(x.getKind())
               .Case("min", "vminimum.bf16")
               .Case("max", "vmaximum.bf16")
               .Default("v" + x.getKind().str() + ".bf16");
    in.rd = x.getDst();
    in.rs1 = x.getLhs();
    in.rs2 = x.getRhs();
  } else if (auto x = dyn_cast<VPUUnaryOp>(op)) {
    name = x.getKind() == "mov" ? "vmov" : "v" + x.getKind().str() + ".bf16";
    in.rd = x.getDst();
    in.rs1 = x.getSrc();
  } else if (auto x = dyn_cast<VPUPackOp>(op)) {
    name = x.getDirection() == "bf16_to_fp8" ? "vpack.bf16.fp8"
                                             : "vunpack.fp8.bf16";
    in.rd = x.getDst();
    in.rs2 = x.getSrc();
    in.rs1 = x.getScaleReg();
  } else if (auto x = dyn_cast<VPUReduceOp>(op)) {
    StringRef kind = x.getKind(); // col_sum, row_max, ...
    std::string reduction = "vred" + kind.drop_front(4).str();
    name = reduction + (kind.starts_with("row") ? ".row.bf16" : ".bf16");
    in.rd = x.getDst();
    in.rs1 = x.getSrc();
  } else if (auto x = dyn_cast<VLIOp>(op)) {
    name = "vli." + x.getMode().str();
    in.rd = x.getDst();
    in.imm = x.getImmediate();
  } else if (auto x = dyn_cast<XLUTransposeOp>(op)) {
    name = "vtrpose.xlu";
    in.rd = x.getDst();
    in.rs1 = x.getSrc();
  } else if (auto x = dyn_cast<ALURegOp>(op)) {
    name = x.getKind().str();
    in.rd = x.getDst();
    in.rs1 = x.getLhs();
    in.rs2 = x.getRhs();
  } else if (auto x = dyn_cast<ALUImmOp>(op)) {
    name = x.getKind().str();
    in.rd = x.getDst();
    in.rs1 = x.getSrc();
    in.imm = signedValue(x.getImmediateAttr());
  } else if (auto x = dyn_cast<BranchOp>(op)) {
    name = x.getKind().str();
    in.rs1 = x.getLhs();
    in.rs2 = x.getRhs();
  } else if (auto x = dyn_cast<JumpOp>(op)) {
    name = x.getKind().str();
    in.rd = x.getDst();
    in.rs1 = x.getBase();
    in.imm = signedValue(x.getOffsetAttr());
  } else if (auto x = dyn_cast<UpperOp>(op)) {
    name = x.getKind().str();
    in.rd = x.getDst();
    in.imm = x.getImmediate();
  } else if (auto x = dyn_cast<CSROp>(op)) {
    name = "cs" + x.getKind().str(); // rrw -> csrrw
    in.rd = x.getDst();
    in.rs1 = x.getSource();
    in.imm = x.getAddress();
  } else if (auto x = dyn_cast<TrapOp>(op)) {
    name = x.getKind().str();
  } else if (isa<FenceOp>(op)) {
    name = "fence";
  } else if (auto x = dyn_cast<ScalarLoadOp>(op)) {
    name = x.getKind().str();
    in.rd = x.getDst();
    in.rs1 = x.getBase();
    in.imm = signedValue(x.getOffsetAttr());
  } else if (auto x = dyn_cast<ScalarStoreOp>(op)) {
    name = x.getKind().str();
    in.rs2 = x.getSrc();
    in.rs1 = x.getBase();
    in.imm = signedValue(x.getOffsetAttr());
  }
  in.op = findOp(name);
  if (!in.op)
    return op->emitOpError("has no entry in the timing model");
  return in;
}

bool isNop(Operation *op) {
  if (auto alu = dyn_cast<ALUImmOp>(op))
    return alu.getDst() == 0;
  if (auto alu = dyn_cast<ALURegOp>(op))
    return alu.getDst() == 0;
  return false;
}

Operation *createNop(OpBuilder &builder, Location loc, Value state) {
  return ALUImmOp::create(builder, loc, StateType::get(builder.getContext()),
                          state, builder.getStringAttr("addi"),
                          builder.getI32IntegerAttr(0),
                          builder.getI32IntegerAttr(0),
                          builder.getI32IntegerAttr(0));
}

// delay N stalls N + 1 cycles, and N has 12 bits.
std::vector<uint32_t> idleDelays(int idle) {
  std::vector<uint32_t> out;
  while (idle > 0) {
    int n = std::min(idle, 4096);
    out.push_back(n - 1);
    idle -= n;
  }
  return out;
}

struct Insertion {
  std::vector<uint32_t> delays;
  bool guard = false;
  std::string reason;
};

bool mergeInto(RegValues &into, const RegValues &from) {
  bool changed = false;
  for (int r = 1; r < 32; r++) {
    if (into[r] && (!from[r] || *from[r] != *into[r])) {
      into[r].reset();
      changed = true;
    }
  }
  return changed;
}

struct Issued {
  size_t index;
  Footprint f;
  int cycle;
};

constexpr int kMaxSearch = 100000;

LogicalResult timeBlock(ArrayRef<Operation *> ops, ArrayRef<Instr> instrs,
                        size_t begin, size_t end, bool fallsThrough,
                        RegValues regs, std::vector<Insertion> &before) {
  ReservationTable table;
  std::vector<Issued> issued;
  std::array<std::vector<size_t>, 8> pending;
  int nextFree = 0;

  auto name = [&](size_t i) { return ops[i]->getName().getStringRef().str(); };
  auto earliest = [&](const Instr &in, const Footprint &f, int cycle,
                      std::string &reason) {
    for (const Issued &x : issued) {
      Dependence d = dependence(instrs[x.index], x.f, in, f);
      if (d.distance > 0 && x.cycle + d.distance > cycle) {
        cycle = x.cycle + d.distance;
        reason = d.reason + " after " + name(x.index);
      }
    }
    return cycle;
  };
  auto drained = [&] {
    int cycle = 0;
    for (const Issued &x : issued)
      cycle = std::max(cycle, x.cycle + x.f.doneAge + 1);
    return cycle;
  };
  auto checkDma = [&](size_t i, const Footprint &f) -> LogicalResult {
    for (int ch = 0; ch < 8; ++ch)
      for (size_t k : pending[ch]) {
        EdgeKind kind;
        if (conflictsAtCompletion(issued[k].f, f, kind))
          return ops[i]->emitOpError()
                 << edgeKindName(kind) << " conflict with "
                 << name(issued[k].index) << " on channel " << ch
                 << ", which may still be in flight; a delay cannot cover a "
                    "DMA transfer, so add atlas.dma_wait first";
      }
    const OpInfo &op = *instrs[i].op;
    if (op.engine == Engine::Dma && op.opClass != OpClass::DmaWait &&
        !pending[op.channel].empty())
      ops[i]->emitWarning()
          << "reuses DMA channel " << op.channel << " while "
          << name(issued[pending[op.channel].back()].index)
          << " may still be in flight; the timing model needs an "
             "atlas.dma_wait here, which a delay cannot replace";
    return success();
  };
  auto place = [&](size_t i, const Footprint &f, int cycle,
                   const std::string &reason) {
    if (cycle > nextFree)
      before[i] = {idleDelays(cycle - nextFree), false, reason};
    const Instr &in = instrs[i];
    table.reserve(in, f, cycle);
    if (in.op->opClass == OpClass::DmaWait) {
      table.extendForWait(cycle);
      pending[in.op->channel].clear();
    }
    issued.push_back({i, f, cycle});
    if (in.op->engine == Engine::Dma && in.op->opClass != OpClass::DmaWait)
      pending[in.op->channel].push_back(issued.size() - 1);
    applyScalar(in, regs);
    nextFree = cycle + naturalGap(in);
  };
  auto search = [&](size_t i, int &cycle, std::string &reason,
                    function_ref<std::string(int)> fits) -> LogicalResult {
    for (int start = cycle;; ++cycle) {
      std::string why = fits(cycle);
      if (why.empty())
        return success();
      reason = why;
      if (cycle - start > kMaxSearch)
        return ops[i]->emitOpError("found no free issue cycle: ") << why;
    }
  };

  for (size_t i = begin; i < end; ++i) {
    const Instr &in = instrs[i];
    Footprint f = footprintOf(in, regs);
    if (!f.error.empty())
      return ops[i]->emitOpError(f.error);
    if (failed(checkDma(i, f)))
      return failure();
    std::string reason;
    int cycle = earliest(in, f, nextFree, reason);

    if (in.op->opClass == OpClass::Halt) {
      for (int ch = 0; ch < 8; ++ch)
        if (!pending[ch].empty())
          return ops[i]->emitOpError()
                 << "halts while DMA channel " << ch
                 << " may still be in flight; add atlas.dma_wait first";
      // A halt neither drains in-flight work nor waits for a delay, so its
      // stall ends on a NOP, reusing one that is already there.
      for (const Issued &x : issued)
        if (x.cycle + x.f.doneAge > cycle) {
          cycle = x.cycle + x.f.doneAge;
          reason = "halt waits for " + name(x.index) + " to finish";
        }
      int idle = cycle - nextFree;
      if (idle > 0) {
        size_t prev = i - 1;
        bool reuse =
            i > begin && isNop(ops[prev]) && before[prev].delays.empty();
        if (reuse)
          before[prev] = {idleDelays(idle), false, reason};
        else
          before[i] = {idleDelays(idle - 1), true, reason};
        nextFree = cycle;
      }
      place(i, f, cycle, reason);
      continue;
    }

    if (isControlFlow(*in.op)) {
      // The slot issues next; the successors start drained two cycles later.
      size_t s = i + 1;
      const Instr &slot = instrs[s];
      RegValues after = regs;
      applyScalar(in, after);
      Footprint sf = footprintOf(slot, after);
      if (!sf.error.empty())
        return ops[s]->emitOpError(sf.error);
      if (sf.doneAge > 0 || slot.op->opClass == OpClass::Halt)
        return ops[s]->emitOpError(
            "delay-slot instruction must be single-cycle scalar work");
      if (failed(checkDma(s, sf)))
        return failure();
      if (drained() - 2 > cycle) {
        cycle = drained() - 2;
        reason = "this block finishes before the branch's successors start";
      }
      auto fits = [&](int c) -> std::string {
        std::string why = table.conflict(in, f, c);
        if (!why.empty())
          return why;
        std::string slotReason;
        int slotCycle = earliest(slot, sf, c + 1, slotReason);
        Dependence d = dependence(in, f, slot, sf);
        if (c + d.distance > slotCycle) {
          slotCycle = c + d.distance;
          slotReason = d.reason + " after " + name(i);
        }
        if (slotCycle > c + 1)
          return "delay slot: " + slotReason;
        ReservationTable withBranch = table;
        withBranch.reserve(in, f, c);
        why = withBranch.conflict(slot, sf, c + 1);
        return why.empty() ? why : "delay slot: " + why;
      };
      if (failed(search(i, cycle, reason, fits)))
        return failure();
      place(i, f, cycle, reason);
      place(s, sf, cycle + 1, "");
      i = s;
      continue;
    }

    if (failed(search(i, cycle, reason,
                      [&](int c) { return table.conflict(in, f, c); })))
      return failure();
    place(i, f, cycle, reason);
  }

  if (fallsThrough && drained() > nextFree)
    before[end] = {idleDelays(drained() - nextFree), false,
                   "this block finishes before the next one starts"};
  for (int ch = 0; ch < 8; ++ch)
    if (!pending[ch].empty())
      ops[issued[pending[ch].back()].index]->emitWarning()
          << "may still be in flight when its block ends; DMA hazards are "
             "checked only within a block";
  return success();
}

LogicalResult insertDelays(ModuleOp module) {
  SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(module, words, /*llvmBlock=*/false)))
    return failure();
  OpBuilder builder(module.getContext());
  SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      ops.push_back(&op);

  for (Operation *op : ops) {
    if (isa<DelayOp>(op))
      return op->emitOpError(
          "is not allowed in the input; insert-atlas-delays computes every "
          "delay");
    if (auto upper = dyn_cast<UpperOp>(op); upper && upper.getKind() == "auipc")
      return op->emitOpError(
          "reads its own instruction index, which inserting delays changes");
    if (auto jump = dyn_cast<JumpOp>(op)) {
      if (jump.getKind() == "jalr")
        return op->emitOpError(
            "has a register target that inserting delays could invalidate");
      if (jump.getDst() != 0)
        return op->emitOpError(
            "writes a link value that inserting delays changes");
    }
  }

  // Targets kept as operations survive insertion; nullptr is the stream end.
  llvm::DenseMap<Operation *, Operation *> targetOf;
  for (auto [index, op] : llvm::enumerate(ops)) {
    int64_t offset;
    if (auto branch = dyn_cast<BranchOp>(op))
      offset = signedValue(branch.getOffsetBytesAttr());
    else if (auto jump = dyn_cast<JumpOp>(op))
      offset = signedValue(jump.getOffsetAttr());
    else
      continue;
    // ScalarCore.scala applies the byte displacement >> 1 to its word PC.
    int64_t target = static_cast<int64_t>(index) + offset / 2;
    int64_t size = static_cast<int64_t>(ops.size());
    if (target < 0 || target > size)
      return op->emitOpError("targets outside the instruction stream");
    targetOf[op] = target == size ? nullptr : ops[target];
  }

  size_t n = ops.size();
  llvm::DenseMap<Operation *, size_t> indexOf;
  std::vector<Instr> instrs;
  for (auto [index, op] : llvm::enumerate(ops)) {
    indexOf[op] = index;
    auto in = toInstr(op);
    if (failed(in))
      return failure();
    instrs.push_back(*in);
  }
  auto targetIndex = [&](size_t i) {
    Operation *target = targetOf.lookup(ops[i]);
    return target ? indexOf.lookup(target) : n;
  };

  std::set<size_t> leaders = {0};
  for (size_t i = 0; i < n; ++i) {
    if (isControlFlow(*instrs[i].op)) {
      size_t target = targetIndex(i);
      if (target > 0 && target < n && isControlFlow(*instrs[target - 1].op))
        return ops[i]->emitOpError("targets a delay slot");
      leaders.insert(target);
      leaders.insert(i + 2);
    }
    if (instrs[i].op->opClass == OpClass::Halt)
      leaders.insert(i + 1);
  }
  std::vector<size_t> starts;
  for (size_t leader : leaders)
    if (leader < n)
      starts.push_back(leader);
  size_t blocks = starts.size();
  std::vector<size_t> blockAt(n);
  for (size_t b = 0; b < blocks; ++b)
    blockAt[starts[b]] = b;
  auto endOf = [&](size_t b) { return b + 1 < blocks ? starts[b + 1] : n; };
  auto endsInBranch = [&](size_t b) {
    size_t e = endOf(b);
    return e - starts[b] >= 2 && isControlFlow(*instrs[e - 2].op);
  };

  std::vector<SmallVector<size_t, 2>> succs(blocks);
  for (size_t b = 0; b < blocks; ++b) {
    size_t e = endOf(b);
    if (endsInBranch(b)) {
      if (targetIndex(e - 2) < n)
        succs[b].push_back(blockAt[targetIndex(e - 2)]);
      if (instrs[e - 2].op->opClass == OpClass::Branch && e < n)
        succs[b].push_back(blockAt[e]);
    } else if (instrs[e - 1].op->opClass != OpClass::Halt && e < n) {
      succs[b].push_back(blockAt[e]);
    }
  }

  // Unlike npu_model's all-zero reset, a restarted core keeps its registers.
  std::vector<RegValues> entry(blocks, unknownRegs());
  std::vector<bool> reached(blocks, false);
  reached[0] = true;
  std::deque<size_t> work = {0};
  while (!work.empty()) {
    size_t b = work.front();
    work.pop_front();
    RegValues regs = entry[b];
    for (size_t i = starts[b]; i < endOf(b); ++i)
      applyScalar(instrs[i], regs);
    for (size_t s : succs[b]) {
      if (!reached[s]) {
        entry[s] = regs;
        reached[s] = true;
        work.push_back(s);
      } else if (mergeInto(entry[s], regs)) {
        work.push_back(s);
      }
    }
  }

  std::vector<Insertion> before(n);
  for (size_t b = 0; b < blocks; ++b) {
    size_t e = endOf(b);
    bool fallsThrough = e < n && !endsInBranch(b) &&
                        instrs[e - 1].op->opClass != OpClass::Halt;
    if (failed(timeBlock(ops, instrs, starts[b], e, fallsThrough, entry[b],
                         before)))
      return failure();
  }

  Type stateType = StateType::get(module.getContext());
  for (size_t i = 0; i < n; ++i) {
    const Insertion &insertion = before[i];
    if (insertion.delays.empty() && !insertion.guard)
      continue;
    Operation *op = ops[i];
    builder.setInsertionPoint(op);
    Value state = op->getOperand(0);
    StringAttr reason = builder.getStringAttr(insertion.reason);
    for (uint32_t cycles : insertion.delays) {
      auto delay =
          DelayOp::create(builder, op->getLoc(), stateType, state, cycles);
      delay->setAttr("atlas.reason", reason);
      state = delay.getResult();
    }
    if (insertion.guard) {
      Operation *nop = createNop(builder, op->getLoc(), state);
      nop->setAttr("atlas.reason", builder.getStringAttr(
                                       "a halt does not wait for a delay"));
      state = nop->getResult(0);
    }
    op->setOperand(0, state);
  }

  llvm::DenseMap<Operation *, int64_t> word;
  int64_t count = 0;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      word[&op] = count++;
  for (auto [op, target] : targetOf) {
    int64_t offset = 2 * ((target ? word[target] : count) - word[op]);
    if (auto branch = dyn_cast<BranchOp>(op))
      branch.setOffsetBytesAttr(builder.getI32IntegerAttr(offset));
    else
      cast<JumpOp>(op).setOffsetAttr(builder.getI32IntegerAttr(offset));
  }

  words.clear();
  return collectAtlasWords(module, words, /*llvmBlock=*/false);
}

struct InsertAtlasDelaysPass
    : PassWrapper<InsertAtlasDelaysPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertAtlasDelaysPass)

  StringRef getArgument() const final { return "insert-atlas-delays"; }
  StringRef getDescription() const final {
    return "Insert the minimum in-order delays of the npu_model rtl-match "
           "timing model, ported from atlas-compiler-experiments, into a "
           "stream without atlas.delay";
  }

  void runOnOperation() override {
    if (failed(insertDelays(getOperation())))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerInsertAtlasDelaysPass() {
  PassRegistration<InsertAtlasDelaysPass>();
}
