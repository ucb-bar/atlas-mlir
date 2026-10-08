#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/Builders.h"
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
  } else if (auto x = dyn_cast<DelayOp>(op)) {
    name = "delay";
    in.imm = x.getCycles();
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

} // namespace

bool mlir::atlas::isNop(Operation *op) {
  if (auto alu = dyn_cast<ALUImmOp>(op))
    return alu.getDst() == 0;
  if (auto alu = dyn_cast<ALURegOp>(op))
    return alu.getDst() == 0;
  return false;
}

static Operation *createNop(OpBuilder &builder, Location loc, Value state) {
  return ALUImmOp::create(builder, loc, StateType::get(builder.getContext()),
                          state, builder.getStringAttr("addi"),
                          builder.getI32IntegerAttr(0),
                          builder.getI32IntegerAttr(0),
                          builder.getI32IntegerAttr(0));
}

// delay N stalls N + 1 cycles, and N has 12 bits.
std::vector<uint32_t> mlir::atlas::idleDelays(int idle) {
  std::vector<uint32_t> out;
  while (idle > 0) {
    int n = std::min(idle, 4096);
    out.push_back(n - 1);
    idle -= n;
  }
  return out;
}

static bool mergeInto(RegValues &into, const RegValues &from) {
  bool changed = false;
  for (int r = 1; r < 32; r++) {
    if (into[r] && (!from[r] || *from[r] != *into[r])) {
      into[r].reset();
      changed = true;
    }
  }
  return changed;
}

size_t AtlasStream::blockEnd(size_t block) const {
  return block + 1 < starts.size() ? starts[block + 1] : ops.size();
}

bool AtlasStream::endsInBranch(size_t block) const {
  size_t end = blockEnd(block);
  return end - starts[block] >= 2 && isControlFlow(*instrs[end - 2].op);
}

bool AtlasStream::endsInHalt(size_t block) const {
  return instrs[blockEnd(block) - 1].op->opClass == OpClass::Halt;
}

bool AtlasStream::fallsThrough(size_t block) const {
  return blockEnd(block) < ops.size() && !endsInBranch(block) &&
         !endsInHalt(block);
}

FailureOr<AtlasStream> mlir::atlas::readAtlasStream(ModuleOp module,
                                                 AtlasStreamReadMode mode) {
  bool verification = mode == AtlasStreamReadMode::Verification;
  SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(module, words, /*llvmBlock=*/false,
                               /*skipGeneratedCheck=*/verification)))
    return failure();
  AtlasStream s;
  SmallVector<Operation *> &ops = s.ops;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      ops.push_back(&op);

  for (Operation *op : ops) {
    if (!verification && isa<DelayOp>(op))
      return op->emitOpError(
          "is not allowed in the input; delays come from the timing model");
    if (auto upper = dyn_cast<UpperOp>(op);
        !verification && upper && upper.getKind() == "auipc")
      return op->emitOpError(
          "reads its own instruction index, which inserting delays changes");
    if (auto jump = dyn_cast<JumpOp>(op)) {
      if (jump.getKind() == "jalr")
        return op->emitOpError(
            verification ? "has an unproven JALR target for DMA memory verification"
                         : "has a register target that inserting delays could invalidate");
      if (!verification && jump.getDst() != 0)
        return op->emitOpError(
            "writes a link value that inserting delays changes");
    }
  }

  // Targets kept as operations survive insertion; nullptr is the stream end.
  llvm::DenseMap<Operation *, Operation *> &targetOf = s.targetOf;
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
  std::vector<Instr> &instrs = s.instrs;
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
  std::vector<size_t> &starts = s.starts;
  for (size_t leader : leaders)
    if (leader < n)
      starts.push_back(leader);
  size_t blocks = starts.size();
  if (!blocks)
    return s;
  std::vector<size_t> blockAt(n);
  for (size_t b = 0; b < blocks; ++b)
    blockAt[starts[b]] = b;

  s.succs.resize(blocks);
  std::vector<SmallVector<size_t, 2>> &succs = s.succs;
  for (size_t b = 0; b < blocks; ++b) {
    size_t e = s.blockEnd(b);
    if (s.endsInBranch(b)) {
      if (targetIndex(e - 2) < n)
        succs[b].push_back(blockAt[targetIndex(e - 2)]);
      if (instrs[e - 2].op->opClass == OpClass::Branch && e < n)
        succs[b].push_back(blockAt[e]);
    } else if (instrs[e - 1].op->opClass != OpClass::Halt && e < n) {
      succs[b].push_back(blockAt[e]);
    }
  }

  // Unlike npu_model's all-zero reset, a restarted core keeps its registers.
  s.entry.assign(blocks, unknownRegs());
  std::vector<RegValues> &entry = s.entry;
  std::vector<bool> reached(blocks, false);
  reached[0] = true;
  std::deque<size_t> work = {0};
  while (!work.empty()) {
    size_t b = work.front();
    work.pop_front();
    RegValues regs = entry[b];
    for (size_t i = starts[b]; i < s.blockEnd(b); ++i)
      applyScalar(instrs[i], regs);
    for (size_t next : succs[b]) {
      if (!reached[next]) {
        entry[next] = regs;
        reached[next] = true;
        work.push_back(next);
      } else if (mergeInto(entry[next], regs)) {
        work.push_back(next);
      }
    }
  }

  return s;
}

LogicalResult mlir::atlas::checkAtlasStream(const AtlasStream &s) {
  auto name = [&](size_t i) {
    return s.ops[i]->getName().getStringRef().str();
  };
  for (size_t b = 0; b < s.starts.size(); ++b) {
    RegValues regs = s.entry[b];
    std::array<std::vector<std::pair<size_t, Footprint>>, 8> pending;
    for (size_t i = s.starts[b]; i < s.blockEnd(b); ++i) {
      const Instr &in = s.instrs[i];
      const OpInfo &op = *in.op;
      Footprint f = footprintOf(in, regs);
      if (!f.error.empty())
        return s.ops[i]->emitOpError(f.error);
      if (i > s.starts[b] && isControlFlow(*s.instrs[i - 1].op) &&
          (f.doneAge > 0 || op.opClass == OpClass::Halt))
        return s.ops[i]->emitOpError(
            "delay-slot instruction must be single-cycle scalar work");
      for (int ch = 0; ch < 8; ++ch)
        for (const auto &[k, dma] : pending[ch]) {
          EdgeKind kind;
          if (conflictsAtCompletion(dma, f, kind))
            return s.ops[i]->emitOpError()
                   << edgeKindName(kind) << " conflict with " << name(k)
                   << " on channel " << ch
                   << ", which may still be in flight; a delay cannot cover "
                      "a DMA transfer, so add atlas.dma_wait first";
        }
      bool command = op.engine == Engine::Dma && op.opClass != OpClass::DmaWait;
      if (command && !pending[op.channel].empty())
        s.ops[i]->emitWarning()
            << "reuses DMA channel " << op.channel << " while "
            << name(pending[op.channel].back().first)
            << " may still be in flight; the timing model needs an "
               "atlas.dma_wait here, which a delay cannot replace";
      if (op.opClass == OpClass::Halt)
        for (int ch = 0; ch < 8; ++ch)
          if (!pending[ch].empty())
            return s.ops[i]->emitOpError()
                   << "halts while DMA channel " << ch
                   << " may still be in flight; add atlas.dma_wait first";
      if (op.opClass == OpClass::DmaWait)
        pending[op.channel].clear();
      if (command)
        pending[op.channel].push_back({i, f});
      applyScalar(in, regs);
    }
    for (int ch = 0; ch < 8; ++ch)
      if (!pending[ch].empty())
        s.ops[pending[ch].back().first]->emitWarning()
            << "may still be in flight when its block ends; DMA hazards are "
               "checked only within a block";
  }
  return success();
}

LogicalResult mlir::atlas::writeAtlasStream(ModuleOp module,
                                            const AtlasStream &s,
                                            ArrayRef<size_t> order,
                                            ArrayRef<DelayInsertion> before) {
  OpBuilder builder(module.getContext());
  Type stateType = StateType::get(module.getContext());
  Operation *prev = &module.getBody()->front();
  Value state = prev->getResult(0);
  auto append = [&](Operation *op) {
    op->moveAfter(prev);
    op->setOperand(0, state);
    prev = op;
    state = op->getResult(0);
  };
  for (size_t i : order) {
    Operation *op = s.ops[i];
    const DelayInsertion &insertion = before[i];
    StringAttr reason = builder.getStringAttr(insertion.reason);
    for (uint32_t cycles : insertion.delays) {
      builder.setInsertionPointAfter(prev);
      auto delay =
          DelayOp::create(builder, op->getLoc(), stateType, state, cycles);
      delay->setAttr("atlas.reason", reason);
      append(delay);
    }
    if (insertion.guard) {
      builder.setInsertionPointAfter(prev);
      Operation *nop = createNop(builder, op->getLoc(), state);
      nop->setAttr("atlas.reason", builder.getStringAttr(
                                       "a halt does not wait for a delay"));
      append(nop);
    }
    append(op);
  }

  llvm::DenseMap<Operation *, int64_t> word;
  int64_t count = 0;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      word[&op] = count++;
  // A target starts a block, and reordering may put another op first.
  llvm::DenseMap<Operation *, Operation *> first;
  for (size_t start : s.starts)
    first[s.ops[start]] = s.ops[order[start]];
  for (auto [op, target] : s.targetOf) {
    int64_t aim = target ? word[first.lookup(target)] : count;
    int64_t offset = 2 * (aim - word[op]);
    if (auto branch = dyn_cast<BranchOp>(op))
      branch.setOffsetBytesAttr(builder.getI32IntegerAttr(offset));
    else
      cast<JumpOp>(op).setOffsetAttr(builder.getI32IntegerAttr(offset));
  }

  SmallVector<uint32_t> words;
  return collectAtlasWords(module, words, /*llvmBlock=*/false);
}
