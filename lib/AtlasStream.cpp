#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include <atomic>
#include <deque>
#include <limits>
#include <set>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

int64_t signedValue(IntegerAttr attr) { return attr.getValue().getSExtValue(); }

FailureOr<Instr> toInstr(Operation *op) {
  Instr in;
  if (Attribute raw = op->getAttr("atlas.complete")) {
    auto complete = dyn_cast<BoolAttr>(raw);
    if (!complete)
      return op->emitOpError("atlas.complete requires a boolean completion annotation");
    in.release = complete.getValue();
  }
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
    name = vpuMnemonic(x.getKind().str());
    in.rd = x.getDst();
    in.rs1 = x.getLhs();
    in.rs2 = x.getRhs();
  } else if (auto x = dyn_cast<VPUUnaryOp>(op)) {
    name = vpuMnemonic(x.getKind().str());
    in.rd = x.getDst();
    in.rs1 = x.getSrc();
  } else if (auto x = dyn_cast<VPUPackOp>(op)) {
    name = x.getDirection() == "bf16_to_fp8" ? "vpack.bf16.fp8"
                                             : "vunpack.fp8.bf16";
    in.rd = x.getDst();
    in.rs2 = x.getSrc();
    in.rs1 = x.getScaleReg();
  } else if (auto x = dyn_cast<VPUReduceOp>(op)) {
    name = vpuMnemonic(x.getKind().str());
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
    // The selected ABI publishes completion through this CSR. Dropping an
    // optional annotation cannot turn an actual publication into an ordinary op.
    if (x.getAddress() == 0xc10 &&
        (x.getKind() == "rrw" || x.getKind() == "rrwi" || x.getSource() != 0))
      in.release = true;
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

Instr mlir::atlas::atlasNopInstr() {
  Instr nop;
  nop.op = findOp("addi");
  return nop;
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

static bool mergeInto(RegValues &into, const RegValues &from, int first = 1) {
  bool changed = false;
  for (int r = first; r < 32; r++) {
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
  size_t end = blockEnd(block), width = delaySlots ? 2 : 1;
  return end - starts[block] >= width && isControlFlow(*instrs[end - width].op);
}

size_t AtlasStream::redirectOf(size_t block) const {
  assert(endsInBranch(block) && "the block ends in a branch or jump");
  return blockEnd(block) - (delaySlots ? 2 : 1);
}

bool AtlasStream::endsInHalt(size_t block) const {
  return instrs[blockEnd(block) - 1].op->opClass == OpClass::Halt;
}

bool AtlasStream::fallsThrough(size_t block) const {
  return blockEnd(block) < ops.size() && !endsInBranch(block) &&
         !endsInHalt(block);
}

const AtlasLoop *AtlasStream::loop(size_t block) const {
  const AtlasLoop *loop = findAtlasLoopAt(loops, starts[block]);
  return loop && loop->begin == starts[block] ? loop : nullptr;
}

const AtlasLoopFacts &AtlasStream::factsOf(const AtlasLoop &loop) const {
  assert(&loop >= loops.data() && &loop < loops.data() + loops.size() &&
         "facts exist for this stream's own loops");
  return loopFacts[&loop - loops.data()];
}

bool AtlasStream::dmaMayCross(size_t block) const {
  if (block + 1 >= starts.size())
    return false;
  return loop(block) || (fallsThrough(block) && loop(block + 1));
}

static std::atomic<unsigned> decodeCount{0};

unsigned mlir::atlas::atlasStreamDecodeCount() { return decodeCount; }

FailureOr<AtlasStream> mlir::atlas::readAtlasStream(
    ModuleOp module, AtlasStreamReadMode mode,
    llvm::SmallVectorImpl<uint32_t> *words) {
  bool verification = mode == AtlasStreamReadMode::Verification;
  ++decodeCount;
  SmallVector<uint32_t> local;
  SmallVectorImpl<uint32_t> &encoded = words ? *words : local;
  if (failed(verification ? encodeAtlasWords(module, encoded)
                          : verifyAtlasArtifact(module, /*llvmBlock=*/false, encoded)))
    return failure();
  AtlasStream s;
  s.delaySlots = atlasHasDelaySlots(module);
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
      if (s.delaySlots && target > 0 && target < n &&
          isControlFlow(*instrs[target - 1].op))
        return ops[i]->emitOpError("targets a delay slot");
      leaders.insert(target);
      leaders.insert(i + (s.delaySlots ? 2 : 1));
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
      size_t redirect = s.redirectOf(b);
      if (targetIndex(redirect) < n)
        succs[b].push_back(blockAt[targetIndex(redirect)]);
      if (instrs[redirect].op->opClass == OpClass::Branch && e < n)
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

  // A loop's back-edge target and the instruction after it and its delay slot
  // are leaders, and nothing between them can be, so each loop fills one block.
  for (const AtlasLoop &loop : findAtlasLoops(ops, s.delaySlots)) {
    size_t block = blockAt[loop.begin];
    assert(s.blockEnd(block) == loop.end() && "an analyzed loop fills one block");
    AtlasLoopFacts facts;
    if (block > 0 && s.fallsThrough(block - 1)) {
      facts.entry = entry[block - 1];
      for (size_t i = starts[block - 1]; i < loop.begin; ++i)
        applyScalar(instrs[i], facts.entry);
    }
    facts.access = proveAtlasLoopAccess(s, loop, facts.entry);
    s.loops.push_back(loop);
    s.loopFacts.push_back(std::move(facts));
  }
  return s;
}

// DMA_CONFIG writes one shared ScalarCore register, irrespective of channel.
// Its value, like the scalar operands, must agree on every incoming CFG edge.
std::vector<std::optional<uint32_t>>
mlir::atlas::atlasDMAUpperWordEntries(const AtlasStream &s) {
  auto transfer = [&](size_t block, std::optional<uint32_t> &base) {
    RegValues regs = s.entry[block];
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      if (auto config = dyn_cast<DMAConfigOp>(s.ops[i]))
        base = regs[config.getBaseReg()];
      applyScalar(s.instrs[i], regs);
    }
    return success();
  };
  // Unknown sticks: a disagreeing edge resets the entry for good.
  auto join = [](std::optional<uint32_t> &into, const std::optional<uint32_t> &incoming) {
    if (!into || into == incoming)
      return false;
    into.reset();
    return true;
  };
  return std::move(atlasForwardEntries<std::optional<uint32_t>>(s, std::nullopt, transfer, join)->entries);
}

void mlir::atlas::applyAtlasScaleRegister(const Instr &in, RegValues &regs) {
  if (in.op->opClass == OpClass::ScaleImm)
    regs[in.rd] = static_cast<uint32_t>(in.imm) & 0xff;
  else if (in.op->opClass == OpClass::ScaleLoad)
    regs[in.rd].reset();
}

std::vector<RegValues>
mlir::atlas::atlasScaleRegisterEntries(const AtlasStream &s) {
  auto transfer = [&](size_t block, RegValues &regs) {
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i)
      applyAtlasScaleRegister(s.instrs[i], regs);
    return success();
  };
  auto join = [](RegValues &into, const RegValues &incoming) { return mergeInto(into, incoming, /*first=*/0); };
  return std::move(atlasForwardEntries(s, RegValues{}, transfer, join)->entries);
}

bool mlir::atlas::atlasBlockExits(const AtlasStream &s, size_t block) {
  if (s.succs[block].empty())
    return true;
  if (!s.endsInBranch(block))
    return false;
  size_t redirect = s.redirectOf(block);
  return !s.targetOf.lookup(s.ops[redirect]) || (s.instrs[redirect].op->opClass == OpClass::Branch && s.blockEnd(block) == s.ops.size());
}

LogicalResult mlir::atlas::checkAtlasStream(
    const AtlasStream &s, const TimingProvider &provider) {
  if (s.ops.empty())
    return failure();
  std::string missing = validateTimingProvider(provider);
  if (!missing.empty())
    return s.ops.front()->emitOpError(missing);
  std::string scope = provider.validateScope(s);
  if (!scope.empty())
    return s.ops.front()->emitOpError(scope);
  auto name = [&](size_t i) {
    return s.ops[i]->getName().getStringRef().str();
  };
  // Transfers stay in flight across a block boundary only into and out of an
  // analyzed loop (AtlasStream::dmaMayCross), so a block starts with the
  // transfers the block before it ended with.
  std::array<std::vector<std::pair<size_t, Footprint>>, kAtlasDMAChannels> pending;
  auto anyPending = [&] { return llvm::any_of(pending, [](const auto &p) { return !p.empty(); }); };
  auto meetsPending = [&](size_t i, const Footprint &f) -> LogicalResult {
    for (int ch = 0; ch < kAtlasDMAChannels; ++ch)
      for (const auto &[k, dma] : pending[ch]) {
        auto conflict = provider.dmaConflict(dma, f);
        if (!conflict.error.empty())
          return s.ops[i]->emitOpError(conflict.error);
        if (conflict.value.conflict)
          return s.ops[i]->emitOpError()
                 << edgeKindName(conflict.value.kind) << " conflict with " << name(k)
                 << " on channel " << ch
                 << ", which may still be in flight; a delay cannot cover "
                    "a DMA transfer, so add atlas.dma_wait first";
      }
    return success();
  };
  for (size_t b = 0; b < s.starts.size(); ++b) {
    RegValues regs = s.entry[b];
    // An analyzed loop's addresses change with each iteration, so its instructions
    // meet the pending transfers as it runs from the registers it is entered
    // with, not once with the registers its block joins.
    const AtlasLoop *loop = s.loop(b);
    if (loop && anyPending()) {
      RegValues iteration = s.factsOf(*loop).entry;
      auto exit = runAtlasLoop(
          s.ops, *loop, [&]() -> const RegValues & { return iteration; },
          [&](size_t i) -> LogicalResult {
            Footprint f = provider.footprint(s.instrs[i], iteration);
            applyScalar(s.instrs[i], iteration);
            if (!f.error.empty())
              return s.ops[i]->emitOpError(f.error);
            return meetsPending(i, f);
          });
      if (failed(exit))
        return failure();
      if (*exit != AtlasLoopExit::Exited)
        return s.ops[loop->branch]->emitOpError(
            "analyzed loop beside a pending DMA transfer has no decidable exit");
    }
    for (size_t i = s.starts[b]; i < s.blockEnd(b); ++i) {
      const Instr &in = s.instrs[i];
      const OpInfo &op = *in.op;
      Footprint f = provider.footprint(in, regs);
      if (!f.error.empty())
        return s.ops[i]->emitOpError(f.error);
      if (f.doneAge < 0)
        return s.ops[i]->emitOpError("timing provider returned a negative completion age");
      if (s.delaySlots && i > s.starts[b] && isControlFlow(*s.instrs[i - 1].op) &&
          (f.doneAge > 0 || op.opClass == OpClass::Halt))
        return s.ops[i]->emitOpError(
            "delay-slot instruction must be single-cycle scalar work");
      if (in.release)
        for (int ch = 0; ch < kAtlasDMAChannels; ++ch)
          if (!pending[ch].empty())
            return s.ops[i]->emitOpError("completion publication requires DMA.WAIT for pending channel ") << ch;
      if (!loop && failed(meetsPending(i, f)))
        return failure();
      bool command = op.opClass == OpClass::DmaLoad || op.opClass == OpClass::DmaStore;
      if (command && !pending[op.channel].empty())
        s.ops[i]->emitWarning()
            << "reuses DMA channel " << op.channel << " while "
            << name(pending[op.channel].back().first)
            << " may still be in flight; the timing model needs an "
               "atlas.dma_wait here, which a delay cannot replace";
      if (op.opClass == OpClass::Halt)
        for (int ch = 0; ch < kAtlasDMAChannels; ++ch)
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
    // Timing is checked one block at a time, so elsewhere a transfer must
    // complete in the block that starts it, as verifyAtlasTimedStream requires.
    if (!s.dmaMayCross(b))
      for (int ch = 0; ch < kAtlasDMAChannels; ++ch)
        if (!pending[ch].empty())
          return s.ops[pending[ch].back().first]->emitOpError(
              "may still be in flight when its block ends; timing requires DMA "
              "completion before a block boundary, so add atlas.dma_wait in "
              "this block");
  }
  return success();
}

LogicalResult mlir::atlas::writeAtlasStream(ModuleOp module,
                                            const AtlasStream &s,
                                            ArrayRef<size_t> order,
                                            ArrayRef<DelayInsertion> before,
                                            ArrayRef<DelayInsertion> after,
                                            StringRef timedBy) {
  assert((after.empty() || after.size() == s.starts.size()) &&
         "trailing delays are given per block");
  OpBuilder builder(module.getContext());
  Type stateType = StateType::get(module.getContext());
  Operation *prev = &module.getBody()->front();
  Value state = prev->getResult(0);
  std::vector<Operation *> emittedFirst(s.ops.size(), nullptr);
  size_t current = 0;
  auto append = [&](Operation *op) {
    if (!emittedFirst[current])
      emittedFirst[current] = op;
    op->moveAfter(prev);
    op->setOperand(0, state);
    prev = op;
    state = op->getResult(0);
  };
  // Padding takes the CFG ownership of the op it serves, so it lies in that
  // op's source block (and edge) region.
  auto carryCFGOwner = [&](Operation *owner, Operation *padding) {
    for (StringRef name : {kAtlasTagCFGBlock, kAtlasTagCFGEdge, kAtlasTagCFGHelper})
      if (Attribute value = owner->getAttr(name))
        padding->setAttr(name, value);
  };
  auto emitDelays = [&](const DelayInsertion &insertion, Operation *owner) {
    StringAttr reason = builder.getStringAttr(insertion.reason);
    for (uint32_t cycles : insertion.delays) {
      builder.setInsertionPointAfter(prev);
      auto delay =
          DelayOp::create(builder, owner->getLoc(), stateType, state, cycles);
      delay->setAttr("atlas.reason", reason);
      carryCFGOwner(owner, delay);
      append(delay);
    }
  };
  bool fillSlots = !s.delaySlots && !timedBy.empty();
  size_t block = 0;
  for (size_t position = 0; position < order.size(); ++position) {
    size_t i = order[position];
    current = i;
    Operation *op = s.ops[i];
    const DelayInsertion &insertion = before[i];
    emitDelays(insertion, op);
    if (insertion.guard) {
      builder.setInsertionPointAfter(prev);
      Operation *nop = createNop(builder, op->getLoc(), state);
      nop->setAttr("atlas.reason", builder.getStringAttr(
                                       "a halt does not wait for a delay"));
      carryCFGOwner(op, nop);
      append(nop);
    }
    append(op);
    // A block's trailing delays follow its last op and belong to it, so a
    // branch to the next block skips them.
    if (position + 1 == s.blockEnd(block)) {
      bool redirectLast = s.endsInBranch(block) && i == s.redirectOf(block);
      assert((s.delaySlots || !s.endsInBranch(block) || redirectLast ||
              (fillSlots && order[position - 1] == s.redirectOf(block))) &&
             "a pending delay slot is filled by the op after its redirect");
      if (fillSlots && redirectLast) {
        builder.setInsertionPointAfter(prev);
        Operation *nop = createNop(builder, op->getLoc(), state);
        nop->setAttr("atlas.reason", builder.getStringAttr(
                                         "no instruction can fill the delay slot"));
        carryCFGOwner(op, nop);
        append(nop);
      }
      if (!after.empty())
        emitDelays(after[block], op);
      ++block;
    }
  }

  llvm::DenseMap<Operation *, int64_t> word;
  int64_t count = 0;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      word[&op] = count++;
  // A target starts a block, and reordering may put another op first.
  llvm::DenseMap<Operation *, Operation *> first;
  for (size_t start : s.starts)
    first[s.ops[start]] = emittedFirst[order[start]];
  for (auto [op, target] : s.targetOf) {
    int64_t aim = target ? word[first.lookup(target)] : count;
    int64_t offset = 2 * (aim - word[op]);
    if (auto branch = dyn_cast<BranchOp>(op))
      branch.setOffsetBytesAttr(builder.getI32IntegerAttr(offset));
    else
      cast<JumpOp>(op).setOffsetAttr(builder.getI32IntegerAttr(offset));
  }

  bool timed = !timedBy.empty();
  if (fillSlots)
    module->removeAttr(kAtlasDelaySlotsPending);
  module->setAttr(kAtlasTimingState, builder.getStringAttr(timed ? "timed" : "untimed"));
  if (timed)
    module->setAttr(kAtlasTimingProvider, builder.getStringAttr(timedBy));
  else
    module->removeAttr(kAtlasTimingProvider);
  SmallVector<uint32_t> words;
  return verifyAtlasArtifact(module, /*llvmBlock=*/false, words);
}

LogicalResult mlir::atlas::verifyAtlasTimingState(ModuleOp module, bool requireTimed) {
  Attribute rawState = module->getAttr(kAtlasTimingState);
  Attribute rawProvider = module->getAttr(kAtlasTimingProvider);
  if (!rawState) {
    if (rawProvider)
      return module.emitOpError("timing provider requires an explicit timing state");
    return success();
  }
  auto state = dyn_cast<StringAttr>(rawState);
  if (!state || (state.getValue() != "untimed" && state.getValue() != "timed"))
    return module.emitOpError("unknown Atlas timing state");
  if (rawProvider) {
    auto provider = dyn_cast<StringAttr>(rawProvider);
    if (!provider || provider.getValue().empty())
      return module.emitOpError("Atlas timing provider must be a nonempty string");
  }
  if (state.getValue() == "timed" && !rawProvider)
    return module.emitOpError("timed Atlas stream requires its timing provider");
  if (state.getValue() == "timed" && !atlasHasDelaySlots(module))
    return module.emitOpError("timed Atlas stream requires its delay slots");
  if (requireTimed && state.getValue() != "timed")
    return module.emitOpError("untimed Atlas stream requires scheduling or delay insertion before executable emission");
  return success();
}

LogicalResult mlir::atlas::verifyAtlasTimedStream(
    const AtlasStream &s, const TimingProvider &provider) {
  if (failed(checkAtlasStream(s, provider)))
    return failure();
  struct Issued { size_t index; Footprint footprint; int cycle; };
  // As in checkAtlasStream, DMA stays in flight only into and out of an analyzed loop.
  std::array<Operation *, kAtlasDMAChannels> pendingDMA{};
  for (size_t block = 0; block < s.starts.size(); ++block) {
    RegValues regs = s.entry[block];
    auto reservations = provider.createReservations();
    if (!reservations.error.empty())
      return s.ops[s.starts[block]]->emitOpError(reservations.error);
    if (!reservations.value)
      return s.ops[s.starts[block]]->emitOpError("timing provider returned no reservation state");
    TimingReservations &table = *reservations.value;
    std::vector<Issued> issued;
    int cycle = 0;
    int drained = 0;
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      const Instr &in = s.instrs[i];
      Footprint f = provider.footprint(in, regs);
      if (!f.error.empty())
        return s.ops[i]->emitOpError(f.error);
      if (f.doneAge < 0 || f.doneAge >= std::numeric_limits<int>::max() - cycle)
        return s.ops[i]->emitOpError("timing provider completion age exceeds nonnegative cycle domain");
      for (const Issued &prior : issued) {
        auto rule = provider.dependence(s.instrs[prior.index], prior.footprint, in, f);
        if (!rule.error.empty())
          return s.ops[i]->emitOpError(rule.error);
        const Dependence &d = rule.value;
        if (d.distance < 0)
          return s.ops[i]->emitOpError("timing provider returned a negative issue distance");
        if (cycle - prior.cycle < d.distance)
          return s.ops[i]->emitOpError("insufficient issue spacing: ") << d.reason
                 << "; requires " << d.distance << ", got " << cycle - prior.cycle;
      }
      auto conflict = table.conflict(in, f, cycle);
      if (!conflict.error.empty())
        return s.ops[i]->emitOpError(conflict.error);
      if (!conflict.value.empty())
        return s.ops[i]->emitOpError("timing resource conflict: ") << conflict.value;
      if (in.release && cycle < drained)
        return s.ops[i]->emitOpError("completion publication precedes fixed-latency completion");
      if (in.op->opClass == OpClass::Halt) {
        if (i > 0 && isa<DelayOp>(s.ops[i - 1]))
          return s.ops[i]->emitOpError("halt does not wait for a preceding DELAY; require a scalar guard instruction");
        if (cycle < drained)
          return s.ops[i]->emitOpError("halt precedes completion of in-flight fixed-latency work");
      }
      if (isControlFlow(*in.op) && int64_t(cycle) + 2 < drained)
        return s.ops[i]->emitOpError("redirect reaches a successor before fixed-latency work drains");
      std::string reservationError = table.reserve(in, f, cycle);
      if (reservationError.empty() && in.op->opClass == OpClass::DmaWait)
        reservationError = table.onWait(in, cycle);
      if (!reservationError.empty())
        return s.ops[i]->emitOpError(reservationError);
      if (in.op->opClass == OpClass::DmaLoad || in.op->opClass == OpClass::DmaStore)
        pendingDMA[in.op->channel] = s.ops[i];
      else if (in.op->opClass == OpClass::DmaWait)
        pendingDMA[in.op->channel] = nullptr;
      issued.push_back({i, f, cycle});
      drained = std::max(drained, cycle + f.doneAge + 1);
      applyScalar(in, regs);
      auto gap = provider.issueGap(in);
      if (!gap.error.empty())
        return s.ops[i]->emitOpError(gap.error);
      if (gap.value <= 0 || cycle > std::numeric_limits<int>::max() - gap.value)
        return s.ops[i]->emitOpError("timing provider issue gap exceeds positive cycle domain");
      cycle += gap.value;
    }
    if (!s.dmaMayCross(block))
      for (Operation *launch : pendingDMA)
        if (launch)
          return launch->emitOpError("timing verification requires DMA completion before a block boundary");
    if (!s.endsInBranch(block) && !s.endsInHalt(block)) {
      int idle = 0;
      if (s.fallsThrough(block))
        for (size_t i = s.blockEnd(block); i < s.ops.size() && isa<DelayOp>(s.ops[i]); ++i) {
          auto gap = provider.issueGap(s.instrs[i]);
          if (!gap.error.empty())
            return s.ops[i]->emitOpError(gap.error);
          if (gap.value <= 0 || idle > std::numeric_limits<int>::max() - gap.value)
            return s.ops[i]->emitOpError("timing provider boundary gap exceeds positive cycle domain");
          idle += gap.value;
        }
      if (int64_t(drained) > int64_t(cycle) + idle)
        return s.ops[s.blockEnd(block) - 1]->emitOpError("fixed-latency work remains live across a block boundary or stream end");
    }
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasTiming(
    const AtlasVerificationContext &ctx, const TimingProvider &provider) {
  ModuleOp module = ctx.module;
  if (failed(verifyAtlasTimingState(module)))
    return failure();
  if (auto selected = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
      selected && canonicalTimingProviderId(selected.getValue()) != provider.id)
    return module.emitOpError("supplied timing policy disagrees with retained provider identity");
  assert(ctx.stream && "a timed stream is decoded");
  return verifyAtlasTimedStream(*ctx.stream, provider);
}

LogicalResult mlir::atlas::verifyAtlasTiming(
    ModuleOp module, const TimingProvider &provider) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasTiming(*ctx, provider);
}

FailureOr<TimingProvider> mlir::atlas::selectAtlasTimingProvider(
    ModuleOp module, StringRef requested) {
  auto retained = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
  if (!requested.empty() && retained &&
      canonicalTimingProviderId(retained.getValue()) !=
          canonicalTimingProviderId(requested))
    return module.emitOpError("supplied timing policy disagrees with retained provider identity");
  std::string id =
      !requested.empty() ? requested.str() : retained ? retained.getValue().str() : kBuiltinTimingProviderId.str();
  auto provider = lookupTimingProvider(id, module);
  if (!provider.error.empty())
    return module.emitOpError(provider.error);
  return std::move(provider.value);
}

LogicalResult mlir::atlas::verifyAtlasTiming(const AtlasVerificationContext &ctx) {
  auto provider = selectAtlasTimingProvider(ctx.module);
  return failed(provider) ? failure() : verifyAtlasTiming(ctx, *provider);
}

LogicalResult mlir::atlas::verifyAtlasTiming(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/false, /*requireStream=*/true);
  return failed(ctx) ? failure() : verifyAtlasTiming(*ctx);
}

namespace {
struct VerifyAtlasTimingPass
    : PassWrapper<VerifyAtlasTimingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasTimingPass)
  VerifyAtlasTimingPass() = default;
  VerifyAtlasTimingPass(const VerifyAtlasTimingPass &other) : PassWrapper(other) {}
  Option<std::string> provider{*this, "provider",
                               llvm::cl::desc("Timing provider (default: atlas.timing_provider, else builtin)"),
                               llvm::cl::init("")};
  StringRef getArgument() const final { return "verify-atlas-timing"; }
  StringRef getDescription() const final {
    return "Check actual issue spacing using the selected registered timing provider";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<uint32_t> words;
    if (failed(verifyAtlasArtifact(module, /*llvmBlock=*/false, words)))
      return signalPassFailure();
    auto selected = selectAtlasTimingProvider(module, provider.getValue());
    if (failed(selected) || failed(verifyAtlasTiming(module, *selected)))
      return signalPassFailure();
    Builder builder(module.getContext());
    module->setAttr(kAtlasTimingState, builder.getStringAttr("timed"));
    module->setAttr(kAtlasTimingProvider, builder.getStringAttr(selected->id));
  }
};
}

void mlir::atlas::registerVerifyAtlasTimingPass() {
  PassRegistration<VerifyAtlasTimingPass>();
}

std::unique_ptr<Pass> mlir::atlas::createVerifyAtlasTimingPass() {
  return std::make_unique<VerifyAtlasTimingPass>();
}
