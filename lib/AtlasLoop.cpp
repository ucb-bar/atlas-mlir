#include "Atlas/AtlasLoop.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

void AtlasVmemRange::cover(const AtlasVmemRange &other) {
  if (other.empty())
    return;
  if (empty()) {
    *this = other;
    return;
  }
  begin = std::min(begin, other.begin);
  end = std::max(end, other.end);
}

std::optional<int64_t> mlir::atlas::atlasRedirectDistance(Operation *op) {
  if (auto branch = dyn_cast<BranchOp>(op))
    return branch.getOffsetBytesAttr().getValue().getSExtValue() / 2;
  if (auto jump = dyn_cast<JumpOp>(op); jump && jump.getKind() == "jal")
    return jump.getOffsetAttr().getValue().getSExtValue() / 2;
  return std::nullopt;
}

namespace {
// Keep lifecycle/control effects out of a summarized loop. Scalar loads
// invalidate their destination in applyScalar; a data-dependent address or
// branch therefore becomes unknown rather than being guessed from memory.
bool isBodyInstruction(Operation *op) {
  if (auto load = dyn_cast<ScalarLoadOp>(op))
    return load.getKind() != "seli" && load.getKind() != "seld";
  return isa<DelayOp, ScalarStoreOp, ALUImmOp, ALURegOp, UpperOp>(op);
}

} // namespace

std::vector<AtlasLoop> mlir::atlas::findAtlasLoops(ArrayRef<Operation *> ops,
                                                   bool delaySlots) {
  SmallVector<std::pair<size_t, int64_t>> redirects; // (redirect, target)
  for (auto [pc, op] : llvm::enumerate(ops)) {
    if (auto jump = dyn_cast<JumpOp>(op); jump && jump.getKind() == "jalr")
      return {};
    if (auto distance = atlasRedirectDistance(op))
      redirects.push_back({pc, int64_t(pc) + *distance});
  }
  std::vector<AtlasLoop> loops;
  for (auto [pc, target] : redirects) {
    auto branch = dyn_cast<BranchOp>(ops[pc]);
    if (!branch || target < 0 || target >= int64_t(pc) ||
        (delaySlots && pc + 1 >= ops.size()))
      continue;
    AtlasLoop loop;
    loop.begin = size_t(target);
    loop.branch = pc;
    loop.delaySlot = delaySlots;
    bool body =
        llvm::all_of(ops.slice(loop.begin, loop.branch - loop.begin),
                     isBodyInstruction) &&
        (!delaySlots || isBodyInstruction(ops[pc + 1]));
    bool otherEntry = llvm::any_of(redirects, [&](auto redirect) {
      return redirect.first != pc && redirect.second >= 0 &&
             loop.contains(size_t(redirect.second));
    });
    if (body && !otherEntry)
      loops.push_back(loop);
  }
  return loops;
}

const AtlasLoop *mlir::atlas::findAtlasLoopAt(ArrayRef<AtlasLoop> loops, size_t pc) {
  // A supported body has no interior branch; discovered loops are disjoint.
  auto loop = llvm::partition_point(
      loops, [&](const AtlasLoop &l) { return l.end() <= pc; });
  return loop != loops.end() && loop->contains(pc) ? &*loop : nullptr;
}

std::optional<bool> mlir::atlas::atlasBranchTaken(BranchOp branch,
                                                const RegValues &regs) {
  auto lhs = regs[branch.getLhs()], rhs = regs[branch.getRhs()];
  if (!lhs || !rhs)
    return std::nullopt;
  StringRef kind = branch.getKind();
  if (kind == "beq")
    return *lhs == *rhs;
  if (kind == "bne")
    return *lhs != *rhs;
  if (kind == "blt")
    return int32_t(*lhs) < int32_t(*rhs);
  if (kind == "bge")
    return int32_t(*lhs) >= int32_t(*rhs);
  if (kind == "bltu")
    return *lhs < *rhs;
  return *lhs >= *rhs;
}

FailureOr<AtlasLoopExit>
mlir::atlas::runAtlasLoop(ArrayRef<Operation *> ops, const AtlasLoop &loop,
                         function_ref<const RegValues &()> regs,
                         function_ref<LogicalResult(size_t pc)> step) {
  auto branch = cast<BranchOp>(ops[loop.branch]);
  for (unsigned iteration = 0; iteration < kAtlasLoopIterationBound; ++iteration) {
    std::optional<bool> again;
    for (size_t pc = loop.begin; pc < loop.end(); ++pc) {
      if (pc == loop.branch)
        again = atlasBranchTaken(branch, regs());
      if (failed(step(pc)))
        return failure();
    }
    if (!again)
      return AtlasLoopExit::UnknownBranch;
    if (!*again)
      return AtlasLoopExit::Exited;
  }
  return AtlasLoopExit::AnalysisLimit;
}

std::optional<AtlasVmemAccess>
mlir::atlas::proveAtlasLoopAccess(const AtlasStream &stream, const AtlasLoop &loop,
                                 const RegValues &entry) {
  RegValues regs = entry;
  AtlasVmemAccess access;
  bool known = true;
  auto step = [&](size_t pc) {
    const Instr &in = stream.instrs[pc];
    OpClass kind = in.op->opClass;
    if (kind == OpClass::ScalarLoad || kind == OpClass::ScalarStore) {
      ScalarAccess touched = scalarAccess(in, regs);
      // ScalarCore slices the address to VMEM, so one past it may alias anything.
      if (!touched.first || *touched.first + touched.bytes > kAtlasVmemBytes)
        known = false;
      else
        (touched.write ? access.writes : access.reads)
            .cover({uint64_t(*touched.first), uint64_t(*touched.first + touched.bytes)});
    }
    applyScalar(in, regs);
    return success();
  };
  FailureOr<AtlasLoopExit> exit =
      runAtlasLoop(stream.ops, loop,
                   [&]() -> const RegValues & { return regs; }, step);
  if (failed(exit) || *exit != AtlasLoopExit::Exited || !known)
    return std::nullopt;
  return access;
}
