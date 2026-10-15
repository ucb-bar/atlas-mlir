#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
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
bool isPackHelper(Operation *op) {
  auto helper = op->getAttrOfType<StringAttr>(kAtlasTagCFGHelper);
  return helper && helper.getValue() == "pack";
}

// A loop body instruction: the PACK's scalar copy work, or a delay.
bool isBodyInstruction(Operation *op, int32_t source) {
  if (isa<DelayOp>(op))
    return true;
  if (!isPackHelper(op) || contractTag(op, kAtlasTagCFGSource) != source)
    return false;
  if (auto load = dyn_cast<ScalarLoadOp>(op))
    return load.getKind() != "seli" && load.getKind() != "seld";
  return isa<ScalarStoreOp, ALUImmOp, ALURegOp, UpperOp>(op);
}

unsigned scalarWidth(const Instr &in) {
  StringRef name = in.op->name;
  return name == "lw" || name == "sw" ? 4 : name.starts_with("lh") || name == "sh" ? 2 : 1;
}
} // namespace

std::vector<AtlasPackLoop> mlir::atlas::findAtlasPackLoops(ArrayRef<Operation *> ops) {
  SmallVector<std::pair<size_t, int64_t>> redirects; // (redirect, target)
  for (auto [pc, op] : llvm::enumerate(ops)) {
    if (auto jump = dyn_cast<JumpOp>(op); jump && jump.getKind() == "jalr")
      return {};
    if (auto distance = atlasRedirectDistance(op))
      redirects.push_back({pc, int64_t(pc) + *distance});
  }
  std::vector<AtlasPackLoop> loops;
  for (auto [pc, target] : redirects) {
    auto branch = dyn_cast<BranchOp>(ops[pc]);
    auto source = branch ? contractTag(branch, kAtlasTagCFGSource) : std::nullopt;
    if (!source || !isPackHelper(branch) || target < 0 || target >= int64_t(pc) ||
        pc + 1 >= ops.size() || !isExactAtlasNop(ops[pc + 1]))
      continue;
    AtlasPackLoop loop;
    loop.begin = size_t(target);
    loop.branch = pc;
    loop.source = *source;
    bool body = llvm::all_of(ops.slice(loop.begin, loop.branch - loop.begin),
                             [&](Operation *op) { return isBodyInstruction(op, *source); });
    bool otherEntry = llvm::any_of(redirects, [&](auto redirect) {
      return redirect.first != pc && redirect.second >= 0 && loop.contains(size_t(redirect.second));
    });
    if (body && !otherEntry)
      loops.push_back(loop);
  }
  return loops;
}

const AtlasPackLoop *mlir::atlas::findAtlasPackLoopAt(ArrayRef<AtlasPackLoop> loops, size_t pc) {
  // Loops hold no branch but their own, so they are disjoint and in order.
  auto loop = llvm::partition_point(loops, [&](const AtlasPackLoop &l) { return l.end() <= pc; });
  return loop != loops.end() && loop->contains(pc) ? &*loop : nullptr;
}

std::optional<bool> mlir::atlas::atlasBranchTaken(BranchOp branch, const RegValues &regs) {
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

FailureOr<AtlasPackLoopExit>
mlir::atlas::runAtlasPackLoop(ArrayRef<Operation *> ops, const AtlasPackLoop &loop,
                              function_ref<const RegValues &()> regs,
                              function_ref<LogicalResult(size_t pc)> step) {
  auto branch = cast<BranchOp>(ops[loop.branch]);
  for (unsigned iteration = 0; iteration < kAtlasPackLoopIterationBound; ++iteration) {
    std::optional<bool> again;
    for (size_t pc = loop.begin; pc < loop.end(); ++pc) {
      if (pc == loop.branch)
        again = atlasBranchTaken(branch, regs());
      if (failed(step(pc)))
        return failure();
    }
    if (!again)
      return AtlasPackLoopExit::UnknownBranch;
    if (!*again)
      return AtlasPackLoopExit::Exited;
  }
  return AtlasPackLoopExit::Unbounded;
}

std::optional<AtlasVmemAccess>
mlir::atlas::proveAtlasPackLoopAccess(const AtlasStream &stream, const AtlasPackLoop &loop) {
  RegValues regs = loop.entry;
  AtlasVmemAccess access;
  bool known = true;
  auto step = [&](size_t pc) {
    const Instr &in = stream.instrs[pc];
    OpClass kind = in.op->opClass;
    if (kind == OpClass::ScalarLoad || kind == OpClass::ScalarStore) {
      std::optional<long long> address = scalarAddress(in, regs);
      // ScalarCore slices the address to VMEM, so one past it may alias anything.
      if (!address || *address + scalarWidth(in) > kVmemBytes)
        known = false;
      else
        (kind == OpClass::ScalarStore ? access.writes : access.reads)
            .cover({uint64_t(*address), uint64_t(*address) + scalarWidth(in)});
    }
    applyScalar(in, regs);
    return success();
  };
  FailureOr<AtlasPackLoopExit> exit =
      runAtlasPackLoop(stream.ops, loop, [&]() -> const RegValues & { return regs; }, step);
  if (failed(exit) || *exit != AtlasPackLoopExit::Exited || !known)
    return std::nullopt;
  return access;
}
