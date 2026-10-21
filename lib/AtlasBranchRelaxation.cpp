#include "Atlas/AtlasBranchRelaxation.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
using namespace mlir;
using namespace mlir::atlas;
namespace {
StringRef inverse(StringRef kind) {
  return llvm::StringSwitch<StringRef>(kind)
      .Case("beq", "bne").Case("bne", "beq")
      .Case("blt", "bge").Case("bge", "blt")
      .Case("bltu", "bgeu").Case("bgeu", "bltu").Default("");
}
bool nop(Operation *op) {
  auto a = dyn_cast<ALUImmOp>(op);
  return a && a.getKind() == "addi" && !a.getDst() && !a.getSrc() &&
         !a.getImmediate();
}
}

LogicalResult mlir::atlas::relaxAtlasBranches(ModuleOp module) {
  const size_t width = atlasHasDelaySlots(module) ? 2 : 1;
  SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op)) ops.push_back(&op);
  llvm::DenseMap<Operation *, Operation *> targets;
  for (auto [pc, op] : llvm::enumerate(ops)) {
    std::optional<int64_t> offset;
    if (auto b = dyn_cast<BranchOp>(op)) offset = b.getOffsetBytesAttr().getValue().getSExtValue();
    else if (auto j = dyn_cast<JumpOp>(op); j && j.getKind() == "jal") offset = j.getOffsetAttr().getValue().getSExtValue();
    if (!offset) continue;
    int64_t dest = int64_t(pc) + *offset / 2;
    if (*offset % 2 || dest < 0 || dest > int64_t(ops.size()))
      return op->emitOpError("invalid branch destination during layout");
    targets[op] = dest == int64_t(ops.size()) ? nullptr : ops[dest];
  }
  OpBuilder builder(module.getContext());
  while (true) {
    llvm::DenseMap<Operation *, int64_t> positions;
    for (auto [pc, op] : llvm::enumerate(ops)) positions[op] = pc;
    bool expanded = false;
    for (size_t pc = 0; pc < ops.size(); ++pc) {
      auto branch = dyn_cast<BranchOp>(ops[pc]);
      if (!branch) continue;
      Operation *target = targets.lookup(branch);
      int64_t offset = 2 * ((target ? positions.lookup(target) : int64_t(ops.size())) - int64_t(pc));
      if (offset >= -4096 && offset <= 4094) continue;
      if (pc + width >= ops.size() || inverse(branch.getKind()).empty() ||
          (width == 2 && isa<BranchOp, JumpOp>(ops[pc + 1])))
        return branch.emitOpError("cannot relax branch without a preserved delay slot and fallthrough");
      Operation *fallthrough = ops[pc + width], *slot = ops[pc + width - 1];
      builder.setInsertionPointAfter(slot);
      auto create = [&](StringRef name, ArrayRef<NamedAttribute> attrs) {
        OperationState state(branch.getLoc(), name);
        state.addOperands(slot->getResult(0));
        state.addTypes(slot->getResult(0).getType());
        state.addAttributes(attrs);
        Operation *result = builder.create(state);
        if (auto owner = branch->getAttr(kAtlasTagCFGBlock)) result->setAttr(kAtlasTagCFGBlock, owner);
        slot = result;
        builder.setInsertionPointAfter(result);
        return result;
      };
      auto attr = [&](StringRef name, Attribute value) { return builder.getNamedAttr(name, value); };
      Operation *jump = create("atlas.jump", {
          attr("kind", builder.getStringAttr("jal")), attr("dst", builder.getI32IntegerAttr(0)),
          attr("base", builder.getI32IntegerAttr(0)), attr("offset", builder.getI32IntegerAttr(0))});
      jump->setAttr("atlas.branch_trampoline", builder.getUnitAttr());
      Operation *delay = nullptr;
      if (width == 2)
        delay = create("atlas.alu_imm", {
            attr("kind", builder.getStringAttr("addi")), attr("dst", builder.getI32IntegerAttr(0)),
            attr("src", builder.getI32IntegerAttr(0)), attr("immediate", builder.getI32IntegerAttr(0))});
      branch->setAttr("atlas.relaxed_branch", builder.getStringAttr(branch.getKind()));
      branch.setKindAttr(builder.getStringAttr(inverse(branch.getKind())));
      targets[jump] = target;
      targets[branch] = fallthrough;
      ops.insert(ops.begin() + pc + width, jump);
      if (delay) ops.insert(ops.begin() + pc + width + 1, delay);
      expanded = true;
      break; // Recompute every affected offset; expansion is monotone.
    }
    if (expanded) continue;
    for (auto [op, target] : targets) {
      int64_t offset = 2 * ((target ? positions.lookup(target) : int64_t(ops.size())) - positions.lookup(op));
      if (auto b = dyn_cast<BranchOp>(op)) b.setOffsetBytesAttr(builder.getI32IntegerAttr(offset));
      else {
        if (offset < -1048576 || offset > 1048574)
          return op->emitOpError("long jump exceeds selected ISA range");
        cast<JumpOp>(op).setOffsetAttr(builder.getI32IntegerAttr(offset));
      }
    }
    break;
  }
  Value state;
  for (Operation &op : module.getBody()->getOperations()) {
    if (state) op.setOperand(0, state);
    state = op.getResult(0);
  }
  auto limit = atlasProgramWordLimit(module);
  if (failed(limit)) return failure();
  if (ops.size() > *limit)
    return module.emitOpError("program exceeds the ") << *limit << "-word policy limit after branch relaxation";
  return verifyAtlasRelaxedBranches(module);
}

LogicalResult mlir::atlas::verifyAtlasRelaxedBranches(ModuleOp module) {
  const size_t width = atlasHasDelaySlots(module) ? 2 : 1;
  SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations()) if (!isa<StartOp>(op)) ops.push_back(&op);
  for (auto [pc, op] : llvm::enumerate(ops)) {
    if (auto raw = op->getAttr("atlas.relaxed_branch")) {
      auto original = dyn_cast<StringAttr>(raw);
      auto branch = dyn_cast<BranchOp>(op);
      auto jump = pc + 2 * width - 1 < ops.size() ? dyn_cast<JumpOp>(ops[pc + width]) : JumpOp();
      if (!branch || !original || inverse(original.getValue()) != branch.getKind() ||
          branch.getOffsetBytes() != 4 * width || !jump || jump.getKind() != "jal" || jump.getDst() ||
          !jump->hasAttr("atlas.branch_trampoline") || (width == 2 && !nop(ops[pc + 3])))
        return op->emitOpError("malformed relaxed conditional branch");
    }
    if (op->hasAttr("atlas.branch_trampoline") &&
        (pc < width || !ops[pc - width]->hasAttr("atlas.relaxed_branch")))
      return op->emitOpError("orphan branch trampoline");
  }
  return success();
}
