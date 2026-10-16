#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace mlir::atlas::timing {
struct AtlasPackLoopTimingProofBuilder {
  static TimingContext make(const Instr &in, std::shared_ptr<const uint8_t> region,
                            const std::array<uint32_t, 32> &addresses) {
    return TimingContext(new InstructionTimingContext(in, std::move(region), addresses));
  }
};
} // namespace mlir::atlas::timing

namespace {
std::optional<bool> taken(const Instr &in, const RegValues &regs) {
  if (in.rs1 < 0 || in.rs1 >= 32 || in.rs2 < 0 || in.rs2 >= 32 ||
      !regs[in.rs1] || !regs[in.rs2]) return std::nullopt;
  uint32_t a = *regs[in.rs1], b = *regs[in.rs2];
  const std::string &kind = in.op->name;
  if (kind == "beq") return a == b;
  if (kind == "bne") return a != b;
  if (kind == "blt") return int32_t(a) < int32_t(b);
  if (kind == "bge") return int32_t(a) >= int32_t(b);
  if (kind == "bltu") return a < b;
  if (kind == "bgeu") return a >= b;
  return std::nullopt;
}
bool helper(Operation *op) {
  auto tag = op->getAttrOfType<StringAttr>(kAtlasTagCFGHelper);
  return tag && tag.getValue() == "pack";
}
bool nop(const Instr &in) {
  return in.op && in.op->name == "addi" && in.rd == 0 && in.rs1 == 0 &&
      in.imm == 0 && !in.release;
}
} // namespace

std::vector<TimingContext> mlir::atlas::proveAtlasScalarLoopTimingContexts(
    const AtlasStream &stream, size_t begin, size_t branch, const RegValues &entry) {
  if (begin >= branch || branch + 1 >= stream.instrs.size()) return {};
  std::vector<TimingContext> result(branch + 2 - begin);
  if (!stream.instrs[branch].op || stream.instrs[branch].op->opClass != OpClass::Branch ||
      !nop(stream.instrs[branch + 1])) return result;
  std::vector<std::array<uint32_t, 32>> addresses(result.size());
  RegValues regs = entry;
  for (unsigned row = 0; row < 32; ++row) {
    std::optional<bool> again;
    for (size_t pc = begin; pc < branch + 2; ++pc) {
      const Instr &in = stream.instrs[pc];
      if (!in.op) return result;
      if (pc == branch) again = taken(in, regs);
      else if (in.op->opClass != OpClass::Alu && in.op->opClass != OpClass::Delay &&
               in.op->opClass != OpClass::ScalarLoad && in.op->opClass != OpClass::ScalarStore)
        return result;
      if (in.op->opClass == OpClass::ScalarLoad || in.op->opClass == OpClass::ScalarStore) {
        if ((in.op->name != "lw" && in.op->name != "sw") ||
            in.rs1 < 0 || in.rs1 >= 32 || !regs[in.rs1]) return result;
        int64_t raw = int64_t(*regs[in.rs1]) + signExtend(in.imm, 12);
        uint32_t effective = uint32_t(raw) & 0x1fffff;
        if (raw < 0 || uint64_t(raw) != effective || effective % 4 ||
            uint64_t(effective) + 4 > uint64_t(kVmemBytes)) return result;
        addresses[pc - begin][row] = effective;
      }
      applyScalar(in, regs);
    }
    // Earlier exit, another iteration or an undecidable branch gives no proof.
    if (!again || *again != (row < 31)) return result;
  }
  auto region = std::make_shared<const uint8_t>(0);
  for (size_t pc = begin; pc < branch; ++pc) {
    const Instr &in = stream.instrs[pc];
    if (in.op->name == "lw" || in.op->name == "sw")
      result[pc - begin] = AtlasPackLoopTimingProofBuilder::make(in, region, addresses[pc - begin]);
  }
  return result;
}

void mlir::atlas::populateAtlasInstructionTimingContexts(AtlasStream &stream) {
  stream.timingContexts.assign(stream.instrs.size(), {});
  // An indirect redirect could create an unrepresented entry into a candidate.
  for (const Instr &in : stream.instrs)
    if (in.op->name == "jalr") return;
  for (size_t block = 1; block < stream.starts.size(); ++block) {
    size_t begin = stream.starts[block], end = stream.blockEnd(block);
    if (end < begin + 3 || !stream.fallsThrough(block - 1)) continue;
    size_t branch = end - 2;
    Operation *redirect = stream.ops[branch];
    auto source = contractTag(redirect, kAtlasTagCFGSource);
    if (stream.instrs[branch].op->opClass != OpClass::Branch || !helper(redirect) || !source ||
        stream.targetOf.lookup(redirect) != stream.ops[begin] || !nop(stream.instrs[end - 1])) continue;
    bool shape = true;
    for (size_t predecessor = 0; predecessor < stream.succs.size(); ++predecessor)
      for (size_t successor : stream.succs[predecessor])
        if (successor == block && predecessor != block && predecessor + 1 != block) shape = false;
    for (size_t pc = begin; pc < branch; ++pc) {
      const Instr &in = stream.instrs[pc];
      if (in.op->opClass == OpClass::Delay) continue;
      if (!helper(stream.ops[pc]) || contractTag(stream.ops[pc], kAtlasTagCFGSource) != source ||
          (in.op->opClass != OpClass::Alu && in.op->name != "lw" && in.op->name != "sw")) shape = false;
    }
    if (!shape) continue;
    // Exclude the self edge: the fall-through preheader establishes true entry.
    RegValues entry = stream.entry[block - 1];
    for (size_t pc = stream.starts[block - 1]; pc < begin; ++pc)
      applyScalar(stream.instrs[pc], entry);
    auto contexts = proveAtlasScalarLoopTimingContexts(stream, begin, branch, entry);
    std::copy(contexts.begin(), contexts.end(), stream.timingContexts.begin() + begin);
  }
}
