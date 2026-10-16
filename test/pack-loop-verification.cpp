#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "VerificationTestSupport.h"
#include <functional>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
// One PACK; lowering emits its relayout loop between the boundary input and the MXU.
constexpr StringLiteral sourceText = R"mlir(module {
  func.func @pack() -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64
  } {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %p = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %s2 = "atlas.virtual_output_bf16"(%s1, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
})mlir";

const FixedResourcePlacement &fixed() {
  static VirtualAllocationPlan plan;
  return plan.fixed();
}

SmallVector<Operation *> opsOf(ModuleOp module) {
  SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      ops.push_back(&op);
  return ops;
}

Operation *first(ModuleOp module, function_ref<bool(Operation *)> wanted) {
  for (Operation *op : opsOf(module))
    if (wanted(op))
      return op;
  return nullptr;
}

bool helper(Operation *op) { return op->hasAttr(kAtlasTagCFGHelper); }
bool backEdge(Operation *op) { return isa<BranchOp>(op) && helper(op); }
bool copyLoad(Operation *op) { return isa<ScalarLoadOp>(op) && helper(op) && cast<ScalarLoadOp>(op).getKind() == "lw"; }
void setI32(Operation *op, StringRef name, int64_t value) { op->setAttr(name, Builder(op->getContext()).getI32IntegerAttr(value)); }
int64_t i32(Operation *op, StringRef name) { return op->getAttrOfType<IntegerAttr>(name).getInt(); }

// Clones `op` and threads the clone into the state chain just before `anchor`.
Operation *cloneBefore(Operation *op, Operation *anchor) {
  OpBuilder builder(anchor);
  Operation *clone = builder.clone(*op);
  clone->setOperand(0, anchor->getOperand(0));
  anchor->setOperand(0, clone->getResult(0));
  return clone;
}

// The lowered loop: its shape, the facts the decoded stream adds, and the
// VMEM its 32 iterations touch.
void decoded(MLIRContext &context, StringRef pipeline) {
  auto module = runPipeline(context, sourceText, pipeline);
  check(bool(module), "PACK source lowers through " + pipeline.str());
  if (!module)
    return;
  SmallVector<Operation *> ops = opsOf(*module);
  std::vector<AtlasPackLoop> loops = findAtlasPackLoops(ops);
  check(loops.size() == 1, "one PACK relayout loop");
  if (loops.size() != 1)
    return;
  const AtlasPackLoop &loop = loops[0];
  check(backEdge(ops[loop.branch]) && copyLoad(ops[loop.begin]) && contractTag(ops[loop.branch], kAtlasTagCFGSource) == loop.source,
        "the loop runs from the first copy load to its own helper back-edge");
  check(findAtlasPackLoopAt(loops, loop.begin) == &loop && findAtlasPackLoopAt(loops, loop.end() - 1) == &loop &&
            !findAtlasPackLoopAt(loops, loop.begin - 1) && !findAtlasPackLoopAt(loops, loop.end()),
        "the loop holds its delay slot and nothing outside it");

  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && stream->packLoops.size() == 1, "the decoded stream records the loop");
  if (failed(stream) || stream->packLoops.size() != 1)
    return;
  const AtlasPackLoop &decoded = stream->packLoops[0];
  size_t block = llvm::find(stream->starts, loop.begin) - stream->starts.begin();
  check(block < stream->starts.size() && stream->blockEnd(block) == loop.end() &&
            stream->packLoop(block) == &decoded && decoded.branch == loop.branch,
        "the loop fills one block");
  for (size_t b = 0; b + 1 < stream->starts.size(); ++b)
    check(stream->dmaMayCross(b) == (b + 1 == block || b == block), "DMA may cross only into and out of the loop");
  const AtlasPackLoopFacts &facts = stream->factsOf(decoded);
  uint64_t raw = uint64_t(fixed().packWord) * 4, relayout = uint64_t(fixed().packRelayoutWord) * 4;
  check(facts.entry[fixed().packRowReg] == 0u && facts.entry[fixed().packRowsReg] == 32u,
        "the loop is entered with its row counter and limit");
  check(facts.access && facts.access->reads.begin == raw && facts.access->reads.end == raw + 1024 &&
            facts.access->writes.begin == relayout && facts.access->writes.end == relayout + 1024,
        "the loop reads its raw scratch and writes its relayout scratch, 1 KiB each");
}

// Each change makes the back-edge stop being a PACK relayout loop.
void refused(MLIRContext &context) {
  const std::pair<StringRef, std::function<void(ModuleOp)>> cases[] = {
      {"a copy load without the helper tag", [](ModuleOp m) { first(m, copyLoad)->removeAttr(kAtlasTagCFGHelper); }},
      {"a copy load of another source operation", [](ModuleOp m) { Operation *l = first(m, copyLoad); setI32(l, kAtlasTagCFGSource, i32(l, kAtlasTagCFGSource) + 1); }},
      {"a forward helper branch", [](ModuleOp m) { setI32(first(m, backEdge), "offset_bytes", 4); }},
      {"a delay slot that writes a register", [](ModuleOp m) { setI32(first(m, backEdge)->getNextNode(), "dst", 5); }},
      {"a scale load in the body", [](ModuleOp m) { first(m, copyLoad)->setAttr("kind", StringAttr::get(m.getContext(), "seld")); }},
      {"a second entry into the body", [](ModuleOp m) {
         // A copy of the back-edge, placed before the loop, jumps forward into it.
         Operation *branch = first(m, backEdge);
         Operation *entry = cloneBefore(branch, first(m, copyLoad));
         setI32(entry, "offset_bytes", 4);
         cloneBefore(branch->getNextNode(), entry->getNextNode());
       }},
  };
  for (const auto &[name, mutate] : cases) {
    auto module = runPipeline(context, sourceText, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits");
    if (!module)
      return;
    mutate(*module);
    check(findAtlasPackLoops(opsOf(*module)).empty(), "not a PACK loop: " + name.str());
  }

  // A loop the run cannot decide is still a loop, with unknown access.
  auto module = runPipeline(context, sourceText, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits");
  if (!module)
    return;
  auto limit = [](Operation *op) {
    auto a = dyn_cast<ALUImmOp>(op);
    return a && helper(op) && a.getDst() == fixed().packRowsReg;
  };
  setI32(first(*module, limit), "src", fixed().scalarTemporary);
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && stream->packLoops.size() == 1 && !stream->factsOf(stream->packLoops[0]).access,
        "an unknown row limit leaves the loop's access unknown");
}
} // namespace

void atlas_test::runPackLoop(MLIRContext &context) {
  decoded(context, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits");
  decoded(context, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits,insert-atlas-delays");
  refused(context);
}
