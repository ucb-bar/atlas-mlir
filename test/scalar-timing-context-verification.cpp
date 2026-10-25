#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;
using namespace atlas_test;

namespace {
constexpr StringLiteral source = R"mlir(module {
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

OwningOpRef<ModuleOp> lower(MLIRContext &context) {
  return runPipeline(context, source, "lower-atlas-virtual-to-machine,insert-atlas-dma-waits");
}
const FixedResourcePlacement &fixed() {
  static VirtualAllocationPlan plan;
  return plan.fixed();
}
bool noContexts(const std::vector<TimingContext> &contexts) {
  return llvm::none_of(contexts, [](const TimingContext &c) { return bool(c); });
}
void setI32(Operation *op, StringRef key, int64_t value) {
  op->setAttr(key, Builder(op->getContext()).getI32IntegerAttr(value));
}

void negative(MLIRContext &context) {
  auto module = lower(context);
  check(bool(module), "PACK context fixture lowers", diagnostics);
  if (!module) return;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && stream->loops.size() == 1, "PACK context fixture has one proved loop");
  if (failed(stream) || stream->loops.size() != 1) return;
  const auto &loop = stream->loops.front();
  RegValues entry = stream->factsOf(loop).entry;
  for (uint32_t rows : {31, 33, 0, 257}) {
    RegValues changed = entry;
    changed[fixed().packRowsReg] = rows;
    check(noContexts(proveAtlasPackLoopTimingContexts(*stream, loop, changed)),
          "only a decided exact 32-iteration replay carries timing proofs");
  }
  RegValues unknown = entry;
  unknown[fixed().packRowsReg].reset();
  check(noContexts(proveAtlasPackLoopTimingContexts(*stream, loop, unknown)), "unknown branch exit has no timing proof");
  unknown = entry;
  unknown[fixed().packSourceRegs[0]].reset();
  check(noContexts(proveAtlasPackLoopTimingContexts(*stream, loop, unknown)), "unknown source address has no timing proof");
  for (uint32_t bad : {uint32_t(*entry[fixed().packSourceRegs[0]] + 1), uint32_t(kAtlasVmemBytes - 496),
                       uint32_t(0x200000), uint32_t(0xfffffffc)}) {
    RegValues changed = entry;
    changed[fixed().packSourceRegs[0]] = bad;
    check(noContexts(proveAtlasPackLoopTimingContexts(*stream, loop, changed)),
          "unaligned, final-row out-of-range and wrapping addresses fail closed");
  }
  // A generic loop may prove DMA byte accesses without qualifying scalar
  // compute overlap: PACK provenance is an independent requirement.
  auto genericModule = lower(context);
  if (genericModule) {
    auto tagged = readAtlasStream(*genericModule, AtlasStreamReadMode::Verification);
    if (succeeded(tagged) && tagged->loops.size() == 1) {
      for (Operation *op : tagged->ops) {
        op->removeAttr(kAtlasTagCFGHelper);
        op->removeAttr(kAtlasTagCFGSource);
      }
      const auto &genericLoop = tagged->loops.front();
      check(noContexts(proveAtlasPackLoopTimingContexts(
                *tagged, genericLoop, tagged->factsOf(genericLoop).entry)),
            "a generic loop cannot mint PACK timing contexts directly");
      auto generic = readAtlasStream(*genericModule, AtlasStreamReadMode::Verification);
      check(succeeded(generic) && generic->loops.size() == 1 &&
                generic->factsOf(generic->loops.front()).access.has_value() &&
                noContexts(generic->timingContexts),
            "generic DMA loop access proof does not establish scalar-overlap evidence");
    }
  }
  // The preheader itself, rather than caller-supplied claims, supplies entry facts.
  for (Operation &op : module->getBody()->getOperations()) {
    auto alu = dyn_cast<ALUImmOp>(&op);
    if (alu && alu.getDst() == fixed().packRowsReg) {
      setI32(&op, "src", fixed().scalarTemporary);
      break;
    }
  }
  auto unknownStream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(unknownStream) && noContexts(unknownStream->timingContexts),
        "a tagged loop with an unknown true preheader cannot create timing proofs");
}
} // namespace

void atlas_test::runScalarTimingContext(MLIRContext &context) {
  negative(context);
}
