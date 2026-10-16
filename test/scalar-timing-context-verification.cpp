#include "Atlas/AtlasContractAttr.h"
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

// Test-only bounds discovery; production recognition is performed by the
// timing-context adapter using actual targets and complete incoming CFG edges.
struct TestLoop {
  size_t begin, branch;
  size_t end() const { return branch + 2; }
  bool contains(size_t pc) const { return begin <= pc && pc < end(); }
};
std::optional<TestLoop> loopOf(const AtlasStream &stream) {
  for (size_t b = 0; b < stream.starts.size(); ++b) {
    size_t end = stream.blockEnd(b);
    if (!stream.endsInBranch(b)) continue;
    Operation *op = stream.ops[end - 2];
    auto helper = op->getAttrOfType<StringAttr>(kAtlasTagCFGHelper);
    if (helper && helper.getValue() == "pack") return TestLoop{stream.starts[b], end - 2};
  }
  return std::nullopt;
}
RegValues trueEntry(const AtlasStream &stream, const TestLoop &loop) {
  size_t block = llvm::find(stream.starts, loop.begin) - stream.starts.begin();
  RegValues entry = stream.entry[block - 1];
  for (size_t pc = stream.starts[block - 1]; pc < loop.begin; ++pc)
    applyScalar(stream.instrs[pc], entry);
  return entry;
}

void positive(MLIRContext &context) {
  auto module = lower(context);
  if (!module) return;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && bool(loopOf(*stream)), "context source has a bounded PACK loop");
  if (failed(stream) || !loopOf(*stream)) return;
  const auto loop = *loopOf(*stream);
  unsigned load = 0, store = 0;
  TimingContext first;
  for (size_t pc = 0; pc < stream->instrs.size(); ++pc) {
    const Instr &in = stream->instrs[pc];
    const TimingContext &proof = stream->timingContexts[pc];
    bool memory = loop.contains(pc) && (in.op->name == "lw" || in.op->name == "sw");
    check(bool(proof) == memory, "only scalar copy instructions carry contexts");
    if (!proof) continue;
    check(proof->matches(in), "context binds all decoded instruction fields");
    Instr changed = in;
    ++changed.imm;
    check(!proof->matches(changed), "a changed effective-address immediate invalidates the context");
    changed = in;
    ++changed.rs1;
    check(!proof->matches(changed), "a changed address role invalidates the context");
    if (!first) first = proof;
    check(first->sameRegion(*proof), "all copy contexts share one replay region");
    bool isLoad = in.op->name == "lw";
    unsigned copy = isLoad ? load++ : store++;
    for (unsigned row = 0; row < 32; ++row) {
      uint32_t expected = isLoad ? fixed().packWord * 4 + 512 * (copy % 2) + 16 * row + 4 * (copy / 2)
                                : fixed().packRelayoutWord * 4 + 32 * row + 16 * (copy % 2) + 4 * (copy / 2);
      check(proof->scalarAddresses()[row] == expected, "every replayed row has its exact effective byte address");
    }
  }
  check(load == 8 && store == 8, "all sixteen scalar instructions have complete 32-row proofs");
  auto reread = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(reread) && !first->sameRegion(*reread->timingContexts[loop.begin]),
        "a new decoded snapshot recomputes its own immutable region");

  std::vector<Instr> nodes(stream->instrs.begin() + loop.begin, stream->instrs.begin() + loop.end());
  std::vector<TimingContext> contexts(stream->timingContexts.begin() + loop.begin,
                                      stream->timingContexts.begin() + loop.end());
  auto provider = builtinTimingProvider();
  unsigned queries = 0;
  provider.contextualDependence = [&](const Instr &a, const Footprint &fa, const TimingContext &ca,
                                     const Instr &b, const Footprint &fb, const TimingContext &cb) {
    if (ca) { check(ca->matches(a), "graph retains the first instruction's proof"); ++queries; }
    if (cb) { check(cb->matches(b), "graph retains the second instruction's proof"); ++queries; }
    return provider.dependence(a, fa, b, fb);
  };
  auto graph = buildTimingGraph(nodes, trueEntry(*stream, loop), provider, nullptr, &contexts);
  check(graph.error.empty() && queries > 0, "graph construction supplies contexts to dependence queries");
  contexts.pop_back();
  check(!buildTimingGraph(nodes, unknownRegs(), provider, nullptr, &contexts).error.empty(),
        "graph rejects a context vector for a different instruction snapshot");
}

void negative(MLIRContext &context) {
  auto module = lower(context);
  if (!module) return;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  if (failed(stream) || !loopOf(*stream)) return;
  const auto loop = *loopOf(*stream);
  RegValues entry = trueEntry(*stream, loop);
  for (uint32_t rows : {31, 33, 0, 257}) {
    RegValues changed = entry;
    changed[fixed().packRowsReg] = rows;
    check(noContexts(proveAtlasScalarLoopTimingContexts(*stream, loop.begin, loop.branch, changed)),
          "only a decided exact 32-iteration replay carries timing proofs");
  }
  RegValues unknown = entry;
  unknown[fixed().packRowsReg].reset();
  check(noContexts(proveAtlasScalarLoopTimingContexts(*stream, loop.begin, loop.branch, unknown)), "unknown branch exit has no timing proof");
  unknown = entry;
  unknown[fixed().packSourceRegs[0]].reset();
  check(noContexts(proveAtlasScalarLoopTimingContexts(*stream, loop.begin, loop.branch, unknown)), "unknown source address has no timing proof");
  for (uint32_t bad : {uint32_t(*entry[fixed().packSourceRegs[0]] + 1), uint32_t(kVmemBytes - 496),
                       uint32_t(0x200000), uint32_t(0xfffffffc)}) {
    RegValues changed = entry;
    changed[fixed().packSourceRegs[0]] = bad;
    check(noContexts(proveAtlasScalarLoopTimingContexts(*stream, loop.begin, loop.branch, changed)),
          "unaligned, final-row out-of-range and wrapping addresses fail closed");
  }
  size_t block = llvm::find(stream->starts, loop.begin) - stream->starts.begin();
  AtlasStream alternate = *stream;
  alternate.succs.back().push_back(block);
  populateAtlasInstructionTimingContexts(alternate);
  check(noContexts(alternate.timingContexts), "a second incoming CFG edge cannot acquire timing proofs");
  AtlasStream indirect = *stream;
  indirect.instrs.back().op = findOp("jalr");
  populateAtlasInstructionTimingContexts(indirect);
  check(noContexts(indirect.timingContexts), "an unproven indirect redirect excludes timing proofs");
  AtlasStream unsupported = *stream;
  unsupported.instrs[loop.begin].op = findOp("lb");
  populateAtlasInstructionTimingContexts(unsupported);
  check(noContexts(unsupported.timingContexts), "a tagged unsupported scalar width cannot acquire word proofs");
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
  positive(context);
  negative(context);
}
