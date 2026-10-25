#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasBufferContractVerification.h"
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

void positive(MLIRContext &context) {
  auto module = lower(context);
  check(bool(module), "ungrouped PACK context fixture lowers", diagnostics);
  if (!module) return;
  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && stream->loops.size() == 1, "context source has one proved PACK loop");
  if (failed(stream) || stream->loops.size() != 1) return;
  const auto &loop = stream->loops.front();
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
  check(succeeded(reread) && first && reread->timingContexts[loop.begin] &&
            !first->sameRegion(*reread->timingContexts[loop.begin]),
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
  auto graph = buildTimingGraph(nodes, stream->entry[1], provider, nullptr, &contexts);
  check(graph.error.empty() && queries > 0, "graph construction supplies contexts to dependence queries");
  contexts.pop_back();
  check(!buildTimingGraph(nodes, unknownRegs(), provider, nullptr, &contexts).error.empty(),
        "graph rejects a context vector for a different instruction snapshot");
}

void grouped(MLIRContext &context) {
  auto module = lower(context);
  check(bool(module), "grouped PACK context fixture lowers", diagnostics);
  if (!module) return;
  auto original = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(original) && original->loops.size() == 1, "grouped fixture has one proved PACK loop");
  if (failed(original) || original->loops.size() != 1) return;
  const auto &oldLoop = original->loops.front();
  check(!oldLoop.delaySlot, "grouped fixture starts before delay-slot insertion");
  if (oldLoop.delaySlot) return;
  Operation *branch = original->ops[oldLoop.branch];
  std::vector<Operation *> memory, updates, oldBody;
  for (size_t pc = oldLoop.begin; pc < oldLoop.branch; ++pc) {
    Operation *op = original->ops[pc];
    oldBody.push_back(op);
    if (isa<ScalarLoadOp, ScalarStoreOp>(op)) memory.push_back(op);
    else if (isa<ALUImmOp>(op)) updates.push_back(op);
  }
  // Construct two rows independently of the model lowering. Scalar memory
  // accesses keep their decoded order; all four loop updates move to the end.
  OpBuilder builder(branch);
  unsigned bodyWords = 0;
  Value state = oldBody.front()->getOperand(0);
  auto clone = [&](Operation *op) {
    Operation *copy = builder.clone(*op);
    copy->setOperand(0, state);
    state = copy->getResult(0);
    ++bodyWords;
    return copy;
  };
  for (unsigned row = 0; row < kPackRelayoutUnrollFactor; ++row)
    for (Operation *op : memory) {
      Operation *copy = clone(op);
      int64_t offset = op->getAttrOfType<IntegerAttr>("offset").getInt();
      offset += row * (isa<ScalarLoadOp>(op) ? kAtlasMregRowBytes / 2 : kAtlasMregRowBytes);
      setI32(copy, "offset", offset);
    }
  for (Operation *op : updates) {
    Operation *copy = clone(op);
    setI32(copy, "immediate", op->getAttrOfType<IntegerAttr>("immediate").getInt() * kPackRelayoutUnrollFactor);
  }
  branch->setOperand(0, state);
  for (Operation *op : llvm::reverse(oldBody)) op->erase();
  setI32(branch, "offset_bytes", -2 * int64_t(bodyWords));

  auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(stream) && stream->loops.size() == 1, "grouped copy remains a single decidable loop");
  if (failed(stream) || stream->loops.size() != 1) return;
  const auto &loop = stream->loops.front();
  constexpr unsigned group = kPackRelayoutUnrollFactor;
  constexpr unsigned iterations = kAtlasMregRows / group;
  unsigned loads = 0, stores = 0;
  for (size_t pc = loop.begin; pc < loop.end(); ++pc) {
    const auto &in = stream->instrs[pc];
    const auto &proof = stream->timingContexts[pc];
    bool isLoad = in.op->name == "lw", isStore = in.op->name == "sw";
    check(bool(proof) == (isLoad || isStore), "grouped copy proves every scalar word and no loop update");
    if (!proof) continue;
    unsigned copy = isLoad ? loads++ : stores++;
    unsigned rowInGroup = copy / kPackRelayoutRowWords;
    unsigned word = copy % kPackRelayoutRowWords;
    for (unsigned ordinal = 0; ordinal < proof->scalarAddresses().size(); ++ordinal) {
      unsigned row = group * (ordinal % iterations) + rowInGroup;
      uint32_t expected = isLoad ? fixed().packWord * 4 + kAtlasRegisterBytes / 2 * (word % 2) +
                                      kAtlasMregRowBytes / 2 * row + 4 * (word / 2)
                                : fixed().packRelayoutWord * 4 + kAtlasMregRowBytes * row +
                                      kAtlasMregRowBytes / 2 * (word % 2) + 4 * (word / 2);
      check(proof->scalarAddresses()[ordinal] == expected,
            "grouped proof repeats exact finite cases at the shared iteration ordinal");
    }
  }
  check(loads == group * kPackRelayoutRowWords && stores == loads,
        "grouped copy carries complete load and store proofs");
  expectVerified("grouped replay preserves every PACK interleave word", true, {},
                 [&] { return verifyAtlasGeneratedBufferContract(*module); });

  RegValues incomplete = stream->factsOf(loop).entry;
  incomplete[fixed().packRowsReg] = kAtlasMregRows - group;
  check(noContexts(proveAtlasPackLoopTimingContexts(*stream, loop, incomplete)),
        "a grouped replay missing one group has no timing proof");
  AtlasStream malformed = *stream;
  malformed.instrs[loop.begin].op = findOp("addi");
  check(noContexts(proveAtlasPackLoopTimingContexts(malformed, loop, stream->factsOf(loop).entry)),
        "unequal decoded copy counts cannot create grouped timing proofs");
  auto reread = readAtlasStream(*module, AtlasStreamReadMode::Verification);
  check(succeeded(reread) && stream->timingContexts[loop.begin] &&
            !stream->timingContexts[loop.begin]->sameRegion(*reread->timingContexts[loop.begin]),
        "grouped contexts are recomputed for each decoded snapshot");
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
  positive(context);
  grouped(context);
  negative(context);
}
