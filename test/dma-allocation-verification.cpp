#include "Atlas/AtlasDMAAllocationVerification.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
using Assignments = SmallVector<VirtualDMAAssignment>;

// Each DMA operation's state result keys its transfer, in source order.
struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<Value> handles;
  Fixture(MLIRContext &context, StringRef text, StringRef name = "fixture is typed SSA")
      : module(parse(context, text, name)) {
    if (!module)
      return;
    function = firstFunction(*module);
    function.walk([&](Operation *op) {
      if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op))
        handles.push_back(op->getResult(0));
    });
  }
};

void expect(Fixture &fixture, StringRef name, const Assignments &assignments, bool valid, ArrayRef<StringRef> fragments = {}) {
  expectVerified(name, valid, fragments, [&] { return verifyAtlasDMAAllocation(fixture.function, assignments); });
}

std::string source(StringRef body, StringRef result) {
  return R"mlir(module { func.func @f() -> !atlas.virtual_state {
    %addr = arith.constant -2147483648 : i32
    %n8 = arith.constant 1024 : i32
    %n16 = arith.constant 2048 : i32
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
)mlir" + body.str() + "    return " + result.str() + " : !atlas.virtual_state\n} }";
}

const std::string mixed = source(R"mlir(
    %s1, %x = "atlas.virtual_dma_load_fp8"(%s0, %addr, %n8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %y = "atlas.virtual_dma_load_bf16"(%s1, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s5 = "atlas.virtual_dma_store_bf16"(%s2, %y, %addr, %n16) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %s6 = "atlas.virtual_dma_store_fp8"(%s5, %x, %addr, %n8) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> !atlas.virtual_state
)mlir", "%s6");

const std::string partialRelease = source(R"mlir(
    %s1, %x = "atlas.virtual_dma_load_bf16"(%s0, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %y = "atlas.virtual_dma_load_fp8"(%s1, %addr, %n8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s4, %z = "atlas.virtual_dma_load_bf16"(%s2, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
)mlir", "%s4");

Assignments mixedAssignments(Fixture &fixture) {
  return {{fixture.handles[0], {7, 1, 17, 0x2000, 1, 2, 3}},
          {fixture.handles[1], {2, 2, 41, 0x2100, 1, 2, 3}},
          {fixture.handles[2], {7, 2, 99, 0x2100, 31, 30, 29}},
          {fixture.handles[3], {2, 1, 7, 0x2000, 1, 2, 3}}};
}

void completenessTests(Fixture &f, Fixture &foreign) {
  auto base = mixedAssignments(f);
  constexpr StringLiteral untracked = "foreign, stale, or untracked value", notLaunch = "does not refer to a DMA operation's state result";
  mutated(f, base, "alternative placements and reverse load/store completion", [](Assignments &) {});
  mutated(f, base, "assignment order does not define launch order", [](Assignments &a) { std::reverse(a.begin(), a.end()); });
  mutated(f, base, "missing store assignment", [](Assignments &a) { a.pop_back(); }, {"missing DMA assignment", "assigned transfer"});
  mutated(f, base, "duplicate assignment", [](Assignments &a) { a.push_back(a[0]); }, {"duplicate DMA assignment"});
  mutated(f, base, "null handle", [](Assignments &a) { a[0].transfer = Value(); }, {untracked});
  mutated(f, base, "foreign source handle", [&](Assignments &a) { a[0].transfer = foreign.handles[0]; }, {untracked});
  mutated(f, base, "known non-DMA scalar", [&](Assignments &a) { a[0].transfer = f.function.getBody().front().front().getResult(0); }, {notLaunch});
  mutated(f, base, "a load's tile is not its transfer key",
          [&](Assignments &a) { a[0].transfer = f.handles[0].getDefiningOp()->getResult(1); }, {notLaunch});
}

void geometryTests(Fixture &f) {
  auto base = mixedAssignments(f);
  auto placed = [&](StringRef name, unsigned index, auto mutate, StringRef diagnostic = {}) {
    SmallVector<StringRef> fragments;
    if (!diagnostic.empty())
      fragments = {diagnostic, "assigned transfer"};
    mutated(f, base, name, [&](Assignments &a) { mutate(a[index].placement); }, fragments);
  };
  constexpr StringLiteral capacity = "staging range exceeds selected VMEM capacity", helper = "helper register must be in x1..x31",
                          distinct = "helper registers must be distinct within a transfer";
  for (unsigned index = 0; index < 4; ++index)
    placed("format halves", index, [](auto &p) { p.halves = p.halves == 1 ? 2 : 1; }, "halves do not match the transfer format");
  placed("channel eight", 0, [](auto &p) { p.channel = 8; }, "DMA channel is outside [0, 7]");
  placed("unaligned word address", 0, [](auto &p) { ++p.stagingWord; }, "staging word address must be aligned to 1 KiB");
  // ScalarCore AtlasMemMap gives 0x180000 bytes (393216 words); LSU transfers are 1024 bytes.
  placed("FP8 range ends exactly at VMEM top", 0, [](auto &p) { p.stagingWord = 393216 - 256; });
  placed("BF16 range ends exactly at VMEM top", 1, [](auto &p) { p.stagingWord = 393216 - 512; });
  placed("FP8 range beyond VMEM", 0, [](auto &p) { p.stagingWord = 393216; }, capacity);
  placed("BF16 second half beyond VMEM", 1, [](auto &p) { p.stagingWord = 393216 - 256; }, capacity);
  placed("word end must not wrap uint32", 0, [](auto &p) { p.stagingWord = 0xffffff00; }, capacity);
  for (unsigned reg : {0u, 32u}) {
    placed("invalid staging helper", 0, [=](auto &p) { p.stagingReg = reg; }, helper);
    placed("invalid DRAM helper", 0, [=](auto &p) { p.dramReg = reg; }, helper);
    placed("invalid size helper", 0, [=](auto &p) { p.sizeReg = reg; }, helper);
  }
  placed("staging and DRAM helpers alias", 0, [](auto &p) { p.dramReg = p.stagingReg; }, distinct);
  placed("staging and size helpers alias", 0, [](auto &p) { p.sizeReg = p.stagingReg; }, distinct);
  placed("DRAM and size helpers alias", 0, [](auto &p) { p.sizeReg = p.dramReg; }, distinct);
  mutated(f, base, "arbitrary bounded IDs", [](Assignments &a) {
    a[0].placement.id = 0x7fffffff;
    a[3].placement.id = 0;
  });
  placed("negative i32 bit pattern ID", 0, [](auto &p) { p.id = 0x80000000; }, "DMA transfer id must fit a nonnegative i32");
  placed("duplicate ID after earlier completion", 2, [&](auto &p) { p.id = base[0].placement.id; }, "duplicate DMA transfer id across static launches");
}

// Atomic DMA has no pending interval in virtual IR: transfers may share a
// channel or staging window, and insert-atlas-dma-waits completes the earlier
// one first. The generated DMA checks verify actual overlap.
void reuseTests(MLIRContext &context, Fixture &f, Fixture &released) {
  auto base = mixedAssignments(f);
  mutated(f, base, "loads may share a channel", [](Assignments &a) { a[1].placement.channel = a[0].placement.channel; });
  mutated(f, base, "stores may share a staging window",
          [](Assignments &a) { a[3].placement.stagingWord = a[2].placement.stagingWord + 256; });
  Assignments partial = {{released.handles[0], {5, 2, 1, 0x3100, 4, 7, 9}},
                         {released.handles[1], {1, 1, 2, 0x3300, 4, 7, 9}},
                         {released.handles[2], {5, 2, 3, 0x3100, 4, 7, 9}}};
  mutated(released, partial, "a later load reuses an earlier load's channel and window", [](Assignments &) {});
  Fixture empty(context, source("", "%s0"), "no DMA fixture");
  if (empty.module)
    expect(empty, "no DMA requires no assignments", {}, true);
}
} // namespace

void atlas_test::runDMAAllocation(MLIRContext &context) {
  Fixture fixture(context, mixed, "mixed fixture"), foreign(context, mixed, "foreign fixture"),
      released(context, partialRelease, "partial release fixture");
  if (!fixture.module || !foreign.module || !released.module)
    return;
  completenessTests(fixture, foreign);
  geometryTests(fixture);
  reuseTests(context, fixture, released);
}
