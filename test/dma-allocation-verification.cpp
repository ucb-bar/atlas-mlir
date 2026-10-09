#include "VerificationTestSupport.h"
#include "Atlas/AtlasDMAAllocationVerification.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
using Assignments = SmallVector<VirtualDMAAssignment>;


// These typed SSA fixtures deliberately bypass the compiler's older single-
// pending admission limit. This independent API checks source-handle ownership;
// the tests do not ask the allocator to choose their channels or windows.
struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<Value> transfers;

  bool initialize(MLIRContext &context, StringRef source, StringRef name) {
    module = parseSourceString<ModuleOp>(source, &context);
    check(bool(module), name, "fixture failed to parse");
    if (!module)
      return false;
    bool valid = succeeded(verify(*module));
    check(valid, name, "fixture is not valid typed SSA");
    if (!valid)
      return false;
    function = *module->getOps<func::FuncOp>().begin();
    function.walk([&](Operation *op) {
      if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
              VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op))
        transfers.push_back(op->getResult(1));
    });
    return true;
  }
};

void expect(Fixture &fixture, StringRef name, const Assignments &assignments,
            bool valid, ArrayRef<StringRef> fragments = {}) {
  expectVerified(name, valid, fragments, [&] { return verifyAtlasDMAAllocation(fixture.function, assignments); });
}

constexpr StringLiteral mixed = R"mlir(
module {
  func.func @mixed() -> !atlas.virtual_state {
    %addr = arith.constant -2147483648 : i32
    %n8 = arith.constant 1024 : i32
    %n16 = arith.constant 2048 : i32
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_dma_load_fp8"(%s0, %addr, %n8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s2, %b = "atlas.virtual_dma_load_bf16"(%s1, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %s3, %y = "atlas.virtual_dma_await_bf16"(%s2, %b) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s4, %x = "atlas.virtual_dma_await_fp8"(%s3, %a) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s5, %c = "atlas.virtual_dma_store_bf16"(%s4, %y, %addr, %n16) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s6, %d = "atlas.virtual_dma_store_fp8"(%s5, %x, %addr, %n8) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %s7 = "atlas.virtual_dma_wait"(%s6, %d) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %s8 = "atlas.virtual_dma_wait"(%s7, %c) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %s8 : !atlas.virtual_state
  }
})mlir";

constexpr StringLiteral partialRelease = R"mlir(
module {
  func.func @partial_release() -> !atlas.virtual_state {
    %addr = arith.constant -2147483648 : i32
    %n8 = arith.constant 1024 : i32
    %n16 = arith.constant 2048 : i32
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_dma_load_bf16"(%s0, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %s2, %b = "atlas.virtual_dma_load_fp8"(%s1, %addr, %n8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %s3, %x = "atlas.virtual_dma_await_bf16"(%s2, %a) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s4, %c = "atlas.virtual_dma_load_bf16"(%s3, %addr, %n16) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %s5, %z = "atlas.virtual_dma_await_bf16"(%s4, %c) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s6, %y = "atlas.virtual_dma_await_fp8"(%s5, %b) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    return %s6 : !atlas.virtual_state
  }
})mlir";

Assignments mixedAssignments(Fixture &fixture) {
  return {{fixture.transfers[0], {7, 1, 17, 0x2000, 1, 2, 3}},
          {fixture.transfers[1], {2, 2, 41, 0x2100, 1, 2, 3}},
          {fixture.transfers[2], {7, 2, 99, 0x2100, 31, 30, 29}},
          {fixture.transfers[3], {2, 1, 7, 0x2000, 1, 2, 3}}};
}

void completenessTests(Fixture &fixture, Fixture &foreign) {
  auto base = mixedAssignments(fixture);
  expect(fixture, "alternative placements and reverse load/store completion", base, true);
  auto changed = base;
  std::reverse(changed.begin(), changed.end());
  expect(fixture, "assignment order does not define launch order", changed, true);
  changed = base;
  changed.pop_back();
  expect(fixture, "missing store assignment", changed, false, {"missing DMA assignment", "assigned transfer"});
  changed = base;
  changed.push_back(base[0]);
  expect(fixture, "duplicate assignment", changed, false, {"duplicate DMA assignment"});
  changed = base;
  changed[0].transfer = Value();
  expect(fixture, "null handle", changed, false, {"foreign, stale, or untracked value"});
  changed = base;
  changed[0].transfer = foreign.transfers[0];
  expect(fixture, "foreign source handle", changed, false, {"foreign, stale, or untracked value"});
  changed = base;
  changed[0].transfer = fixture.function.getBody().front().front().getResult(0);
  expect(fixture, "known non-DMA scalar", changed, false, {"does not refer to a transfer launch handle"});
  changed = base;
  changed[0].transfer = fixture.transfers[0].getDefiningOp()->getResult(0);
  expect(fixture, "launch state result is not its transfer handle", changed, false, {"does not refer to a transfer launch handle"});
}

void geometryTests(Fixture &fixture) {
  auto base = mixedAssignments(fixture);
  auto reject = [&](StringRef name, unsigned index, auto mutate, StringRef diagnostic) {
    auto changed = base;
    mutate(changed[index].placement);
    expect(fixture, name, changed, false, {diagnostic, "assigned transfer"});
  };
  for (unsigned index = 0; index < 4; ++index)
    reject("format halves", index, [](auto &p) { p.halves = p.halves == 1 ? 2 : 1; }, "halves do not match the transfer format");
  reject("channel eight", 0, [](auto &p) { p.channel = 8; }, "DMA channel is outside [0, 7]");
  reject("unaligned word address", 0, [](auto &p) { ++p.stagingWord; }, "staging word address must be aligned to 1 KiB");
  // ScalarCore AtlasMemMap gives 0x180000 bytes; LSU transfers are 1024 bytes.
  // The expected endpoints are written independently of verifier constants.
  auto changed = base;
  changed[0].placement.stagingWord = 393216 - 256;
  expect(fixture, "FP8 range ends exactly at VMEM top", changed, true);
  changed = base;
  changed[1].placement.stagingWord = 393216 - 512;
  expect(fixture, "BF16 range ends exactly at VMEM top", changed, true);
  reject("FP8 range beyond VMEM", 0, [](auto &p) { p.stagingWord = 393216; }, "staging range exceeds selected VMEM capacity");
  reject("BF16 second half beyond VMEM", 1, [](auto &p) { p.stagingWord = 393216 - 256; }, "staging range exceeds selected VMEM capacity");
  reject("word end must not wrap uint32", 0, [](auto &p) { p.stagingWord = 0xffffff00; }, "staging range exceeds selected VMEM capacity");
  for (unsigned reg : {0u, 32u}) {
    reject("invalid staging helper", 0, [=](auto &p) { p.stagingReg = reg; }, "helper register must be in x1..x31");
    reject("invalid DRAM helper", 0, [=](auto &p) { p.dramReg = reg; }, "helper register must be in x1..x31");
    reject("invalid size helper", 0, [=](auto &p) { p.sizeReg = reg; }, "helper register must be in x1..x31");
  }
  reject("staging and DRAM helpers alias", 0, [](auto &p) { p.dramReg = p.stagingReg; }, "helper registers must be distinct within a transfer");
  reject("staging and size helpers alias", 0, [](auto &p) { p.sizeReg = p.stagingReg; }, "helper registers must be distinct within a transfer");
  reject("DRAM and size helpers alias", 0, [](auto &p) { p.sizeReg = p.dramReg; }, "helper registers must be distinct within a transfer");
  changed = base;
  changed[0].placement.id = 0x7fffffff;
  changed[3].placement.id = 0;
  expect(fixture, "arbitrary bounded IDs", changed, true);
  reject("negative i32 bit pattern ID", 0, [](auto &p) { p.id = 0x80000000; }, "DMA transfer id must fit a nonnegative i32");
  reject("duplicate ID after earlier completion", 2, [&](auto &p) { p.id = base[0].placement.id; }, "duplicate DMA transfer id across static launches");
}

void ownershipTests(Fixture &fixture, Fixture &released) {
  auto base = mixedAssignments(fixture);
  auto changed = base;
  changed[1].placement.channel = base[0].placement.channel;
  expect(fixture, "live loads share channel", changed, false, {"simultaneously owned DMA channel", "new transfer", "pending transfer", "channel 7"});
  changed = base;
  changed[3].placement.channel = base[2].placement.channel;
  expect(fixture, "live stores share channel", changed, false, {"simultaneously owned DMA channel"});
  changed = base;
  changed[1].placement.stagingWord = base[0].placement.stagingWord - 256;
  changed[1].placement.stagingReg = 4;
  changed[1].placement.dramReg = 5;
  changed[1].placement.sizeReg = 6;
  expect(fixture, "partial VMEM overlap despite different helper names", changed, false, {"overlapping simultaneously owned DMA staging ranges", "new transfer", "pending transfer", "VMEM words [7936, 8448)", "VMEM words [8192, 8448)"});
  changed = base;
  changed[3].placement.stagingWord = base[2].placement.stagingWord + 256;
  expect(fixture, "FP8 store overlaps BF16 second half", changed, false, {"overlapping simultaneously owned DMA staging ranges"});
  Assignments partial = {{released.transfers[0], {5, 2, 1, 0x3100, 4, 7, 9}},
                         {released.transfers[1], {1, 1, 2, 0x3300, 4, 7, 9}},
                         {released.transfers[2], {5, 2, 3, 0x3100, 4, 7, 9}}};
  expect(released, "await releases only its handle; completed resources reused", partial, true);
  changed = partial;
  changed[2].placement.channel = partial[1].placement.channel;
  expect(released, "await one does not release other channel", changed, false, {"simultaneously owned DMA channel"});
  changed = partial;
  changed[2].placement.stagingWord = partial[1].placement.stagingWord;
  expect(released, "await one does not release other staging", changed, false, {"overlapping simultaneously owned DMA staging ranges"});
}

std::string replace(StringRef source, StringRef before, StringRef after) {
  std::string result = source.str();
  size_t position = result.find(before.str());
  check(position != std::string::npos, "completion fixture transformation");
  if (position != std::string::npos)
    result.replace(position, before.size(), after.str());
  return result;
}

void completionTests(MLIRContext &context) {
  Fixture empty;
  if (empty.initialize(context, R"mlir(module { func.func @no_dma() -> !atlas.virtual_state {
        %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
        return %s : !atlas.virtual_state
      } })mlir", "no DMA fixture"))
    expect(empty, "no DMA requires no assignments", {}, true);

  // These typed SSA mutations exercise defensive ownership diagnostics rather
  // than the API's admitted-source precondition or compiler admission verifier.
  StringRef finalWait = "    %s8 = \"atlas.virtual_dma_wait\"(%s7, %c) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state\n";
  std::string source = replace(mixed, finalWait, "");
  source = replace(source, "return %s8", "return %s7");
  Fixture missing;
  if (missing.initialize(context, source, "missing completion fixture"))
    expect(missing, "missing completion at block exit", mixedAssignments(missing), false, {"ownership remains pending at block exit"});

  source = replace(mixed, "    return %s8", "    %s9 = \"atlas.virtual_dma_wait\"(%s8, %c) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state\n    return %s9");
  Fixture repeated;
  if (repeated.initialize(context, source, "repeated completion fixture"))
    expect(repeated, "same handle cannot complete twice", mixedAssignments(repeated), false, {"completion requires a pending block-local transfer handle"});

  source = replace(mixed, "    %s3, %y =", "    cf.br ^next(%s2 : !atlas.virtual_state)\n  ^next(%edge: !atlas.virtual_state):\n    %s3, %y =");
  source = replace(source, "(%s2, %b)", "(%edge, %b)");
  Fixture crossing;
  if (crossing.initialize(context, source, "cross-block completion fixture"))
    expect(crossing, "DMA handle cannot remain pending across a block edge", mixedAssignments(crossing), false, {"ownership remains pending at block exit"});
}
} // namespace

void atlas_test::runDMAAllocation(MLIRContext &context) {
  Fixture fixture, foreign, released;
  if (fixture.initialize(context, mixed, "mixed fixture") && foreign.initialize(context, mixed, "foreign fixture") &&
      released.initialize(context, partialRelease, "partial release fixture")) {
    completenessTests(fixture, foreign);
    geometryTests(fixture);
    ownershipTests(fixture, released);
    completionTests(context);
  }
}
