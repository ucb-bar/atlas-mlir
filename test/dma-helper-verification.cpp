#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "VerificationTestSupport.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral state = "!atlas.virtual_state";
constexpr StringLiteral clobber = "DMA helper write clobbers live scalar value";
constexpr StringLiteral pendingBase = "pending DMA staging base is not preserved";
constexpr StringLiteral halfSize = "persistent DMA half-size helper is not preserved";
using Policy = DMAAwaitBasePolicy;
using P = DMATransferPlacement;

// One DMA launch and completion between boundary I/O, inside a "loop", before a "join", or on one arm of a "diamond".
// `%keep` is defined before the launch or, with `keepAfterLaunch`, after it; `keepLive` reads it (or, with
// `useOperands`, the address and size) after completion.
std::string source(bool bf16, bool store, bool keepAfterLaunch = false, bool keepLive = true, StringRef cfg = "", bool useOperands = false) {
  std::string s = state.str(), fmt = bf16 ? "bf16" : "fp8", tile = "!atlas.virtual_" + fmt;
  std::string event = store ? "!atlas.virtual_dma_store" : "!atlas.virtual_dma_load_" + fmt;
  std::string body = "module { func.func @test() -> " + s + " {\n    %s0 = \"atlas.virtual_start\"() : () -> " + s + "\n";
  body += "    %addr = arith.constant -2147483648 : i32\n    %size = arith.constant " + std::to_string(bf16 ? 2048 : 1024) + " : i32\n";
  std::string keep = "    %keep = arith.constant 7 : i32\n";
  if (!keepAfterLaunch)
    body += keep;
  body += "    %ready, %observable = \"atlas.virtual_input_bf16\"(%s0) {index = 1 : i32} : (" + s + ") -> (" + s + ", !atlas.virtual_bf16)\n";
  std::string before = "%ready", final = "%done";
  if (cfg == "loop") {
    body += "    cf.br ^loop(%ready : " + s + ")\n  ^loop(%state: " + s + "):\n";
    before = "%state";
  } else if (cfg == "diamond") {
    body += "    %choose = arith.constant true\n    cf.cond_br %choose, ^left(%ready : " + s + "), ^right(%ready : " + s + ")\n  ^left(%state: " + s + "):\n";
    before = "%state";
  }
  if (store) {
    body += "    %input, %tile = \"atlas.virtual_input_" + fmt + "\"(" + before + ") {index = 0 : i32} : (" + s + ") -> (" + s + ", " + tile + ")\n";
    before = "%input";
  }
  body += "    %issued, %event = \"atlas.virtual_dma_" + std::string(store ? "store_" : "load_") + fmt + "\"(" + before + (store ? ", %tile" : "");
  body += ", %addr, %size) : (" + s + (store ? ", " + tile : "") + ", i32, i32) -> (" + s + ", " + event + ")\n";
  if (keepAfterLaunch)
    body += keep;
  body += store ? "    %done = \"atlas.virtual_dma_wait\"(%issued, %event)" : "    %done, %tile = \"atlas.virtual_dma_await_" + fmt + "\"(%issued, %event)";
  body += " : (" + s + ", " + event + ") -> " + (store ? s : "(" + s + ", " + tile + ")") + "\n";
  if (cfg == "join" || cfg == "diamond") {
    body += "    cf.br ^exit(%done : " + s + ")\n";
    if (cfg == "diamond")
      body += "  ^right(%right_io: " + s + "):\n    cf.br ^exit(%right_io : " + s + ")\n";
    body += "  ^exit(%edge: " + s + "):\n";
    final = "%edge";
  }
  if (keepLive)
    body += useOperands ? "    %after = arith.addi %addr, %size : i32\n" : "    %after = arith.addi %keep, %keep : i32\n";
  if (cfg == "loop") {
    body += "    %again = arith.constant false\n    cf.cond_br %again, ^loop(%done : " + s + "), ^exit(%done : " + s + ")\n  ^exit(%edge: " + s + "):\n";
    final = "%edge";
  }
  body += "    %out = \"atlas.virtual_output_bf16\"(" + final + ", %observable) {index = 0 : i32} : (" + s + ", !atlas.virtual_bf16) -> " + s + "\n";
  return body + "    return %out : " + s + "\n} }";
}

// Every register and DMA placement is chosen here, independent of the allocator. Unique baseline registers
// (scalars from x10, so %addr is x10, %size x11 and %keep x12) isolate helper writes from SSA overlap.
struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualDMAAssignment> transfers;
  VirtualDMAAssignment dma;
  Value addr, size, keep;

  Fixture(MLIRContext &context, StringRef text, bool bf16, bool admit = true) {
    module = parseSourceString<ModuleOp>(text, &context);
    bool valid = module && succeeded(verify(*module)) && (!admit || succeeded(verifyAtlasVirtualModule(*module)));
    check(valid, admit ? "fixture is an admitted virtual CFG" : "fixture is verified typed SSA");
    if (!valid) {
      module = {};
      return;
    }
    function = firstFunction(*module);
    unsigned scalar = 10, tensor = 0;
    function.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (result.getType().isInteger(1) || result.getType().isInteger(32))
          registers.push_back({result, scalar++});
        else if (isa<VirtualBF16Type, VirtualFP8Type>(result.getType()))
          registers.push_back({result, std::exchange(tensor, tensor + 2)});
      }
      if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
        dma = {op->getResult(1), {6, bf16 ? 2u : 1u, 37, 0x2100, 4, 7, 9}};
        transfers.push_back(dma);
      }
    });
    addr = registers[0].value;
    size = registers[1].value;
    for (auto assignment : registers)
      if (assignment.value.getType().isInteger(32) && assignment.reg == 12)
        keep = assignment.value;
  }
  explicit operator bool() const { return bool(module); }
  unsigned reg(Value value) const {
    for (auto assignment : registers)
      if (assignment.value == value)
        return assignment.reg;
    llvm_unreachable("fixture value has no claimed register");
  }
};

FixedResourcePlacement fixedPlacement() {
  FixedResourcePlacement fixed{};
  fixed.tensorTemporary = 62; fixed.scalarTemporary = 27; fixed.halfSizeReg = 2;
  fixed.haltReg = 1; fixed.inputBaseReg = 6; fixed.inputDramReg = 1; fixed.outputBaseReg = 8; fixed.outputDramReg = 3;
  fixed.dmaBaseReg = 4; fixed.dmaDramReg = 7; fixed.dmaSizeReg = 9;
  return fixed;
}

void expectAssignments(Fixture &fixture, StringRef name, ArrayRef<VirtualDMAAssignment> assignments, bool valid,
                       ArrayRef<StringRef> fragments = {}, Policy policy = Policy::Preserved,
                       const FixedResourcePlacement &fixed = fixedPlacement()) {
  expectVerified(name, valid, fragments,
                 [&] { return verifyAtlasRegisterAllocation(fixture.function, fixture.registers, fixed, {}, assignments, policy); });
}

// Checks the fixture's single transfer under `placement`.
void expect(Fixture &fixture, StringRef name, P placement, bool valid, ArrayRef<StringRef> fragments = {},
            Policy policy = Policy::Preserved) {
  VirtualDMAAssignment dma{fixture.dma.transfer, placement};
  expectAssignments(fixture, name, dma, valid, fragments, policy);
}

void launchTests(MLIRContext &context, bool bf16, bool store) {
  Fixture f(context, source(bf16, store), bf16), self(context, source(bf16, store, false, true, "", true), bf16);
  if (!f || !self)
    return;
  auto base = f.dma.placement;
  expect(f, "independent placement baseline", base, true);
  expectAssignments(f, "omitted DMA assignments cannot bypass helper checks", {}, false, {"missing DMA assignment"});
  for (unsigned P::*role : {&P::dramReg, &P::sizeReg, &P::stagingReg}) {
    auto changed = base;
    changed.*role = f.reg(f.keep);
    // Reverse liveness meets a BF16 load's later await staging write first.
    StringRef phase = role == &P::dramReg   ? "DRAM capture"
                      : role == &P::sizeReg ? "size capture"
                      : bf16 && !store      ? "await staging materialization"
                                            : "launch staging materialization";
    expect(f, "helper clobbers live-after scalar", changed, false, {clobber, phase, "x12", "live scalar"});
  }
  auto changed = base;
  changed.dramReg = f.reg(f.size);
  expect(f, "first copy destroys unread size operand", changed, false, {"DMA DRAM helper write clobbers size operand before capture", "x11"});
  changed = self.dma.placement;
  changed.dramReg = self.reg(self.addr);
  changed.sizeReg = self.reg(self.size);
  expect(self, "exact self copies preserve live source operands", changed, true);
  changed = base;
  changed.sizeReg = f.reg(f.addr);
  expect(f, "size copy reuses address after its capture", changed, true);
  for (Value dead : {f.addr, f.size}) {
    changed = base;
    changed.stagingReg = f.reg(dead);
    expect(f, "staging write reuses fully captured dead operand", changed, true);
  }
}

void awaitTests(MLIRContext &context, bool bf16, bool store) {
  Fixture live(context, source(bf16, store, true), bf16), dead(context, source(bf16, store, true, false), bf16);
  if (!live || !dead)
    return;
  auto changed = live.dma.placement;
  changed.stagingReg = live.reg(live.keep);
  for (auto policy : {Policy::Preserved, Policy::Rematerialized}) {
    bool preserved = policy == Policy::Preserved;
    SmallVector<StringRef> fragments;
    if (!store)
      fragments = !bf16 && preserved ? SmallVector<StringRef>{pendingBase} : SmallVector<StringRef>{clobber, "await staging materialization", "x12"};
    expect(live, "await staging policy and store wait", changed, store, fragments, policy);
    changed.stagingReg = dead.reg(dead.keep);
    bool valid = store || !preserved;
    expect(dead, "dead scalar reuse still preserves an unread staging base", changed, valid, valid ? ArrayRef<StringRef>() : ArrayRef<StringRef>(pendingBase), policy);
  }
}

void pendingTests(MLIRContext &context) {
  for (bool bf16 : {false, true}) {
    std::string text = source(bf16, false, true, false);
    Fixture f(context, text, bf16), restored(context, replace(text, "%keep = arith.constant 7", "%keep = arith.constant 8448"), bf16);
    if (!f || !restored)
      continue;
    auto changed = f.dma.placement;
    changed.stagingReg = f.reg(f.keep);
    expect(f, "scalar result destroys pending first-half base", changed, false, {pendingBase});
    expect(f, "rematerialized await permits scalar reuse", changed, true, {}, Policy::Rematerialized);
    changed = restored.dma.placement;
    changed.stagingReg = restored.reg(restored.keep);
    expect(restored, "known scalar restoration preserves pending first-half base", changed, true);
  }

  // Typed SSA bypasses only the baseline admission limit on pending handles.
  Fixture two(context, R"mlir(module { func.func @two() -> !atlas.virtual_state {
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %addr = arith.constant -2147483648 : i32
  %size = arith.constant 2048 : i32
  %small = arith.constant 1024 : i32
  %s1, %a = "atlas.virtual_dma_load_bf16"(%s0, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
  %s2, %b = "atlas.virtual_dma_load_fp8"(%s1, %addr, %small) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
  %s3, %x = "atlas.virtual_dma_await_bf16"(%s2, %a) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %s4, %y = "atlas.virtual_dma_await_fp8"(%s3, %b) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %s5 = "atlas.virtual_output_bf16"(%s4, %x) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %s5 : !atlas.virtual_state
} })mlir", true, false);
  if (!two)
    return;
  SmallVector<VirtualDMAAssignment> base = {{two.transfers[0].transfer, {6, 2, 37, 0x2100, 4, 7, 9}},
                                            {two.transfers[1].transfer, {7, 1, 38, 0x2400, 6, 8, 3}}};
  expectAssignments(two, "BF16 first-half read precedes own second-half base write", base, true);
  for (unsigned P::*role : {&P::dramReg, &P::sizeReg, &P::stagingReg}) {
    auto changed = base;
    changed[1].placement.*role = 4;
    expectAssignments(two, "other DMA helper destroys retained staging base", changed, false, {pendingBase});
    expectAssignments(two, "rematerialized awaits allow cross-DMA helper reuse", changed, true, {}, Policy::Rematerialized);
  }
  auto restored = base;
  restored[0].placement.stagingWord = 1024;
  restored[1].placement.sizeReg = 4;
  expectAssignments(two, "same-value DMA size capture preserves another pending base", restored, true);
}

void halfSizeTests(MLIRContext &context) {
  std::string load = source(true, false, false, false);
  for (bool bf16 : {false, true}) {
    Fixture f(context, source(bf16, false, false, false), bf16);
    if (!f)
      continue;
    auto changed = f.dma.placement;
    changed.sizeReg = 2;
    expect(f, "persistent half-size content checked at later boundary read", changed, !bf16, bf16 ? ArrayRef<StringRef>(halfSize) : ArrayRef<StringRef>());
  }

  // The boundary output moves before the load, so the load follows the last boundary read.
  std::string output = "    %out = \"atlas.virtual_output_bf16\"(%ready, %observable) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state\n";
  std::string text = replace(load, replace(output, "%ready", "%done"), "");
  text = replace(replace(text, "    %issued,", output + "    %issued,"), "load_bf16\"(%ready", "load_bf16\"(%out");
  if (Fixture f(context, replace(text, "return %out", "return %done"), true); f) {
    auto changed = f.dma.placement;
    changed.sizeReg = 2;
    expect(f, "half-size helper may be reused after last boundary read", changed, true);
  }

  text = replace(load, "    %out =", "    %small = arith.constant 1024 : i32\n"
      "    %restore, %second = \"atlas.virtual_dma_load_fp8\"(%done, %addr, %small) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)\n"
      "    %restored, %unused = \"atlas.virtual_dma_await_fp8\"(%restore, %second) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n"
      "    %out =");
  if (Fixture f(context, replace(text, "output_bf16\"(%done", "output_bf16\"(%restored"), true); f) {
    SmallVector<VirtualDMAAssignment> assignments = {{f.transfers[0].transfer, {6, 2, 37, 0x2100, 4, 7, 2}},
                                                     {f.transfers[1].transfer, {6, 1, 38, 0x2300, 4, 7, 2}}};
    expectAssignments(f, "FP8 capture restores half-size before later boundary", assignments, true);
  }
}

void halfSizeCFGTests(MLIRContext &context) {
  for (bool bf16 : {false, true})
    for (bool loop : {false, true}) {
      Fixture f(context, source(bf16, false, false, false, loop ? "loop" : "diamond"), bf16);
      if (!f)
        continue;
      auto changed = f.dma.placement;
      expect(f, loop ? "loop half-size baseline" : "diamond half-size baseline", changed, true);
      changed.sizeReg = 2;
      expect(f, loop ? "backedge helper content reaches exit boundary" : "join merges helper contents from both predecessor paths",
             changed, !bf16, bf16 ? ArrayRef<StringRef>(halfSize) : ArrayRef<StringRef>());
    }
}

void packHelperTests(MLIRContext &context) {
  Fixture f(context, replace(source(true, false, false, false), "    %out =",
      "    %packed = \"atlas.virtual_pack_fp8\"(%tile) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n    %out ="), true);
  if (!f)
    return;
  for (auto &assignment : f.registers)
    if (assignment.value.getType().isInteger(1) || assignment.value.getType().isInteger(32))
      assignment.reg += 8;
  auto fixed = fixedPlacement();
  fixed.scaleReg = 3; fixed.outputWord = 65536; fixed.packWord = 32768; fixed.packRelayoutWord = 33024;
  fixed.packSourceRegs = {10, 11}; fixed.packDestinationReg = 12; fixed.packRowReg = 13; fixed.packRowsReg = 14; fixed.packTemporaryRegs = {16, 17};
  expectAssignments(f, "pack preserves unrelated persistent half-size helper", f.transfers, true, {}, Policy::Preserved, fixed);
  fixed.packRowsReg = fixed.halfSizeReg;
  expectAssignments(f, "pack row-count write destroys later boundary half-size", f.transfers, false, {halfSize}, Policy::Preserved, fixed);
  fixed.packRowsReg = 14;
  fixed.packTemporaryRegs[0] = fixed.halfSizeReg;
  expectAssignments(f, "pack temporary makes later boundary half-size unknown", f.transfers, false, {halfSize}, Policy::Preserved, fixed);
}

constexpr StringLiteral scalarSwap = R"mlir(
module { func.func @scalar_swap() -> !atlas.virtual_state {
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %addr = arith.constant -2147483648 : i32
  %size = arith.constant 2048 : i32
  %a = arith.constant 7 : i32
  %b = arith.constant 9 : i32
  %first = arith.constant true
  %stop = arith.constant false
  %issued, %event = "atlas.virtual_dma_load_bf16"(%s0, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
  %ready, %tile = "atlas.virtual_dma_await_bf16"(%issued, %event) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  cf.br ^loop(%ready, %a, %b, %first : !atlas.virtual_state, i32, i32, i1)
^loop(%loop_io: !atlas.virtual_state, %left: i32, %right: i32, %again: i1):
  cf.cond_br %again, ^loop(%loop_io, %right, %left, %stop : !atlas.virtual_state, i32, i32, i1), ^exit(%loop_io : !atlas.virtual_state)
^exit(%exit_io: !atlas.virtual_state):
  %out = "atlas.virtual_output_bf16"(%exit_io, %tile) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %out : !atlas.virtual_state
} })mlir";

void scalarCopyHelperTests(MLIRContext &context) {
  for (bool cycle : {false, true}) {
    std::string text = cycle ? scalarSwap.str() : replace(scalarSwap.str(), "%loop_io, %right, %left, %stop", "%loop_io, %left, %right, %stop");
    Fixture f(context, text, true);
    if (!f)
      continue;
    for (Block &block : f.function.getBody())
      if (block.getNumArguments() == 4)
        for (unsigned i = 1; i < 4; ++i)
          f.registers.push_back({block.getArgument(i), 11 + i});
    auto fixed = fixedPlacement();
    expectAssignments(f, "scalar edge-copy baseline preserves half-size", f.transfers, true, {}, Policy::Preserved, fixed);
    fixed.halfSizeReg = fixed.scalarTemporary;
    expectAssignments(f, cycle ? "swap cycle clobbers aliased half-size helper" : "noncyclic edge avoids scalar temporary write", f.transfers,
                      !cycle, cycle ? ArrayRef<StringRef>(halfSize) : ArrayRef<StringRef>(), Policy::Preserved, fixed);
  }
}

void cfgTests(MLIRContext &context) {
  Fixture through(context, source(false, false, false, true, "join"), false),
      awaitThrough(context, source(true, false, true, true, "join"), true), loop(context, source(false, false, false, false, "loop"), false);
  if (!through || !awaitThrough || !loop)
    return;
  auto changed = through.dma.placement;
  expect(through, "CFG live-through baseline", changed, true);
  changed.dramReg = through.reg(through.keep);
  expect(through, "launch helper clobbers successor live-through scalar", changed, false, {clobber, "DRAM capture"});
  changed = awaitThrough.dma.placement;
  changed.stagingReg = awaitThrough.reg(awaitThrough.keep);
  expect(awaitThrough, "BF16 await clobbers successor live-through scalar", changed, false, {clobber, "await staging materialization"});
  changed = loop.dma.placement;
  expect(loop, "loop baseline", changed, true);
  changed.stagingReg = loop.reg(loop.addr);
  expect(loop, "backedge keeps address operand live after launch", changed, false, {clobber, "launch staging materialization", "x10"});
}
} // namespace

void atlas_test::runDMAHelper(MLIRContext &context) {
  for (bool bf16 : {false, true})
    for (bool store : {false, true}) {
      launchTests(context, bf16, store);
      awaitTests(context, bf16, store);
    }
  cfgTests(context);
  pendingTests(context);
  halfSizeTests(context);
  halfSizeCFGTests(context);
  packHelperTests(context);
  scalarCopyHelperTests(context);
}
