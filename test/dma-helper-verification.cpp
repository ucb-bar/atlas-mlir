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
constexpr StringLiteral halfSize = "persistent DMA half-size helper is not preserved";
using Policy = DMAAwaitBasePolicy;
using P = DMATransferPlacement;

// One atomic DMA between boundary I/O, inside a "loop", before a "join", or on one arm of a "diamond". A load's
// tile is unused, so lowering places its staging VLOADs before the block's terminator. `%keep` is defined before the
// DMA or, with `keepAfterLaunch`, after it; `keepLive` reads it (or, with `useOperands`, the address and size)
// after that terminator's block point. `%held` keeps the address and size in registers unless `immediates`, when they
// are DMA-only constants that each launch builds.
std::string source(bool bf16, bool store, bool keepAfterLaunch = false, bool keepLive = true, StringRef cfg = "", bool useOperands = false,
                   bool immediates = false) {
  std::string s = state.str(), fmt = bf16 ? "bf16" : "fp8", tile = "!atlas.virtual_" + fmt;
  std::string body = "module { func.func @test() -> " + s + " {\n    %s0 = \"atlas.virtual_start\"() : () -> " + s + "\n";
  body += "    %addr = arith.constant -2147483648 : i32\n    %size = arith.constant " + std::to_string(bf16 ? 2048 : 1024) + " : i32\n";
  std::string keep = "    %keep = arith.constant 7 : i32\n";
  if (!keepAfterLaunch)
    body += keep;
  if (!immediates)
    body += "    %held = arith.addi %addr, %size : i32\n";
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
  body += std::string(store ? "    %done" : "    %done, %tile") + " = \"atlas.virtual_dma_" + (store ? "store_" : "load_") + fmt + "\"(" + before + (store ? ", %tile" : "");
  body += ", %addr, %size) : (" + s + (store ? ", " + tile : "") + ", i32, i32) -> " + (store ? s : "(" + s + ", " + tile + ")") + "\n";
  if (keepAfterLaunch)
    body += keep;
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

// Unique baseline registers (scalars from x10 in definition order, so %addr is x10, %size x11 and a %keep defined
// before the DMA x12) isolate helper writes from SSA overlap. A DMA immediate claims none but keeps its number.
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
        if (isVirtualDMAImmediate(result))
          ++scalar;
        else if (result.getType().isInteger(1) || result.getType().isInteger(32))
          registers.push_back({result, scalar++});
        else if (isa<VirtualBF16Type, VirtualFP8Type>(result.getType()))
          registers.push_back({result, std::exchange(tensor, tensor + 2)});
      }
      if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
        if (transfers.empty()) {
          addr = op->getOperand(op->getNumOperands() - 2);
          size = op->getOperand(op->getNumOperands() - 1);
        }
        dma = {op->getResult(0), {6, bf16 ? 2u : 1u, 37, 0x2100, 4, 7, 9}};
        transfers.push_back(dma);
      }
      if (auto constant = dyn_cast<arith::ConstantOp>(op);
          constant && !keep && constant.getValue() == IntegerAttr::get(IntegerType::get(&context, 32), 7))
        keep = constant.getResult();
    });
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
    StringRef phase = role == &P::dramReg   ? "DRAM capture"
                      : role == &P::sizeReg ? "size capture"
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

// A load's staging VLOADs run before the join's terminator, after `%keep` is defined and while it is live into the
// exit block; a store, which must be in the return block, issues no VLOAD.
void stagingLoadTests(MLIRContext &context, bool bf16, bool store) {
  StringRef cfg = store ? "" : "join";
  Fixture live(context, source(bf16, store, true, true, cfg), bf16), dead(context, source(bf16, store, true, false, cfg), bf16);
  if (!live || !dead)
    return;
  std::string keep = "x" + std::to_string(live.reg(live.keep));
  for (auto policy : {Policy::Preserved, Policy::Rematerialized}) {
    auto changed = live.dma.placement;
    changed.stagingReg = live.reg(live.keep);
    SmallVector<StringRef> fragments;
    if (!store)
      fragments = {clobber, "staging VLOAD materialization", keep};
    expect(live, "staging VLOAD write meets a live scalar", changed, store, fragments, policy);
    changed = dead.dma.placement;
    changed.stagingReg = dead.reg(dead.keep);
    expect(dead, "staging VLOAD write may reuse a dead scalar", changed, true, {}, policy);
  }
}

// A DMA-only address or size is an immediate: each launch builds it in its command register, so it claims no register
// and its capture always writes, even into a register a live value holds.
void immediateTests(MLIRContext &context, bool bf16, bool store) {
  Fixture f(context, source(bf16, store, false, true, "", false, true), bf16);
  if (!f)
    return;
  check(isVirtualDMAImmediate(f.addr) && isVirtualDMAImmediate(f.size), "DMA-only constants are immediates");
  auto base = f.dma.placement;
  expect(f, "immediate placement baseline", base, true);
  f.registers.push_back({f.addr, 13});
  expect(f, "an immediate cannot claim a register", base, false, {"claims a DMA immediate"});
  f.registers.pop_back();
  for (unsigned P::*role : {&P::dramReg, &P::sizeReg}) {
    auto changed = base;
    changed.*role = f.reg(f.keep);
    expect(f, "immediate capture clobbers live-after scalar", changed, false,
           {clobber, role == &P::dramReg ? "DRAM capture" : "size capture", "x12"});
  }
  // The later boundary read sees the size the launch built.
  auto changed = base;
  changed.sizeReg = 2;
  expect(f, "immediate size capture sets persistent half-size content", changed, !bf16,
         bf16 ? ArrayRef<StringRef>(halfSize) : ArrayRef<StringRef>());
}

// Two loads share helper registers: each launch and each staging VLOAD sets the registers it reads.
void sharedHelperTests(MLIRContext &context) {
  Fixture two(context, R"mlir(module { func.func @two() -> !atlas.virtual_state {
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %addr = arith.constant -2147483648 : i32
  %size = arith.constant 2048 : i32
  %small = arith.constant 1024 : i32
  %s1, %x = "atlas.virtual_dma_load_bf16"(%s0, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %s2, %y = "atlas.virtual_dma_load_fp8"(%s1, %addr, %small) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %s5 = "atlas.virtual_output_bf16"(%s2, %x) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %s5 : !atlas.virtual_state
} })mlir", true, false);
  if (!two)
    return;
  SmallVector<VirtualDMAAssignment> base = {{two.transfers[0].transfer, {6, 2, 37, 0x2100, 4, 7, 9}},
                                            {two.transfers[1].transfer, {7, 1, 38, 0x2400, 6, 8, 3}}};
  expectAssignments(two, "loads with separate helpers", base, true);
  for (unsigned P::*role : {&P::dramReg, &P::sizeReg, &P::stagingReg}) {
    auto changed = base;
    changed[1].placement.*role = 4;
    expectAssignments(two, "the second load reuses the first load's staging helper", changed, true);
  }
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
  text = replace(replace(text, "    %done, %tile =", output + "    %done, %tile ="), "load_bf16\"(%ready", "load_bf16\"(%out");
  if (Fixture f(context, replace(text, "return %out", "return %done"), true); f) {
    auto changed = f.dma.placement;
    changed.sizeReg = 2;
    expect(f, "half-size helper may be reused after last boundary read", changed, true);
  }

  text = replace(load, "    %out =", "    %small = arith.constant 1024 : i32\n"
      "    %restored, %unused = \"atlas.virtual_dma_load_fp8\"(%done, %addr, %small) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n"
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
  %issued, %tile = "atlas.virtual_dma_load_bf16"(%s0, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  cf.br ^loop(%issued, %a, %b, %first : !atlas.virtual_state, i32, i32, i1)
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
  expect(awaitThrough, "BF16 staging VLOAD clobbers successor live-through scalar", changed, false, {clobber, "staging VLOAD materialization"});
  changed = loop.dma.placement;
  changed.stagingReg = loop.reg(loop.addr);
  // The loaded tile's staging VLOADs, before the loop's branch, meet the address first in reverse liveness.
  expect(loop, "backedge keeps address operand live after launch", changed, false, {clobber, "staging VLOAD materialization", "x10"});
}
} // namespace

void atlas_test::runDMAHelper(MLIRContext &context) {
  for (bool bf16 : {false, true})
    for (bool store : {false, true}) {
      launchTests(context, bf16, store);
      stagingLoadTests(context, bf16, store);
      immediateTests(context, bf16, store);
    }
  cfgTests(context);
  sharedHelperTests(context);
  halfSizeTests(context);
  halfSizeCFGTests(context);
  packHelperTests(context);
  scalarCopyHelperTests(context);
}
