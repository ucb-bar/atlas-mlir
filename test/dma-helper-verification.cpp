#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "Atlas/AtlasTypes.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
unsigned checks = 0, failures = 0;
constexpr StringLiteral state = "!atlas.virtual_state";
constexpr StringLiteral clobber = "DMA helper write clobbers live scalar value";
constexpr StringLiteral pendingBase = "pending DMA staging base is not preserved";
constexpr StringLiteral halfSize = "persistent DMA half-size helper is not preserved";

void check(bool condition, StringRef name, StringRef detail = {}) {
  ++checks;
  if (!condition) {
    ++failures;
    llvm::errs() << "FAIL: " << name << '\n' << detail << '\n';
  }
}

std::string source(bool bf16, bool store, bool keepAfterLaunch = false,
                   bool keepLive = true, StringRef cfg = "", bool useOperands = false) {
  std::string tile = bf16 ? "!atlas.virtual_bf16" : "!atlas.virtual_fp8";
  std::string event = store ? "!atlas.virtual_dma_store" : (bf16 ? "!atlas.virtual_dma_load_bf16" : "!atlas.virtual_dma_load_fp8");
  std::string fmt = bf16 ? "bf16" : "fp8";
  std::string body = "module { func.func @test() -> " + state.str() + " {\n";
  body += "    %s0 = \"atlas.virtual_start\"() : () -> " + state.str() + "\n";
  body += "    %addr = arith.constant -2147483648 : i32\n    %size = arith.constant " + std::to_string(bf16 ? 2048 : 1024) + " : i32\n";
  if (!keepAfterLaunch)
    body += "    %keep = arith.constant 7 : i32\n";
  body += "    %ready, %observable = \"atlas.virtual_input_bf16\"(%s0) {index = 1 : i32} : (" + state.str() + ") -> (" + state.str() + ", !atlas.virtual_bf16)\n";
  std::string before = "%ready";
  if (cfg == "loop") {
    body += "    cf.br ^loop(%ready : " + state.str() + ")\n  ^loop(%state: " + state.str() + "):\n";
    before = "%state";
  }
  if (store) {
    body += "    %input, %tile = \"atlas.virtual_input_" + fmt + "\"(" + before + ") {index = 0 : i32} : (" + state.str() + ") -> (" + state.str() + ", " + tile + ")\n";
    before = "%input";
  }
  body += "    %issued, %event = \"atlas.virtual_dma_" + std::string(store ? "store_" : "load_") + fmt + "\"(" + before;
  if (store)
    body += ", %tile";
  body += ", %addr, %size) : (" + state.str() + (store ? ", " + tile : "") + ", i32, i32) -> (" + state.str() + ", " + event + ")\n";
  if (keepAfterLaunch)
    body += "    %keep = arith.constant 7 : i32\n";
  body += store ? "    %done = \"atlas.virtual_dma_wait\"(%issued, %event)" : "    %done, %tile = \"atlas.virtual_dma_await_" + fmt + "\"(%issued, %event)";
  body += " : (" + state.str() + ", " + event + ") -> " + (store ? state.str() : "(" + state.str() + ", " + tile + ")") + "\n";
  std::string final = "%done";
  if (cfg == "join") {
    body += "    cf.br ^exit(%done : " + state.str() + ")\n  ^exit(%edge: " + state.str() + "):\n";
    final = "%edge";
  }
  if (keepLive)
    body += useOperands ? "    %after = arith.addi %addr, %size : i32\n" : "    %after = arith.addi %keep, %keep : i32\n";
  if (cfg == "loop") {
    body += "    %again = arith.constant false\n    cf.cond_br %again, ^loop(%done : " + state.str() + "), ^exit(%done : " + state.str() + ")\n  ^exit(%edge: " + state.str() + "):\n";
    final = "%edge";
  }
  body += "    %out = \"atlas.virtual_output_bf16\"(" + final + ", %observable) {index = 0 : i32} : (" + state.str() + ", !atlas.virtual_bf16) -> " + state.str() + "\n";
  return body + "    return %out : " + state.str() + "\n} }";
}

// Every claimed register and DMA placement is chosen here independently of the
// allocator. Unique baseline registers isolate helper writes from SSA overlap.
struct Fixture {
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  SmallVector<VirtualRegisterAssignment> registers;
  SmallVector<VirtualDMAAssignment> transfers;
  VirtualDMAAssignment dma;
  Value addr, size, keep;

  bool initialize(MLIRContext &context, StringRef text, bool bf16, bool admit = true) {
    module = parseSourceString<ModuleOp>(text, &context);
    check(bool(module), "fixture parses");
    if (!module)
      return false;
    bool valid = succeeded(verify(*module)) && (!admit || succeeded(verifyAtlasVirtualModule(*module)));
    check(valid, admit ? "fixture is an admitted virtual CFG" : "fixture is verified typed SSA");
    if (!valid)
      return false;
    function = *module->getOps<func::FuncOp>().begin();
    unsigned scalar = 10, tensor = 0;
    function.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        Type type = result.getType();
        if (type.isInteger(1) || type.isInteger(32))
          registers.push_back({result, scalar++});
        else if (isa<VirtualBF16Type, VirtualFP8Type>(type)) {
          registers.push_back({result, tensor});
          tensor += 2;
        }
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
    return true;
  }
};

unsigned reg(Fixture &fixture, Value value) {
  for (auto assignment : fixture.registers)
    if (assignment.value == value)
      return assignment.reg;
  llvm_unreachable("fixture value has no claimed register");
}

FixedResourcePlacement fixedPlacement() {
  FixedResourcePlacement fixed{};
  fixed.tensorTemporary = 62;
  fixed.scalarTemporary = 27;
  fixed.oneReg = 28;
  fixed.zeroReg = 5;
  fixed.halfSizeReg = 2;
  fixed.haltReg = 1;
  fixed.inputBaseReg = 6;
  fixed.inputDramReg = 1;
  fixed.outputBaseReg = 8;
  fixed.outputDramReg = 3;
  fixed.dmaBaseReg = 4;
  fixed.dmaDramReg = 7;
  fixed.dmaSizeReg = 9;
  return fixed;
}

void expectAssignments(Fixture &fixture, StringRef name,
                       ArrayRef<VirtualDMAAssignment> assignments, bool valid,
                       ArrayRef<StringRef> fragments = {},
                       DMAAwaitBasePolicy policy = DMAAwaitBasePolicy::Preserved,
                       const FixedResourcePlacement *suppliedFixed = nullptr) {
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  ScopedDiagnosticHandler handler(fixture.function.getContext(), [&](Diagnostic &diagnostic) {
    diagnostic.print(stream);
    stream << '\n';
    for (const Diagnostic &note : diagnostic.getNotes()) {
      note.print(stream);
      stream << '\n';
    }
    return success();
  });
  bool accepted = succeeded(verifyAtlasRegisterAllocation(fixture.function, fixture.registers,
      suppliedFixed ? *suppliedFixed : fixedPlacement(), {}, assignments, policy));
  stream.flush();
  check(accepted == valid, name, diagnostics);
  if (valid)
    check(diagnostics.empty(), name, diagnostics);
  for (StringRef fragment : fragments)
    check(StringRef(diagnostics).contains(fragment), name, diagnostics);
}

void expect(Fixture &fixture, StringRef name, DMATransferPlacement placement,
            bool valid, ArrayRef<StringRef> fragments = {},
            DMAAwaitBasePolicy policy = DMAAwaitBasePolicy::Preserved,
            bool supplyDMA = true) {
  VirtualDMAAssignment dma{fixture.dma.transfer, placement};
  ArrayRef<VirtualDMAAssignment> assignments = supplyDMA ? ArrayRef<VirtualDMAAssignment>(dma) : ArrayRef<VirtualDMAAssignment>();
  expectAssignments(fixture, name, assignments, valid, fragments, policy);
}

void launchTests(MLIRContext &context, bool bf16, bool store) {
  Fixture fixture, self;
  if (!fixture.initialize(context, source(bf16, store), bf16) ||
      !self.initialize(context, source(bf16, store, false, true, "", true), bf16))
    return;
  auto base = fixture.dma.placement;
  expect(fixture, "independent placement baseline", base, true);
  expect(fixture, "omitted DMA assignments cannot bypass helper checks", base, false, {"missing DMA assignment"}, DMAAwaitBasePolicy::Preserved, false);
  for (unsigned role = 0; role < 3; ++role) {
    auto changed = base;
    if (role == 0) changed.dramReg = reg(fixture, fixture.keep);
    if (role == 1) changed.sizeReg = reg(fixture, fixture.keep);
    if (role == 2) changed.stagingReg = reg(fixture, fixture.keep);
    StringRef phase = role == 0 ? "DRAM capture" : role == 1 ? "size capture" : "launch staging materialization";
    if (role == 2 && bf16 && !store)
      phase = "await staging materialization"; // Reverse liveness checks encounter the later write first.
    expect(fixture, "helper clobbers live-after scalar", changed, false, {clobber, phase, "x12", "live scalar"});
  }
  auto changed = base;
  changed.dramReg = reg(fixture, fixture.size);
  expect(fixture, "first copy destroys unread size operand", changed, false, {"DMA DRAM helper write clobbers size operand before capture", "x11"});
  changed = self.dma.placement;
  changed.dramReg = reg(self, self.addr);
  changed.sizeReg = reg(self, self.size);
  expect(self, "exact self copies preserve live source operands", changed, true);
  changed = base;
  changed.sizeReg = reg(fixture, fixture.addr);
  expect(fixture, "size copy reuses address after its capture", changed, true);
  for (Value dead : {fixture.addr, fixture.size}) {
    changed = base;
    changed.stagingReg = reg(fixture, dead);
    expect(fixture, "staging write reuses fully captured dead operand", changed, true);
  }
}

void awaitTests(MLIRContext &context, bool bf16, bool store) {
  Fixture live, dead;
  if (!live.initialize(context, source(bf16, store, true), bf16) ||
      !dead.initialize(context, source(bf16, store, true, false), bf16))
    return;
  auto changed = live.dma.placement;
  changed.stagingReg = reg(live, live.keep);
  for (auto policy : {DMAAwaitBasePolicy::Preserved, DMAAwaitBasePolicy::Rematerialized}) {
    bool valid = store;
    bool preserved = policy == DMAAwaitBasePolicy::Preserved;
    expect(live, "await staging policy and store wait", changed, valid,
           valid ? ArrayRef<StringRef>() : !bf16 && preserved ? ArrayRef<StringRef>({pendingBase}) : ArrayRef<StringRef>({clobber, "await staging materialization", "x12"}), policy);
    changed.stagingReg = reg(dead, dead.keep);
    expect(dead, "dead scalar reuse still preserves an unread staging base", changed,
           store || !preserved, store || !preserved ? ArrayRef<StringRef>() : ArrayRef<StringRef>({pendingBase}), policy);
  }
}

constexpr StringLiteral twoLoads = R"mlir(
module { func.func @two() -> !atlas.virtual_state {
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
} })mlir";

void pendingTests(MLIRContext &context) {
  for (bool bf16 : {false, true}) {
    Fixture fixture, restored;
    if (!fixture.initialize(context, source(bf16, false, true, false), bf16))
      continue;
    auto changed = fixture.dma.placement;
    changed.stagingReg = reg(fixture, fixture.keep);
    expect(fixture, "scalar result destroys pending first-half base", changed, false, {pendingBase});
    expect(fixture, "rematerialized await permits scalar reuse", changed, true, {}, DMAAwaitBasePolicy::Rematerialized);
    std::string text = source(bf16, false, true, false);
    text.replace(text.find("%keep = arith.constant 7"), std::string("%keep = arith.constant 7").size(), "%keep = arith.constant 8448");
    if (restored.initialize(context, text, bf16)) {
      changed = restored.dma.placement;
      changed.stagingReg = reg(restored, restored.keep);
      expect(restored, "known scalar restoration preserves pending first-half base", changed, true);
    }
  }

  Fixture two;
  // Typed SSA bypasses only the baseline admission limit on pending handles.
  if (!two.initialize(context, twoLoads, true, false))
    return;
  SmallVector<VirtualDMAAssignment> base = {
      {two.transfers[0].transfer, {6, 2, 37, 0x2100, 4, 7, 9}},
      {two.transfers[1].transfer, {7, 1, 38, 0x2400, 6, 8, 3}}};
  expectAssignments(two, "BF16 first-half read precedes own second-half base write", base, true);
  for (unsigned role = 0; role < 3; ++role) {
    auto changed = base;
    if (role == 0) changed[1].placement.dramReg = 4;
    if (role == 1) changed[1].placement.sizeReg = 4;
    if (role == 2) changed[1].placement.stagingReg = 4;
    expectAssignments(two, "other DMA helper destroys retained staging base", changed, false, {pendingBase});
    expectAssignments(two, "rematerialized awaits allow cross-DMA helper reuse", changed, true, {}, DMAAwaitBasePolicy::Rematerialized);
  }
  auto restored = base;
  restored[0].placement.stagingWord = 1024;
  restored[1].placement.sizeReg = 4;
  expectAssignments(two, "same-value DMA size capture preserves another pending base", restored, true);
}

void halfSizeTests(MLIRContext &context) {
  for (bool bf16 : {false, true}) {
    Fixture fixture;
    if (!fixture.initialize(context, source(bf16, false, false, false), bf16))
      continue;
    auto changed = fixture.dma.placement;
    changed.sizeReg = 2;
    expect(fixture, "persistent half-size content checked at later boundary read", changed,
           !bf16, bf16 ? ArrayRef<StringRef>({halfSize}) : ArrayRef<StringRef>());
  }

  Fixture lastRead, restored;
  std::string text = source(true, false, false, false);
  std::string output = "    %out = \"atlas.virtual_output_bf16\"(%ready, %observable) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state\n";
  auto outputPosition = text.find("    %out =");
  text.erase(outputPosition, text.find('\n', outputPosition) - outputPosition + 1);
  auto launchPosition = text.find("    %issued,");
  text.insert(launchPosition, output);
  auto before = text.find("load_bf16\"(%ready");
  text.replace(before, std::string("load_bf16\"(%ready").size(), "load_bf16\"(%out");
  text.replace(text.find("return %out"), std::string("return %out").size(), "return %done");
  if (lastRead.initialize(context, text, true)) {
    auto changed = lastRead.dma.placement;
    changed.sizeReg = 2;
    expect(lastRead, "half-size helper may be reused after last boundary read", changed, true);
  }

  text = source(true, false, false, false);
  text.insert(text.find("    %out ="),
      "    %small = arith.constant 1024 : i32\n"
      "    %restore, %second = \"atlas.virtual_dma_load_fp8\"(%done, %addr, %small) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)\n"
      "    %restored, %unused = \"atlas.virtual_dma_await_fp8\"(%restore, %second) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n");
  auto stateOperand = text.find("output_bf16\"(%done");
  text.replace(stateOperand, std::string("output_bf16\"(%done").size(), "output_bf16\"(%restored");
  if (restored.initialize(context, text, true)) {
    SmallVector<VirtualDMAAssignment> assignments = {
        {restored.transfers[0].transfer, {6, 2, 37, 0x2100, 4, 7, 2}},
        {restored.transfers[1].transfer, {6, 1, 38, 0x2300, 4, 7, 2}}};
    expectAssignments(restored, "FP8 capture restores half-size before later boundary", assignments, true);
  }
}

std::string halfSizeDiamond(bool bf16) {
  std::string fmt = bf16 ? "bf16" : "fp8";
  std::string tile = "!atlas.virtual_" + fmt;
  std::string event = "!atlas.virtual_dma_load_" + fmt;
  return "module { func.func @diamond() -> !atlas.virtual_state {\n"
      "  %s0 = \"atlas.virtual_start\"() : () -> !atlas.virtual_state\n"
      "  %addr = arith.constant -2147483648 : i32\n"
      "  %size = arith.constant " + std::to_string(bf16 ? 2048 : 1024) + " : i32\n"
      "  %choose = arith.constant true\n"
      "  %ready, %observable = \"atlas.virtual_input_bf16\"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n"
      "  cf.cond_br %choose, ^left(%ready : !atlas.virtual_state), ^right(%ready : !atlas.virtual_state)\n"
      "^left(%left_io: !atlas.virtual_state):\n"
      "  %issued, %event = \"atlas.virtual_dma_load_" + fmt + "\"(%left_io, %addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, " + event + ")\n"
      "  %done, %tile = \"atlas.virtual_dma_await_" + fmt + "\"(%issued, %event) : (!atlas.virtual_state, " + event + ") -> (!atlas.virtual_state, " + tile + ")\n"
      "  cf.br ^join(%done : !atlas.virtual_state)\n"
      "^right(%right_io: !atlas.virtual_state):\n"
      "  cf.br ^join(%right_io : !atlas.virtual_state)\n"
      "^join(%joined: !atlas.virtual_state):\n"
      "  %out = \"atlas.virtual_output_bf16\"(%joined, %observable) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state\n"
      "  return %out : !atlas.virtual_state\n} }";
}

void halfSizeCFGTests(MLIRContext &context) {
  for (bool bf16 : {false, true}) {
    for (bool loop : {false, true}) {
      Fixture fixture;
      std::string text = loop ? source(bf16, false, false, false, "loop") : halfSizeDiamond(bf16);
      if (!fixture.initialize(context, text, bf16))
        continue;
      auto changed = fixture.dma.placement;
      expect(fixture, loop ? "loop half-size baseline" : "diamond half-size baseline", changed, true);
      changed.sizeReg = 2;
      expect(fixture, loop ? "backedge helper content reaches exit boundary" : "join merges helper contents from both predecessor paths",
             changed, !bf16, bf16 ? ArrayRef<StringRef>({halfSize}) : ArrayRef<StringRef>());
    }
  }
}

void packHelperTests(MLIRContext &context) {
  Fixture fixture;
  std::string text = source(true, false, false, false);
  text.insert(text.find("    %out ="),
      "    %packed = \"atlas.virtual_pack_fp8\"(%tile) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n");
  if (!fixture.initialize(context, text, true))
    return;
  for (auto &assignment : fixture.registers)
    if (assignment.value.getType().isInteger(1) || assignment.value.getType().isInteger(32))
      assignment.reg += 8;
  auto fixed = fixedPlacement();
  fixed.scaleReg = 3;
  fixed.outputWord = 65536;
  fixed.packWord = 32768;
  fixed.packRelayoutWord = 33024;
  fixed.packSourceRegs = {10, 11};
  fixed.packDestinationReg = 12;
  fixed.packRowReg = 13;
  fixed.packRowsReg = 14;
  fixed.packTemporaryRegs = {16, 17};
  expectAssignments(fixture, "pack preserves unrelated persistent half-size helper", fixture.transfers,
      true, {}, DMAAwaitBasePolicy::Preserved, &fixed);
  fixed.packRowsReg = fixed.halfSizeReg;
  expectAssignments(fixture, "pack row-count write destroys later boundary half-size", fixture.transfers,
      false, {halfSize}, DMAAwaitBasePolicy::Preserved, &fixed);
  fixed.packRowsReg = 14;
  fixed.packTemporaryRegs[0] = fixed.halfSizeReg;
  expectAssignments(fixture, "pack temporary makes later boundary half-size unknown", fixture.transfers,
      false, {halfSize}, DMAAwaitBasePolicy::Preserved, &fixed);
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
    Fixture fixture;
    std::string text = scalarSwap.str();
    if (!cycle) {
      StringRef swapped = "%loop_io, %right, %left, %stop";
      text.replace(text.find(swapped.str()), swapped.size(), "%loop_io, %left, %right, %stop");
    }
    if (!fixture.initialize(context, text, true))
      continue;
    for (Block &block : fixture.function.getBody())
      if (block.getNumArguments() == 4)
        for (unsigned i = 1; i < 4; ++i)
          fixture.registers.push_back({block.getArgument(i), 11 + i});
    auto fixed = fixedPlacement();
    expectAssignments(fixture, "scalar edge-copy baseline preserves half-size", fixture.transfers,
        true, {}, DMAAwaitBasePolicy::Preserved, &fixed);
    fixed.halfSizeReg = fixed.scalarTemporary;
    expectAssignments(fixture, cycle ? "swap cycle clobbers aliased half-size helper" : "noncyclic edge avoids scalar temporary write",
        fixture.transfers, !cycle, cycle ? ArrayRef<StringRef>({halfSize}) : ArrayRef<StringRef>(),
        DMAAwaitBasePolicy::Preserved, &fixed);
  }
}

void cfgTests(MLIRContext &context) {
  Fixture through, awaitThrough, loop;
  if (!through.initialize(context, source(false, false, false, true, "join"), false) ||
      !awaitThrough.initialize(context, source(true, false, true, true, "join"), true) ||
      !loop.initialize(context, source(false, false, false, false, "loop"), false))
    return;
  auto changed = through.dma.placement;
  expect(through, "CFG live-through baseline", changed, true);
  changed.dramReg = reg(through, through.keep);
  expect(through, "launch helper clobbers successor live-through scalar", changed, false, {clobber, "DRAM capture"});
  changed = awaitThrough.dma.placement;
  changed.stagingReg = reg(awaitThrough, awaitThrough.keep);
  expect(awaitThrough, "BF16 await clobbers successor live-through scalar", changed, false, {clobber, "await staging materialization"});
  changed = loop.dma.placement;
  expect(loop, "loop baseline", changed, true);
  changed.stagingReg = reg(loop, loop.addr);
  expect(loop, "backedge keeps address operand live after launch", changed, false, {clobber, "launch staging materialization", "x10"});
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect>();
  MLIRContext context(registry);
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
  llvm::outs() << checks << " DMA helper checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
