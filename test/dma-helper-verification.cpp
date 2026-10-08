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
  VirtualDMAAssignment dma;
  Value addr, size, keep;

  bool initialize(MLIRContext &context, StringRef text, bool bf16) {
    module = parseSourceString<ModuleOp>(text, &context);
    check(bool(module), "fixture parses");
    if (!module)
      return false;
    bool valid = succeeded(verify(*module)) && succeeded(verifyAtlasVirtualModule(*module));
    check(valid, "fixture is an admitted virtual CFG");
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
      if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op))
        dma = {op->getResult(1), {6, bf16 ? 2u : 1u, 37, 0x2100, 4, 7, 9}};
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

void expect(Fixture &fixture, StringRef name, DMATransferPlacement placement,
            bool valid, ArrayRef<StringRef> fragments = {},
            DMAAwaitBasePolicy policy = DMAAwaitBasePolicy::Preserved,
            bool supplyDMA = true) {
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
  VirtualDMAAssignment dma{fixture.dma.transfer, placement};
  ArrayRef<VirtualDMAAssignment> assignments = supplyDMA ? ArrayRef<VirtualDMAAssignment>(dma) : ArrayRef<VirtualDMAAssignment>();
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
  bool accepted = succeeded(verifyAtlasRegisterAllocation(fixture.function, fixture.registers, fixed, {}, assignments, policy));
  stream.flush();
  check(accepted == valid, name, diagnostics);
  if (valid)
    check(diagnostics.empty(), name, diagnostics);
  for (StringRef fragment : fragments)
    check(StringRef(diagnostics).contains(fragment), name, diagnostics);
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
  // Preserved FP8 acceptance proves only that no helper WRITE clobbers the
  // scalar; preserving the hidden pending base until VLOAD is a separate proof.
  Fixture live, dead;
  if (!live.initialize(context, source(bf16, store, true), bf16) ||
      !dead.initialize(context, source(bf16, store, true, false), bf16))
    return;
  auto changed = live.dma.placement;
  changed.stagingReg = reg(live, live.keep);
  for (auto policy : {DMAAwaitBasePolicy::Preserved, DMAAwaitBasePolicy::Rematerialized}) {
    bool valid = store || (!bf16 && policy == DMAAwaitBasePolicy::Preserved);
    expect(live, "await staging policy and store wait", changed, valid,
           valid ? ArrayRef<StringRef>() : ArrayRef<StringRef>({clobber, "await staging materialization", "x12"}), policy);
    changed.stagingReg = reg(dead, dead.keep);
    expect(dead, "await staging may overwrite a dead scalar", changed, true, {}, policy);
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
  llvm::outs() << checks << " DMA helper checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
