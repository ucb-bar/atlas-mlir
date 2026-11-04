#include "Atlas/AtlasStream.h"
#include "VerificationTestSupport.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
struct Program : IssuedStream {
  using IssuedStream::IssuedStream;
  Operation *addi(int dst, int src, int immediate) {
    return emit("atlas.alu_imm", {text("kind", "addi"), i("dst", dst), i("src", src), i("immediate", immediate)});
  }
  void toggle(int reg) {
    emit("atlas.alu_imm", {text("kind", "xori"), i("dst", reg), i("src", reg), i("immediate", 256)});
  }
  void prefix() { constant(4, 256); constant(5, 0); constant(7, 0x90000000); constant(9, 1024); }
  void launch(int channel = 0) {
    emit("atlas.dma", {text("direction", "load"), i("channel", channel), i("reg", 4), i("dram", 7), i("size", 9)});
  }
  void wait(int channel = 0) { emit("atlas.dma_wait", {i("channel", channel)}); }
  void load(int base = 5) {
    emit("atlas.vload", {i("dst", 0), i("base", base), i("offset", 0), text("format", "raw")});
    emit("atlas.delay", {i("cycles", 66)});
    nop();
  }
  Operation *branch(int reg) {
    return emit("atlas.branch", {text("kind", "bne"), i("lhs", reg), i("rhs", 0), i("offset_bytes", 0)});
  }
  void expect(StringRef name, bool valid, ArrayRef<StringRef> fragments = {}) {
    expectVerified(name, valid, fragments, [&] {
      auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
      return failed(stream) ? failure() : verifyAtlasTimedStream(*stream, timing::npuModelTimingProvider());
    });
  }
  void expectDMA(StringRef name, bool valid, size_t cap) {
    expectVerified(name, valid, {}, [&] {
      auto stream = readAtlasStream(*module, AtlasStreamReadMode::Verification);
      return failed(stream) ? failure() : checkAtlasDMACFG(*stream, timing::npuModelTimingProvider(), cap);
    });
  }
};

void crossBlock(MLIRContext &context) {
  Program p(context);
  p.prefix(); p.launch();
  Operation *jump = p.jump(); p.nop();
  size_t target = p.ops.size();
  p.wait(); p.halt(); p.aim(jump, target);
  p.expect("pending DMA crosses a jump and completes in its successor", true);
  Program fallthrough(context); fallthrough.prefix(); fallthrough.launch();
  Operation *never = fallthrough.branch(0); fallthrough.nop(); fallthrough.wait(); fallthrough.halt();
  fallthrough.aim(never, fallthrough.ops.size());
  fallthrough.expect("known false branch does not create a pending-DMA stream exit", true);
}

void alternating(MLIRContext &context, bool overlap) {
  Program p(context);
  p.prefix(); p.constant(10, 4);
  size_t loop = p.ops.size();
  p.launch(); p.load(overlap ? 4 : 5); p.wait();
  p.toggle(4); p.toggle(5); p.addi(10, 10, -1);
  Operation *back = p.branch(10); p.nop(); p.halt(); p.aim(back, loop);
  p.expect(overlap ? "alternating loop rejects a real overlap" : "four alternating-buffer iterations retain correlated bases",
           !overlap, overlap ? ArrayRef<StringRef>{"conflict"} : ArrayRef<StringRef>{});
}

void captured(MLIRContext &context, bool overlap) {
  Program p(context);
  p.prefix(); p.constant(5, 256); p.launch(); p.constant(4, 0);
  // Rewriting x4 after launch must not rewrite the pending transfer's captured window.
  p.load(overlap ? 5 : 4); p.wait(); p.halt();
  p.expect(overlap ? "pending footprint survives a base overwrite" : "base overwrite permits a disjoint vector access",
           !overlap, overlap ? ArrayRef<StringRef>{"conflict"} : ArrayRef<StringRef>{});
}

void delayedBranch(MLIRContext &context, bool known) {
  Program p(context);
  p.prefix();
  if (known) p.constant(10, 1);
  p.launch();
  Operation *branch = p.branch(known ? 10 : 27);
  p.addi(10, 0, 0); // The decision uses the value before this delay slot executes.
  p.load(4); // Reachable only for an unknown or false branch, and overlaps the DMA.
  size_t target = p.ops.size();
  p.wait(); p.halt(); p.aim(branch, target);
  p.expect(known ? "branch decision precedes its predicate-changing delay slot" : "unknown condition retains the unsafe successor",
           known, known ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"conflict"});
}

void conditions(MLIRContext &context) {
  struct Case { StringRef kind; uint32_t lhs, rhs; };
  const Case cases[] = {{"beq", 0, 0}, {"bne", uint32_t(-1), 0}, {"blt", uint32_t(-1), 0},
                        {"bge", 0, uint32_t(-1)}, {"bltu", 0, uint32_t(-1)}, {"bgeu", uint32_t(-1), 0}};
  for (const Case &test : cases) {
    Program p(context); p.prefix(); p.constant(10, test.lhs); p.constant(11, test.rhs); p.launch();
    Operation *branch = p.emit("atlas.branch", {p.text("kind", test.kind), p.i("lhs", 10), p.i("rhs", 11), p.i("offset_bytes", 0)});
    p.nop(); p.load(4);
    size_t target = p.ops.size(); p.wait(); p.halt(); p.aim(branch, target);
    p.expect("known " + test.kind.str() + " prunes its unsafe fallthrough", true);
  }
}

void widening(MLIRContext &context) {
  Program p(context); p.prefix(); p.constant(10, 0);
  size_t loop = p.ops.size();
  p.addi(10, 10, 1); // Infinitely many scalar values, irrelevant to DMA addresses.
  p.launch(); p.load(); p.wait();
  Operation *back = p.jump(); p.nop(); p.aim(back, loop); p.halt();
  p.expectDMA("growing irrelevant scalar values converge without losing disjoint constant buffers", true, 2);
  Program unsafe(context); unsafe.prefix(); unsafe.constant(4, 0); unsafe.constant(5, 256);
  size_t body = unsafe.ops.size();
  unsafe.launch(); unsafe.load(); unsafe.wait(); unsafe.addi(5, 5, 256);
  Operation *repeat = unsafe.jump(); unsafe.nop(); unsafe.aim(repeat, body); unsafe.halt();
  unsafe.expectDMA("widened vector base must not prove disjointness from a captured transfer", false, 2);
  {
    Program custom(context); custom.constant(10, 0);
    size_t body = custom.ops.size(); custom.addi(10, 10, 1);
    Operation *back = custom.jump(); custom.nop(); custom.aim(back, body); custom.halt();
    auto stream = readAtlasStream(*custom.module, AtlasStreamReadMode::Verification);
    auto provider = timing::npuModelTimingProvider();
    provider.id = "synthetic-no-widening";
    expectVerified("custom provider fails closed at the last disjunct without queued work", false, {}, [&] {
      return failed(stream) ? failure() : checkAtlasDMACFG(*stream, provider, 1);
    });
  }
}

void memoryHazards(MLIRContext &context) {
  for (StringRef direction : {StringRef("load"), StringRef("store")}) {
    for (StringRef access : {StringRef("vstore"), StringRef("scalar_store")}) {
      Program p(context); p.prefix();
      p.emit("atlas.dma", {p.text("direction", direction), p.i("channel", 0), p.i("reg", 4), p.i("dram", 7), p.i("size", 9)});
      if (access == "vstore")
        p.emit("atlas.vstore", {p.i("src", 0), p.i("base", 4), p.i("offset", 0), p.text("format", "raw")});
      else {
        p.constant(6, 1024); // Scalar addresses count bytes, DMA bases count words.
        p.emit("atlas.scalar_store", {p.text("kind", "sw"), p.i("src", 0), p.i("base", 6), p.i("offset", 0)});
      }
      p.wait(); p.halt();
      p.expect("pending DMA " + direction.str() + " rejects overlapping " + access.str(), false, {"conflict"});
    }
  }
  {
    Program p(context); p.prefix(); p.launch(); p.constant(6, 1024);
    p.emit("atlas.scalar_load", {p.text("kind", "lw"), p.i("dst", 10), p.i("base", 6), p.i("offset", 0)});
    p.wait(); p.halt(); p.expect("scalar load cannot read a pending DMA destination", false, {"RAW conflict"});
  }
  {
    Program p(context); p.prefix(); p.launch(); p.launch(1); p.wait(); p.wait(1); p.halt();
    p.expect("cross-channel DMA overlap remains invalid", false, {"conflict"});
  }
  {
    Program p(context); p.prefix(); p.constant(10, 2);
    size_t loop = p.ops.size(); p.launch(); p.addi(4, 4, 256); p.addi(10, 10, -1);
    Operation *back = p.branch(10); p.nop(); p.aim(back, loop);
    p.constant(5, 256); p.load(); p.wait(); p.halt();
    p.expect("repeated launch cannot overwrite an older pending window", false, {"conflict"});
  }
}

void failureCases(MLIRContext &context) {
  {
    Program p(context); p.prefix(); p.launch(); p.load(27); p.wait(); p.halt();
    p.expect("unknown vector base cannot prove DMA disjointness", false);
  }
  {
    Program p(context); p.prefix(); p.launch(); p.halt();
    p.expect("halt with pending DMA fails", false);
  }
  {
    Program p(context); p.prefix(); p.launch();
    Operation *jump = p.jump(); p.nop(); p.aim(jump, p.ops.size());
    p.expect("jump to stream end with pending DMA fails", false);
  }
  {
    Program p(context); p.prefix(); p.launch(); p.wait(1); p.load(4); p.wait(); p.halt();
    p.expect("wait on another channel leaves DMA ownership live", false, {"conflict"});
  }
  {
    Program p(context); p.emit("atlas.delay", {p.i("cycles", 128)}); p.halt();
    p.expect("DELAY directly before halt remains invalid", false, {"halt does not wait for a preceding DELAY"});
  }
  {
    Program p(context); p.prefix(); p.load(); p.halt();
    // Remove the explicit drain, leaving fixed-latency work before the halt.
    p.ops[p.ops.size() - 3]->setAttr("cycles", p.b.getI32IntegerAttr(0));
    p.expect("fixed-latency completion remains required", false);
  }
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &diagnostic) {
    llvm::raw_string_ostream out(diagnostics); out << diagnostic << '\n';
    for (const Diagnostic &note : diagnostic.getNotes()) out << note << '\n';
    return success();
  });
  crossBlock(context);
  alternating(context, false); alternating(context, true);
  captured(context, false); captured(context, true);
  delayedBranch(context, true); delayedBranch(context, false);
  conditions(context); widening(context);
  memoryHazards(context); failureCases(context);
  llvm::outs() << checks << " CFG DMA timing checks, " << failures << " failures\n";
  return failures != 0;
}
