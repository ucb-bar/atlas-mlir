#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasGuardedPack.h"
#include "Atlas/AtlasPipeline.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasStreamVerification.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "VerificationTestSupport.h"
#include "mlir/Pass/PassInstrumentation.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
constexpr StringLiteral finalizationSource = R"mlir(module {
  func.func @finalization() -> !atlas.virtual_state attributes {
    atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64
  } {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io, %x = "atlas.virtual_input_bf16"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %flag = arith.constant 0 : i1
    cf.cond_br %flag, ^left(%io : !atlas.virtual_state), ^right(%io : !atlas.virtual_state)
  ^left(%ls: !atlas.virtual_state):
    %a = "atlas.virtual_vpu_unary"(%x) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    cf.br ^join(%ls, %a : !atlas.virtual_state, !atlas.virtual_bf16)
  ^right(%rs: !atlas.virtual_state):
    cf.br ^join(%rs, %x : !atlas.virtual_state, !atlas.virtual_bf16)
  ^join(%js: !atlas.virtual_state, %v: !atlas.virtual_bf16):
    %out = "atlas.virtual_output_bf16"(%js, %v) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %out : !atlas.virtual_state
  }
})mlir";

using FinalMutation = std::function<void(ModuleOp)>;
struct BeforeFinalization final : PassInstrumentation {
  BeforeFinalization(FinalMutation mutate, bool &seen)
      : mutate(std::move(mutate)), seen(seen) {}
  void runBeforePass(Pass *pass, Operation *operation) override {
    if (pass->getArgument() == "finalize-atlas-mandatory-compilation") {
      seen = true;
      mutate(cast<ModuleOp>(operation));
    }
  }
  FinalMutation mutate;
  bool &seen;
};

LogicalResult compileFinalization(ModuleOp module, bool legacy,
                                  FinalMutation mutate = {}, bool *seen = nullptr) {
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());
  if (legacy) {
    pm.addPass(createSpillAtlasVirtualPass());
    pm.addPass(createVerifyAtlasVirtualStreamPass());
    pm.addPass(createLowerAtlasPackRelayoutPass());
    pm.addPass(createSpillAtlasVirtualPass());
    pm.addPass(createVerifyAtlasVirtualStreamPass());
    pm.addPass(createLowerAtlasVirtualToMachinePass());
    pm.addPass(createInsertAtlasDMAWaitsPass());
    pm.addPass(createVerifyAtlasMachineStreamPass());
    pm.addPass(createVerifyAtlasGeneratedSchedulePass());
    pm.addPass(createInsertAtlasDelaysPass());
    pm.addPass(createVerifyAtlasMachineStreamPass());
    pm.addPass(createVerifyAtlasTimingPass());
    pm.addPass(createVerifyAtlasGeneratedSchedulePass());
  } else {
    AtlasCompileOptions options;
    buildAtlasCompileUnscheduledPipeline(pm, options);
    if (mutate)
      pm.addInstrumentation(std::make_unique<BeforeFinalization>(std::move(mutate), *seen));
  }
  return pm.run(module);
}

void finalization(MLIRContext &context) {
  for (unsigned variant = 0; variant != 3; ++variant) {
    auto source = parseSourceString<ModuleOp>(finalizationSource, &context);
    Builder b(&context);
    if (variant)
      (*source)->setAttr("atlas.timing_assumptions", b.getArrayAttr({b.getStringAttr("stale")}));
    if (variant == 2) {
      auto word = b.getDictionaryAttr({b.getNamedAttr("address", b.getI64IntegerAttr(0x90002000)),
                                      b.getNamedAttr("value", b.getI64IntegerAttr(0))});
      (*source)->setAttr("atlas.dram_contract", b.getDictionaryAttr({
          b.getNamedAttr("name", b.getStringAttr("pipeline-test")),
          b.getNamedAttr("words", b.getArrayAttr({word}))}));
    }
    OwningOpRef<ModuleOp> legacy(cast<ModuleOp>((*source)->clone()));
    diagnostics.clear();
    bool complete = succeeded(compileFinalization(*source, false)) &&
                    succeeded(compileFinalization(*legacy, true));
    check(complete, "new and legacy finalization complete", diagnostics);
    if (!complete) continue;
    std::string actual, expected;
    llvm::raw_string_ostream out(actual), old(expected);
    source->print(out); legacy->print(old);
    check(actual == expected, "mandatory finalization preserves all machine IR and metadata");
    SmallVector<uint32_t> words, oldWords;
    check(succeeded(encodeAtlasWords(*source, words)) &&
          succeeded(encodeAtlasWords(*legacy, oldWords)) && words == oldWords,
          "mandatory finalization preserves encoded instruction words");
    auto assumptions = (*source)->getAttrOfType<ArrayAttr>("atlas.timing_assumptions");
    check(variant == 2 ? assumptions == b.getArrayAttr({b.getStringAttr("immutable-dram:pipeline-test")})
                      : !assumptions,
          "mandatory finalization replaces or removes stale timing assumptions");
  }
  const std::pair<StringRef, FinalMutation> faults[] = {
      {"shortened delays", [](ModuleOp module) {
        Builder b(module.getContext());
        for (auto delay : module.getOps<DelayOp>()) delay->setAttr("cycles", b.getI32IntegerAttr(0));
      }},
      {"pending DMA", [](ModuleOp module) {
        module->setAttr(kAtlasDMAWaitsPending, UnitAttr::get(module.getContext()));
      }},
      {"missing source contract", [](ModuleOp module) { module->removeAttr(kAtlasCFGContract); }},
      {"corrupt source contract", [](ModuleOp module) {
        module->setAttr(kAtlasCFGContract, Builder(module.getContext()).getDictionaryAttr({}));
      }},
      {"escaping LLVM target", [](ModuleOp module) {
        for (auto jump : module.getOps<JumpOp>()) {
          jump->setAttr("offset", Builder(module.getContext()).getI32IntegerAttr(1048574));
          break;
        }
      }},
      {"untimed boundary", [](ModuleOp module) {
        module->setAttr(kAtlasTimingState, Builder(module.getContext()).getStringAttr("untimed"));
      }},
      {"custom provider boundary", [](ModuleOp module) {
        module->setAttr(kAtlasTimingProvider, Builder(module.getContext()).getStringAttr("custom"));
      }},
      {"instruction memory limit", [](ModuleOp module) {
        module->setAttr(kAtlasProgramSizeLimit, Builder(module.getContext()).getI64IntegerAttr(4));
      }},
  };
  for (const auto &[name, mutate] : faults) {
    auto module = parseSourceString<ModuleOp>(finalizationSource, &context);
    bool seen = false;
    diagnostics.clear();
    bool rejected = failed(compileFinalization(*module, false, mutate, &seen));
    check(seen, "negative control reached final mandatory boundary: " + name.str());
    check(rejected && !diagnostics.empty(), "final mandatory boundary rejects " + name.str(), diagnostics);
  }
}

struct Program : IssuedStream {
  using IssuedStream::IssuedStream;
  Operation *addi(int dst, int src, int immediate) {
    return emit("atlas.alu_imm", {text("kind", "addi"), i("dst", dst), i("src", src), i("immediate", immediate)});
  }
  void toggle(int reg) {
    emit("atlas.alu_imm", {text("kind", "xori"), i("dst", reg), i("src", reg), i("immediate", 256)});
  }
  void prefix() { constant(4, 256); constant(5, 0); constant(7, 0x90000000); constant(9, 1024); }
  void launch(int channel = 0, StringRef direction = "load", int size = 9) {
    emit("atlas.dma", {text("direction", direction), i("channel", channel), i("reg", 4), i("dram", 7), i("size", size)});
  }
  void wait(int channel = 0) { emit("atlas.dma_wait", {i("channel", channel)}); }
  void contract(uint32_t value) {
    auto word = b.getDictionaryAttr({b.getNamedAttr("address", b.getI64IntegerAttr(0x90001000)),
                                     b.getNamedAttr("value", b.getI64IntegerAttr(value))});
    (*module)->setAttr("atlas.dram_contract", b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr("test-metadata-v1")),
        b.getNamedAttr("words", b.getArrayAttr({word}))}));
  }
  void metadata(bool completed = true, int upper = 0) {
    constant(19, upper);
    emit("atlas.dma_config", {i("channel", 0), i("base_reg", 19)});
    constant(14, 0x2000); constant(15, 0x90001000); constant(16, 32); constant(17, 0x8000);
    emit("atlas.dma", {text("direction", "load"), i("channel", 2), i("reg", 14), i("dram", 15), i("size", 16)});
    if (completed) wait(2);
  }
  void scalarLoad(int dst) {
    emit("atlas.scalar_load", {text("kind", "lw"), i("dst", dst), i("base", 17), i("offset", 0)});
    emit("atlas.delay", {i("cycles", 4)});
    nop();
  }
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

void footprintState(MLIRContext &context) {
  {
    Program p(context); p.prefix(); p.constant(10, 2);
    size_t loop = p.ops.size();
    p.launch(); p.load(); p.wait();
    p.addi(5, 5, 256); p.addi(10, 10, -1);
    Operation *back = p.branch(10); p.nop(); p.halt(); p.aim(back, loop);
    // The same VLOAD is disjoint on its first visit, then overlaps DMA.
    p.expect("footprints follow a vector base changed by the previous iteration",
             false, {"RAW conflict"});
  }
  {
    Program p(context); p.constant(10, 2);
    size_t loop = p.ops.size();
    p.addi(10, 10, -1);
    Operation *back = p.branch(10); p.nop(); p.halt(); p.aim(back, loop);
    auto stream = readAtlasStream(*p.module, AtlasStreamReadMode::Verification);
    auto provider = timing::npuModelTimingProvider();
    // Keep the built-in id deliberately: callback identity, not its label,
    // determines whether the audited register-independent rules apply.
    provider.footprint = [](const timing::Instr &in, const timing::RegValues &regs) {
      auto result = timing::footprintOf(in, regs);
      if (in.op->opClass == timing::OpClass::Alu && in.rd == 10 && regs[10] == 1)
        result.error = "custom second-iteration footprint refusal";
      return result;
    };
    expectVerified("same-id custom footprint still runs on every loop visit", false,
                   {"custom second-iteration footprint refusal"}, [&] {
      return failed(stream) ? failure() : checkAtlasDMACFG(*stream, provider);
    });
  }
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

void metadataContracts(MLIRContext &context) {
  for (int variant = 0; variant != 10; ++variant) {
    Program p(context); p.prefix();
    if (variant != 1) p.contract(variant == 2 ? 256 : 0);
    p.metadata(variant != 3, variant == 4 ? 1 : 0);
    if (variant >= 5 && variant <= 7) {
      StringRef kind = variant == 5 ? "sb" : variant == 6 ? "sh" : "sw";
      p.emit("atlas.scalar_store", {p.text("kind", kind), p.i("src", 0), p.i("base", 17), p.i("offset", 0)});
    }
    if (variant == 8) {
      p.emit("atlas.vstore", {p.i("src", 0), p.i("base", 14), p.i("offset", 0), p.text("format", "raw")});
      p.emit("atlas.delay", {p.i("cycles", 66)}); p.nop();
    }
    if (variant == 9) {
      p.constant(15, 0x90002000);
      p.emit("atlas.dma", {p.text("direction", "load"), p.i("channel", 2), p.i("reg", 14), p.i("dram", 15), p.i("size", 16)});
      p.wait(2);
    }
    p.scalarLoad(5); p.launch(); p.load(); p.wait(); p.halt();
    p.expect("exact metadata input " + std::to_string(variant), variant == 0,
             variant == 0 ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"RAW conflict"});
  }
  for (uint32_t bound : {1u, 2u}) {
    Program p(context); p.prefix(); p.contract(bound); p.metadata(); p.scalarLoad(10);
    p.constant(5, (kAtlasVmemBytes - 1024) / 4);
    size_t loop = p.ops.size(); p.load(); p.addi(5, 5, 256); p.addi(10, 10, -1);
    Operation *back = p.branch(10); p.nop(); p.halt(); p.aim(back, loop);
    p.expect("loaded loop bound " + std::to_string(bound), bound == 1,
             bound == 1 ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"past the end of VMEM"});
  }
  {
    Program p(context); p.prefix(); p.contract(0); p.metadata();
    p.constant(7, 0x90001000); p.launch(0, "store"); p.wait(); p.halt();
    p.expect("device store cannot overwrite contracted DRAM", false, {"immutable DRAM contract"});
  }
  {
    Program p(context); p.prefix(); p.contract(0); p.metadata();
    p.scalarLoad(5); p.launch(); p.load(); p.wait();
    p.emit("atlas.delay", {p.i("cycles", 128)}); p.halt();
    p.expect("a data contract cannot suppress DELAY-before-halt", false, {"halt does not wait for a preceding DELAY"});
  }
  {
    Program p(context); p.prefix(); p.contract(0);
    p.emit("atlas.dma_config", {p.i("channel", 0), p.i("base_reg", 0)});
    p.constant(14, 0x2000); p.constant(15, 0x90001000); p.constant(16, 32); p.constant(17, 0x8000);
    // Launch path queues the safe singleton first. The equally deep no-launch
    // path forces a hull before the merge runs, without queuing its singleton.
    Operation *chooseLaunch = p.branch(27); p.nop();
    Operation *noLaunchToMerge = p.jump(); p.nop();
    size_t launchPath = p.ops.size();
    p.emit("atlas.dma", {p.text("direction", "load"), p.i("channel", 2), p.i("reg", 14), p.i("dram", 15), p.i("size", 16)});
    Operation *launchToMerge = p.jump(); p.nop();
    size_t merge = p.ops.size(); p.wait(2); p.scalarLoad(5);
    p.launch(); p.load(); p.wait(); p.halt(); p.aim(chooseLaunch, launchPath);
    p.aim(noLaunchToMerge, merge); p.aim(launchToMerge, merge);
    p.expectDMA("widened may-pending metadata never becomes a definite constant", false, 1);
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
    p.expect("repeated launch cannot overwrite an older pending window", false, {"reuses DMA channel 0", "atlas.dma_wait"});
  }
}

void commandRing(MLIRContext &context) {
  for (bool drain : {false, true}) {
    Program p(context); p.prefix(); p.constant(9, 32); p.launch();
    p.constant(4, 512);
    for (unsigned n = 0; n < 8; ++n) {
      if (drain && n == 7) p.wait();
      p.launch(1); p.wait(1);
    }
    p.wait(); p.halt();
    p.expect(drain ? "full drain permits the ninth command" : "ninth command cannot overwrite pending slot zero",
             drain, drain ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"pending DMA command ring slot"});
  }
  // Slot zero can be reused while slot one is still pending. A blanket limit
  // of eight launches since the last full drain would reject this safe case.
  for (unsigned count : {7u, 8u}) {
    Program p(context); p.prefix(); p.constant(9, 32); p.launch();
    p.constant(4, 512); p.launch(1); p.wait(); p.constant(4, 768);
    Operation *jump = p.jump(); p.nop();
    size_t target = p.ops.size();
    for (unsigned n = 0; n < count; ++n) { p.launch(2); p.wait(2); }
    p.wait(1); p.halt(); p.aim(jump, target);
    p.expect(count == 7 ? "retired slot zero is reusable across a CFG edge while slot one stays live"
                       : "ring index survives CFG edges and protects pending slot one",
             count == 7, count == 7 ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"pending DMA command ring slot"});
  }
  {
    Program p(context); p.prefix(); p.launch(); p.constant(4, 512); p.launch(1);
    for (int i = 0; i < 16; ++i) {
      int channel = i % 2;
      p.wait(channel); p.constant(4, channel == 0 ? 256 : 512); p.launch(channel);
    }
    p.wait(); p.wait(1); p.halt();
    p.expect("retired command slots permit eighteen launches without a complete drain", true);
  }
  {
    Program p(context); p.prefix(); p.constant(9, 32); p.launch();
    p.constant(4, 512); p.constant(10, 0);
    size_t loop = p.ops.size();
    p.launch(1); p.wait(1); p.addi(10, 10, 1);
    Operation *back = p.jump(); p.nop(); p.aim(back, loop); p.halt();
    p.expectDMA("widening cannot forget a live ring slot when launch count becomes unknown", false, 2);
  }
  {
    Program p(context); p.prefix(); p.constant(9, 32); p.launch();
    p.constant(10, 4); size_t loop = p.ops.size(); p.addi(10, 10, -1);
    Operation *back = p.branch(10); p.nop(); p.aim(back, loop);
    p.constant(4, 512); p.launch(1); p.wait(1); p.wait(); p.halt();
    p.expectDMA("widening unrelated registers preserves a known free ring slot", true, 2);
  }
}

void dmaSizes(MLIRContext &context) {
  for (StringRef direction : {StringRef("load"), StringRef("store")}) {
    for (uint32_t size : {0u, 1u, 31u, 33u, 8191u, 8192u, 8224u, 16384u, uint32_t(-32)}) {
      Program p(context); p.prefix(); p.constant(9, size);
      p.launch(0, direction); p.wait(); p.halt();
      p.expect("DMA " + direction.str() + " rejects size " + std::to_string(size), false,
               {"DMA size must be nonzero, a multiple of 32 bytes, and less than 8192 bytes"});
    }
    for (uint32_t size : {32u, 4064u, 4096u}) {
      Program p(context); p.prefix(); p.constant(9, size);
      p.launch(0, direction); p.wait(); p.halt();
      p.expect("DMA " + direction.str() + " accepts size " + std::to_string(size), true);
    }
    for (uint32_t size : {4128u, 8160u}) {
      Program p(context); p.prefix(); p.constant(9, size);
      p.launch(0, direction); p.wait(); p.halt();
      bool load = direction == "load";
      p.expect("DMA " + direction.str() + (load ? " accepts size " : " rejects size ") + std::to_string(size), load,
               load ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"DMA store size must not exceed 4096 bytes"});
    }
    {
      Program p(context); p.prefix(); p.launch(0, direction, 27);
      // Even an access below the known base may alias an unknown-sized DMA.
      p.emit("atlas.vstore", {p.i("src", 0), p.i("base", 5), p.i("offset", 0), p.text("format", "raw")});
      p.wait(); p.halt();
      p.expect("unknown DMA " + direction.str() + " size remains conservative", false, {"conflict"});
    }
    {
      Program p(context); p.prefix(); p.launch(0, direction, 27); p.wait(); p.halt();
      p.expect("unknown DMA " + direction.str() + " size permits an explicit wait", true);
    }
  }
}

void channelReuse(MLIRContext &context) {
  for (StringRef first : {StringRef("load"), StringRef("store")}) {
    for (StringRef second : {StringRef("load"), StringRef("store")}) {
      {
        Program p(context); p.prefix(); p.launch(3, first); p.constant(4, 512); p.launch(3, second);
        p.wait(3); p.halt();
        p.expect("pending " + first.str() + " rejects same-channel " + second.str(), false,
                 {"reuses DMA channel 3", "atlas.dma_wait"});
      }
      {
        Program p(context); p.prefix(); p.launch(3, first); p.constant(4, 512); p.launch(4, second);
        p.wait(3); p.wait(4); p.halt();
        p.expect("disjoint " + first.str() + " and " + second.str() + " permit different channels", true);
      }
      {
        Program p(context); p.prefix(); p.launch(3, first); p.wait(3); p.launch(3, second);
        p.wait(3); p.halt();
        p.expect("waited " + first.str() + " permits same-channel " + second.str(), true);
      }
    }
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
  registry.insert<AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.disableMultithreading();
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &diagnostic) {
    llvm::raw_string_ostream out(diagnostics); out << diagnostic << '\n';
    for (const Diagnostic &note : diagnostic.getNotes()) out << note << '\n';
    return success();
  });
  crossBlock(context);
  alternating(context, false); alternating(context, true);
  footprintState(context);
  captured(context, false); captured(context, true);
  delayedBranch(context, true); delayedBranch(context, false);
  conditions(context); widening(context);
  metadataContracts(context);
  memoryHazards(context); commandRing(context); dmaSizes(context); channelReuse(context); failureCases(context);
  finalization(context);
  llvm::outs() << checks << " CFG DMA timing checks, " << failures << " failures\n";
  return failures != 0;
}
