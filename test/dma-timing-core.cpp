// Standalone checks of shared graph semantics; the selected evidence loader
// and actual command-domain admission are exercised by test_rtl_timing.py.
#include "Atlas/AtlasTiming.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace mlir::atlas::timing;

static void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

static Instr instruction(const char *name, int rd = 0, int rs1 = 0,
                         int rs2 = 0, int imm = 0) {
  return {findOp(name), rd, rs1, rs2, imm};
}

// Fixture for the selected capture policy, using legacy finite VLS footprints
// solely to exercise graph ordering independently of the evidence artifact.
static Footprint captured(const Instr &in, const RegValues &regs) {
  Footprint f = footprintOf(in, regs);
  if (in.op->opClass == OpClass::DmaConfig) {
    f.dmaAsync = false;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      access.atCompletion = false;
  } else if (in.op->opClass == OpClass::DmaLoad ||
             in.op->opClass == OpClass::DmaStore) {
    f.exclusiveVmemUntilWait = true;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      if (access.res == Res::XReg || access.res == Res::DmaBase)
        access.atCompletion = false;
    Access dram{Res::Dram, in.op->opClass == OpClass::DmaStore,
                0, 1, 0, 0};
    dram.anywhere = true;
    dram.atCompletion = true;
    f.accesses.push_back(dram);
  }
  return f;
}

static bool hasEdge(const DepGraph &graph, int from, int to) {
  return std::any_of(graph.edges.begin(), graph.edges.end(),
                     [&](const Edge &edge) {
                       return edge.from == from && edge.to == to;
                     });
}

int main() {
  RegValues regs = unknownRegs();
  regs[1] = 0x90000000;
  regs[2] = 128;
  regs[5] = 0;
  regs[6] = 256;
  regs[8] = 65792;
  const Instr load = instruction("dma.load.ch0", 6, 1, 2);
  const Instr store = instruction("dma.store.ch1", 1, 6, 2);
  const Instr config = instruction("dma.config.ch7", 0, 5);
  const Instr wait0 = instruction("dma.wait.ch0");
  const Instr wait1 = instruction("dma.wait.ch1");
  const Instr overwrite = instruction("addi", 1, 0, 0, 17);
  const Instr vload = instruction("vload", 4, 8);

  for (const Instr &command : {load, store, config}) {
    Footprint f = footprintOf(command, regs);
    check(f.dmaAsync && f.dmaCycles > 0,
          "legacy DMA commands must preserve asynchronous model behavior");
    check(std::all_of(f.accesses.begin(), f.accesses.end(),
                      [](const Access &access) { return access.atCompletion; }),
          "legacy operands must remain completion-time accesses");
  }
  Footprint dma = captured(load, regs);
  Footprint cfg = captured(config, regs);
  Footprint scalar = captured(overwrite, regs);
  Footprint vls = captured(vload, regs);
  EdgeKind kind;
  check(!cfg.dmaAsync && cfg.dmaCycles == 0,
        "captured DMA config must not create a transfer generation");
  check(!conflictsAtCompletion(dma, scalar, kind),
        "captured scalar operands must be reusable immediately after launch");
  check(!conflictsAtCompletion(dma, cfg, kind),
        "captured global base may be reconfigured after launch");
  check(conflictsAtCompletion(dma, vls, kind),
        "pending DMA must exclude even disjoint-bank VLS");
  check(conflictsAtCompletion(captured(store, regs), vls, kind),
        "pending DMA must exclude read/read VMEM overlap");
  Dependence drain = dependence(vload, vls, load, dma);
  check(drain.distance >= vls.doneAge + 1,
        "DMA launch must drain preceding finite VMEM work");

  DepGraph graph = buildGraph({load, overwrite, config, wait0, vload},
                             regs, 0, nullptr, captured);
  check(hasEdge(graph, 3, 4),
        "matching wait must guard later VLS even after synchronous config");
  check(dependence(load, dma, overwrite, scalar).distance == 1,
        "captured operand overwrite needs only the launch-time WAR order");
  check(!hasEdge(graph, 1, 3),
        "local wait must not invent completion-time scalar dependencies");
  check(graph.footprints[0].dmaCycles == 0,
        "captured DMA must carry no fabricated completion latency");

  graph = buildGraph({load, wait0, store, wait1, vload},
                     regs, 0, nullptr, captured);
  check(hasEdge(graph, 1, 2) && hasEdge(graph, 3, 4),
        "successive transfer generations must preserve their matching waits");

  // A same-channel synchronous config is not a local transfer: incoming
  // completion guards still belong to the wait, not to that config.
  IncomingDma incoming;
  incoming[0].push_back(dma);
  graph = buildGraph({instruction("dma.config.ch0", 0, 5), wait0, vload},
                     regs, 0, &incoming, captured);
  check(hasEdge(graph, 1, 2),
        "synchronous config must not hide incoming DMA completion guards");

  TargetTiming selected(captured);
  const Instr mxu = instruction("vmatmul.mxu0", 0, 4, 0);
  const Instr push = instruction("vmatpush.weight.mxu0", 0, 6);
  const Instr vpu = instruction("vmov", 4, 6);
  check(!selected.resolve(mxu, regs).error.empty() &&
        !selected.resolve(vpu, regs).error.empty(),
        "selected callbacks must not enable unsupported compute policies");
  check(TargetTiming().resolve(mxu, regs).error.empty() &&
        TargetTiming().resolve(vpu, regs).error.empty(),
        "default legacy compute resolution must remain available");
  check(dependence(mxu, {}, push, {}).distance == 63 &&
        dependence(mxu, {}, push, {}, selected).distance == 0,
        "selected pair rules must not inherit model MXU sequencing");
  check(unitCapacity(Unit::MxuCompute, 0) == 3 &&
        unitCapacity(Unit::MxuCompute, 1) == 2 &&
        unitCapacity(Unit::MxuCompute, 0, selected) == 1,
        "selected unit capacity must not inherit model in-flight capacity");
  check(!ReservationTable(selected).conflict(mxu, {}, 0).empty() &&
        !ReservationTable(selected).conflict(vpu, {}, 0).empty(),
        "selected reservations must fail closed even if resolution is bypassed");
  const Instr xlu = instruction("vtrpose.xlu", 4, 6);
  Footprint transpose = selected.resolve(xlu, regs);
  check(transpose.error.empty() && transpose.doneAge == 65,
        "XLU generic accesses and single-engine hold remain supported");
  ReservationTable xluReservations(selected);
  xluReservations.reserve(xlu, transpose, 0);
  check(!xluReservations.conflict(xlu, transpose, 65).empty() &&
        xluReservations.conflict(xlu, transpose, 66).empty(),
        "XLU whole-engine reservation must release at cycle 66");

  // This fixture supplies reviewed row accesses independently of the policy
  // gate; production selected rules must come from the evidence provider.
  TargetTiming multiplyTarget([](const Instr &in, const RegValues &values) {
    Footprint f = captured(in, values);
    if (in.op->name == "vmul.bf16") {
      f.vpuLive = 0;
      f.holds.push_back({Unit::Vpu, 0, 0, 65});
    }
    return f;
  });
  multiplyTarget.computePolicies.insert(TargetTiming::ComputePolicy::VmulBf16);
  const Instr multiply = instruction("vmul.bf16", 4, 0, 2);
  Footprint multiplication = multiplyTarget.resolve(multiply, regs);
  check(!selected.resolve(multiply, regs).error.empty() &&
        multiplication.error.empty(),
        "VMUL policy must require an explicit opt-in");
  for (const Instr &unsupported : {vpu, instruction("vadd.bf16", 4, 0, 2),
                                  instruction("vpack.bf16.fp8", 4, 0, 2), mxu})
    check(!multiplyTarget.resolve(unsupported, regs).error.empty() &&
          !ReservationTable(multiplyTarget).conflict(unsupported, {}, 0).empty(),
          "VMUL opt-in must not enable another VPU or MXU operation");
  OpInfo wrongEngine = *multiply.op;
  wrongEngine.engine = Engine::Mxu0;
  check(!multiplyTarget.allowsOperation(wrongEngine),
        "a VMUL mnemonic on another engine must not bypass policy admission");
  check(unitCapacity(Unit::Vpu, 0, multiplyTarget) == 1,
        "selected VPU whole-engine capacity must be one");
  ReservationTable multiplyReservations(multiplyTarget);
  multiplyReservations.reserve(multiply, multiplication, 0);
  check(!multiplyReservations.conflict(multiply, multiplication, 65).empty() &&
        multiplyReservations.conflict(multiply, multiplication, 66).empty(),
        "selected VPU hold must include the final write and release at 66");
  ReservationTable waitReservations(multiplyTarget);
  waitReservations.reserve(multiply, multiplication, 10);
  waitReservations.extendForWait(5);
  check(!waitReservations.conflict(multiply, multiplication, 5).empty(),
        "unknown wait must conservatively extend a future VPU engine hold");
  Footprint physicalAlias = multiplyTarget.resolve(
      instruction("vmul.bf16", 4, 0, 32), regs);
  check(!ReservationTable(multiplyTarget)
             .conflict(instruction("vmul.bf16", 4, 0, 32), physicalAlias, 0)
             .empty(),
        "selected VMUL must not inherit model sharing for aliased physical reads");

  std::cout << "DMA timing core checks passed\n";
}
