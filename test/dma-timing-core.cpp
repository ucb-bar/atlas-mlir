// Shared graph and reservation semantics of a selected target, using legacy
// footprints rewritten to the dma=wait capture policy.
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

static Footprint captured(const Instr &in, const RegValues &regs) {
  Footprint f = footprintOf(in, regs);
  if (in.op->opClass == OpClass::DmaConfig) {
    f.dmaAsync = false;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      access.atCompletion = false;
  } else if (f.dmaAsync) {
    f.exclusiveVmemUntilWait = true;
    f.dmaCycles = 0;
    for (Access &access : f.accesses)
      if (access.res == Res::XReg || access.res == Res::DmaBase)
        access.atCompletion = false;
    f.accesses.push_back({Res::Dram, in.op->opClass == OpClass::DmaStore, 0,
                          1, 0, 0, true, true});
  }
  if (in.op->name == "vmul.bf16") {
    f.vpuLive = 0;
    f.holds.push_back({Unit::Vpu, 0, 0, 65});
  }
  return f;
}

static bool hasEdge(const DepGraph &graph, int from, int to) {
  return std::any_of(graph.edges.begin(), graph.edges.end(),
                     [&](const Edge &e) { return e.from == from && e.to == to; });
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
  const TargetTiming selected{captured};

  for (const Instr &command : {load, store, config}) {
    Footprint f = footprintOf(command, regs);
    check(f.dmaAsync && f.dmaCycles > 0 &&
              std::all_of(f.accesses.begin(), f.accesses.end(),
                          [](const Access &a) { return a.atCompletion; }),
          "legacy DMA must stay asynchronous with completion-time accesses");
  }
  Footprint dma = captured(load, regs);
  Footprint cfg = captured(config, regs);
  Footprint scalar = captured(overwrite, regs);
  Footprint vls = captured(vload, regs);
  EdgeKind kind;
  check(!conflictsAtCompletion(dma, scalar, kind) &&
            !conflictsAtCompletion(dma, cfg, kind),
        "captured scalar operands and base may be reused after launch");
  check(conflictsAtCompletion(dma, vls, kind) &&
            conflictsAtCompletion(captured(store, regs), vls, kind),
        "pending DMA must exclude all VMEM work");
  check(dependence(vload, vls, load, dma).distance >= vls.doneAge + 1,
        "DMA launch must drain preceding finite VMEM work");
  check(dependence(load, dma, overwrite, scalar).distance == 1,
        "captured operand overwrite needs only launch-time WAR order");

  DepGraph graph = buildGraph({load, overwrite, config, wait0, vload}, regs, 0,
                              nullptr, selected);
  check(hasEdge(graph, 3, 4) && !hasEdge(graph, 1, 3) &&
            graph.footprints[0].dmaCycles == 0,
        "the wait guards later VLS without completion-time scalar edges");
  graph = buildGraph({load, wait0, store, wait1, vload}, regs, 0, nullptr,
                     selected);
  check(hasEdge(graph, 1, 2) && hasEdge(graph, 3, 4),
        "successive transfers must keep their matching waits");
  // A same-channel synchronous config is not a local transfer.
  IncomingDma incoming;
  incoming[0].push_back(dma);
  graph = buildGraph({instruction("dma.config.ch0", 0, 5), wait0, vload}, regs,
                     0, &incoming, selected);
  check(hasEdge(graph, 1, 2), "config must not hide incoming DMA guards");

  const Instr mxu = instruction("vmatmul.mxu0", 0, 4, 0);
  const Instr push = instruction("vmatpush.weight.mxu0", 0, 6);
  check(dependence(mxu, {}, push, {}).distance == 63 &&
            dependence(mxu, {}, push, {}, selected).distance == 0,
        "selected target must drop model MXU sequencing");
  check(unitCapacity(Unit::MxuCompute, 0) == 3 &&
            unitCapacity(Unit::MxuCompute, 0, selected) == 1 &&
            unitCapacity(Unit::Vpu, 0, selected) == 1,
        "selected units have capacity one");

  const Instr xlu = instruction("vtrpose.xlu", 4, 6);
  Footprint transpose = selected.resolve(xlu, regs);
  ReservationTable xluTable(selected);
  xluTable.reserve(xlu, transpose, 0);
  check(transpose.doneAge == 65 &&
            !xluTable.conflict(xlu, transpose, 65).empty() &&
            xluTable.conflict(xlu, transpose, 66).empty(),
        "XLU reservation must release at cycle 66");

  const Instr multiply = instruction("vmul.bf16", 4, 0, 2);
  Footprint product = selected.resolve(multiply, regs);
  ReservationTable vpuTable(selected);
  vpuTable.reserve(multiply, product, 0);
  check(!vpuTable.conflict(multiply, product, 65).empty() &&
            vpuTable.conflict(multiply, product, 66).empty(),
        "VPU hold must release at cycle 66");
  ReservationTable waitTable(selected);
  waitTable.reserve(multiply, product, 10);
  waitTable.extendForWait(5);
  check(!waitTable.conflict(multiply, product, 5).empty(),
        "an unknown wait must extend a future VPU hold");
  const Instr alias = instruction("vmul.bf16", 4, 0, 32);
  check(!ReservationTable(selected)
             .conflict(alias, selected.resolve(alias, regs), 0)
             .empty(),
        "selected VPU must not share aliased physical reads");

  std::cout << "DMA timing core checks passed\n";
}
