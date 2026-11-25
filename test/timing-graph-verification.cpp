// Retain the original all-pairs graph as an oracle for candidate indexing.
#include "Atlas/AtlasTiming.h"
#include "VerificationTestSupport.h"
#include <algorithm>
#include <map>
#include <random>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;
using namespace atlas_test;

namespace {
struct BruteEdges {
  DepGraph &g;
  std::map<std::pair<int, int>, int> index;
  void add(int from, int to, int distance, EdgeKind kind, const std::string &reason) {
    auto [it, inserted] = index.emplace(std::make_pair(from, to), g.edges.size());
    if (inserted) g.edges.push_back({from, to, distance, kind, reason});
    else if (distance > g.edges[it->second].distance)
      g.edges[it->second] = {from, to, distance, kind, reason};
  }
};
DepGraph bruteForceGraph(const std::vector<Instr> &instrs,
                         const RegValues &entry,
                         const IncomingDma *incomingDma) {
  DepGraph g;
  g.nodes = instrs;
  int n = static_cast<int>(instrs.size());
  RegValues regs = entry;
  for (const Instr &in : instrs) {
    g.footprints.push_back(footprintOf(in, regs));
    applyScalar(in, regs);
  }

  BruteEdges edges{g};
  for (int b = 0; b < n; b++)
    for (int a = 0; a < b; a++) {
      Dependence d =
          dependence(g.nodes[a], g.footprints[a], g.nodes[b], g.footprints[b]);
      if (d.distance > 0)
        edges.add(a, b, d.distance, d.kind, d.reason);
    }

  // A DMA's completion is known only once its dma.wait issues, so later
  // conflicting accesses wait for that dma.wait.
  for (int d = 0; d < n; d++) {
    const OpInfo &op = *g.nodes[d].op;
    if (op.opClass != OpClass::DmaLoad && op.opClass != OpClass::DmaStore)
      continue;
    int wait = -1;
    for (int k = d + 1; k < n && wait < 0; k++)
      if (g.nodes[k].op->opClass == OpClass::DmaWait &&
          g.nodes[k].op->channel == op.channel)
        wait = k;
    for (int k = d + 1; k < n; k++) {
      EdgeKind kind;
      if (k == wait ||
          !conflictsAtCompletion(g.footprints[d], g.footprints[k], kind))
        continue;
      if (wait >= 0 && wait < k)
        edges.add(wait, k, 1, kind,
                  std::string(edgeKindName(kind)) + " with " + op.name +
                      " (done once the wait issues)");
      else
        edges.add(d, k, 1, EdgeKind::Order,
                  op.name + " may still be in flight (no dma.wait in between)");
    }
  }

  // A wait for a transfer from an earlier block guards conflicting accesses
  // and channel reuse.
  for (int w = 0; w < n; w++) {
    const OpInfo &op = *g.nodes[w].op;
    if (op.opClass != OpClass::DmaWait)
      continue;
    bool local = false;
    for (int d = 0; d < w; d++)
      if ((g.nodes[d].op->opClass == OpClass::DmaLoad ||
           g.nodes[d].op->opClass == OpClass::DmaStore) &&
          g.nodes[d].op->channel == op.channel)
        local = true;
    if (local)
      continue;
    for (int k = w + 1; k < n; k++) {
      bool guarded = g.nodes[k].op->opClass == OpClass::DmaLoad ||
                     g.nodes[k].op->opClass == OpClass::DmaStore;
      if (incomingDma) {
        guarded = guarded && g.nodes[k].op->channel == op.channel;
        for (const Footprint &dma : (*incomingDma)[op.channel]) {
          EdgeKind kind;
          guarded |= conflictsAtCompletion(dma, g.footprints[k], kind);
        }
      } else {
        // The transfer latched its registers at launch, so only the VMEM it
        // moves at completion may conflict.
        for (const Access &a : g.footprints[k].accesses)
          if (a.res == Res::Vmem)
            guarded = true;
      }
      if (guarded)
        edges.add(w, k, 1, EdgeKind::Order,
                  op.name + " guards a transfer started in an earlier block");
    }
  }

  g.in.assign(n, {});
  g.out.assign(n, {});
  for (int e = 0; e < static_cast<int>(g.edges.size()); e++) {
    g.out[g.edges[e].from].push_back(e);
    g.in[g.edges[e].to].push_back(e);
  }
  return g;
}

bool equalGraphs(const DepGraph &a, const DepGraph &b) {
  if (a.edges.size() != b.edges.size() || a.in != b.in || a.out != b.out)
    return false;
  for (size_t i = 0; i < a.edges.size(); ++i) {
    const Edge &x = a.edges[i], &y = b.edges[i];
    if (std::tie(x.from, x.to, x.distance, x.kind, x.reason) !=
        std::tie(y.from, y.to, y.distance, y.kind, y.reason)) return false;
  }
  return true;
}

std::vector<Instr> mixedStream(unsigned seed) {
  std::vector<std::string> names = {
      "addi", "add", "lui", "csrrw", "fence", "lw", "lhu", "sw", "sb",
      "seli", "seld", "vload", "vstore", "vadd.bf16", "vrelu.bf16",
      "vredsum.row.bf16", "vredmax.bf16", "vpack.bf16.fp8", "vunpack.fp8.bf16",
      "vli.all", "vli.one", "vtrpose.xlu"};
  for (int unit = 0; unit < 2; ++unit)
    for (const char *name : {"vmatpush.weight", "vmatpush.acc.fp8", "vmatpush.acc.bf16",
                            "vmatpop.fp8.acc", "vmatpop.bf16.acc", "vmatmul", "vmatmul.acc"})
      names.push_back(std::string(name) + ".mxu" + std::to_string(unit));
  for (unsigned channel = 0; channel < kAtlasDMAChannels; ++channel)
    for (const char *name : {"dma.config", "dma.load", "dma.store", "dma.wait"})
      names.push_back(std::string(name) + ".ch" + std::to_string(channel));
  std::mt19937 rng(seed);
  std::shuffle(names.begin(), names.end(), rng);
  std::vector<Instr> result{{findOp("dma.wait.ch0")}};
  for (unsigned i = 0; i < 128; ++i) {
    const OpInfo *op = findOp(names[i < names.size() ? i : rng() % names.size()]);
    Instr in{op, int(rng() % 32) * 2, int(rng() % 32) * 2, int(rng() % 32) * 2};
    if (hasOperand(*op, "xd")) in.rd = 4 + in.rd % 28;
    if (hasOperand(*op, "ed")) in.rd %= 32;
    if (hasOperand(*op, "x1") || hasOperand(*op, "e1") || hasOperand(*op, "@")) in.rs1 %= 32;
    if (hasOperand(*op, "x2")) in.rs2 %= 32;
    if (hasOperand(*op, "ad") || hasOperand(*op, "wd")) in.rd %= 2;
    if (hasOperand(*op, "a2") || hasOperand(*op, "w2")) in.rs2 %= 2;
    if (op->opClass == OpClass::VLoad || op->opClass == OpClass::VStore) in.rd = rng() % 64;
    if (op->opClass == OpClass::Alu) in.imm = int(rng() % 5) * 1024;
    // Keep DMA address/size registers intact so known-entry streams include
    // real completion accesses, rather than only refused size footprints.
    if (op->opClass == OpClass::DmaLoad) {
      in.rd = 1; in.rs1 = 2; in.rs2 = 3;
    } else if (op->opClass == OpClass::DmaStore) {
      in.rd = 2; in.rs1 = 1; in.rs2 = 3;
    }
    in.release = op->opClass == OpClass::Csr && (seed + i) % 2 == 0;
    result.push_back(in);
  }
  result.push_back({findOp("ecall")});
  return result;
}
} // namespace

void atlas_test::runTimingGraph(MLIRContext &) {
  for (unsigned seed = 0; seed < 24; ++seed) {
    auto instructions = mixedStream(seed);
    for (bool known : {false, true}) {
      RegValues entry = unknownRegs();
      if (known) {
        // Include disjoint pages that collide modulo the index, page/bank
        // boundaries, and transfers of several different aligned sizes.
        for (unsigned reg = 1; reg < entry.size(); ++reg)
          entry[reg] = 0x2000 + ((reg * 17 + seed) % 512) * 256;
        entry[1] = seed % 4 == 0 ? (63 * 1024 + 512) / 4
                   : seed % 4 == 1 ? (255 * 1024 + 512) / 4 : 0x2000;
        entry[2] = 0x80000000;
        entry[3] = seed % 3 == 0 ? 32 : seed % 3 == 1 ? 1024 : 2048;
      }
      IncomingDma incoming;
      incoming[0].push_back(footprintOf({findOp("dma.load.ch0"), 1, 2, 3}, entry));
      incoming[1].push_back(footprintOf({findOp("dma.store.ch1"), 2, 1, 3}, entry));
      for (const IncomingDma *pending :
           {static_cast<const IncomingDma *>(nullptr),
            static_cast<const IncomingDma *>(&incoming)}) {
        auto expected = bruteForceGraph(instructions, entry, pending);
        auto actual = buildGraph(instructions, entry, pending);
        check(equalGraphs(actual, expected), "indexed graph matches every ordered brute-force edge",
              "seed " + std::to_string(seed) + (known ? ", known addresses" : ", unknown addresses"));
      }
    }
  }
}
