#!/usr/bin/env python3
"""Generate lib/AtlasIssueRulesTable.inc from npu_model's issue-rules profiles.
Rebuild every instruction variant and require exact item and reservation equality.
The rebuild mirrors footprintOf and reservation rules in lib/AtlasTiming.cpp; change both together when rules change.
Write: python3 tools/atlas_issue_rules_table.py --npu-model CHECKOUT --revision COMMIT
Check: python3 tools/atlas_issue_rules_table.py --npu-model CHECKOUT --check
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import re
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "lib/AtlasIssueRulesTable.inc"
SOURCE = "npu_model/scheduling/issue_rules.py"
ROWS, SRC, SRC2, DST, RD, RS, RS2 = 32, 2, 4, 6, 5, 6, 7


def fail(message: str):
    raise SystemExit(f"issue-rules table: {message}")


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load(root: Path):
    path = root / SOURCE
    sys.path.insert(0, str(root))
    from npu_model.scheduling import issue_rules
    expect(Path(issue_rules.__file__).resolve() == path.resolve(),
           f"selected issue-rules source {path} differs from imported module: {issue_rules.__file__}")
    return issue_rules, hashlib.sha256(path.read_bytes()).hexdigest()


class Facts:
    def __init__(self, ir, op):
        self.p = ir.profile(op)

    def _one(self, items, resource):
        found = [i for i in items if i.resource == tuple(resource)]
        expect(len(found) == 1, f"{self.p.name}: expected one item on {resource}, found {len(found)}")
        return found[0]

    def hold(self, *resource):
        h = self._one(self.p.holds, resource)
        return (h.first, h.last)

    def gate(self, *resource):
        g = self._one(self.p.gates, resource)
        return (g.first, g.last)

    def probe(self, *resource):
        return self._one(self.p.probes, resource).at

    def counted(self, *resource):
        c = self._one(self.p.counted, resource)
        return (c.first, c.last, c.capacity)

    def first(self, resource, write):
        ages = [r.at for r in self.p.rows if r.resource == tuple(resource) and r.write == write]
        expect(bool(ages), f"{self.p.name}: no {'write' if write else 'read'} of {resource}")
        return min(ages)

    def step(self, reg, write):
        port = [h for h in self.p.holds if h.resource == ("mrf_port", reg % 32, "w" if write else "r")]
        expect(len(port) >= 1, f"{self.p.name}: no port hold for m{reg}")
        return (port[0].last - port[0].first) // (ROWS - 1)


def derive(ir) -> dict:
    Op, t = ir.Op, {}
    f = Facts(ir, Op("scalar", rd=RD, rs=(RS,)))
    t["scalarSourceReadAge"] = f.first(("xrf", RS), False)
    expect(t["scalarSourceReadAge"] == 0,
           "the compiler reads scalar sources at issue (lib/AtlasTiming.cpp scalarRegs, VLOAD/VSTORE, and DMA)")
    at = f.probe("scalar_writeback")
    t["kScalarTiming"] = (f.first(("xrf", RD), True) + 1, (at, at))
    f = Facts(ir, Op("sload", rd=RD, rs=(RS,), vmem_addr=0))
    t["kScalarLsuTiming"] = (t["scalarSourceReadAge"], f.first(("vmem",), False), f.hold("vmem_bank", 0),
                             f.first(("xrf", RD), True) + 1, f.hold("lsu_scalar"), f.gate("scalar_writeback"))
    f = Facts(ir, Op("vload", mreg=SRC, rs=(RS,), vmem_addr=0))
    t["kVloadTiming"] = (f.first(("vmem",), False), f.first(("mrf", SRC), True), f.p.writes_free,
                        f.hold("vload_path"), f.hold("vmem_bank", 0))
    f = Facts(ir, Op("vstore", mreg=SRC, rs=(RS,), vmem_addr=0))
    t["kVstoreTiming"] = (f.first(("mrf", SRC), False), f.first(("vmem",), True), f.p.reads_free - 1,
                         f.hold("vstore_path"), f.hold("vmem_bank", 0))
    t["kMxuTiming"] = []
    for u in (0, 1):
        mm = Facts(ir, Op("matmul", u, mreg=SRC, acc=0, wslot=0))
        pop = Facts(ir, Op("pop_bf16", u, mreg=SRC, acc=0))
        push = Facts(ir, Op("weight", u, mreg=SRC, wslot=0))
        drain = mm.gate("wslot_drain", 0, 0) if u == 0 else mm.gate("wslot_gate", 1, 0)
        t["kMxuTiming"].append(((mm.gate("acc_row0_gate", u, 0)[1] + 1, pop.p.writes_free),
            mm.first(("wslot", u, 0), False), mm.p.reads_free - 1, mm.counted("inflight", u)[2], drain[1] + 1,
            (mm.hold("mxu_port", u, "P0"), pop.hold("mxu_port", u, "W0"), push.hold("wpath", u),
             pop.hold("acc_rport", u, 0))))
    t["kMxuSourceReadAge"] = Facts(ir, Op("matmul", 0, mreg=SRC, acc=0, wslot=0)).first(("mrf", SRC), False)
    t["kMxuPushResultAge"] = Facts(ir, Op("weight", 0, mreg=SRC, wslot=0)).first(("wslot", 0, 0), True)
    t["kMxuPopResultAge"] = Facts(ir, Op("pop_bf16", 0, mreg=SRC, acc=0)).first(("mrf", SRC), True)
    t["kMxu1WeightUseAfterPush"] = Facts(ir, Op("weight", 1, mreg=SRC, wslot=0)).gate("wslot_gate", 1, 0)[1] + 1
    t["kMxu1AccumulatorUseAfterPush"] = Facts(ir, Op("push_fp8", 1, mreg=SRC, acc=0)).gate("acc_pushgate", 1, 0)[1] + 1
    sets = [ir.VPU_TWO_INPUT, ir.VPU_SINGLE, ir.VPU_ROW_REDUCE, ir.VPU_COL_REDUCE, ir.VPU_VLI]
    single = ir.VPU_SINGLE - {"vpack.bf16.fp8", "vunpack.fp8.bf16"}
    expect(sum(map(len, sets)) == len(set().union(*sets)) and set().union(*sets) == ir.VPU_OPS
           and ir.VPU_SINGLE - single == {"vpack.bf16.fp8", "vunpack.fp8.bf16"},
           "VPU mnemonic classes overlap or pack and unpack moved")
    vli_pair = sorted(n for n in ir.VPU_VLI if len(ir.profile(Op("vpu", name=n, mdst=DST)).writes) == 2)
    classes = {"Elementwise": sorted(ir.VPU_TWO_INPUT | single),
               "Pack": ["vpack.bf16.fp8"], "Unpack": ["vunpack.fp8.bf16"],
               "RowSum": ["vredsum.row.bf16"], "RowMinMax": sorted(ir.VPU_ROW_REDUCE - {"vredsum.row.bf16"}),
               "ColReduce": sorted(ir.VPU_COL_REDUCE), "LoadImmPair": vli_pair,
               "LoadImmSingle": sorted(ir.VPU_VLI - set(vli_pair))}
    for cls, names in classes.items():
        f = Facts(ir, vpu_op(ir, names[0]))
        p = f.p
        reads = (f.first(("mrf", SRC), False) if SRC in p.reads else 0,
                 f.first(("mrf", SRC + 1), False) if SRC + 1 in p.reads else 0, 1)
        writes = (f.first(("mrf", DST), True), f.first(("mrf", DST + 1), True) if DST + 1 in p.writes else 0,
                  f.step(DST, True))
        t[f"kVpu{cls}Timing"] = (reads, writes, p.reads_free - 1 if p.reads else 0, p.writes_free)
        if cls == "ColReduce":
            second = min(r.at for r in p.rows if r.resource == ("mrf", SRC) and r.at > reads[0] + ROWS - 1)
            t["kVpuColReducePassInterval"] = second - reads[0]
    t["kVpuIssueSlots"] = Facts(ir, vpu_op(ir, "vmov")).counted("vpu_slot")[2]
    for cls, names in [("TwoInput", ir.VPU_TWO_INPUT), ("SingleInput", single), ("RowReduce", ir.VPU_ROW_REDUCE),
                       ("ColReduce", classes["ColReduce"]), ("LoadImmPair", classes["LoadImmPair"]),
                       ("LoadImmSingle", classes["LoadImmSingle"])]:
        t[f"kVpu{cls}Ops"] = sorted(names)
    groups = [(n, g.resource[1]) for n in sorted(ir.VPU_OPS) for g in ir.profile(vpu_op(ir, n)).gates
              if g.resource[0] == "vpu_group"]
    expect(len({n for n, _ in groups}) == len(groups), "a VPU op is in two lane groups; vpuLogicGroup returns one")
    t["kVpuLogicGroups"] = sorted(groups, key=lambda ng: (ng[1], ng[0]))
    f = Facts(ir, Op("xlu", mreg=SRC, mdst=DST))
    t["kXluTiming"] = (f.first(("mrf", SRC), False), f.first(("mrf", DST), True), f.p.reads_free - 1,
                      f.p.writes_free, f.hold("xlu"))
    t["vpuClass"] = {n: cls for cls, names in classes.items() for n in names}
    return t


def vpu_op(ir, name, src=SRC, src2=SRC2, dst=DST):
    return ir.Op("vpu", name=name, mreg=None if name in ir.VPU_VLI else src,
                 mreg2=src2 if name in ir.VPU_TWO_INPUT else None, mdst=dst)


class Rebuilt:
    def __init__(self, ir):
        self.ir, self.items = ir, []
        self.reads, self.writes, self.reads_free, self.writes_free, self.ignores = set(), set(), 0, 0, False

    def add(self, *items):
        self.items += items

    def mrf(self, reg, first, write, step=1, shared=False, port=True):
        ir = self.ir
        if port:
            self.add(ir.Hold(("mrf_port", reg % 32, "w" if write else "r"), first, first + step * (ROWS - 1),
                             (reg, first) if shared else None))
        self.add(*[ir.RowAccess(("mrf", reg), r, first + step * r, write) for r in range(ROWS)])
        self.add(*[ir.RowAccess(("mrf_phys", reg % 32), reg // 32 * ROWS + r, first + step * r, write)
                   for r in range(ROWS)])
        (self.writes if write else self.reads).add(reg)

    def xrf(self, regs, age, write=False):
        self.add(*[self.ir.RowAccess(("xrf", r), 0, age, write) for r in regs if r])

    def vmem(self, addr, first, count, write, bank):
        ir = self.ir
        self.add(ir.Hold(("vmem_bank", addr // ir.VMEM_BANK_BYTES), *bank))
        self.add(*[ir.RowAccess(("vmem",), addr // ir.LINE_BYTES + i, first + i, write) for i in range(count)])


def rebuild(ir, t, op) -> Rebuilt:
    Hold, Row, Gate, Probe, Counted = ir.Hold, ir.RowAccess, ir.Gate, ir.Probe, ir.Counted
    b, k = Rebuilt(ir), op.kind
    if k in ("scalar", "seli", "branch", "sload", "sstore", "vload", "vstore", "dma"):
        b.xrf(op.rs, t["scalarSourceReadAge"])
    if k in ("scalar", "seli", "branch"):
        if k == "seli" or op.rd:
            b.add(Probe(("scalar_writeback",), t["kScalarTiming"][1][0]))
        if op.rd:
            b.xrf([op.rd], t["kScalarTiming"][0] - 1, True)
        if k == "branch":
            # The encoder rejects branches and jumps in a branch delay slot.
            b.add(Gate(("delay_slot",), 1, 1), Probe(("delay_slot",), 0))
    elif k in ("sload", "sstore"):
        b.vmem(op.vmem_addr, t["kScalarLsuTiming"][1], 1, k == "sstore", t["kScalarLsuTiming"][2])
        if k == "sload":
            b.add(Hold(("lsu_scalar",), *t["kScalarLsuTiming"][4]),
                  Gate(("scalar_writeback",), *t["kScalarLsuTiming"][5]))
            if op.rd:
                b.xrf([op.rd], t["kScalarLsuTiming"][3] - 1, True)
    elif k == "vload":
        b.mrf(op.mreg, t["kVloadTiming"][1], True)
        b.add(Hold(("vload_path",), *t["kVloadTiming"][3]))
        b.vmem(op.vmem_addr, t["kVloadTiming"][0], ROWS, False, t["kVloadTiming"][4])
        b.writes_free, b.ignores = t["kVloadTiming"][2], True
    elif k == "vstore":
        b.mrf(op.mreg, t["kVstoreTiming"][0], False)
        b.add(Hold(("vstore_path",), *t["kVstoreTiming"][3]))
        b.vmem(op.vmem_addr, t["kVstoreTiming"][1], ROWS, True, t["kVstoreTiming"][4])
        b.reads_free = t["kVstoreTiming"][2] + 1
    elif op.mxu is not None:
        u, acc, slot = op.mxu, ("acc", op.mxu, op.acc), op.wslot
        (row0, popped), first, release, capacity, overwrite, windows = t["kMxuTiming"][u]
        source_ports, result_ports, push_stream, accumulator_read = windows
        src_age, push_age = t["kMxuSourceReadAge"], t["kMxuPushResultAge"]
        if k.startswith("pop"):
            b.add(Hold(("acc_rport", u, op.acc), *accumulator_read))
            b.add(*[Row(acc, r, src_age + r, False) for r in range(ROWS)])
            for i, reg in enumerate([op.mreg, op.mreg + 1] if k == "pop_bf16" else [op.mreg]):
                b.add(Hold(("mxu_port", u, f"W{i}"), *result_ports))
                b.mrf(reg, t["kMxuPopResultAge"], True)
            b.writes_free = popped
            b.add(Probe(("acc_row0_gate", u, op.acc)))
            return b
        ports = ["P0"] if k.startswith("matmul") else ["P0", "P1"] if k == "push_bf16" else \
            ["P1" if op.push_port in (None, 1) else "P0"]
        for i, port in enumerate(ports):
            b.add(Hold(("mxu_port", u, port), *source_ports))
            b.mrf(op.mreg + i, src_age, False)
        b.reads_free = release + 1
        if k == "weight":
            b.add(Hold(("wpath", u), *push_stream))
            b.add(*[Row(("wslot", u, slot), c, push_age + c, True) for c in range(ROWS)])
            if u == 0:
                b.add(Probe(("wslot_drain", 0, slot)))
            else:
                b.add(Gate(("wslot_gate", 1, slot), 0, t["kMxu1WeightUseAfterPush"] - 1),
                      Probe(("wslot_gate", 1, slot)))
        elif k.startswith("push"):
            b.add(Hold(("apath", u), *push_stream), Hold(("acc_wport", u, op.acc), *push_stream))
            b.add(*[Row(acc, r, push_age + r, True) for r in range(ROWS)], Probe(("acc_row0_gate", u, op.acc)))
            if u == 1:
                b.add(Gate(("acc_pushgate", 1, op.acc), 0, t["kMxu1AccumulatorUseAfterPush"] - 1))
                if k == "push_fp8":
                    b.add(Probe(("acc_pushgate", 1, op.acc)))
        else:
            b.add(Counted(("inflight", u), 0, row0 + ROWS - 2, capacity))
            b.add(Hold(("acc_wport", u, op.acc), row0 - 1, row0 + ROWS - 2))
            b.add(*[Row(acc, r, row0 - 1 + r, True) for r in range(ROWS)])
            b.add(Gate(("acc_row0_gate", u, op.acc), 0, row0 - 1), Probe(("acc_row0_gate", u, op.acc)))
            if k == "matmul_acc":
                b.add(Hold(("acc_rport", u, op.acc), *accumulator_read))
                b.add(*[Row(acc, r, src_age + r, False) for r in range(ROWS)])
            if u == 0:
                b.add(*[Row(("wslot", 0, slot), c, first + c, False) for c in range(ROWS)])
                b.add(Gate(("wslot_drain", 0, slot), 0, overwrite - 1))
                # The drain gate covers MXU0's second weight read at 63+c.
                b.add(*[Row(("wslot", 0, slot), c, first + c + 2 * (ROWS - 1), False) for c in range(ROWS)])
            else:
                b.add(Gate(("wslot_gate", 1, slot), 0, overwrite - 1), Probe(("wslot_gate", 1, slot)),
                      Probe(("acc_pushgate", 1, op.acc)))
                # MXU1's slot gates cover the full-slot weight reads.
                b.add(*[Row(("wslot", 1, slot), c, at, False) for c in range(ROWS) for at in (first, ROWS)])
    elif k == "vpu":
        rebuild_vpu(ir, t, op, b)
    elif k == "xlu":
        b.mrf(op.mreg, t["kXluTiming"][0], False)
        b.mrf(op.mdst, t["kXluTiming"][1], True)
        b.add(Hold(("xlu",), *t["kXluTiming"][4]))
        b.reads_free, b.writes_free = t["kXluTiming"][2] + 1, t["kXluTiming"][3]
    elif k == "dma":
        # Lowering inserts dma.wait before reusing a DMA channel.
        b.add(Hold(("dma_channel", op.channel), 0, ir.FOREVER))
    elif k not in ("dma_wait", "other"):
        fail(f"no compiler structure for {k}")
    return b


def rebuild_vpu(ir, t, op, b) -> None:
    name = op.name
    cls = t["vpuClass"][name]
    reads, writes, release, latency = t[f"kVpu{cls}Timing"]
    if cls not in ("LoadImmPair", "LoadImmSingle"):
        sources = [op.mreg] + ([op.mreg2] if name in ir.VPU_TWO_INPUT else [])
        for src in sources:
            passes = (0, t["kVpuColReducePassInterval"]) if cls == "ColReduce" else (0,)
            for offset in passes:
                b.mrf(src, offset + reads[0], False, shared=True)
                if cls != "Unpack":
                    b.mrf(src + 1, offset + reads[1], False, shared=True)
        b.reads_free = release + 1
    b.mrf(op.mdst, writes[0], True, writes[2], port=writes[2] == 1)
    if writes[2] != 1:
        expect(cls == "Pack", f"{name}: only vpack may write with a stride")
        expect(latency - 1 + writes[0] > writes[0] + writes[2] * (ROWS - 1), "a second vpack could write in the holes")
        # A strided pack holds its write port; no complete writer fits in the holes.
        b.add(ir.Hold(("mrf_port", op.mdst % 32, "w"), writes[0], writes[0] + writes[2] * (ROWS - 1)))
    if cls not in ("Pack", "LoadImmSingle"):
        b.mrf(op.mdst + 1, writes[1], True)
    b.writes_free = latency
    live = latency - 2
    double = name in ir.VPU_TWO_INPUT or cls in ("RowSum", "RowMinMax")
    b.add(ir.Counted(("vpu_slot",), 0, live, t["kVpuIssueSlots"]), ir.Gate(("vpu_any",), 0, live),
          ir.Probe(("vpu_double",)), ir.Gate(("vpu_op", name), 0, live), ir.Probe(("vpu_op", name)))
    if double:
        b.add(ir.Gate(("vpu_double",), 0, live), ir.Probe(("vpu_any",)))
    for n, g in t["kVpuLogicGroups"]:
        if n == name:
            b.add(ir.Gate(("vpu_group", g), 0, live), ir.Probe(("vpu_group", g)))


def variants(ir):
    Op = ir.Op
    yield from (Op("scalar", rd=RD, rs=(RS, RS2)), Op("scalar", rs=(RS,)), Op("scalar"), Op("seli"),
                Op("branch", rs=(RS, RS2)), Op("branch", rd=RD, rs=(RS,)))
    for addr in (0, 3 * ir.VMEM_BANK_BYTES + 5 * ir.LINE_BYTES):
        yield from (Op("sload", rd=RD, rs=(RS,), vmem_addr=addr), Op("sload", rs=(RS,), vmem_addr=addr))
        yield Op("sstore", rs=(RS, RS2), vmem_addr=addr % ir.VMEM_BANK_BYTES + ir.VMEM_BANK_BYTES)
        for reg in (SRC, 40):
            yield from (Op("vload", mreg=reg, rs=(RS,), vmem_addr=addr),
                        Op("vstore", mreg=reg, rs=(RS,), vmem_addr=addr))
    for u in (0, 1):
        for kind in ir.MXU_KINDS:
            for reg, acc, slot in ((SRC, 0, 0), (40, 1, 1), (SRC, 0, 1), (SRC, 1, 0)):
                yield Op(kind, u, mreg=reg, acc=acc, wslot=slot)
                if kind in ("weight", "push_fp8"):
                    yield Op(kind, u, mreg=reg, acc=acc, wslot=slot, push_port=0)
    for name in sorted(ir.VPU_OPS):
        yield from (vpu_op(ir, name), vpu_op(ir, name, 34, 44, 12), vpu_op(ir, name, 34, 44, 38))
    yield from (Op("xlu", mreg=SRC, mdst=DST), Op("xlu", mreg=40, mdst=33))
    yield from (Op("dma", channel=0, rs=(RD, RS, RS2)), Op("dma", channel=7, rs=(RS,)),
                Op("dma_wait", channel=0), Op("other"))


def verify(ir, t) -> None:
    alu, load = t["kScalarTiming"][1][0], t["kScalarLsuTiming"][5]
    expect(alu <= load[0] and load[1] - load[0] < t["kScalarLsuTiming"][4][1] + 1,
           "the compiler holds the scalar write port both ways; issue-rules' one-way gate is equal only when an ALU "
           "uses the port no later than a load does, and loads cannot overlap on it")
    expect(all(m[3] >= 1 for m in t["kMxuTiming"]) and t["kVpuIssueSlots"] >= 1, "capacities must be positive")
    expect(all(m[5][0][0] == 0 for m in t["kMxuTiming"]), "issue-rules picks a push's read port at issue and the "
           "compiler over the whole window; they agree only when source ports are taken from issue")
    for op in variants(ir):
        p, b = ir.profile(op), rebuild(ir, t, op)
        want = collections.Counter(b.items)
        got = collections.Counter(p.holds + p.rows + p.gates + p.probes + p.counted)
        extra, missing = got - want, want - got
        expect(not extra and not missing, f"{p.name} {op}: issue-rules has {list(extra)[:3]}, "
                                          f"the compiler structure gives {list(missing)[:3]}")
        expect((p.reads, p.writes, p.reads_free, p.writes_free, p.writer_ignores_readers)
               == (frozenset(b.reads), frozenset(b.writes), b.reads_free, b.writes_free, b.ignores),
               f"{p.name}: register reservations differ from the compiler's")
    for u in (0, 1):
        reads = [r for r in ir.profile(ir.Op("matmul", u, mreg=SRC, acc=0, wslot=0)).rows if r.resource[0] == "wslot"]
        written = {r.row: r.at for r in ir.profile(ir.Op("weight", u, mreg=SRC, wslot=0)).rows
                   if r.resource[0] == "wslot"}
        war = max(r.at - written[r.row] for r in reads)
        raw = max(written[r.row] - r.at + 1 for r in reads)
        modeled_raw = max(written[c] - (t["kMxuTiming"][u][1] + c) + 1 for c in written)
        expect(war <= t["kMxuTiming"][u][4], f"mxu{u}: weight drain gate does not cover read {war}")
        expect(raw <= (t["kMxu1WeightUseAfterPush"] if u == 1 else modeled_raw), f"mxu{u}: push-to-matmul gap {raw}")
    port = t["kMxuTiming"][1][5][0]
    expect(t["kMxuTiming"][1][4] <= port[1] - port[0] + 1, "mxu1: slot gate outlasts the source port")


def c(v):
    if isinstance(v, (tuple, list)):
        return "{" + ", ".join(map(c, v)) + "}"
    return f'"{v}"' if isinstance(v, str) else str(v)


def render(ir, t, digest, revision):
    lines = ["// Generated by tools/atlas_issue_rules_table.py; do not edit.",
             f"// Source: npu_model revision {revision}.",
             f"// {SOURCE} sha256 {digest}.",
             "// Regenerate: python3 tools/atlas_issue_rules_table.py --npu-model <checkout> --revision <commit>",
             f'static_assert(kLineBytes == {ir.LINE_BYTES} && kVmemBankBytes == {ir.VMEM_BANK_BYTES}, '
             '"issue-rules VMEM geometry changed");',
    ]
    for cls in ("Scalar", "ScalarLsu", "Vload", "Vstore"):
        lines.append(f"constexpr {cls}Timing k{cls}Timing = {c(t[f'k{cls}Timing'])};")
    lines += ["constexpr MxuTiming kMxuTiming[] = {",
              "    // {accumulatorRow0, poppedMreg}, weightReadAge, sourceReadRelease, computeCapacity, "
              "weightOverwriteAfterCompute,",
              "    // {sourcePorts, resultPorts, pushStream, accumulatorRead}"]
    for u in (0, 1):
        lines.append(f"    {c(t['kMxuTiming'][u])}, {' ' if u else ''}// MXU{u}")
    lines += ["};", "constexpr int " + ", ".join(f"{n} = {t[n]}" for n in
              ("kMxuSourceReadAge", "kMxuPushResultAge", "kMxuPopResultAge")) + ";",
              "// MXU1's sequencer also blocks a weight or accumulator slot while a push into it is still writing.",
              "constexpr int " + ", ".join(f"{n} = {t[n]}" for n in
              ("kMxu1WeightUseAfterPush", "kMxu1AccumulatorUseAfterPush")) + ";"]
    for cls in ("Elementwise", "Pack", "Unpack", "RowSum", "RowMinMax", "ColReduce", "LoadImmPair", "LoadImmSingle"):
        lines.append(f"constexpr VpuTiming kVpu{cls}Timing = {c(t[f'kVpu{cls}Timing'])};")
    lines += [f'constexpr int kVpuColReducePassInterval = {t["kVpuColReducePassInterval"]};',
              f'constexpr unsigned kVpuIssueSlots = {t["kVpuIssueSlots"]};']
    for declaration, values in [
            (f"const char *kVpu{cls}Ops", t[f"kVpu{cls}Ops"])
            for cls in ("TwoInput", "SingleInput", "RowReduce", "ColReduce", "LoadImmPair", "LoadImmSingle")
    ] + [("std::pair<const char *, int> kVpuLogicGroups", t["kVpuLogicGroups"])]:
        expect(values, f"{declaration}: the compiler requires a nonempty array")
        lines.append(f"constexpr {declaration}[] = {{")
        for i, value in enumerate(values):
            item = c(value) + ("," if i + 1 < len(values) else "};")
            space = "" if lines[-1].endswith("{") else " "
            if len(lines[-1] + space + item) > 120:
                lines.append("    " + item)
            else:
                lines[-1] += space + item
    lines.append(f"constexpr XluTiming kXluTiming = {c(t['kXluTiming'])};")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npu-model", type=Path, required=True)
    parser.add_argument("--revision", help="source revision to record; otherwise reuse one with the same digest")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if the generated include is stale")
    args = parser.parse_args()
    ir, digest = load(args.npu_model)
    t = derive(ir)
    verify(ir, t)
    old = args.output.read_text() if args.output.exists() else ""
    recorded = re.search(r"revision (\S+)\.\n// \S+ sha256 (\S+)\.", old)
    revision = args.revision or (recorded.group(1) if recorded and recorded.group(2) == digest else None)
    if revision is None and not args.check:
        fail("pass --revision to record the source revision")
    if args.check and (revision is None or render(ir, t, digest, revision) != old):
        print(f"{args.output} is stale; rerun {Path(__file__).name}", file=sys.stderr)
        return 1
    if not args.check:
        args.output.write_text(render(ir, t, digest, revision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
