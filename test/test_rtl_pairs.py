"""Pair admission, operand relations and live command capacity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from rtl_test_support import (CONSUMERS, FACTS, HALT, OPT, PAIRS, ROOT, VERIFY, addi, block, busy,
    decode_mxu, export, facts, gapped, issue_cycles, issue_gap, mxu, pair_spec,
    pairs_variant, program, run, selection, separated, spaced, stream, transpose, vls, vpu,)

SETS = [(0, 2, 4), (8, 10, 12), (16, 18, 20), (24, 26, 28)]


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file(), "requires timing facts, pair facts and built tools")
class PairAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = facts()
        cls.document = json.loads(Path(PAIRS).read_text())
        cls.with_pairs, cls.plain = selection(pairs=PAIRS), selection()

    def verify(self, source, option=None):
        return run(OPT, source, option or self.with_pairs, VERIFY)

    def boundary(self, source, gap, option=None, offset=0):
        accepted = self.verify(source(gap), option)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        if gap == 1:
            return ""
        early = self.verify(source(gap - 1), option)
        self.assertNotEqual(early.returncode, 0)
        self.assertRegex(early.stderr, rf"first permitted cycle {gap + offset}\b|requires {gap}, got {gap - 1}\b")
        return early.stderr

    def record(self, document, first, second, **relation):
        return next(r for r in document["op_pairs"] if (r["first"], r["second"]) == (first, second)
                    and r["relation"] == {"mreg_banks": "disjoint", **relation})

    def test_vpu_admission_boundaries(self):
        cases = [("exp", "relu"), ("relu", "exp"), ("rcp", "sqrt"), ("mov", "exp"),
                 ("fp8unpack", "relu"), ("vliAll", "relu"), ("csum", "relu"),
                 ("exp", "exp"), ("exp", "exp2"), ("relu", "add"), ("add", "relu"),
                 ("rsum", "relu"), ("exp", "rsum"), ("vliAll", "vliCol")]
        for first, second in cases:
            with self.subTest(first=first, second=second):
                a, b = vpu(first, SETS[0]), vpu(second, SETS[1])
                gap = self.record(self.document, f"vpu.{first}", f"vpu.{second}")["stable_gap"]
                stderr = self.boundary(lambda g: gapped([], a, b, g), gap)
                if gap > 1:
                    self.assertIn("no pair fact admits", stderr)
                reverse = self.record(self.document, f"vpu.{second}", f"vpu.{first}")["stable_gap"]
                for consumer in CONSUMERS:
                    timed = run(OPT, program([a, b, HALT]), self.with_pairs, consumer, VERIFY)
                    self.assertEqual(timed.returncode, 0, timed.stderr)
                    placed = issue_gap(timed.stdout, a, b)
                    self.assertEqual(abs(placed), gap if placed > 0 else reverse)

    def test_mxu_admission_and_data_boundaries(self):
        for unit in (0, 1):
            cases = [("matmul", "pop_acc_bf16", 0, 0, 64 if unit == 0 else 4), ("matmul", "pop_acc_bf16", 1, 0, 1),
                     ("matmul", "matmul", 1, 0, 32), ("matmul", "matmul_acc", 0, 1, 64 if unit == 0 else 32),
                     ("matmul", "push_weight", 0, 0, 94 if unit == 0 else 34), ("matmul", "push_weight", 0, 1, 1),
                     ("push_weight", "matmul", 0, 0, 33), ("push_weight", "matmul", 0, 1, 1),
                     ("push_acc_fp8", "matmul_acc", 0, 0, 2 if unit == 0 else 33),
                     ("push_acc_fp8", "matmul_acc", 1, 0, 1)]
            for first, second, acc, slot, gap in cases:
                with self.subTest(unit=unit, first=first, second=second, acc=acc, slot=slot):
                    a = mxu(first, unit, reg=12)
                    b = mxu(second, unit, reg=24, acc=acc, slot=slot)
                    self.boundary(lambda g: spaced([a, b], g), gap)
                    hold = busy(block(self.facts, f"mxu{unit}.{first}")) + 1
                    self.boundary(lambda g: spaced([a, b], g), hold, self.plain)

    def test_operand_relations(self):
        a, b = vpu("exp", (8, 10, 16)), vpu("relu", (48, 50, 28))
        gap = block(self.facts, "vpu.exp")["next_issue_age"]
        stderr = self.boundary(lambda g: gapped([], a, b, g), gap)
        self.assertIn("no pair fact admits vrelu.bf16", stderr)
        for unit in (0, 1):
            read_only = [mxu("push_weight", unit, reg=12), mxu("matmul", unit, reg=12, slot=1)]
            self.boundary(lambda g: spaced(read_only, g), 32)
            a = mxu("matmul", unit, reg=12)
            window = busy(block(self.facts, f"mxu{unit}.matmul")) + 1
            for dst, gap in ((12, 33), (44, 1)):
                with self.subTest(unit=unit, dst=dst):
                    b = mxu("pop_acc_bf16", unit, reg=dst, acc=1)
                    self.boundary(lambda g: spaced([a, b], g), gap)
                    with tempfile.TemporaryDirectory() as directory:
                        path = pairs_variant(directory, lambda d: d.update(op_pairs=[
                            r for r in d["op_pairs"] if r["relation"]["mreg_banks"] == "disjoint"]))
                        stderr = self.boundary(lambda g: spaced([a, b], g), window, selection(pairs=path))
                    self.assertIn("no pair fact admits", stderr)
                    self.assertIn(f"MXU in-flight matmuls {unit} held by vmatmul.mxu{unit}", stderr)

    def test_unusable_facts_fall_back_to_isolated_holds(self):
        cases = [("vpu.exp", "vpu.relu", {}, vpu("exp", SETS[0]), vpu("relu", SETS[1]),
                  block(self.facts, "vpu.exp")["next_issue_age"],
                  [None, {}, {"mreg_banks": "shared"}, {"mreg_banks": "disjoint", "accumulator": "other"}]),
                 ("mxu0.matmul", "mxu0.pop_acc_bf16", {"accumulator": "other"},
                  mxu("matmul", 0, reg=12), mxu("pop_acc_bf16", 0, reg=24, acc=1),
                  busy(block(self.facts, "mxu0.matmul")) + 1,
                  [None, {"mreg_banks": "disjoint", "accumulator": "different"},
                   {"mreg_banks": "disjoint", "accumulator": "other", "port": "read0"},
                   {"mreg_banks": "disjoint"}, "short sweep"])]
        with tempfile.TemporaryDirectory() as directory:
            for first, second, relation, a, b, gap, mutations in cases:
                for mutation in mutations:
                    with self.subTest(first=first, mutation=mutation):
                        def mutate(document):
                            record = self.record(document, first, second, **relation)
                            if mutation is None:
                                document["op_pairs"].remove(record)
                            elif mutation == "short sweep":
                                record["gaps"][1] = gap - 2
                            else:
                                record["relation"] = mutation
                        path = pairs_variant(directory, mutate)
                        option = selection(pairs=path)
                        stderr = self.boundary(lambda g: spaced([a, b], g), gap, option)
                        self.assertIn("no pair fact admits", stderr)
                        if first.startswith("mxu"):
                            same = mxu("pop_acc_bf16", 0, reg=24)
                            self.boundary(lambda g: spaced([a, same], g), 64, option)

    def test_capacity_and_every_live_xlu_hold(self):
        cases = [("vpu", [vpu(n, SETS[i]) for i, n in enumerate(("exp", "relu", "sin"))],
                  1, block(self.facts, "vpu.exp")["next_issue_age"], "VPU slots busy"),
                 ("mxu0", [mxu("matmul", 0, reg=r, acc=a) for r, a in ((12, 0), (16, 1), (4, 0))],
                  32, busy(block(self.facts, "mxu0.matmul")) + 1, "all 2 mxu0 slots busy")]
        for engine, ops, second_cycle, third_cycle, diagnostic in cases:
            with self.subTest(engine=engine):
                prefix = [ops[0], *separated(second_cycle)]
                stderr = self.boundary(lambda g: gapped(prefix, ops[1], ops[2], g), third_cycle - second_cycle, offset=second_cycle)
                self.assertIn(diagnostic, stderr)
                self.assertIn(f"first permitted cycle {third_cycle}", stderr)
                for consumer in CONSUMERS:
                    timed = run(OPT, program([*ops, HALT]), self.with_pairs, consumer, VERIFY)
                    self.assertEqual(timed.returncode, 0, timed.stderr)
                    cycles = [c for op, _, c in issue_cycles(timed.stdout) if op == ops[0][0]]
                    self.assertEqual(cycles, [0, second_cycle, third_cycle])
                if engine == "mxu0":
                    with tempfile.TemporaryDirectory() as directory:
                        path = pairs_variant(directory, lambda d: next(
                            c for c in d["capacity"] if c["engine"] == engine).update(live=3))
                        self.assertEqual(self.verify(spaced(ops, 32), selection(pairs=path)).returncode, 0)
        exp, relu = vpu("exp", SETS[0]), vpu("relu", SETS[1])
        gap = block(self.facts, "vpu.relu")["first_free_age"]
        self.assertIn("busy", self.boundary(lambda g: gapped([exp], relu, transpose(40, 42), g), gap, offset=1))

    def test_chain_sweeps(self):
        for engine in ("vpu", "mxu"):
            raw = pair_spec(ROOT / f"tools/rtl_extract/specs/sequences/core_{engine}.yaml")
            rows = [(name, length, gap) for name, lengths in raw["expect"].items() for length, gap in lengths.items()]
            def scan(row):
                name, length, stable = row
                entries = raw["sequences"][name]["ops"]
                ops = ([vpu(entries[i % len(entries)]["op"], SETS[i % len(SETS)]) for i in range(length)]
                       if engine == "vpu" else [decode_mxu(e["inputs"]["instruction"]) for e in entries[:length]])
                verdicts = []
                for gap in range(1, stable + 4 if engine == "vpu" else 200):
                    verdicts.append(self.verify(spaced(ops, gap)))
                    if engine == "mxu" and gap >= stable and verdicts[-1].returncode == 0 and all(
                            self.verify(spaced(ops, g)).returncode == 0 for g in range(gap + 1, gap + 3)):
                        return row, verdicts
                return row, verdicts
            with ThreadPoolExecutor(8) as pool:
                scanned = list(pool.map(scan, rows))
            matched = 0
            for (name, length, stable), verdicts in scanned:
                with self.subTest(engine=engine, name=name, length=length):
                    accepted = [g for g, result in enumerate(verdicts, 1) if result.returncode == 0]
                    self.assertTrue(accepted)
                    self.assertGreaterEqual(min(accepted), stable)
                    matched += min(accepted) == stable
                    if engine == "vpu":
                        self.assertEqual(min(accepted), stable)
                        self.assertTrue(all(r.returncode == 0 for r in verdicts[stable - 1:]))
                    elif min(accepted) > stable:
                        self.assertRegex(verdicts[stable - 1].stderr, r"mxu\d slots busy|requires \d+, got \d+")
            if engine == "mxu":
                self.assertGreaterEqual(matched, 109)

    def test_without_applicable_pairs_preserves_schedules(self):
        ops = [vpu(n, SETS[i]) for i, n in enumerate(("exp", "relu", "exp"))]
        for consumer in CONSUMERS:
            timed = run(OPT, program([*ops, HALT]), self.plain, consumer, VERIFY)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            holds = [block(self.facts, f"vpu.{n}")["next_issue_age"] for n in ("exp", "relu")]
            self.assertEqual([c for op, _, c in issue_cycles(timed.stdout) if op == "vpu_unary"],
                             [0, holds[0], sum(holds)])
        with tempfile.TemporaryDirectory() as directory:
            path = pairs_variant(directory, lambda d: d.update(
                op_pairs=[r for r in d["op_pairs"] if r["engine"] not in ("mxu0", "mxu1")],
                capacity=[c for c in d["capacity"] if c["engine"] not in ("mxu0", "mxu1")]))
            source = program([addi(6, 0, 256), vls("vload", 0), vpu("add", SETS[0]), vls("vstore", 4, offset=32), HALT])
            cases = [("vls_add", source, self.with_pairs)]
            cases += [(f"tile{u}", program([mxu("push_weight", u), mxu("matmul", u),
                                          mxu("pop_acc_bf16", u), HALT]), selection(pairs=path)) for u in (0, 1)]
            for name, source, option in cases:
                for consumer in CONSUMERS:
                    with self.subTest(name=name, consumer=consumer):
                        timed = [run(OPT, source, o, consumer, VERIFY) for o in (self.plain, option)]
                        for result in timed:
                            self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(stream(timed[0].stdout), stream(timed[1].stdout))

    def test_export_admission(self):
        source = program([mxu("matmul", 0, reg=12), mxu("pop_acc_bf16", 0, reg=24), HALT])
        for option, units in ((self.with_pairs, ["mxu0", "mxu1"]), (self.plain, None)):
            timed = run(OPT, source, option, CONSUMERS[0], VERIFY)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            applicability = export(timed.stdout)["applicability"]
            self.assertEqual(applicability["admission"].get("mxu_pipelined_units"), units)
            self.assertEqual(applicability["admission"]["mxu_commands_sequenced_per_unit"], units is None)
            self.assertEqual("same_mxu_pipelining" in applicability["unsupported"], units is None)
            if units:
                self.assertEqual(applicability["pair_facts"]["live_window"],
                                 {"mxu0": "last_event_age", "mxu1": "last_event_age", "vpu": "next_issue_age"})


if __name__ == "__main__":
    unittest.main()
