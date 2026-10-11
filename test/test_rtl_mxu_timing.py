"""Selected MXU timing from RTL-computed facts (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING).

Pair results come from tools/rtl_extract/targets/atlas/pairs/mxu{0,1}.yaml (expect), or from
the rtl_extract.op_pairs.v1 files named by ATLAS_OP_PAIRS (os.pathsep-separated) when set.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import yaml

from test_delay_insertion import OPT, EMIT, ROOT, addi, nop, program, run, without_delays
from test_rtl_timing import (CONSUMERS, FACTS, HALT, MARKER, VERIFY, block, delay, export, facts,
                             facts_variant, selected, vls)

PAIR_SPECS = {u: yaml.safe_load((ROOT / f"tools/rtl_extract/targets/atlas/pairs/mxu{u}.yaml").read_text())
              for u in (0, 1)}
OPERATIONS = ("push_weight", "push_acc_fp8", "push_acc_bf16", "pop_acc_fp8", "pop_acc_bf16", "matmul", "matmul_acc")
# Event group -> (export resource, write, operand); the compiler supplies this structure.
STRUCTURE = {
    "push_weight": {"mreg_read1": ("mreg", False, "reg"), "weight_write": ("weight", True, "slot")},
    "push_acc_fp8": {"mreg_read1": ("mreg", False, "reg"), "acc_load": ("acc", True, "acc")},
    "push_acc_bf16": {"mreg_read0": ("mreg", False, "reg"), "mreg_read1": ("mreg", False, "reg+1"),
                      "acc_load": ("acc", True, "acc")},
    "pop_acc_fp8": {"acc_store": ("acc", False, "acc"), "mreg_write0": ("mreg", True, "reg")},
    "pop_acc_bf16": {"acc_store": ("acc", False, "acc"), "mreg_write0": ("mreg", True, "reg"),
                     "mreg_write1": ("mreg", True, "reg+1")},
    "matmul": {"mreg_read0": ("mreg", False, "reg"), "acc_write": ("acc", True, "acc")},
    "matmul_acc": {"mreg_read0": ("mreg", False, "reg"), "acc_read": ("acc", False, "acc"),
                   "acc_write": ("acc", True, "acc")},
}
EXCLUDED = {("MXU in-flight matmuls", 0), ("MXU in-flight matmuls", 1), ("VPU", 0), ("XLU", 0),
            ("VLOAD path", 0), ("VSTORE path", 0)}


def mxu(operation, unit, reg=2, acc=0, slot=0, scale=0):
    if operation.startswith("push_"):
        kind = {"push_weight": "weight_fp8", "push_acc_fp8": "acc_fp8", "push_acc_bf16": "acc_bf16"}[operation]
        target = slot if operation == "push_weight" else acc
        return ("mxu_push", f'kind = "{kind}", unit = {unit} : i32, src = {reg} : i32, slot = {target} : i32')
    if operation.startswith("pop_"):
        fp8 = operation == "pop_acc_fp8"
        return ("mxu_pop", f'format = "{"fp8" if fp8 else "bf16"}", unit = {unit} : i32, dst = {reg} : i32, '
                           f'slot = {acc} : i32, scale_reg = {scale if fp8 else 0} : i32')
    return ("mxu_matmul", f"unit = {unit} : i32, src = {reg} : i32, weight_slot = {slot} : i32, "
                          f"acc_slot = {acc} : i32, accumulate = {str(operation == 'matmul_acc').lower()}")


def busy(record):
    """Last age of any event, MREG reads retained through their response."""
    latency = record["assumptions"]["scratchpad_read_latency"]
    return max(g["last_age"] + (latency if k.startswith("mreg_read") else 0)
               for k, g in record["events"].items() if g["count"])


def spacer(gap):
    """Ops between two issues `gap` cycles apart (DELAY N occupies N+1 cycles)."""
    return [] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]


def pair_results():
    """(unit, name, first, second, second operands, stable_gap) for every MXU pair."""
    stable = {}
    for path in filter(None, os.environ.get("ATLAS_OP_PAIRS", "").split(os.pathsep)):
        for record in json.loads(Path(path).read_text())["op_pairs"]:
            stable[record["name"]] = record["stable_gap"]
    rows = []
    for unit, spec in PAIR_SPECS.items():
        for name, pair in spec["pairs"].items():
            inputs = pair.get("second_inputs", {})
            operands = {"reg": inputs.get("io_cmd_bits_mregId", 2), "acc": inputs.get("io_cmd_bits_accSel", 0),
                        "slot": inputs.get("io_cmd_bits_weightSlot", 0)}
            gap = stable.get(f"mxu{unit}.{name}", spec["expect"][name]["stable_gap"])
            rows.append((unit, name, pair["first"], pair["second"], operands, gap))
    return rows


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedMXUTimingTest(unittest.TestCase):
    def check(self, ops, accepted, *passes, diagnostic=None, **options):
        result = selected(ops, *passes, **options)
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)
        return result

    def gap_accepted(self, first, second, gap, drain, **options):
        return selected([first, *spacer(gap), second, delay(drain), nop(), HALT], VERIFY, **options).returncode == 0

    def assert_minimum_gap(self, first, second, gap, drain, **options):
        self.assertTrue(self.gap_accepted(first, second, gap, drain, **options), f"gap {gap} rejected")
        self.assertFalse(self.gap_accepted(first, second, gap - 1, drain, **options), f"gap {gap - 1} accepted")

    def test_footprints_come_from_facts(self):
        document = facts()
        operands = {"reg": 6, "acc": 1, "slot": 1}
        for unit in (0, 1):
            for operation in OPERATIONS:
                record = block(document, f"mxu{unit}.{operation}")
                end = busy(record)
                expected = set()
                for group, (resource, write, operand) in STRUCTURE[operation].items():
                    event = record["events"][group]
                    name, _, offset = operand.partition("+")
                    value = operands[name] + int(offset or 0)
                    first = value * 32 if resource == "mreg" else (unit * 2 + value) * 32
                    expected.add((resource, write, first, event["count"], event["first_age"], event["step"]))
                if operation.startswith("matmul"):
                    last = record["events"]["acc_write"]["last_age"]
                    expected |= {("weight", False, (unit * 2 + 1) * 32, 32, age, 0) for age in (0, last)}
                if operation == "pop_acc_fp8":
                    expected.add(("ereg", False, 3, 1, 0, 1))
                for consumer in CONSUMERS:
                    with self.subTest(unit=unit, operation=operation, consumer=consumer):
                        result = self.check([mxu(operation, unit, scale=3, **operands), HALT], True, consumer, VERIFY)
                        footprint = next(x for x in export(result.stdout)["instructions"]
                                         if x["mnemonic"].endswith(f".mxu{unit}"))["footprint"]
                        self.assertEqual({(a["resource"], a["write"], a["first"], a["count"], a["age"], a["step"])
                                          for a in footprint["accesses"]}, expected)
                        self.assertEqual(footprint["done_age"], end)
                        self.assertEqual({(h["unit"], h["index"]) for h in footprint["holds"]}, EXCLUDED)
                        self.assertTrue(all(h["from"] == 0 and h["to"] == end for h in footprint["holds"]))
                        self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_pair_results_bound_the_compiler_gap(self):
        document = facts()
        rows = pair_results()
        self.assertEqual(len(rows), 12)
        for unit, name, first, second, operands, stable in rows:
            with self.subTest(pair=f"mxu{unit}.{name}"):
                gap = busy(block(document, f"mxu{unit}.{first}")) + 1
                self.assertGreaterEqual(gap, stable)
                a, b = mxu(first, unit), mxu(second, unit, **operands)
                self.assert_minimum_gap(a, b, gap, busy(block(document, f"mxu{unit}.{second}")))
                scheduled = export(self.check([a, b, HALT], True, CONSUMERS[0], VERIFY).stdout)
                issues = [x["logical_issue_cycle"] for x in scheduled["instructions"] if "mxu" in x["mnemonic"]]
                self.assertEqual(issues[1] - issues[0], gap)
                self.check([a, b, HALT], True, CONSUMERS[1], VERIFY)

    def hold_end(self, op, unit):
        result = self.check([addi(6, 0, 256), op, HALT], True, CONSUMERS[0], VERIFY)
        footprint = export(result.stdout)["instructions"][1]["footprint"]
        return max(h["to"] for h in footprint["holds"] if h["unit"] == unit)

    def test_mxus_and_vector_engines_are_mutually_excluded(self):
        document = facts()
        matmul0, matmul1 = busy(block(document, "mxu0.matmul")), busy(block(document, "mxu1.matmul"))
        add = ("vpu_binary", 'kind = "add", dst = 40 : i32, lhs = 20 : i32, rhs = 50 : i32')
        cases = [
            (mxu("matmul", 0), mxu("push_weight", 1, reg=22), matmul0 + 1),
            (mxu("matmul", 1), mxu("matmul", 0, reg=22), matmul1 + 1),
            (mxu("matmul", 1), ("xlu_transpose", "dst = 40 : i32, src = 22 : i32"), matmul1 + 1),
            (mxu("matmul", 1), add, matmul1 + 1),
            (mxu("pop_acc_bf16", 0), vls("vstore", 22), busy(block(document, "mxu0.pop_acc_bf16")) + 1),
            (vls("vload", 22), mxu("push_weight", 1), self.hold_end(vls("vload", 22), "VLOAD path") + 1),
            (vls("vstore", 22), mxu("pop_acc_bf16", 1), self.hold_end(vls("vstore", 22), "VSTORE path") + 1),
            (add, mxu("matmul", 0), self.hold_end(add, "VPU") + 1),
        ]
        for first, second, gap in cases:
            with self.subTest(first=first[1], second=second[1]):
                ops = lambda g: [addi(6, 0, 256), first, *spacer(g), second, delay(130), nop(), HALT]
                self.assertEqual(selected(ops(gap), VERIFY).returncode, 0)
                rejected = selected(ops(gap - 1), VERIFY)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("busy", rejected.stderr)

    def test_mxu_tile_program_is_scheduled_and_rechecked(self):
        ops = [("alu_imm", 'kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32'),
               vls("vload", 0), ("alu_imm", 'kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 512 : i32'),
               vls("vload", 2), mxu("push_weight", 0), mxu("matmul", 0), mxu("pop_acc_bf16", 0, reg=4),
               ("alu_imm", 'kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32'),
               vls("vstore", 4, 8), ("alu_imm", 'kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32'),
               vls("vstore", 5, 8), ("alu_imm", 'kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32'),
               MARKER, HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(ops, True, consumer, VERIFY)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                self.assertNotEqual(run(OPT, without_delays(result.stdout), VERIFY).returncode, 0)
                for entrypoint in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
                    self.assertEqual(run(OPT, result.stdout, entrypoint).returncode, 0)
                    self.assertNotEqual(run(OPT, without_delays(result.stdout), entrypoint).returncode, 0)
        # The marker publishes only after the last accumulator write and both stores.
        self.check([mxu("matmul", 0), addi(1, 0, 1), MARKER, delay(100), nop(), HALT], False, VERIFY,
                   diagnostic="requires all prior asynchronous writes to be complete")

    def test_facts_drive_the_mxu_boundaries(self):
        matmul = mxu("matmul", 1)
        second = mxu("matmul", 1, reg=22)
        original = busy(block(facts(), "mxu1.matmul")) + 1
        self.assert_minimum_gap(matmul, second, original, 60)
        with tempfile.TemporaryDirectory(prefix="atlas-mxu-facts-") as directory:
            def slower(d):
                block(d, "mxu1.matmul")["events"]["compute_busy"].update(last_age=40, count=40)
            path = facts_variant(directory, slower, "slower.json")
            self.assert_minimum_gap(matmul, second, 41, 60, path=path)

            def later(d):
                block(d, "mxu0.push_weight")["events"]["weight_write"].update(first_age=2, last_age=33)
            path = facts_variant(directory, later, "later.json")
            footprint = export(self.check([mxu("push_weight", 0), HALT], True, CONSUMERS[0], VERIFY,
                                          path=path).stdout)["instructions"][0]["footprint"]
            self.assertIn(("weight", 2), {(a["resource"], a["age"]) for a in footprint["accesses"]})
            self.assertEqual(footprint["done_age"], 33)

            for name, mutate, diagnostic in (
                    ("missing", lambda d: d["op_timing"].remove(block(d, "mxu1.matmul")),
                     "mxu1.matmul: facts block is missing"),
                    ("unresolved", lambda d: block(d, "mxu1.matmul").update(events=None),
                     "mxu1.matmul: facts block is unresolved"),
                    ("unmodeled", lambda d: block(d, "mxu1.matmul")["events"]["acc_load"].update(
                        first_age=1, last_age=32, count=32, step=1), "event group acc_load is not modeled"),
                    ("busy", lambda d: block(d, "mxu1.matmul")["events"].pop("compute_busy"),
                     "event group compute_busy is missing"),
                    ("uneven", lambda d: block(d, "mxu1.matmul")["events"]["acc_write"].update(last_age=40),
                     "does not sweep")):
                path = facts_variant(directory, mutate, f"{name}.json")
                for consumer in CONSUMERS:
                    with self.subTest(mutation=name, consumer=consumer):
                        self.check([matmul, HALT], False, consumer, path=path, diagnostic=diagnostic)
                self.check([mxu("matmul", 0), HALT], True, CONSUMERS[0], VERIFY, path=path)

    def test_pending_dma_requires_wait_before_mxu(self):
        from test_rtl_dma_timing import setup, transfer, wait
        self.check([*setup(), transfer(), mxu("matmul", 1), wait(), HALT], False, CONSUMERS[0], dma="wait")
        for consumer in CONSUMERS:
            self.check([*setup(), transfer(), wait(), mxu("matmul", 1), HALT], True, consumer, VERIFY, dma="wait")

    def test_unselected_mxu_rules_are_unchanged(self):
        # Built-in (npu_model) rules: MXU1 back-to-back matmuls share read port 0 for 32 cycles;
        # an MXU0 pop waits for row 0 of the matmul on its accumulator (age 63).
        for first, second, gap in ((mxu("matmul", 1), mxu("matmul", 1, reg=22), 32),
                                   (mxu("matmul", 0), mxu("pop_acc_bf16", 0, reg=22), 64)):
            result = run(OPT, program([first, second, HALT]), CONSUMERS[0])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"cycles = {gap - 2} : i32", result.stdout)
            self.assertNotIn("atlas.rtl_evidence", result.stdout)


if __name__ == "__main__":
    unittest.main()
