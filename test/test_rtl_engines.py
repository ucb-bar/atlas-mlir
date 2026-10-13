"""Selected RTL engine footprints, DMA policies, and cross-engine overlap.

Requires ATLAS_OOT_BIN_DIR and ATLAS_OP_TIMING. MXU pair expectations come from
RTL extraction specs or the ATLAS_OP_PAIRS result files when set.
"""
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from rtl_test_support import (
    BANK_WORDS, CHANNEL_REUSE, CONSUMERS, DMA_SECOND, EIGHT_SLOTS, EMIT, FACTS, HALT,
    IN_FLIGHT, KERNELS, MARKER, MXU_WORDS, OPT, PROVIDER, ROOT, RTLTestSupport, VERIFY, WAITED,
    addi, binary, block, busy, config, delay, dma_setup, export, facts, facts_variant, gapped,
    issue_gap, mxu, nop, pack, program, reduce, run, selected, selection, separated, stream,
    transfer, transpose, unary, vli, vls, wait, without_delays, word_base,
)


PAIR_SPECS = {u: yaml.safe_load((ROOT / f"tools/rtl_extract/targets/atlas/pairs/mxu{u}.yaml").read_text())
              for u in (0, 1)}
OPERATIONS = tuple(MXU_WORDS.values())
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
OWN_UNIT = "MXU in-flight matmuls"  # the only hold: a command sequences its own MXU (pairs/core_mxu*.yaml)


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
            assert "first_inputs" not in pair, f"mxu{unit}.{name}: first-operation overlays are not mapped"
            operands = {"reg": inputs.get("io_cmd_bits_mregId", 2), "acc": inputs.get("io_cmd_bits_accSel", 0),
                        "slot": inputs.get("io_cmd_bits_weightSlot", 0)}
            gap = stable.get(f"mxu{unit}.{name}", spec["expect"][name]["stable_gap"])
            rows.append((unit, name, pair["first"], pair["second"], operands, gap))
    return rows


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedMXUTimingTest(RTLTestSupport, unittest.TestCase):

    def gap_accepted(self, first, second, gap, drain, **options):
        return selected([first, *separated(gap), second, delay(drain), nop(), HALT], VERIFY, **options).returncode == 0

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
                        self.assertEqual({(h["unit"], h["index"]) for h in footprint["holds"]}, {(OWN_UNIT, unit)})
                        self.assertTrue(all(h["from"] == 0 and h["to"] == end for h in footprint["holds"]))
                        self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_pair_results_bound_the_compiler_gap(self):
        document = facts()
        rows = pair_results()
        self.assertTrue(rows)
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

    def test_mxu_overlaps_other_engines_at_the_mreg_rules(self):
        """Disjoint operands: MXU against the other MXU, VPU, XLU and VLS at gap 1 in both orders (the AtlasCore sweeps
        of pairs/core_mxu*.yaml). Shared registers wait for the MREG release rules and a shared physical bank for its
        port, accepted at the facts-derived boundary and rejected one cycle earlier; same-MXU commands keep the busy rule."""
        document = facts()
        latency = block(document, "vlsu.vload")["assumptions"]["scratchpad_read_latency"]
        group = lambda name, key: block(document, name)["events"][key]
        add = ("vpu_binary", 'kind = "add", dst = 40 : i32, lhs = 36 : i32, rhs = 50 : i32')
        transpose = ("xlu_transpose", "dst = 44 : i32, src = 46 : i32")
        others = {"add": add, "transpose": transpose, "vload": vls("vload", 56), "vstore": vls("vstore", 58)}
        disjoint = [(mxu(op, u), other) for u in (0, 1) for op in OPERATIONS for other in others.values()]
        disjoint += [(mxu(a, u), mxu(b, 1 - u, reg=22)) for u in (0, 1) for a in OPERATIONS for b in OPERATIONS]
        ops = lambda first, second, g: [addi(6, 0, 256), first, *separated(g), second, delay(130), nop(), HALT]
        for first, second in disjoint + [(b, a) for a, b in disjoint]:
            with self.subTest(first=first[1], second=second[1]):
                result = selected(ops(first, second, 1), VERIFY)
                self.assertEqual(result.returncode, 0, result.stderr)
        for u in (0, 1):
            pop, matmul = f"mxu{u}.pop_acc_bf16", f"mxu{u}.matmul"
            port = group(matmul, "mreg_read0")["last_age"] - group("vlsu.vstore", "mreg_read")["first_age"] + 1
            cases = [  # (first, second, boundary): RAW, WAR and one physical bank
                (mxu("pop_acc_bf16", u, reg=22), vls("vstore", 22), group(pop, "mreg_write0")["last_age"] + 1),
                (vls("vload", 22), mxu("matmul", u, reg=22), group("vlsu.vload", "mreg_write")["last_age"] + 1),
                (mxu("matmul", u, reg=22), vls("vload", 22), group(matmul, "mreg_read0")["last_age"] + latency + 1),
                # The LSU reports a VSTORE's source as read until it drains (pairs/core_mxu_cross.yaml).
                (vls("vstore", 22), mxu("pop_acc_bf16", u, reg=22),
                 max(group("vlsu.vstore", "mreg_read")["last_age"] + latency + 1, block(document, "vlsu.vstore")["first_free_age"])),
                (mxu("matmul", u, reg=12), vls("vstore", 44), port),
                (mxu("matmul", u), mxu("pop_acc_bf16", u, reg=22), busy(block(document, matmul)) + 1)]
            for first, second, gap in cases:
                with self.subTest(first=first[1], second=second[1], gap=gap):
                    self.assertGreater(gap, 1)
                    self.assertEqual(selected(ops(first, second, gap), VERIFY).returncode, 0)
                    rejected = selected(ops(first, second, gap - 1), VERIFY)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertRegex(rejected.stderr, "reserved for|reported read until|port busy|MXU in-flight matmuls")

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
                   diagnostic="atlas.complete waits for prior fixed-latency work")

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

    def test_mxu_runs_beside_a_pending_transfer(self):
        # pairs/core_dma_mxu.yaml: every MXU command keeps its isolated events beside a DMA load or store, and the
        # transfer keeps its beats; the launch needs no MXU drain.
        for unit in (0, 1):
            for first, second in ((transfer(), mxu("matmul", unit)), (mxu("matmul", unit), transfer())):
                for consumer in CONSUMERS:
                    with self.subTest(unit=unit, first=first[0], consumer=consumer):
                        result = self.check([*dma_setup(), first, second, wait(), HALT], True, consumer, VERIFY, dma="wait")
                        self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                        order = [op for op, _, _ in stream(result.stdout)]
                        self.assertLess(order.index("mxu_matmul"), order.index("dma_wait"))
                        if consumer == CONSUMERS[0]:
                            self.assertEqual(issue_gap(result.stdout, first, second), 1)

    def test_unselected_mxu_rules_are_unchanged(self):
        # Built-in (npu_model) rules: MXU1 back-to-back matmuls share read port 0 for 32 cycles;
        # an MXU0 pop waits for row 0 of the matmul on its accumulator (age 63).
        for first, second, gap in ((mxu("matmul", 1), mxu("matmul", 1, reg=22), 32),
                                   (mxu("matmul", 0), mxu("pop_acc_bf16", 0, reg=22), 64)):
            result = run(OPT, program([first, second, HALT]), CONSUMERS[0])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"cycles = {gap - 2} : i32", result.stdout)
            self.assertNotIn("atlas.rtl_evidence", result.stdout)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedDMATimingTest(unittest.TestCase):
    def check_consumers(self, ops, accepted):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked = selected(ops, consumer, VERIFY, dma="wait")
                self.assertEqual(checked.returncode == 0, accepted, checked.stderr)
                if accepted:
                    self.assertEqual(run(EMIT, checked.stdout).returncode, 0, checked.stderr)
        return checked

    def test_both_consumers_and_final_export(self):
        for consumer in CONSUMERS:
            checked = selected(WAITED, consumer, VERIFY, dma="wait")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn('dma = "wait"', checked.stdout)
            exported = run(EMIT, checked.stdout, "--rtl-timing-json")
            self.assertEqual(exported.returncode, 0, exported.stderr)
            data = json.loads(exported.stdout)
            self.assertEqual(data["schema"], "atlas.resolved_rtl_timing.v1")
            self.assertEqual(data["provider"], {"id": PROVIDER, "version": 1, "dma_policy": "wait"})
            self.assertIn("dma.load.ch0..7", data["applicability"]["supported_operations"])
            self.assertIsNone(data["applicability"]["dma_domain"]["completion_latency"])
            launch = next(i for i in data["instructions"] if i["mnemonic"] == "dma.load.ch0")
            retired = next(i for i in data["instructions"] if i["mnemonic"] == "dma.wait.ch0")
            self.assertIsNone(launch["footprint"]["done_age"])
            self.assertIsNone(launch["footprint"]["dma_cycles"])
            memory = [a for a in launch["footprint"]["accesses"] if a["resource"] == "vmem"]
            self.assertTrue(memory)
            for access in memory:
                self.assertTrue(access["at_completion"])
                self.assertIsNone(access["age"])
                self.assertIsNone(access["step"])
            self.assertGreater(retired["issue_epoch"], launch["issue_epoch"])
            self.assertEqual(retired["epoch_offset"], 0)
            self.assertNotIn("logical_issue_cycle", launch)
            self.assertIn("minimum_issue_cycle", launch)

    def test_accepted_sequences(self):
        cases = [[addi(5, 0, 31), config(), HALT],  # config is synchronous, not a transfer
                 # Captured scalar operands and the global base may change while pending.
                 [*dma_setup(), transfer(), *word_base(1, 0x90001000), addi(6, 0, 32), addi(2, 0, 32),
                  addi(5, 0, 1), config(), wait(), addi(5, 0, 0), config(), HALT]]
        cases += [[*dma_setup(), transfer(channel=c), wait(c), transfer("store", c), wait(c), HALT] for c in range(8)]
        for ops in cases:
            with self.subTest(ops=ops):
                self.check_consumers(ops, True)

    def test_pending_is_released_only_by_its_wait(self):
        prefix = [*dma_setup(), transfer()]
        for ops in ([*prefix, wait(1), HALT], [*prefix, transfer(channel=1), wait(1), HALT],
                    [*prefix, addi(5, 0, 1), config(), vls("vload"), wait(), HALT],
                    [*prefix, MARKER, wait(), HALT], [*prefix, HALT], [*prefix, delay(100), HALT],
                    [wait(), HALT], [*prefix, wait(), wait(), HALT]):
            with self.subTest(ops=ops):
                self.check_consumers(ops, False)

    def test_queued_transfers_follow_the_evidence_rules(self):
        base = [*dma_setup(), *DMA_SECOND]
        accepted = {
            "two channels on disjoint windows": [*base, transfer(), transfer(channel=1, reg=7, dram=3), wait(), wait(1), HALT],
            "two loads of one DRAM span": [*base, transfer(), transfer(channel=1, reg=7), wait(), wait(1), HALT],
            "VLS off the pending window": [*base, transfer(), vls("vload", 4, 9), wait(), HALT],
            "wait right after launch": [*base, transfer(), wait(), HALT],
        }
        rejected = {
            "same channel before its wait": ([*base, transfer(), transfer(reg=7, dram=3), wait(), wait(), HALT], CHANNEL_REUSE),
            "overlapping windows": ([*base, transfer(), transfer(channel=1, dram=3), wait(), wait(1), HALT], IN_FLIGHT),
            "load and store of one DRAM span": ([*base, transfer(), transfer("store", 1, 7), wait(), wait(1), HALT], IN_FLIGHT),
            "VLS on the pending window": ([*base, transfer(), vls("vload"), wait(), HALT], IN_FLIGHT),
        }
        windows = [op for c in range(9) for op in word_base(10 + c, 64 * c)]
        launches = [transfer(channel=c % 8, reg=10 + c) for c in range(8)]
        eight = [*base, *windows, *launches, *(wait(c) for c in range(8)), HALT]
        ninth_early = [*base, *windows, *launches, wait(1), transfer(channel=1, reg=18), *(wait(c) for c in range(8) if c != 1), wait(1), HALT]
        ninth_late = [*base, *windows, *launches, wait(0), transfer(channel=0, reg=18), *(wait(c) for c in range(1, 8)), wait(0), HALT]
        accepted["eight transfers pending"] = eight
        accepted["ninth launch after launch 0 waited"] = ninth_late
        # The scheduler keeps waits ahead of later launches, so it cannot move launch 0's wait past the ninth.
        accepted["ninth launch on another channel after launch 0 waited"] = [
            *base, *windows, *launches, wait(1), wait(0), transfer(channel=1, reg=18), *(wait(c) for c in range(2, 8)), wait(1), HALT]
        rejected["ninth launch before launch 0 waited"] = (ninth_early, EIGHT_SLOTS)
        for name, ops in accepted.items():
            with self.subTest(case=name):
                checked = self.check_consumers(ops, True)
                exported = json.loads(run(EMIT, checked.stdout, "--rtl-timing-json").stdout)
                self.assertEqual(exported["applicability"]["dma_domain"]["pending_limit"], 8)
        for name, (ops, diagnostic) in rejected.items():
            with self.subTest(case=name):
                self.assertIn(diagnostic, self.check_consumers(ops, False).stderr)

    def test_operand_domain_boundaries_and_unknowns(self):
        for vmem, offset, size, base in ((0, 0, 32, 0), (392192, 0xffffffe0, 32, 31), (392192, 0x90000000, 4096, 31)):
            self.check_consumers([*dma_setup(vmem, offset, size, base), transfer(), wait(), HALT], True)
        for vmem, offset, size, base in ((1, 0, 32, 0), (393216, 0, 32, 0), (65528, 0, 64, 0),
                                         (0, 1, 32, 0), (0, 0xffffffe0, 64, 0), (0, 0, 0, 0),
                                         (0, 0, 31, 0), (0, 0, 33, 0), (0, 0, 4128, 0), (0, 0, 32, 32)):
            with self.subTest(vmem=vmem, offset=offset, size=size, base=base):
                self.check_consumers([*dma_setup(vmem, offset, size, base), transfer(), wait(), HALT], False)
        self.check_consumers([addi(6, 0, 0), transfer(), wait(), HALT], False)

    def test_final_boundaries_recheck_the_wait(self):
        checked = selected(WAITED, VERIFY, dma="wait")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        # Replace the wait with a NOP, keeping the state chain intact.
        unsafe = re.sub(r'(%\w+) = "atlas.dma_wait"\((%\w+)\)[^\n]*',
                        lambda m: m[1] + ' = "atlas.alu_imm"(' + m[2] + ') <{dst = 0 : i32, immediate = 0 : i32, '
                        'kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state', checked.stdout)
        self.assertNotEqual(unsafe, checked.stdout)
        for tool, args in ((OPT, [VERIFY]), (EMIT, []), (OPT, ["--convert-atlas-to-llvm"]),
                           (OPT, ["--convert-atlas-to-llvm-calls"])):
            with self.subTest(args=args):
                self.assertNotEqual(run(tool, unsafe, *args).returncode, 0)
        staged = run(OPT, checked.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertEqual(run(OPT, staged.stdout, "--finalize-atlas-llvm-calls").returncode, 0)
        # Drop the staged wait call and renumber the remaining words.
        lines, index = [], 0
        for line in staged.stdout.splitlines():
            if 'atlas.source_op = "atlas.dma_wait"' in line:
                continue
            if "atlas.word_index =" in line:
                line = re.sub(r"atlas.word_index = [0-9]+ : i32", f"atlas.word_index = {index} : i32", line)
                index += 1
            lines.append(line)
        self.assertEqual(len(lines), len(staged.stdout.splitlines()) - 1)
        finalized = run(OPT, "\n".join(lines), "--finalize-atlas-llvm-calls")
        self.assertNotEqual(finalized.returncode, 0)
        self.assertIn("DMA", finalized.stderr)

    def test_dma_policy_must_be_explicit_and_known(self):
        unselected = selected(WAITED, "--insert-atlas-delays")
        self.assertNotEqual(unselected.returncode, 0)
        self.assertIn("dma=wait", unselected.stderr)
        for policy in ("latency", "none", "true"):
            with self.subTest(policy=policy):
                result = selected(WAITED, VERIFY, dma=policy)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsupported DMA policy", result.stderr)
        plain = selected([HALT], VERIFY)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertNotIn("dma =", plain.stdout)
        self.assertEqual(json.loads(run(EMIT, plain.stdout, "--rtl-timing-json").stdout)["provider"]["dma_policy"],
                         "none")


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedXLUTimingTest(RTLTestSupport, unittest.TestCase):

    def test_both_consumers_export_facts_numbers(self):
        record = block(facts(), "xlu.transpose")
        busy = record["first_free_age"] - 1
        for src, dst in ((3, 7), (5, 5), (63, 0)):
            for consumer in CONSUMERS:
                with self.subTest(src=src, dst=dst, consumer=consumer):
                    result = self.check([transpose(src, dst), HALT], True, consumer, VERIFY)
                    footprint = next(x for x in export(result.stdout)["instructions"]
                                     if "xlu" in x["mnemonic"])["footprint"]
                    accesses = footprint["accesses"]
                    self.assertEqual(sorted(x["age"] for x in accesses),
                                     sorted(record["events"][g]["first_age"] for g in ("read", "write")))
                    self.assertTrue(all(x["count"] == 32 and x["step"] == 1 for x in accesses))
                    self.assertEqual(footprint["done_age"], busy)
                    self.assertEqual(footprint["write_release"], record["events"]["write"]["last_age"])
                    self.assertEqual(footprint["read_release"], record["events"]["read"]["last_age"] +
                                     record["assumptions"]["scratchpad_read_latency"])
                    self.assertEqual({h["unit"]: h["to"] for h in footprint["holds"]},
                                     {u: busy for u in ("XLU", "VPU")})
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_facts_drive_the_busy_boundary(self):
        # DELAY N occupies N+1 cycles; the launches are N+2 apart and unrelated.
        boundary = block(facts(), "xlu.transpose")["first_free_age"]
        for gap in (boundary - 1, boundary):
            self.check([transpose(), delay(gap - 2), transpose(22, 24), delay(64), nop(), HALT],
                       gap >= boundary, VERIFY)
        with tempfile.TemporaryDirectory(prefix="atlas-xlu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "xlu.transpose").update(first_free_age=70))
            for gap in (69, 70):
                with self.subTest(gap=gap):
                    self.check([transpose(), delay(gap - 2), transpose(22, 24), delay(68), nop(), HALT],
                               gap >= 70, VERIFY, path=slower)
            missing = facts_variant(directory, lambda d: d["op_timing"].remove(block(d, "xlu.transpose")))
            for consumer in CONSUMERS:
                rejected = self.check([transpose(), HALT], False, consumer, path=missing)
                self.assertIn("xlu.transpose: facts block is missing", rejected.stderr)
            self.check([addi(6, 0, 256), vls("vload"), delay(33), nop(), HALT], True, VERIFY, path=missing)

    def test_vls_transpose_chain_and_marker_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 256), vls("vload", 3),
               transpose(3, 35), vls("vstore", 35, 8), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.check(ops, True, consumer, VERIFY)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            self.assertNotEqual(run(OPT, without_delays(result.stdout), VERIFY).returncode, 0)
            self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.check([transpose(), *suffix], False, VERIFY)
        for src, dst in ((-1, 7), (64, 7), (3, -1), (3, 64)):
            self.check([transpose(src, dst), HALT], False, CONSUMERS[0])

    def test_xlu_runs_beside_a_pending_transfer(self):
        for ops in ([*dma_setup(), transfer(), transpose(), wait(), HALT], [*dma_setup(), transfer(), wait(), transpose(), HALT]):
            for consumer in CONSUMERS:
                result = self.check(ops, True, consumer, VERIFY, dma="wait")
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)


VPU_OPERATIONS = {
    **{f"v{kind}.bf16": (binary(kind=kind), f"vpu.{kind}") for kind in ("add", "sub", "mul")},
    "vmaximum.bf16": (binary(kind="max"), "vpu.pairmax"), "vminimum.bf16": (binary(kind="min"), "vpu.pairmin"),
    "vmov": (unary("mov"), "vpu.mov"), "vrecip.bf16": (unary("recip"), "vpu.rcp"),
    **{f"v{kind}.bf16": (unary(kind), f"vpu.{kind}")
       for kind in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")},
    "vlog2.bf16": (unary("log2"), "vpu.log"),
    "vredsum.row.bf16": (reduce("row_sum"), "vpu.rsum"), "vredmax.row.bf16": (reduce("row_max"), "vpu.rmax"),
    "vredmin.row.bf16": (reduce("row_min"), "vpu.rmin"), "vredsum.bf16": (reduce("col_sum"), "vpu.csum"),
    "vredmax.bf16": (reduce("col_max"), "vpu.cmax"), "vredmin.bf16": (reduce("col_min"), "vpu.cmin"),
    "vpack.bf16.fp8": (pack("bf16_to_fp8"), "vpu.fp8pack"),
    "vunpack.fp8.bf16": (pack("fp8_to_bf16"), "vpu.fp8unpack"),
    "vli.all": (vli("all"), "vpu.vliAll"), "vli.row": (vli("row"), "vpu.vliRow"),
    "vli.col": (vli("col"), "vpu.vliCol"), "vli.one": (vli("one"), "vpu.vliOne"),
}


def expected_mreg_accesses(record):
    """(age, count, step) MReg accesses a block implies; a column reduction
    reads its 64 source rows twice, kept as two passes."""
    expected = []
    for group in record["events"].values():
        if group["count"] == 128:
            expected += [(group["first_age"], 64, group["step"]), (group["first_age"] + 64, 64, group["step"])]
        elif group["count"]:
            expected.append((group["first_age"], group["count"], group["step"]))
    return sorted(expected)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedVPUTimingTest(RTLTestSupport, unittest.TestCase):

    def test_every_vpu_operation_takes_its_numbers_from_the_facts(self):
        document = facts()
        for mnemonic, (operation, name) in VPU_OPERATIONS.items():
            record = block(document, name)
            busy = record["first_free_age"] - 1
            for consumer in CONSUMERS:
                with self.subTest(mnemonic=mnemonic, consumer=consumer):
                    result = self.check([operation, HALT], True, consumer, VERIFY)
                    self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                    self.assertNotEqual(run(OPT, without_delays(result.stdout), VERIFY).returncode, 0)
                    footprint = next(i for i in export(result.stdout)["instructions"]
                                     if i["mnemonic"] == mnemonic)["footprint"]
                    self.assertEqual(sorted((a["age"], a["count"], a["step"]) for a in footprint["accesses"]
                                            if a["resource"] == "mreg"), expected_mreg_accesses(record))
                    self.assertEqual(footprint["done_age"], busy)
                    self.assertEqual(footprint["write_release"], record["next_issue_age"])
                    # The VPU slot frees at next_issue_age; the XLU stays excluded until first_free_age.
                    issued = record["next_issue_age"] - 1
                    self.assertEqual({h["unit"]: (h["from"], h["to"]) for h in footprint["holds"]},
                                     {"VPU": (0, issued), **({"XLU": (issued + 1, busy)} if issued < busy else {})})
                    if mnemonic == "vmul.bf16":
                        self.assertEqual(footprint["read_release"], record["events"]["read0"]["last_age"] +
                                         record["assumptions"]["scratchpad_read_latency"])

    def test_register_domain(self):
        for pair, accepted in (((62, 2, 4), True), ((0, 60, 62), True), ((1, 2, 4), False), ((0, 3, 4), False),
                               ((0, 2, 5), False), ((0, 2, 64), False), ((-2, 2, 4), False), ((0, 0, 4), False),
                               ((0, 32, 4), False), ((0, 2, 0), False), ((0, 2, 34), False)):
            for consumer in CONSUMERS:
                with self.subTest(pair=pair, consumer=consumer):
                    self.check([binary(*pair), HALT], accepted, consumer)
        # Physical bank aliases (register mod 32) pass the dialect verifier and fail in the provider.
        for op in (unary("mov", 0, 32), reduce("row_sum", 32, 0), reduce("col_sum", 0, 0),
                   pack("bf16_to_fp8", 0, 1), pack("fp8_to_bf16", 4, 4), unary("square", 2, 34)):
            with self.subTest(op=op):
                self.assertIn(PROVIDER, self.check([op, HALT], False, CONSUMERS[0]).stderr)

    def test_facts_drive_the_serialized_boundary(self):
        # The next VPU command may issue at next_issue_age and the XLU at first_free_age; a VSTORE waits only
        # for the data it reads.
        record = block(facts(), "vpu.mul")
        written = max(record["next_issue_age"], record["events"]["write0"]["last_age"]) + 1
        for next_op, boundary, reason in ((binary(6, 8, 10), record["next_issue_age"], "timing resource conflict"),
                                          (transpose(8, 12), record["first_free_age"], "timing resource conflict"),
                                          (vls("vstore", 4), written, "m4 reserved for writing"),
                                          (vls("vstore", 12), 1, None)):
            for gap in (boundary - 1, boundary) if boundary > 1 else (boundary,):
                with self.subTest(gap=gap, next_op=next_op):
                    between = [] if gap == 1 else [delay(gap - 2)]
                    result = self.check([addi(6, 0, 0), binary(), *between, next_op, delay(64), nop(), HALT],
                                        gap >= boundary, VERIFY)
                    if gap < boundary:
                        self.assertIn(reason, result.stderr)
        with tempfile.TemporaryDirectory(prefix="atlas-vpu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "vpu.mul").update(first_free_age=70, next_issue_age=69))
            for next_op, boundary in ((binary(6, 8, 10), 69), (transpose(8, 12), 70)):
                for gap in (boundary - 1, boundary):
                    with self.subTest(gap=gap, next_op=next_op):
                        self.check([addi(6, 0, 0), binary(), delay(gap - 2), next_op, delay(68), nop(), HALT],
                                   gap >= boundary, VERIFY, path=slower)
            unresolved = facts_variant(directory, lambda d: block(d, "vpu.mul").update(events=None))
            self.assertIn("vpu.mul: facts block is unresolved",
                          self.check([binary(), HALT], False, CONSUMERS[0], path=unresolved).stderr)
            self.check([binary(kind="add"), HALT], True, CONSUMERS[0], VERIFY, path=unresolved)
            narrow = facts_variant(directory, lambda d: block(d, "vpu.mul")["events"]["write0"].update(count=32))
            self.assertIn("does not sweep 1x64", self.check([binary(), HALT], False, CONSUMERS[0], path=narrow).stderr)

    def test_producer_consumer_and_completion_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 1024), vls("vload", 0), vls("vload", 1, offset=32),
               vls("vload", 2, offset=64), vls("vload", 3, offset=96), binary(), vls("vstore", 4, 8),
               vls("vstore", 5, 8, 32), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.check(ops, True, consumer, VERIFY)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            unsafe = without_delays(result.stdout)
            self.assertNotEqual(run(OPT, unsafe, VERIFY).returncode, 0)
            self.assertNotEqual(run(EMIT, unsafe).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.check([binary(), *suffix], False, VERIFY)

    def test_llvm_reconstruction_preserves_evidence_and_rechecks(self):
        result = self.check([binary(), HALT], True, CONSUMERS[0], VERIFY)
        unsafe = without_delays(result.stdout)
        for entrypoint in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            self.assertNotEqual(run(OPT, unsafe, entrypoint).returncode, 0)
        staged = run(OPT, result.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        finalized = run(OPT, staged.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        self.assertIn("op_timing_sha256", finalized.stdout)

    def test_vpu_runs_beside_a_pending_transfer(self):
        for ops in ([*dma_setup(), transfer(), binary(), wait(), HALT],
                    [*dma_setup(), transfer(), wait(), transpose(0, 2), binary(2, 4, 6), HALT]):
            for consumer in CONSUMERS:
                result = self.check(ops, True, consumer, VERIFY, dma="wait")
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)


ENGINE_FIRST = {
    "vload": ("vlsu.vload", word_base(6, 0), vls("vload", 4, 6)),
    "vstore": ("vlsu.vstore", word_base(8, 2 * BANK_WORDS), vls("vstore", 9, 8)),
    "add": ("vpu.add", [], binary(16, 20, 24, "add")),
    "rmax": ("vpu.rmax", [], reduce("row_max", 12, 14)),
    "xlu": ("xlu.transpose", [], transpose(30, 40)),
    "mxu": ("mxu0.matmul", [], mxu("matmul", 0, reg=50)),
}
ENGINE_SECOND = {
    "vload": ("vlsu.vload", word_base(7, 3 * BANK_WORDS), vls("vload", 5, 7)),
    "vstore": ("vlsu.vstore", word_base(9, 4 * BANK_WORDS), vls("vstore", 11, 9)),
    "relu": ("vpu.relu", [], unary("relu", 0, 6)),
    "exp": ("vpu.exp", [], unary("exp", 2, 28)),
    "xlu": ("xlu.transpose", [], transpose(31, 42)),
    "mxu": ("mxu1.matmul", [], mxu("matmul", 1, reg=51)),
}


def expected_gap(document, first, second):
    if first.startswith("mxu") or second.startswith("mxu"):
        return 1  # MXU0 and MXU1 differ; an MXU meets the other engines only at MREG ports and release rules
    if first.startswith("vpu.") and second.startswith("vpu."):
        return block(document, first)["next_issue_age"]
    if first.startswith("vlsu.") != second.startswith("vlsu.") or {first, second} == {"vlsu.vload", "vlsu.vstore"}:
        return 1  # VLOAD, VSTORE and the VPU/XLU pair overlap; banks and MREGs are disjoint
    return block(document, first)["first_free_age"]


def pairs():
    for x, (fa, sa, a) in ENGINE_FIRST.items():
        for y, (fb, sb, b) in ENGINE_SECOND.items():
            yield f"{x}->{y}", fa, fb, sa + sb, a, b


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedOverlapTest(unittest.TestCase):
    def verify(self, source, path=None):
        options = [selection(path)] if path else [selection()]
        return run(OPT, source, *options, VERIFY)

    def test_issue_gap_boundaries_come_from_the_facts(self):
        document = facts()
        for name, fa, fb, setup, a, b in pairs():
            gap = expected_gap(document, fa, fb)
            with self.subTest(pair=name, gap=gap):
                accepted = self.verify(gapped(setup, a, b, gap))
                self.assertEqual(accepted.returncode, 0, accepted.stderr)
                if gap > 1:
                    rejected = self.verify(gapped(setup, a, b, gap - 1))
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertRegex(rejected.stderr, "timing resource conflict: .* busy; issue cycle .*, first permitted cycle")

    def test_both_consumers_issue_at_the_boundary(self):
        document = facts()
        for name, fa, fb, setup, a, b in pairs():
            for consumer in CONSUMERS:
                with self.subTest(pair=name, consumer=consumer):
                    result = run(OPT, program([*setup, a, b, HALT]), selection(), consumer, VERIFY)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    gap, expected = issue_gap(result.stdout, a, b), expected_gap(document, fa, fb)
                    if consumer == CONSUMERS[0]:
                        self.assertEqual(gap, expected)
                    elif gap > 0:  # the scheduler may issue b first, or b's scalar setup between them
                        self.assertTrue(expected <= gap <= expected + len(setup), (gap, expected))

    def test_vpu_issue_window_fails_closed(self):
        setup = []
        add, relu = ENGINE_FIRST["add"][2], ENGINE_SECOND["relu"][2]
        with tempfile.TemporaryDirectory(prefix="atlas-overlap-facts-") as directory:
            for value in (None, 0, 67):
                path = facts_variant(directory, lambda d: block(d, "vpu.add").update(next_issue_age=value))
                for consumer in CONSUMERS:
                    with self.subTest(next_issue_age=value, consumer=consumer):
                        result = run(OPT, program([add, HALT]), selection(path), consumer)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"{PROVIDER}: vpu.add: facts block lacks a next_issue_age", result.stderr)
            # A later next_issue_age moves only the VPU->VPU boundary.
            path = facts_variant(directory, lambda d: block(d, "vpu.add").update(next_issue_age=60))
            for b, boundary in ((relu, 60), (ENGINE_SECOND["xlu"][2], 66)):
                with self.subTest(second=b[0]):
                    self.assertEqual(self.verify(gapped(setup, add, b, boundary), path).returncode, 0)
                    self.assertNotEqual(self.verify(gapped(setup, add, b, boundary - 1), path).returncode, 0)


    def test_vstore_source_is_reported_read_until_it_drains(self):
        # The MREG bank tracker reports a VSTORE's source as read through its busy window: a VPU, XLU or MXU pop
        # writing it waits for first_free_age, while a VLOAD (which checks only pending writes) follows the read
        # release (pairs/core_mxu_cross.yaml: vstore_transpose_to_m9, vstore_vli_all_m8, vstore_mxu*_pop_acc_bf16_m8).
        store = block(facts(), "vlsu.vstore")
        drained = store["first_free_age"]
        released = store["events"]["mreg_read"]["last_age"] + store["assumptions"]["scratchpad_read_latency"] + 1
        setup = [*word_base(8, 2 * BANK_WORDS), *word_base(7, 3 * BANK_WORDS)]
        writers = {"transpose": (transpose(30, 9), drained), "vli": (vli("all", 8), drained),
                   "pop": (("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 8 : i32, slot = 0 : i32, scale_reg = 0 : i32'), drained),
                   "vload": (vls("vload", 9, 7), released)}
        for name, (writer, boundary) in writers.items():
            with self.subTest(writer=name):
                self.assertGreater(boundary, 1)
                self.assertEqual(self.verify(gapped(setup, vls("vstore", 9, 8), writer, boundary)).returncode, 0)
                rejected = self.verify(gapped(setup, vls("vstore", 9, 8), writer, boundary - 1))
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("reported read until" if boundary == drained else "reserved for reading", rejected.stderr)

    def test_example_kernels(self):
        for name, kernel in KERNELS.items():
            for consumer in CONSUMERS:
                with self.subTest(kernel=name, consumer=consumer):
                    result = run(OPT, program(kernel()), selection(), consumer, VERIFY)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(run(EMIT, result.stdout).returncode, 0)


if __name__ == "__main__":
    unittest.main()
