"""Compiler issue gaps never undercut the RTL pair sweeps (rtl_extract.op_pairs.v1).

For each pair record, a program issues the pair's two operations with the pair file's operand
overlays (tools/rtl_extract/targets/atlas/pairs), separated by DELAY, and --verify-atlas-rtl-timing
finds the smallest accepted issue gap; it must be at least the record's stable_gap. Pairs the
selected resolver rejects are reported as unsupported. An informational legacy run lists pairs
whose built-in rules (insert-atlas-delays without a selection) fall below stable_gap.

ATLAS_PAIR_SWEEPS lists sweep files (os.pathsep-separated); the default is pairs-vlsu.json and
pairs-vpu.json in ../explore next to ATLAS_OP_TIMING's directory. Missing files are skipped.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import unittest

from test_delay_insertion import OPT, ROOT, addi, nop, program, run, stream
from test_rtl_timing import FACTS, HALT, RESOLVER, VERIFY, delay, selection, vls, word_base
from test_rtl_vmul_timing import reduce, unary, vli, vmul

sys.path.insert(0, str(ROOT / "tools"))

LIMIT = 400  # widest gap probed; every pair here drains well before it
BINARY = {"add": "add", "sub": "sub", "mul": "mul", "pairmax": "max", "pairmin": "min"}
UNARY = {"mov": "mov", "rcp": "recip", "log": "log2",
         **{k: k for k in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")}}
REDUCE = {"rsum": "row_sum", "rmax": "row_max", "rmin": "row_min",
          "csum": "col_sum", "cmax": "col_max", "cmin": "col_min"}
VLI = {"vliAll": "all", "vliRow": "row", "vliCol": "col", "vliOne": "one"}


def sweep_paths():
    listed = os.environ.get("ATLAS_PAIR_SWEEPS")
    if listed:
        return [Path(p) for p in listed.split(os.pathsep) if p]
    if not FACTS:
        return []
    explore = Path(FACTS).resolve().parent.parent / "explore"
    return [explore / "pairs-vlsu.json", explore / "pairs-vpu.json"]


def records():
    found = []
    for path in sweep_paths():
        if path.is_file():
            document = json.loads(path.read_text())
            if document.get("schema") != "rtl_extract.op_pairs.v1":
                raise AssertionError(f"{path}: expected schema rtl_extract.op_pairs.v1")
            found += [(path.name, r) for r in document["op_pairs"]]
    return found


def pair_files():
    from rtl_extract import pairs
    return pairs, pairs.load("atlas")


def operation(engine, name, fields, base):
    """(setup, op) for one RTL command, or None when the operation has no compiler mapping."""
    if engine == "vlsu" and name in ("vload", "vstore"):
        return word_base(base, fields["io_cmd_bits_vmemLineAddr"] * 8), vls(name, fields["io_cmd_bits_mregBank"], base)
    if engine == "vpu":
        vs1, vs2, vd = (fields[f"io_cmd_bits_{k}"] for k in ("vs1", "vs2", "vd"))
        if name in BINARY:
            return [], vmul(vs1, vs2, vd, BINARY[name])
        if name in UNARY:
            return [], unary(UNARY[name], vs1, vd)
        if name in REDUCE:
            return [], reduce(REDUCE[name], vs1, vd)
        if name in VLI:
            return [], vli(VLI[name], vd)
    if engine in ("mxu0", "mxu1"):
        unit, src = engine[-1], fields["io_cmd_bits_mregId"]
        acc, slot = fields.get("io_cmd_bits_accSel", 0), fields.get("io_cmd_bits_weightSlot", 0)
        if name in ("matmul", "matmul_acc"):
            return [], ("mxu_matmul", f"unit = {unit} : i32, src = {src} : i32, weight_slot = {slot} : i32, "
                                      f"acc_slot = {acc} : i32, accumulate = {str(name == 'matmul_acc').lower()}")
        if name in ("push_weight", "push_acc_fp8", "push_acc_bf16"):
            kind, at = ("weight_fp8", slot) if name == "push_weight" else (name[5:], acc)
            return [], ("mxu_push", f'kind = "{kind}", unit = {unit} : i32, src = {src} : i32, slot = {at} : i32')
        if name in ("pop_acc_bf16", "pop_acc_fp8"):
            return [], ("mxu_pop", f'format = "{name[8:]}", unit = {unit} : i32, dst = {src} : i32, '
                                   f'slot = {acc} : i32, scale_reg = 0 : i32')
    return None


def pair_program(record):
    """(setup, first, second) for a record, from its pair file's commands and overlays."""
    pairs, files = pair_files()
    engine, key = record["engine"], record["name"].split(".", 1)[-1]
    raw, spec = files.get(engine, ({"pairs": {}}, None))
    pair = raw["pairs"].get(key)
    if not pair or (pair["first"], pair["second"]) != (record["first"], record["second"]):
        return None
    sides = []
    for side, base in (("first", 6), ("second", 8)):
        overlay = record.get(f"{side}_inputs", pair.get(f"{side}_inputs", {}))
        commands = pairs.commands(spec, pair[side], overlay)
        if list(commands) != [0]:
            return None
        sides.append(operation(engine, pair[side], {**spec["inputs"], **commands[0]}, base))
    if None in sides:
        return None
    return sides[0][0] + sides[1][0], sides[0][1], sides[1][1]


def separated(gap):
    return [] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]


def gapped(setup, first, second, gap):
    return program([*setup, first, *separated(gap), second, delay(LIMIT), nop(), HALT])


def issue_cycles(printed):
    """(op, sorted fields, issue cycle) for each printed stream op."""
    out, cycle = [], 0
    for op, fields, _ in stream(printed):
        out.append((op, sorted(fields.split(", ")), cycle))
        cycle += int(fields.split("cycles = ")[1].split(" ")[0]) + 1 if op == "delay" else 1
    return out


def issue_gap(printed, first, second):
    """Issue cycle of `second` minus that of `first` (negative if reordered)."""
    ops = issue_cycles(printed)
    find = lambda op, skip=-1: next(i for i, (name, fields, _) in enumerate(ops)
                                    if i != skip and (name, fields) == (op[0], sorted(op[1].split(", "))))
    a = find(first)
    return ops[find(second, a)][2] - ops[a][2]


def selected_gap(setup, first, second, limit):
    """Smallest gap accepted by --verify-atlas-rtl-timing, None if none up to `limit`, or the rejection."""
    option = selection()
    widest = run(OPT, gapped(setup, first, second, limit), option, VERIFY)
    if widest.returncode:
        return None, widest.stderr.strip().splitlines()[0] if widest.stderr else "rejected"
    accepts = lambda g: run(OPT, gapped(setup, first, second, g), option, VERIFY).returncode == 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(1, limit + 1, 16):
            batch = range(start, min(start + 16, limit + 1))
            accepted = list(pool.map(accepts, batch))
            if True in accepted:
                return batch[accepted.index(True)], None
    return limit, None


def consumer_gap(setup, first, second, *options):
    result = run(OPT, program([*setup, first, second, HALT]), *options, "--insert-atlas-delays")
    return issue_gap(result.stdout, first, second) if result.returncode == 0 else None


@unittest.skipUnless(FACTS and OPT.is_file(), "requires explicit RTL timing facts and built Atlas tools")
class RTLGapCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        if not cls.records:
            raise unittest.SkipTest("no pair sweep files (ATLAS_PAIR_SWEEPS)")

    def test_selected_gaps_cover_stable_gaps(self):
        rows, unsupported, unmapped = [], [], []
        for source, record in self.records:
            built = pair_program(record)
            if not built:
                unmapped.append(record["name"])
                continue
            stable = record["stable_gap"]
            gap, why = selected_gap(*built, max(LIMIT // 2, stable + 1))
            if gap is None:
                unsupported.append((record["name"], why))
                continue
            rows.append((record["name"], stable, gap, consumer_gap(*built, selection())))
            with self.subTest(pair=record["name"], source=source):
                self.assertGreaterEqual(gap, stable, f"compiler accepts gap {gap} < RTL stable_gap {stable}")
        print("\nselected compiler gap vs RTL stable_gap (slack = compiler - stable; delays = insert-atlas-delays gap)")
        print(f"  {'pair':34} {'stable':>6} {'compiler':>8} {'slack':>5} {'delays':>6}")
        for name, stable, gap, delays in rows:
            print(f"  {name:34} {stable:6} {gap:8} {gap - stable:5} {str(delays):>6}")
        for name, why in unsupported:
            print(f"  {name:34} unsupported: {why}")
        self.assertFalse(unmapped, f"pair records without a compiler operand mapping: {unmapped}")
        self.assertTrue(rows or unsupported)

    def test_legacy_rules_below_stable_gap_are_listed(self):
        print("\nlegacy (unselected) insert-atlas-delays gap vs RTL stable_gap (informational; '<' marks below)")
        below = []
        for _, record in self.records:
            built = pair_program(record)
            gap = built and consumer_gap(*built)
            low = gap is not None and gap < record["stable_gap"]
            below += [record["name"]] if low else []
            print(f"  {record['name']:34} stable {record['stable_gap']:4} legacy {str(gap):>4} {'<' if low else ''}")
        print(f"  below stable_gap: {below or 'none'}")

    def test_unsupported_pairs_are_resolver_rejections(self):
        # A rejection at the widest gap must come from the selected resolver, not from a malformed program.
        for _, record in self.records:
            built = pair_program(record)
            if built:
                result = run(OPT, gapped(*built, LIMIT), selection(), VERIFY)
                if result.returncode:
                    with self.subTest(pair=record["name"]):
                        self.assertIn(RESOLVER, result.stderr)


if __name__ == "__main__":
    unittest.main()
