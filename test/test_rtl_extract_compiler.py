"""Compiler cross-check: RTL-computed op timing against the compiler's built-in rules (lib/AtlasTiming.cpp).

test/rtl-timing-facts-map.yaml states which fact is compared with which compiler quantity, and
under which convention; a disagreement means the compiler constants or the RTL changed.

Compiler side: ATLAS_TIMING_PROBE names a built test/rtl-timing-facts-probe, or ATLAS_LLVM_INCLUDES
(os.pathsep-separated LLVM source and build include directories) builds one in a temporary directory:
  c++ -std=c++17 -O2 -I include -I <llvm-project>/llvm/include -I <llvm-build>/include \\
      test/rtl-timing-facts-probe.cpp lib/AtlasTiming.cpp -o build/rtl-extract/probe/rtl-timing-facts-probe
RTL side: ATLAS_OP_TIMING names an op_timing JSON from tools/extract-rtl-timing.py, or ATLAS_HW_IR and
ATLAS_HW_EXPORTER compute one. The tests skip otherwise. `python3 test/test_rtl_extract_compiler.py --table`
prints every comparison.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
MAP = yaml.safe_load((REPO / "test/rtl-timing-facts-map.yaml").read_text())
FIELDS = ("first_age", "last_age", "count", "step")
STORAGE = {"MReg", "Acc", "Weight", "Vmem"}  # every compiler access to these must meet a compared fact


def probe_path(workdir):
    if os.environ.get("ATLAS_TIMING_PROBE"):
        return os.environ["ATLAS_TIMING_PROBE"]
    includes = [p for p in os.environ.get("ATLAS_LLVM_INCLUDES", "").split(os.pathsep) if p]
    if not includes:
        return None
    exe = Path(workdir) / "rtl-timing-facts-probe"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-I", REPO / "include",
                    *[a for p in includes for a in ("-I", p)], REPO / "test/rtl-timing-facts-probe.cpp",
                    REPO / "lib/AtlasTiming.cpp", "-o", exe], check=True)
    return exe


def run_probe(exe, mnemonics):
    out = json.loads(subprocess.run([exe, *mnemonics], check=True, stdout=subprocess.PIPE, text=True).stdout)
    return {i["operation"]: i for i in out["instances"]}


def load_facts():
    if os.environ.get("ATLAS_OP_TIMING"):
        records = json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())["records"]
    elif os.environ.get("ATLAS_HW_IR") and os.environ.get("ATLAS_HW_EXPORTER"):
        from rtl_extract.ir import load_document
        from rtl_extract.runner import extract, modules
        from rtl_extract.spec import available_engines, load_target
        specs = load_target("atlas", available_engines("atlas"))
        document = load_document(os.environ["ATLAS_HW_IR"], sorted({m for s in specs.values() for m in modules(s)}),
                                 os.environ["ATLAS_HW_EXPORTER"])
        records = [r for s in specs.values() for r in extract(document, s)]
    else:
        return None
    return {r["name"]: r for r in records}


def mapped_ops():
    return {name: (family, mnemonic) for family in MAP["families"].values() for name, mnemonic in family["ops"].items()}


def summary(ages):
    """The extractor's event summary (rtl_extract.summaries.group) of a list of touch ages."""
    ages = sorted(ages)
    steps = {b - a for a, b in zip(ages, ages[1:])}
    return {"first_age": ages[0] if ages else None, "last_age": ages[-1] if ages else None,
            "count": len(ages), "step": steps.pop() if len(steps) == 1 else None}


def select(instance, selector):
    """Every access the selector names; at least one must exist."""
    resource, direction, *register = selector.split()
    matches = [i for i, a in enumerate(instance["footprint"]["accesses"])
               if a["resource"] == resource and a["write"] == (direction == "w")]
    if register:
        name, _, offset = register[0].partition("+")
        element = instance[name] + int(offset or 0)
        matches = [i for i in matches if instance["footprint"]["accesses"][i]["first"] ==
                   (element * 32 if resource == "MReg" else element)]
    if not matches:
        raise AssertionError(f"{instance['operation']}: selector {selector!r} matches no access")
    return matches


def compiler_value(expr, instance, record):
    name, _, offset = (part.strip() for part in expr.partition("+"))
    footprint = instance["footprint"]
    if name.startswith("hold."):
        value = max((h["to"] for h in footprint["holds"] if h["unit"] == name[5:]), default=0)
    elif name.startswith("events."):
        value = fact_value(record, name)
    else:
        value = footprint[name]
    return value + int(offset or 0)


def fact_value(record, path):
    value = record
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def compare(facts, probe):
    """Rows (record, mnemonic, quantity, fact, compiler, relation, ok)."""
    rows = []
    for name, (family, mnemonic) in mapped_ops().items():
        record, instance = facts.get(name), probe[mnemonic]
        if record is None:
            rows.append((name, mnemonic, "record", "missing", "-", "=", False))
            continue
        if instance["footprint"]["error"]:
            rows.append((name, mnemonic, "error", "", instance["footprint"]["error"], "=", False))
        accesses, claimed = instance["footprint"]["accesses"], set()
        for group, selectors in family["events"].items():
            chosen = sorted({i for s in selectors for i in select(instance, s)})
            claimed.update(chosen)
            ages = [accesses[i]["age"] + n * accesses[i]["step"] for i in chosen for n in range(accesses[i]["count"])]
            fact = tuple((record["events"] or {}).get(group, {}).get(k) for k in FIELDS)
            computed = tuple(summary(ages)[k] for k in FIELDS)
            rows.append((name, mnemonic, f"events.{group}", fact, computed, "=", fact == computed))
        for i, a in enumerate(accesses):
            if a["resource"] in STORAGE and i not in claimed:
                rows.append((name, mnemonic, "unmatched compiler access", "-", f"{a['resource']} {a['first']}", "=", False))
        for fact_path, expr, *relation in family.get("scalars", []):
            relation = relation[0] if relation else "="
            fact, computed = fact_value(record, fact_path), compiler_value(expr, instance, record)
            ok = fact is not None and (computed == fact if relation == "=" else computed >= fact)
            rows.append((name, mnemonic, f"{fact_path} ~ {expr}", fact, computed, relation, ok))
    return rows


def coverage(facts):
    """Facts neither compared nor declared not modeled."""
    ops, problems = mapped_ops(), []
    for name, record in facts.items():
        if name in MAP["not_modeled_records"]:
            continue
        if name not in ops:
            problems.append(f"{name}: record not mapped")
            continue
        if record["status"] != "computed":
            problems.append(f"{name}: {record['status']}: {record.get('reason')}")
            continue
        family = ops[name][0]
        for group, event in (record["events"] or {}).items():
            if event["count"] and group not in family["events"] and group not in family.get("not_modeled", {}):
                problems.append(f"{name}: event group {group} neither compared nor not_modeled")
    return problems


def table(rows):
    def text(v):
        return "..".join(map(str, v[:2])) + f" n{v[2]} s{v[3]}" if isinstance(v, tuple) else str(v)
    lines = [f"{'record':28} {'mnemonic':22} {'quantity':46} {'rtl':>16} rel {'compiler':>16}  ok"]
    lines += [f"{r[0]:28} {r[1]:22} {r[2]:46} {text(r[3]):>16} {r[5]:^3} {text(r[4]):>16}  {'ok' if r[6] else 'MISMATCH'}"
              for r in rows]
    return "\n".join(lines)


class CompilerCrossCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.workdir)
        exe = probe_path(cls.workdir)
        if exe is None:
            raise unittest.SkipTest("requires ATLAS_TIMING_PROBE or ATLAS_LLVM_INCLUDES")
        cls.facts = load_facts()
        if cls.facts is None:
            raise unittest.SkipTest("requires ATLAS_OP_TIMING, or ATLAS_HW_IR and ATLAS_HW_EXPORTER")
        cls.probe = run_probe(exe, sorted({m for _, m in mapped_ops().values()}))

    def test_compiler_rules_match_rtl_facts(self):
        rows = compare(self.facts, self.probe)
        failed = [r for r in rows if not r[6]]
        self.assertEqual(failed, [], "\n" + table(failed))

    def test_every_fact_compared_or_not_modeled(self):
        self.assertEqual(coverage(self.facts), [])

    def test_not_modeled_records_exist(self):
        self.assertLessEqual(set(MAP["not_modeled_records"]), set(self.facts))

    def test_perturbed_fact_is_reported(self):
        facts = copy.deepcopy(self.facts)
        facts["vpu.add"]["events"]["write0"]["first_age"] += 1
        facts["vlsu.vload"]["first_free_age"] += 1
        failed = {(r[0], r[2].split(" ")[0]) for r in compare(facts, self.probe) if not r[6]}
        self.assertEqual(failed, {("vpu.add", "events.write0"), ("vlsu.vload", "first_free_age")})


if __name__ == "__main__":
    if "--table" in sys.argv:
        with tempfile.TemporaryDirectory() as workdir:
            facts, exe = load_facts(), probe_path(workdir)
            if facts is None or exe is None:
                raise SystemExit("set ATLAS_OP_TIMING (or ATLAS_HW_IR/ATLAS_HW_EXPORTER) and ATLAS_TIMING_PROBE (or ATLAS_LLVM_INCLUDES)")
            rows = compare(facts, run_probe(exe, sorted({m for _, m in mapped_ops().values()})))
            print(table(rows))
            print(f"{sum(r[6] for r in rows)}/{len(rows)} comparisons hold; uncovered: {coverage(facts)}")
            raise SystemExit(0 if all(r[6] for r in rows) else 1)
    unittest.main()
