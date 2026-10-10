"""Compiler cross-check: RTL-computed op timing against the compiler's built-in rules (lib/AtlasTiming.cpp).

test/rtl-timing-facts-map.yaml states which fact is compared with which compiler quantity, and
under which convention; a disagreement means the compiler constants or the RTL changed.

Compiler side: ATLAS_TIMING_PROBE names a built test/rtl-timing-facts-probe, or ATLAS_LLVM_INCLUDES
(os.pathsep-separated LLVM source and build include directories) builds one in a temporary directory:
  c++ -std=c++17 -O2 -I include -I <llvm-project>/llvm/include -I <llvm-build>/include \\
      test/rtl-timing-facts-probe.cpp lib/AtlasTiming.cpp -o build/rtl-extract/probe/rtl-timing-facts-probe
RTL side: ATLAS_OP_TIMING names a merlin.op_timing.v1 JSON from tools/extract-rtl-timing.py (its
op_timing blocks), or ATLAS_HW_IR and ATLAS_HW_EXPORTER compute them. The tests skip otherwise. `python3 test/test_rtl_extract_compiler.py --table`
prints every comparison.
ATLAS_EMIT (atlas-emit) additionally checks every spec decode word against the compiler's instruction encoding.
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
from rtl_extract import facts  # noqa: E402
from rtl_extract.ir import load_document  # noqa: E402
from rtl_extract.runner import extract, modules  # noqa: E402
from rtl_extract.spec import available_engines, load_target  # noqa: E402
EMIT = os.environ.get("ATLAS_EMIT")
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
        document = json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())
        if document.get("schema") != facts.SCHEMA:
            raise ValueError(f"{os.environ['ATLAS_OP_TIMING']}: expected schema {facts.SCHEMA}, found {document.get('schema')!r}")
        records = document["op_timing"]
    elif os.environ.get("ATLAS_HW_IR") and os.environ.get("ATLAS_HW_EXPORTER"):
        specs = load_target("atlas", available_engines("atlas"))
        document = load_document(os.environ["ATLAS_HW_IR"], sorted({m for s in specs.values() for m in modules(s)}),
                                 os.environ["ATLAS_HW_EXPORTER"])
        records = [facts.block(r) for s in specs.values() for r in extract(document, s)]
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
        if record["events"] is None:
            problems.append(f"{name}: unresolved: {record['evidence']}")
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


def compiler_ops():
    """Compiler machine op (generic form, zero operands) for every declared decode word."""
    regs = {"dst": 0, "src": 0}
    ops = {("vpu", n): ("vpu_binary", {"kind": k, "dst": 0, "lhs": 0, "rhs": 0})
           for n, k in {"add": "add", "sub": "sub", "mul": "mul", "pairmin": "min", "pairmax": "max"}.items()}
    ops.update({("vpu", n): ("vpu_unary", {"kind": k, **regs}) for n, k in {
        "mov": "mov", "rcp": "recip", "exp": "exp", "exp2": "exp2", "square": "square", "cube": "cube", "relu": "relu",
        "sin": "sin", "cos": "cos", "tanh": "tanh", "log": "log2", "sqrt": "sqrt"}.items()})
    ops.update({("vpu", n): ("vpu_reduce", {"kind": k, **regs}) for n, k in {
        "csum": "col_sum", "cmin": "col_min", "cmax": "col_max", "rsum": "row_sum", "rmin": "row_min", "rmax": "row_max"}.items()})
    ops.update({("vpu", n): ("vpu_pack", {"direction": d, **regs, "scale_reg": 0})
                for n, d in {"fp8pack": "bf16_to_fp8", "fp8unpack": "fp8_to_bf16"}.items()})
    ops.update({("vpu", f"vli{m.title()}"): ("vli", {"mode": m, "dst": 0, "immediate": 0}) for m in ("one", "col", "row", "all")})
    for unit in (0, 1):
        engine = f"mxu{unit}"
        ops.update({(engine, f"push_{k}"): ("mxu_push", {"kind": k.replace("weight", "weight_fp8"), "unit": unit, "src": 0, "slot": 0})
                    for k in ("weight", "acc_fp8", "acc_bf16")})
        ops.update({(engine, f"pop_acc_{f}"): ("mxu_pop", {"format": f, "unit": unit, "dst": 0, "slot": 0, "scale_reg": 0}) for f in ("fp8", "bf16")})
        ops.update({(engine, n): ("mxu_matmul", {"unit": unit, "src": 0, "weight_slot": 0, "acc_slot": 0, "accumulate": a})
                    for n, a in (("matmul", False), ("matmul_acc", True))})
    ops[("xlu", "vtrpose")] = ("xlu_transpose", regs)
    ops[("vlsu", "vload")] = ("vload", {"dst": 0, "base": 0, "offset": 0, "format": "raw"})
    ops[("vlsu", "vstore")] = ("vstore", {"src": 0, "base": 0, "offset": 0, "format": "raw"})
    ops.update({("scalar_lsu", k): ("scalar_load", {"kind": k, "dst": 0, "base": 0, "offset": 0}) for k in ("lw", "seld")})
    ops[("scalar_lsu", "sw")] = ("scalar_store", {"kind": "sw", "src": 0, "base": 0, "offset": 0})
    return ops


def mlir(ops):
    def attr(v):
        return ("true" if v else "false") if isinstance(v, bool) else f"{v} : i32" if isinstance(v, int) else f'"{v}"'
    lines = ['  %s0 = "atlas.start"() : () -> !atlas.state']
    for i, (name, attrs) in enumerate(ops, 1):
        body = ", ".join(f"{k} = {attr(v)}" for k, v in attrs.items())
        lines.append(f'  %s{i} = "atlas.{name}"(%s{i - 1}) {{{body}}} : (!atlas.state) -> !atlas.state')
    return "module {\n" + "\n".join(lines) + "\n}\n"


@unittest.skipUnless(EMIT, "requires ATLAS_EMIT (atlas-emit)")
class CompilerWordsTest(unittest.TestCase):
    def test_decode_words_match_compiler_encoding(self):
        declared = {(e, n): w for e, s in load_target("atlas").items() for n, w in s.get("decode", {}).get("words", {}).items()}
        ops = compiler_ops()
        self.assertEqual(set(declared), set(ops))
        keys = sorted(ops)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.mlir"
            path.write_text(mlir([ops[k] for k in keys]))
            words = [int(w, 16) for w in subprocess.check_output([EMIT, str(path)], text=True).split()]
        self.assertEqual(dict(zip(keys, words)), {k: declared[k] for k in keys})


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
