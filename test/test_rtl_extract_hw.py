"""Engine, DMA and compiler timing regressions against retained hardware IR."""
from collections import Counter
import copy
import importlib.util
import stat
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import derive, facts, pairs, runner
from rtl_extract.ir import load_document
from rtl_extract.runner import extract, modules
from rtl_extract.spec import available_engines, load_system, load_specs
REPO = Path(__file__).resolve().parents[1]
EMIT = os.environ.get("ATLAS_EMIT")
MAP = yaml.safe_load((REPO / "test/rtl-timing-facts-map.yaml").read_text())
STORAGE = {"MReg", "Acc", "Weight", "Vmem"}
HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
TIMING = ("first_age", "last_age", "count", "step")

def mismatches(expected, actual, path=""):
    if not isinstance(actual, dict):
        return [f"{path}: missing"]
    found = []
    for key, value in expected.items():
        if isinstance(value, dict):
            found += mismatches(value, actual.get(key), f"{path}.{key}" if path else key)
        elif actual.get(key) != value:
            found.append(f"{path}.{key}: expected {value!r}, computed {actual.get(key)!r}")
    return found

def find(document, name):
    return next(m for m in document["modules"] if m["name"] == name)

def add_stage(module, port):
    """Insert one register between ``port``'s driver and the port."""
    ports = {p["name"]: p for p in module["ports"]}
    result = f"mutation.{port}"
    module["operations"].append({"id": result, "kind": "seq.firreg", "operands": [ports[port]["value"], ports["clock"]["value"]],
                                 "results": [result], "result_types": ["i1"], "attributes": {}, "has_regions": False})
    ports[port]["value"] = result

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class EngineCase(unittest.TestCase):
    engine = None
    @classmethod
    def setUpClass(cls):
        if cls.engine is None:
            raise unittest.SkipTest("engine base class")
        cls.spec = load_specs([cls.engine])[cls.engine]
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER)
        cls.records = cls.extract(cls.document)
    @classmethod
    def extract(cls, document, spec=None):
        return {r["name"].split(".", 1)[1]: r for r in extract(document, spec or cls.spec)}

    def mutated(self, module, port):
        document = copy.deepcopy(self.document)
        add_stage(find(document, module), port)
        return document

    def assert_delayed(self, mutated, base, group, others=()):
        """``group`` starts and ends exactly one age later with the same count; ``others`` are unchanged."""
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        for key in ("first_age", "last_age"):
            self.assertEqual(mutated["events"][group][key], base["events"][group][key] + 1)
        self.assertEqual(mutated["events"][group]["count"], base["events"][group]["count"])
        for other in others:
            self.assertEqual(mutated["events"][other], base["events"][other])

    def test_records_computed(self):
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertEqual(record["status"], "computed", record.get("reason"))

    def test_regression_values_match_expect(self):
        self.assertTrue(self.spec["expect"])
        for name, expected in self.spec["expect"].items():
            with self.subTest(name):
                self.assertEqual(mismatches(expected, self.records[name]), [])

class XluTest(EngineCase):
    engine = "xlu"

    def test_payload_dependent_busy_rejected(self):
        document = copy.deepcopy(self.document)
        ports = {p["name"]: p for p in find(document, self.spec["module"])["ports"]}
        ports[self.spec["busy"]]["value"] = ports["io_mregReadResp_bits"]["value"]
        self.assertTrue(all(r["status"] == "unresolved" and "unapproved input" in r["reason"] for r in self.extract(document).values()))

    def test_extra_write_valid_stage_delays_writes(self):
        mutated = self.extract(self.mutated(self.spec["module"], self.spec["events"]["write"]["valid"]))["transpose"]
        self.assert_delayed(mutated, self.records["transpose"], "write", ["read"])

class Mxu0Test(EngineCase):
    engine = "mxu0"

class Mxu1Test(EngineCase):
    engine = "mxu1"

    def test_extra_tree_output_stage_delays_writes_by_one(self):
        mutated = self.extract(self.mutated("InnerProductTrees", "io_out_valid"))
        for op in ("matmul", "matmul_acc"):
            with self.subTest(op):
                self.assertEqual(mutated[op]["status"], "computed", mutated[op].get("reason"))
                self.assertEqual(mutated[op]["events"]["acc_write"]["first_age"], self.records[op]["events"]["acc_write"]["first_age"] + 1)
                self.assertEqual(mutated[op]["events"]["mreg_read0"], self.records[op]["events"]["mreg_read0"])

class ScalarLsuTest(EngineCase):
    engine = "scalar_lsu"

    def test_extra_response_stage_in_partner_delays_writeback(self):
        mutated = self.extract(self.mutated(self.spec["partner"]["module"], "io_scalarResp_valid"))["load"]
        base = self.records["load"]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        self.assertEqual(mutated["events"]["load_write"]["first_age"], base["events"]["load_write"]["first_age"] + 1)
        self.assertEqual(mutated["events"]["vmem_read"], base["events"]["vmem_read"])

class VlsuTest(EngineCase):
    engine = "vlsu"
    @staticmethod
    def timing(record):
        return ({g: {k: e[k] for k in TIMING} for g, e in record["events"].items()}, record["first_free_age"], record["next_issue_age"])

    def test_operands_do_not_move_timing(self):
        for op in self.spec["operations"]:
            with self.subTest(op):
                self.assertEqual(self.timing(self.records[f"{op}/operands"]), self.timing(self.records[op]))

    def test_extra_write_valid_stage_delays_writes(self):
        for op, group in (("vload", "mreg_write"), ("vstore", "vmem_write")):
            with self.subTest(op):
                mutated = self.extract(self.mutated(self.spec["module"], self.spec["events"][group]["valid"]))[op]
                base = self.records[op]
                self.assert_delayed(mutated, base, group, [g for g in base["events"] if g != group])

class VpuTest(EngineCase):
    engine = "vpu"

    def test_all_operations_and_busy_release(self):
        self.assertEqual(len(self.records), 29)
        self.assertNotIn("fp8", self.records)
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertGreaterEqual(record["first_free_age"], record["next_issue_age"])

    def bypassed(self):
        """Relu's valid register bypassed: its response valid follows the request directly."""
        document = copy.deepcopy(self.document)
        ports = {p["name"]: p for p in find(document, "Relu")["ports"]}
        ports["io_resp_valid"]["value"] = ports["io_req_valid"]["value"]
        return document

    def test_bypassed_valid_stage_advances_writes(self):
        mutated = self.extract(self.bypassed(), {**self.spec, "checks": []})["relu"]  # the family check would reject the mutant
        base = self.records["relu"]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        for group in ("write0", "write1"):
            if base["events"][group]["count"]:
                self.assertLess(mutated["events"][group]["first_age"], base["events"][group]["first_age"])

    def test_bypassed_valid_stage_violates_family_check(self):
        mutated = self.extract(self.bypassed())["relu"]
        self.assertEqual(mutated["status"], "unresolved")
        self.assertIn("Check failed", mutated["reason"])

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class ResponseLatencyTest(unittest.TestCase):

    def test_every_memory_response_is_one_cycle(self):
        specs = load_specs()
        document = load_document(HW_IR, sorted({m for s in specs.values() for m in modules(s)}), EXPORTER)
        for name, spec in specs.items():
            resolved, notes = derive.resolve(document, spec)
            for group, event in resolved["events"].items():
                if "response" in event:
                    with self.subTest(f"{name}.{group}"):
                        note = notes["scratchpad_read_derivation"][group]
                        self.assertEqual((note["measured"], note["memory_read_latency"], event["response"]["latency"]), (1, 1, 1))

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class SystemTest(unittest.TestCase):
    """The decoder-cut AtlasCore spec reproduces each engine's isolated events with every core assertion quiet."""
    SAME = {("vlsu", "vload"): ("vload", {"vmem_read": "vmem_read", "mreg_write": "mreg_write"}),
            ("vlsu", "vstore"): ("vstore", {"mreg_read": "mreg_read", "vmem_write": "vmem_write"}),
            ("xlu", "transpose"): ("transpose", {"read": "xlu_read", "write": "xlu_write"}),
            ("vpu", "add"): ("add", {"read0": "vpu_read0", "read1": "vpu_read1", "write0": "vpu_write0"})}
    @classmethod
    def setUpClass(cls):
        cls.spec = load_system("core")
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER, assertions=True)
        cls.records = {r["name"].split(".", 1)[1]: r for r in extract(cls.document, cls.spec)}

    def test_records_computed_with_invariants(self):
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertEqual(record["status"], "computed", record.get("reason"))

    def test_engine_events_match(self):
        engines = load_specs(sorted({e for e, _ in self.SAME}))
        for (engine, op), (system_op, groups) in self.SAME.items():
            spec = {**engines[engine], "operations": {op: engines[engine]["operations"][op]}, "variants": {}, "checks": []}
            alone = extract(self.document, spec)[0]
            with self.subTest(f"{engine}.{op}"):
                for group, system_group in groups.items():
                    mine, theirs = self.records[system_op]["events"][system_group], alone["events"][group]
                    self.assertEqual({k: mine[k] for k in TIMING}, {k: theirs[k] for k in TIMING}, group)
                    self.assertEqual(mine["values"], {k: theirs["values"][k] for k in mine["values"]}, group)

def dma_groups(trace):
    """Multiset of DMA events without ages or source tags (tags count up across transfers): (group, fields)."""
    return Counter((g, tuple(sorted((k, v) for k, v in f.items() if k != "tag"))) for e in trace for g, f in e["events"].items() if g.startswith("dma_") or g.startswith("dram."))

def vls_events(trace, shift=0):
    return sorted((g, e["age"] - shift, tuple(sorted(f.items()))) for e in trace for g, f in e["events"].items() if not (g.startswith("dma_") or g.startswith("dram.")))

def span(trace, signal):
    ages = [e["age"] for e in trace if e["signals"][signal]]
    return (ages[0], ages[-1]) if ages else None
DMA_RESPONSES = (("fast", {"accept": [1], "latency": 1}, 450),
                 ("stalled", {"accept": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], "latency": 43}, 450),
                 ("slow", {"accept": [1] + [0] * 9, "latency": 61}, 900))

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwareDMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load(["core_dma"])["core_dma"]
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER, True)
        cls.concrete, _ = derive.resolve(cls.document, cls.spec, {})
        cls.circuit = runner.build_single(cls.document, cls.concrete)

    def paired(self, first, second, gap, spec=None, second_inputs=None):
        spec = spec or self.concrete
        timeline = {**pairs.commands(spec, first, {}), **pairs.commands(spec, second, second_inputs or {}, gap)}
        trace, problem = pairs.run(self.circuit, spec, timeline)
        self.assertIsNone(problem)
        self.assertEqual(runner.failures(self.circuit, trace), [])
        return trace

    def alone(self, name, spec=None):
        spec = spec or self.concrete
        trace, problem = pairs.run(self.circuit, spec, pairs.commands(spec, name, {}))
        self.assertIsNone(problem)
        return trace

    def test_pair_windows(self):
        window = copy.deepcopy(self.raw)
        for pair, expected in self.raw["expect"].items():
            stable = expected["stable_gap"]
            window["pairs"][pair]["gaps"] = "1..2" if stable is None else f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(self.document, window, self.spec, jobs=8):
            pair = record["name"].split(".", 1)[1]
            expected = self.raw["expect"][pair]
            with self.subTest(record["name"]):
                self.assertEqual(record["invariants"]["checked"], 177)
                self.assertEqual(record["stable_gap"], expected["stable_gap"], record["results"])
                self.assertEqual(record["min_gap"], expected["stable_gap"], record["results"])  # first accepted in the window
                if "rejected_by" in expected:
                    self.assertRegex(record["results"][0]["result"], expected["rejected_by"])
                if "disjoint" in expected:
                    self.assertEqual(record["shared_control"]["disjoint"], expected["disjoint"], record["shared_control"])

    def test_second_channel_launch_is_queued_not_dropped(self):
        first, second = self.alone("dma_load0"), self.alone("dma_load1")
        trace = self.paired("dma_load0", "dma_load1", 1)
        # Every beat of both transfers happens exactly once: 64 requests with distinct addresses, 64 distinct writes.
        self.assertEqual(dma_groups(trace), dma_groups(first) + dma_groups(second))
        requests = [f for e in trace for g, f in e["events"].items() if g == "dram.request"]
        self.assertEqual(len({f["address"] for f in requests}), 64)
        # Requests are dispatched in launch order: the second transfer's first beat follows the first's last.
        ages = [e["age"] for e in trace for g in e["events"] if g == "dram.request"]
        self.assertEqual(ages, sorted(ages))
        self.assertEqual([f["address"] >> 12 & 1 for f in requests], [0] * 32 + [1] * 32)
        # Channel busy is per channel: channel 0 retires while channel 1 is still in flight; channel 1 busy from launch.
        self.assertEqual(span(trace, "dma_channel0"), span(first, "dma_channel0"))
        self.assertEqual(span(trace, "dma_channel1")[0], 2)
        self.assertGreater(span(trace, "dma_channel1")[1], span(trace, "dma_channel0")[1])
        self.assertEqual(span(trace, "dma_launched1"), (2, 2))

    def test_vls_on_other_lines_keeps_its_events_while_the_dma_waits(self):
        first, second = self.alone("dma_load0"), self.alone("vload_bank3")
        gap = 40
        trace = self.paired("dma_load0", "vload_bank3", gap)
        self.assertEqual(vls_events(trace, gap), vls_events(second))
        self.assertEqual(dma_groups(trace), dma_groups(first))
        self.assertNotEqual(pairs.events(trace, {}), sorted(pairs.events(first, {}) + pairs.events(second, {}, gap)))
        writes = lambda t: [e["age"] for e in t if "dma_vmem_write" in e["events"]]
        self.assertGreater(writes(trace)[-1], writes(first)[-1])
        # On another bank nothing moves.
        trace = self.paired("dma_load0", "vload", gap)
        self.assertEqual(pairs.events(trace, {}), sorted(pairs.events(first, {}) + pairs.events(self.alone("vload"), {}, gap)))

    def test_dma_store_reads_wait_for_the_lsu(self):
        first, second = self.alone("dma_store0"), self.alone("vload_bank3")
        trace = self.paired("dma_store0", "vload_bank3", 1)
        self.assertEqual(vls_events(trace, 1), vls_events(second))
        self.assertEqual(dma_groups(trace), dma_groups(first))
        reads = lambda t: [e["age"] for e in t if "dma_vmem_read" in e["events"]]
        self.assertEqual(reads(first), list(range(1, 33)))
        self.assertGreater(reads(trace)[-1], 33)

    def test_same_channel_reuse_is_not_protected(self):
        first = self.alone("dma_load0")
        trace = self.paired("dma_load0", "dma_load0", 10, second_inputs={"dma_vmem_addr": 197120})
        # Channel 0 busy falls when the first transfer retires although the second transfer's beats are outstanding.
        self.assertEqual(span(trace, "dma_channel0"), span(first, "dma_channel0"))
        later = [e["age"] for e in trace if "dma_vmem_write" in e["events"] and e["age"] > span(first, "dma_channel0")[1]]
        self.assertEqual(len(later), 32)

    def test_disjoint_bank_results_hold_under_other_response_patterns(self):
        for label, overlay, limit in DMA_RESPONSES:
            spec = copy.deepcopy(self.spec)
            spec["responders"]["dram"].update(overlay)
            spec["limit"] = limit
            window = copy.deepcopy(self.raw)
            window["pairs"] = {"load0_vload": {**self.raw["pairs"]["load0_vload"], "gaps": "1..2"},
                               "load0_add": {**self.raw["pairs"]["load0_add"], "gaps": "1..2"},
                               "load0_load1": {**self.raw["pairs"]["load0_load1"], "gaps": "1..1"}}
            results = {r["name"]: r for r in pairs.extract(self.document, window, spec, jobs=5)}
            with self.subTest(label):
                self.assertEqual(results["core_dma.load0_vload"]["stable_gap"], 1, results["core_dma.load0_vload"]["results"])
                self.assertEqual(results["core_dma.load0_add"]["stable_gap"], 1, results["core_dma.load0_add"]["results"])
                self.assertTrue(results["core_dma.load0_vload"]["shared_control"]["disjoint"])
                self.assertIn("missing dram.request events", results["core_dma.load0_load1"]["results"][0]["result"])

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwareMXUBesideDMATests(unittest.TestCase):
    """pairs/core_dma_mxu.yaml: every MXU command on both units against a DMA load or store, in both orders."""
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load(["core_dma_mxu"])["core_dma_mxu"]
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER, True)

    def test_pair_windows(self):
        # The full sweep accepts every gap; gaps 1 and 2 for every pair, where the commands are closest.
        window = copy.deepcopy(self.raw)
        for pair in window["pairs"].values():
            pair["gaps"] = "1..2"
        records = pairs.extract(self.document, window, self.spec, jobs=8)
        self.assertEqual({r["name"].split(".", 1)[1] for r in records}, set(self.raw["expect"]))
        for record in records:
            expected = self.raw["expect"][record["name"].split(".", 1)[1]]
            with self.subTest(record["name"]):
                self.assertEqual(record["invariants"], {"checked": 177, "excluded": []})
                self.assertEqual((record["min_gap"], record["stable_gap"]), (expected["min_gap"], expected["stable_gap"]), record["results"])
                self.assertEqual(record["shared_control"]["disjoint"], expected["disjoint"], record["shared_control"])

    def test_mxu_keeps_its_events_under_other_response_patterns(self):
        for label, overlay, limit in DMA_RESPONSES:
            spec = copy.deepcopy(self.spec)
            spec["responders"]["dram"].update(overlay)
            spec["limit"] = limit
            window = copy.deepcopy(self.raw)
            # The transfer first: the accept pattern repeats by absolute age, so a shifted launch would see another phase.
            window["pairs"] = {n: {**self.raw["pairs"][n], "gaps": "1..2"} for n in
                               ("load0_mxu0_matmul", "store0_mxu1_pop_acc_bf16", "store0_mxu0_push_acc_bf16", "load0_mxu1_matmul_acc")}
            for record in pairs.extract(self.document, window, spec, jobs=8):
                with self.subTest(label=label, pair=record["name"]):
                    self.assertEqual(record["stable_gap"], 1, record["results"])


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
        specs = load_specs(available_engines())
        document = load_document(os.environ["ATLAS_HW_IR"], sorted({m for s in specs.values() for m in modules(s)}), os.environ["ATLAS_HW_EXPORTER"])
        records = [facts.block(r) for s in specs.values() for r in extract(document, s)]
    else:
        return None
    return {r["name"]: r for r in records}

def mapped_ops():
    return {name: (family, mnemonic) for family in MAP["families"].values() for name, mnemonic in family["ops"].items()}

def summary(ages):
    """Independent event summary for compiler access ages."""
    ages = sorted(ages)
    steps = {b - a for a, b in zip(ages, ages[1:])}
    return {"first_age": ages[0] if ages else None, "last_age": ages[-1] if ages else None, "count": len(ages), "step": steps.pop() if len(steps) == 1 else None}

def select(instance, selector):
    """Every access the selector names; at least one must exist."""
    resource, direction, *register = selector.split()
    matches = [i for i, a in enumerate(instance["footprint"]["accesses"]) if a["resource"] == resource and a["write"] == (direction == "w")]
    if register:
        name, _, offset = register[0].partition("+")
        element = instance[name] + int(offset or 0)
        matches = [i for i in matches if instance["footprint"]["accesses"][i]["first"] == (element * 32 if resource == "MReg" else element)]
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
            fact = tuple((record["events"] or {}).get(group, {}).get(k) for k in TIMING)
            computed = tuple(summary(ages)[k] for k in TIMING)
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
        scalars = {path.split(".")[1] for row in family.get("scalars", []) for path in row[:2] if path.startswith("events.")}
        for group, event in (record["events"] or {}).items():
            if event["count"] and group not in family["events"] and group not in scalars | set(family.get("not_modeled", {})):
                problems.append(f"{name}: event group {group} neither compared nor not_modeled")
    return problems

def table(rows):

    def text(v):
        return "..".join(map(str, v[:2])) + f" n{v[2]} s{v[3]}" if isinstance(v, tuple) else str(v)
    lines = [f"{'record':28} {'mnemonic':22} {'quantity':46} {'rtl':>16} rel {'compiler':>16}  ok"]
    lines += [f"{r[0]:28} {r[1]:22} {r[2]:46} {text(r[3]):>16} {r[5]:^3} {text(r[4]):>16}  {'ok' if r[6] else 'MISMATCH'}" for r in rows]
    return "\n".join(lines)

class CompilerCrossCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exe = os.environ.get("ATLAS_TIMING_PROBE") or Path(os.environ.get("ATLAS_OOT_BIN_DIR", "")) / "rtl-timing-facts-probe"
        if not Path(exe).is_file():
            raise unittest.SkipTest("requires a built rtl-timing-facts-probe (ATLAS_OOT_BIN_DIR or ATLAS_TIMING_PROBE)")
        cls.facts = load_facts()
        if cls.facts is None:
            raise unittest.SkipTest("requires ATLAS_OP_TIMING, or ATLAS_HW_IR and ATLAS_HW_EXPORTER")
        cls.probe = run_probe(exe, sorted({m for _, m in mapped_ops().values()}))

    def test_compiler_rules_and_fact_coverage(self):
        rows = compare(self.facts, self.probe)
        failed = [r for r in rows if not r[6]]
        self.assertEqual(failed, [], "\n" + table(failed))
        self.assertLessEqual(set(MAP["not_modeled_records"]), set(self.facts))
        self.assertEqual(coverage(self.facts), [])

    def test_perturbed_fact_is_reported(self):
        facts = copy.deepcopy(self.facts)
        facts["vpu.add"]["events"]["write0"]["first_age"] += 1
        facts["vlsu.vload"]["first_free_age"] += 1
        facts["mxu0.matmul"]["events"]["compute_busy"]["last_age"] += 1
        failed = {(r[0], r[2].split(" ")[0]) for r in compare(facts, self.probe) if not r[6]}
        self.assertEqual(failed, {("vpu.add", "events.write0"), ("vlsu.vload", "first_free_age"), ("mxu0.matmul", "events.compute_busy.last_age")})

def compiler_ops():
    """Compiler machine op (generic form, zero operands) for every declared decode word."""
    regs = {"dst": 0, "src": 0}
    ops = {("vpu", n): ("vpu_binary", {"kind": k, "dst": 0, "lhs": 0, "rhs": 0}) for n, k in {"add": "add", "sub": "sub", "mul": "mul", "pairmin": "min", "pairmax": "max"}.items()}
    ops.update({("vpu", n): ("vpu_unary", {"kind": k, **regs}) for n, k in {
        "mov": "mov", "rcp": "recip", "exp": "exp", "exp2": "exp2", "square": "square", "cube": "cube", "relu": "relu",
        "sin": "sin", "cos": "cos", "tanh": "tanh", "log": "log2", "sqrt": "sqrt"}.items()})
    ops.update({("vpu", n): ("vpu_reduce", {"kind": k, **regs}) for n, k in {
        "csum": "col_sum", "cmin": "col_min", "cmax": "col_max", "rsum": "row_sum", "rmin": "row_min", "rmax": "row_max"}.items()})
    ops.update({("vpu", n): ("vpu_pack", {"direction": d, **regs, "scale_reg": 0}) for n, d in {"fp8pack": "bf16_to_fp8", "fp8unpack": "fp8_to_bf16"}.items()})
    ops.update({("vpu", f"vli{m.title()}"): ("vli", {"mode": m, "dst": 0, "immediate": 0}) for m in ("one", "col", "row", "all")})
    for unit in (0, 1):
        engine = f"mxu{unit}"
        ops.update({(engine, f"push_{k}"): ("mxu_push", {"kind": k.replace("weight", "weight_fp8"), "unit": unit, "src": 0, "slot": 0}) for k in ("weight", "acc_fp8", "acc_bf16")})
        ops.update({(engine, f"pop_acc_{f}"): ("mxu_pop", {"format": f, "unit": unit, "dst": 0, "slot": 0, "scale_reg": 0}) for f in ("fp8", "bf16")})
        ops.update({(engine, n): ("mxu_matmul", {"unit": unit, "src": 0, "weight_slot": 0, "acc_slot": 0, "accumulate": a}) for n, a in (("matmul", False), ("matmul_acc", True))})
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
        declared = {(e, n): w for e, s in load_specs().items() for n, w in s.get("decode", {}).get("words", {}).items()}
        ops = compiler_ops()
        self.assertEqual(set(declared), set(ops))
        keys = sorted(ops)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.mlir"
            path.write_text(mlir([ops[k] for k in keys]))
            words = [int(w, 16) for w in subprocess.check_output([EMIT, str(path)], text=True).split()]
        self.assertEqual(dict(zip(keys, words)), {k: declared[k] for k in keys})
TOOL = Path(__file__).resolve().parents[1] / "tools/retain-ee290-hw-ir.py"
FIRTOOL = """#!/bin/sh
[ "$1" = --version ] && { echo stub-firtool 1; exit 0; }
while [ $# -gt 0 ]; do case "$1" in -o) out=$2;; esac; shift; done
cat > "$out" <<'IR'
%s
IR
"""
HW = "hw.module @AtlasCore() {\n  seq.firreg\n}\nhw.module @ScalarCore() {\n}\nhw.module @DmaEngine() {\n}"
CIRCT_OPT = '#!/bin/sh\n[ "$1" = --version ] && echo stub-circt-opt\nexit 0\n'

class RetainTest(unittest.TestCase):

    def run_tool(self, hw):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, text in (("firtool", FIRTOOL % hw), ("circt-opt", CIRCT_OPT)):
            (tmp / name).write_text(text)
            (tmp / name).chmod((tmp / name).stat().st_mode | stat.S_IXUSR)
        (tmp / "in.fir").write_text("circuit AtlasCore :\n")
        (tmp / "opts.txt").write_text("disallowLocalVariables\n")
        (tmp / "anno.json").write_text(json.dumps([{"class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation", "filename": "/elsewhere/h.json"}]))
        result = subprocess.run([sys.executable, str(TOOL), "--firrtl", str(tmp / "in.fir"), "--annotations", str(tmp / "anno.json"),
                                 "--lowering-options", str(tmp / "opts.txt"), "--firtool", str(tmp / "firtool"), "--circt-opt", str(tmp / "circt-opt"),
                                 "--output", str(tmp / "out")], capture_output=True, text=True)
        return tmp / "out", result

    def test_verified_manifest_and_hierarchy_redirect(self):
        out, result = self.run_tool(HW)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "verified")
        self.assertEqual(manifest["structure"]["sequential_op_occurrences"]["seq.firreg"], 1)
        self.assertEqual(manifest["firtool"]["version"], "stub-firtool 1")
        self.assertEqual([c["stage"] for c in manifest["commands"]], ["lower", "verify"])
        lower = manifest["commands"][0]["argv"]
        self.assertIn("--ir-hw", lower)
        self.assertEqual(json.loads((out / "retained-hw.anno.json").read_text())[0]["filename"], str(out / "top_module_hierarchy.json"))

    def test_missing_atlas_module_is_rejected(self):
        out, result = self.run_tool(HW.replace("@DmaEngine", "@Other"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["state"], "structure_check_failed")
SPEC = importlib.util.spec_from_file_location("retain_hw_ir", TOOL)
retain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retain)

class RetainedHWIRTests(unittest.TestCase):

    def test_debug_matches_recorded_timing_lowering_policy(self):
        command = retain.lowering_command("firtool", Path("input.fir"), Path("debug.anno.json"), "noAlwaysComb\n", Path("out"), "debug")
        self.assertEqual(command, ["firtool", "--format=fir", "--export-module-hierarchy", "--verify-each=true",
            "--warn-on-unprocessed-annotations", "--disable-annotation-classless",
            "--disable-annotation-unknown", "--mlir-timing", "--mlir-print-debuginfo",
            "--lowering-options=noAlwaysComb", "-O=debug", "--preserve-values=named",
            "--annotation-file=debug.anno.json", "--ir-hw", "-o", "out/ee290.debug.hw.mlir", "input.fir"])

    def test_default_keeps_sequential_memory_replacement(self):
        command = retain.lowering_command("firtool", Path("input.fir"), Path("hw.anno.json"), "noAlwaysComb", Path("out"), "legacy")
        self.assertIn("--repl-seq-mem", command)
        self.assertIn("--repl-seq-mem-file=out/mems.conf", command)
        self.assertNotIn("-O=debug", command)
        self.assertNotIn("--preserve-values=named", command)
        self.assertEqual(command[-2:], ["out/ee290.hw.mlir", "input.fir"])

    def test_hierarchy_redirect_keeps_other_annotations(self):
        inline = {"class": "firrtl.transforms.BlackBoxInlineAnno", "name": "Toy.sv", "text": "module Toy; endmodule"}
        sidecar = [{"class": cls, "filename": "/old/file.json"} for cls in retain.HIERARCHY] + [inline]
        prepared = retain.prepare_annotations(sidecar, Path("out"))
        self.assertEqual(prepared[-1], inline)
        self.assertEqual([a["filename"] for a in prepared[:-1]], [str(Path("out") / name) for name in retain.HIERARCHY.values()])

    def test_sha256_hashes_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.write_bytes(b"abc")
            self.assertEqual(retain.sha256(source), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
if __name__ == "__main__":
    unittest.main()
