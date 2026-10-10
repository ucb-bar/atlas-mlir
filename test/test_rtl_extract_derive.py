"""Derived spec inputs (bundle ports, memory response latency, decoded command codes).

Synthetic tests need no hardware IR. The hardware class (ATLAS_HW_IR, ATLAS_HW_EXPORTER) checks that the
derivations reproduce the values the Atlas specs used to declare.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import derive
from rtl_extract.ir import load_document
from rtl_extract.runner import extract, modules
from rtl_extract.spec import load_spec, load_target, variant

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
EMIT = os.environ.get("ATLAS_EMIT")


def op(ident, kind, operands, typ="i1", **attrs):
    return {"id": ident, "kind": kind, "operands": operands, "results": [ident], "result_types": [typ],
            "attributes": attrs, "has_regions": False}


def mod(name, inputs, outputs, operations):
    ports = [{"name": n, "direction": "input", "value": n, "type": t} for n, t in inputs]
    ports += [{"name": n, "direction": "output", "value": v, "type": t} for n, v, t in outputs]
    return {"name": name, "ports": ports, "arguments": [{"id": n, "type": t} for n, t in inputs], "operations": operations}


def inst(name, module, pins, results):
    """``pins``: {port: value}; ``results``: [(port, type)], result ids are ``name.port``."""
    return {"id": name, "kind": "hw.instance", "operands": list(pins.values()), "results": [f"{name}.{p}" for p, _ in results],
            "result_types": [t for _, t in results], "has_regions": False,
            "attributes": {"moduleName": f"@{module}", "instanceName": name, "argNames": json.dumps(list(pins)),
                           "resultNames": json.dumps([p for p, _ in results])}}


CLOCKED = [("clock", "!seq.clock"), ("reset", "i1")]


def document(stages=1, firmem=(1,)):
    """Sys instantiates a decoder (instruction bits 26:25 are the op), an engine and a memory answering after ``stages``."""
    zero = op("zero", "hw.constant", [], value="false")
    one = op("one", "hw.constant", [], value="true")
    engine = mod("Eng", CLOCKED + [("io_cmd_valid", "i1"), ("io_cmd_bits_op", "i2"), ("io_cmd_bits_tag", "i3"),
                                   ("io_resp_valid", "i1"), ("io_resp_bits", "i8")],
                 [("io_req_valid", "req", "i1"), ("io_req_bits_tag", "tag", "i3"), ("io_req_bits_data", "io_resp_bits", "i8"),
                  ("io_busy", "busy", "i1")],
                 [zero, one, op("go", "hw.constant", [], "i2", value="1 : i2"),
                  op("isgo", "comb.icmp", ["io_cmd_bits_op", "go"], predicate="0 : i64"),
                  op("start", "comb.and", ["io_cmd_valid", "isgo"]),
                  op("req", "seq.firreg", ["start", "clock", "reset", "zero"]),
                  op("tag", "seq.firreg", ["io_cmd_bits_tag", "clock"], "i3"),
                  op("hold", "comb.mux", ["io_resp_valid", "zero", "busy"]),
                  op("next", "comb.mux", ["start", "one", "hold"]),
                  op("busy", "seq.firreg", ["next", "clock", "reset", "zero"])])
    chain = [op(f"stage{i}", "seq.firreg", [f"stage{i - 1}" if i else "io_eng_valid", "clock", "reset", "zero"]) for i in range(stages)]
    memory = mod("Mem", CLOCKED + [("io_eng_valid", "i1"), ("io_eng_bits_row", "i3"), ("io_other_valid", "i1")],
                 [("io_engResp_valid", f"stage{stages - 1}", "i1")],
                 [zero] + chain + [op(f"bank{i}", "seq.firmem", [], "!seq.firmem<8 x 8>", readLatency=f"{n} : i32") for i, n in enumerate(firmem)])
    decoder = mod("Dec", [("io_instr", "i32")], [("io_op", "op", "i2")], [op("op", "comb.extract", ["io_instr"], "i2", lowBit="25 : i32")])
    system = mod("Sys", CLOCKED + [("instr", "i32")], [],
                 [zero, op("three", "hw.constant", [], "i3", value="3 : i3"),
                  op("s1_instr", "hw.wire", ["instr"], "i32", name="s1_instr"),
                  inst("dec", "Dec", {"io_instr": "s1_instr"}, [("io_op", "i2")]),
                  inst("eng", "Eng", {"clock": "clock", "reset": "reset", "io_cmd_valid": "zero", "io_cmd_bits_op": "dec.io_op",
                                      "io_cmd_bits_tag": "three", "io_resp_valid": "mem.io_engResp_valid", "io_resp_bits": "zero"},
                       [("io_req_valid", "i1"), ("io_req_bits_tag", "i3"), ("io_req_bits_data", "i8"), ("io_busy", "i1")]),
                  inst("mem", "Mem", {"clock": "clock", "reset": "reset", "io_eng_valid": "eng.io_req_valid",
                                      "io_eng_bits_row": "eng.io_req_bits_tag", "io_other_valid": "zero"}, [("io_engResp_valid", "i1")])])
    return {"modules": [system, engine, memory, decoder]}


SPEC = {"engine": "toy", "module": "Eng", "system": "Sys", "command": "io_cmd", "limit": 40, "busy": "io_busy",
        "inputs": {"io_cmd_bits_tag": 5},
        "decode": {"input": "io_cmd_bits_op", "instruction": {"instance": "dec", "port": "io_instr"},
                   "words": {"go": 1 << 25, "stop": 2 << 25}},
        "events": {"req": {"bundle": "io_req", "response": {"input": "io_resp_valid", "memory": "Mem"}}},
        "operations": {"go": {"commands": {0: {"io_cmd_valid": 1, "io_cmd_bits_op": "$go", "io_cmd_bits_tag": 2}}}},
        "variants": {"slow": {"events": {"req": {"response": {"latency": 3}}}}}}


def load(value):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{value['engine']}.yaml"
        path.write_text(yaml.safe_dump(value))
        return load_spec(path)


class SyntheticTests(unittest.TestCase):
    def resolve(self, value=SPEC, doc=None):
        return derive.resolve(doc or document(), load(value))

    def test_ports_expand_from_bundles(self):
        spec, _ = self.resolve()
        self.assertEqual(spec["inputs"], {"io_cmd_valid": 0, "io_cmd_bits_op": 0, "io_cmd_bits_tag": 5, "io_resp_valid": 0})
        self.assertEqual(spec["events"]["req"]["valid"], "io_req_valid")
        self.assertEqual(spec["events"]["req"]["fields"], {"tag": "io_req_bits_tag"})  # data depends on io_resp_bits: payload

    def test_explicit_fields_override(self):
        value = copy.deepcopy(SPEC)
        value["events"]["req"]["fields"] = {"t": "io_req_bits_tag"}
        self.assertEqual(self.resolve(value)[0]["events"]["req"]["fields"], {"t": "io_req_bits_tag"})

    def test_missing_bundle_fails(self):
        value = copy.deepcopy(SPEC)
        value["events"]["req"]["bundle"] = "io_absent"
        with self.assertRaisesRegex(ValueError, "exactly one module"):
            self.resolve(value)
        doc = document()
        doc["modules"][1]["ports"][2]["name"] = "io_start"
        with self.assertRaisesRegex(ValueError, "Missing command port"):
            self.resolve(doc=doc)

    def test_response_latency_measured(self):
        spec, notes = self.resolve()
        self.assertEqual(spec["events"]["req"]["response"]["latency"], 1)
        self.assertEqual(notes["scratchpad_read_derivation"]["req"],
                         {"memory": "Sys/mem", "request": "io_eng_valid", "response": "io_engResp_valid",
                          "measured": 1, "memory_read_latency": 1, "used": 1})

    def test_extra_response_register_wins_and_both_are_reported(self):
        spec, notes = self.resolve(doc=document(stages=2))
        self.assertEqual(spec["events"]["req"]["response"]["latency"], 2)
        self.assertEqual({k: notes["scratchpad_read_derivation"]["req"][k] for k in ("measured", "memory_read_latency")},
                         {"measured": 2, "memory_read_latency": 1})

    def test_declared_latency_overrides_and_reports_measured(self):
        spec = load(SPEC)
        resolved, notes = derive.resolve(document(), variant(spec, "slow"))
        self.assertEqual(resolved["events"]["req"]["response"]["latency"], 3)
        self.assertEqual((notes["scratchpad_read_derivation"]["req"]["measured"], notes["scratchpad_read_derivation"]["req"]["used"]), (1, 3))

    def test_response_fails_closed(self):
        cases = {"precedes its read data": document(firmem=(2,)),
                 "mixes memory read latencies": document(firmem=(1, 2))}
        doc = document()
        system = doc["modules"][0]["operations"]
        system[-1]["operands"][-1] = "eng.io_req_valid"  # the request valid also drives io_other_valid
        cases["drives 2"] = doc
        doc = document()
        doc["modules"][0]["operations"][4]["operands"][5] = "zero"  # engine response not driven by the memory
        cases["driven by 0"] = doc
        doc = document()
        doc["modules"][0]["operations"].append(copy.deepcopy(doc["modules"][0]["operations"][-1]) | {"id": "mem2"})
        doc["modules"][0]["operations"][-1]["attributes"] = {**doc["modules"][0]["operations"][-1]["attributes"], "instanceName": "mem2"}
        doc["modules"][0]["operations"][-1]["results"] = ["mem2.io_engResp_valid"]
        cases["2 times"] = doc
        for message, doc in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.resolve(doc=doc)

    def test_codes_decoded_from_instruction_words(self):
        spec, _ = self.resolve()
        self.assertEqual(spec["operations"]["go"]["commands"][0]["io_cmd_bits_op"], 1)
        self.assertEqual(derive.decode(document(), load(SPEC), {}), {"go": 1, "stop": 2})

    def test_decode_fails_closed(self):
        clash = copy.deepcopy(SPEC)
        clash["decode"]["words"]["again"] = 1 << 25
        idle = copy.deepcopy(SPEC)
        idle["decode"]["words"]["nop"] = 0x13
        cases = {"decode to the same code": (clash, document()), "all-zero word": (idle, document())}
        doc = document()
        doc["modules"][0]["operations"][3]["operands"] = ["instr"]  # decoder input driven straight by a port
        cases["no nameable driver"] = (SPEC, doc)
        doc = document()
        decoder = doc["modules"][3]
        decoder["operations"].append(op("late", "seq.firreg", ["op", "clock"], "i2"))
        decoder["ports"][-1]["value"] = "late"
        decoder["ports"].insert(0, {"name": "clock", "direction": "input", "value": "clock", "type": "!seq.clock"})
        doc["modules"][0]["operations"][3]["operands"].insert(0, "clock")
        doc["modules"][0]["operations"][3]["attributes"]["argNames"] = '["clock", "io_instr"]'
        cases["depends on state"] = (SPEC, doc)
        for message, (value, doc) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                derive.decode(doc, load(value), {})

    def test_records_computed_with_derived_inputs(self):
        records = {r["name"]: r for r in extract(document(), load(SPEC))}
        base, slow = records["toy.go"], records["toy.go/slow"]
        self.assertEqual((base["status"], base["events"]["req"]["first_age"], base["first_free_age"]), ("computed", 1, 3))
        self.assertEqual(base["events"]["req"]["values"], {"tag": [2]})
        self.assertEqual(base["assumptions"]["scratchpad_read_latency"], 1)
        self.assertEqual((slow["first_free_age"], slow["assumptions"]["scratchpad_read_latency"]), (5, 3))

    def test_derivation_failure_is_unresolved(self):
        doc = document()
        doc["modules"][0]["operations"][4]["operands"][5] = "zero"
        records = extract(doc, load(SPEC))
        self.assertTrue(all(r["status"] == "unresolved" and "driven by 0" in r["reason"] for r in records))

    def test_modules_include_system(self):
        self.assertEqual(modules(load(SPEC)), ["Eng", "Sys"])

    def test_spec_validation(self):
        cases = {"valid and bundle": lambda s: s["events"]["req"].update(valid="io_req_valid"),
                 "latency or memory": lambda s: s["events"]["req"]["response"].pop("memory"),
                 "need a system": lambda s: s.pop("system"),
                 "absent from decode.words": lambda s: s["operations"]["go"]["commands"][0].update(io_cmd_bits_op="$halt"),
                 "needs codes or decode.words": lambda s: (s.pop("decode"), s["operations"]["go"]["commands"][0].pop("io_cmd_bits_op"),
                                                           s["events"]["req"]["response"].update(latency=1),
                                                           s.update(operation_table={"template": {"commands": {0: {"io_cmd_valid": 1}}}})),
                 "undeclared inputs": lambda s: s["operations"]["go"]["commands"][0].update(other=1)}
        for message, mutate in cases.items():
            value = copy.deepcopy(SPEC)
            mutate(value)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                load(value)

    def test_operation_table_from_decode_words(self):
        value = copy.deepcopy(SPEC)
        value["operations"] = {}
        value["operation_table"] = {"template": {"commands": {0: {"io_cmd_valid": 1, "io_cmd_bits_op": "$code"}}}}
        spec = load(value)
        self.assertEqual(spec["operations"]["stop"]["commands"][0]["io_cmd_bits_op"], "$stop")
        self.assertEqual(derive.resolve(document(), spec)[0]["operations"]["stop"]["commands"][0]["io_cmd_bits_op"], 2)


# Values the Atlas specs declared before derivation (field names were mreg/bank; RTL suffixes now).
DECLARED_CODES = {
    "vpu": {"add": 1, "sub": 2, "mul": 3, "rcp": 4, "sqrt": 5, "sin": 6, "cos": 7, "tanh": 8, "log": 9, "exp": 10, "exp2": 11,
            "square": 12, "cube": 13, "rsum": 14, "csum": 15, "fp8pack": 17, "fp8unpack": 18, "relu": 19, "rmax": 20, "rmin": 21,
            "cmax": 22, "cmin": 23, "pairmax": 24, "pairmin": 25, "mov": 26, "vliOne": 27, "vliCol": 28, "vliRow": 29, "vliAll": 30},
    "mxu0": {"push_weight": 0, "push_acc_fp8": 1, "push_acc_bf16": 2, "pop_acc_fp8": 3, "pop_acc_bf16": 4, "matmul": 5, "matmul_acc": 6},
    "vlsu": {"vload": 1, "vstore": 2},
    "scalar_lsu": {"lw": 3, "seld": 9},
}
DECLARED_CODES["mxu1"] = DECLARED_CODES["mxu0"]
# Codes the specs got wrong before derivation; the RTL never reads them for timing (the records are unchanged).
CORRECTED_CODES = {"xlu": {"vtrpose": 1}, "scalar_lsu": {"sw": 8}}   # declared: 0 (XluEngine ignores op), 0 (store data mask only)
DECLARED_INPUTS = {
    "xlu": {"io_cmd_valid", "io_cmd_bits_op", "io_cmd_bits_srcMregId", "io_cmd_bits_dstMregId", "io_mregReadResp_valid"},
    "vpu": {"io_cmd_valid", "io_cmd_bits_op", "io_cmd_bits_vs1", "io_cmd_bits_vs2", "io_cmd_bits_vd",
            "io_mregReadResp0_valid", "io_mregReadResp1_valid"},
    "vlsu": {"io_cmd_valid", "io_cmd_bits_op", "io_cmd_bits_mregBank", "io_cmd_bits_vmemLineAddr", "io_vmemVecReadData_valid",
             "io_mregReadResp_valid"},
    "mxu0": {"io_cmd_valid", "io_cmd_bits_op", "io_cmd_bits_mregId", "io_cmd_bits_accSel", "io_cmd_bits_weightSlot",
             "io_mregReadResp0_valid", "io_mregReadResp1_valid"},
    "scalar_lsu": {"issueScalarLoad", "issueScalarStore", "hostStart", "mem_cmd", "io_vmemScalarReadData_valid"},
}
DECLARED_INPUTS["mxu1"] = DECLARED_INPUTS["mxu0"]
# Declared fields per bundle (previously named mreg/bank/row); payload bits are excluded, scalar ports had none.
DECLARED_FIELDS = {"io_vmemVecRead": {"bankIdx", "bankAddr"}, "io_vmemVecWrite": {"bankIdx", "bankAddr"},
                   "io_vmemScalarRead": set(), "io_vmemScalarWrite": set(), "io_scalarResp": set()}


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class AtlasDerivationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_target("atlas")
        cls.document = load_document(HW_IR, sorted({m for s in cls.specs.values() for m in modules(s)}), EXPORTER)
        cls.resolved = {name: derive.resolve(cls.document, spec) for name, spec in cls.specs.items()}

    def test_response_latency_is_one_cycle(self):
        for name, (spec, notes) in self.resolved.items():
            for group, event in spec["events"].items():
                if "response" in event:
                    with self.subTest(f"{name}.{group}"):
                        note = notes["scratchpad_read_derivation"][group]
                        self.assertEqual((note["measured"], note["memory_read_latency"], event["response"]["latency"]), (1, 1, 1))

    def test_codes_match_declared_tables(self):
        for name, spec in self.specs.items():
            with self.subTest(name):
                codes = derive.decode(self.document, spec, {})
                self.assertEqual(codes, {**DECLARED_CODES.get(name, {}), **CORRECTED_CODES.get(name, {})})

    def test_ports_cover_declared_lists(self):
        for name, (spec, _) in self.resolved.items():
            with self.subTest(name):
                extra = set(spec["inputs"]) - DECLARED_INPUTS[name]
                self.assertLessEqual(DECLARED_INPUTS[name], set(spec["inputs"]))
                self.assertTrue(all(i.startswith(f"{spec['command']}_bits_") for i in extra), extra)

    def test_bundle_fields_are_the_declared_control_fields(self):
        for name, spec in self.specs.items():
            for group, event in spec["raw"]["events"].items():
                if "bundle" in event:
                    fields = self.resolved[name][0]["events"][group]["fields"]
                    with self.subTest(f"{name}.{group}"):
                        self.assertEqual(set(fields), DECLARED_FIELDS.get(event["bundle"], {"mregId", "row"}))
                        self.assertTrue(all(signal == f"{event['bundle']}_bits_{field}" for field, signal in fields.items()))


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
    unittest.main()
