"""The reference coverage audit names the rule for every source node and every refusal in one walk."""

import contextvars
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

import numpy as np

from atlas_mlir_native.reference_eval import EvaluationError, device, evaluate
from atlas_mlir_native.reference_eval import __main__ as cli
from atlas_mlir_native.reference_eval.coverage import Audit, audit
from atlas_mlir_native.reference_eval.decisions import SelectedDecisions
from atlas_mlir_native.reference_eval.evaluator import _AUDIT

_RULES = {"execution": "host", "host_order": {"default": "lexicographic-f32"}}


def _ref(node_id):
    return {"node_id": node_id, "value_id": node_id + ":v0"}


def _node(node_id, target, args, dtype, shape):
    return {"id": node_id, "op": "call_function", "target": target, "args": args, "kwargs": {},
            "results": [{"id": node_id + ":v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}


def _decisions(placed):
    decisions = object.__new__(SelectedDecisions)
    decisions._by_node = {node: [{"id": "r_" + node, "placement": "host", "implementation": "source_float_arithmetic"}]
                          for node in placed}
    decisions._bindings, decisions._softmax, decisions.record = {}, {}, {"applied": {}}
    return decisions


def _graph():
    x = {"id": "x", "op": "placeholder", "target": "x", "input_kind": "USER_INPUT",
         "results": [{"id": "x:v0", "dtype": "float32", "shape": [4], "device": "cpu"}]}
    return [x,
            _node("a", "aten.mul.Tensor", [_ref("x"), 2.0], "float32", (4,)),             # placed: host rule
            _node("ones", "aten.ones.default", [[4]], "float32", (4,)),                   # compile-time constant
            _node("c", "aten.unknown_op.default", [_ref("x")], "float32", (4,)),          # no handler
            _node("d", "aten.mul.Tensor", [_ref("c"), 3.0], "float32", (4,)),             # no placement
            _node("e", "aten.to.device", [_ref("a"), "cpu", "torch.int64"], "int64", (4,)),  # placed; host rule refuses
            _node("f", "aten.to.device", [_ref("x"), "cpu", "torch.int64"], "int64", (4,)),  # no placement, nor host rule
            {"id": "out", "op": "output", "target": "output", "args": [_ref("a")], "kwargs": {}, "results": []}]


class CoverageAuditTest(TestCase):
    def test_every_refusal_is_listed_at_once_with_its_kind_and_every_other_node_its_rule(self):
        x = {"x": np.arange(1, 5, dtype=np.float32).view(np.uint32)}
        with self.assertRaisesRegex(EvaluationError, "source node c"):
            evaluate(_graph(), {}, x, _RULES, decisions=_decisions({"a", "e"}))
        report = audit(_graph(), {}, x, _RULES, decisions=_decisions({"a", "e"}))
        rows = {row["id"]: row for row in report["per_node"]}
        self.assertEqual(sorted(node for node, row in rows.items() if row["status"] == "refused"), ["c", "d", "e", "f"])
        self.assertEqual({node: rows[node]["kind"] for node in "cdef"},
                         {"c": "handler", "d": "placement", "e": "conversion", "f": "placement"})
        self.assertEqual(rows["f"]["host_rule"], "unsupported source dtype conversion f32 to i64")
        self.assertEqual(rows["d"]["host_rule"], "evaluates")
        self.assertEqual((rows["x"]["rules"], rows["a"]["rules"], rows["ones"]["rules"], rows["out"]["rules"]),
                         (["input"], ["host"], ["host"], ["output"]))
        self.assertEqual(rows["ones"]["decision"], "compile_time_constant")
        self.assertEqual(rows["e"]["signature"], {"operands": ["f32"], "scalars": 0, "results": ["i64"], "form": "out-of-place"})
        self.assertEqual((report["nodes"], report["covered"], report["refused"], report["missing_rules"]), (8, 4, 4, 4))
        self.assertEqual(sum(group["count"] for group in report["groups"]), 4)

    def test_the_audit_leaves_an_ordinary_evaluation_unchanged(self):
        x = {"x": np.arange(1, 5, dtype=np.float32).view(np.uint32)}
        graph = [node for node in _graph() if node["id"] in {"x", "a", "out"}]
        plain = evaluate(graph, {}, x, _RULES, decisions=_decisions({"a"}))
        report = audit(graph, {}, x, _RULES, decisions=_decisions({"a"}))
        np.testing.assert_array_equal(plain.outputs.raw.view(np.float32), np.arange(2, 10, 2, dtype=np.float32))
        self.assertEqual((report["refused"], report["covered"]), (0, 3))
        evaluate(graph, {}, x, _RULES, decisions=_decisions({"a"}))  # no audit state outlives its walk


def _device_rules():
    root = os.environ.get("ATLAS_NPU_MODEL_ROOT")
    if not root:
        raise AssertionError("ATLAS_NPU_MODEL_ROOT must name the pinned model; qualification never skips")
    return {"npu_model_revision": device.MODEL_REVISION, "numerics": "rtl", "selected_unit": "mxu1",
            "k_order": "ascending-k32", "npu_model_root": root}


def _qlinear_graph(output=True):
    def placeholder(name, dtype, shape):
        return {"id": name, "op": "placeholder", "target": name, "input_kind": "USER_INPUT",
                "results": [{"id": name + ":v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}
    graph = [placeholder("a", "uint8", (2, 3)), placeholder("ax", "int32", ()), placeholder("b", "uint8", (4, 3)),
             placeholder("bx", "int32", ()),
             _node("y", "atlas.qlinear.default", [_ref("a"), _ref("ax"), _ref("b"), _ref("bx"), None, "bfloat16",
                                                  "atlas_e4m3"], "bfloat16", (2, 4))]
    return graph + ([{"id": "out", "op": "output", "target": "output", "args": [_ref("y")], "kwargs": {}, "results": []}]
                    if output else [])


_QLINEAR = {"a": np.array([[0x38, 0x18, 0x40], [0x30, 0x38, 0x20]], np.uint8), "ax": np.array(1, np.int32),
            "b": np.array([[0x38, 0x18, 0x18], [0x40, 0x38, 0x30], [0x28, 0x20, 0x38], [0x38, 0x38, 0x38]], np.uint8),
            "bx": np.array(-2, np.int32)}


class AuditIsolationTest(TestCase):
    def plain(self):
        return evaluate(_qlinear_graph(), {}, _QLINEAR, _device_rules()).outputs.raw

    def test_device_arithmetic_runs_unaudited_after_inside_and_after_a_failing_audit(self):
        expected = self.plain()
        self.assertTrue(np.any(expected != 0x3F80))  # the audit stand-in is all 0x3F80; the real readout is not
        report = audit(_qlinear_graph(), {}, _QLINEAR, _device_rules())
        rows = {row["id"]: row for row in report["per_node"]}
        self.assertEqual((rows["y"]["rules"], rows["y"]["device_numerics"], report["refused"]), (["device"], "stand-in", 0))
        self.assertIsNone(_AUDIT.get())
        np.testing.assert_array_equal(self.plain(), expected)
        # An ordinary evaluation inside an active audit computes real arithmetic and records nothing in that audit.
        outer = Audit()

        def nested():
            _AUDIT.set(outer)
            return self.plain()
        np.testing.assert_array_equal(contextvars.copy_context().run(nested), expected)
        self.assertEqual(outer.rows, [])
        # An audit that raises (no output node) still leaves no audit active.
        with self.assertRaisesRegex(EvaluationError, "no output node"):
            audit(_qlinear_graph(output=False), {}, _QLINEAR, _device_rules())
        self.assertIsNone(_AUDIT.get())
        np.testing.assert_array_equal(self.plain(), expected)
        # No stand-in leaves an audit, and an audit takes no observer.
        self.assertIsNone(evaluate(_qlinear_graph(), {}, _QLINEAR, _device_rules(), audit=Audit()).outputs)
        with self.assertRaisesRegex(TypeError, "no observer"):
            evaluate(_qlinear_graph(), {}, _QLINEAR, _device_rules(), audit=Audit(), observer=lambda *args: None)

    def test_the_bypass_checks_shapes_exactly_as_the_contraction_does(self):
        unit = device._rules(_device_rules(), contraction=True, k_extent=32)
        codes = lambda shape: np.full(shape, 0x38, np.uint8)
        for a_shape, b_shape in (((3, 32), (2, 5, 32)), ((2, 1, 3, 32), (4, 5, 32))):  # right operand adds batches
            with self.subTest(a=a_shape, b=b_shape):
                real = device._contraction(codes(a_shape), np.int32(0), codes(b_shape), np.int32(0), unit)
                recorder = Audit()
                recorder.begin({"id": "y", "target": "atlas.qk_matmul.default"})
                token = _AUDIT.set(recorder)
                try:
                    stand_in = device._contraction(codes(a_shape), np.int32(0), codes(b_shape), np.int32(0), unit)
                finally:
                    _AUDIT.reset(token)
                self.assertEqual(stand_in.shape, real.shape)
        for a_shape, b_shape in (((2, 3, 32), (3, 5, 32)),):
            for audited in (False, True):
                with self.subTest(audited=audited), self.assertRaisesRegex(ValueError, "do not broadcast"):
                    recorder = Audit()
                    recorder.begin({"id": "y", "target": "atlas.qk_matmul.default"})
                    token = _AUDIT.set(recorder if audited else None)
                    try:
                        device._contraction(codes(a_shape), np.int32(0), codes(b_shape), np.int32(0), unit)
                    finally:
                        _AUDIT.reset(token)


class AuditCompletenessTest(TestCase):
    def test_a_refused_output_is_recorded_and_stand_in_dependence_is_transitive(self):
        x = {"x": np.arange(1, 5, dtype=np.float32).view(np.uint32)}
        graph = _graph()[:4]  # x, a, ones, and c, which has no handler
        graph += [_node("v", "aten.view.default", [_ref("c"), [4]], "float32", (4,)),
                  # Read only by value id, through the view, from the stand-in for c.
                  _node("b", "aten.bucketize.Tensor", [_ref("x"), {"value_id": "v:v0"}], "int64", (4,)),
                  {"id": "out", "op": "output", "target": "output", "args": [{"node_id": "missing", "value_id": "missing:v0"}],
                   "kwargs": {}, "results": []}]
        report = audit(graph, {}, x, _RULES, decisions=_decisions({"a", "b"}))
        rows = {row["id"]: row for row in report["per_node"]}
        self.assertEqual((rows["out"]["status"], rows["out"]["kind"]), ("refused", "rule"))
        self.assertEqual((rows["b"]["status"], rows["b"]["kind"], rows["b"]["value_dependent"]), ("refused", "rule", True))
        self.assertEqual(rows["v"]["status"], "covered")
        self.assertNotEqual(report["refused"], 0)

    def test_the_cli_audit_writes_no_expected_archive_and_never_reports_expected(self):
        x = {"x": np.arange(1, 5, dtype=np.float32).view(np.uint32)}
        cases = {"COVERED": [node for node in _graph() if node["id"] in {"x", "ones", "out2"}] + [
                     {"id": "out", "op": "output", "target": "output", "args": [_ref("ones")], "kwargs": {}, "results": []}],
                 "GAPS": _graph()}
        for status, graph in cases.items():
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory) / "audit"
                argv = ["reference_eval", directory, "--capture", "--receipt-sha256", "0" * 64, "--audit", "--out", str(dest)]
                with mock.patch.object(cli, "capture_sources", return_value=(Path(directory), graph, {"artifacts": {
                            "frontend-trace.json": "0" * 64}})), \
                        mock.patch.object(cli, "safetensor_constants", return_value={}), \
                        mock.patch.object(cli, "capture_inputs", return_value=x), \
                        mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                    code = cli.main()
                record = json.loads((dest / "reference.json").read_text())
                self.assertEqual((record["status"], code), (status, 0 if status == "COVERED" else 1))
                self.assertFalse((dest / "expected.npz").exists())
                self.assertNotIn("expected_sha256", record)
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "audit"
            with mock.patch.object(sys, "argv", ["reference_eval", directory, "--audit", "--out", str(dest)]), \
                    mock.patch("builtins.print"):
                self.assertEqual(cli.main(), 1)
            self.assertEqual(json.loads((dest / "reference.json").read_text())["reason"], "--audit requires --capture")


if __name__ == "__main__":
    main()
