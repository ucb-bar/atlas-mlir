"""Fail-closed curve extraction from source and configuration."""
from copy import copy
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_curve as curve
import atlas_model_schema as schema
from model_evidence import NativeModelFixture


class CurveContractTest(unittest.TestCase):
    def test_knee_empty_curves_and_nonpositive_latency_refuse(self):
        cfg = SimpleNamespace(dma_beat_bytes=64)
        params, data = {"latency_mode": "unloaded"}, {"curves": {"100": [[100, 20]]}}
        curve._check_curve_configuration(cfg, params, data)
        for field, value in (("latency_mode", "knee"), ("min_latency", 0), ("min_latency", -1)):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    curve.UnsupportedCurve, "unsupported capability: dma_memory_params\\." + field):
                curve._check_curve_configuration(cfg, dict(params, **{field: value}), data)
        with self.assertRaisesRegex(curve.UnsupportedCurve, "unsupported capability: curve\\.curves"):
            curve._check_curve_configuration(cfg, params, {"curves": {}})


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native curve contracts")
class NativeCurveTest(NativeModelFixture, unittest.TestCase):
    def test_derivation_allocates_no_architecture_and_issues_no_beats(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.dma import DmaExecutionUnit
        from npu_model.hardware.memory_backend import CurveMemoryBackend
        with patch.object(ArchState, "__init__", side_effect=AssertionError("allocated state")), \
                patch.object(DmaExecutionUnit, "tick", side_effect=AssertionError("ticked DMA")), \
                patch.object(CurveMemoryBackend, "issue", side_effect=AssertionError("issued beat")):
            facts = curve.derive_curve(self.cfg)
        self.assertEqual(facts["curve_sha256"], schema.LOCK["curve_sha256"])
        self.assertEqual(facts["source_sha256"], schema.LOCK["dma_source_sha256"])
        shared = facts["source_contract"]
        self.assertEqual(shared["source_shapes"], schema.DMA_SOURCE_SHAPES)
        self.assertEqual(shared["edge_order"], facts["proposal_contract"]["source_evidence"]["edge_order"])
        self.assertEqual(shared["core_tick_shape"], facts["proposal_contract"]["source_evidence"]["core_tick_shape"])
        self.assertTrue(facts["estimate_only"])
        self.assertEqual(facts["semantics"]["address_dependence"], "none")

    def test_frequency_queue_and_history_are_independent_configuration_inputs(self):
        original = curve.derive_curve(self.cfg)
        cfg = copy(self.cfg)
        cfg.dma_memory_params = dict(self.cfg.dma_memory_params, frequency_ghz=0.75,
                                     queue_depth=13, window_beats=7)
        facts = curve.derive_curve(cfg)
        self.assertEqual(facts["queue_depth"], 13)
        self.assertEqual(facts["window_beats"], 7)
        self.assertEqual(facts["max_in_flight"], self.cfg.dma_max_in_flight)
        for first, second in zip(original["curves"], facts["curves"]):
            self.assertAlmostEqual(second["cycles_per_beat"], first["cycles_per_beat"] * 1.5)
            self.assertAlmostEqual(second["unloaded_latency_cycles"], first["unloaded_latency_cycles"] * 1.5)

    def test_selected_curve_bytes_change_facts_and_bind_their_digest(self):
        from atlas_model_schema import sha256
        original = curve.derive_curve(self.cfg)
        data = json.loads(curve.resolve_curve_path(self.cfg).read_text())
        for points in data["curves"].values():
            for point in points:
                point[0] *= 0.8
                point[1] *= 1.25
        cfg = copy(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text(json.dumps(data))
            cfg.dma_memory_params = dict(self.cfg.dma_memory_params, curves=str(path))
            changed = curve.derive_curve(cfg)
            self.assertEqual(changed["curve_sha256"], sha256(path.read_bytes()))
        self.assertNotEqual(changed["curve_sha256"], original["curve_sha256"])
        for first, second in zip(original["curves"], changed["curves"]):
            self.assertAlmostEqual(second["cycles_per_beat"], first["cycles_per_beat"] / 0.8)
            self.assertAlmostEqual(second["unloaded_latency_cycles"], first["unloaded_latency_cycles"] * 1.25)

    def test_controller_capability_refusals_identify_the_field(self):
        for field, value in (("latency_mode", "mess"), ("window_beats", 0),
                             ("window_beats", 1.5), ("queue_depth", False),
                             ("frequency_ghz", 0), ("converge", 0.5),
                             ("enforce_peak", False), ("beat_bytes", 16),
                             ("unknown_knob", 2)):
            cfg = copy(self.cfg)
            cfg.dma_memory_params = dict(self.cfg.dma_memory_params, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(
                    curve.UnsupportedCurve, "unsupported capability: dma_memory_params\\." + field):
                curve.derive_curve(cfg)

    def test_unreviewed_source_mutation_rejected(self):
        from npu_model.hardware import memory_backend
        path = Path(memory_backend.__file__)
        source = path.read_text()
        mutated = source.replace("self._last_ready = 0.0", "self._last_ready = 1.0")
        self.assertNotEqual(source, mutated)
        with patch.object(curve, "_read", return_value=mutated):
            with self.assertRaisesRegex(curve.UnsupportedCurve, "source semantics or bytes"):
                curve.derive_curve(self.cfg)

    def test_core_order_and_wait_predicate_mutations_refuse(self):
        original_read = curve._read
        for before, after in (("target = self._tick_s1()", "target = None"),
                              ("and state.check_flag(insn.funct3):", "and not state.check_flag(insn.funct3):")):
            def mutated(path):
                source = original_read(path)
                if Path(path).name == "core.py":
                    self.assertIn(before, source)
                    return source.replace(before, after)
                return source
            with self.subTest(before=before), patch.object(curve, "_read", side_effect=mutated):
                with self.assertRaisesRegex(curve.UnsupportedCurve, "completion visibility"):
                    curve.derive_curve(self.cfg)

    def test_source_bound_refuses_unknown_ready_expression_independently_of_pin(self):
        from npu_model.hardware import core, dma, memory_backend
        facts = curve.derive_curve(self.cfg)
        backend = Path(memory_backend.__file__).read_text().replace(
            "self._last_ready + 1.0 / peak_bw", "self._last_ready + 2.0 / peak_bw")
        with self.assertRaisesRegex(curve.UnsupportedModelTiming, "DMA proof premise"):
            curve._proposal_contract(Path(core.__file__).read_text(), ast.parse(Path(dma.__file__).read_text()),
                                     ast.parse(backend), facts["curves"], facts["min_latency"],
                                     facts["rounding_epsilon_cycles"], facts["issue_interval_cycles"])

if __name__ == "__main__":
    unittest.main()
