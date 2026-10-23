"""Fail-closed curve extraction and independent native DMA validation."""
from copy import copy
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_curve as curve
import atlas_model_schema as schema
from model_evidence import isolate_model_imports


class CurveContractTest(unittest.TestCase):
    def test_prohibited_paths_rejected_before_any_filesystem_access(self):
        with patch.object(Path, "is_symlink", side_effect=AssertionError("accessed prohibited path")):
            for name in ("/tmp/HAMMER/curve.json", "/tmp/vLsI/../curve.json"):
                with self.assertRaisesRegex(curve.UnsupportedCurve, "prohibited"):
                    curve._safe(name)

    def test_symlink_ancestor_rejected_before_descendant_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "link"
            link.symlink_to(Path(directory) / "unvisited")
            with self.assertRaisesRegex(curve.UnsupportedCurve, "symlink"):
                curve._safe(link / "curve.json")

    def test_old_lock_without_dma_contract_remains_loadable(self):
        value = {k: v for k, v in schema.LOCK.items() if k not in {"curve_sha256", "dma_source_sha256"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(value))
            self.assertEqual(schema.load_lock(path), value)
            value["curve_sha256"] = "bad"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "curve_sha256"):
                schema.load_lock(path)

    def test_generated_cpp_identity_covers_curve_source_and_ast_pins(self):
        generated = schema.generate(schema.VOCABULARY)
        self.assertIn("&dmaIdentity()", generated)
        self.assertIn(schema.LOCK["curve_sha256"], generated)
        for name, digest in schema.LOCK["dma_source_sha256"].items():
            self.assertIn(digest, generated)
            self.assertIn('"shape:' + name + '"', generated)
            self.assertIn(schema.SOURCE_SHAPES[name], generated)


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native curve contracts")
class NativeCurveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        from atlas_model_timing import configuration
        cls.cfg, _ = configuration()

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

    def test_scaling_changes_source_converted_facts(self):
        original = curve.derive_curve(self.cfg)
        scaled = copy(self.cfg)
        scaled.dma_memory_params = dict(self.cfg.dma_memory_params, channels=2, frequency_ghz=1.0)
        changed = curve.derive_curve(scaled)
        self.assertEqual(changed["curve_sha256"], original["curve_sha256"])
        for first, second in zip(original["curves"], changed["curves"]):
            self.assertEqual(second["cycles_per_beat"], first["cycles_per_beat"])
            self.assertEqual(second["unloaded_latency_cycles"], first["unloaded_latency_cycles"] * 2)

    def test_unknown_modes_or_malformed_curve_rejected(self):
        for values in ({"latency_mode": "mess"}, {"latency_mode": "knee"},
                       {"converge": 0.5}, {"enforce_peak": False}, {"curves": {}}, {"beat_bytes": 16},
                       {"min_latency": 0}, {"min_latency": -1}):
            cfg = copy(self.cfg)
            cfg.dma_memory_params = dict(self.cfg.dma_memory_params, **values)
            with self.subTest(values=values), self.assertRaises(curve.UnsupportedCurve):
                curve.derive_curve(cfg)
        cfg = copy(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text('{"latencyMode": "unloaded", "curves": {}}\n')
            cfg.dma_memory_params = dict(self.cfg.dma_memory_params, curves=str(path))
            with self.assertRaisesRegex(curve.UnsupportedCurve, "unsupported capability: curve.curves"):
                curve.derive_curve(cfg)

    def test_frequency_queue_and_history_are_independent_configuration_inputs(self):
        from atlas_validate_curve import FactBackend, _stream
        from npu_model.hardware.memory_backend import make_memory_backend
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
        stores = [False] * 11 + [True] * 17 + [False, True] * 9
        self.assertEqual(_stream(FactBackend(facts), stores, stalls=True),
                         _stream(make_memory_backend(cfg), stores, stalls=True))

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

    def test_independent_reference_detects_timing_tamper(self):
        from atlas_validate_curve import FactBackend, _stream
        from npu_model.hardware.memory_backend import make_memory_backend
        facts = curve.derive_curve(self.cfg)
        native = _stream(make_memory_backend(self.cfg), [False] * 65)
        self.assertEqual(native, _stream(FactBackend(facts), [False] * 65))
        facts["curves"][-1]["cycles_per_beat"] *= 1.01
        self.assertNotEqual(native, _stream(FactBackend(facts), [False] * 65))

    def test_bound_covers_every_history_curve_and_extracts_wait_visibility(self):
        import math
        facts = curve.derive_curve(self.cfg)
        contract = facts["proposal_contract"]
        maximum = max(facts["min_latency"], *(r["unloaded_latency_cycles"] for r in facts["curves"]),
                      *(r["cycles_per_beat"] for r in facts["curves"]))
        self.assertEqual(contract["per_beat_bound_cycles"], math.ceil(maximum) + facts["issue_interval_cycles"])
        self.assertEqual(contract["response_to_retire_cycles"], 0)
        self.assertEqual(contract["retire_to_wait_issue_cycles"], 1)
        self.assertTrue(contract["preconditions"]["engine_backend_drained_at_launch"])
        self.assertEqual(contract["preconditions"]["maximum_cycle"], 2147483647)
        self.assertEqual(contract["policy"]["classification"], "compiler_policy")

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

    def test_core_positive_and_one_short_wait_edges(self):
        from atlas_validate_curve import _core_wait_observation
        for store in (False, True):
            early = _core_wait_observation(self.cfg, store=store, beats=1)
            age = early["retired_cycle"] - early["launch_cycle"]
            short = _core_wait_observation(self.cfg, store=store, beats=1, wait_age=age)
            positive = _core_wait_observation(self.cfg, store=store, beats=1, wait_age=age + 1)
            self.assertEqual(short["wait_checks"][0], dict(cycle=short["retired_cycle"], fired=False, stall="dma.wait"))
            self.assertEqual(positive["wait_checks"], [dict(cycle=positive["retired_cycle"] + 1, fired=True, stall=None)])

    def test_mixed_queued_grant_denials_preserve_native_backend_parity(self):
        from atlas_validate_curve import _mixed_engine_observation
        facts = curve.derive_curve(self.cfg)
        native = _mixed_engine_observation(self.cfg, facts, deny_grants=True)
        self.assertEqual(native, _mixed_engine_observation(self.cfg, facts, fact_backend=True, deny_grants=True))
        denied = [access for event in native["trace"] for access in (event["vmem_read"], event["vmem_write"])
                  if access is not None and not access[2]]
        self.assertTrue(denied)

    def test_near_epsilon_latency_bound_uses_ceiling_before_issue_interval(self):
        from atlas_validate_curve import _engine_observation
        cfg = copy(self.cfg)
        cfg.dma_max_in_flight = 1
        # Construct selected curve bytes in native units; use the converter's
        # resulting values, never observations, to build the bound.
        data = json.loads(curve.resolve_curve_path(cfg).read_text())
        for points in data["curves"].values():
            for point in points:
                point[0] = 1000000.0
                point[1] = 1.000000001 / float(cfg.dma_memory_params.get("frequency_ghz", data.get("frequencyGHz", 1.0)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epsilon.json"
            path.write_text(json.dumps(data))
            cfg.dma_memory_params = dict(cfg.dma_memory_params, curves=str(path), min_latency=1)
            facts = curve.derive_curve(cfg)
            self.assertEqual(facts["proposal_contract"]["per_beat_bound_cycles"], 3)
            native = _engine_observation(cfg, store=False, beats=4, history=[False] * 31)
            self.assertLessEqual(native["retired_cycle"] - native["launch_cycle"],
                                 curve.proposal_duration(facts, store=False, beats=4))


if __name__ == "__main__":
    unittest.main()
