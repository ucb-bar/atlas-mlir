"""Selected-model identity, decoding and execution failures must fail closed."""
from copy import deepcopy
import json
import pickle
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_scorer import ModelScorer, bind_curve_configuration, model_instruction, selected_model, _model_content
from atlas_model_profiles import PeriodicAcceptance
from atlas_model_timing import configuration, configuration_values, encoded, sha256
from check_program_decode import decode_binary
from model_evidence import isolate_model_imports


class ScorerGuardsTest(unittest.TestCase):
    def test_assertions_cannot_be_disabled(self):
        with patch("atlas_model_scorer.sys.flags") as flags:
            flags.optimize = 1
            with self.assertRaisesRegex(ValueError, "assertions"):
                ModelScorer(Path("/tmp"))({})

    def test_each_model_identity_component_is_required(self):
        keys = ("source_sha256", "numerical_data_sha256", "configuration_sha256")
        for key in keys:
            evidence = dict(zip(keys, ("source", "data", "config")))
            evidence[key] = "changed"
            with self.subTest(key=key), patch("atlas_model_scorer.selected_model",
                                             return_value=(Path("/tmp"), "source", "data", "config", None)):
                with self.assertRaisesRegex(ValueError, "differs from provider evidence"):
                    ModelScorer(Path("/tmp"))({"model_evidence": evidence})

    def test_configuration_identity_and_execution_use_one_instance(self):
        cfg, values = object(), {"timing": 0.75}
        factory = Mock(return_value=(cfg, values))
        with patch("atlas_model_scorer._model_content", return_value=(Path("/tmp"), "source", "data")), \
                patch("atlas_model_scorer.configuration", factory), \
                patch("atlas_model_scorer.check_configuration") as check:
            identity = selected_model(Path("/tmp"), include_configuration=True)
        factory.assert_called_once_with()
        check.assert_called_once_with(cfg)
        self.assertEqual(identity[:4], (Path("/tmp"), "source", "data", sha256(encoded(values))))
        self.assertIs(identity[4], cfg)

    def test_periodic_acceptance_validates_and_preserves_absolute_phase_when_pickled(self):
        acceptance = PeriodicAcceptance(7, 3)
        restored = pickle.loads(pickle.dumps(acceptance))
        self.assertEqual([restored(cycle) for cycle in range(20, 40)],
                         [cycle % 7 < 3 for cycle in range(20, 40)])
        for period, prefix in ((0, 0), (-1, 0), (3, -1), (3, 4), (True, 1), (3, 1.0)):
            with self.subTest(period=period, prefix=prefix), self.assertRaises(ValueError):
                PeriodicAcceptance(period, prefix)

    def test_selected_curve_bytes_are_bound_before_execution(self):
        raw = b'{"curves":{"100":[[100,20]]},"frequencyGHz":0.75}\n'
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            selected.write_bytes(raw)
            cfg = SimpleNamespace(dma_memory_params={"curves": str(selected), "frequency_ghz": 0.75})
            with patch("atlas_model_scorer.resolve_curve_path", return_value=selected) as resolve:
                bind_curve_configuration(cfg, {"dma_curve_sha256": sha256(raw)})
            resolve.assert_called_once_with(cfg)
            self.assertEqual(cfg.dma_memory_params, {"curves": json.loads(raw), "frequency_ghz": 0.75})

    def test_curve_identity_mismatch_and_legacy_omission_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            selected.write_text('{"curves":{}}\n')
            cfg = SimpleNamespace(dma_memory_params={"curves": str(selected)})
            for evidence in ({"dma_curve_sha256": "changed"}, {}):
                with self.subTest(evidence=evidence), patch("atlas_model_scorer.resolve_curve_path", return_value=selected):
                    with self.assertRaisesRegex(ValueError, "DMA curve differs from provider evidence"):
                        bind_curve_configuration(cfg, evidence)
                self.assertEqual(cfg.dma_memory_params["curves"], str(selected))


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for model execution")
class ModelScorerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        isolate_model_imports(cls, str(root))
        _model_content.cache_clear()
        cls.addClassCleanup(_model_content.cache_clear)
        _, source, data, config = selected_model(root)
        cls.scorer = ModelScorer(root)
        cls.evidence = dict(source_sha256=source, numerical_data_sha256=data, configuration_sha256=config)
        cls.inventory = json.loads((Path(__file__).parents[1] / "docs/selected-variant-inventory.json").read_text())

    def program(self, words):
        return dict(model_evidence=self.evidence, instructions=[dict(word_u32=word) for word in words])

    def test_shift_immediate_decodes_without_function_bits(self):
        from npu_model.configs import isa_definition as isa
        for word in (0x40115093, 0x00115093, 0x00111093):
            with self.subTest(word=word):
                row = decode_binary(self.inventory, word.to_bytes(4, "little"))["instructions"][0]
                instruction = model_instruction(row, self.inventory, isa)
                self.assertEqual(instruction.to_bytecode(), word)
                self.assertEqual(instruction.imm & 31, 1)

    def test_cold_cycles_profiles_and_exhaustion(self):
        # ADDI x1, x0, 1; CSRRW x0, completion, x1; ECALL.
        program = self.program([0x00100093, 0xc1009073, 0x00000073])
        for profile in ("fifo43", "curve"):
            self.assertEqual(self.scorer(program, profile), 3)
        with self.assertRaisesRegex(ValueError, "exhausted before ECALL"):
            self.scorer(self.program([0x00100093]))

    def test_nondefault_curve_filename_scaling_and_timing_configuration(self):
        cfg, _ = configuration()
        from atlas_derive_curve import resolve_curve_path
        raw = resolve_curve_path(cfg).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            selected.write_bytes(raw)

            def selected_configuration():
                cfg, _ = configuration()
                cfg.vpu_simple_op_latency_cycles = 6
                cfg.dma_memory_params = dict(cfg.dma_memory_params, curves=str(selected), frequency_ghz=0.75)
                return cfg, configuration_values(cfg)

            with patch("atlas_model_scorer.configuration", side_effect=selected_configuration):
                _, source, data, config = selected_model(self.scorer.model)
                evidence = dict(source_sha256=source, numerical_data_sha256=data,
                                configuration_sha256=config, dma_curve_sha256=sha256(raw))
                program = dict(self.program([0x00100093, 0xc1009073, 0x00000073]), model_evidence=evidence)
                self.assertEqual(self.scorer(program), 3)
                selected.write_bytes(raw + b"\n")
                with self.assertRaisesRegex(ValueError, "DMA curve differs from provider evidence"):
                    self.scorer(program)

    def test_fifo43_native_backend_matches_previous_callable_trajectories(self):
        from atlas_validate_curve import _stream
        from npu_model.hardware.memory_backend import make_memory_backend
        old, _ = configuration()
        # Historical behavior is an independent regression oracle, not profile input.
        old.dma_memory_backend = "fixed"
        old.dma_memory_params = dict(latency=43, cycles_per_beat=1, accept_fn=lambda cycle: cycle % 11 < 4)
        current, receipt = self.scorer.configured_profile(self.program([]), "fifo43")
        self.assertEqual(current.dma_memory_backend, "fixed")
        self.assertEqual(receipt["provenance"], "synthetic-rtl-harness-target")
        for stores in ([False] * 65, [True] * 65, [False, True] * 65):
            for gap, stalls in ((0, False), (0, True), (13, True)):
                with self.subTest(stores=stores[:4], gap=gap, stalls=stalls):
                    before, after = make_memory_backend(old), make_memory_backend(current)
                    expected = _stream(before, stores, gap=gap, stalls=stalls)
                    self.assertEqual(expected, _stream(after, stores, gap=gap, stalls=stalls))
                    after.reset()
                    self.assertEqual(expected, _stream(after, stores, gap=gap, stalls=stalls))

    def test_profile_receipts_bind_effective_inputs_without_changing_base_identity(self):
        document = self.program([])
        receipts = {profile: self.scorer.profile_receipt(document, profile) for profile in ("curve", "fifo43")}
        for profile, receipt in receipts.items():
            payload = dict(receipt)
            self.assertEqual(payload.pop("sha256"), sha256(encoded(payload)))
            self.assertEqual(json.loads(json.dumps(receipt)), receipt)
            self.assertEqual(receipt["model_identity"], self.evidence)
            self.assertEqual(receipt, pickle.loads(pickle.dumps(self.scorer)).profile_receipt(document, profile))
        self.assertNotEqual(receipts["curve"]["sha256"], receipts["fifo43"]["sha256"])
        self.assertIsInstance(receipts["curve"]["configuration"]["dma_memory_params"]["curves"], dict)
        self.assertEqual(selected_model(self.scorer.model)[-1], self.evidence["configuration_sha256"])
        for group, field in (("dma_memory_params", "latency"), ("dma_memory_params", "cycles_per_beat"),
                             ("accept_pattern", "period_cycles"), ("accept_pattern", "accepted_prefix_cycles")):
            target = deepcopy(self.scorer._targets)
            target["fifo43"][group][field] += 1
            with self.subTest(field=field), patch("atlas_model_scorer.synthetic_targets", return_value=target):
                changed = ModelScorer(self.scorer.model)
                self.assertNotEqual(changed.profile_receipt(document, "fifo43")["sha256"], receipts["fifo43"]["sha256"])
                self.assertEqual(self.scorer.profile_receipt(document, "fifo43"), receipts["fifo43"])
        _, values = configuration()
        self.assertNotEqual(sha256(encoded(values)), receipts["fifo43"]["sha256"])

    def test_default_curve_materialization_preserves_backend_state_and_trajectories(self):
        from atlas_validate_curve import _stream
        from npu_model.hardware.memory_backend import make_memory_backend
        old, _ = configuration()
        old.dma_memory_params = dict(old.dma_memory_params,
                                     curves=str(self.scorer.model / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json"))
        current, _ = configuration()
        bind_curve_configuration(current, self.evidence)
        old_backend, current_backend = make_memory_backend(old), make_memory_backend(current)
        # The dict input intentionally has no filesystem path. All execution
        # state, including converted curves, scaling and pacing, is identical.
        execution_state = lambda backend: {key: value for key, value in vars(backend).items()
                                          if key not in {"path", "source"}}
        self.assertEqual(execution_state(old_backend), execution_state(current_backend))
        for stores in ([False] * 65, [True] * 65, [False, True] * 65,
                       [False] * 32 + [True] * 32 + [False] * 65):
            with self.subTest(stores=stores[:4], count=len(stores)):
                self.assertEqual(_stream(make_memory_backend(old), stores),
                                 _stream(make_memory_backend(current), stores))


if __name__ == "__main__":
    unittest.main()
