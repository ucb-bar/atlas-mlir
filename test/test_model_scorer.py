"""Selected-model identity, decoding and execution failures must fail closed."""
from copy import deepcopy
from io import StringIO
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


def _stream(backend, stores, *, translation=0, stalls=False, gap=0):
    from npu_model.hardware.memory_backend import BeatRequest
    issued, completed, blocked, held = [], [], 0, 0
    max_pending = 0
    index = 0
    for cycle in range(200000):
        response = backend.response(cycle)
        if response is not None:
            if stalls and cycle % 31 < 7:
                held += 1
                assert backend.response(cycle) == response
            else:
                completed.append([cycle, response.source, response.is_store])
                backend.pop_response(cycle)
        if index < len(stores) and cycle >= index * gap:
            if backend.can_accept(cycle):
                backend.issue(BeatRequest(index, translation + index * 32, stores[index], cycle))
                issued.append([cycle, index, stores[index]])
                index += 1
            else:
                blocked += 1
        max_pending = max(max_pending, backend.outstanding)
        if index == len(stores) and not backend.outstanding:
            return dict(issued=issued, completed=completed, blocked_cycles=blocked,
                        held_cycles=held, max_pending=max_pending)
    raise AssertionError("backend regression did not drain")


class ScorerGuardsTest(unittest.TestCase):
    def test_assertions_cannot_be_disabled(self):
        with patch("atlas_model_scorer.sys.flags") as flags:
            flags.optimize = 1
            with self.assertRaisesRegex(ValueError, "assertions"):
                ModelScorer(Path("/tmp"))({})

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

    def test_configuration_values_preserves_floats(self):
        self.assertEqual(configuration_values(SimpleNamespace(frequency=0.75, latency=1.25)),
                         {"frequency": 0.75, "latency": 1.25, "ignore_runtime_errors": False})

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
            with patch("atlas_model_scorer.resolve_curve_path", return_value=selected):
                bind_curve_configuration(cfg, {"dma_curve_sha256": sha256(raw)})
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

    def decode(self, word):
        from npu_model.configs import isa_definition as isa
        row, = decode_binary(self.inventory, word.to_bytes(4, "little"))["instructions"]
        instruction = model_instruction(row, self.inventory, isa)
        self.assertEqual(instruction.to_bytecode(), word)
        return row, instruction

    def test_shift_immediate_decodes_without_function_bits(self):
        for word in (0x40115093, 0x00115093, 0x00111093):
            with self.subTest(word=word):
                _, instruction = self.decode(word)
                self.assertEqual(instruction.imm & 31, 1)

    def test_every_selected_encoding_reencodes_through_the_model(self):
        for mode in self.inventory["variants"]:
            word = int(mode["rtl_bitpat"].replace("?", "0"), 2)
            with self.subTest(mode=mode["id"]):
                row, _ = self.decode(word)
                self.assertEqual(row["mode_id"], mode["id"])

    def test_nonzero_exponent_operand_survives_decode_and_reencoding(self):
        from npu_model.util.converter import stream_to_instrs
        for assembly in ("vpack.bf16.fp8 m7, m34, e31", "vmatpop.fp8.acc.mxu0 m9, acc1, e17",
                         "vmatpop.fp8.acc.mxu1 m11, acc1, e23"):
            original, = stream_to_instrs(StringIO(assembly))
            with self.subTest(assembly=assembly):
                row, instruction = self.decode(original.to_bytecode())
                self.assertEqual(row["fields"]["scale_reg"], int(original.es1))
                self.assertEqual(instruction.es1, original.es1)

    def test_signed_bundled_offsets_preserve_the_encoded_memory_operands(self):
        from npu_model.util.converter import stream_to_instrs
        for mnemonic, register in (("lw", "x4"), ("sw", "x4"), ("seld", "e4"), ("vload", "m35"), ("vstore", "m35")):
            for literal, expected in (("-2048", -2048), ("-1", -1), ("-0x10", -16),
                                      ("-0b100", -4), ("-0o10", -8), ("2047", 2047)):
                assembly = f"{mnemonic} {register}, {literal}(x3)"
                original, = stream_to_instrs(StringIO(assembly))
                with self.subTest(assembly=assembly):
                    row, instruction = self.decode(original.to_bytecode())
                    self.assertEqual(row["fields"]["offset"], expected)
                    self.assertEqual(instruction.rs1, 3)
                    self.assertEqual(instruction.imm & 0xfff, expected & 0xfff)

    def test_cold_cycles_profiles_and_exhaustion(self):
        # ADDI x1, x0, 1; CSRRW x0, completion, x1; ECALL.
        program = self.program([0x00100093, 0xc1009073, 0x00000073])
        for profile in ("fifo43", "curve"):
            self.assertEqual(self.scorer(program, profile), 3)
        with self.assertRaisesRegex(ValueError, "exhausted before ECALL"):
            self.scorer(self.program([0x00100093]))

    def test_fifo43_native_backend_matches_previous_callable_trajectories(self):
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
        _, values = configuration()
        self.assertNotEqual(sha256(encoded(values)), receipts["fifo43"]["sha256"])


if __name__ == "__main__":
    unittest.main()
