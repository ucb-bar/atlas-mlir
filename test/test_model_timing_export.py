"""External MXU observation restores methods and preserves sampled results."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_observer import observe_mxu
import atlas_model_timing as timing
from atlas_model_timing import compress, reference_facts, result_signature


class Staged:
    def __init__(self, unit, uop):
        self.state, self.unit, self.uop = unit.arch_state, unit.mxu, uop
        self.value = object()

    def sample(self, location, index, value):
        return value

    def result(self, location):
        return self.value


class ModelObserverTest(unittest.TestCase):
    def setUp(self):
        self.unit = SimpleNamespace(arch_state=object(), mxu="mxu0", cycle=7)
        self.uop = SimpleNamespace(insn=SimpleNamespace(mnemonic="vmatmul.mxu0"))
        self.staged = Staged(self.unit, self.uop)
        self.events = []
        self.methods = (Staged.sample, Staged.result)

    def tearDown(self):
        self.assertEqual((Staged.sample, Staged.result), self.methods)

    def observe(self, sink=None):
        return observe_mxu(Staged, self.unit, self.uop, sink or (lambda **event: self.events.append(event)))

    def test_samples_and_equal_value_inferred_writes_are_not_change_detection(self):
        value = object()
        with self.observe() as finish:
            for _ in range(2):
                self.assertIs(self.staged.sample(("acc", 0), 3, value), value)
                self.assertIs(self.staged.result(("acc", 0)), self.staged.value)
            self.assertEqual([event["kind"] for event in self.events], ["acc_read"] * 2)
            finish()
            self.assertEqual([event["kind"] for event in self.events], ["acc_read"] * 2 + ["acc_write"] * 2)
            self.assertEqual([event["row"] for event in self.events[:2]], [3, 3])
            self.assertNotIn("row", self.events[-1])
            finish()
            self.assertEqual(len(self.events), 4)

    def test_weight_results_and_foreign_operations(self):
        self.uop.insn.mnemonic = "vmatpush.weight.mxu0"
        with self.observe() as finish:
            self.staged.sample(("wb", 0), 0, 0)
            self.staged.result(("mrf", 0))
            for attribute in ("state", "unit", "uop"):
                foreign = Staged(self.unit, self.uop)
                setattr(foreign, attribute, object())
                foreign.sample(("acc", 0), 0, 0)
                foreign.result(("wb", 0))
            self.staged.result(("wb", 0))
            finish()
        self.assertEqual([event["kind"] for event in self.events], ["weight_write"])

    def test_failed_tick_discards_pending_writes(self):
        with self.assertRaisesRegex(RuntimeError, "tick failed"):
            with self.observe():
                self.staged.result(("acc", 0))
                raise RuntimeError("tick failed")
        self.assertEqual(self.events, [])
        with self.observe() as finish:
            finish()
        self.assertEqual(self.events, [])

    def test_sink_failure_restores_both_methods(self):
        def failed_sink(**event):
            raise RuntimeError("sink failed")
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            with self.observe(failed_sink) as finish:
                self.staged.result(("acc", 0))
                finish()

    def test_original_method_failure_restores_both_methods(self):
        def failed_result(staged, location):
            raise RuntimeError("original failed")
        with patch.object(Staged, "result", failed_result):
            with self.assertRaisesRegex(RuntimeError, "original failed"):
                with self.observe():
                    self.staged.result(("acc", 0))
            self.assertIs(Staged.result, failed_result)

    def test_original_sample_failure_restores_both_methods(self):
        def failed_sample(staged, location, index, value):
            raise RuntimeError("sample failed")
        with patch.object(Staged, "sample", failed_sample):
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                with self.observe():
                    self.staged.sample(("acc", 0), 0, 0)
            self.assertIs(Staged.sample, failed_sample)

    def test_revision_source_and_numerical_data_are_independently_pinned(self):
        expected = (timing.MODEL_REVISION, timing.SOURCE_SHA256, timing.NUMERICAL_DATA_SHA256)
        for changed in range(3):
            values = list(expected)
            values[changed] = "unexpected"
            with self.subTest(field=changed), patch.object(timing.subprocess, "check_output", return_value=values[0]), \
                    patch.object(timing, "source_digest", return_value=(values[1], [])), \
                    patch.object(timing, "numerical_data", return_value=(values[2], [])):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    timing.model_identity(Path("unused-model"))

    def test_nested_observer_is_rejected_without_disabling_outer_observer(self):
        with self.observe() as finish:
            with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                with self.observe():
                    pass
            self.staged.result(("acc", 0))
            finish()
        self.assertEqual(len(self.events), 1)

    def test_uncovered_result_site_is_rejected(self):
        self.uop.insn.mnemonic = "unknown"
        with self.assertRaisesRegex(RuntimeError, "uncovered MXU result"):
            with self.observe():
                self.staged.result(("acc", 0))

    def test_event_sweeps_reject_missing_or_repeated_edges(self):
        self.assertEqual(compress([3, 1, 2]), dict(count=3, first_age=1, last_age=3, step=1))
        for ages in ([], [1, 1], [1, 2, 4]):
            with self.subTest(ages=ages), self.assertRaises(ValueError):
                compress(ages)

    def test_unpinned_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "facts.json"
            reference.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                reference_facts(reference)

    def test_noninterference_signature_includes_each_state_and_progress(self):
        fields = ("initial_state_sha256", "final_state_sha256", "architectural_states", "control_states", "pre_edges",
                  "first_free_age", "complete_count", "total_instructions", "busy_cycles", "instruction_word", "configuration")
        raw = dict.fromkeys(fields, "value")
        raw["uop_id"] = 1
        expected = result_signature(raw)
        raw["uop_id"] = 2
        self.assertEqual(result_signature(raw), expected)
        for field in fields:
            with self.subTest(field=field):
                changed = dict(raw, **{field: "changed"})
                self.assertNotEqual(result_signature(changed), expected)


if __name__ == "__main__":
    unittest.main()
