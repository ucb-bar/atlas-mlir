"""Fixed replay policy and checked source edges."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_replay import replay, semantic_digest, validate_program, prediction_error
from atlas_model_diagnostics import UnsupportedModelTiming
from model_replay_fixture import replay_fixture
from model_evidence import NativeModelFixture


class ReplayContractTest(unittest.TestCase):
    def test_fixed_policy_is_exact_and_unknown_structures_fail_closed(self):
        program = replay_fixture()
        self.assertEqual(validate_program(program), program)
        for mutate in (lambda p: p.update(start=[["cycle", [0]]]),
                       lambda p: p.update(schema="atlas.projected_core_replay.v1"),
                       lambda p: p["source"].update(clock_step=2),
                       lambda p: p["projection_policy"].update(dma="native")):
            changed = deepcopy(program)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_program(changed)
            self.assertNotEqual(semantic_digest(changed), semantic_digest(program))

    def test_fixed_recurrence_and_prediction_error(self):
        graph = dict(preds=[[], [], [0]], order=[0, 1, 2], proposal=dict(replay=replay_fixture(), dma_capacity=8,
            costs=[dict(unit=0, busy=10, latency=12, frontend=2, queued_dma=True, drains=False),
                   dict(unit=1, busy=3, latency=3, frontend=1, queued_dma=False, drains=False),
                   dict(unit=1, busy=1, latency=1, frontend=1, queued_dma=False, drains=False)]))
        self.assertEqual(replay(graph, (0, 1, 2)), (14, 10))
        result = prediction_error([graph], 20)
        self.assertEqual((result['predicted_cycles'], result['signed_error_cycles']), (14, -6))
        self.assertIn('excludes', result['approximation_scope'])


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native Core contracts")
class NativeReplayContractTest(NativeModelFixture, unittest.TestCase):
    def test_reordered_core_or_new_implicit_stalls_refuse_without_running_native(self):
        import atlas_derive_frontend as frontend
        from atlas_derive_replay import derive_replay
        original = frontend.read_source
        for old, new in (("target = self._tick_s1()", "target = None"),
                         ('if self.delay_counter:', 'if self.delay_counter or self.exus:')):
            def mutated(path):
                text = original(path)
                return text.replace(old, new) if Path(path).name == 'core.py' else text
            with patch.object(frontend, 'read_source', side_effect=mutated):
                with self.assertRaises(UnsupportedModelTiming):
                    derive_replay(self.cfg)


if __name__ == '__main__':
    unittest.main()
