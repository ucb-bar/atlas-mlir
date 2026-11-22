"""Scheduling of the composed qualification and per-step work changes when work runs, never its bytes or verdicts."""
import threading
import unittest
from unittest.mock import patch

from atlas_mlir import deployment_gate as gate, deployment_link as d
from atlas_mlir.chiprunner import composed


class StepWorkerTests(unittest.TestCase):
    def test_workers_follow_usable_cores_with_a_cap(self):
        for cores, expected in ((1, 1), (3, 3), (32, 32), (256, d.MAX_STEP_WORKERS)):
            with self.subTest(cores=cores), patch('os.sched_getaffinity', return_value=set(range(cores))):
                self.assertEqual(d.step_workers(), expected)

    def test_parallel_keeps_item_order_and_results(self):
        items = [f'step_{i:03d}' for i in range(50)]
        with patch.object(d, 'step_workers', return_value=8):
            self.assertEqual(list(d._parallel(str.upper, items).items()), [(i, i.upper()) for i in items])


class FakeCalls:
    """composed.execute stand-in: records overlap and returns or raises per label."""

    def __init__(self, positive=None, negative=None, overlap=True):
        self.outcomes = dict(positive=positive or 'journal', negative=negative or composed.BoundaryError('boundary mismatch: b/y'))
        self.started, self.labels = threading.Barrier(2, timeout=5) if overlap else None, []

    def __call__(self, bundle, logs, backend, mutation=False, transport_factory=None):
        label = logs.name
        self.labels.append((label, mutation, backend, transport_factory))
        if self.started is not None:
            self.started.wait()
        outcome = self.outcomes[label]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class QualifyTests(unittest.TestCase):
    """The gate's concurrent qualification gives the runner's sequential verdicts."""

    def qualify(self, calls, **options):
        with patch.object(composed, 'execute', calls):
            return gate._qualify('/bundle', '/logs', **options)

    def test_positive_and_negative_run_at_once(self):
        calls = FakeCalls()  # Each call waits for the other, so a sequential schedule would time out.
        self.assertEqual(self.qualify(calls), {'positive': 'journal', 'negative': 'expected numerical failure'})
        self.assertEqual(sorted(row[:3] for row in calls.labels),
                         [('negative', True, 'sim'), ('positive', False, 'sim')])

    def test_verdicts_match_the_sequential_runner(self):
        with self.assertRaisesRegex(ValueError, 'did not change'):
            self.qualify(FakeCalls(negative='completed'))
        with self.assertRaisesRegex(RuntimeError, 'positive broke'):
            self.qualify(FakeCalls(positive=RuntimeError('positive broke'), negative=RuntimeError('negative broke')))
        with self.assertRaisesRegex(RuntimeError, 'negative broke'):
            self.qualify(FakeCalls(negative=RuntimeError('negative broke')))
        with self.assertRaises(composed.BoundaryError):
            self.qualify(FakeCalls(positive=composed.BoundaryError('positive mismatch')))

    def test_options_reach_both_sessions(self):
        transport = object()
        calls = FakeCalls()
        self.qualify(calls, backend='protocol-test', transport_factory=transport)
        self.assertEqual({row[2:] for row in calls.labels}, {('protocol-test', transport)})


if __name__ == '__main__':
    unittest.main()
