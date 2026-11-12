"""Resource-only planning never qualifies or publishes a machine artifact."""

from collections import Counter
import json
from pathlib import Path
import re
import tempfile
import unittest

from test_virtual_lowering import EXAMPLES, run, virtual_chain, virtual_pressure


class ResourcePlanningReportTest(unittest.TestCase):
    def report(self, source, *options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resources.json'
            result = run('atlas-opt', source,
                         '--lower-atlas-virtual-to-machine='
                         f'resource-planning-report={path}', *options)
            return result, json.loads(path.read_text())

    def test_resource_query_preserves_virtual_ir_and_is_unqualified(self):
        source = virtual_chain(2)
        canonical = run('atlas-opt', source)
        result, report = self.report(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, canonical.stdout)
        self.assertEqual(report['schema'], 'atlas.virtual_resource_plan.v1')
        self.assertTrue(report['planning_succeeded'])
        self.assertFalse(report['machine_artifact_qualified'])
        self.assertNotIn('fits', report)
        self.assertEqual(report['failures'], [])
        self.assertEqual(sum(block['planned_words'] for block in report['cfg_blocks']),
                         report['untimed_words']['words'])
        # The returned IR still cannot cross the final artifact boundary.
        self.assertNotEqual(run('atlas-emit', result.stdout).returncode, 0)

    def test_owner_counts_are_lower_bounds_after_waits_and_both_timing_paths(self):
        for name in ('virtual_bf16_loop_program.mlir',
                     'virtual_bf16_dynamic_branch_program.mlir',
                     'virtual_dma_mxu.mlir'):
            source = (EXAMPLES / name).read_text()
            result, report = self.report(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(report['planned_helper_control'])
            for timing in ('--insert-atlas-delays', '--schedule-atlas-stream'):
                with self.subTest(source=name, timing=timing):
                    final = run('atlas-opt', source, '--lower-atlas-virtual-to-machine',
                                '--insert-atlas-dma-waits', timing)
                    self.assertEqual(final.returncode, 0, final.stderr)
                    self.assertIn(report['cfg_contract'], final.stdout)
                    counts = Counter(map(int, re.findall(
                        r'atlas.virtual_cfg_block = (\d+) : i32', final.stdout)))
                    for block in report['cfg_blocks']:
                        self.assertGreaterEqual(counts[block['id']], block['planned_words'])

    def test_helper_control_is_explicitly_reported(self):
        source = (EXAMPLES / 'virtual_fp8_two_layer_mlp.mlir').read_text()
        result, report = self.report(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report['planned_helper_control'])
        self.assertFalse(report['machine_artifact_qualified'])

    def test_allocation_failure_is_not_a_successful_plan(self):
        result, report = self.report(virtual_pressure(32))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report['planning_succeeded'])
        self.assertFalse(report['machine_artifact_qualified'])
        self.assertNotIn('cfg_contract', report)
        self.assertTrue(report['failures'])

    def test_qualification_and_resource_reports_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('atlas-opt', virtual_chain(1),
                         '--lower-atlas-virtual-to-machine='
                         f'resource-planning-report={directory}/plan.json '
                         f'allocation-report={directory}/allocation.json')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('mutually exclusive', result.stderr)


if __name__ == '__main__':
    unittest.main()
