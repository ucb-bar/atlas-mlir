"""Fair policy expansion and conservative refusal in the optional provider."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import compare_atlas_materializations as subject


@dataclass
class Cost:
    score: int


class MaterializationsTest(unittest.TestCase):
    def run_comparison(self, *, refuse=False, mutate=False):
        sources = {'native': 'native', 'hand-u0': 'legacy', 'egg-00': 'joint'}
        candidates = [SimpleNamespace(source_sha256=subject.digest(s.encode()), source=s.encode())
                      for s in sources.values()]
        details = {name: dict(source_sha256=subject.digest(source.encode()), choices=[])
                   for name, source in sources.items()}
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / 'compiler'
            compiler.write_bytes(b'compiler')

            def share(source, **kwargs):
                calls.append((source, kwargs['max_distance'], kwargs['max_constants']))
                if mutate:
                    compiler.write_bytes(b'changed')
                return SimpleNamespace(source=source if refuse else source + '-shared',
                    evidence=dict(selected='baseline' if refuse else 'shared', shares=[{}]))

            with patch.object(subject, 'search', return_value=(candidates, {'details': details})), \
                 patch.object(subject, 'enumerate_hand', return_value=iter([
                     ('native', 'native', (), None), ('hand-exhaustive-000', 'joint', (0,), None)])), \
                 patch.object(subject, 'parse_program', side_effect=lambda source: source), \
                 patch.object(subject, 'bounded_splat_sharing', side_effect=share):
                rows, report = subject.compare(None, SimpleNamespace(estimate=lambda source: Cost(len(source))),
                    atlas_opt=compiler, budget=1)
        return rows, report, calls

    def test_same_policy_set_and_shared_budget_with_common_controls(self):
        rows, report, calls = self.run_comparison()
        strategies = report['strategy_pool']
        for name in ('hand', 'egglog'):
            self.assertEqual({None, (16, 2), (64, 2)},
                {row['policy'] for row in strategies if row['strategy'] == name})
        self.assertEqual(1, report['measurement_budget_per_strategy'])
        self.assertEqual(6, len(calls))  # Duplicate hand/egg source shares one probe cache.
        self.assertEqual(report['aliases']['egg-00-share-d16-c2'],
                         report['aliases']['hand-exhaustive-000-share-d16-c2'])
        selected = report['strategy_selections']
        self.assertEqual(6, sum(row['strategy'] == 'control' for row in selected))
        self.assertEqual(1, sum(row['strategy'] == 'hand' for row in selected))
        self.assertEqual(1, sum(row['strategy'] == 'egglog' for row in selected))
        self.assertEqual(6, len(rows))

    def test_refused_policy_has_no_effective_shares_and_exact_baseline_alias(self):
        rows, report, _ = self.run_comparison(refuse=True)
        self.assertEqual(3, len(rows))
        for row in report['strategy_pool']:
            self.assertIsNone(row['effective_policy'])
            self.assertEqual(0, row['applied_shares'])
            self.assertNotIn('identical_splat_sharing', row['families'])
        self.assertEqual(report['aliases']['egg-00'], report['aliases']['egg-00-share-d64-c2'])

    def test_changed_compiler_refuses_comparison(self):
        with self.assertRaisesRegex(ValueError, 'compiler changed'):
            self.run_comparison(mutate=True)

    def test_invalid_policy_and_budget_refused_before_tool_access(self):
        for kwargs in ({'budget': 0}, {'budget': 17}, {'policies': ((0, 2),)},
                       {'policies': ((True, 2),)}, {'policies': ((16, 2), (16, 2))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                subject.compare(None, None, atlas_opt='/absent/compiler', **kwargs)


if __name__ == '__main__':
    unittest.main()
