"""Selection must admit exact evidence and preserve target-dependent rankings."""
import importlib.util
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'tools/select_atlas_candidates.py'
SPEC = importlib.util.spec_from_file_location('selection', PATH)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


class CandidateSelectionTests(unittest.TestCase):
    def test_memory_latency_can_reverse_winner(self):
        result = selection.rank({'baseline': {'fast': 200, 'slow': 400},
                                 'a': {'fast': 100, 'slow': 300},
                                 'b': {'fast': 110, 'slow': 250}},
                                'baseline', ['fast', 'slow'])
        self.assertEqual(result['winners'], {'fast': 'a', 'slow': 'b'})
        self.assertEqual(result['minimax_regret_candidate'], 'b')
        self.assertEqual(result['metrics']['baseline']['dominated_by'], ['a', 'b'])
        self.assertEqual(result['within_five_percent_all_targets'], [])

    def test_partial_candidate_cannot_win_robustness(self):
        result = selection.rank({'baseline': {'fast': 200, 'slow': 400},
                                 'partial': {'fast': 1}}, 'baseline', ['fast', 'slow'])
        self.assertEqual(result['minimax_regret_candidate'], 'baseline')
        self.assertNotIn('partial', result['metrics'])

    def test_rtl_only_target_has_no_model_ranking(self):
        self.assertEqual(selection.rank({}, 'baseline', []), {'status': 'NO_TARGETS'})

    def test_missing_baseline_is_incomplete(self):
        self.assertEqual(selection.rank({}, 'baseline', ['rtl'])['status'], 'INCOMPLETE_BASELINE')

    def receipt(self):
        return {'status': 'PASS', 'program_sha256': 'binary', 'cycles': [100, 99],
                'correctness': {'outputs': True, 'oracle': True, 'guards': True}}

    def test_guard_mismatch_stops_even_if_adapter_claims_pass(self):
        value = self.receipt()
        value['correctness']['guards'] = False
        with self.assertRaises(selection.CorrectnessDifference):
            selection.check_measurement(value, 'binary')

    def test_no_implicit_guard_qualification(self):
        value = self.receipt()
        del value['correctness']['guards']
        with self.assertRaises(ValueError):
            selection.check_measurement(value, 'binary')

    def test_binary_identity_is_part_of_measurement(self):
        with self.assertRaises(ValueError):
            selection.check_measurement(self.receipt(), 'another-binary')

    def test_cycle_difference_is_accepted(self):
        value = self.receipt()
        value['cycles'] = [1, 123456]
        selection.check_measurement(value, 'binary')

    def test_failed_status_cannot_hide_explicit_guard_difference(self):
        value = self.receipt()
        value.update(status='FAIL', correctness={'guards': False})
        with self.assertRaises(selection.CorrectnessDifference):
            selection.check_measurement(value, 'binary')

    def test_restricted_raw_name_is_rejected_before_filesystem_inspection(self):
        with patch.object(Path, 'is_symlink', side_effect=AssertionError('filesystem inspected')):
            with self.assertRaises(ValueError):
                selection.safe('/unaccessed/HAMMER/input')

    def test_restricted_identifier_is_refused_before_directory_creation(self):
        with self.assertRaises(ValueError):
            selection.identifier('candidate-VLSI')

    def test_source_alias_is_refused_before_its_descendants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'target').mkdir()
            (root / 'alias').symlink_to(root / 'target', target_is_directory=True)
            with self.assertRaises(ValueError):
                selection.safe(root / 'alias/not-created')

    def test_missing_measurement_receipt_stops_before_next_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.mlir'
            source.write_text('mock source')
            adapter = root / 'adapter.py'
            adapter.write_text('''import json, pathlib, sys
action, source, output = sys.argv[1:]
out = pathlib.Path(output)
if action == 'compile':
 import hashlib
 record = dict(status='PASS', source_sha256=hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(), program=str(out), program_sha256='mock-binary')
 (out/'receipt.json').write_text(json.dumps(record))
else:
 with open(pathlib.Path(source).parent/'calls.txt','a') as log: log.write('measure\\n')
 sys.exit(7)
''')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            candidate = dict(source=str(source), source_sha256=digest, admitted=True,
                             estimated_cost=1, proof=dict(path=str(source), sha256=digest))
            manifest = dict(schema='atlas.measured-selection.v1', baseline='baseline', max_candidates=2,
                            deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat(),
                            candidates=[dict(candidate,id='baseline'),dict(candidate,id='other')],
                            targets=[dict(id='first',kind='model'),dict(id='second',kind='model')],
                            compile_argv=[sys.executable,str(adapter),'compile',str(source),'{output}'],
                            measure_argv=[sys.executable,str(adapter),'measure',str(source),'{output}'])
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            process = subprocess.run([sys.executable,str(PATH),str(path),'--output',str(root/'result')],
                                     capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0)
            self.assertEqual((root/'calls.txt').read_text().splitlines(),['measure'])
            result = json.loads((root/'result/results.json').read_text())
            self.assertEqual(result['status'],'ERROR')
            self.assertNotIn('other',result['candidates'])


if __name__ == '__main__':
    unittest.main()
