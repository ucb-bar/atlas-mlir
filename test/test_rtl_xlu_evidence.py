"""Mutation checks for the conditional XLU receipt and observation validator."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('xlu_check', ROOT / 'tools/check-rtl-xlu.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def controlled_trace(case):
    origins = [(0, case['src'], case['dst'])]
    if case['accept_second']:
        origins.append((case['gap'], case['second_src'], case['second_dst']))
    rows = []
    for cycle in range(max(case['gap'], 0) + 75):
        row = dict(cycle=cycle, launch=int(cycle == 0 or cycle == case['gap']), capture=0,
                   busy_launch=0, read_active=0, read_active_reg=0, write_active=0,
                   write_active_reg=0, mreg_read=0, mreg_read_reg=0, mreg_read_row=0,
                   mreg_response=0, mreg_write=0, mreg_write_reg=0, mreg_write_row=0)
        for age, src, dst in origins:
            row['capture'] |= int(cycle == age)
            if age + 1 <= cycle <= age + 32:
                row.update(mreg_read=1, mreg_read_reg=src, mreg_read_row=cycle - age - 1)
            if age + 1 + case['delay'] <= cycle <= age + 32 + case['delay']:
                row['mreg_response'] = 1
            if age + 33 + case['delay'] <= cycle <= age + 64 + case['delay']:
                row.update(mreg_write=1, mreg_write_reg=dst,
                           mreg_write_row=cycle - age - 33 - case['delay'])
            if age + 1 <= cycle <= age + 32 + case['delay']:
                row.update(read_active=1, read_active_reg=src)
            if age + 1 <= cycle <= age + 64 + case['delay']:
                row.update(write_active=1, write_active_reg=dst)
        row['busy_launch'] = int(row['launch'] and row['write_active'])
        rows.append(row)
    summary = dict(complete=True, data_and_guards_pass=True, reads=32 * len(origins),
                   writes=32 * len(origins), captures=len(origins),
                   busy_launches=int(case['gap'] >= 0 and not case['accept_second']))
    return rows, summary


class XluEvidenceTests(unittest.TestCase):
    def test_control_cases_cover_aliases_boundary_loss_and_latency_assumption(self):
        for case in CHECK.CASES:
            with self.subTest(case=case['name']):
                rows, summary = controlled_trace(case)
                self.assertTrue(all(CHECK.check_trace(case, rows, summary).values()))

    def test_mutated_stream_register_row_and_cycle_reject(self):
        case = CHECK.CASES[0]
        for field, cycle in [('mreg_read_reg', 1), ('mreg_read_row', 32),
                             ('mreg_write_reg', 34), ('mreg_write_row', 65), ('cycle', 34)]:
            with self.subTest(field=field):
                rows, summary = controlled_trace(case)
                rows[cycle][field] += 1
                self.assertFalse(all(CHECK.check_trace(case, rows, summary).values()))

    def test_missing_response_premature_release_and_false_numerical_result_reject(self):
        case = CHECK.CASES[0]
        for mutation in ['response', 'source_release', 'engine_release', 'numerical', 'count', 'missing_cycle']:
            with self.subTest(mutation=mutation):
                rows, summary = controlled_trace(case)
                if mutation == 'response': rows[33]['mreg_response'] = 0
                elif mutation == 'source_release': rows[33]['read_active'] = 0
                elif mutation == 'engine_release': rows[65]['write_active'] = 0
                elif mutation == 'numerical': summary['data_and_guards_pass'] = False
                elif mutation == 'count': summary['captures'] += 1
                else: rows.pop()
                self.assertFalse(all(CHECK.check_trace(case, rows, summary).values()))

    def test_busy_attempt_is_not_successful_capture(self):
        case = CHECK.CASES[6]
        rows, summary = controlled_trace(case)
        rows[65]['capture'] = 1
        rows[65]['busy_launch'] = 0
        self.assertFalse(all(CHECK.check_trace(case, rows, summary).values()))

    def test_slice_preserves_complete_module_and_assertions(self):
        data = b'''module {
  hw.module private @XluEngine(in %clock : !seq.clock) {
    sv.fatal 1 loc(#loc0)
    hw.output
  } loc(#loc0)
}
#loc0 = loc("x.scala":1:1)
'''
        raw, preamble, standalone, extraction = CHECK.selected_slice(data)
        self.assertIn(b'sv.fatal 1 loc(#loc0)', raw)
        self.assertIn(b'sv.fatal 1', standalone)
        self.assertNotIn(b'loc(#loc0)', standalone)
        self.assertTrue(extraction['assertions_preserved'])
        self.assertTrue(extraction['selected_module_has_assertions'])
        with self.assertRaises(CHECK.CheckError):
            CHECK.selected_slice(data.replace(b'@XluEngine', b'@OtherEngine'))

    def test_manifest_identity_rejects_before_tools_or_output_creation(self):
        artifact_root = ROOT / 'build/rtl-timing'
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            directory = Path(directory)
            hardware = directory / 'input.mlir'
            hardware.write_bytes(b'identity checked before parsing\n')
            member = dict(path=str(hardware), bytes=hardware.stat().st_size,
                          sha256=hashlib.sha256(hardware.read_bytes()).hexdigest())
            manifest = directory / 'manifest.json'
            for field, value in [('sha256', '0' * 64), ('bytes', 0)]:
                bad = copy.deepcopy(member)
                bad[field] = value
                manifest.write_text(json.dumps(dict(schema='atlas.retained_hw_ir.v0', state='verified',
                                                    target_config='EE290SimConfig', hardware_ir=bad)))
                output = directory / ('rejected-' + field)
                with self.assertRaisesRegex(CHECK.INDEX.IndexError, 'hash/size mismatch'):
                    CHECK.run(SimpleNamespace(manifest=manifest, output=output))
                self.assertFalse(output.exists())

    def test_unreviewed_source_identity_and_missing_snapshot_reject(self):
        with self.assertRaisesRegex(CHECK.CheckError, 'outside the reviewed'):
            CHECK.checked_sources({'sources': [dict(path='XLU.scala', sha256='0' * 64,
                                                   captured_copy=dict(sha256='0' * 64))]},
                                  ROOT / 'build/rtl-timing/inventory.json')
        with self.assertRaisesRegex(CHECK.CheckError, 'Missing reviewed'):
            CHECK.checked_sources({'sources': []}, ROOT / 'build/rtl-timing/inventory.json')


if __name__ == '__main__':
    unittest.main()
