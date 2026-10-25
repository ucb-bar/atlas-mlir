"""Footprint algebra regressions independent of measured experiment catalogs."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_overlap import SourceRules
from test_derive_overlap import fixtures, event, release


class OverlapRulesTests(unittest.TestCase):

    def test_register_lifetimes_follow_selected_reservation_releases(self):
        cases = (("mxu0.matmul", dict(mreg_read0=event(0, 31)), 40),
                 ("xlu.transpose", dict(read=event(0, 31), write=event(35, 66)), 41))
        for name, events, available in cases:
            with self.subTest(name=name):
                facts, contract = fixtures()
                facts["op_timing"] = [dict(name=name, resolved=True, first_free_age=80, events=events)]
                facts["reservation_releases"] = {name: dict(mreg_read=release(available), mreg_write=release(80))}
                registers = SourceRules(facts, contract).registers(name)
                self.assertEqual((registers[0].release, registers[0].requests[-1]), (available, (31, 31)))

    def test_missing_or_early_register_releases_fail_closed(self):
        name = "mxu0.matmul"
        for releases in ({}, {name: {}}, {name: dict(mreg_read=release(31))},
                         {name: dict(mreg_read=dict(first_age=0, release_age=32, available_age=True))}):
            with self.subTest(releases=releases):
                facts, contract = fixtures()
                facts["op_timing"] = [dict(name=name, resolved=True, first_free_age=80,
                                           events=dict(mreg_read0=event(0, 31)))]
                facts["reservation_releases"] = releases
                with self.assertRaisesRegex(ValueError, "mreg_read reservation release"):
                    SourceRules(facts, contract).registers(name)

    def test_source_event_pairs_derive_suffix_safe_minima(self):
        model = SourceRules(*fixtures())
        cases = [
            ('relu', 'relu', '0=2,1=3', 65), ('fp8pack', 'relu', '2~3', 32),
            ('relu', 'fp8pack', '1~2', 1), ('fp8pack', 'relu', '2~1', 1),
            ('exp2', 'fp8pack', '2~2', 31), ('fp8pack', 'exp2', '2~2', 64),
            ('relu', 'fp8pack', '3~2', 63),
        ]
        for first, second, banks, minimum in cases:
            with self.subTest(first=first, second=second, banks=banks):
                first, second, relation = ('vpu.' + first, 'vpu.' + second, dict(mreg_banks=banks))
                self.assertEqual(model.minimum_gap(first, second, relation), minimum)
