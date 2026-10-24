"""Footprint algebra regressions independent of measured experiment catalogs."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_overlap_rules import relation_class
from atlas_derive_overlap import SourceRules
from test_derive_overlap import fixtures


def event(first, count, step=1):
    return dict(first_age=first, last_age=first + (count - 1) * step, count=count, step=step)

def reservation(available):
    return dict(first_age=0, release_age=available - 1, available_age=available)

def model_rules(blocks, releases):
    return SourceRules(dict(schema='atlas.model_timing.v1', age_convention='issue_edge_zero', model={}, op_timing=blocks, reservation_releases=releases), fixtures()[1])


def rules():
    return SourceRules(*fixtures())

class OverlapRulesTests(unittest.TestCase):

    def test_register_lifetimes_follow_selected_reservation_releases(self):
        for name, events, available in (('mxu0.matmul', dict(mreg_read0=event(0, 32)), 40), ('xlu.transpose', dict(read=event(0, 32), write=event(35, 32)), 41)):
            with self.subTest(name=name):
                block = dict(name=name, resolved=True, first_free_age=80, events=events)
                releases = {name: dict(mreg_read=reservation(available), mreg_write=reservation(80))}
                model = model_rules([block], releases)
                self.assertEqual(model.registers(name)[0].release, available)
                self.assertEqual(model.registers(name)[0].requests[-1], (31, 31))

    def test_missing_or_early_register_releases_fail_closed(self):
        name = 'mxu0.matmul'
        block = dict(name=name, resolved=True, first_free_age=80, events=dict(mreg_read0=event(0, 32)))
        for releases in ({}, {name: {}}, {name: dict(mreg_read=reservation(31))}, {name: dict(mreg_read=dict(first_age=0, release_age=32, available_age=True))}):
            with self.subTest(releases=releases):
                model = model_rules([block], releases)
                with self.assertRaisesRegex(ValueError, 'mreg_read reservation release'):
                    model.registers(name)




    def test_relation_family_keeps_slot_and_alias_constraints(self):
        self.assertEqual(relation_class(dict(mreg_banks='2~3')), dict(mreg_banks='bank_alias'))
        self.assertEqual(relation_class(dict(mreg_banks='0=0,1~1', accumulator='same')), dict(mreg_banks='mixed_alias', accumulator='same'))
        with self.assertRaises(ValueError):
            rules().minimum_gap('vpu.relu', 'vpu.exp2', dict(mreg_banks='invalid'))

    def test_source_event_pairs_derive_suffix_safe_minima(self):
        model = rules()
        cases = [('relu', 'relu', '0=2,1=3', 65), ('fp8pack', 'relu', '2~3', 32), ('relu', 'fp8pack', '1~2', 1), ('fp8pack', 'relu', '2~1', 1), ('exp2', 'fp8pack', '2~2', 31), ('fp8pack', 'exp2', '2~2', 64), ('relu', 'fp8pack', '3~2', 63)]
        for first, second, banks, minimum in cases:
            with self.subTest(first=first, second=second, banks=banks):
                first, second, relation = ('vpu.' + first, 'vpu.' + second, dict(mreg_banks=banks))
                self.assertEqual(model.minimum_gap(first, second, relation), minimum)
