"""Executable physical paths bound counted loops without charging both arms."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from atlas_mlir_native.restoration_budget import _block_path_words, instruction_bound


ROOT = Path(__file__).resolve().parents[1]


def header_program():
    # Header words 0..7 mirror region_087's words 310..317: continue costs
    # five words; exit costs six, including their respective delay slots.
    controls = [dict(kind='sequential'),
                dict(kind='conditional_direct', delay_slot_word_index=2,
                     fallthrough_word_index=3, target_word_index=6),
                dict(kind='sequential'), dict(kind='sequential'),
                dict(kind='jump_direct', delay_slot_word_index=5, target_word_index=8),
                dict(kind='sequential'),
                dict(kind='jump_direct', delay_slot_word_index=7, target_word_index=9),
                dict(kind='sequential'), dict(kind='trap'), dict(kind='trap')]
    rows = [dict(word_index=i, fields={'atlas.virtual_cfg_block': 0 if i < 8 else i-7}, control=control)
            for i, control in enumerate(controls)]
    for slot, branch in ((2, 1), (5, 4), (7, 6)):
        rows[slot]['delay_slot_for_word_index'] = branch
    return rows


class ExecutableBlockPathTest(unittest.TestCase):
    def test_header_counts_branch_delay_and_one_edge_arm(self):
        costs, refined = _block_path_words(header_program(), {0: [2, 1], 1: [], 2: []})
        self.assertEqual(costs[0], {2: 5, 1: 6})
        self.assertEqual(refined, {0, 1, 2})

    def test_arms_to_same_successor_take_longest_path(self):
        rows = header_program()
        rows[6]['control']['target_word_index'] = 8
        costs, refined = _block_path_words(rows, {0: [1], 1: [], 2: []})
        self.assertEqual(costs[0], {1: 6})
        self.assertIn(0, refined)

    def test_unknown_or_inconsistent_controls_retain_all_words(self):
        for mutation in ('missing', 'indirect', 'delay', 'fallthrough', 'target'):
            with self.subTest(mutation=mutation):
                rows = deepcopy(header_program())
                if mutation == 'missing':
                    rows[1].pop('control')
                elif mutation == 'indirect':
                    rows[1]['control']['kind'] = 'jump_register'
                elif mutation == 'delay':
                    rows[1]['control']['delay_slot_word_index'] = 3
                elif mutation == 'fallthrough':
                    rows[1]['control']['fallthrough_word_index'] = 4
                else:
                    rows[1]['control']['target_word_index'] = 100
                costs, refined = _block_path_words(rows, {0: [2, 1], 1: [], 2: []})
                self.assertEqual(costs[0], {2: 8, 1: 8})
                self.assertNotIn(0, refined)

    def test_instruction_limit_is_supplied_by_caller(self):
        machine = ('atlas.virtual_cfg_contract = {blocks = [{id = 0 : i32, condition = -1 : i32, '
                   'args = array<i32>, operations = array<i32>}], edges = [], operations = [], '
                   'output_buffer = 0 : i32, values = []}')
        rows = [dict(word_index=i, fields={'atlas.virtual_cfg_block': 0},
                     control=dict(kind='sequential' if i == 0 else 'trap')) for i in range(2)]
        for cap in (1, 2, 1_000_000):
            proof = instruction_bound('', machine, {'instructions': rows}, max_instructions=cap)
            self.assertEqual(proof['max_instructions'], cap)
            self.assertEqual(proof['admitted'], cap >= 2)
        for cap in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'invalid instruction limit'):
                instruction_bound('', machine, {'instructions': rows}, max_instructions=cap)

    def test_region087_worst_case_shape_is_admitted(self):
        fixture = json.loads((ROOT / 'test/fixtures/dynamic-head-quantizer-bound.json').read_text())
        proof = instruction_bound(**fixture)
        self.assertEqual(proof['instruction_bound'], 988_850)
        self.assertTrue(proof['admitted'])
        header = next(block for block in proof['blocks'] if block['block'] == 15)
        self.assertEqual(header['encoded_words'], 8)
        self.assertEqual(header['successor_path_words'], [{'successor': 17, 'words': 5},
                                                         {'successor': 16, 'words': 6}])
        self.assertEqual(header['continue_visits'], 32*1024)
        self.assertEqual(header['exit_visits'], 32)
        self.assertEqual(header['instruction_bound'], 32*(1024*5+6))


if __name__ == '__main__':
    unittest.main()
