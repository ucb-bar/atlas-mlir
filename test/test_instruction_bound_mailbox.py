"""Mailbox staging and scalar load scheduling are charged to entry."""
import json
import os
from pathlib import Path
import unittest

from test_composed_initialization_budget import source
from atlas_mlir_native.binding import _run, _scheduled_virtual_pipeline, NATIVE_CHECKS
from atlas_mlir_native.restoration_budget import instruction_bound


@unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'), 'select Atlas tools')
class MailboxInstructionBoundTest(unittest.TestCase):
    def test_all_mailbox_words_have_entry_ownership(self):
        text = source(iterations=1).replace('%counter,%one', '%counter, %one')
        text = text.replace('@proof()', '@proof(%choose: i32)').replace(
            'atlas.input_dram_base =',
            'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
        tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
        machine = _run([str(tools/'atlas-opt'), *_scheduled_virtual_pipeline(), '-'], input_text=text)
        _run([str(tools/'atlas-opt'), *NATIVE_CHECKS, '--verify-atlas-timing', '-'], input_text=machine)
        program = json.loads(_run([str(tools/'atlas-emit'), '--program-json', '-'], input_text=machine))
        bound = instruction_bound(text, machine, program)
        entry = bound['blocks'][0]
        self.assertEqual(entry['maximum_visits'], 1)
        words = program['instructions']
        argument = next(row for row in words if 'atlas.virtual_scalar_argument' in row['fields'])
        self.assertEqual(argument['fields']['kind'], 'lw')
        self.assertEqual(argument['fields']['atlas.virtual_cfg_block'], 0)
        self.assertEqual(entry['encoded_words'], sum(row['fields']['atlas.virtual_cfg_block'] == 0 for row in words))
        self.assertTrue(bound['admitted'])
        del argument['fields']['atlas.virtual_cfg_block']
        with self.assertRaisesRegex(ValueError, 'unique CFG ownership'):
            instruction_bound(text, machine, program)
