"""Only the declared argument mailbox may bypass source-DMA ownership."""

import re
import unittest

from test_composed_initialization_budget import source
from test_virtual_ssa import run


class ArgumentMailboxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = source(iterations=1).replace('@proof()', '@proof(%choose: i1)').replace(
            'atlas.input_dram_base =',
            'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
        result = run('atlas-opt', text, '--atlas-compile-unscheduled')
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.machine = result.stdout

    def test_canonical_mailbox_is_accepted(self):
        result = run('atlas-opt', self.machine, '--verify-atlas-generated-schedule')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tile_zero_cannot_name_another_declared_mailbox(self):
        changed = self.machine.replace('atlas.control_dram_base = 2415925248 : i64',
                                       'atlas.control_dram_base = 2415926272 : i64', 1)
        self.assertNotEqual(changed, self.machine)
        result = run('atlas-opt', changed, '--verify-atlas-generated-schedule')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not prove output reads initialized', result.stderr)

    def test_matching_tile_metadata_cannot_shorten_the_mailbox(self):
        # Change both the unowned DMA's size operand and its tile record. The
        # generic tile correspondence still agrees, but the ABI requires 1024.
        lines = self.machine.splitlines()
        dma = next(i for i, line in enumerate(lines)
                   if '"atlas.dma"' in line and 'atlas.virtual_tile_command = 0 : i32' in line)
        size_reg = re.search(r'size = (\d+) : i32', lines[dma]).group(1)
        defining = next(i for i in range(dma - 1, -1, -1)
                        if f'dst = {size_reg} : i32' in lines[i])
        self.assertIn('immediate = 1024 : i32', lines[defining])
        lines[defining] = lines[defining].replace('immediate = 1024 : i32',
                                                 'immediate = 32 : i32', 1)
        changed = '\n'.join(lines)
        tile_zero = re.search(r'\{after = array<i32>, bytes = 1024 : i32, channel = 0 : i32, '
                              r'dram_byte = -1879042048 : i32, id = 0 : i32, kind = "dma_load"[^{}]*\}',
                              changed)
        self.assertIsNotNone(tile_zero)
        changed = changed.replace(tile_zero.group(),
                                  tile_zero.group().replace('bytes = 1024 : i32', 'bytes = 32 : i32'), 1)
        result = run('atlas-opt', changed, '--verify-atlas-generated-schedule')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not prove output reads initialized', result.stderr)


if __name__ == '__main__':
    unittest.main()
