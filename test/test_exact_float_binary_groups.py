"""Exact binary lane grouping preserves rejection and publication boundaries."""
import os
import unittest
from unittest.mock import patch
from test_exact_float_device import request, scalar_execute, host_reference


class BinaryLaneGroupTest(unittest.TestCase):
    def setUp(self):
        from float_isa_c import executor
        executor.cache_clear()
        option = patch.dict(os.environ, ATLAS_EXACT_BINARY_LANE_GROUPS='1')
        option.start()
        self.addCleanup(option.stop)
        self.addCleanup(executor.cache_clear)

    def test_grouping_requires_explicit_option(self):
        from atlas_mlir_native.exact_float_device import prepare_plan, virtual_program
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        plan = prepare_plan(request('bf16_add', 1024))
        grouped = virtual_program(plan)
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_LANE_GROUPS='0'):
            compact = virtual_program(plan)
        os.environ.pop('ATLAS_EXACT_BINARY_LANE_GROUPS')
        self.assertEqual(virtual_program(plan), compact)
        self.assertEqual((compact.count('byte_offset = 6 '), grouped.count('byte_offset = 6 ')), (0, 4))
        self.assertEqual(compact.count('virtual_dma_load'), grouped.count('virtual_dma_load') + 2)
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_LANE_GROUPS='yes'), self.assertRaises(NativeCompilationError):
            virtual_program(plan)

    def test_group_boundaries_and_multitile_publication(self):
        for operation in ('bf16_add', 'bf16_sub'):
            for count in (1, 3, 4, 5, 31, 33, 1024, 2048):
                source = request(operation, count)
                raw = [b'\x80\x3f' * count, b'\x00\x3f' * count]
                self.assertEqual(scalar_execute(source, raw)[:2], host_reference(source, raw))
                for operand in (0, 1):
                    invalid = list(raw)
                    invalid[operand] = invalid[operand][:-2] + b'\x81\x7f'
                    output, status, _ = scalar_execute(source, invalid)
                    self.assertNotEqual(status, 0)
                    self.assertEqual(output, bytes([0xa5]) * len(output))

    def test_qualified_full_tile_instruction_work(self):
        source = request('bf16_add', 1024)
        raw = [b'\x80\x3f' * 1024, b'\x00\x3f' * 1024]
        output, status, instructions = scalar_execute(source, raw)
        self.assertEqual((output, status), host_reference(source, raw))
        # The original counted validation/classification route exceeds 132k.
        self.assertLess(instructions, 110000)


if __name__ == '__main__':
    unittest.main()
