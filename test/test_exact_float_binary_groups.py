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
        self.assertEqual(compact.count('virtual_dma_load'), grouped.count('virtual_dma_load'))
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


class BinaryPreflightTest(unittest.TestCase):
    def setUp(self):
        from float_isa_c import executor
        executor.cache_clear()
        self.addCleanup(executor.cache_clear)

    def test_distinct_tile_routes_and_late_rejection(self):
        # The small pair is outside the qualified interval; the subnormal
        # pair needs scalar arithmetic. A later tile must keep its own route.
        domains = ((0x3f80, 0x3b80), (0x2000, 0x1c00), (1, 127))
        for operation in ('bf16_add', 'bf16_sub'):
            for first, second in ((domains[0], domains[1]),
                                  (domains[2], domains[0])):
                source = request(operation, 2048)
                pairs = [first] * 1024 + [second] * 1024
                raw = [b''.join(pair[i].to_bytes(2, 'little') for pair in pairs)
                       for i in (0, 1)]
                self.assertEqual(scalar_execute(source, raw)[:2], host_reference(source, raw))
                raw[1] = raw[1][:-2] + bytes.fromhex('c17f')
                output, status, _ = scalar_execute(source, raw)
                self.assertNotEqual(status, 0)
                self.assertEqual(output, bytes([0xa5]) * len(output))

    def test_qualified_inputs_skip_exception_checks(self):
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_LANE_GROUPS='0'):
            source = request('bf16_add', 1024)
            raw = [bytes.fromhex('803f') * 1024, bytes.fromhex('003f') * 1024]
            output, status, instructions = scalar_execute(source, raw)
            self.assertEqual((output, status), host_reference(source, raw))
            # The prior compact implementation takes over 132k instructions.
            self.assertLess(instructions, 80000)


if __name__ == '__main__':
    unittest.main()
