"""Exact binary lane grouping preserves rejection and publication boundaries."""
import unittest
from test_exact_float_device import request, scalar_execute, host_reference


class BinaryLaneGroupTest(unittest.TestCase):
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
