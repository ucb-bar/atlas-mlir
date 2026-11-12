"""Pointwise ABI proposals retain the checked solver gate and exact bounds."""
from unittest.mock import patch
import unittest

from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.buffer_allocation import Buffer, BufferAllocation, check_buffer_allocation
from merlin.semantic_compiler.target_binding import NativeCompilationError
from atlas_mlir_native.pointwise import _checked_pointwise_allocation


class PointwiseAllocationTests(unittest.TestCase):
    def test_non_model_extents_keep_lexicographic_source_order(self):
        bank = StorageBank('external', 'test', 19, 'tile')
        buffers = tuple(Buffer(name, bank.name, size, 0, 1) for name, size in zip(('a', 'b', 'out'), (5, 7, 3)))
        addresses = _checked_pointwise_allocation(buffers, bank)
        self.assertEqual(addresses, {'a': 0, 'b': 5, 'out': 12})
        check_buffer_allocation(buffers, (bank,), addresses)

    def test_alignment_and_holes_remain_checked(self):
        bank = StorageBank('external', 'test', 32, 'tile')
        buffers = (Buffer('a', bank.name, 3, 0, 1, alignment=4),
                   Buffer('b', bank.name, 5, 0, 1, alignment=8),
                   Buffer('out', bank.name, 3, 0, 1, alignment=4))
        addresses = _checked_pointwise_allocation(buffers, bank)
        self.assertEqual(addresses, {'a': 0, 'b': 8, 'out': 4})
        check_buffer_allocation(buffers, (bank,), addresses)

    def test_true_insufficient_capacity_is_resource_limit(self):
        bank = StorageBank('external', 'test', 3, 'tile')
        buffers = (Buffer('a', bank.name, 2, 0, 1), Buffer('out', bank.name, 2, 0, 1))
        with self.assertRaises(NativeCompilationError) as caught:
            _checked_pointwise_allocation(buffers, bank)
        self.assertEqual(caught.exception.status, 'resource_limit')
        self.assertIn('unsatisfiable', caught.exception.reason)

    def test_unknown_solver_result_remains_unknown_even_with_valid_proposal(self):
        bank = StorageBank('external', 'test', 3, 'tile')
        buffers = (Buffer('a', bank.name, 1, 0, 1), Buffer('out', bank.name, 1, 0, 1))
        with patch('merlin.semantic_compiler.buffer_allocation.allocate_buffers',
                   return_value=BufferAllocation('unknown', reason='test budget exhausted')) as allocate:
            with self.assertRaises(NativeCompilationError) as caught:
                _checked_pointwise_allocation(buffers, bank)
        self.assertEqual(caught.exception.status, 'search_timeout')
        self.assertIn('test budget exhausted', caught.exception.reason)
        self.assertTrue(all(item.fixed_start is not None for item in allocate.call_args.args[0]))


if __name__ == '__main__':
    unittest.main()
