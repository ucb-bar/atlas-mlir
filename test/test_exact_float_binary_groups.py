"""Exact binary lane grouping preserves rejection and publication boundaries."""

from dataclasses import replace
import unittest
from atlas_mlir_native.exact_float_device import prepare_plan, virtual_program
from atlas_mlir_native.options import CodegenOptions, codegen_options, current_codegen_options
from float_isa_c import executor
from test_exact_float_device import request, scalar_execute, host_reference


class BinaryLaneGroupTest(unittest.TestCase):
    def setUp(self):
        executor.cache_clear()
        option = codegen_options(replace(current_codegen_options(), exact_binary_lane_groups=True, exact_binary_vector_preflight=False, native_pipeline='mandatory'))
        option.__enter__()
        self.addCleanup(option.__exit__, None, None, None)
        self.addCleanup(executor.cache_clear)

    def test_grouping_requires_explicit_option(self):
        plan = prepare_plan(request('bf16_add', 1024))
        grouped = virtual_program(plan)
        with codegen_options(replace(current_codegen_options(), exact_binary_lane_groups=False)):
            compact = virtual_program(plan)
        with codegen_options(CodegenOptions()):
            self.assertEqual(virtual_program(plan), compact)
        self.assertEqual((compact.count('byte_offset = 6 '), grouped.count('byte_offset = 6 ')), (0, 4))
        # The default keeps its original validation/compute reloads; explicitly
        # grouped preflight retains a single immutable tile through computation.
        self.assertEqual((compact.count('virtual_dma_load'), grouped.count('virtual_dma_load')), (4, 2))
        self.assertNotIn('%routes =', compact)
        self.assertIn('%routes =', grouped)

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
        executor.cache_clear()
        # Exercise cached scalar preflight independently of both vector
        # preflight and grouped lanes, using the explicit route-cache opt-in.
        option = codegen_options(replace(current_codegen_options(), exact_binary_lane_groups=False, exact_binary_vector_preflight=False, exact_binary_routes=True, native_pipeline='mandatory'))
        option.__enter__()
        self.addCleanup(option.__exit__, None, None, None)
        self.addCleanup(executor.cache_clear)

    def test_route_cache_is_independent_of_native_pipeline(self):
        plan = prepare_plan(request('bf16_add', 1024))
        for pipeline in ('mandatory', 'optimized'):
            with self.subTest(pipeline=pipeline):
                with codegen_options(replace(current_codegen_options(), native_pipeline=pipeline,
                                             exact_binary_routes=False)):
                    compact = virtual_program(plan)
                with codegen_options(replace(current_codegen_options(), native_pipeline=pipeline,
                                             exact_binary_routes=True)):
                    cached = virtual_program(plan)
                self.assertNotIn('%routes =', compact)
                self.assertIn('%routes =', cached)

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
        with codegen_options(replace(current_codegen_options(), exact_binary_lane_groups=False)):
            source = request('bf16_add', 1024)
            raw = [bytes.fromhex('803f') * 1024, bytes.fromhex('003f') * 1024]
            output, status, instructions = scalar_execute(source, raw)
            self.assertEqual((output, status), host_reference(source, raw))
            # The preserved default reloads and reclassifies; this explicitly
            # selected scalar preflight skips unnecessary exception checks.
            self.assertLess(instructions, 80000)


if __name__ == '__main__':
    unittest.main()
