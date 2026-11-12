"""Execution budgets enlarge counted dispatches without enlarging IMEM."""
from pathlib import Path
import json
import os
import tempfile
import unittest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.limits import functional_instruction_limit
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_exact_float_device import request, host_reference, scalar_execute

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('ATLAS_OOT_BIN_DIR', ROOT / 'build/bin'))


class DispatchBudgetTest(unittest.TestCase):
    def test_scoped_budget_preserves_default_and_complete_tail(self):
        source = request('bf16_add', 131075)
        self.assertEqual(functional_instruction_limit(), 1_000_000)
        self.assertEqual(len(application.prepare_plan(source).launches), 129)
        with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                            execution_instruction_limit=128_000_000)):
            plan = application.prepare_plan(source)
            self.assertEqual(plan.chunk_elements, 131072)
            self.assertEqual(len(plan.launches), 2)
            self.assertEqual(plan.launches[-1]['input_bytes'], 6)
            self.assertEqual(plan.launches[-1]['output_offset_bytes'], 262144)
        self.assertEqual(functional_instruction_limit(), 1_000_000)

    @unittest.skipUnless((BIN / 'atlas-opt').is_file(), 'select built Atlas tools')
    def test_larger_dispatch_replays_bound_and_obeys_static_budget(self):
        source = request('bf16_add', 2048)
        with tempfile.TemporaryDirectory(prefix='dispatch-budget-') as directory:
            compiled = Path(directory) / 'compiled'
            options = CodegenOptions(execution_instruction_limit=2_000_000)
            with codegen_options(options):
                device.compile_source(None, source, target_source=ROOT,
                                      destination=compiled, limits=None)
                bound = json.loads((compiled / 'float-instruction-bound.json').read_text())
                self.assertGreater(bound['instruction_bound'], 1_000_000)
                self.assertLessEqual(bound['instruction_bound'], 2_000_000)
                self.assertEqual(bound['max_instructions'], 2_000_000)
                device.prepare_c_package(source, compiled, 'larger', target_source=ROOT)
            # An artifact compiled with a larger budget cannot pass the smaller
            # caller's independent execution-bound replay.
            with self.assertRaisesRegex(ValueError, 'instruction bound'):
                device.prepare_c_package(source, compiled, 'smaller', target_source=ROOT)
            with codegen_options(CodegenOptions(execution_instruction_limit=2_000_000,
                                                program_size_limit_bytes=128)):
                with self.assertRaises((NativeCompilationError, ValueError)):
                    device.compile_source(None, source, target_source=ROOT,
                                          destination=Path(directory) / 'too-large', limits=None)

    @unittest.skipUnless((BIN / 'atlas-opt').is_file(), 'select built Atlas tools')
    def test_large_binary_with_late_invalid_input_keeps_output_private(self):
        from float_isa_c import executor
        self.addCleanup(executor.cache_clear)
        count = 16385
        source = request('bf16_add', count)
        raw = [bytes.fromhex('803f') * count, bytes.fromhex('003f') * count]
        limit = 32_000_000
        with codegen_options(CodegenOptions(execution_instruction_limit=limit)):
            actual, status, steps = scalar_execute(source, raw, max_instructions=limit)
            self.assertEqual((actual, status), host_reference(source, raw))
            self.assertLess(steps, limit)
            raw[1] = raw[1][:-2] + bytes.fromhex('c17f')
            actual, status, _ = scalar_execute(source, raw, max_instructions=limit)
            self.assertNotEqual(status, 0)
            self.assertEqual(actual, bytes([0xa5]) * len(actual))

    @unittest.skipUnless((BIN / 'atlas-opt').is_file(), 'select built Atlas tools')
    def test_counted_contraction_records_and_replays_selected_budget(self):
        from atlas_mlir_native import quantized_loop, quantized_loop_c
        from test_already_quantized import source_request
        from test_looped_mnk import build_snapshot
        from merlin.semantic_compiler.snapshot import open_native_snapshot
        from merlin.semantic_compiler.search import SearchLimits

        source = source_request(32, 32, 64, batch=())
        with tempfile.TemporaryDirectory(prefix='counted-budget-') as directory:
            compiled = Path(directory) / 'compiled'
            with codegen_options(CodegenOptions(execution_instruction_limit=2_000_000)):
                existing = os.environ.get('ATLAS_TEST_NATIVE_SNAPSHOT')
                snapshot = open_native_snapshot(Path(existing)) if existing else build_snapshot(Path(directory) / 'native')
                from atlas_mlir_native.looped_tiling import tuning_key
                selected_key = tuning_key(snapshot, source, {}, (), SearchLimits())
                with codegen_options(CodegenOptions()):
                    default_key = tuning_key(snapshot, source, {}, (), SearchLimits())
                self.assertNotEqual(selected_key, default_key)
                self.assertEqual(selected_key['execution_instruction_limit'], 2_000_000)
                self.assertNotIn('execution_instruction_limit', default_key)
                quantized_loop.compile_source(snapshot, source, target_source=ROOT,
                                             destination=compiled, limits=SearchLimits())
                route = json.loads((compiled / 'quantized-route.json').read_text())
                self.assertEqual(route['physical_instruction_limit'], 2_000_000)
                self.assertEqual(route['launches_per_invocation'], 1)
                quantized_loop_c.prepare_c_package(source, compiled, 'contraction', target_source=ROOT)
            with self.assertRaisesRegex(ValueError, 'selected profile'):
                quantized_loop_c.prepare_c_package(source, compiled, 'wrong_budget', target_source=ROOT)


if __name__ == '__main__':
    unittest.main()
