"""Complete rows batch synchronously without publishing partial results."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.row_loop import prepare_plan
from test_tiled_softmax import source_request


class BoundedRowPlanTest(unittest.TestCase):
    def setUp(self):
        from atlas_mlir_native.options import CodegenOptions, codegen_options
        self.options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        self.options.__enter__()
        self.addCleanup(self.options.__exit__, None, None, None)

    def test_selected_base_offsets_and_explicit_plan_override(self):
        from atlas_mlir_native.binding import DRAM_BASE
        from atlas_mlir_native.options import CodegenOptions, codegen_options

        base = 0xa0000000
        source = source_request((65, 64))
        with codegen_options(CodegenOptions(io_memory_base=base, io_memory_bytes=67108864)):
            plan = prepare_plan(source)
            self.assertEqual(plan.address(base, 0), base)
            self.assertEqual(plan.address(base, 1), base + plan.readonly_bytes)
            self.assertEqual(plan.address(base + plan.readonly_bytes, 1),
                             plan.writable_base + plan.writable_bytes)
            # The body retains invocation addresses while an explicit plan
            # override selects a separate destination frame.
            override = prepare_plan(source, io_memory_base=DRAM_BASE)
            self.assertEqual(override.address(base, 0), DRAM_BASE)
            self.assertEqual(override.address(base + override.readonly_bytes, 1),
                             override.writable_base + override.writable_bytes)

    def test_window512_keeps_rows_and_all_groups(self):
        plan = prepare_plan(source_request((16384, 512), masked=True))
        self.assertEqual((plan.groups, plan.groups_per_launch, plan.launches), (512, 32, 16))
        self.assertEqual(plan.groups_per_launch * plan.tiling.column_tiles, 512)
        execution = plan.execution_record()
        self.assertEqual(execution['groups_per_launch'], 32)
        self.assertEqual(execution['launches_per_invocation'], 16)

    def test_fixed64_retains_one_loop_and_prime_tail_has_complete_rows(self):
        plan = prepare_plan(source_request((2048, 64), masked=True))
        self.assertEqual((plan.groups, plan.groups_per_launch, plan.launches), (64, 64, 1))
        self.assertNotIn('groups_per_launch', plan.execution_record())
        plan = prepare_plan(source_request((8193, 64), masked=True))
        self.assertEqual((plan.groups, plan.groups_per_launch, plan.launches), (257, 1, 257))
        self.assertEqual(plan.tiling.regions[-1].valid_rows, 1)


@unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR')
                     and os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select native tools and original model')
class BoundedRowRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.row_loop import compile_row_loop
        from atlas_mlir_native.softmax_c import prepare_c_package
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import ROOT, build_snapshot

        # A smaller test envelope exercises the exact production batching path
        # with three launches, two full row groups, and a one-row tail.
        cls.patch = patch('atlas_mlir_native.row_loop.ROW_LOOP_COLUMN_TILES_PER_LAUNCH', 2)
        cls.patch.start()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        snapshot = build_snapshot(cls.root / 'snapshot')
        root = cls.root / 'case-batched-row-loop'
        request = source_request((65, 64))
        compile_row_loop(snapshot, request, target_source=ROOT, destination=root, limits=SearchLimits())
        plan = prepare_plan(request)
        if plan.launches != 3 or plan.groups_per_launch != 1:
            raise AssertionError('test did not select repeated complete-row batches')
        record, files = prepare_c_package(root / 'source-request.json', root, 'layer', target_source=ROOT)
        package = root / 'c-package'
        package.mkdir()
        for filename, data in files.items():
            path = package / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        cls.cases = [(root, plan.tiling, [root], record, package)]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.patch.stop()

    def authenticate_model_source(self):
        from test_mxu1_cap448 import authenticate_model
        authenticate_model(Path(os.environ['IMPORT_MODEL_ROOT']))

    # Reuse the independent reference and host failure harness (random VMEM, exact outputs, no early reuse,
    # late read failure, and timeout recovery), then authenticate every model module the run imported.
    def test_numeric_and_failure_paths(self):
        from test_softmax_c_package import SoftmaxCPackageTest
        SoftmaxCPackageTest.test_c_calls_all_row_groups_without_partial_outputs_or_early_reuse(self)
        self.authenticate_model_source()


if __name__ == '__main__':
    unittest.main()
