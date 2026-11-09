"""Complete rows batch synchronously without publishing partial results."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from atlas_mlir_native.row_loop import prepare_plan
from test_tiled_softmax import source_request


class BoundedRowPlanTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        self.options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        self.options.__enter__()
        self.addCleanup(self.options.__exit__, None, None, None)

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
        root = Path(os.environ['IMPORT_MODEL_ROOT']).resolve()
        pin = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
        checked = 0
        for key, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if not (key == 'npu_model' or key.startswith('npu_model.')) or not filename:
                continue
            path = Path(filename).resolve()
            relative = path.relative_to(root)
            pinned = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', f'{pin}:{relative}'])
            self.assertEqual(path.read_bytes(), pinned)
            checked += 1
        self.assertGreater(checked, 0)

    # Reuse the established independent numerical/reference and host failure
    # harness: random initial VMEM, exact outputs, no read/reuse before wait,
    # late read failure preserving caller output, and timeout recovery checks.
    def test_numeric_and_failure_paths(self):
        from test_softmax_c_package import SoftmaxCPackageTest
        SoftmaxCPackageTest.test_c_calls_all_row_groups_without_partial_outputs_or_early_reuse(self)
        self.authenticate_model_source()


if __name__ == '__main__':
    unittest.main()
