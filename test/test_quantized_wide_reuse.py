"""Wider bounded activation caching keeps the original ordered arithmetic."""
import os
from math import prod
from unittest.mock import patch
import unittest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_operand_reuse import activation_residency, WIDE_CACHE_TILES
from atlas_mlir_native.restoration_budget import instruction_bound
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture
from test_quantized_operand_reuse import cycle


class WideReuseTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        options = codegen_options(CodegenOptions(io_memory_bytes=64*1024*1024))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_recorded_candidate_bounds_and_tamper_rejection(self):
        for k in (32, 2048, 2049, 5632, 6144, 6145, 8192):
            plan = prepare_source(source_request(m=1,n=33,k=k,batch=()))
            cache = activation_residency(plan, 'wide-cache')
            self.assertEqual(cache.tile_count, min((k+31)//32,WIDE_CACHE_TILES))
            layout = execution_layout(plan,operand_reuse='wide-cache')
            with patch.dict(os.environ,ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
                virtual_source(plan,layout)
            layout['operand_reuse_cached_tiles'] -= 1
            with self.assertRaisesRegex(ValueError,'schedule differs'):
                virtual_source(plan,layout)
        plan = prepare_source(source_request(m=1,n=32,k=5632,batch=()))
        self.assertIsNone(activation_residency(plan,'wide-cache'))

    def test_native_wide_publication_and_rebinding(self):
        import tempfile
        from pathlib import Path
        from atlas_mlir_native import quantized_loop, quantized_loop_c
        from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
        from merlin.semantic_compiler.materialization import canonical_request
        from merlin.semantic_compiler.search import SearchLimits
        from test_exponent_restoration import ROOT
        from test_looped_mnk import build_snapshot
        source = source_request(m=1,n=33,k=5632,batch=())
        canonical = canonical_request(source,source_identity_policy='provenance')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = SelectionCapture(build_snapshot(root/'snapshot'))
            with patch.dict(os.environ,ATLAS_QUANTIZED_OPERAND_REUSE='wide-cache'):
                quantized_loop.compile_source(capture,canonical,target_source=ROOT,
                    destination=root/'compiled',limits=SearchLimits())
            with patch.dict(os.environ,ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
                bind_counted(source,canonical,root/'compiled',root/'bound',capture,target_source=ROOT)
                record, files = quantized_loop_c.prepare_c_package(source,root/'bound','wide',target_source=ROOT)
                self.assertEqual(record['output_node_ids'],list(source.outputs))
                self.assertTrue(files)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned model')
    def test_full_row_and_streamed_tail_match_reference_and_read_counts(self):
        for m,n,k,batch in ((1,33,5632,()),(33,35,6145,()),(1,33,2049,(2,))):
            plan = prepare_source(source_request(m=m,n=n,k=k,batch=batch))
            layouts,programs = [],[]
            for policy in ('ordered','wide-cache'):
                layout = execution_layout(plan,operand_reuse=policy)
                virtual = virtual_source(plan,layout)
                machine,physical = compile_fixture(virtual)
                self.assertTrue(instruction_bound(virtual,machine,physical)['admitted'])
                self.assertLessEqual(4*len(physical['instructions']),131072)
                layouts.append(layout);programs.append(physical)
            memory = memory_fixture(plan,layouts[0])
            for refused in (False,True):
                if refused:
                    at = layouts[0]['exponent_addresses'][0]-DRAM_BASE
                    memory[at:at+4] = (1000).to_bytes(4,'little')
                expected,baseline = cycle(programs[0],layouts[0],memory)
                actual,reused = cycle(programs[1],layouts[1],memory)
                self.assertEqual(actual,expected)
                at = layouts[0]['status_address']-DRAM_BASE
                self.assertEqual(bool(int.from_bytes(actual[at:at+4],'little')),refused)
                saved = prod(batch)*((m+31)//32)*((n+31)//32-1)*min((k+31)//32,192)*1024
                self.assertEqual(32*(baseline.load_beats-reused.load_beats),saved)
                self.assertEqual(baseline.store_beats,reused.store_beats)


if __name__ == '__main__':
    unittest.main()
