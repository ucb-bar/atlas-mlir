"""Wider bounded activation caching keeps the original ordered arithmetic."""
from atlas_mlir_native.options import codegen_options, current_codegen_options

from dataclasses import replace
import os
from math import prod
import unittest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_operand_reuse import activation_residency, WIDE_CACHE_TILES
from test_already_quantized import source_request
from test_quantized_operand_reuse import LargeArenaTest, admitted_programs, compare_cycles, export_recorded_policy, memory_fixture


class WideReuseTest(LargeArenaTest):
    def test_recorded_candidate_bounds_and_tamper_rejection(self):
        for k in (32, 2048, 2049, 5632, 6144, 6145, 8192):
            plan = prepare_source(source_request(m=1,n=33,k=k,batch=()))
            cache = activation_residency(plan, 'wide-cache')
            self.assertEqual(cache.tile_count, min((k+31)//32,WIDE_CACHE_TILES))
            layout = execution_layout(plan,operand_reuse='wide-cache')
            with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered')):
                virtual_source(plan,layout)
            layout['operand_reuse_cached_tiles'] -= 1
            with self.assertRaisesRegex(ValueError,'schedule differs'):
                virtual_source(plan,layout)
        plan = prepare_source(source_request(m=1,n=32,k=5632,batch=()))
        self.assertIsNone(activation_residency(plan,'wide-cache'))

    def test_native_wide_publication_and_rebinding(self):
        import tempfile
        from pathlib import Path
        from merlin.semantic_compiler.materialization import canonical_request
        source = source_request(m=1,n=33,k=5632,batch=())
        canonical = canonical_request(source,source_identity_policy='provenance')
        with tempfile.TemporaryDirectory() as directory:
            export_recorded_policy(self,source,canonical,Path(directory),'wide',quantized_operand_reuse='wide-cache')

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned model')
    def test_full_row_and_streamed_tail_match_reference_and_read_counts(self):
        for m,n,k,batch in ((1,33,5632,()),(33,35,6145,()),(1,33,2049,(2,))):
            plan = prepare_source(source_request(m=m,n=n,k=k,batch=batch))
            layouts = [execution_layout(plan,operand_reuse=policy) for policy in ('ordered','wide-cache')]
            programs = admitted_programs(self,[virtual_source(plan,layout) for layout in layouts])
            saved = prod(batch)*((m+31)//32)*((n+31)//32-1)*min((k+31)//32,192)*1024
            for _,baseline,reused in compare_cycles(self,programs,layouts,memory_fixture(plan,layouts[0])):
                self.assertEqual(32*(baseline.load_beats-reused.load_beats),saved)
                self.assertEqual(baseline.store_beats,reused.store_beats)


if __name__ == '__main__':
    unittest.main()
