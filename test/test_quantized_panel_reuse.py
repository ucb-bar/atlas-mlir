"""Paired rows share operands without changing per-output K order."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.exponent_restoration import Restoration
from atlas_mlir_native.quantized_loop import execution_layout,virtual_source as ordinary_source
from atlas_mlir_native.quantized_panel_reuse import admitted,virtual_source,PANEL_K_TILES
from atlas_mlir_native.region_composition import rematerialize_scalar_constants
from atlas_mlir_native.restoration_budget import fallback_tiles,instruction_bound
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture
from test_quantized_operand_reuse import cycle


def panel_virtual(plan,layout):
    return rematerialize_scalar_constants(virtual_source(plan,layout,
        Restoration(plan,layout,scalar_fallback=bool(fallback_tiles(plan)))))


class PanelReuseTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions,codegen_options
        options=codegen_options(CodegenOptions(io_memory_bytes=64*1024*1024))
        options.__enter__()
        self.addCleanup(options.__exit__,None,None,None)

    def test_small_odd_batched_and_unbounded_geometries_decline(self):
        for m,n,k,batch in ((32,64,1024,()),(65,64,1024,()),(64,32,1024,()),
                            (64,64,65,()),(64,64,6145,()),(64,64,1024,(2,))):
            plan=prepare_source(source_request(m=m,n=n,k=k,batch=batch))
            self.assertFalse(admitted(plan))
            with self.assertRaises(ValueError):
                panel_virtual(plan,execution_layout(plan))
            fallback=execution_layout(plan,operand_reuse='paired-cache')
            wide=execution_layout(plan,operand_reuse='wide-cache')
            self.assertEqual(ordinary_source(plan,fallback),ordinary_source(plan,wide))
        for m in (33,64,97,128):
            plan=prepare_source(source_request(m=m,n=35,k=1024,batch=()))
            self.assertTrue(admitted(plan))

    def test_recorded_policy_replay_and_native_rebinding(self):
        import tempfile
        from pathlib import Path
        from atlas_mlir_native import quantized_loop,quantized_loop_c
        from atlas_mlir_native.selection_materialization import SelectionCapture,bind_counted
        from merlin.semantic_compiler.materialization import canonical_request
        from merlin.semantic_compiler.search import SearchLimits
        from test_exponent_restoration import ROOT
        from test_looped_mnk import build_snapshot
        source=source_request(m=33,n=35,k=1025,batch=())
        plan=prepare_source(source)
        layout=execution_layout(plan,operand_reuse='paired-cache')
        self.assertEqual(layout['operand_reuse'],'paired-cache')
        for key in ('operand_reuse_panel_m_tiles','operand_reuse_panel_k_tiles','operand_reuse_cached_tiles'):
            changed={**layout,key:1}
            with self.assertRaisesRegex(ValueError,'schedule differs'):
                ordinary_source(plan,changed)
        canonical=canonical_request(source,source_identity_policy='provenance')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            capture=SelectionCapture(build_snapshot(root/'native'))
            with patch.dict(os.environ,ATLAS_QUANTIZED_OPERAND_REUSE='paired-cache'):
                quantized_loop.compile_source(capture,canonical,target_source=ROOT,
                    destination=root/'compiled',limits=SearchLimits())
            with patch.dict(os.environ,ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
                bind_counted(source,canonical,root/'compiled',root/'bound',capture,target_source=ROOT)
                record,files=quantized_loop_c.prepare_c_package(source,root/'bound','paired',target_source=ROOT)
                self.assertEqual(record['output_node_ids'],list(source.outputs))
                self.assertTrue(files)
                self.assertEqual(ordinary_source(plan,layout),panel_virtual(plan,layout))

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned model')
    def test_optimized_pipeline_matches_original_memory(self):
        plan=prepare_source(source_request(m=33,n=35,k=1025,batch=()))
        layout=execution_layout(plan,operand_reuse='paired-cache')
        baseline=execution_layout(plan,operand_reuse='ordered')
        _,reference=compile_fixture(ordinary_source(plan,baseline))
        virtual=ordinary_source(plan,layout)
        machine,program=compile_fixture(virtual,pipeline='optimized')
        self.assertTrue(instruction_bound(virtual,machine,program)['admitted'])
        memory=memory_fixture(plan,layout)
        expected,_=cycle(reference,baseline,memory)
        actual,_=cycle(program,layout,memory)
        self.assertEqual(actual,expected)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned model')
    def test_paired_panels_match_full_row_cache_with_exact_traffic(self):
        for m,n,k in ((33,35,1025),(97,65,2049),(64,65,5632)):
            source=source_request(m=m,n=n,k=k,batch=())
            source=replace(source,nodes=tuple(replace(node,type=replace(node.type,shape=(m,1)))
                if node.id=='sx' else node for node in source.nodes))
            plan=prepare_source(source)
            layout=execution_layout(plan,operand_reuse='wide-cache')
            memory=memory_fixture(plan,layout)
            programs=[]
            for virtual in (ordinary_source(plan,layout),panel_virtual(plan,layout)):
                machine,program=compile_fixture(virtual)
                self.assertTrue(instruction_bound(virtual,machine,program)['admitted'])
                self.assertLessEqual(4*len(program['instructions']),131072)
                programs.append(program)
            for refused in (False,True):
                if refused:
                    at=layout['exponent_addresses'][1]-DRAM_BASE+((n-1)//32*512+(n-1)%32)*4
                    memory[at:at+4]=(1000).to_bytes(4,'little')
                expected,old=cycle(programs[0],layout,memory)
                actual,new=cycle(programs[1],layout,memory)
                self.assertEqual(actual,expected)
                at=layout['status_address']-DRAM_BASE
                self.assertEqual(bool(int.from_bytes(actual[at:at+4],'little')),refused)
                mc,nc,kc=((v+31)//32 for v in (m,n,k))
                phases=(kc+PANEL_K_TILES-1)//PANEL_K_TILES
                saved_weights=(mc//2)*nc*kc*1024
                partial_reads=phases*mc*nc*2048
                self.assertEqual(32*(old.load_beats-new.load_beats),saved_weights-partial_reads)
                self.assertEqual(32*(new.store_beats-old.store_beats),partial_reads)
                self.assertLess(new.load_beats+new.store_beats,old.load_beats+old.store_beats)
                if not refused:
                    self.assertLess(new.cycles,old.cycles)


if __name__=='__main__':
    unittest.main()
