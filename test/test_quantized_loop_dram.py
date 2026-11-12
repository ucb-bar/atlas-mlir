"""Counted contraction admission includes every selected DRAM frame byte."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native import quantized_loop as loop, quantized_loop_c as native_c
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop_sizing import size_counted, untimed_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_quantized_epilogue import case

ROOT = Path(__file__).resolve().parents[1]
CACHE_DRAM = 67108864
SMALL_DRAM = 4325376


def cache_source(m=4, n=2048, k=2048, *, batch=(1,), op='atlas.qlinear',
                 batched_rhs=False, scalar_lhs=False):
    source = source_request(m, n, k, batch)
    nodes = []
    for node in source.nodes:
        if node.id == 'sx':
            node = replace(node, type=replace(node.type, shape=() if scalar_lhs else (*batch, 1, 1)))
        elif batched_rhs and node.id in ('w', 'sw'):
            node = replace(node, type=replace(node.type,
                shape=(*batch, n, k) if node.id == 'w' else (*batch, 1, 1)))
        elif node.id == 'y':
            node = replace(node, op=op)
            if op == 'atlas.qk_matmul':
                node = replace(node, type=replace(node.type, dtype='f32'),
                    attrs=tuple((key, 'float32' if key == 'output_dtype' else value)
                                for key, value in node.attrs))
        nodes.append(node)
    return replace(source, nodes=tuple(nodes))


class QuantizedLoopDramTest(unittest.TestCase):
    def test_synthetic_small_profile_includes_complete_status_and_exponent_frame(self):
        for source,expected in ((cache_source(),4526080),(cache_source(k=5632),11980800)):
            plan = prepare_source(source)
            with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)):
                layout = loop.execution_layout(plan)
                self.assertEqual(layout['status_address']+layout['status_bytes']-DRAM_BASE,expected)
            with codegen_options(CodegenOptions(io_memory_bytes=SMALL_DRAM)):
                selection = size_counted(plan,target_source=ROOT).record()
                self.assertEqual((selection['route'],selection['reason']),('k32_tiles','selected_dram_window'))
                with self.assertRaisesRegex(NativeCompilationError,'selected DRAM window'):
                    loop.execution_layout(plan)

    def test_tiny_frame_alignment_status_page_and_invalid_window(self):
        plan = prepare_source(source_request(1,1,1,batch=()))
        with codegen_options(CodegenOptions(io_memory_bytes=12288)):
            layout = loop.execution_layout(plan)
            self.assertEqual(layout['input_buffer_bytes'],[1024,1024])
            self.assertEqual(layout['input_addresses'],[DRAM_BASE,DRAM_BASE+2048])
            self.assertEqual(layout['exponent_addresses'],[DRAM_BASE+4096,DRAM_BASE+6144])
            self.assertEqual(layout['output_addresses'],[DRAM_BASE+8192])
            self.assertEqual((layout['status_address'],layout['status_bytes']),(DRAM_BASE+10240,2048))
        for window in (2048,10240):
            with self.subTest(window=window),codegen_options(CodegenOptions(io_memory_bytes=window)):
                self.assertEqual(size_counted(plan,target_source=ROOT).record()['reason'],'selected_dram_window')
                with self.assertRaisesRegex(NativeCompilationError,'selected DRAM window'):
                    loop.execution_layout(plan)
        for window in (12287,12289):
            with self.subTest(window=window),codegen_options(CodegenOptions(io_memory_bytes=window)):
                with self.assertRaisesRegex(NativeCompilationError,'whole tiles'):
                    size_counted(plan,target_source=ROOT)

    def test_only_epilogue_footprint_overflow_retains_original_device_operations(self):
        from atlas_mlir_native.quantized_epilogue import alternatives,prepare_source as prepare_epilogue
        source,contraction,choices = case(m=1,n=1,k=1,batch=())
        with codegen_options(CodegenOptions(io_memory_bytes=12288)):
            self.assertEqual(loop.execution_layout(prepare_source(contraction))['status_address']+2048-DRAM_BASE,12288)
            epilogue = prepare_epilogue(source)
            self.assertEqual(size_counted(epilogue,target_source=ROOT).record()['reason'],'selected_dram_window')
            adapter = SimpleNamespace()
            self.assertEqual(alternatives(adapter,source,choices),choices)
            self.assertEqual(adapter._quantized_epilogue_groups,{})

    def test_epilogue_tool_timeout_is_not_silently_declined(self):
        from atlas_mlir_native.quantized_epilogue import alternatives
        source,_,choices = case(m=1,n=1,k=1,batch=())
        with patch('atlas_mlir_native.quantized_loop_sizing.size_counted',
                   side_effect=NativeCompilationError('search_timeout','synthetic tool timeout')):
            with self.assertRaisesRegex(NativeCompilationError,'synthetic tool timeout'):
                alternatives(SimpleNamespace(),source,choices)

    def test_retained_sizing_detects_changed_evidence_without_permissions(self):
        from atlas_mlir_native import quantized_loop_sizing as sizing
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory/'resource.json'
            path.write_bytes(b'original resource plan')
            retained = sizing._retain_checked(directory, {'native_publication': False})
            self.assertEqual(retained.read('resource.json'), b'original resource plan')
            path.write_bytes(b'changed resource plan')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                retained.record()
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                retained.read('resource.json')

    def test_identical_checked_timeout_is_retained_without_retry(self):
        from atlas_mlir_native import quantized_loop_sizing as sizing
        plan = prepare_source(source_request())
        identity = dict(pipeline='mandatory', capture_postspill=False,
                        effective_pipeline_arguments=['--atlas-compile-unscheduled'])
        def resource_plan(command, **kwargs):
            option = next(arg for arg in command if 'resource-planning-report=' in arg)
            path = Path(option.split('resource-planning-report=', 1)[1].split(' ', 1)[0])
            path.write_text(json.dumps({'cfg_contract': 'contract'}))
            return 'prepared virtual'
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)), \
             patch.object(sizing, '_CACHE', {}), \
             patch.object(sizing, '_identity', return_value=identity), \
             patch.object(loop, 'virtual_source', return_value='source virtual'), \
             patch.object(sizing, '_run', side_effect=resource_plan) as planning, \
             patch.object(sizing, 'untimed_bound', return_value=dict(
                 instruction_bound=1, exceeds_physical_instruction_limit=False)), \
             patch('atlas_mlir_native.region_composition.compile_virtual_with_receipt',
                   side_effect=NativeCompilationError('search_timeout', 'checked wall exhausted')) as checked:
            for attempt in range(2):
                with self.assertRaisesRegex(NativeCompilationError, 'checked wall exhausted') as caught:
                    size_counted(plan, target_source=ROOT)
                self.assertEqual(caught.exception.status, 'search_timeout')
                if attempt:
                    self.assertIn('retained identical sizing failure', caught.exception.reason)
            planning.assert_called_once()
            checked.assert_called_once()

    def test_late_success_does_not_replace_retained_same_key_timeout(self):
        from atlas_mlir_native import quantized_loop_sizing as sizing
        plan = prepare_source(source_request())
        identity = dict(pipeline='mandatory', capture_postspill=False,
                        pipeline_arguments=['--atlas-compile-unscheduled'],
                        effective_pipeline_arguments=['--atlas-compile-unscheduled'])

        def resource_plan(command, **kwargs):
            if '--program-json' in command:
                return json.dumps(dict(selected_rtl_revision=sizing.RTL_REVISION, instructions=[]))
            option = next(arg for arg in command if 'resource-planning-report=' in arg)
            path = Path(option.split('resource-planning-report=', 1)[1].split(' ', 1)[0])
            path.write_text(json.dumps(dict(cfg_contract='contract', untimed_words=dict(limit=32768))))
            return 'prepared virtual'

        attempts = []
        def finish_after_other_probe_failed(*args, **kwargs):
            attempts.append(kwargs['destination'])
            if len(attempts) == 2:
                raise NativeCompilationError('search_timeout', 'checked wall exhausted')
            # Both probes missed the cache. Complete the second probe's checked
            # timeout before publishing the first probe's successful decision.
            with self.assertRaisesRegex(NativeCompilationError, 'checked wall exhausted'):
                size_counted(plan, target_source=ROOT)
            return 'checked machine with contract'

        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)), \
             patch.object(sizing, '_CACHE', {}), \
             patch.object(sizing, '_identity', return_value=identity), \
             patch.object(loop, 'virtual_source', return_value='source virtual'), \
             patch.object(sizing, '_run', side_effect=resource_plan) as planning, \
             patch.object(sizing, 'untimed_bound', return_value=dict(
                 instruction_bound=1, exceeds_physical_instruction_limit=False)), \
             patch.object(sizing, 'instruction_bound', return_value=dict(instruction_bound=1, admitted=True)), \
             patch('atlas_mlir_native.region_composition.compile_virtual_with_receipt',
                   side_effect=finish_after_other_probe_failed) as checked:
            result = size_counted(plan, target_source=ROOT)
            self.assertEqual(result.record()['reason'], 'physical_bound_admitted')
            with self.assertRaisesRegex(NativeCompilationError, 'retained identical sizing failure') as caught:
                size_counted(plan, target_source=ROOT)
            self.assertEqual(caught.exception.status, 'search_timeout')
            self.assertEqual(planning.call_count, 3)  # Two plans and one successful emission.
            self.assertEqual(checked.call_count, 2)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'),'select native compiler tools')
    def test_only_explicit_instruction_memory_report_selects_resource_fallback(self):
        from atlas_mlir_native import quantized_loop_sizing as sizing
        source = replace(cache_source(n=256),source_identity='instruction-memory-report-control')
        def refused(command,**kwargs):
            option = next(arg for arg in command if 'resource-planning-report=' in arg)
            path = Path(option.split('resource-planning-report=',1)[1].split(' ',1)[0])
            path.write_text(json.dumps(dict(schema='atlas.virtual_resource_plan.v1',
                planning_succeeded=False,machine_artifact_qualified=False,
                untimed_words=dict(words=32769,limit=32768))))
            raise NativeCompilationError('compile_error','synthetic explicit IMEM refusal')
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)),patch.object(sizing,'_run',side_effect=refused):
            result = size_counted(prepare_source(source),target_source=ROOT).record()
            self.assertEqual((result['route'],result['reason']),('k32_tiles','encoded_instruction_memory'))
            self.assertEqual((result['planned_words'],result['program_word_limit']),(32769,32768))
        source = replace(source,source_identity='timeout-is-not-an-IMEM-refusal')
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)),patch.object(sizing,'_run',
                side_effect=NativeCompilationError('search_timeout','synthetic timeout')):
            with self.assertRaisesRegex(NativeCompilationError,'synthetic timeout'):
                size_counted(prepare_source(source),target_source=ROOT)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'),'select native compiler tools')
    def test_actual_cache64_profile_keeps_all_fitting_projections_counted(self):
        cases = (
            ('Q',cache_source(),4526080),('K',cache_source(n=256),626688),
            ('QK',cache_source(n=4,k=64,batch=(1,32),op='atlas.qk_matmul',batched_rhs=True),329728),
            ('V',cache_source(n=256),626688),
            ('PV',cache_source(n=64,k=4,batch=(1,32),op='atlas.pv_matmul',batched_rhs=True,scalar_lhs=True),299008),
            ('O',cache_source(),4526080),('up',cache_source(n=5632),12324864),
            ('gate',cache_source(n=5632),12324864),('down',cache_source(k=5632),11980800))
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)):
            for name,source,expected in cases:
                with self.subTest(name=name):
                    selection = size_counted(prepare_source(source),target_source=ROOT).record()
                    self.assertEqual(selection['counted_frame_bytes'],expected)
                    self.assertEqual(selection['route'],'counted')
                    self.assertLessEqual(selection['physical_instruction_bound'],1000000)
                    self.assertFalse(selection['native_publication'])
                    self.assertEqual(selection['selected_io_memory_bytes'],CACHE_DRAM)
                    self.assertTrue(selection['checks_covered_by_immutable_mandatory_result'])
                    self.assertEqual(len(selection['checked_commands']),1)
                    self.assertTrue(selection['checked_commands'][0][0].endswith('/atlas-emit'))

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'),'select native compiler tools')
    def test_shape_budget_rejects_large_image_before_full_proof_and_removes_old_cutoff(self):
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)):
            large = size_counted(prepare_source(source_request(512,5632,2048,batch=())),target_source=ROOT)
            record = large.record()
            self.assertEqual((record['route'],record['reason']),('k32_tiles','untimed_instruction_bound'))
            self.assertGreater(record['untimed_instruction_bound'],1000000)
            self.assertNotIn('checked/atlas-machine.mlir',dict(large.files))
            report = json.loads(large.read('resource.json'))
            for field in ('planned_helper_control','machine_artifact_qualified'):
                changed = dict(report);changed[field] = True
                with self.assertRaises(ValueError):
                    untimed_bound(large.read('source-virtual.mlir').decode(),changed)
            # This complete K32 frame lies beyond the removed368/352 split,
            # but its actual checked image fits; it must remain device-looped.
            narrow = size_counted(prepare_source(source_request(32,11776,32,batch=())),target_source=ROOT).record()
            self.assertEqual(narrow['route'],'counted')
            self.assertLessEqual(narrow['physical_instruction_bound'],1000000)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'),'select native compiler tools')
    def test_checked_cache_is_profile_bound_and_detects_changed_files(self):
        plan = prepare_source(source_request())
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)):
            selected = size_counted(plan,target_source=ROOT)
            self.assertIs(size_counted(plan,target_source=ROOT),selected)
            path = selected.directory/'checked/atlas-machine.mlir'
            original = path.read_bytes()
            try:
                path.chmod(0o644);path.write_bytes(original+b'\n')
                with self.assertRaisesRegex(ValueError,'cache changed'):
                    size_counted(plan,target_source=ROOT)
            finally:
                path.write_bytes(original);path.chmod(0o444)
        with codegen_options(CodegenOptions(io_memory_bytes=SMALL_DRAM)):
            other = size_counted(plan,target_source=ROOT)
            self.assertNotEqual(other.record()['identity']['profile_sha256'],selected.record()['identity']['profile_sha256'])
            self.assertIsNot(other,selected)


if __name__ == '__main__':
    unittest.main()
