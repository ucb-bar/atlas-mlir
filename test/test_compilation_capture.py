"""Post-spill audit capture is opt-in and preserves single-call publication."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_looped_mnk import ROOT, build_snapshot

BIN=Path(os.environ.get('ATLAS_OOT_BIN_DIR',ROOT/'build/bin'))


class CompilationCaptureOptionsTest(unittest.TestCase):
    def test_default_compile_does_not_request_or_parse_ir_dumps(self):
        from atlas_mlir_native import binding
        from atlas_mlir_native.region_composition import compile_virtual_with_receipt

        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)
            tools=destination/'tools';tools.mkdir()
            for name in ('atlas-opt','atlas-emit'):(tools/name).write_bytes(name.encode())
            with patch.dict(os.environ, {'ATLAS_CAPTURE_POSTSPILL':'0'}):
                with patch.object(binding,'_run',return_value='machine') as run:
                    result=compile_virtual_with_receipt('virtual',tools=tools,pipeline='mandatory',destination=destination)
            self.assertEqual(result,'machine')
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0],[str(tools/'atlas-opt'),*binding.NATIVE_PIPELINES['mandatory'],'-'])
            receipt=json.loads((destination/'compilation-pipeline.json').read_text())
            self.assertNotIn('spilled_virtual_sha256',receipt)
            self.assertFalse((destination/'mandatory-spilled-virtual.mlir').exists())


@unittest.skipUnless((BIN/'atlas-opt').is_file() and shutil.which('cargo'), 'select Atlas tools and offline Rust bridge')
class CompilationCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        cls.root=Path(cls.temporary.name)
        cls.snapshot=build_snapshot(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def check_capture(self, destination, manifest):
        receipt=json.loads((destination/'compilation-pipeline.json').read_text())
        self.assertEqual(receipt['argv'][-2:],['--atlas-compile-unscheduled','-'])
        self.assertEqual(receipt['wall_limit_s'],30)
        for key,name in [('input_sha256','atlas-virtual.mlir'),('output_sha256','atlas-machine.mlir'),
                         ('spilled_virtual_sha256','mandatory-spilled-virtual.mlir')]:
            self.assertEqual(receipt[key],hashlib.sha256((destination/name).read_bytes()).hexdigest())
            if name!='atlas-machine.mlir':self.assertIn(name,manifest['stage_artifacts'])
        self.assertIn('compilation-pipeline.json',manifest['stage_artifacts'])

    @patch.dict(os.environ, {'ATLAS_CAPTURE_POSTSPILL':'1'})
    def test_pointwise_capture_survives_publication_and_export(self):
        from atlas_mlir_native.binding import TARGET_IDENTITY
        from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY, compile_pointwise, plan_abi, prepare_source
        from atlas_mlir_native.pointwise_c import prepare_c_package
        from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        type_=TensorType((2,3),'bf16','raw-bit-pattern')
        inputs=tuple(SemanticNode(key,'input',(),type_,effect='input') for key in ('a','b'))
        result=SemanticNode('result','bf16_mul',('a','b'),type_,
            attrs=(('numeric_semantics','ieee754-bf16-mul-rne-v1'),),index_maps=pointwise_identity_maps(2,2))
        source=KernelRequest((*inputs,result),('result',),('external',),(('a','external'),('b','external')),TARGET_IDENTITY,'diagnostic')
        precision=prepare_source(source,bf16_policy=BF16_POINTWISE_POLICY)
        abi=plan_abi(precision.request);destination=self.root/'pointwise'
        manifest=compile_pointwise(self.snapshot,precision.request,fixed_inputs=abi['fixed_inputs'],
            fixed_outputs=tuple(abi['fixed_outputs']),target_source=ROOT,destination=destination,limits=SearchLimits())
        self.check_capture(destination,manifest)
        record,files=prepare_c_package(source,destination,'probe',target_source=ROOT,bf16_policy=BF16_POINTWISE_POLICY)
        self.assertTrue(files)
        spilled=destination/'mandatory-spilled-virtual.mlir';original=spilled.read_bytes()
        spilled.write_bytes(original+b'\n')
        with self.assertRaises((ValueError,NativeCompilationError)):
            prepare_c_package(source,destination,'probe',target_source=ROOT,bf16_policy=BF16_POINTWISE_POLICY)

    @patch.dict(os.environ, {'ATLAS_CAPTURE_POSTSPILL':'1'})
    def test_row_verification_and_export_do_not_compile_again(self):
        from atlas_mlir_native import binding, row_loop
        from atlas_mlir_native.softmax_c import prepare_c_package
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from test_tiled_softmax import source_request

        source=source_request((2,33));destination=self.root/'row-loop'
        manifest=row_loop.compile_row_loop(self.snapshot,source,target_source=ROOT,destination=destination,limits=SearchLimits())
        self.check_capture(destination,manifest)
        original=binding._run
        with patch.object(row_loop,'_run',wraps=original) as row_run, patch.object(binding,'_run',wraps=original) as native_run:
            row_loop.verify_row_loop(source,destination,target_source=ROOT)
            record,files=prepare_c_package(destination/'source-request.json',destination,'probe',target_source=ROOT)
        self.assertTrue(files)
        commands=[call.args[0] for call in (*row_run.call_args_list,*native_run.call_args_list)]
        self.assertFalse(any(any(option in {'--atlas-compile','--atlas-compile-unscheduled'} for option in command) for command in commands))
        self.assertTrue(any('--verify-atlas-timing' in command for command in commands))
        self.assertTrue(any('--program-json' in command for command in commands))
        receipt=destination/'compilation-pipeline.json';changed=json.loads(receipt.read_text());changed['wall_limit_s']=31
        receipt.write_text(json.dumps(changed))
        with self.assertRaises((ValueError,NativeCompilationError)):row_loop.verify_row_loop(source,destination,target_source=ROOT)

    @patch.dict(os.environ, {'ATLAS_CAPTURE_POSTSPILL':'0'})
    def test_default_row_publication_and_export_without_ir_capture(self):
        from atlas_mlir_native import row_loop
        from atlas_mlir_native.softmax_c import prepare_c_package
        from merlin.semantic_compiler.search import SearchLimits
        from test_tiled_softmax import source_request

        source=source_request((2,33));destination=self.root/'row-loop-default'
        manifest=row_loop.compile_row_loop(self.snapshot,source,target_source=ROOT,destination=destination,limits=SearchLimits())
        receipt=json.loads((destination/'compilation-pipeline.json').read_text())
        self.assertNotIn('spilled_virtual_sha256',receipt)
        self.assertFalse(any(option.startswith('--mlir-print-ir-') for option in receipt['argv']))
        self.assertFalse((destination/'mandatory-spilled-virtual.mlir').exists())
        self.assertIn('compilation-pipeline.json',manifest['stage_artifacts'])
        row_loop.verify_row_loop(source,destination,target_source=ROOT)
        _,files=prepare_c_package(destination/'source-request.json',destination,'probe_default',target_source=ROOT)
        self.assertTrue(files)


if __name__=='__main__':unittest.main()
