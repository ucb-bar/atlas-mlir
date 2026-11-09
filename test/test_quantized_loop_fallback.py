"""Oversized complete sources retain checked K32 dispatch and private outputs."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native import quantized_loop, quantized_loop_c
from merlin.semantic_compiler.model import ConstantBinding
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_already_quantized import source_request
from test_quantized_epilogue import case

ROOT = Path(__file__).resolve().parents[1]


def constant_source():
    source = source_request(512, 705, 32, batch=())
    values = {'w': bytes([0x38]) * (705 * 32), 'sw': (-7).to_bytes(4, 'little', signed=True) * 705}
    return replace(source,
        nodes=tuple(replace(node, op='constant', effect='constant') if node.id in values else node
                    for node in source.nodes),
        constants=tuple(ConstantBinding(key, 'i8' if key == 'w' else 'i32-le', data.hex())
                        for key, data in values.items()),
        input_storages=tuple(row for row in source.input_storages if row[0] not in values))


class QuantizedLoopFallbackTest(unittest.TestCase):
    def setUp(self):
        # This small diagnostic profile is a last-resort ABI control, not the
        # cache phase's selected 64 MiB window.
        context = codegen_options(CodegenOptions(io_memory_bytes=12288))
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_window512_uses_counted_rows_before_last_resort_tiles(self):
        for m, n, k, batch, fallback in (
                (64, 5632, 2048, (), False), (64, 2048, 5632, (), False),
                (64, 64, 64, (32,), False), (1, 65, 64, (32,), False),
                (512, 5632, 2048, (), False), (512, 512, 64, (32,), False)):
            with codegen_options(CodegenOptions(io_memory_bytes=67108864)), self.subTest(m=m, n=n, k=k, batch=batch):
                self.assertEqual(quantized_loop.uses_tiled_fallback(prepare_source(source_request(m, n, k, batch))), fallback)
        source = constant_source()
        with tempfile.TemporaryDirectory(prefix='atlas-row-refusal-') as directory, \
             patch('atlas_mlir_native.already_quantized.compile_source', return_value='tiles') as compile_tiles, \
             patch('atlas_mlir_native.quantized_loop_sizing.publish_route'), \
             patch.object(quantized_loop, 'virtual_source', side_effect=AssertionError('whole image attempted')):
            self.assertEqual(quantized_loop.compile_source('snapshot', source, target_source=ROOT,
                             destination=directory, limits='limits'), 'tiles')
            refusal = json.loads((Path(directory)/'row-block-refusals.json').read_text())
            self.assertEqual(refusal['source_request_digest'], source.digest())
        self.assertIs(compile_tiles.call_args.args[1], source)
        self.assertEqual(compile_tiles.call_args.kwargs['limits'], 'limits')

    def test_synthetic_small_profile_epilogue_retains_separate_device_choices(self):
        from atlas_mlir_native.quantized_epilogue import prepare_source as prepare_epilogue, alternatives
        source, _, choices = case(m=512, n=705, k=32, batch=())
        self.assertTrue(quantized_loop.uses_tiled_fallback(prepare_epilogue(source)))
        adapter = type('Adapter', (), {})()
        self.assertEqual(alternatives(adapter, source, choices), choices)
        self.assertEqual(adapter._quantized_epilogue_groups, {})

    @unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select native compiler tools')
    def test_512_k32_source_exports_only_used_tile_and_literal_constant_binding(self):
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        source = constant_source()
        original = source.record()
        payloads = {item.node_id: bytes.fromhex(item.data_hex) for item in source.constants}
        with tempfile.TemporaryDirectory(prefix='atlas-bounded-qlinear-') as directory:
            root = Path(directory)
            snapshot = build_snapshot(root / 'native')
            compiled = root / 'compiled'
            quantized_loop.compile_source(snapshot, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
            self.assertFalse((compiled / 'atlas-virtual.mlir').exists())
            self.assertEqual(json.loads((compiled / 'already-quantized-plan.json').read_text()), prepare_source(source).record())
            self.assertFalse((compiled / 'continue').exists())
            for role in ('reset',):
                manifest = json.loads((compiled / role / 'manifest.json').read_text())
                self.assertEqual(manifest['binary_sha256'], sha256((compiled / role / 'program.bin').read_bytes()).hexdigest())
            record, files = quantized_loop_c.prepare_c_package(source, compiled, 'bounded', target_source=ROOT)
            self.assertEqual(record['execution_route'], 'bounded_k32_tile_dispatch')
            self.assertEqual(record['launches_per_invocation'], 16 * 23)
            self.assertEqual(record['native_program_paths'], ['reset'])
            self.assertEqual(record['device_launch_schedule']['total_launches'], 16 * 23)
            self.assertIn('host', record['restoration_placement'])
            self.assertEqual(record['source_interface']['failure_output'], 'unchanged')
            self.assertEqual(record['precision_plan']['source_request_digest'], source.digest())
            self.assertTrue(files)
        self.assertEqual(source.record(), original)
        self.assertEqual({item.node_id: bytes.fromhex(item.data_hex) for item in source.constants}, payloads)


if __name__ == '__main__':
    unittest.main()
