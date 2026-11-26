"""Opt-in end-to-end comparison with an independently emitted legacy application.

Set ATLAS_PORT_SNAPSHOT, ATLAS_PORT_LEGACY_APPLICATION, and ATLAS_OOT_BIN_DIR.
The fixture and legacy codegen options are consumed unchanged. No boards or
reference evaluator are used; this checks compiler output bytes only.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.compiler.build.regions import compile_semantic_application
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.search.exploration import SearchLimits
from merlin.compiler.semantic.target_binding import open_native_snapshot


@unittest.skipUnless(os.environ.get('ATLAS_PORT_SNAPSHOT') and os.environ.get('ATLAS_PORT_LEGACY_APPLICATION'),
                     'requires explicit native snapshot and independently compiled legacy application')
class RefactorApplicationTest(unittest.TestCase):
    def test_k32_emits_identical_device_programs(self):
        fixture = Path(__file__).parent / 'fixtures/refactor-k32'
        source = KernelRequest.from_record(json.loads((fixture / 'source-request.json').read_bytes()))
        original = source.record()
        policy = json.loads((fixture / 'precision-policy.json').read_bytes())
        baseline = Path(os.environ['ATLAS_PORT_LEGACY_APPLICATION'])
        legacy = json.loads((baseline / 'manifest.json').read_bytes())
        snapshot = open_native_snapshot(Path(os.environ['ATLAS_PORT_SNAPSHOT']))
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy['source_sha256'])
        with tempfile.TemporaryDirectory() as directory, codegen_options(CodegenOptions(**legacy['codegen_options'])):
            destination = Path(directory) / 'application'
            actual = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
                target_source=Path(__file__).resolve().parents[1], destination=destination,
                limits=SearchLimits(), name='capsule', capacity=legacy['application_memory_bytes'])
            self.assertEqual(source.record(), original)
            self.assertEqual(len(actual['segments']), len(legacy['segments']))
            self.assertGreater(len(actual['segments']), 0)
            for row, old in zip(actual['segments'], legacy['segments'], strict=True):
                self.assertEqual(row['path'], old['path'])
                self.assertEqual(row['binary_sha256'], old['binary_sha256'])
                current, previous = destination / row['path'], baseline / old['path']
                for name in ('atlas-virtual.mlir', 'atlas-machine.mlir', 'physical-program.json', 'program.bin'):
                    self.assertEqual((current / name).read_bytes(), (previous / name).read_bytes(), str(current / name))
                self.assertEqual(hashlib.sha256((current / 'program.bin').read_bytes()).hexdigest(), row['binary_sha256'])
            # The adapter's interface claims must survive the real publication/replay.
            for region in actual['regions']:
                self.assertIn('compiler_recipe', region['description'])


if __name__ == '__main__':
    unittest.main()
