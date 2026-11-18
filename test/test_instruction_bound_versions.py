"""Both authenticated instruction proof versions replay at each existing exporter."""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.search import SearchLimits

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('ATLAS_OOT_BIN_DIR', ROOT / 'build/bin'))


@unittest.skipUnless((BIN / 'atlas-opt').is_file(), 'select built Atlas tools')
class InstructionBoundVersionsTest(unittest.TestCase):
    def check_versions(self, source, compile_source, export, artifact, snapshot=None):
        with tempfile.TemporaryDirectory() as directory:
            compiled = Path(directory) / 'publication'
            compile_source(snapshot, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
            path = compiled / artifact
            fresh = json.loads(path.read_text())
            self.assertEqual(fresh['schema'], 'atlas.restoration_instruction_bound.v2')
            export(source, compiled, 'probe', target_source=ROOT)
            # Retain every original v1 field, including restoration admission.
            legacy = instruction_bound((compiled / 'atlas-virtual.mlir').read_text(),
                (compiled / 'atlas-machine.mlir').read_text(),
                json.loads((compiled / 'physical-program.json').read_text()),
                schema='atlas.restoration_instruction_bound.v1')
            for key in ('scalar_fallback_tile_budget', 'restoration_admission'):
                if key in fresh:
                    legacy[key] = fresh[key]
            def publish(record):
                path.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
                manifest_path = compiled / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                sizing_path = compiled / 'quantized-sizing.json'
                if sizing_path.exists():
                    sizing = json.loads(sizing_path.read_text())
                    sizing['physical_instruction_bound'] = record['instruction_bound']
                    sizing_path.write_text(json.dumps(sizing))
                    route_path = compiled / 'quantized-route.json'
                    route = json.loads(route_path.read_text())
                    route['sizing_sha256'] = sha256(sizing_path.read_bytes()).hexdigest()
                    route_path.write_text(json.dumps(route))
                    for member in (sizing_path, route_path):
                        manifest['stage_artifacts'][member.name] = sha256(member.read_bytes()).hexdigest()
                manifest['stage_artifacts'][artifact] = sha256(path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
            publish(legacy)
            export(source, compiled, 'probe', target_source=ROOT)
            for record in (legacy, fresh):
                with self.subTest(schema=record['schema']):
                    publish(dict(record, instruction_bound=record['instruction_bound'] - 1))
                    with self.assertRaisesRegex(ValueError, 'instruction (bound|proof) differs'):
                        export(source, compiled, 'probe', target_source=ROOT)
            publish(dict(fresh, schema='atlas.restoration_instruction_bound.v999'))
            with self.assertRaisesRegex(ValueError, 'unsupported schema'):
                export(source, compiled, 'probe', target_source=ROOT)

    def test_float_v1_and_v2(self):
        from atlas_mlir_native.exact_float_device import compile_source, prepare_c_package
        from test_exact_float_device import request
        self.check_versions(request('bf16_to_f32', 64), compile_source, prepare_c_package,
                            'float-instruction-bound.json')

    def test_layout_v1_and_v2(self):
        from atlas_mlir_native.layout_device import compile_source, prepare_c_package
        from test_layout_device import request
        self.check_versions(request(shapes=((3, 7),), shape=(21,)), compile_source,
                            prepare_c_package, 'layout-instruction-bound.json')

    def test_restoration_v1_and_v2(self):
        from atlas_mlir_native.quantized_loop import compile_source
        from atlas_mlir_native.quantized_loop_c import prepare_c_package
        from merlin.semantic_compiler.snapshot import open_native_snapshot
        from test_quantized_loop import shared
        selected = os.environ.get('ATLAS_TEST_NATIVE_SNAPSHOT')
        if selected:
            snapshot = open_native_snapshot(Path(selected))
        else:
            from test_looped_mnk import build_snapshot
            with tempfile.TemporaryDirectory() as directory:
                snapshot = build_snapshot(Path(directory) / 'native')
                self.check_versions(shared, compile_source, prepare_c_package,
                                    'restoration-instruction-bound.json', snapshot)
            return
        self.check_versions(shared, compile_source, prepare_c_package,
                            'restoration-instruction-bound.json', snapshot)
