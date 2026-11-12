"""Bounded contraction exports publish real leaves and exact call order."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from atlas_mlir_native.already_quantized import POLICY, prepare_source
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_precision import CONTRACTION_POLICY
from atlas_mlir_native.binding import TARGET_IDENTITY, AtlasNativeBinding
from atlas_mlir_native.quantized_loop import compile_source, uses_tiled_fallback
from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
from merlin.compile.application_handoff import application_handoff
from merlin.compile.launch_schedule import expand_launch_schedule
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import ConstantBinding
from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
from merlin.semantic_compiler.target_binding import verify_native_publication
from test_already_quantized import source_request
from test_looped_mnk import build_snapshot
from test_quantized_loop_fallback import constant_source
from test_quantized_loop_dram import CACHE_DRAM, cache_source


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                     'select native compiler tools')
class BoundedContractionHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='atlas-contraction-handoff-')
        cls.root = Path(cls.temporary.name)
        with codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM)):
            cls.snapshot = build_snapshot(cls.root / 'native')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        context = codegen_options(CodegenOptions(io_memory_bytes=CACHE_DRAM))
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def check_handoff(self, source, name, roles=None, groups=None, launches=None, *, counted=False, rows=False):
        original = source.record()
        plan = prepare_source(source)
        self.assertEqual(uses_tiled_fallback(plan),not (counted or rows))
        policy = dict(schema='atlas.diagnostic_application_precision.v1', source_sha256='0'*64,
            target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY,
            row_softmax=SOFTMAX_POLICY, host_support=['transpose', 'reshape', 'copy'],
            already_quantized=POLICY)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256='0'*64)
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(dict(selected[0].metadata)['kind'], 'already_quantized')
        application = self.root / name
        segments = application / 'segments'
        compiled = segments / name
        compile_source(self.snapshot, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
        self.assertEqual((compiled / 'manifest.json').exists(),counted)
        row_schedule = None
        if rows:
            from atlas_mlir_native.quantized_row_blocks import verify_publication
            row_schedule, envelope = verify_publication(source, compiled, target_source=ROOT)
            roles = tuple(row['path'] for row in envelope['segments'])
            launches = len(row_schedule.launches)
            self.assertEqual({path.name for path in compiled.iterdir() if path.is_dir()}, {'segments'})
            self.assertLess(launches, 65536)
            self.assertLessEqual(launches, plan.m)
        else:
            self.assertEqual({path.name for path in compiled.iterdir() if path.is_dir()},set() if counted else set(roles))
        lowered = adapter.export_region(source, selected[0], name, segments, ROOT)
        package_root = self.root / (name + '-c-package')
        package_root.mkdir()
        for member, payload in lowered.files.items():
            path = package_root / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-Wno-misleading-indentation', '-fsyntax-only', name+'.c'],
                       cwd=package_root, check=True, capture_output=True, timeout=30)
        expected_paths = ['segments/'+name+('' if role=='.' else '/'+role) for role in roles]
        self.assertEqual(list(lowered.programs), expected_paths)
        if not counted:
            self.assertEqual(lowered.description['native_program_paths'],list(roles))
        self.assertEqual(lowered.description['precision_plan']['source_request_digest'], source.digest())
        self.assertEqual(lowered.description['launches_per_invocation'], launches)
        images, executions, payloads = [], {}, {}
        for index, (role, relative) in enumerate(zip(roles, expected_paths, strict=True)):
            leaf = application / relative
            manifest = json.loads((leaf / 'manifest.json').read_text())
            body = row_schedule.bodies[index] if rows else source if counted else plan.reset if role=='reset' else plan.continuation
            verify_native_publication(leaf, manifest, engine='merlin_native',
                request_digest=body.digest(), target_identity=source.target_identity)
            binary = (leaf / 'program.bin').read_bytes()
            digest = sha256(binary).hexdigest()
            self.assertEqual(digest, manifest['binary_sha256'])
            self.assertEqual(json.loads((leaf / 'source-request.json').read_text()), body.record())
            if rows:
                proof = json.loads((leaf / 'restoration-instruction-bound.json').read_text())
                self.assertTrue(proof['admitted'])
                self.assertIs(type(proof['instruction_bound']), int)
                self.assertGreater(proof['instruction_bound'], 0)
                self.assertLessEqual(proof['instruction_bound'], 1000000)
                launch = next(row for row in row_schedule.launches if row['body_index'] == index)
                spans = {row['node_id']: row for row in launch['input_slices']}
                parent_constants = {row.node_id: bytes.fromhex(row.data_hex) for row in source.constants}
                for binding in body.constants:
                    span = spans[binding.node_id]
                    self.assertEqual(bytes.fromhex(binding.data_hex), parent_constants[binding.node_id][
                        span['offset_bytes']:span['offset_bytes']+span['bytes']])
            images.append(digest)
            payloads[digest] = binary
            executions[digest] = json.loads((leaf / 'execution_plan.json').read_text())
        interface = dict(source=name+'.c', header=lowered.header, entry=lowered.symbol,
                         inputs=list(lowered.inputs))
        files = {'c-package/'+path: data for path, data in lowered.files.items()}
        region = dict(id='contraction', placement='accelerator', native_programs=expected_paths,
                      description=lowered.description)

        def emit(row):
            return application_handoff(source, interface, [row], files, application)

        package = emit(region)
        abi = json.loads(package['handoff/abi.json'])
        call = abi['calls'][0]
        self.assertEqual(call['images'], images)
        if counted:
            self.assertNotIn('device_launch_schedule',call)
            self.assertEqual(launches,1)
            self.assertEqual(lowered.description['restoration_placement'],'device')
        else:
            schedule = call['device_launch_schedule']
            if groups is not None:
                self.assertEqual(schedule['groups'],groups)
            self.assertEqual(schedule['total_launches'],launches)
            self.assertEqual(schedule['images'],images)
            expected = (tuple(images[row['body_index']] for row in row_schedule.launches) if rows else
                        tuple(images[run['image']] for group in groups for _ in range(group['repeat'])
                              for run in group['runs'] for _ in range(run['count'])))
            self.assertEqual(expand_launch_schedule(schedule,image_sha256s=images),expected)
            if rows:
                self.assertEqual(schedule, lowered.description['device_launch_schedule'])
                self.assertEqual(lowered.description['execution_route'], 'checked_counted_row_blocks')
                self.assertEqual(lowered.description['restoration_placement'], 'device')
                self.assertEqual(lowered.description['row_block_schedule'], row_schedule.record())
        self.assertNotIn('launches_per_call', call)
        self.assertEqual(len(abi['programs']), len(set(images)))
        for record in abi['programs']:
            digest = record['sha256']
            self.assertEqual(package['handoff/'+record['path']], payloads[digest])
            self.assertEqual(record['execution'], executions[digest])
        self.assertEqual({row['id']: package['handoff/'+row['path']] for row in abi['constants']},
                         {row.node_id: bytes.fromhex(row.data_hex) for row in source.constants})
        self.assertEqual((compiled / 'manifest.json').exists(),counted)
        self.assertEqual(source.record(), original)
        if not counted:
            bad = deepcopy(region)
            bad['description']['device_launch_schedule']['total_launches'] += 1
            with self.assertRaisesRegex(ValueError,'declared total'):
                emit(bad)
        binary_path = application / expected_paths[0] / 'program.bin'
        binary = binary_path.read_bytes()
        try:
            binary_path.write_bytes(bytes([binary[0] ^ 1]) + binary[1:])
            with self.assertRaisesRegex(ValueError, 'changed during handoff'):
                emit(region)
        finally:
            binary_path.write_bytes(binary)
        return region, emit

    def test_window512_exports_counted_rows_and_exact_genuine_child_schedule(self):
        source = source_request(512, 2048, 2048, batch=())
        region, emit = self.check_handoff(source, 'large', rows=True)
        bad = deepcopy(region)
        if len(set(region['description']['device_launch_schedule']['images'])) > 1:
            bad['native_programs'].reverse()
        else:
            # A single shared child has no distinct order to reverse; an extra
            # path must still refuse rather than alter the exact call ledger.
            bad['native_programs'] *= 2
        with self.assertRaisesRegex(ValueError, 'ordered native programs'):
            emit(bad)
        # Rehashing the enclosing envelope cannot authenticate a changed
        # parent byte view: export rebuilds the schedule from the real source.
        from atlas_mlir_native.quantized_loop_c import prepare_c_package
        compiled = self.root/'large/segments/large'
        schedule_path = compiled/'row-block-schedule.json'
        manifest_path = compiled/'row-block-manifest.json'
        saved_schedule, saved_manifest = schedule_path.read_bytes(), manifest_path.read_bytes()
        try:
            changed = json.loads(saved_schedule)
            changed['launches'][0]['input_slices'][0]['offset_bytes'] += 1
            schedule_path.write_text(json.dumps(changed))
            envelope = json.loads(saved_manifest)
            envelope['stage_artifacts']['row-block-schedule.json'] = sha256(schedule_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'source, schedule, or target profile differs'):
                prepare_c_package(source, compiled, 'rejected', target_source=ROOT)
        finally:
            schedule_path.write_bytes(saved_schedule)
            manifest_path.write_bytes(saved_manifest)

    def test_k32_exports_only_used_reset_image_and_literal_constants(self):
        # A deliberately small diagnostic window preserves reset-only fallback
        # ABI coverage without introducing a production dimension threshold.
        with codegen_options(CodegenOptions(io_memory_bytes=12288)):
            self.snapshot = build_native_snapshot(AtlasNativeBinding.profile(),
                destination=self.root/'small-native/snapshot',crate=packaged_egg_bridge_path(),
                cargo_target_dir=self.root/'native/cargo',source_revision='small-window-handoff-test')
            self.check_handoff(constant_source(),'single',('reset',),
                [dict(repeat=368,runs=[dict(image=0,count=1)])],368)

    def test_cache64_keeps_checked_counted_image_and_unchanged_weights(self):
        source = cache_source()
        values = {'w': bytes([0x38])*(2048*2048),
                  'sw': (-7).to_bytes(4,'little',signed=True)*2048}
        source = replace(source,
            nodes=tuple(replace(node,op='constant',effect='constant') if node.id in values else node
                        for node in source.nodes),
            constants=tuple(ConstantBinding(key,'i8' if key=='w' else 'i32-le',data.hex())
                            for key,data in values.items()),
            input_storages=tuple(row for row in source.input_storages if row[0] not in values))
        # The actual cache profile equals the fixed64 profile: one checked
        # counted launch retains device restoration and literal source weights.
        self.check_handoff(source,'cache_q',('.',),None,1,counted=True)
        compiled = self.root/'cache_q/segments/cache_q'
        manifest = json.loads((compiled/'manifest.json').read_text())
        self.assertEqual(manifest['io_memory_bytes'],CACHE_DRAM)
        route_path = compiled/'quantized-route.json'
        original = route_path.read_bytes()
        from atlas_mlir_native.quantized_loop_c import prepare_c_package
        for field,value in (('selected_io_memory_bytes',4325376),('launches_per_invocation',4096),
                ('source_request_digest','0'*64),('physical_instruction_limit',1000001),
                ('restoration_placement','host')):
            changed = json.loads(original);changed[field] = value
            try:
                route_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    prepare_c_package(source,compiled,'rejected',target_source=ROOT)
            finally:
                route_path.write_bytes(original)
        # Updating every enclosing hash cannot turn an invented bound into a
        # proof: the exporter rederives it from the verified physical words.
        names = ('manifest.json','quantized-route.json','quantized-sizing.json',
                 'restoration-instruction-bound.json')
        saved = {name:(compiled/name).read_bytes() for name in names}
        try:
            bound = json.loads(saved['restoration-instruction-bound.json'])
            sizing = json.loads(saved['quantized-sizing.json'])
            route = json.loads(saved['quantized-route.json'])
            parent = json.loads(saved['manifest.json'])
            bound['instruction_bound'] -= 1
            sizing['physical_instruction_bound'] = bound['instruction_bound']
            for name,value in (('restoration-instruction-bound.json',bound),('quantized-sizing.json',sizing)):
                (compiled/name).write_text(json.dumps(value))
            route['sizing_sha256'] = sha256((compiled/'quantized-sizing.json').read_bytes()).hexdigest()
            (compiled/'quantized-route.json').write_text(json.dumps(route))
            for name in names[1:]:
                parent['stage_artifacts'][name] = sha256((compiled/name).read_bytes()).hexdigest()
            (compiled/'manifest.json').write_text(json.dumps(parent))
            with self.assertRaisesRegex(ValueError,'physical instruction proof differs'):
                prepare_c_package(source,compiled,'rejected',target_source=ROOT)
        finally:
            for name,payload in saved.items():
                (compiled/name).write_bytes(payload)



if __name__ == '__main__':
    unittest.main()
