"""Automatic normal application compilation shares and verifies resident C bodies."""
import ctypes
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options
from atlas_mlir_native import quantized_loop
from atlas_mlir_native.quantized_resident import pack_resident_input
from atlas_mlir_native.resident_loop_image import ResidentLoopImage
from merlin.compiler.build.regions import compile_semantic_application, verify_semantic_application
from merlin.compiler.partition.semantic_regions import extract_semantic_region
from merlin.compiler.semantic.search.exploration import SearchLimits
from atlas_test_fixtures.semantic_regions import plan_for
import test_resident_loop_image as fixture
from test_resident_loop_c_application import renamed, dense_expected


def application_fixture(base, selected, trips):
    requests = [renamed(base, f'trip_{trip}_')[0] for trip in range(trips)]
    source = replace(base,
        nodes=tuple(node for request in requests for node in request.nodes if node.effect == 'input')+
              tuple(request.nodes[-1] for request in requests),
        outputs=tuple(request.outputs[0] for request in requests),
        output_storages=('external',)*trips,
        input_storages=tuple(row for request in requests for row in request.input_storages))
    plan = plan_for(source, tuple((trip,) for trip in range(trips)))
    plan = replace(plan, selected=tuple(replace(row, placement='accelerator',
        implementation='synthetic_atlas_counted', metadata=(('kind', 'already_quantized'),)) for row in plan.selected))
    regions = [extract_semantic_region(source, (request.outputs[0],)) for request in requests]
    bindings = {}
    for trip, region in enumerate(regions):
        initial = deepcopy(selected.resident_bindings[base.digest()])
        bindings[region.digest()] = {f'trip_{trip}_{key}':dict(row, address=row['address']+trip*8192)
                                    for key, row in initial.items()}
    return source, plan, regions, replace(selected, resident_bindings=bindings)


class SyntheticAdapter(AtlasApplicationAdapter):
    """A fixed synthetic partition uses Atlas's actual native/provider runtime."""
    def __init__(self, plan):
        self.synthetic_plan = plan
        self.compiles = 0

    def plan(self, source):
        return self.synthetic_plan

    def materialization_identity(self, region, alternative):
        return dict(target_choice=alternative.to_dict(), source_identity_policy='preserve',
                    compilation_identity=dict(synthetic='resident-c-automatic-v1'))

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        self.compiles += 1
        quantized_loop.compile_source(snapshot, region, destination=segments/name,
                                      target_source=target_source, limits=limits)
        return self.export_region(region, alternative, name, segments, target_source)


class ResidentLoopCAutomaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory)/'automatic-c')):
                fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            fixture.ResidentLoopImageTest.setUpClass.__func__(cls)

    tearDownClass = classmethod(fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def test_automatic_loop_compiles_once_replays_table_and_executes_normal_driver(self):
        import torch
        model = Path(os.environ.get('ATLAS_NPU_MODEL_ROOT', os.environ['NPU_MODEL_ROOT']))
        sys.path[:0] = [str(model), str(fixture.ROOT), str(fixture.ROOT/'tools'),
                       str(fixture.ROOT.parent/'merlin/examples/atlas/whole-model/unchanged_fx')]
        import functional_runtime
        from functional_runtime import FunctionalModelRuntime, PhysicalFunctionalInterpreter
        torch.set_num_threads(1)
        testcase = self

        class SophiaLakeInterpreter(PhysicalFunctionalInterpreter):
            def __init__(self, config, **kwargs):
                super().__init__(replace(config, dram_base=fixture.SOPHIALAKE_BASE,
                    dram_size=fixture.SOPHIALAKE_BYTES), **kwargs)

            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                testcase.assertEqual(result.state.cfg.dram_size, fixture.SOPHIALAKE_BYTES)
                return result

        for trips in (2, 22):
            with self.subTest(trips=trips):
                source, plan, regions, selected = application_fixture(self.source, self.selected, trips)
                adapter = SyntheticAdapter(plan)
                artifact = self.root/f'application-{trips}'
                with codegen_options(selected), patch.object(quantized_loop, 'compile_source',
                        wraps=quantized_loop.compile_source) as compiled:
                    manifest = compile_semantic_application(source, adapter=adapter, snapshot=self.snapshot,
                        target_source=fixture.ROOT, destination=artifact, limits=SearchLimits(),
                        name='model', capacity=262144)
                self.assertEqual(compiled.call_count, 1)
                self.assertEqual(adapter.compiles, 1)
                self.assertTrue(manifest['dispatch_loop'])
                rows = json.loads((artifact/'compiled-regions.json').read_bytes())
                self.assertEqual([row.get('compiled_body_index') for row in rows], [None]+[0]*(trips-1))
                self.assertEqual(len(tuple((artifact/'c-package/regions').iterdir())), 1)
                self.assertEqual(len(manifest['segments']), 1)
                combined = (artifact/'c-package/combined/model_combined.c').read_text()
                self.assertEqual(combined.count('static const uint32_t region_000_resident_image_template['), 1)
                receipt = rows[0]['description']['resident_loop_runtime']
                self.assertEqual(len(receipt['trip_bindings']), trips)
                with codegen_options(selected):
                    checked = verify_semantic_application(source, adapter=SyntheticAdapter(plan), snapshot=self.snapshot,
                        target_source=fixture.ROOT, artifact=artifact)
                self.assertEqual(checked['status'], 'checked')
                # Change a live address in both emitted C forms, then recompute the
                # claimed file hashes. Fresh source replay must still reject it.
                original_manifest = (artifact/'manifest.json').read_bytes()
                address = receipt['trip_bindings'][-1]['addresses'][0]
                altered_files = {}
                for relative in ('c-package/regions/region_000/region_000.c', 'c-package/combined/model_combined.c'):
                    path = artifact/relative
                    original = path.read_bytes()
                    needle = f'{address}u'.encode()
                    self.assertIn(needle, original)
                    altered = original.replace(needle, f'{address+8192}u'.encode(), 1)
                    altered_files[relative] = original
                    path.write_bytes(altered)
                    manifest['stage_artifacts'][relative] = sha256(altered).hexdigest()
                (artifact/'manifest.json').write_text(json.dumps(manifest))
                with codegen_options(selected), self.assertRaisesRegex(ValueError, 'differs from replayed'):
                    verify_semantic_application(source, adapter=SyntheticAdapter(plan), snapshot=self.snapshot,
                        target_source=fixture.ROOT, artifact=artifact)
                for relative, original in altered_files.items():
                    (artifact/relative).write_bytes(original)
                (artifact/'manifest.json').write_bytes(original_manifest)
                package = artifact/'c-package'
                driver = self.root/f'native-driver-{trips}'
                driver.mkdir()
                (driver/'harness.c').write_text('''#include "combined/model_combined.h"
static model_context context;static atlas_host host;
int setup(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get) {
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 return model_bind(&context,&host,(atlas_tile_memory_io){put,get,0},8u);
}
int run(const void *const *inputs,void *const *outputs) { return model_run(&context,inputs,outputs); }
void interleave(void) { static const uint32_t image[]={0x13u,0x73u};(void)atlas_host_ensure_loaded(&host,image,2u); }
uint64_t loads(void) { return host.image_loads; }
uint64_t reuses(void) { return host.image_reuses; }
''')
                compiler = shutil.which('cc')
                self.assertIsNotNone(compiler, 'C compiler required')
                subprocess.run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC',
                    'combined/model_combined.c', str(driver/'harness.c'), '-I.', '-o', str(driver/'model.so')],
                    cwd=package, check=True, capture_output=True)
                with codegen_options(selected):
                    image = ResidentLoopImage(regions[0], artifact/'segments/region_000', target_source=fixture.ROOT)
                    bound = [image.bind(region) for region in regions]
                values = [fixture.fixture_values(trip) for trip in range(trips)]
                # Encode the trip in five logical FP8 weights so all 22
                # immutable payloads differ, with exact small-integer sums.
                for trip, data in enumerate(values):
                    for bit in range(5):
                        data['w'][0, bit] = 0x40 if trip & (1 << bit) else 0x38
                self.assertEqual(len({data['w'].tobytes() for data in values}), trips)
                immutable = [(row['address'], pack_resident_input(region, row['node_id'],
                    values[trip]['sw' if row['node_id'].endswith('sw') else 'w'].tobytes()))
                    for trip, (region, current) in enumerate(zip(regions, bound, strict=True))
                    for row in current.execution_plan['resident_bindings']]
                runtime = FunctionalModelRuntime(fixture.ROOT, model, [current.binary for current in bound],
                    self.root/f'model-{trips}', memory_bytes=selected.io_memory_bytes, memory_base=DRAM_BASE,
                    image_plans={sha256(current.binary).hexdigest():current.execution_plan for current in bound},
                    trace_payloads=False, immutable_loads=immutable,
                    capacity_base=selected.io_memory_capacity_base, capacity_bytes=selected.io_memory_capacity_bytes)
                library = ctypes.CDLL(str(driver/'model.so'))
                library.setup.argtypes = [type(callback) for callback in runtime.callbacks]
                library.run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
                library.loads.restype = library.reuses.restype = ctypes.c_uint64
                self.assertEqual(library.setup(*runtime.callbacks), 0)

                def execute(data):
                    buffers = [ctypes.create_string_buffer(item[key].tobytes()) for item in data for key in ('a', 'sx', 'w', 'sw')]
                    # Keep every source boundary valid, while deliberately giving
                    # the provider incorrect dense resident host operands. Device
                    # allocations supply the authenticated immutable values.
                    for index in range(trips):
                        ctypes.memset(buffers[index*4+2], 0, len(buffers[index*4+2]))
                        ctypes.memset(buffers[index*4+3], 0, len(buffers[index*4+3]))
                    inputs = (ctypes.c_void_p*len(buffers))(*(ctypes.addressof(buffer) for buffer in buffers))
                    outputs = [ctypes.create_string_buffer(30) for _ in data]
                    pointers = (ctypes.c_void_p*trips)(*(ctypes.addressof(buffer) for buffer in outputs))
                    before = len(runtime.launches)
                    with patch.object(functional_runtime, 'PhysicalFunctionalInterpreter', SophiaLakeInterpreter):
                        self.assertEqual(library.run(inputs, pointers), 0, runtime.errors)
                    self.assertEqual(len(runtime.launches)-before, trips)
                    self.assertEqual([buffer.raw for buffer in outputs], [dense_expected(item) for item in data])
                    return [buffer.raw for buffer in outputs]

                first = execute(values)
                for address, payload in immutable:
                    self.assertEqual(bytes(runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)]), payload)
                changed = deepcopy(values)
                changed[-1]['w'][-1, -1] = 0x40 if changed[-1]['w'][-1, -1] == 0x38 else 0x38
                address = bound[-1].execution_plan['input_addresses'][1]
                payload = pack_resident_input(regions[-1], regions[-1].nodes[2].id, changed[-1]['w'].tobytes())
                runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)] = payload
                library.interleave()
                second = execute(changed)
                self.assertEqual(first[:-1], second[:-1])
                self.assertNotEqual(first[-1], second[-1])
                result = runtime.finish()
                self.assertFalse(result['errors'])
                self.assertEqual(len(result['launches']), trips*2)
                self.assertEqual(library.loads(), trips*2+1)
                self.assertEqual(library.reuses(), 0)
                record = dict(schema='atlas.resident_loop_c_automatic_test.v1', trips=trips,
                    exact=True, executed_later_weight_mutation=True, native_compilations=compiled.call_count,
                    shared_template_count=1,
                    distinct_immutable_weight_payloads=trips,
                    compiled_body_indices=[row.get('compiled_body_index') for row in rows],
                    fresh_source_verification=True, rehashed_address_table_tamper_rejected=True,
                    per_decode_starts=[trips, trips], prefix_starts=0, suffix_starts=0,
                    image_loads=library.loads(), image_reuses=library.reuses(),
                    functional_instructions=[row['instructions'] for row in result['launches']])
                (self.root/f'automatic-application-{trips}.json').write_text(json.dumps(record, indent=2)+'\n')
                print('AUTOMATIC_RESIDENT_APPLICATION_EXACT', trips, record, flush=True)


if __name__ == '__main__':
    unittest.main()
