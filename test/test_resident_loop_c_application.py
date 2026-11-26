"""Provider-shared generated C uses each trip's resident allocations in the model."""
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
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_c_context import application_binding
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options
from atlas_mlir_native.quantized_resident import pack_resident_input
import test_resident_loop_image as fixture


def renamed(source, prefix):
    names = {node.id: prefix+node.id for node in source.nodes}
    return replace(source, nodes=tuple(replace(node, id=names[node.id],
        inputs=tuple(names[key] for key in node.inputs)) for node in source.nodes),
        outputs=tuple(names[key] for key in source.outputs),
        input_storages=tuple((names[key], value) for key, value in source.input_storages)), names


def dense_expected(values):
    tile = fixture.expected_tile(values)
    return b''.join(tile[(r*16+c)*2:(r*16+c)*2+2] for r in range(3) for c in range(5))


class ResidentLoopCApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory)/'application-c')):
                fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            fixture.ResidentLoopImageTest.setUpClass.__func__(cls)

    tearDownClass = classmethod(fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def test_shared_provider_body_and_application_context_execute_current_trip(self):
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

        alternative = SimpleNamespace(placement='accelerator', metadata=(('kind', 'already_quantized'),))
        for trips in (2, 22):
            with self.subTest(trips=trips):
                adapter = object.__new__(AtlasApplicationAdapter)
                with codegen_options(self.selected):
                    representative = adapter.export_region(self.source, alternative, 'compiled', self.root, fixture.ROOT)
                self.assertIn('resident_loop_runtime', representative.description)
                packages, sources = [representative], [self.source]
                all_bindings = deepcopy(self.selected.resident_bindings)
                for trip in range(1, trips):
                    source, names = renamed(self.source, f'trip_{trip}_')
                    selected = fixture.moved_options(self.source, self.selected, trip)
                    current = {names[key]: value for key, value in selected.resident_bindings[self.source.digest()].items()}
                    all_bindings[source.digest()] = current
                    selected = replace(selected, resident_bindings={source.digest(): current})
                    with codegen_options(selected), patch.object(type(self.snapshot), 'select',
                            side_effect=AssertionError('sharing repeated selection')):
                        package = adapter.bind_loop_region(representative, source, alternative, names, self.root, fixture.ROOT)
                    self.assertIsNotNone(package)
                    packages.append(replace(package, body_index=0))
                    sources.append(source)
                # Pending state never changes the frozen representative package.
                original_files = dict(representative.files)
                # Match the normal compiler's complete allocation scope when
                # finalization independently replays every current source.
                with codegen_options(replace(self.selected, resident_bindings=all_bindings)):
                    packages = adapter.finalize_loop_regions(tuple(packages), target_source=fixture.ROOT)
                self.assertEqual(representative.files, original_files)
                self.assertTrue(all(package.files is packages[0].files for package in packages))
                receipt = packages[0].description['resident_loop_runtime']
                self.assertEqual(len(receipt['trip_bindings']), trips)
                body = packages[0].files['compiled.c'].decode()
                self.assertEqual(body.count('static const uint32_t compiled_resident_image_template['), 1)
                self.assertNotIn('static const uint32_t compiled_words[', body)
                self.assertEqual(len({row['binding']['binary_sha256'] for row in receipt['trip_bindings']}), trips)
                rows = [dict(placement='accelerator', description=package.description) for package in packages]
                binding = application_binding('model', rows,
                    dict(workspace_bytes=1, inputs=[{}], outputs=[{}], pointer_abi='opaque_native_dense_buffers'))
                work = self.root/f'generated-application-{trips}'
                work.mkdir()
                for name, data in packages[0].files.items():
                    path = work/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                (work/'harness.c').write_text('''#include "compiled.h"
typedef struct { unsigned active; } model_driver_state;
typedef struct { void **items; size_t count; } model_driver_contexts;
static int model_invoke(model_driver_state *s,void *w,size_t z,const void *const *i,void *const *o,const model_driver_contexts *c) {
 (void)s;(void)w;(void)z;(void)i;(void)o;(void)c;return 0;
}
'''+binding.declarations+binding.implementation+'''
static model_context context; static atlas_host host;
int setup(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get) {
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 return model_bind(&context,&host,(atlas_tile_memory_io){put,get,0},8u);
}
int call(uint32_t index,const void *const *inputs,void *const *outputs) {
 if(index>=context.dispatch.count)return 8;
 return compiled_source_dispatch(context.items[index],inputs,outputs);
}
void interleave(void) { static const uint32_t image[]={0x13u,0x73u};(void)atlas_host_ensure_loaded(&host,image,2u); }
uint64_t loads(void) { return host.image_loads; }
uint64_t reuses(void) { return host.image_reuses; }
''')
                compiler = shutil.which('cc')
                self.assertIsNotNone(compiler, 'C compiler is required')
                subprocess.run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC',
                    'compiled.c', 'runtime/atlas_host.c', 'harness.c', '-o', 'application.so'],
                    cwd=work, check=True, capture_output=True)
                values = [fixture.fixture_values(trip) for trip in range(trips)]
                plans = [row['execution_plan'] for row in receipt['trip_bindings']]
                # Reconstruct expected immutable images through the original checked
                # template under each current compiler allocation, independently of C.
                images = []
                for source, plan in zip(sources, plans, strict=True):
                    bindings = {row['node_id']: {key: row[key] for key in ('address', 'bytes', 'immutable')}
                                for row in plan['resident_bindings']}
                    with codegen_options(replace(self.selected, resident_bindings={source.digest(): bindings})):
                        images.append(self.image.bind(source).binary)
                immutable = [(row['address'], pack_resident_input(source, row['node_id'],
                    values[trip]['w' if row['node_id'].endswith('w') and not row['node_id'].endswith('sw') else 'sw'].tobytes()))
                    for trip, (source, plan) in enumerate(zip(sources, plans, strict=True))
                    for row in plan['resident_bindings']]
                runtime = FunctionalModelRuntime(fixture.ROOT, model, images, work/'model',
                    memory_bytes=self.selected.io_memory_bytes, memory_base=DRAM_BASE,
                    image_plans={sha256(image).hexdigest(): plan for image, plan in zip(images, plans, strict=True)},
                    trace_payloads=False, immutable_loads=immutable,
                    capacity_base=self.selected.io_memory_capacity_base,
                    capacity_bytes=self.selected.io_memory_capacity_bytes)
                library = ctypes.CDLL(str(work/'application.so'))
                library.setup.argtypes = [type(callback) for callback in runtime.callbacks]
                library.call.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
                library.loads.restype = library.reuses.restype = ctypes.c_uint64
                self.assertEqual(library.setup(*runtime.callbacks), 0)

                def execute(trip, data):
                    a = ctypes.create_string_buffer(data['a'].tobytes())
                    sx = ctypes.create_string_buffer(data['sx'].tobytes())
                    # Resident host operand pointers are intentionally absent.
                    inputs = (ctypes.c_void_p*4)(ctypes.addressof(a), ctypes.addressof(sx), 0, 0)
                    output = ctypes.create_string_buffer(30)
                    outputs = (ctypes.c_void_p*1)(ctypes.addressof(output))
                    with patch.object(functional_runtime, 'PhysicalFunctionalInterpreter', SophiaLakeInterpreter):
                        self.assertEqual(library.call(trip, inputs, outputs), 0, runtime.errors)
                    self.assertEqual(output.raw, dense_expected(data))

                for trip, data in enumerate(values):
                    if trip == trips-1:
                        library.interleave()
                    execute(trip, data)
                    for address, payload in immutable:
                        self.assertEqual(bytes(runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)]), payload)
                changed = deepcopy(values[-1])
                changed['w'][-1, -1] = 0x40 if changed['w'][-1, -1] == 0x38 else 0x38
                self.assertNotEqual(dense_expected(changed), dense_expected(values[-1]))
                address = plans[-1]['input_addresses'][1]
                payload = pack_resident_input(sources[-1], sources[-1].nodes[2].id, changed['w'].tobytes())
                runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)] = payload
                execute(trips-1, changed)
                result = runtime.finish()
                self.assertFalse(result['errors'])
                self.assertEqual(len(result['launches']), trips+1)
                self.assertEqual(library.loads(), trips+2)
                self.assertEqual(library.reuses(), 0)
                record = dict(schema='atlas.resident_loop_c_application_test.v1', trips=trips,
                    exact=True, executed_last_weight_mutation=True, shared_body_count=1,
                    image_loads=library.loads(), image_reuses=library.reuses(),
                    starts=len(result['launches']), runtime=receipt,
                    functional_instructions=[row['instructions'] for row in result['launches']])
                (self.root/f'resident-application-c-{trips}.json').write_text(json.dumps(record, indent=2)+'\n')
                print('RESIDENT_APPLICATION_C_EXACT', trips, record['functional_instructions'], flush=True)


if __name__ == '__main__':
    unittest.main()
