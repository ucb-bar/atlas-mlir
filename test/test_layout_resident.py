"""Immutable affine operands retain their raw bits and readonly DMA spans."""
from dataclasses import replace
from pathlib import Path
import ctypes
import json
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from merlin.semantic_compiler.codegen import codegen_options, current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from atlas_mlir_native import layout_device as native
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE
from test_layout_device import cases, request, ROOT


def options(source, ids):
    inventory = {item['node_id']: item for item in AtlasApplicationAdapter.inventory_resident_inputs(source, SimpleNamespace(metadata=(("kind", "layout_device"),)))}
    cursor = DRAM_BASE + 262144
    bindings = {}
    for node_id in ids:
        item = inventory[node_id]
        bindings[node_id] = dict(address=cursor, bytes=item['bytes'], immutable=True)
        cursor += item['bytes']
    return replace(current_codegen_options(), io_memory_bytes=262144,
                   io_memory_capacity_base=DRAM_BASE, io_memory_capacity_bytes=1048576,
                   resident_bindings={source.digest(): bindings})


class LayoutResidentTest(unittest.TestCase):
    def test_inventory_and_packing_follow_the_selected_compiler_route(self):
        from test_rmsnorm import source_for
        from test_already_quantized import source_request

        alternative = lambda kind: SimpleNamespace(metadata=(("kind", kind),))
        norm = source_for()
        # A fused normalization contains an internal reshape, but uses its own
        # compiler and has no dense-layout immutable ABI.
        self.assertIsNone(AtlasApplicationAdapter.inventory_resident_inputs(norm, alternative("rmsnorm")))
        with self.assertRaisesRegex(NativeCompilationError, "no immutable residency ABI"):
            AtlasApplicationAdapter.pack_resident_input(norm, "gamma", bytes(256), alternative("rmsnorm"))
        # Recognition failures on an explicitly selected route remain errors.
        with self.assertRaisesRegex(ValueError, "one pure source operation"):
            AtlasApplicationAdapter.inventory_resident_inputs(norm, alternative("layout_device"))
        source = cases()["dense_dma"]
        selected = alternative("layout_device")
        self.assertEqual(AtlasApplicationAdapter.inventory_resident_inputs(source, selected),
                         native.inventory_resident_inputs(source))
        self.assertEqual(AtlasApplicationAdapter.pack_resident_input(source, "input_0", bytes(192), selected), bytes(192))
        self.assertIsNone(AtlasApplicationAdapter.inventory_resident_inputs(source, alternative("scalar_broadcast")))
        quantized = source_request(1, 1, 1, batch=())
        from atlas_mlir_native.quantized_resident import inventory_resident_inputs
        self.assertEqual(AtlasApplicationAdapter.inventory_resident_inputs(quantized, alternative("already_quantized")),
                         inventory_resident_inputs(quantized))

    def test_inventory_and_pack_keep_raw_bits_and_private_dma_padding(self):
        for label, source in cases().items():
            with self.subTest(label=label):
                plan = native.prepare_source(source)
                inventory = AtlasApplicationAdapter.inventory_resident_inputs(source, SimpleNamespace(metadata=(("kind", "layout_device"),)))
                for node, item, size in zip(plan.operands, inventory, plan.input_bytes, strict=True):
                    raw = bytes(i % 256 for i in range(size))
                    packed = native.pack_resident_input(source, node.id, raw)
                    self.assertEqual(packed, raw + bytes(item['bytes'] - size))
                    self.assertEqual(item['layout']['source_dtype'], node.type.dtype)
                    with self.assertRaisesRegex(ValueError, 'byte length'):
                        native.pack_resident_input(source, node.id, raw[:-1])

    def test_mixed_concat_keeps_staging_geometry_and_checks_bound_ranges(self):
        source = cases()['concat']
        baseline = native.prepare_source(source)
        selected = options(source, ('input_1',))
        with codegen_options(selected):
            plan = native.prepare_source(source)
            self.assertEqual(plan.input_addresses, (baseline.input_addresses[0], DRAM_BASE+262144))
            self.assertEqual(plan.output_address, baseline.output_address)
            self.assertEqual(plan.workspace_bytes, baseline.workspace_bytes)
            self.assertIn(str(DRAM_BASE+262144-(1 << 32)), native.virtual_program(plan))
            with self.assertRaises(NativeCompilationError):
                native.prepare_source(source, io_memory_base=DRAM_BASE+262144)
        # An offset layout frame must still protect the entire I/O window.
        bindings = selected.resident_bindings[source.digest()]
        changed = {'input_1': {**bindings['input_1'], 'address': DRAM_BASE}}
        with codegen_options(replace(selected, resident_bindings={source.digest(): changed})):
            with self.assertRaisesRegex(NativeCompilationError, 'overlaps'):
                native.prepare_source(source, io_memory_base=DRAM_BASE+32)
        for update in (dict(address=DRAM_BASE), dict(address=DRAM_BASE+262145),
                       dict(address=DRAM_BASE+1048576), dict(bytes=1), dict(immutable=False)):
            bindings = selected.resident_bindings[source.digest()]
            changed = {'input_1': {**bindings['input_1'], **update}}
            with self.subTest(update=update), codegen_options(replace(selected, resident_bindings={source.digest(): changed})):
                with self.assertRaises(NativeCompilationError):
                    native.prepare_source(source)
        selected = options(source, ('input_0', 'input_1'))
        bindings = selected.resident_bindings[source.digest()]
        changed = {**bindings, 'input_1': {**bindings['input_1'], 'address': bindings['input_0']['address']}}
        with codegen_options(replace(selected, resident_bindings={source.digest(): changed})):
            with self.assertRaises(NativeCompilationError):
                native.prepare_source(source)

    def test_prefix_fallback_refuses_before_creating_a_publication(self):
        source = request('concat', ((2, 2, 65536), (2, 2, 65536)), (2, 4, 65536), axis=1)
        selected = replace(current_codegen_options(), io_memory_bytes=4194304,
            io_memory_capacity_base=DRAM_BASE, io_memory_capacity_bytes=8388608,
            resident_bindings={source.digest(): {'input_1': dict(address=DRAM_BASE+4194304,bytes=524288,immutable=True)}})
        with codegen_options(selected), tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'compiled'
            with self.assertRaisesRegex(NativeCompilationError, 'single device program'):
                native.compile_source(None, source, target_source=ROOT, destination=dest, limits=None)
            self.assertFalse(dest.exists())

    @unittest.skipUnless(shutil.which('cc'), 'C compiler unavailable')
    def test_generated_c_accepts_null_resident_ports_and_never_uploads_them(self):
        source = request('concat', ((32,), (32,)), (64,), axis=0)
        for resident in (('input_1',), ('input_0', 'input_1')):
            with self.subTest(resident=resident), codegen_options(options(source, resident)), tempfile.TemporaryDirectory() as temp:
                dest = Path(temp)
                compiled = dest/'compiled'
                compiled.mkdir()
                plan = native.prepare_source(source)
                program = {'instructions': [{'word_u32': 0x00100073}]}
                bound = dict(schema='probe', admitted=True)
                virtual = native.virtual_program(plan)
                for name, value in {'manifest.json': dict(lowering_pipeline='mandatory'),
                    'source-request.json': source.record(), 'execution_plan.json': plan.record(),
                    'layout-instruction-bound.json': bound}.items():
                    (compiled/name).write_text(json.dumps(value))
                (compiled/'atlas-virtual.mlir').write_text(virtual)
                (compiled/'atlas-machine.mlir').write_text('machine')
                (compiled/'program.bin').write_bytes((0x00100073).to_bytes(4,'little'))
                with patch.object(native, 'verify_native_publication'), patch.object(native, '_run', side_effect=['machine', json.dumps(program)]), patch('atlas_mlir_native.restoration_budget.instruction_bound', return_value=bound):
                    record, files = native.prepare_c_package(source, compiled, 'probe', target_source=ROOT)
                self.assertEqual(record['resident_bindings'], list(plan.resident_bindings))
                self.assertIn(f'{{{DRAM_BASE+262144}ULL,64u,ATLAS_HOST_DEVICE_READ}}', files['probe.c'].decode())
                for name, payload in files.items():
                    path = dest/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                pointers = ','.join('0' if key in resident else 'input' for key in ('input_0','input_1'))
                shim = '''#include "probe.h"
#include <string.h>
static unsigned starts,puts;static atlas_host host;
static uint32_t read32(void *c,uintptr_t a){(void)c;return a==0x40004u?starts:5u;}
static void write32(void *c,uintptr_t a,uint32_t v){(void)c;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n){(void)c;(void)a;(void)v;(void)n;++puts;return 0;}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n){(void)c;(void)a;memset(v,0,n);return 0;}
int invoke(void){
 probe_context region;probe_source_context source;uint8_t input[64]={0},output[128];(void)input;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0}))return -1;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},0,0,20u))return -2;
 if(probe_source_bind(&source,&region,0,0))return -3;
 const void *inputs[]={@POINTERS@};void *outputs[]={output};
 int result=probe_source_dispatch(&source,inputs,outputs);return result?100+result:(int)puts;
}
'''.replace('@POINTERS@', pointers)
                (dest/'shim.c').write_text(shim)
                proc = subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    'probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'], cwd=dest, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(ctypes.CDLL(str(dest/'probe.so')).invoke(), 2-len(resident))


if __name__ == '__main__':
    unittest.main()
