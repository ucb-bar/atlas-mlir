"""Resident layouts preserve dense bytes and the counted transport contract."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import ctypes
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from merlin.semantic_compiler.codegen import codegen_options, current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from atlas_mlir_native import quantized_loop_c as native
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_resident import inventory_resident_inputs, pack_resident_input
from test_already_quantized import source_request

ATLAS = Path(__file__).resolve().parents[1]


def options(source, ids):
    inventory = {item['node_id']: item for item in inventory_resident_inputs(source)}
    cursor = DRAM_BASE+262144
    bindings = {}
    for node_id in ids:
        item = inventory[node_id]
        bindings[node_id] = dict(address=cursor, bytes=item['bytes'], immutable=True)
        cursor += (item['bytes']+2047)//2048*2048
    return replace(current_codegen_options(), io_memory_bytes=262144,
                   io_memory_capacity_base=DRAM_BASE, io_memory_capacity_bytes=1048576,
                   resident_bindings={source.digest(): bindings})


class QuantizedResidentTest(unittest.TestCase):
    def test_packing_preserves_folded_transposes_and_scalar_batch_exponents(self):
        from test_quantized_operand_permutation import fixture

        for operand in ('a', 'w'):
            logical, source = fixture(operand, m=3, n=5, k=7, batch=(2,))
            shape = logical.node(operand).type.shape
            values = np.resize(np.array([0, 8, 0x38, 0x7e, 0x88, 0xfe], np.uint8), shape)
            physical = values.swapaxes(-1, -2).copy()
            self.assertEqual(pack_resident_input(source, operand+'_physical', physical.tobytes()),
                             pack_resident_input(logical, operand, values.tobytes()))
        source = source_request(3, 5, 7, batch=(2,))
        source = replace(source, nodes=tuple(
            replace(node, type=replace(node.type, shape=())) if node.id == 'sx' else
            replace(node, type=replace(node.type, shape=(2,5,7))) if node.id == 'w' else
            replace(node, type=replace(node.type, shape=(2,1,1))) if node.id == 'sw' else node
            for node in source.nodes))
        self.assertEqual(pack_resident_input(source, 'sx', (-127).to_bytes(4,'little',signed=True)),
                         (-127).to_bytes(4,'little',signed=True)+bytes(2044))
        dense = np.array([[-127], [127]], dtype='<i4').tobytes()
        self.assertEqual(pack_resident_input(source, 'sw', dense),
                         dense[:4]+bytes(2044)+dense[4:]+bytes(2044))

    def test_pack_matches_counted_tiles_and_pages_with_tails(self):
        source = source_request(35, 37, 65, batch=(2,))
        for item in inventory_resident_inputs(source):
            node_id = item['node_id']
            shape = source.node(node_id).type.shape
            values = np.full(shape, 0x38, np.uint8) if item['kind'] == 'fp8' else np.arange(np.prod(shape), dtype='<i4').reshape(shape)
            packed = pack_resident_input(source, node_id, values.tobytes())
            self.assertEqual(len(packed), item['bytes'])
            layout = item['layout']
            if item['kind'] == 'fp8':
                dense = values.reshape(layout['batch_count'], *shape[-2:])
                for batch in range(layout['batch_count']):
                    for row in range(layout['row_tiles']*32):
                        for k in range(layout['k_tiles']*32):
                            at = ((batch*layout['row_tiles']+row//32)*layout['k_tiles']+k//32)*1024+row%32*32+k%32
                            self.assertEqual(packed[at], int(dense[batch,row,k]) if row<shape[-2] and k<shape[-1] else 0)
            elif layout['mode'] == 'row':
                for row in range(layout['rows']):
                    self.assertEqual(packed[(row//32)*2048+(row%32)*4:][:4], values.tobytes()[row*4:][:4])
                self.assertEqual(packed[128:2048], bytes(1920))
            else:
                self.assertEqual(packed[:4], values.tobytes()[:4])
        with self.assertRaisesRegex(ValueError, 'finite normal'):
            pack_resident_input(source, 'w', bytes([1])*np.prod(source.node('w').type.shape))
        with self.assertRaisesRegex(ValueError, 'byte length'):
            pack_resident_input(source, 'sw', b'')

    def test_layout_binds_exact_spans_and_preserves_transient_geometry(self):
        source = source_request(3, 5, 7, batch=())
        plan = prepare_source(source)
        baseline = execution_layout(plan)
        selected = options(source, ('w', 'sw'))
        with codegen_options(selected):
            layout = execution_layout(plan)
            self.assertEqual(layout['input_addresses'][1], DRAM_BASE+262144)
            self.assertEqual(layout['exponent_addresses'][1], DRAM_BASE+264192)
            self.assertEqual(layout['output_addresses'], baseline['output_addresses'])
            signed_address = layout['input_addresses'][1]-(1 << 32)
            self.assertIn(f'arith.constant {signed_address} : i32', virtual_source(plan, layout))
            self.assertEqual([item['node_id'] for item in layout['resident_bindings']], ['sw', 'w'])
        for update in (dict(address=DRAM_BASE), dict(address=DRAM_BASE+262145),
                       dict(address=DRAM_BASE+1048576), dict(bytes=1), dict(immutable=False)):
            bindings = {key: dict(value) for key, value in selected.resident_bindings[source.digest()].items()}
            bindings['w'].update(update)
            with self.subTest(update=update), codegen_options(replace(selected, resident_bindings={source.digest(): bindings})):
                with self.assertRaises(NativeCompilationError):
                    execution_layout(plan)
        bindings = selected.resident_bindings[source.digest()]
        overlapping = {**bindings, 'sw': {**bindings['sw'], 'address': bindings['w']['address']}}
        with codegen_options(replace(selected, resident_bindings={source.digest(): overlapping})):
            with self.assertRaises(NativeCompilationError):
                execution_layout(plan)

    def test_resident_fallback_is_an_explicit_refusal(self):
        from atlas_mlir_native import quantized_loop_sizing as sizing
        source = source_request(1, 1, 1, batch=())
        with codegen_options(options(source, ('w',))), patch.object(sizing, '_size_counted') as mock:
            mock.return_value.record.return_value = dict(route='k32_tiles', reason='selected_dram_window')
            with self.assertRaisesRegex(NativeCompilationError, 'require an admitted counted'):
                sizing.size_counted(prepare_source(source))

    @unittest.skipUnless(shutil.which('cc'), 'C compiler unavailable')
    def test_generated_c_keeps_read_spans_and_omits_only_resident_writes(self):
        source = source_request(1, 1, 1, batch=())
        for resident in (('w',), ('sw',), ('w', 'sw'), ('a', 'sx', 'w', 'sw')):
            with self.subTest(resident=resident), codegen_options(options(source, resident)), tempfile.TemporaryDirectory() as temp:
                dest = Path(temp)
                compiled = dest/'compiled'
                compiled.mkdir()
                plan = prepare_source(source)
                layout = execution_layout(plan)
                records = {'already-quantized-plan.json': plan.record(), 'source-request.json': source.record(),
                    'manifest.json': {'target_inventory_sha256': sha256((ATLAS/'docs/selected-variant-inventory.json').read_bytes()).hexdigest()},
                    'execution_plan.json': layout, 'physical-program.json': {'instructions': [{'word_u32': 0x00100073}]}}
                for key, value in records.items():
                    (compiled/key).write_text(json.dumps(value))
                (compiled/'program.bin').write_bytes((0x00100073).to_bytes(4,'little'))
                with patch.object(native, 'verify_native_publication'), patch.object(native, 'selected_route', return_value='counted'):
                    record, files = native.prepare_c_package(source, compiled, 'probe', target_source=ATLAS)
                self.assertEqual(record['resident_bindings'], layout['resident_bindings'])
                for key, value in files.items():
                    path = dest/key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value)
                pointers = ','.join('0' if node_id in resident else '&code' if node_id in ('a','w') else '&exponent'
                                    for node_id in plan.input_ids)
                shim = '''#include "probe.h"
#include <string.h>
static unsigned starts,puts;static atlas_host host;
static uint32_t read32(void *c,uintptr_t a){(void)c;return a==0x40004u?starts:5u;}
static void write32(void *c,uintptr_t a,uint32_t v){(void)c;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n){(void)c;(void)a;(void)v;(void)n;++puts;return 0;}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n){(void)c;(void)a;memset(v,0,n);return 0;}
int invoke(void){
 probe_context region;probe_source_context source;
 uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES];
 uint8_t code=0x38;int32_t exponent=0;uint16_t out=0xffff;
 (void)code;(void)exponent;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0}))return -1;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u))return -2;
 if(probe_source_bind(&source,&region,converted,sizeof(converted)))return -3;
 const void *inputs[]={@POINTERS@};void *outputs[]={&out};
 int result=probe_source_dispatch(&source,inputs,outputs);
 return result?100+result:out?200:(int)puts;
}
'''.replace('@POINTERS@', pointers)
                (dest/'shim.c').write_text(shim)
                proc = subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                                       'probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'], cwd=dest, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                lib = ctypes.CDLL(str(dest/'probe.so'))
                self.assertEqual(lib.invoke(), 4-len(resident))


if __name__ == '__main__':
    unittest.main()
