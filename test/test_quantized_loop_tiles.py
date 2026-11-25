"""On chip, a counted image whose host copy cannot share the I/O window with its frame and dense buffers moves one tile at a time."""
from hashlib import sha256
from math import prod
from pathlib import Path
import ctypes
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native import quantized_loop_c as native
from atlas_mlir_native.quantized_epilogue import prepare_source as prepare_epilogue
from atlas_mlir_native.quantized_loop import execution_layout
from atlas_mlir_native.options import CodegenOptions, codegen_options
from test_already_quantized import source_request
from test_quantized_epilogue import case
from test_quantized_loop import head, shared, output_read_spans

ROOT = Path(__file__).resolve().parents[1]
BASE = 0x200C0000

SHIM = r'''#include "probe.h"
#include <string.h>
static atlas_host host; static probe_context region; static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES],memory[@BYTES@u];
static unsigned starts,puts,gets; static const uint8_t *result; static int mode;
static int span(uintptr_t a,size_t n) { return a>=@BASE@u && a-@BASE@u<=@BYTES@u && n<=@BYTES@u-(a-@BASE@u); }
static uint32_t read32(void *c,uintptr_t a) { (void)c; return a==0x40004u?starts:5u; }
static void write32(void *c,uintptr_t a,uint32_t v) {
  (void)c; if (a!=0x40018u || v!=1u) return;
  uint32_t status=mode==1; ++starts;
  memcpy(memory+(@OUT@u-@BASE@u),result,@OUTBYTES@u); memcpy(memory+(@STATUS@u-@BASE@u),&status,4u);
}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) {
  (void)c; ++puts; if (!span(a,n)) return 1; memcpy(memory+(a-@BASE@u),v,n); return 0;
}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) {
  (void)c; ++gets; if ((mode==2 && a==@OUT@u) || (mode==3 && gets==@READS@u)) return 1;
  if (!span(a,n)) return 1; memcpy(v,memory+(a-@BASE@u),n); return 0;
}
const uint8_t *device_memory(void) { return memory; }
unsigned start_count(void) { return starts; } unsigned write_count(void) { return puts; } unsigned read_count(void) { return gets; }
int invoke(const void *const *inputs,void *out,const uint8_t *res,int fail) {
  starts=puts=gets=0; result=res; mode=fail; memset(memory,0xa5,sizeof(memory));
  if (atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0})) return -1;
  if (probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u)) return -2;
  if (probe_source_bind(&source,&region,converted,sizeof(converted))) return -3;
  void *outputs[]={out}; return probe_source_dispatch(&source,inputs,outputs);
}
'''


def options(window, onchip):
    return codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=window,
                                          application_deployment='onchip' if onchip else None))


def package(source, plan, execution, window, onchip):
    with tempfile.TemporaryDirectory() as temp, options(window, onchip):
        compiled = Path(temp)
        records = {'already-quantized-plan.json': plan.record(), 'source-request.json': source.record(),
                   'manifest.json': {'target_inventory_sha256': sha256((ROOT/'docs/selected-variant-inventory.json').read_bytes()).hexdigest()},
                   'execution_plan.json': execution, 'physical-program.json': {'instructions': [{'word_u32': 0x00100073}]}}
        for key, value in records.items():
            (compiled/key).write_text(json.dumps(value))
        (compiled/'program.bin').write_bytes((0x00100073).to_bytes(4, 'little'))
        with patch.object(native, 'verify_native_publication'), patch.object(native, 'selected_route', return_value='counted'):
            return native.prepare_c_package(source, compiled, 'probe', target_source=ROOT)


def value(source, key):
    node = source.node(key)
    if node.type.dtype == 'i8':
        return np.resize(np.array([0, 8, 0x38, 0x77, 0x78, 0x7e, 0x88, 0xfe], np.uint8), node.type.shape)
    if node.type.dtype == 'i32':
        return (np.arange(prod(node.type.shape), dtype='<i4') % 97 - 48).reshape(node.type.shape)
    if node.type.dtype == 'bf16':
        return np.resize(np.array([0x8000, 0x3f80, 0xbf80, 0x4000, 0x3c00], '<u2'), node.type.shape)
    return np.resize(np.array([-0.0, 1.00390625, -1.01171875, 2.0], '<f4'), node.type.shape)


class TileTransferTest(unittest.TestCase):
    def test_head_sized_contraction_declares_one_tile_of_host_scratch_on_chip(self):
        # A 1x6272 by 6272x2 contraction has a 409,600-byte frame; with its host image it needs 819,200 bytes.
        source = source_request(m=1, n=2, k=6272, batch=())
        plan = native.prepare_source(source)
        with options(786432, True):
            execution = execution_layout(plan)
        frame = execution['status_address']+execution['status_bytes']-BASE
        self.assertEqual(frame, 409600)
        self.assertEqual(package(source, plan, execution, 786432, False)[0]['workspace_bytes'], frame)
        self.assertEqual(package(source, plan, execution, 786432, True)[0]['workspace_bytes'], 2048)

    def test_dense_inputs_and_output_sharing_the_window_also_require_tile_transfers_on_chip(self):
        # A 1x5632 by 5632x32 contraction: its 368,640-byte frame and host image fit 786,432 bytes, but with the
        # 185,988 dense source bytes and the 64-byte output and its converted copy they do not.
        source = source_request(m=1, n=32, k=5632, batch=())
        plan = native.prepare_source(source)
        with options(786432, True):
            execution = execution_layout(plan)
        frame = execution['status_address']+execution['status_bytes']-BASE
        dense = sum(prod(source.node(key).type.shape)*(4 if source.node(key).type.dtype == 'i32' else 1)
                    for key in plan.input_ids)
        self.assertEqual((frame, dense), (368640, 185988))
        self.assertLessEqual(2*frame, 786432)
        self.assertEqual(package(source, plan, execution, 786432, False)[0]['workspace_bytes'], frame)
        self.assertEqual(package(source, plan, execution, 786432, True)[0]['workspace_bytes'], 2048)

    def test_tile_transfers_leave_device_memory_and_outputs_unchanged(self):
        sources = [(shared, native.prepare_source(shared)), (head, native.prepare_source(head))]
        sources += [(source, prepare_epilogue(source)) for source in (case('bf16', 0)[0], case('f32', 1)[0])]
        for source, plan in sources:
            with self.subTest(output=source.outputs[0], dtype=source.node(source.outputs[0]).type.dtype):
                with options(786432, True):
                    execution = execution_layout(plan)
                window = execution['input_addresses'][-1]+execution['input_buffer_bytes'][-1]-BASE \
                    if len(execution['input_addresses']) == 3 else execution['status_address']+execution['status_bytes']-BASE
                runs = {}
                for onchip in (False, True):
                    record, files = package(source, plan, execution, window, onchip)
                    runs[onchip] = (record['workspace_bytes'], *self.run_package(source, plan, execution, window, files))
                full, tiles = runs[False], runs[True]
                self.assertEqual(full[0], window)
                self.assertEqual(tiles[0], 2048)
                # The same bytes reach every device buffer, and the restored outputs are equal.
                self.assertEqual(full[1], tiles[1])
                np.testing.assert_array_equal(full[2], tiles[2])
                self.assertTrue(np.any(full[2].view(np.uint8)))
                batches, (mc, nc, kc) = prod(plan.batch_shape), ((v+31)//32 for v in (plan.m, plan.n, plan.k))
                rhs = batches if source.node(plan.operand_ids[1]).type.shape[:-2] else 1
                pages = sum(geometry['pages'] for geometry in execution['exponent_layouts'])
                extra = batches*mc*nc if len(execution['input_addresses']) == 3 else 0
                reads=1+len(output_read_spans(plan))
                self.assertEqual(full[3], (1, 2+pages+bool(extra), reads))
                self.assertEqual(tiles[3], (1, batches*mc*kc+rhs*nc*kc+pages+extra, reads))

    def run_package(self, source, plan, execution, window, files):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for key, payload in files.items():
                path = root/key; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
            text = SHIM
            for key, item in {'BASE': BASE, 'BYTES': window, 'OUT': execution['output_addresses'][0],
                              'OUTBYTES': execution['output_bytes'], 'STATUS': execution['status_address'],
                              'READS': 1+len(output_read_spans(plan))}.items():
                text = text.replace('@'+key+'@', str(item))
            (root/'shim.c').write_text(text)
            proc = subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation',
                                   '-shared', '-fPIC', 'probe.c', 'runtime/atlas_host.c', 'shim.c', '-o', 'probe.so'],
                                  cwd=root, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lib = ctypes.CDLL(str(root/'probe.so'))
            lib.invoke.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]
            lib.device_memory.restype = ctypes.POINTER(ctypes.c_uint8*window)
            values = [value(source, key) for key in plan.input_ids]
            inputs = (ctypes.c_void_p*len(values))(*(item.ctypes.data for item in values))
            output = np.empty(source.node(plan.output_id).type.shape, '<u2' if plan.output_dtype == 'bf16' else '<f4')
            device = (0x3f80+np.arange(execution['output_bytes']//2, dtype=np.uint16) % 64).astype('<u2')
            for mode, status in ((1, 2), (2, 3), (3, 3)):
                output.view(np.uint8).fill(0xa5)
                self.assertEqual(lib.invoke(ctypes.cast(inputs, ctypes.c_void_p), output.ctypes.data, device.ctypes.data, mode), status)
                self.assertTrue(np.all(output.view(np.uint8) == 0xa5))
            output.view(np.uint8).fill(0xa5)
            self.assertEqual(lib.invoke(ctypes.cast(inputs, ctypes.c_void_p), output.ctypes.data, device.ctypes.data, 0), 0)
            return (bytes(lib.device_memory().contents), output.copy(),
                    (lib.start_count(), lib.write_count(), lib.read_count()))


if __name__ == '__main__':
    unittest.main()
