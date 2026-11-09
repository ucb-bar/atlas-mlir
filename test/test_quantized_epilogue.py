"""Quantized epilogues preserve intake, source guards and five-input C ABI."""
import ctypes
from dataclasses import replace
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source as prepare_dispatch
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_epilogue import alternatives, prepare_source
from atlas_mlir_native import quantized_loop_c as native_c
from atlas_mlir_native.quantized_loop import execution_layout
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, CycleInterval, RegionAlternative, ValueRepresentation
from test_already_quantized import source_request
from test_quantized_loop import packed

ROOT = Path(__file__).resolve().parents[1]


def case(dtype='bf16', operand=0, *, escape=False, m=3, n=35, k=65, batch=(2,)):
    contraction = source_request(m, n, k, batch)
    if dtype == 'f32':
        original = contraction.nodes[-1]
        result = replace(original, type=replace(original.type, dtype='f32'),
            attrs=tuple((key, 'float32' if key == 'output_dtype' else value) for key, value in original.attrs))
        contraction = replace(contraction, nodes=(*contraction.nodes[:-1], result))
    result = contraction.nodes[-1]
    other = SemanticNode('other', 'input', (), result.type, effect='input')
    inputs = (result.id, other.id) if operand == 0 else (other.id, result.id)
    product = SemanticNode('product', dtype + '_mul', inputs, result.type,
        (('numeric_semantics', 'ieee754-' + dtype + '-mul-rne-v1'),),
        index_maps=pointwise_identity_maps(len(result.type.shape), 2))
    source = replace(contraction, nodes=(*contraction.nodes, other, product),
        outputs=(product.id, result.id) if escape else (product.id,),
        output_storages=('external', 'external') if escape else ('external',),
        input_storages=(*contraction.input_storages, (other.id, 'external')))
    ports = lambda keys: tuple(BufferRepresentation(key, ValueRepresentation('host_memory', 'dense', source.node(key).type.dtype))
                               for key in dict.fromkeys(keys))
    choices = tuple(RegionAlternative('r' + str(index), (index,), 'source', 'accelerator',
        CycleInterval.unknown('test'), ports(node.inputs), ports((node.id,)), metadata=(('kind', kind),))
        for index, (node, kind) in enumerate(((result, 'already_quantized'), (product, 'pointwise'))))
    return source, contraction, choices


def packed_extra(codes, plan):
    batches, mc, nc = prod(plan.batch_shape), (plan.m + 31) // 32, (plan.n + 31) // 32
    out = np.zeros((batches, mc, nc, 1024), dtype='<u2')
    dense = codes.reshape(batches, plan.m, plan.n)
    for batch in range(batches):
        for row in range(plan.m):
            for column in range(plan.n):
                at = ((column % 32) // 16) * 512 + (row % 32) * 16 + column % 16
                out[batch, row // 32, column // 32, at] = dense[batch, row, column]
    return out


def packed_exponents(value, geometry):
    out = np.zeros((geometry['pages'], 512), dtype='<i4')
    if geometry['mode'] == 'row':
        dense = value.reshape(geometry['batches'], geometry['rows'])
        for batch in range(geometry['batches']):
            for row in range(geometry['rows']):
                out[batch * geometry['tiles'] + row // 32, row % 32] = dense[batch, row]
    else:
        out[:, 0] = value.reshape(-1)
    return out


class QuantizedEpilogueTest(unittest.TestCase):
    def test_plan_keeps_contraction_intake_and_ordered_arithmetic(self):
        for dtype in ('bf16', 'f32'):
            for operand in (0, 1):
                source, original, _ = case(dtype, operand)
                old, plan = prepare_dispatch(original), prepare_dispatch(source)
                self.assertEqual(plan.source, source)
                self.assertEqual((plan.region_id, plan.epilogue_other_id, plan.epilogue_contract_operand),
                                 ('product', 'other', operand))
                self.assertEqual(plan.output_dtype, dtype)
                self.assertEqual(plan.input_ids, (*old.input_ids, 'other'))
                self.assertEqual((plan.operand_ids, plan.exponent_ids, plan.batch_shape, plan.m, plan.n, plan.k),
                                 (old.operand_ids, old.exponent_ids, old.batch_shape, old.m, old.n, old.k))
                self.assertEqual(plan.reset.nodes, old.reset.nodes)
                self.assertEqual(plan.continuation.nodes, old.continuation.nodes)
                self.assertEqual(plan.epilogue_precision.request.nodes[-1].op, 'mul_bf16_zero')
                self.assertEqual(plan.record()['schema'], 'atlas.already_quantized_epilogue.v1')
                self.assertEqual(old.record()['schema'], 'atlas.already_quantized_contraction.v1')

    def test_only_adjacent_dense_nonescaping_original_products_are_selected(self):
        for dtype in ('bf16', 'f32'):
            source, _, choices = case(dtype)
            adapter = SimpleNamespace()
            candidates = [row for row in alternatives(adapter, source, choices)
                          if dict(row.metadata)['kind'] == 'quantized_epilogue']
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].nodes, (0, 1))
            self.assertEqual(len(adapter._quantized_epilogue_groups), 1)
            escaped, _, choices = case(dtype, escape=True)
            self.assertEqual(len(alternatives(adapter, escaped, choices)), 2)
            self.assertEqual(len(alternatives(adapter, source, (choices[0], replace(choices[1], placement='host')))), 2)
            product = source.nodes[-1]
            changed = replace(product, attrs=(('numeric_semantics', 'changed-rounding'),))
            with self.assertRaises(ValueError):
                prepare_source(replace(source, nodes=(*source.nodes[:-1], changed)))
            larger = replace(source.node('other'), type=replace(product.type, shape=(*product.type.shape[:-1], product.type.shape[-1] + 1)))
            with self.assertRaises(ValueError):
                prepare_source(replace(source, nodes=tuple(larger if node.id == 'other' else node for node in source.nodes)))

    def test_sizing_timeout_propagates_and_proven_refusal_declines(self):
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        source, _, choices = case()
        timeout = NativeCompilationError('search_timeout', 'selected tool exceeded its wall limit')
        with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', side_effect=timeout):
            with self.assertRaises(NativeCompilationError) as raised:
                alternatives(SimpleNamespace(), source, choices)
        self.assertIs(raised.exception, timeout)
        refusal = SimpleNamespace(record=lambda: {'route': 'k32_tiles'})
        adapter = SimpleNamespace()
        with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', return_value=refusal):
            self.assertEqual(alternatives(adapter, source, choices), choices)
        self.assertEqual(adapter._quantized_epilogue_groups, {})

    @unittest.skipUnless(shutil.which('cc'), 'C compiler unavailable')
    def test_c_extra_guard_transport_aliases_and_atomic_status(self):
        for dtype, operand in (('bf16', 0), ('f32', 1)):
            with self.subTest(dtype=dtype, operand=operand), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _, _ = case(dtype, operand)
                plan = prepare_source(source)
                layout = execution_layout(plan)
                self.assertEqual(len(layout['input_addresses']), 3)
                self.assertEqual(layout['input_addresses'][2], layout['status_address'] + layout['status_bytes'])
                self.assertEqual(layout['input_buffer_bytes'][2], layout['output_bytes'])
                compiled = root / 'compiled'; compiled.mkdir()
                records = {'already-quantized-plan.json': plan.record(), 'source-request.json': source.record(),
                    'manifest.json': {'target_inventory_sha256': sha256((ROOT / 'docs/selected-variant-inventory.json').read_bytes()).hexdigest()},
                    'execution_plan.json': layout, 'physical-program.json': {'instructions': [{'word_u32': 0x00100073}]}}
                for name, value in records.items(): (compiled / name).write_text(json.dumps(value))
                (compiled / 'program.bin').write_bytes((0x00100073).to_bytes(4, 'little'))
                with patch.object(native_c, 'verify_native_publication'), \
                     patch.object(native_c, 'selected_route', return_value='counted'):
                    record, files = native_c.prepare_c_package(source, compiled, 'probe', target_source=ROOT)
                    malformed = {**layout, 'input_addresses': [*layout['input_addresses'][:2], layout['status_address']]}
                    (compiled / 'execution_plan.json').write_text(json.dumps(malformed))
                    with self.assertRaises(ValueError): native_c.prepare_c_package(source, compiled, 'probe', target_source=ROOT)
                self.assertEqual(record['kind'], 'quantized_epilogue')
                for name, payload in files.items():
                    path = root / name; path.parent.mkdir(exist_ok=True, parents=True); path.write_bytes(payload)
                addresses = [*layout['input_addresses'][:2], *layout['exponent_addresses'], layout['input_addresses'][2]]
                sizes = [*layout['input_buffer_bytes'][:2], *layout['exponent_buffer_bytes'], layout['input_buffer_bytes'][2]]
                shim = r'''#include "probe.h"
#include <string.h>
static atlas_host host; static probe_context region; static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned calls,starts,puts,gets; static const uint8_t *expected[5],*result; static int mode;
static uint32_t read32(void *c,uintptr_t a) { (void)c; ++calls; return a==0x40004u?starts:5u; }
static void write32(void *c,uintptr_t a,uint32_t v) { (void)c; ++calls; if(a==0x40018u&&v==1u) ++starts; }
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) {
  (void)c; ++calls; ++puts; const uintptr_t addresses[]={@ADDRESSES@}; const size_t sizes[]={@SIZES@};
  for(unsigned i=0;i<5u;++i) { if(a==addresses[i]&&n==sizes[i]) return memcmp(v,expected[i],n)!=0; }
  return 1;
}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) {
  (void)c; ++calls; ++gets;
  if(a==@STATUS@&&n==4u) { uint32_t status=mode?1u:0u; memcpy(v,&status,4u); return 0; }
  if(a==@OUT@&&n==@OUTBYTES@) { memcpy(v,result,n); return 0; }
  return 1;
}
unsigned callback_count(void) { return calls; } unsigned start_count(void) { return starts; }
unsigned write_count(void) { return puts; } unsigned read_count(void) { return gets; }
int invoke(const void *a,const void *sx,const void *w,const void *sw,const void *other,void *out,
           const uint8_t *pa,const uint8_t *pw,const uint8_t *pea,const uint8_t *pew,const uint8_t *px,
           const uint8_t *res,int fail) {
  calls=starts=puts=gets=0; expected[0]=pa; expected[1]=pw; expected[2]=pea; expected[3]=pew; expected[4]=px; result=res; mode=fail;
  if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0})) return -1;
  if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u)) return -2;
  if(probe_source_bind(&source,&region,converted,sizeof(converted))) return -3;
  const void *inputs[]={a,sx,w,sw,other}; void *outputs[]={out};
  return probe_source_dispatch(&source,inputs,outputs);
}
'''
                replacements = {'ADDRESSES': ','.join(str(value) + 'u' for value in addresses),
                    'SIZES': ','.join(str(value) + 'u' for value in sizes), 'STATUS': str(layout['status_address']) + 'u',
                    'OUT': str(layout['output_addresses'][0]) + 'u', 'OUTBYTES': str(layout['output_bytes']) + 'u'}
                for key, value in replacements.items(): shim = shim.replace('@' + key + '@', value)
                (root / 'shim.c').write_text(shim)
                process = subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    'probe.c', 'runtime/atlas_host.c', 'shim.c', '-o', 'probe.so'], cwd=root, text=True, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                lib = ctypes.CDLL(str(root / 'probe.so')); lib.invoke.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int]
                a = np.resize(np.array([0, 8, 0x38, 0x78, 0xfe], np.uint8), source.node('a').type.shape)
                w = np.resize(np.array([0x38, 0x08, 0x88, 0, 0x7e], np.uint8), source.node('w').type.shape)
                sx = np.zeros(source.node('sx').type.shape, dtype='<i4')
                sw = np.arange(prod(source.node('sw').type.shape), dtype='<i4').reshape(source.node('sw').type.shape) - 128
                if dtype == 'bf16':
                    other = np.resize(np.array([0, 0x8000, 0x3f80, 0xbf80, 0x4000], '<u2'), source.node('other').type.shape)
                    codes = other
                else:
                    other = np.resize(np.array([0.0, -0.0, 1.00390625, -1.01171875, 2.0], '<f4'), source.node('other').type.shape)
                    codes = np.frombuffer(plan.epilogue_precision.operand_specs[1 - operand].encode(other), dtype='<u2')
                pa, pw = packed(a), packed(w)
                pea, pew = (packed_exponents(value, geometry) for value, geometry in zip((sx, sw), layout['exponent_layouts'], strict=True))
                px = packed_extra(codes, plan)
                device = (0x3f80 + np.arange(layout['output_bytes'] // 2, dtype=np.uint16) % 64).astype('<u2')
                logical = np.zeros((prod(plan.batch_shape), plan.m, plan.n), dtype='<u2')
                for batch in range(prod(plan.batch_shape)):
                    for row in range(plan.m):
                        for column in range(plan.n):
                            at = ((batch * ((plan.m + 31) // 32) + row // 32) * ((plan.n + 31) // 32) + column // 32) * 1024
                            logical[batch, row, column] = device[at + (column % 32 // 16) * 512 + (row % 32) * 16 + column % 16]
                expected = logical if dtype == 'bf16' else (logical.astype('<u4') << 16).view('<f4')
                output = np.empty_like(expected)
                def invoke(mode=0, target=output):
                    output.view(np.uint8).fill(0xa5)
                    return lib.invoke(*(value.ctypes.data for value in (a, sx, w, sw, other, target, pa, pw, pea, pew, px, device)), mode)
                frozen = [value.tobytes() for value in (a, sx, w, sw, other)]
                self.assertEqual(invoke(), 0); np.testing.assert_array_equal(output, expected)
                self.assertEqual((lib.start_count(), lib.write_count(), lib.read_count()), (1, 5, 2))
                self.assertEqual(invoke(1), 2); self.assertTrue(np.all(output.view(np.uint8) == 0xa5))
                self.assertEqual((lib.start_count(), lib.write_count(), lib.read_count()), (1, 5, 1))
                self.assertEqual(invoke(target=other), 1); self.assertEqual(lib.callback_count(), 0)
                saved = other.flat[-1]; other.flat[-1] = 0x7f80 if dtype == 'bf16' else np.nan
                self.assertEqual(invoke(), 2); self.assertEqual(lib.callback_count(), 0)
                self.assertTrue(np.all(output.view(np.uint8) == 0xa5)); other.flat[-1] = saved
                self.assertEqual([value.tobytes() for value in (a, sx, w, sw, other)], frozen)


if __name__ == '__main__':
    unittest.main()
