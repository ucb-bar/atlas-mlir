"""Remove only a private exact cast of an admitted BF16 contraction readout."""
import ctypes
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source as prepare_dispatch
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_narrowing import alternatives, prepare_source
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_quantized_epilogue import case as product_case

ROOT = Path(__file__).resolve().parents[1]


def case(**kwargs):
    _, contraction, choices = product_case('f32', **kwargs)
    result = contraction.nodes[-1]
    cast = SemanticNode('narrowed', 'f32_to_bf16', (result.id,),
        replace(result.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(result.type.shape), 1))
    source = replace(contraction, nodes=(*contraction.nodes, cast), outputs=(cast.id,))
    port = replace(choices[0].outputs[0], buffer=cast.id,
                   representation=replace(choices[0].outputs[0].representation, dtype='bf16'))
    choices = (choices[0], replace(choices[1], inputs=choices[0].outputs, outputs=(port,),
                                  metadata=(('kind', 'exact_float_device'),)))
    return source, contraction, choices


class QuantizedNarrowingTest(unittest.TestCase):
    def test_real_application_adapter_selects_the_cast_group(self):
        from atlas_mlir_native.already_quantized import POLICY
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from test_application_float_arithmetic import policy

        selected = {**policy(), 'already_quantized': POLICY}
        base = AtlasApplicationAdapter(selected, original_source_sha256=selected['source_sha256'])
        source, _, _ = case()
        self.assertEqual([dict(row.metadata)['kind'] for row in base.plan(source).selected],
                         ['already_quantized', 'exact_float_device'])
        group, = AdjacentDeviceAdapter(base).plan(source).selected
        self.assertEqual(dict(group.metadata)['kind'], 'quantized_narrowing')

    def test_private_exact_cast_keeps_arithmetic_and_device_program(self):
        for m, n, k, batch in ((3,35,65,(2,)), (32,32,32,()), (1,33,2049,())):
            source, contraction, _ = case(m=m, n=n, k=k, batch=batch)
            old, plan = prepare_dispatch(contraction), prepare_dispatch(source)
            self.assertEqual(plan.input_ids, old.input_ids)
            self.assertEqual(plan.output_id, 'narrowed')
            self.assertEqual(plan.output_dtype, 'bf16')
            self.assertEqual(plan.reset.nodes, old.reset.nodes)
            self.assertEqual(plan.continuation.nodes, old.continuation.nodes)
            self.assertEqual(plan.record()['schema'], 'atlas.already_quantized_narrowing.v1')
            self.assertEqual(execution_layout(plan), execution_layout(old))
            self.assertEqual(virtual_source(plan, execution_layout(plan)),
                             virtual_source(old, execution_layout(old)))

    def test_rejects_changed_contracts_and_preserves_smaller_choices(self):
        source, _, choices = case()
        base = SimpleNamespace(plan=lambda request: SimpleNamespace(selected=choices))
        adapter = AdjacentDeviceAdapter(base)
        rows = adapter.application_region_alternatives(source)
        self.assertEqual(len(rows), 3)
        self.assertEqual(dict(rows[-1].metadata)['kind'], 'quantized_narrowing')
        self.assertEqual(rows[-1].nodes, (0, 1))
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(dict(selected[0].metadata)['kind'], 'quantized_narrowing')
        cast = source.nodes[-1]
        changed = [replace(source, outputs=('y', 'narrowed'), output_storages=('external', 'external'))]
        for node in (replace(cast, attrs=()), replace(cast, index_maps=()),
                     replace(cast, type=replace(cast.type, dtype='f32')),
                     replace(cast, inputs=('a',)), replace(cast, op='bf16_to_f32')):
            changed.append(replace(source, nodes=(*source.nodes[:-1], node)))
        extra = replace(source.nodes[0], id='unused')
        changed.append(replace(source, nodes=(extra, *source.nodes),
                               input_storages=(('unused','external'),*source.input_storages)))
        for request in changed:
            with self.subTest(request=request.digest()), self.assertRaises(ValueError):
                prepare_source(request)
        self.assertEqual(len(alternatives(adapter, changed[0], choices)), 2)
        self.assertEqual(len(alternatives(adapter, source,
            (choices[0], replace(choices[1], placement='host')))), 2)
        refusal = SimpleNamespace(record=lambda: {'route': 'k32_tiles'})
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', return_value=refusal):
            self.assertEqual(alternatives(adapter, source, choices), choices)
            from atlas_mlir_native.quantized_loop import compile_source
            with self.assertRaisesRegex(NativeCompilationError, 'separate narrowing cast'):
                compile_source(None, source, target_source=ROOT, destination=None, limits=None)
        from atlas_mlir_native.quantized_loop_c import prepare_c_package
        with patch('atlas_mlir_native.quantized_loop_c.selected_route', return_value='k32_tiles'):
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'separate narrowing cast'):
                prepare_c_package(source, directory, 'probe', target_source=ROOT)
        timeout = NativeCompilationError('search_timeout', 'native timeout')
        with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', side_effect=timeout):
            with self.assertRaises(NativeCompilationError):
                alternatives(adapter, source, choices)

    def test_every_finite_bf16_widening_has_identity_host_narrowing(self):
        from test_exact_float_device import host_reference, request
        codes = np.arange(65536, dtype='<u2')
        codes = codes[(codes & 0x7f80) != 0x7f80]
        widened = (codes.astype('<u4') << 16).tobytes()
        # Independent Merlin integer RNE cast, including both zeros and all
        # subnormals (a superset of the contraction's admitted output domain).
        result, status = host_reference(request('f32_to_bf16', len(codes)), [widened])
        self.assertEqual(status, 0)
        self.assertEqual(result, codes.tobytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select pinned model')
    def test_cycle_model_preserves_ordered_result_padding_and_rejection(self):
        from atlas_mlir_native.binding import DRAM_BASE
        from test_exponent_restoration import compile_fixture
        from test_quantized_operand_reuse import cycle
        from test_quantized_loop import packed
        from test_quantized_epilogue import packed_extra, packed_exponents

        source, _, _ = case(m=3, n=35, k=65, batch=(2,))
        plan = prepare_source(source)
        layout = execution_layout(plan)
        _, program = compile_fixture(virtual_source(plan, layout))
        memory = bytearray([0xa5])*(layout['status_address']-DRAM_BASE+layout['status_bytes']+4096)
        for key, address in zip(plan.operand_ids, layout['input_addresses']):
            payload = packed(np.full(source.node(key).type.shape, 0x38, np.uint8)).tobytes()
            memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)] = payload
        sx = np.zeros(source.node('sx').type.shape, '<i4')
        sw = (np.arange(35, dtype='<i4')%3-1).reshape(35,1)
        for values, geometry, address in zip((sx,sw), layout['exponent_layouts'], layout['exponent_addresses']):
            payload = packed_exponents(values, geometry).tobytes()
            memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)] = payload
        actual, _ = cycle(program, layout, memory)
        # 65 products of exactly one, then an exact per-column power of two.
        expected = np.broadcast_to((0x4282+sw[:,0]*128).astype('<u2'), (2,3,35))
        wanted = bytearray(memory)
        at = layout['output_addresses'][0]-DRAM_BASE
        payload = packed_extra(expected, plan).tobytes()
        wanted[at:at+len(payload)] = payload
        status = layout['status_address']-DRAM_BASE
        wanted[status:status+layout['status_bytes']] = bytes(layout['status_bytes'])
        self.assertEqual(actual, wanted)
        # Last exponent page rejects after earlier output tiles have completed.
        at = layout['exponent_addresses'][1]-DRAM_BASE+2048
        memory[at:at+4] = (1000).to_bytes(4, 'little')
        rejected, _ = cycle(program, layout, memory)
        self.assertEqual(int.from_bytes(rejected[status:status+4], 'little'), 1)

    def test_native_binding_exports_bf16_and_atomic_source_dispatch(self):
        from atlas_mlir_native import quantized_loop, quantized_loop_c
        from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
        from atlas_mlir_native.quantized_narrowing import export_group
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        from test_quantized_epilogue import packed_extra

        source, _, _ = case(m=3, n=35, k=65, batch=(2,))
        canonical = replace(source, source_identity='canonical-narrowing-test')
        plan = prepare_source(source)
        layout = execution_layout(plan)
        with tempfile.TemporaryDirectory(prefix='atlas-narrowing-') as directory:
            root = Path(directory)
            capture = SelectionCapture(build_snapshot(root/'native'))
            quantized_loop.compile_source(capture, canonical, target_source=ROOT,
                destination=root/'compiled', limits=SearchLimits())
            bind_counted(source, canonical, root/'compiled', root/'bound', capture, target_source=ROOT)
            package = export_group(plan, root/'bound', 'probe', target_source=ROOT)
            record, files = quantized_loop_c.prepare_c_package(source, root/'bound', 'probe', target_source=ROOT)
            self.assertEqual(record['kind'], 'quantized_narrowing')
            self.assertEqual(record['output_node_ids'], ['narrowed'])
            self.assertEqual(record['source_workspace_bytes'], 2*2*3*35)
            self.assertEqual(package.inputs, plan.input_ids)
            self.assertEqual(package.outputs, ('narrowed',))
            self.assertEqual(package.description['kind'], 'quantized_narrowing')
            for name, payload in files.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            shim = r'''#include "probe.h"
#include <string.h>
static atlas_host host;
static probe_context region;
static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES], converted[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned starts,calls; static int fail; static const uint8_t *result;
static uint32_t read32(void *c,uintptr_t a) { (void)c; ++calls; return a==0x40004u?starts:5u; }
static void write32(void *c,uintptr_t a,uint32_t v) { (void)c; ++calls; if(a==0x40018u&&v==1u) ++starts; }
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) { (void)c;(void)a;(void)v;(void)n; ++calls; return 0; }
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) {
  (void)c; ++calls;
  if(a==@STATUS@&&n==4u) { uint32_t status=fail?1u:0u; memcpy(v,&status,4u); return 0; }
  if(a==@OUTPUT@&&n==@BYTES@) { memcpy(v,result,n); return 0; }
  return 1;
}
unsigned call_count(void) { return calls; }
int invoke(const void *a,const void *sx,const void *w,const void *sw,void *out,const uint8_t *res,int reject) {
  starts=calls=0; result=res; fail=reject;
  if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0})) return -1;
  if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u)) return -2;
  if(probe_source_bind(&source,&region,converted,sizeof(converted))) return -3;
  const void *inputs[]={a,sx,w,sw}; void *outputs[]={out};
  return probe_source_dispatch(&source,inputs,outputs);
}
'''
            for key, value in {'STATUS':layout['status_address'], 'OUTPUT':layout['output_addresses'][0],
                               'BYTES':layout['output_bytes']}.items():
                shim = shim.replace('@'+key+'@', str(value)+'u')
            (root/'shim.c').write_text(shim)
            process = subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                'probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'],cwd=root,text=True,capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            lib = ctypes.CDLL(str(root/'probe.so'))
            lib.invoke.argtypes = [ctypes.c_void_p]*6+[ctypes.c_int]
            a = np.full(source.node('a').type.shape, 0x38, np.uint8)
            w = np.full(source.node('w').type.shape, 0x38, np.uint8)
            sx = np.zeros(source.node('sx').type.shape, '<i4')
            sw = np.zeros(source.node('sw').type.shape, '<i4')
            expected = np.resize(np.array([0,0x8000,0x3f80,0xbf80,0x0080,0x7f7f],'<u2'), (2,3,35))
            device = packed_extra(expected, plan)
            output = np.empty_like(expected)
            def invoke(fail=0, target=output):
                output.view(np.uint8).fill(0xa5)
                return lib.invoke(*(v.ctypes.data for v in (a,sx,w,sw,target,device)), fail)
            self.assertEqual(invoke(), 0)
            np.testing.assert_array_equal(output, expected)
            self.assertEqual(invoke(1), 2)
            self.assertTrue(np.all(output.view(np.uint8)==0xa5))
            self.assertEqual(invoke(target=a), 1)
            self.assertEqual(lib.call_count(), 0)
            a.flat[-1] = 0x7f
            self.assertEqual(invoke(), 2)
            self.assertEqual(lib.call_count(), 0)
            self.assertTrue(np.all(output.view(np.uint8)==0xa5))


if __name__ == '__main__':
    unittest.main()
