"""Keep ordered contraction and product semantics across a private result cast."""
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from atlas_mlir_native import quantized_loop, quantized_loop_c
from atlas_mlir_native.already_quantized import prepare_source as prepare_dispatch
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_epilogue import alternatives as epilogues, prepare_source as prepare_epilogue
from atlas_mlir_native.quantized_epilogue_narrowing import alternatives, prepare_source
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, ValueRepresentation
from test_quantized_epilogue import case as epilogue_case

ROOT = Path(__file__).resolve().parents[1]


def case(operand=0, **geometry):
    producer, _, choices = epilogue_case('f32', operand, **geometry)
    product = producer.nodes[-1]
    cast = SemanticNode('narrowed', 'f32_to_bf16', (product.id,), replace(product.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(product.type.shape), 1))
    source = replace(producer, nodes=(*producer.nodes, cast), outputs=(cast.id,))
    last = replace(choices[-1], id='cast', nodes=(2,), inputs=choices[-1].outputs,
        outputs=(BufferRepresentation(cast.id, ValueRepresentation('host_memory', 'dense', 'bf16')),),
        metadata=(('kind', 'exact_float_device'),))
    return source, producer, (*choices, last)


@pytest.mark.parametrize('operand', [0, 1])
def test_preserve_product_order_and_intake(operand):
    source, producer, _ = case(operand)
    old, plan = prepare_epilogue(producer), prepare_dispatch(source)
    assert plan.epilogue_contract_operand == old.epilogue_contract_operand == operand
    assert plan.input_ids == old.input_ids
    assert plan.reset.nodes == old.reset.nodes and plan.continuation.nodes == old.continuation.nodes
    assert plan.epilogue_precision.request.nodes == old.epilogue_precision.request.nodes
    assert plan.source == source and plan.output_id == 'narrowed' and plan.output_dtype == 'bf16'
    assert plan.narrowing_input_id == producer.outputs[0]
    assert 'epilogue product' in plan.record()['narrowing']['admission']


def test_selection_refusals_and_tool_errors_remain_visible():
    source, _, choices = case()
    adapter = SimpleNamespace()
    admitted = SimpleNamespace(record=lambda: dict(route='counted', counted_frame_bytes=65536))
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', return_value=admitted):
        choices = epilogues(adapter, source, choices)
        selected = alternatives(adapter, source, choices)[-1]
        assert selected.nodes == (0, 1, 2)
        assert dict(selected.metadata)['kind'] == 'quantized_epilogue_narrowing'
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted',
               return_value=SimpleNamespace(record=lambda: dict(route='k32_tiles'))):
        assert alternatives(adapter, source, choices) == choices
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted',
               side_effect=NativeCompilationError('compile_error', 'test tool failure')):
        with pytest.raises(NativeCompilationError, match='test tool failure'):
            alternatives(adapter, source, choices)
    for changed in (replace(source, outputs=('product', 'narrowed'), output_storages=('external',)*2),
                    replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], attrs=())))):
        with pytest.raises(ValueError):
            prepare_source(changed)


@pytest.mark.skipif(not all(os.environ.get(k) for k in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'ATLAS_TEST_NATIVE_SNAPSHOT')),
    reason='select Rust model, native tools and snapshot')
@pytest.mark.parametrize('m,n,k,batch,operand', [(3, 35, 65, (2,), 0), (33, 5, 33, (), 1), (1, 1, 1, (), 0)])
def test_generated_c_and_rust_preserve_every_output_and_guards(tmp_path, m, n, k, batch, operand):
    source, _, _ = case(operand, m=m, n=n, k=k, batch=batch)
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        snapshot = open_native_snapshot(Path(os.environ['ATLAS_TEST_NATIVE_SNAPSHOT']))
        plan = prepare_source(source)
        compiled = tmp_path/'compiled'
        quantized_loop.compile_source(snapshot, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
        record, files = quantized_loop_c.prepare_c_package(source, compiled, 'probe', target_source=ROOT)
        assert record['launches_per_invocation'] == 1 and record['output_node_ids'] == ['narrowed']
        layout = quantized_loop.execution_layout(plan)
        for name, data in files.items():
            target = tmp_path/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        size = (max(a+b for a,b in zip(layout['input_addresses'], layout['input_buffer_bytes'])) - DRAM_BASE
                + 8191)//4096*4096
        (tmp_path/'shim.c').write_text('''#include "probe.h"
#include <string.h>
static atlas_host host; static probe_context region; static probe_source_context source;
static unsigned char scratch[PROBE_WORKSPACE_BYTES], conversion[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned char *memory; static unsigned starts, transfers;
static uint32_t rd(void *c,uintptr_t a){(void)c;return a==0x40004u?starts:5u;}
static void wr(void *c,uintptr_t a,uint32_t v){(void)c;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *p,size_t n){
 (void)c;++transfers;if(a<0x90000000u||a-0x90000000u+n>@SIZE@u)return 1;
 memcpy(memory+a-0x90000000u,p,n);return 0;}
static int get(void *c,uintptr_t a,uint8_t *p,size_t n){
 (void)c;++transfers;if(a<0x90000000u||a-0x90000000u+n>@SIZE@u)return 1;
 memcpy(p,memory+a-0x90000000u,n);return 0;}
unsigned transfer_count(void){return transfers;}
int invoke(const void *const *inputs,void *output,void *dram){
 memory=dram;starts=transfers=0;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},
  (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}))return 9;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),20u))return 8;
 if(probe_source_bind(&source,&region,conversion,sizeof(conversion)))return 7;
 void *outputs[]={output};return probe_source_dispatch(&source,inputs,outputs);
}'''.replace('@SIZE@', str(size)))
        subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
            'probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'],cwd=tmp_path,check=True,capture_output=True)
        library = ctypes.CDLL(str(tmp_path/'probe.so'))
        library.invoke.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_void_p]
        rng = np.random.default_rng(1986)
        integers = {key:rng.integers(-1,2,size=source.node(key).type.shape,dtype=np.int32) for key in ('a','w')}
        tensors = {key:np.where(v<0,0xb8,np.where(v>0,0x38,0)).astype(np.uint8) for key,v in integers.items()}
        tensors['sx'] = np.zeros(source.node('sx').type.shape,dtype='<i4')
        tensors['sw'] = (np.arange(n,dtype='<i4')%3-1).reshape(source.node('sw').type.shape)
        tensors['other'] = np.resize(np.array([-2.,-.5,-0.,0.,.5,1.,2.],dtype='<f4'),source.node('other').type.shape)
        arrays = [tensors[key] for key in plan.input_ids]
        frozen = [v.tobytes() for v in arrays]
        pointers = (ctypes.c_void_p*len(arrays))(*(v.ctypes.data for v in arrays))
        memory = np.full(size,0xa5,np.uint8)
        start,end = layout['output_addresses'][0]-DRAM_BASE,layout['status_address']-DRAM_BASE+layout['status_bytes']
        memory[start:end]=0
        storage = np.full(16+2*m*n*np.prod(batch,dtype=int)+16,0xa7,np.uint8)
        output = storage[16:-16].view('<u2').reshape((*batch,m,n))
        assert library.invoke(pointers,output.ctypes.data,memory.ctypes.data)==0
        initial=memory.tobytes()
        (tmp_path/'initial.bin').write_bytes(initial)
        (tmp_path/'config.json').write_text(json.dumps(dict(dram_base=DRAM_BASE,dram_size=size,compute='cpu',pack_layout='register_rows')))
        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(compiled/'program.bin'),
            '--config',str(tmp_path/'config.json'),'--load',f'{DRAM_BASE}:{tmp_path/"initial.bin"}',
            '--dump-state',str(tmp_path/'state')],check=True,capture_output=True)
        actual=b''.join((tmp_path/'state'/f'dram_{a:016x}.bin').read_bytes() for a in range(DRAM_BASE,DRAM_BASE+size,4096))
        assert actual[:start]==initial[:start] and actual[end:]==initial[end:]
        memory[:]=np.frombuffer(actual,np.uint8)
        output.fill(0xa7a7)
        assert library.invoke(pointers,output.ctypes.data,memory.ctypes.data)==0
        expected=np.matmul(integers['a'],integers['w'].swapaxes(-1,-2)).astype('<f4')
        expected*=np.exp2(tensors['sw'].reshape(n)).astype('<f4')
        expected*=tensors['other']
        bits=expected.view('<u4');wanted=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
        np.testing.assert_array_equal(output,wanted)
        assert np.all(storage[:16]==0xa7) and np.all(storage[-16:]==0xa7)
        assert [v.tobytes() for v in arrays]==frozen
        assert library.invoke(pointers,tensors['other'].ctypes.data,memory.ctypes.data)==1
        assert library.transfer_count()==0
        saved_other = tensors['other'].flat[-1]
        tensors['other'].flat[-1]=np.nan;output.fill(0xa7a7)
        assert library.invoke(pointers,output.ctypes.data,memory.ctypes.data)==2
        assert library.transfer_count()==0 and np.all(output==0xa7a7)
        tensors['other'].flat[-1] = saved_other
        # A late device-side exponent rejection must also keep the caller's
        # result untouched, even after earlier private output tiles completed.
        tensors['sw'].flat[-1] = 1000
        assert library.invoke(pointers,output.ctypes.data,memory.ctypes.data)==0
        (tmp_path/'rejected.bin').write_bytes(memory.tobytes())
        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(compiled/'program.bin'),
            '--config',str(tmp_path/'config.json'),'--load',f'{DRAM_BASE}:{tmp_path/"rejected.bin"}',
            '--dump-state',str(tmp_path/'rejected-state')],check=True,capture_output=True)
        rejected=b''.join((tmp_path/'rejected-state'/f'dram_{a:016x}.bin').read_bytes()
                          for a in range(DRAM_BASE,DRAM_BASE+size,4096))
        status = layout['status_address'] - DRAM_BASE
        assert int.from_bytes(rejected[status:status+4], 'little') == 1
        memory[:] = np.frombuffer(rejected, np.uint8)
        output.fill(0xa7a7)
        assert library.invoke(pointers,output.ctypes.data,memory.ctypes.data)==2
        assert np.all(output==0xa7a7)
        assert np.all(storage[:16]==0xa7) and np.all(storage[-16:]==0xa7)
