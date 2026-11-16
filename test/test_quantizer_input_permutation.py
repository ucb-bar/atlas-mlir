"""Folded packing preserves quantizer groups, all output bytes and admission."""
import ctypes as ct
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from atlas_mlir_native import activation_quantize_application as application
from atlas_mlir_native import quantizer_input_permutation as folded
from atlas_mlir_native.vector_quantize import prepare_plan as device_plan
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import IndexMap, SemanticNode, pointwise_identity_maps
from test_activation_quantize_device import dynamic_source, host_reference
from test_already_quantized import quantizer_source

ROOT = Path(__file__).resolve().parents[1]


def fixture(shape, axes, mode):
    logical_shape = tuple(shape[axes.index(i)] for i in range(len(shape)))
    if mode == 'static':
        logical = quantizer_source(count=int(np.prod(shape)), fixed=-8)
        logical = replace(logical, nodes=tuple(replace(n,type=replace(n.type,shape=logical_shape))
            if n.id != logical.nodes[-1].id else n for n in logical.nodes))
    else:
        logical = dynamic_source(shape=logical_shape,reduce=mode,rows=2,rule='ref')
    x = logical.nodes[0]
    physical = replace(x,id='physical',type=replace(x.type,shape=shape))
    rank = len(shape)
    view = SemanticNode(x.id,'transpose',(physical.id,),x.type,
        index_maps=(IndexMap(rank,tuple(tuple(int(j==a) for j in range(rank)) for a in axes),(0,)*rank),
                    pointwise_identity_maps(rank,1)[0]))
    source = replace(logical,nodes=(physical,view,*logical.nodes[1:]),input_storages=((physical.id,'external'),))
    return logical,source


def test_source_refuses_escape_nonpermutation_and_changed_type():
    _,source=fixture((3,5,32),(1,0,2),'head')
    group=folded.prepare_source(source)
    assert group.physical.id=='physical' and group.quantizer.nodes[0].id==group.logical.id
    view=group.logical
    bad=[replace(source,outputs=(*source.outputs,view.id),output_storages=('external',)*3)]
    for changed in (replace(view,attrs=(('unknown',True),)),
                    replace(view,type=replace(view.type,dtype='f32')),
                    replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,0,0)),view.index_maps[1]))):
        bad.append(replace(source,nodes=tuple(changed if n.id==view.id else n for n in source.nodes)))
    for request in bad:
        with pytest.raises(ValueError): folded.prepare_source(request)


def test_selector_requires_private_complete_permutation_and_device_consumer():
    from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
    from test_pointwise_narrowing import adapter
    _,source=fixture((3,5,32),(1,0,2),'head')
    base=adapter()
    assert len(base.plan(source).selected)==2
    wrapper=AdjacentDeviceAdapter(base,patterns=('quantizer_input_permutation',))
    selected,=wrapper.plan(source).selected
    assert dict(selected.metadata)['kind']=='quantizer_input_permutation'
    escaped=replace(source,outputs=(*source.outputs,source.nodes[1].id),output_storages=('external',)*3)
    assert len(wrapper.plan(escaped).selected)==2
    # A second external reader forces the original permutation to remain.
    view=source.nodes[1]
    other=SemanticNode('copy','copy',(view.id,),view.type,index_maps=pointwise_identity_maps(len(view.type.shape),1))
    shared=replace(source,nodes=(*source.nodes,other),outputs=(*source.outputs,other.id),output_storages=('external',)*3)
    assert all(dict(row.metadata)['kind']!='quantizer_input_permutation' for row in wrapper.plan(shared).selected)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('shape,axes,mode,memory',[
    ((33,64),(1,0),'static',8192),
    ((16,10,64),(1,0,2),'head',67108864),
    ((3,5,32),(1,0,2),'token',67108864),
    ((3,5,32),(1,0,2),'head_block',67108864),
    ((32,3,2),(2,0,1),'head',67108864),
])
def test_generated_c_rust_keeps_slices_codes_exponents_and_guards(tmp_path,shape,axes,mode,memory):
    logical,source=fixture(shape,axes,mode)
    with codegen_options(CodegenOptions(io_memory_bytes=memory)):
        group=folded.prepare_source(source)
        compiled=tmp_path/'compiled'
        folded.compile_group(group,None,target_source=ROOT,destination=compiled,limits=None)
        exported=folded.export_group(group,compiled,'quant',target_source=ROOT)
        schedule=application.prepare_plan(group.quantizer)
        plans=[device_plan(body) for body in schedule.bodies]
        programs=[compiled/f'quantizer/segments/body_{i:03d}/program.bin' for i in range(len(plans))]
        assert exported.inputs==('physical',) and exported.outputs==source.outputs
        assert exported.description['launches_per_invocation']==len(schedule.launches)
        assert all(p.stat().st_size<=131072 for p in programs)
        for name,data in exported.files.items():
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        size=(max(p.workspace_bytes for p in plans)+4095)//4096*4096+4096
        (tmp_path/'shim.c').write_text('''#include "quant.h"
#include <string.h>
static atlas_host host; static quant_context region; static quant_source_context source;
static unsigned char scratch[QUANT_WORKSPACE_BYTES],memory[@SIZE@];
static uint32_t image[32768]; static size_t words; static unsigned starts,transfers;
static void (*execute)(const uint32_t *,size_t,unsigned char *,size_t);
static uint32_t rd(void *c,uintptr_t a){(void)c;return a==0x40004u?starts:5u;}
static void wr(void *c,uintptr_t a,uint32_t v){(void)c;
 if(a>=0x20000u && a<0x40000u){if(a==0x20000u)words=0;image[(a-0x20000u)/4]=v;++words;}
 else if(a==0x40018u&&v==1u){execute(image,words,memory,sizeof(memory));++starts;}}
static int put(void *c,uintptr_t a,const uint8_t *p,size_t n){(void)c;++transfers;
 if(a<0x90000000u||a-0x90000000u+n>sizeof(memory))return 1;
 memcpy(memory+a-0x90000000u,p,n);return 0;}
static int get(void *c,uintptr_t a,uint8_t *p,size_t n){(void)c;++transfers;
 if(a<0x90000000u||a-0x90000000u+n>sizeof(memory))return 1;
 memcpy(p,memory+a-0x90000000u,n);return 0;}
unsigned transfer_count(void){return transfers;}
int invoke(const void *input,void *codes,void *exponents,void (*run)(const uint32_t *,size_t,unsigned char *,size_t)){
 starts=transfers=words=0;execute=run;memset(memory,0xa5,sizeof(memory));
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},
  (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}))return 9;
 if(quant_bind(&region,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),20u))return 8;
 if(quant_source_bind(&source,&region,0,0))return 7;
 const void *inputs[]={input};void *outputs[]={codes,exponents};
 return quant_source_dispatch(&source,inputs,outputs);
}
'''.replace('@SIZE@',str(size)))
        subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',
            '-shared','-fPIC','quant.c','runtime/atlas_host.c','shim.c','-o','quant.so'],
            cwd=tmp_path,check=True,capture_output=True)
        library=ct.CDLL(str(tmp_path/'quant.so'))
        callback=ct.CFUNCTYPE(None,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t)
        library.invoke.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,callback]
        calls,errors=[],[]
        expected_raw=b''
        def execute(image,words,memory_pointer,memory_size):
            try:
                ordinal=len(calls);call=schedule.launches[ordinal]
                index=call['body_index'];plan=plans[index]
                assert ct.string_at(image,words*4)==programs[index].read_bytes()
                initial=ct.string_at(memory_pointer,memory_size)
                start=call['input_offset_bytes'];count=call['input_bytes']
                assert initial[:count]==expected_raw[start:start+count]
                assert initial[count:plan.input_bytes]==bytes(plan.input_bytes-count)
                work=tmp_path/f'run-{mode}-{sample}-{ordinal}';work.mkdir()
                (work/'initial.bin').write_bytes(initial)
                (work/'config.json').write_text(json.dumps(dict(dram_base=plan.io_memory_base,
                    dram_size=memory_size,pack_layout='register_rows',compute='cpu')))
                subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(programs[index]),
                    '--config',str(work/'config.json'),'--load',f'{plan.io_memory_base}:{work/"initial.bin"}',
                    '--dump-state',str(work/'state')],check=True,capture_output=True)
                actual=b''.join((work/'state'/f'dram_{a:016x}.bin').read_bytes()
                    for a in range(plan.io_memory_base,plan.io_memory_base+memory_size,4096))
                begin=plan.code_address-plan.io_memory_base
                assert actual[:begin]==initial[:begin] and actual[plan.workspace_bytes:]==initial[plan.workspace_bytes:]
                ct.memmove(memory_pointer,actual,len(actual));calls.append(sha256(actual).hexdigest())
            except Exception as error: errors.append(error)
        hook=callback(execute)
        for sample in range(2):
            values=np.resize(np.array([0,0x8000,1,127,0x3f88,0x3ff8,0x7f7f,0x7f80,0xff80,0x7fc1],'<u2'),shape)
            if sample:
                logical_shape=group.logical.type.shape
                row=np.arange(int(np.prod(logical_shape[:-1]))).reshape(*logical_shape[:-1],1)
                finite=((np.arange(logical_shape[-1])%7)-3)*np.exp2((row%9)-4)
                codes=(finite.astype('<f4').view('<u4')>>16).astype('<u2')
                values=codes.transpose(axes).copy()
            expected_raw=values.transpose(np.argsort(axes)).copy().tobytes()
            expected=host_reference(group.quantizer,expected_raw)
            output=ct.create_string_buffer(bytes([0xa5])*(len(expected[0])+32))
            exponents=ct.create_string_buffer(bytes([0xa5])*(len(expected[1])+32))
            before=values.tobytes();calls.clear();errors.clear()
            assert library.invoke(values.ctypes.data,ct.addressof(output)+16,ct.addressof(exponents)+16,hook)==0
            assert not errors,errors
            assert len(calls)==len(schedule.launches)
            assert output.raw[16:16+len(expected[0])]==expected[0]
            assert exponents.raw[16:16+len(expected[1])]==expected[1]
            assert values.tobytes()==before
            for buffer,length in ((output,len(expected[0])),(exponents,len(expected[1]))):
                assert buffer.raw[:16]==bytes([0xa5])*16 and buffer.raw[16+length:]==bytes([0xa5])*16+b'\0'
            saved=output.raw,exponents.raw
            assert library.invoke(values.ctypes.data,values.ctypes.data,ct.addressof(exponents)+16,hook)==1
            assert library.transfer_count()==0 and (output.raw,exponents.raw)==saved and values.tobytes()==before
        source_file=compiled/'source-request.json'
        source_file.write_text(json.dumps(logical.record()))
        with pytest.raises(ValueError,match='differs from its source'):
            folded.export_group(group,compiled,'quant',target_source=ROOT)
