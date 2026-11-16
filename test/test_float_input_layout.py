"""Packed affine inputs retain exact arithmetic, guards and atomic publication."""
import ctypes as ct
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native import float_input_layout as folded
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import IndexMap, SemanticNode, pointwise_identity_maps
from test_exact_float_device import request, host_reference

ROOT = Path(__file__).resolve().parents[1]


def fixture(op, layout, port=0):
    shape=(33,35)
    original=request(op,int(np.prod(shape)))
    logical=replace(original,nodes=tuple(replace(n,type=replace(n.type,shape=shape),
        index_maps=() if n.effect=='input' else pointwise_identity_maps(2,len(n.inputs))) for n in original.nodes))
    operand=logical.nodes[port]
    if layout=='broadcast':
        physical_shape,coefficients,offsets=(35,),((0,1),),(0,)
    elif layout=='slice':
        physical_shape,coefficients,offsets=(33,70),((1,0),(0,1)),(0,35)
    elif layout=='reverse':
        physical_shape,coefficients,offsets=(33,70),((1,0),(0,-2)),(0,69)
    elif layout=='transpose':
        physical_shape,coefficients,offsets=(35,33),((0,1),(1,0)),(0,0)
    else: raise ValueError(layout)
    physical=replace(operand,id='physical',type=replace(operand.type,shape=physical_shape))
    view=SemanticNode(operand.id,'gather',(physical.id,),operand.type,
        index_maps=(IndexMap(2,coefficients,offsets),pointwise_identity_maps(2,1)[0]))
    nodes=tuple(item for n in logical.nodes for item in ((physical,view) if n.id==operand.id else (n,)))
    source=replace(logical,nodes=nodes,input_storages=tuple((n.id,'external') for n in nodes if n.effect=='input'))
    return logical,source


def test_private_selection_and_refused_source_boundaries():
    from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
    from atlas_mlir_native.already_quantized import POLICY
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from test_application_float_arithmetic import policy
    logical,source=fixture('bf16_add','broadcast',1)
    selected_policy={**policy(), 'already_quantized': POLICY,
                     'host_support': ['transpose','reshape','copy','gather'],
                     'host_float_arithmetic': ['bf16_add','bf16_sub','bf16_neg','f32_neg']}
    base=AtlasApplicationAdapter(selected_policy,original_source_sha256=selected_policy['source_sha256'])
    wrapper=AdjacentDeviceAdapter(base,patterns=('float_input_layout',))
    selected,=wrapper.plan(source).selected
    assert dict(selected.metadata)['kind']=='float_input_layout'
    view=next(n for n in source.nodes if n.op=='gather')
    escaped=replace(source,outputs=(*source.outputs,view.id),output_storages=('external',)*2)
    assert len(wrapper.plan(escaped).selected)==2
    other=SemanticNode('copy','copy',(view.id,),view.type,index_maps=pointwise_identity_maps(2,1))
    shared=replace(source,nodes=(*source.nodes,other),outputs=(*source.outputs,other.id),output_storages=('external',)*2)
    assert all(dict(row.metadata)['kind']!='float_input_layout' for row in wrapper.plan(shared).selected)
    changes=[escaped]
    for changed in (replace(view,attrs=(('unknown',True),)),
        replace(view,type=replace(view.type,dtype='f32')),
        replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,)),view.index_maps[1]))):
        changes.append(replace(source,nodes=tuple(changed if n.id==view.id else n for n in source.nodes)))
    for changed in changes:
        with pytest.raises(ValueError): folded.prepare_source(changed)
    # A source port used both through the view and directly must not silently
    # change alias/dedup semantics when the view port is replaced.
    _,same=fixture('bf16_add','transpose')
    physical=replace(same.nodes[0],type=replace(same.nodes[0].type,shape=(33,35)))
    identity=replace(same.nodes[1],index_maps=pointwise_identity_maps(2,1))
    out=replace(same.nodes[-1],inputs=(identity.id,physical.id))
    same=replace(same,nodes=(physical,identity,out),input_storages=((physical.id,'external'),))
    with pytest.raises(ValueError): folded.prepare_source(same)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR')),
    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('op,layout,port',[
    ('bf16_add','broadcast',0),('bf16_sub','broadcast',1),
    ('bf16_to_f32','slice',0),('bf16_neg','slice',0),
    ('f32_to_bf16','reverse',0),('f32_neg','transpose',0),
    ('bf16_to_f32','broadcast',0),('bf16_add','transpose',1),
])
def test_c_rust_affine_slices_domain_and_atomic_output(tmp_path,op,layout,port):
    logical,source=fixture(op,layout,port)
    with codegen_options(CodegenOptions(io_memory_bytes=8192,exact_float_unroll=True)):
        group=folded.prepare_source(source)
        schedule=application.prepare_plan(group.arithmetic)
        assert len(schedule.launches)>1
        compiled=tmp_path/'compiled'
        folded.compile_group(group,None,target_source=ROOT,destination=compiled,limits=None)
        exported=folded.export_group(group,compiled,'packed',target_source=ROOT)
        plans=[device.prepare_plan(body) for body in schedule.bodies]
        programs=[compiled/f'float/segments/body_{i:03d}/program.bin' for i in range(len(plans))]
        assert exported.inputs[port]=='physical'
        assert all(p.stat().st_size<=131072 for p in programs)
        for name,data in exported.files.items():
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (tmp_path/'shim.c').write_text('''#include "packed.h"
#include <string.h>
static atlas_host host; static packed_context region; static packed_source_context source;
static unsigned char scratch[PACKED_WORKSPACE_BYTES],memory[12288];
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
int invoke(const void *const *inputs,void *output,void (*run)(const uint32_t *,size_t,unsigned char *,size_t)){
 starts=transfers=words=0;execute=run;memset(memory,0xa5,sizeof(memory));
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},
  (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}))return 9;
 if(packed_bind(&region,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),20u))return 8;
 if(packed_source_bind(&source,&region,0,0))return 7;
 void *outputs[]={output};return packed_source_dispatch(&source,inputs,outputs);
}
''')
        subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',
            '-shared','-fPIC','packed.c','runtime/atlas_host.c','shim.c','-o','packed.so'],
            cwd=tmp_path,check=True,capture_output=True)
        library=ct.CDLL(str(tmp_path/'packed.so'))
        callback=ct.CFUNCTYPE(None,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t)
        library.invoke.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,callback]
        calls,errors=[],[]
        expected_raw=[]
        trial=0
        def execute(image,words,memory_pointer,memory_size):
            try:
                ordinal=len(calls);call=schedule.launches[ordinal]
                index=call['body_index'];plan=plans[index]
                assert ct.string_at(image,words*4)==programs[index].read_bytes()
                initial=ct.string_at(memory_pointer,memory_size)
                start,count=call['input_offset_bytes'],call['input_bytes']
                for j,raw in enumerate(expected_raw):
                    at=j*plan.input_bytes
                    assert initial[at:at+count]==raw[start:start+count]
                    assert initial[at+count:at+plan.input_bytes]==bytes(plan.input_bytes-count)
                work=tmp_path/f'run-{trial}-{ordinal}';work.mkdir()
                (work/'initial.bin').write_bytes(initial)
                (work/'config.json').write_text(json.dumps(dict(dram_base=plan.io_memory_base,
                    dram_size=memory_size,pack_layout='register_rows',compute='cpu')))
                subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(programs[index]),
                    '--config',str(work/'config.json'),'--load',f'{plan.io_memory_base}:{work/"initial.bin"}',
                    '--dump-state',str(work/'state')],check=True,capture_output=True)
                actual=b''.join((work/'state'/f'dram_{a:016x}.bin').read_bytes()
                    for a in range(plan.io_memory_base,plan.io_memory_base+memory_size,4096))
                output_at=plan.output_address-plan.io_memory_base
                assert actual[:output_at]==initial[:output_at]
                assert actual[plan.workspace_bytes:]==initial[plan.workspace_bytes:]
                ct.memmove(memory_pointer,actual,len(actual));calls.append(ordinal)
            except Exception as error: errors.append(error)
        hook=callback(execute)
        width=schedule.plan.width
        codes=[0,0x8000,1,127,128,0x3f80,0x3f81,0x7f7f,0x7f80,0xff80]
        if op.endswith('_neg'):codes.extend([0x7fc1,0xffff])
        if width==4:
            codes=[word<<16 for word in codes]+[0x3f888000,0x3f898000,0x3f888001,1]
        values=np.resize(np.array(codes,dtype=f'<u{width}'),group.physical.type.shape)
        if layout=='slice': values[:,:35]=0x7fc1 # unread NaNs must not reject the logical slice
        read=group.logical.index_maps[0]
        indices=[tuple(offset+sum(c*d for c,d in zip(row,index,strict=True))
            for row,offset in zip(read.coefficients,read.offsets,strict=True)) for index in np.ndindex(group.logical.type.shape)]
        for invalid in (False,True) if not op.endswith('_neg') else (False,):
            trial+=1
            if invalid: values[indices[-1]]=0x7fc1 if width==2 else 0x7f800001
            logical_raw=b''.join(int(values[index]).to_bytes(width,'little') for index in indices)
            expected_raw=[bytes(schedule.plan.count*width) for _ in schedule.plan.operands]
            expected_raw[port]=logical_raw
            expected,status=host_reference(group.arithmetic,expected_raw)
            buffers=[ct.create_string_buffer(raw) for raw in expected_raw]
            buffers[port]=ct.create_string_buffer(values.tobytes())
            pointers=(ct.c_void_p*len(buffers))(*(ct.addressof(b) for b in buffers))
            output=ct.create_string_buffer(bytes([0xa5])*(len(expected)+32))
            before=[b.raw for b in buffers];calls.clear();errors.clear()
            actual_status=library.invoke(pointers,ct.addressof(output)+16,hook)
            assert not errors,errors
            assert bool(actual_status)==bool(status)
            if invalid:
                assert actual_status==2 and output.raw==bytes([0xa5])*(len(expected)+32)+b'\0'
                if layout!='broadcast':
                    assert len(calls)==len(schedule.launches) # the error occurs in the final slice
            else:
                assert actual_status==0 and len(calls)==len(schedule.launches)
                assert output.raw[16:16+len(expected)]==expected
            assert [b.raw for b in buffers]==before
            assert output.raw[:16]==bytes([0xa5])*16 and output.raw[16+len(expected):]==bytes([0xa5])*16+b'\0'
            saved=output.raw
            assert library.invoke(pointers,ct.addressof(buffers[port]),hook)==1
            assert library.transfer_count()==0 and output.raw==saved and [b.raw for b in buffers]==before
        (compiled/'source-request.json').write_text(json.dumps(logical.record()))
        with pytest.raises(ValueError,match='differs from its source'):
            folded.export_group(group,compiled,'packed',target_source=ROOT)
