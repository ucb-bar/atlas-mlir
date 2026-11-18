"""Focused C transport validation; publication authentication alone is mocked."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from math import prod
import ctypes
import json
import os
import unittest
import shutil
import subprocess
import tempfile
from unittest.mock import patch
import numpy as np
from atlas_mlir_native import quantized_loop_c as native
from atlas_mlir_native.already_quantized import restore_bf16
from atlas_mlir_native.quantized_loop import execution_layout
from merlin.semantic_compiler.model import TensorType

from test_already_quantized import source_request

atlas=Path(__file__).resolve().parents[1]
shared=source_request(m=35,n=37,k=65,batch=(2,))
shared=replace(shared,nodes=tuple(replace(node,type=TensorType((2,35,1),'i32','raw-bit-pattern')) if node.id=='sx' else node for node in shared.nodes))
head=source_request(m=3,n=35,k=65,batch=(2,3))
head=replace(head,nodes=tuple(replace(node,type=TensorType((2,3,35,65),'i8','raw-bit-pattern')) if node.id=='w' else
    replace(node,type=TensorType((2,3,1,1),'i32','raw-bit-pattern')) if node.id in ('sx','sw') else
    replace(node,op='atlas.qk_matmul',type=replace(node.type,dtype='f32'),attrs=tuple((key,'float32' if key=='output_dtype' else value) for key,value in node.attrs)) if node.id=='y' else node for node in head.nodes))

def packed(value):
    batches=int(np.prod(value.shape[:-2]));rows,k=value.shape[-2:];mc,kc=(rows+31)//32,(k+31)//32
    data=np.zeros((batches,mc,kc,32,32),np.uint8)
    dense=value.reshape(batches,rows,k)
    for b in range(batches):
        for r in range(rows):
            for c in range(k):data[b,r//32,c//32,r%32,c%32]=dense[b,r,c]
    return data.ravel()



class QuantizedLoopTest(unittest.TestCase):
    def test_layout_and_virtual_unit_batch_contract(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        for source in (shared,head):
            plan=native.prepare_source(source);layout=execution_layout(plan)
            b=int(np.prod(plan.batch_shape));mc,nc,kc=((v+31)//32 for v in (plan.m,plan.n,plan.k))
            rhs=b if source.node('w').type.shape[:-2] else 1
            self.assertEqual(layout['input_buffer_bytes'],[b*mc*kc*1024,rhs*nc*kc*1024])
            self.assertEqual(layout['output_bytes'],b*mc*nc*2048)
            virtual=virtual_source(plan,layout)
            self.assertIn('!atlas.virtual_mxu_acc<1>',virtual)
            self.assertIn('!atlas.virtual_mxu_weight<1>',virtual)
            self.assertNotIn('!atlas.virtual_mxu_acc<0>',virtual)
            self.assertEqual(virtual.count('\"atlas.virtual_start\"'),1)
            self.assertEqual(virtual.count('atlas.exponent_scratch = \"status\"'),1)
            self.assertIn('atlas.virtual_mxu_load_acc_bf16',virtual)

    def test_actual_lowering_verifies_batched_mxu1_and_shared_rhs(self):
        from atlas_mlir_native.binding import NATIVE_PIPELINES, NATIVE_CHECKS, RTL_REVISION
        from atlas_mlir_native.quantized_loop import virtual_source
        tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR', atlas/'build/bin'))
        if not (tools/'atlas-opt').is_file() or not (tools/'atlas-emit').is_file():
            self.skipTest('Atlas compiler binaries unavailable')
        for source in (shared,head):
            with self.subTest(operation=source.node('y').op):
                plan=native.prepare_source(source)
                if source is shared:
                    # Odd M/N tails, shared RHS and both row-exponent pages
                    # exercise scalar data guards alongside counted DMA.
                    self.assertEqual((plan.m,plan.n,plan.k,plan.batch_shape),(35,37,65,(2,)))
                    self.assertEqual(source.node('sx').type.shape,(2,35,1))
                    self.assertEqual(source.node('sw').type.shape,(37,1))
                virtual=virtual_source(plan,execution_layout(plan))
                lowered=subprocess.run([str(tools/'atlas-opt'),*NATIVE_PIPELINES['mandatory'],'-'],input=virtual,
                    capture_output=True,text=True,timeout=30)
                self.assertEqual(lowered.returncode,0,lowered.stderr)
                verified=subprocess.run([str(tools/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-'],
                    input=lowered.stdout,capture_output=True,text=True,timeout=30)
                self.assertEqual(verified.returncode,0,verified.stderr)
                emitted=subprocess.run([str(tools/'atlas-emit'),'--program-json','-'],
                    input=lowered.stdout,capture_output=True,text=True,timeout=30)
                self.assertEqual(emitted.returncode,0,emitted.stderr)
                program=json.loads(emitted.stdout)
                self.assertEqual(program['selected_rtl_revision'],RTL_REVISION)
                self.assertTrue(program['instructions'])

    @unittest.skipUnless(shutil.which('cc'),'C compiler unavailable')
    def test_generated_c_transports_exponents_and_device_status_atomically(self):
        for source in (shared,head):
            with tempfile.TemporaryDirectory(prefix='atlas-loop-c-') as temp:
                dest=Path(temp);compiled=dest/'compiled';compiled.mkdir()
                plan=native.prepare_source(source);execution=execution_layout(plan)
                records={'already-quantized-plan.json':plan.record(),'source-request.json':source.record(),
                    'manifest.json':{'target_inventory_sha256':sha256((atlas/'docs/selected-variant-inventory.json').read_bytes()).hexdigest()},
                    'execution_plan.json':execution,'physical-program.json':{'instructions':[{'word_u32':0x00100073}]}}
                for key,value in records.items():(compiled/key).write_text(json.dumps(value))
                (compiled/'program.bin').write_bytes((0x00100073).to_bytes(4,'little'))
                with patch.object(native,'verify_native_publication'), \
                     patch.object(native,'selected_route',return_value='counted'):
                    record,files=native.prepare_c_package(source,compiled,'probe',target_source=atlas)
                    wrong={**execution,'status_address':execution['status_address']+2048}
                    (compiled/'execution_plan.json').write_text(json.dumps(wrong))
                    with self.assertRaises(ValueError):
                        native.prepare_c_package(source,compiled,'probe',target_source=atlas)
                for key,value in files.items():
                    path=dest/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
                inputs=execution['input_addresses']+execution['exponent_addresses']
                sizes=execution['input_buffer_bytes']+execution['exponent_buffer_bytes']
                text='''#include "probe.h"
#include <string.h>
static atlas_host host;static probe_context region;static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned calls,starts,puts,gets;static const uint8_t *expected[4],*result;static int mode;
static uint32_t read32(void *c,uintptr_t a){(void)c;++calls;return a==0x40004u?starts:5u;}
static void write32(void *c,uintptr_t a,uint32_t v){(void)c;++calls;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n){
 (void)c;++calls;++puts;const uintptr_t addresses[]={@ADDRESSES@};const size_t sizes[]={@SIZES@};
 for(unsigned i=0;i<4u;++i)if(a==addresses[i]&&n==sizes[i])return memcmp(v,expected[i],n)!=0;return 1;
}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n){
 (void)c;++calls;++gets;if(mode==2)return 1;
 if(a==@STATUS@&&n==4u){uint32_t status=mode==1?1u:0u;memcpy(v,&status,4u);return 0;}
 if(a==@OUT@&&n==@OUTBYTES@){memcpy(v,result,n);return 0;}return 1;
}
unsigned callback_count(void){return calls;}unsigned start_count(void){return starts;}
unsigned write_count(void){return puts;}unsigned read_count(void){return gets;}
int invoke(const void *a,const void *sx,const void *w,const void *sw,void *out,
 const uint8_t *pa,const uint8_t *pw,const uint8_t *pea,const uint8_t *pew,const uint8_t *res,int fail){
 calls=starts=puts=gets=0;expected[0]=pa;expected[1]=pw;expected[2]=pea;expected[3]=pew;result=res;mode=fail;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0}))return -1;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u))return -2;
 if(probe_source_bind(&source,&region,converted,sizeof(converted)))return -3;
 const void *inputs[]={a,sx,w,sw};void *outputs[]={out};return probe_source_dispatch(&source,inputs,outputs);
}
'''
                for key,value in {'ADDRESSES':','.join(str(v)+'u' for v in inputs),'SIZES':','.join(str(v)+'u' for v in sizes),
                    'STATUS':str(execution['status_address'])+'u','OUT':str(execution['output_addresses'][0])+'u',
                    'OUTBYTES':str(execution['output_bytes'])+'u'}.items():text=text.replace('@'+key+'@',value)
                (dest/'shim.c').write_text(text)
                proc=subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-shared','-fPIC',
                    'probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'],cwd=dest,capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
                lib=ctypes.CDLL(str(dest/'probe.so'));lib.invoke.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int]
                a=np.resize(np.array([0,8,0x38,0x77,0x78,0x7e,0x88,0xfe],np.uint8),source.node('a').type.shape)
                w=np.resize(np.array([0x38,0xfe,0x88,0,0x7e,0x08],np.uint8),source.node('w').type.shape)
                # Arbitrary i32 bytes cross this boundary unchanged. Admission is
                # represented by the device's explicit completion status.
                sx=np.arange(prod(source.node('sx').type.shape),dtype=np.int32).reshape(source.node('sx').type.shape)-128
                sw=np.arange(prod(source.node('sw').type.shape),dtype=np.int32).reshape(source.node('sw').type.shape)+127
                def pack_exponents(value,geometry):
                    out=np.zeros((geometry['pages'],512),np.int32)
                    if geometry['mode']=='row':
                        dense=value.reshape(geometry['batches'],geometry['rows'])
                        for b in range(geometry['batches']):
                            for r in range(geometry['rows']):out[b*geometry['tiles']+r//32,r%32]=dense[b,r]
                    else:out[:,0]=value.reshape(-1)
                    return out
                pa,pw=packed(a),packed(w)
                pea,pew=(pack_exponents(v,g) for v,g in zip((sx,sw),execution['exponent_layouts'],strict=True))
                device_words=(0x3f80+np.arange(execution['output_bytes']//2,dtype=np.uint16)%64).astype('<u2')
                batch=prod(plan.batch_shape);m,n=plan.m,plan.n;mc,nc=(m+31)//32,(n+31)//32
                expected=np.empty((batch,m,n),np.uint16)
                for b in range(batch):
                    for r in range(m):
                        for c in range(n):expected[b,r,c]=device_words[((b*mc+r//32)*nc+c//32)*1024+(c%32//16)*512+(r%32)*16+c%16]
                if plan.output_dtype=='f32':expected=(expected.astype(np.uint32)<<16).view(np.float32)
                output=np.empty_like(expected)
                def invoke(mode=0,target=output):
                    output.view(np.uint8).fill(0xa5)
                    return lib.invoke(*(v.ctypes.data for v in (a,sx,w,sw,target,pa,pw,pea,pew,device_words)),mode)
                frozen=[v.tobytes() for v in (a,w,sx,sw)]
                self.assertEqual(invoke(),0);np.testing.assert_array_equal(output,expected)
                self.assertEqual((lib.start_count(),lib.write_count(),lib.read_count()),(1,4,2))
                self.assertEqual(invoke(1),2);self.assertTrue(np.all(output.view(np.uint8)==0xa5))
                self.assertEqual((lib.start_count(),lib.write_count(),lib.read_count()),(1,4,1))
                self.assertEqual(invoke(2),3);self.assertTrue(np.all(output.view(np.uint8)==0xa5))
                self.assertEqual(invoke(target=a),1);self.assertEqual(lib.callback_count(),0)
                self.assertEqual(frozen,[v.tobytes() for v in (a,w,sx,sw)])
                original=int(a.flat[-1]);a.flat[-1]=0x7f
                self.assertEqual(invoke(),2);self.assertEqual(lib.callback_count(),0)
                self.assertTrue(np.all(output.view(np.uint8)==0xa5));a.flat[-1]=original


if __name__ == "__main__":
    unittest.main()
