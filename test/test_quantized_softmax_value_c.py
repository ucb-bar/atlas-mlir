"""Exercise generated source transport with fake MMIO, including atomic failure.

The fake device checks every staged operand and supplies recognizable output
codes. It does not interpret the opaque image or qualify device arithmetic.
"""
from dataclasses import replace
import ctypes
import hashlib
import json
from math import prod
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.binding import DRAM_BASE, TARGET_IDENTITY, TILE_BYTES
from atlas_mlir_native.layout import pack_bf16_pair_halves
from atlas_mlir_native.quantized_softmax_value import prepare_source
from atlas_mlir_native.quantized_softmax_value_c import _checked_execution, _render_package, value_spec
from atlas_mlir_native.resident_graph import mixed_memory_plan
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]


def source(rows, keys, channels, batches=1, rank4=False):
    shape = (1,batches,rows,keys) if rank4 else (batches,rows,keys)
    rank = len(shape)
    tensor = lambda s: TensorType(s,'f32','raw-bit-pattern')
    full, row, column = tensor(shape), tensor(shape[:-1]), tensor((*shape[:-1],1))
    identity = pointwise_identity_maps(rank,1)[0]
    prefix = IndexMap(rank,identity.coefficients[:-1],(0,)*(rank-1))
    reduce_maps = (identity,prefix,prefix)
    broadcast = (identity,IndexMap(rank,(*identity.coefficients[:-1],(0,)*rank),(0,)*rank),identity)
    mask_shape = (1,1,rows,keys) if rank4 else (1,rows,keys)
    mask_coeff = tuple((0,)*rank if i<rank-2 else identity.coefficients[i] for i in range(rank))
    mask_map = IndexMap(rank,mask_coeff,(0,)*rank)
    nodes = [SemanticNode('scores','input',(),full,effect='input'),
        SemanticNode('mask','input',(),tensor(mask_shape),effect='input'),
        SemanticNode('masked','f32_add',('scores','mask'),full,(("numeric_semantics","ieee754-f32-add-rne-v1"),),index_maps=(identity,mask_map,identity)),
        SemanticNode('min_seed','constant',(),row,effect='constant')]
    nodes += [SemanticNode('maximum','source_f32_reduce_max_unqualified',('masked','min_seed'),row,
        (('axis',rank-1),('reduction_order','unresolved'),('numeric_semantics','source-f32-maximum-unqualified-v1'),('scalar_operand_order','input-accumulator')),index_maps=reduce_maps),
        SemanticNode('max_view','reshape',('maximum',),column),
        SemanticNode('shifted','f32_sub',('masked','max_view'),full,(('numeric_semantics','ieee754-f32-sub-rne-v1'),),index_maps=broadcast),
        SemanticNode('exp','source_f32_exp_unqualified',('shifted',),full,(('numeric_semantics','source-f32-exp-unqualified-v1'),),index_maps=pointwise_identity_maps(rank,1)),
        SemanticNode('zero_seed','constant',(),row,effect='constant'),
        SemanticNode('total','source_f32_reduce_add_unqualified',('exp','zero_seed'),row,
            (('axis',rank-1),('reduction_order','unresolved'),('numeric_semantics','source-f32-add-unqualified-v1'),('scalar_add_order','input-accumulator')),index_maps=reduce_maps),
        SemanticNode('sum_view','reshape',('total',),column),
        SemanticNode('probability','f32_div',('exp','sum_view'),full,(('numeric_semantics','ieee754-f32-div-rne-v1'),),index_maps=broadcast)]
    probability = 'probability'
    if rank4:
        nodes.append(SemanticNode('flat_p','reshape',('probability',),tensor((batches,rows,keys))))
        probability = 'flat_p'
    out_type = tensor((batches,rows,channels))
    nodes += [SemanticNode('values','input',(),tensor((batches,keys,channels)),effect='input'),
              SemanticNode('pv_seed','constant',(),out_type,effect='constant')]
    axes = tuple(tuple(int(i==j) for j in range(4)) for i in range(4))
    maps = tuple(IndexMap(4,(axes[0],*pair),(0,)*3) for pair in ((axes[1],axes[3]),(axes[3],axes[2]),(axes[1],axes[2])))
    nodes.append(SemanticNode('context','source_f32_matmul_unqualified',(probability,'values','pv_seed'),out_type,
        (('reduction_order','unresolved'),('numeric_semantics','source-f32-mul-add-unqualified-v1')),index_maps=(*maps,maps[-1])))
    return KernelRequest(tuple(nodes),('context',),('external',),(('scores','external'),('values','external'),('mask','external')),
        TARGET_IDENTITY,'diagnostic',constants=(ConstantBinding('min_seed','f32-le-raw','000080ff'*prod(shape[:-1])),
        ConstantBinding('zero_seed','f32-le-raw','00000000'*prod(shape[:-1])),ConstantBinding('pv_seed','f32-le-raw','00000000'*prod(out_type.shape))))


class QuantizedSoftmaxValueCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temporary.name)
        cls.packages = {}
        for rows,keys,channels,batches,rank4 in ((1,33,32,1,False),(33,35,35,2,True)):
            captured = source(rows,keys,channels,batches,rank4)
            plan = prepare_source(captured,value_scale_exponent=-10)
            _,slots,sizes,_ = mixed_memory_plan(plan.body)
            execution = {role:[dict(node=key,address=DRAM_BASE+slots[key]*TILE_BYTES,bytes=sizes[key]) for key in keys_] for role,keys_ in (
                ('inputs',[n.id for n in plan.body.nodes if n.effect in {'input','constant'}]),('outputs',list(plan.body.outputs)))}
            # An uninterpreted fake image exercises only the normal host loading ABI.
            record,files = _render_package(plan,execution,b'fake',slots,'qsv',ROOT,'separate')
            directory = cls.work/f'{rows}-{keys}-{channels}'
            directory.mkdir()
            for filename,data in files.items():
                path=directory/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            (directory/'client.c').write_text('''#include "qsv.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const float *const *inputs,float *output,int alias) {
  atlas_host host; qsv_context region; qsv_source_context source;
  unsigned char raw[QSV_WORKSPACE_BYTES+32u],conversion[QSV_SOURCE_WORKSPACE_BYTES+32u];
  __builtin_memset(raw,0xa7,sizeof(raw));__builtin_memset(conversion,0xa7,sizeof(conversion));
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
      qsv_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw+16u,QSV_WORKSPACE_BYTES,8u)) return -1;
  if (alias==1) return qsv_source_bind((qsv_source_context *)&region,&region,conversion+16u,QSV_SOURCE_WORKSPACE_BYTES);
  if (qsv_source_bind(&source,&region,conversion+16u,QSV_SOURCE_WORKSPACE_BYTES)) return -2;
  int result=qsv_source_run(&source,inputs,output,alias==2?(atlas_host_result *)output:0);
  for (size_t i=0;i<16u;++i) if (raw[i]!=0xa7 || raw[QSV_WORKSPACE_BYTES+16u+i]!=0xa7 ||
      conversion[i]!=0xa7 || conversion[QSV_SOURCE_WORKSPACE_BYTES+16u+i]!=0xa7) return -3;
  return result;
}''')
            built = subprocess.run([shutil.which('cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                'qsv.c','runtime/atlas_host.c','client.c','-o','client.so'],cwd=directory,capture_output=True)
            if built.returncode:
                raise RuntimeError(built.stderr.decode())
            cls.packages[rows]=(plan,execution,record,directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_tail_rows_channels_affine_broadcast_and_three_masks(self):
        for rows in (1,33):
            self.run_case(rows)

    def test_late_invalid_values_and_fault_leave_output_unchanged(self):
        for fault in ('score','value','mask','negative_zero_mask','device','read'):
            with self.subTest(fault=fault):
                self.run_case(33,fault=fault)

    def test_context_device_and_input_output_aliases_refuse(self):
        for alias in (1,2,3):
            self.run_case(33,alias=alias)

    def test_codec_and_readonly_mutable_slot_coverage(self):
        self.assertEqual(value_spec(-10).encode(np.array([0,-0.,2**-10,-2**-10,448*2**-10,1e20],dtype=np.float32)).tolist(),[0,0,0x38,0xb8,0x7e,0x7e])
        for plan,execution,record,_ in self.packages.values():
            _,slots,_,_=mixed_memory_plan(plan.body)
            ranges=record['device_buffers']
            self.assertEqual(ranges[0]['access'],'READ')
            self.assertEqual(ranges[1]['access'],'READ_WRITE')
            self.assertEqual(ranges[0]['address']+ranges[0]['bytes'],ranges[1]['address'])
            self.assertEqual(ranges[1]['address']+ranges[1]['bytes'],DRAM_BASE+(max(slots.values())+1)*TILE_BYTES)
            self.assertEqual(record['input_node_ids'],['scores','values','mask'])

    def test_source_plan_and_selected_provenance_are_hash_bound(self):
        # Mock only native publication admission: test adversarial source/plan
        # binding independently of native code generation and device execution.
        captured = source(1,33,32)
        plan = prepare_source(captured,value_scale_exponent=-10,provenance_sha256='a'*64)
        _,slots,sizes,liveness = mixed_memory_plan(plan.body)
        row = lambda key: dict(node=key,address=DRAM_BASE+slots[key]*TILE_BYTES,bytes=sizes[key])
        execution = dict(schema='atlas.native_tile_execution_plan.v8',launches_per_invocation=1,
            inputs=[dict(**row(n.id),type=n.type.record()) for n in plan.body.nodes if n.effect in {'input','constant'}],
            outputs=[dict(**row(k),type=plan.body.node(k).type.record()) for k in plan.body.outputs],
            buffers=[row(k) for k in slots],source_liveness_buffers=liveness,backend_verification_tier='verified')
        directory=self.work/'publication-binding';directory.mkdir()
        def write(name,data):
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            return hashlib.sha256(data).hexdigest()
        constants=[]
        for i,binding in enumerate(plan.body.constants):
            raw=bytes.fromhex(binding.data_hex);bf16=binding.encoding=='bf16-le-raw'
            data=pack_bf16_pair_halves(raw) if bf16 else raw
            name=f'constants/constant-{i:03d}.bin'
            constants.append(dict(node_id=binding.node_id,address=row(binding.node_id)['address'],bytes=len(data),
                encoding=binding.encoding,physical_layout='bf16_pair_halves' if bf16 else 'fp8_row_major',path=name,sha256=write(name,data)))
        execution['constant_inputs']=constants
        encoded=lambda value:json.dumps(value).encode()
        write('source-request.json',encoded(plan.body.record()))
        write('execution_plan.json',encoded(execution))
        semantic_hash=write('semantic-source.json',encoded(captured.record()))
        plan_hash=write('quantized-softmax-value-plan.json',encoded(plan.record()))
        manifest=dict(binary_sha256=write('program.bin',b'fake'),constant_inputs=constants,
            target_inventory_sha256=hashlib.sha256((ROOT/'docs/selected-variant-inventory.json').read_bytes()).hexdigest(),
            source_softmax_value_request_digest=captured.digest(),source_softmax_value_plan_sha256=plan_hash,
            stage_artifacts={'semantic-source.json':semantic_hash,'quantized-softmax-value-plan.json':plan_hash})
        write('manifest.json',encoded(manifest))
        module='atlas_mlir_native.quantized_softmax_value_c'
        with patch(module+'.verify_native_publication') as admission,patch(module+'.check_io_memory_manifest'):
            _checked_execution(plan,directory,ROOT)
            self.assertEqual(admission.call_args.kwargs['request_digest'],plan.body.digest())
            wrong=replace(plan,provenance_sha256='b'*64)
            self.assertEqual(wrong.body.digest(),plan.body.digest())
            with self.assertRaisesRegex(ValueError,'provenance binding'):
                _checked_execution(wrong,directory,ROOT)
            changed=replace(plan,provenance_sha256=None)
            with self.assertRaisesRegex(ValueError,'provenance binding'):
                _checked_execution(changed,directory,ROOT)
            execution['buffers'][-1]['bytes']+=1024
            write('execution_plan.json',encoded(execution))
            with self.assertRaisesRegex(ValueError,'buffers or verified tier'):
                _checked_execution(plan,directory,ROOT)

    def run_case(self,rows,*,fault=None,alias=0):
        plan,execution,record,directory=self.packages[rows]
        region=plan.region;batch=prod(region.batch_shape);keys,channels=region.keys,region.channels
        scores=np.resize(np.array([-40.,-2.,0.,1.,40.],dtype=np.float32),batch*rows*keys).copy()
        palette=np.array([0.,-0.,2**-10,-2**-10,448*2**-10,1000.],dtype=np.float32)
        values=np.resize(palette,batch*keys*channels).copy()
        mask=np.zeros((rows,keys),dtype=np.float32);mask[:,1::3]=-np.inf;mask[:,2::3]=np.finfo(np.float32).min
        if rows>1: mask[1,:]=-np.inf
        if fault=='score':scores[-1]=np.nan
        if fault=='value':values[-1]=np.inf
        if fault=='mask':mask[-1,-1]=1.
        if fault=='negative_zero_mask':mask[-1,-1]=-0.
        arrays=[scores,values,mask.reshape(-1)]
        before=[a.tobytes() for a in arrays]
        rd=ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_size_t)
        wr=ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32)
        mem=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t)
        memory=bytearray([0xa5])*(record['device_buffers'][-1]['address']-DRAM_BASE+record['device_buffers'][-1]['bytes']+2048)
        words,errors,calls={},[],[]
        state={'polls':None,'fault':False,'reads':0}
        addresses={r['node']:r['address']-DRAM_BASE for r in execution['inputs']+execution['outputs']}
        def read(_,address):
            if address==0x40004:return len(words)
            if address!=0x40008:errors.append('unexpected CSR read');return 3
            if state['polls'] is None:return 5
            state['polls']+=1
            return 0 if state['polls']==1 else 3 if state['fault'] else 5
        def write(_,address,value):
            if 0x20000<=address<0x40000:words[(address-0x20000)//4]=value;return
            try:
                self.assertEqual((address,value),(0x40018,1))
                group=len(calls);calls.append(True);state['fault']=fault=='device' and group==1;state['polls']=0
                if state['fault']:return
                b,first=divmod(group,(rows+31)//32);first*=32
                for kt in range((keys+31)//32):
                    for keep in (False,True):
                        logical=np.full((32,32),0 if keep else 0xc220,dtype='<u2')
                        for r in range(min(32,rows-first)):
                            for k in range(min(32,keys-kt*32)):
                                key=kt*32+k
                                if mask[first+r,key]==0:
                                    logical[r,k]=0x3f80 if keep else struct.unpack('<I',struct.pack('<f',float(scores[(b*rows+first+r)*keys+key])))[0]>>16
                        key=f'keep_{kt}' if keep else f'scores_{kt}';at=addresses[key]
                        self.assertEqual(bytes(memory[at:at+2048]),pack_bf16_pair_halves(logical.tobytes()))
                encoded=value_spec(-10).encode(values).reshape(batch,keys,channels)
                for nt in range((channels+31)//32):
                    for kt in range((keys+31)//32):
                        tile=np.zeros((32,32),dtype=np.uint8)
                        for d in range(min(32,channels-nt*32)):
                            for k in range(min(32,keys-kt*32)):tile[d,k]=encoded[b,kt*32+k,nt*32+d]
                        at=addresses[f'value_{nt}_{kt}'];self.assertEqual(bytes(memory[at:at+1024]),tile.tobytes())
                for i,binding in enumerate(plan.body.constants):
                    raw=bytes.fromhex(binding.data_hex);packed=pack_bf16_pair_halves(raw) if binding.encoding=='bf16-le-raw' else raw
                    at=addresses[binding.node_id];self.assertEqual(bytes(memory[at:at+len(packed)]),packed)
                for nt,key in enumerate(plan.body.outputs):
                    tile=np.full((32,32),0x7fc0,dtype='<u2')
                    for r in range(min(32,rows-first)):
                        for d in range(min(32,channels-nt*32)):tile[r,d]=0x3f80+(((b*rows+first+r)*channels+nt*32+d)%64)
                    at=addresses[key];memory[at:at+2048]=pack_bf16_pair_halves(tile.tobytes())
            except Exception as error:errors.append(repr(error));state['fault']=True;state['polls']=0
        def transfer(address,pointer,size,put):
            at=address-DRAM_BASE
            if not 0<=at<=len(memory)-size:return 1
            if put:memory[at:at+size]=ctypes.string_at(pointer,size)
            else:
                state['reads']+=1
                if fault=='read' and len(calls)==2:return 1
                ctypes.memmove(pointer,bytes(memory[at:at+size]),size)
            return 0
        callbacks=(rd(read),wr(write),mem(lambda _,a,p,n:transfer(a,p,n,True)),mem(lambda _,a,p,n:transfer(a,p,n,False)))
        lib=ctypes.CDLL(str(directory/'client.so'));lib.call.argtypes=[rd,wr,mem,mem,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_int]
        inputs=(ctypes.c_void_p*3)(*(a.ctypes.data for a in arrays));size=batch*rows*channels*4
        output=ctypes.create_string_buffer(bytes([0xa7])*(size+32),size+32);pointer=arrays[0].ctypes.data if alias==3 else ctypes.addressof(output)+16
        result=lib.call(*callbacks,inputs,pointer,alias)
        self.assertEqual(errors,[]);self.assertEqual([a.tobytes() for a in arrays],before)
        self.assertEqual(output.raw[:16],bytes([0xa7])*16);self.assertEqual(output.raw[-16:],bytes([0xa7])*16)
        if fault or alias:
            self.assertEqual(result,1 if alias else 4 if fault=='device' else 3 if fault=='read' else 2)
            self.assertEqual(output.raw[16:-16],bytes([0xa7])*size)
            self.assertEqual(len(calls),2 if fault in {'device','read'} else 0)
        else:
            self.assertEqual(result,0)
            expected=b''.join(struct.pack('<I',(0x3f80+i%64)<<16) for i in range(batch*rows*channels))
            self.assertEqual(output.raw[16:-16],expected);self.assertEqual(len(calls),plan.groups)


if __name__=='__main__':
    unittest.main()
