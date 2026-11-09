"""C admission, tile transport, failure atomicity, and optional instruction run.

Fake MMIO tests qualify transport only. The separately selected original47
test executes the real images and compares results only after their readback.
"""
from dataclasses import replace
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.hierarchical_rmsnorm import prepare_source, operand_spec
from atlas_mlir_native.hierarchical_rmsnorm_c import (STAGES, _checked_stages, _stage_layout,
    _render_package, prepare_c_package)
from atlas_mlir_native.layout import pack_bf16_pair_halves, unpack_bf16_pair_halves
from merlin.compile.launch_schedule import expand_launch_schedule
from test_hierarchical_rmsnorm import source_request, selected, evaluate_stage

ROOT = Path(__file__).resolve().parents[1]
READ = ct.CFUNCTYPE(ct.c_uint32,ct.c_void_p,ct.c_size_t)
WRITE = ct.CFUNCTYPE(None,ct.c_void_p,ct.c_size_t,ct.c_uint32)
MEMORY = ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t)


def build_client(directory, files):
    directory.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (directory/'shim.c').write_text('''#include "norm.h"
static atlas_host host;
static norm_context region;
static norm_source_context source;
static unsigned char raw[NORM_WORKSPACE_BYTES+32u],converted[NORM_SOURCE_WORKSPACE_BYTES+32u];
int bind(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get) {
  __builtin_memset(raw,0xa7,sizeof(raw));__builtin_memset(converted,0xa7,sizeof(converted));
  if(atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0})) return -1;
  if(norm_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw+16,NORM_WORKSPACE_BYTES,20)) return -2;
  return norm_source_bind(&source,&region,converted+16,NORM_SOURCE_WORKSPACE_BYTES);
}
int call(const float *a,const float *b,float *out,int alias) {
  const float *inputs[]={a,b};
  if(alias==1) return norm_source_bind((norm_source_context *)&region,&region,converted+16,NORM_SOURCE_WORKSPACE_BYTES);
  int status=norm_source_run(&source,inputs,out,alias==2?(atlas_host_result *)out:0);
  for(size_t i=0;i<16u;++i) if(raw[i]!=0xa7 || raw[NORM_WORKSPACE_BYTES+16u+i]!=0xa7 ||
      converted[i]!=0xa7 || converted[NORM_SOURCE_WORKSPACE_BYTES+16u+i]!=0xa7) return -3;
  return status;
}
int dispatch(const void *a,const void *b,void *out) {
  const void *inputs[]={a,b};void *outputs[]={out};return norm_source_dispatch(&source,inputs,outputs);
}
''')
    result=subprocess.run([shutil.which('cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
        'norm.c','runtime/atlas_host.c','shim.c','-o','norm.so'],cwd=directory,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    lib=ct.CDLL(str((directory/'norm.so').resolve()))
    lib.bind.argtypes=[READ,WRITE,MEMORY,MEMORY]
    lib.call.argtypes=[ct.c_void_p]*3+[ct.c_int]
    lib.dispatch.argtypes=[ct.c_void_p]*3
    return lib


class Device:
    """Test-local synchronous MMIO adapter; production model code is untouched."""
    def __init__(self, test, plan, stages, record, *, execute=None, fail_read_at=None, corrupt=False):
        self.test,self.plan,self.stages,self.record=test,plan,stages,record
        self.images={sha256(v['binary']).hexdigest():k for k,v in stages.items()}
        self.schedule=expand_launch_schedule(record['device_launch_schedule'],image_sha256s=record['device_launch_schedule']['images'])
        self.dram=bytearray([0xa5])*(32*2048)
        self.words={};self.calls=[];self.errors=[];self.running=False;self.polls=0
        self.mmio=[];self.memory_events=[]
        self.execute,self.fail_read_at,self.corrupt=execute,fail_read_at,corrupt
        self.previous={k:[] for k in STAGES}
        self.callbacks=READ(self.read),WRITE(self.write),MEMORY(self.put),MEMORY(self.get)

    def read(self,_,address):
        self.mmio.append(('read',address))
        if address==0x40004:return len(self.calls)
        if address!=0x40008:self.errors.append('unexpected CSR');return 3
        if not self.running:return 5
        self.polls+=1
        if self.polls==1:return 0
        self.running=False
        return 3 if self.errors else 5

    def write(self,_,address,value):
        self.mmio.append(('write',address,value))
        if 0x20000<=address<0x40000:
            if address==0x20000:self.words.clear()
            self.words[(address-0x20000)//4]=value;return
        try:
            self.test.assertEqual((address,value),(0x40018,1))
            binary=b''.join(self.words[i].to_bytes(4,'little') for i in range(len(self.words)))
            digest=sha256(binary).hexdigest();kind=self.images[digest]
            self.test.assertEqual(digest,self.schedule[len(self.calls)])
            stage=self.stages[kind];self.calls.append(kind);self.running=True;self.polls=0
            before=bytes(self.dram)
            if self.execute:
                after=self.execute(kind,before)
            else:
                after=self.fake(kind,before)
            self.test.assertEqual(len(after),len(before))
            mutable=[(r['address']-DRAM_BASE,r['bytes']) for r in stage['write_spans']]
            keep=bytearray(b'\1')*len(before)
            for a,s in mutable:keep[a:a+s]=bytes(s)
            self.test.assertEqual(bytes(x for x,k in zip(after,keep) if k),bytes(x for x,k in zip(before,keep) if k))
            self.dram[:]=after
        except Exception as error:self.errors.append(repr(error))

    def put(self,_,address,pointer,count):
        self.memory_events.append(('write',address,count))
        if self.running:self.errors.append('input overwritten before completion');return 1
        # Only immutable external input/constant spans may be written by transport.
        if not any(a<=address and address+count<=a+s and m=='READ'
                   for stage in self.stages.values() for a,s,m in stage['ranges']):
            self.errors.append('write outside declared external inputs');return 1
        at=address-DRAM_BASE;self.dram[at:at+count]=ct.string_at(pointer,count);return 0

    def get(self,_,address,pointer,count):
        self.memory_events.append(('read',address,count))
        if self.running:self.errors.append('early readback');return 1
        if len(self.calls)==self.fail_read_at:return 1
        stage=self.stages[self.calls[-1]]
        if (address,count)!=(stage['output'],stage['output_bytes']):
            self.errors.append('readback outside declared output');return 1
        at=address-DRAM_BASE;ct.memmove(pointer,bytes(self.dram[at:at+count]),count);return 0

    def fake(self,kind,before):
        rows,width=self.plan.region.matrix_shape;tiles,groups=width//32,width//1024
        at=len(self.calls)-1;per=2*tiles+groups+2;rowgroup,step=divmod(at,per)
        stage=self.stages[kind]
        def logical(key):
            a=stage['addresses'][key]-DRAM_BASE
            return np.frombuffer(unpack_bf16_pair_halves(before[a:a+2048]),dtype='<u2').reshape(32,32)
        first=rowgroup*32
        hidden=np.zeros((32,width),dtype='<u2')
        hidden[:min(32,rows-first)]=self.hidden[first:first+32]
        for binding in self.plan.stages[kind].constants:
            self.test.assertEqual(logical(binding.node_id).tobytes(),bytes.fromhex(binding.data_hex))
        if kind=='square_reduce':
            self.test.assertEqual(logical('hidden').tobytes(),hidden[:,step*32:step*32+32].tobytes())
            result=np.repeat((np.arange(32,dtype='<u2')+0x3a80+step)[:,None],32,axis=1)
            if step==0:self.previous={k:[] for k in STAGES}
        elif kind=='reduce':
            index=step-tiles;expected=np.zeros((32,32),dtype='<u2')
            partials=self.previous['square_reduce'][index*32:index*32+32] if index<groups else self.previous['reduce']
            for col,tile in enumerate(partials):expected[:,col]=tile[:,0]
            self.test.assertEqual(logical('partials').tobytes(),expected.tobytes())
            result=np.full((32,32),0x3f00+index,dtype='<u2')
        elif kind=='inverse':
            self.test.assertEqual(logical('total').tobytes(),self.previous['reduce'][-1].tobytes())
            result=np.full((32,32),0x3f80,dtype='<u2')
        else:
            ctile=step-tiles-groups-2
            self.test.assertEqual(logical('hidden').tobytes(),hidden[:,ctile*32:ctile*32+32].tobytes())
            self.test.assertEqual(logical('inverse').tobytes(),self.previous['inverse'][-1].tobytes())
            weights=np.tile(self.gamma[ctile*32:ctile*32+32],(32,1))
            self.test.assertEqual(logical('gamma').tobytes(),weights.tobytes())
            result=np.array([[0 if first+r>=rows else 0x38+(first+r+c+ctile)%7 for c in range(32)] for r in range(32)],dtype='u1')
            if self.corrupt and at==len(self.schedule)-1:result[0,0]=0x7f
        self.previous[kind].append(result.copy())
        after=bytearray(before);a=stage['output']-DRAM_BASE
        payload=result.tobytes() if kind=='publish' else pack_bf16_pair_halves(result.tobytes())
        after[a:a+len(payload)]=payload
        return bytes(after)


class HierarchicalRMSNormCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        cls.work=Path(cls.temporary.name);cls.cases={}
        for rows,width in ((1,2048),(33,1024)):
            source=source_request(width,rows);plan=prepare_source(source,selected(source))
            stages={k:dict(**_stage_layout(plan.stages[k],k),binary=bytes([i+1])*4) for i,k in enumerate(STAGES)}
            record,files=_render_package(plan,stages,'norm',ROOT,'separate')
            lib=build_client(cls.work/f'{rows}-{width}',files)
            cls.cases[rows]=(plan,stages,record,lib)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def run_case(self, rows, fault=None, alias=0):
        plan,stages,record,lib=self.cases[rows];width=plan.width
        h=np.resize(np.array([2**-10,-.125,.25,-0.,1.],dtype=np.float32),rows*width).reshape(rows,width)
        gamma=np.resize(np.array([1.,-.5,0.,2**-24,256.],dtype=np.float32),width)
        if fault=='h':h[-1,-1]=np.inf
        if fault=='upper':h[-1,-1]=2.015625
        if fault=='witness':h[-1,:]=2**-24
        if fault=='gamma':gamma[-1]=np.nan
        if fault=='zero':h[:]=0.
        before=[h.tobytes(),gamma.tobytes()]
        device=Device(self,plan,stages,record,fail_read_at=record['device_launch_schedule']['total_launches'] if fault=='read' else None,corrupt=fault=='output')
        if fault not in {'h','gamma'}:
            device.hidden=operand_spec().encode(h);device.gamma=operand_spec().encode(gamma)
        self.assertEqual(lib.bind(*device.callbacks),0)
        output=(ct.c_uint8*(rows*width*4+32))(*([0xcd]*(rows*width*4+32)));pointer=ct.addressof(output)+16
        result=lib.call(h.ctypes.data,gamma.ctypes.data,h.ctypes.data if alias==3 else pointer,alias)
        self.assertFalse(device.errors,device.errors)
        self.assertEqual(before,[h.tobytes(),gamma.tobytes()])
        if alias or fault in {'h','gamma','upper','witness','read','output'}:
            self.assertEqual(result,1 if alias else 3 if fault=='read' else 2)
            self.assertEqual(bytes(output),b'\xcd'*len(output))
            if alias or fault in {'h','gamma','upper','witness'}:
                self.assertEqual(device.calls,[]);self.assertEqual(device.mmio,[]);self.assertEqual(device.memory_events,[])
        else:
            self.assertEqual(result,0)
            codes=np.empty((rows,width),dtype='u1')
            for r in range(rows):
                for c in range(width):codes[r,c]=0x38+(r+c%32+c//32)%7
            words=[]
            for code in codes.flat:
                e=((int(code)>>3)&15)+120+plan.output_scale_exponent
                words.append((e<<23)|((int(code)&7)<<20))
            self.assertEqual(bytes(output)[16:-16],struct.pack('<'+'I'*len(words),*words))
            self.assertEqual(bytes(output)[:16]+bytes(output)[-16:],b'\xcd'*32)
            self.assertEqual(len(device.calls),record['device_launch_schedule']['total_launches'])

    def test_all_four_images_partial_columns_tail_rows_and_exact_unpack(self):
        for rows in self.cases:self.run_case(rows)

    def test_whole_array_admission_before_first_mmio_and_zero_rows(self):
        for fault in ('h','gamma','upper','witness','zero'):
            with self.subTest(fault=fault):self.run_case(33,fault=fault)

    def test_late_failure_and_invalid_native_code_keep_public_output_atomic(self):
        for fault in ('read','output'):self.run_case(33,fault=fault)

    def test_context_device_and_input_output_aliases_refuse(self):
        for alias in (1,2,3):self.run_case(33,alias=alias)

    def test_source_policy_stage_hashes_constants_and_linked_paths(self):
        source=source_request();plan=prepare_source(source,selected(source));directory=self.work/'binding';directory.mkdir()
        encoded=lambda value:(json.dumps(value,sort_keys=True)+'\n').encode()
        parent=plan.record()|{'compiled_stages':{}}
        for kind,body in plan.stages.items():
            root=directory/kind;root.mkdir();layout=_stage_layout(body,kind)
            execution=dict(schema='atlas.native_tile_execution_plan.v8',input_addresses=list(layout['addresses'].values()),
                output_addresses=[layout['output']],output_bytes=layout['output_bytes'],
                output_physical_layouts=['fp8_row_major' if kind=='publish' else 'bf16_pair_halves'],**layout['expected'])
            constants=[]
            for i,binding in enumerate(body.constants):
                data=pack_bf16_pair_halves(bytes.fromhex(binding.data_hex));name=f'constants/constant-{i:03d}.bin'
                p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
                constants.append(dict(node_id=binding.node_id,address=layout['addresses'][binding.node_id],bytes=len(data),
                    encoding=binding.encoding,physical_layout='bf16_pair_halves',path=name,sha256=sha256(data).hexdigest()))
            execution['constant_inputs']=constants;execution_bytes=encoded(execution)
            binary=b'fake'
            physical_bytes=encoded(dict(schema='atlas.physical_program.v1',word_endianness='little',word_count=1,timing_state='timed',
                instructions=[dict(word_index=0,word_u32=int.from_bytes(binary,'little'))]))
            (root/'physical-program.json').write_bytes(physical_bytes)
            manifest=dict(binary_sha256=sha256(binary).hexdigest(),execution_plan_sha256=sha256(execution_bytes).hexdigest(),
                stage_artifacts={'physical-program.json':sha256(physical_bytes).hexdigest()},
                constant_inputs=constants,target_inventory_sha256=sha256((ROOT/'docs/selected-variant-inventory.json').read_bytes()).hexdigest())
            manifest_bytes=encoded(manifest)
            for name,data in {'manifest.json':manifest_bytes,'execution_plan.json':execution_bytes,'program.bin':binary,'source-request.json':encoded(body.record())}.items():(root/name).write_bytes(data)
            parent['compiled_stages'][kind]=dict(request_digest=body.digest(),program_sha256=sha256(binary).hexdigest(),manifest_path=kind+'/manifest.json',manifest_sha256=sha256(manifest_bytes).hexdigest(),execution_plan_sha256=sha256(execution_bytes).hexdigest())
        (directory/'hierarchical-rmsnorm-plan.json').write_bytes(encoded(parent));(directory/'semantic-source.json').write_bytes(encoded(source.record()))
        module='atlas_mlir_native.hierarchical_rmsnorm_c'
        with patch(module+'.verify_native_publication') as verify,patch(module+'.check_io_memory_manifest'):
            _checked_stages(plan,directory,ROOT);self.assertEqual(verify.call_count,4)
            other=prepare_source(source,selected(source)|{'provenance_sha256':'34'*32})
            with self.assertRaisesRegex(ValueError,'provenance'):_checked_stages(other,directory,ROOT)
            file=directory/'inverse/constants/constant-000.bin';original=file.read_bytes();file.write_bytes(bytes(2048))
            with self.assertRaisesRegex(ValueError,'constant payload'):_checked_stages(plan,directory,ROOT)
            file.write_bytes(original)
            execution=directory/'publish/execution_plan.json';old=execution.read_bytes();execution.write_bytes(old+b' ')
            with self.assertRaisesRegex(ValueError,'stage hashes'):_checked_stages(plan,directory,ROOT)
            execution.write_bytes(old)
            # Rehashing a forged transport plan cannot authorize different memory.
            forged=json.loads(old);forged['buffers'][-1]['bytes']+=1024
            forged_bytes=encoded(forged);execution.write_bytes(forged_bytes)
            manifest_path=directory/'publish/manifest.json';manifest=json.loads(manifest_path.read_bytes())
            manifest['execution_plan_sha256']=sha256(forged_bytes).hexdigest();manifest_bytes=encoded(manifest)
            manifest_path.write_bytes(manifest_bytes)
            parent['compiled_stages']['publish']['execution_plan_sha256']=sha256(forged_bytes).hexdigest()
            parent['compiled_stages']['publish']['manifest_sha256']=sha256(manifest_bytes).hexdigest()
            (directory/'hierarchical-rmsnorm-plan.json').write_bytes(encoded(parent))
            with self.assertRaisesRegex(ValueError,'buffers or verified tier'):_checked_stages(plan,directory,ROOT)
            source_file=directory/'semantic-source.json';source_file.unlink();source_file.symlink_to(directory/'square_reduce/source-request.json')
            with self.assertRaisesRegex(ValueError,'unsafe'):_checked_stages(plan,directory,ROOT)


@unittest.skipUnless(os.environ.get('ATLAS_NORM_C_NATIVE_OUT'),'select a source-bound four-stage compilation')
class HierarchicalRMSNormNativeCTest(unittest.TestCase):
    def test_generated_c_original47_four_image_execution(self):
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from atlas_mlir_native.binding import TARGET_IDENTITY
        compiled=Path(os.environ['ATLAS_NORM_C_NATIVE_OUT']).resolve()
        source_root=Path(os.environ['ATLAS_NORM_C_SOURCE']).resolve()
        source=translate_linalg_text((source_root/'model.mlir').read_text(),entry='forward',target_identity=TARGET_IDENTITY,lowering_policy='diagnostic').request
        parent=json.loads((compiled/'hierarchical-rmsnorm-plan.json').read_text())
        policy={k:parent[k] for k in ('policy','source_request_digest','provenance_sha256','output_scale_exponent')}
        plan=prepare_source(source,policy);stages=_checked_stages(plan,compiled,ROOT)
        record,files=prepare_c_package(source,compiled,'norm',policy=policy,target_source=ROOT)
        work=Path(os.environ['ATLAS_NORM_C_TEST_OUT']).resolve();lib=build_client(work,files)
        (work/'description.json').write_text(json.dumps(record,indent=2)+'\n')
        if os.environ.get('ATLAS_NORM_C_MODEL')!='1':self.skipTest('C export and compile passed; original47 execution not selected')
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        from test_mxu1_cap448 import authenticate_model,MODEL_PIN
        model_before=authenticate_model(Path(os.environ['NPU_MODEL_ROOT']))
        torch.set_num_threads(1)
        inventory=json.loads((ROOT/'docs/selected-variant-inventory.json').read_text());cycles=[]
        decoded={k:decode_binary(inventory,v['binary']) for k,v in stages.items()}
        def execute(kind,before):
            program=InstantiableProgram([model_instruction(row,inventory,isa) for row in decoded[kind]['instructions']])
            program.dram_base=DRAM_BASE;program.memory_regions=[(0,torch.tensor(list(before),dtype=torch.uint8))]
            config=DefaultHardwareConfig();config.arch_state_config=replace(config.arch_state_config,numerics='rtl',randomize_init=True,init_seed=417)
            simulation=Simulation(config,LoggerConfig(filename=str(work/'model.json')),program,verbose=False)
            state=simulation.core.arch_state
            try:
                state.vmem.fill_(0xcd);simulation.run(max_cycles=100000)
                self.assertTrue(state.halted);self.assertEqual(simulation.get_stats().runtime_errors,0)
                cycles.append(simulation.get_stats().cycles)
                return bytes(state.read_dram(DRAM_BASE,len(before)).tolist())
            finally:state.close()
        device=Device(self,plan,stages,record,execute=execute)
        self.assertEqual(lib.bind(*device.callbacks),0)
        with np.load(source_root/'inputs.npz') as values:
            self.assertEqual(len(values.files),1)
            h=values[values.files[0]].copy()
        from safetensors.numpy import load_file
        weights=load_file(source_root/'weights.safetensors');gamma=next(iter(weights.values())).copy()
        self.assertEqual(h.size,plan.region.matrix_shape[0]*plan.width)
        before=(h.tobytes(),gamma.tobytes());output=np.full(h.shape,np.nan,dtype=np.float32)
        self.assertEqual(lib.call(h.ctypes.data,gamma.ctypes.data,output.ctypes.data,0),0,device.errors)
        self.assertFalse(device.errors,device.errors)
        self.assertEqual(before,(h.tobytes(),gamma.tobytes()))
        self.assertEqual(len(device.calls),record['device_launch_schedule']['total_launches'])
        (work/'actual-output-f32.bin').write_bytes(output.tobytes())
        (work/'actual-launches.json').write_text(json.dumps(dict(images=record['device_launch_schedule']['images'],
            kinds=device.calls,cycles=cycles,declared_write_guards=True,immutable_inputs=True),indent=2)+'\n')
        # Independent primitive evaluator is a comparator, invoked only after
        # actual generated-C/native image execution has returned its own bytes.
        hw=operand_spec().encode(h).reshape(-1,plan.width);gw=operand_spec().encode(gamma).reshape(-1)
        expected=np.empty_like(output).view(np.uint32).reshape(-1,plan.width)
        from atlas_mlir_native.hierarchical_rmsnorm import pack_partial_columns
        for first in range(0,len(hw),32):
            hidden=np.zeros((32,plan.width),dtype='<u2');hidden[:min(32,len(hw)-first)]=hw[first:first+32]
            partials=[]
            for ctile in range(plan.width//32):
                result=evaluate_stage(plan.stages['square_reduce'],{'hidden':tuple(map(int,hidden[:,ctile*32:ctile*32+32].flat))})
                partials.append(struct.pack('<1024H',*result))
            totals=[]
            for g in range(plan.width//1024):
                result=evaluate_stage(plan.stages['reduce'],{'partials':struct.unpack('<1024H',pack_partial_columns(partials[g*32:g*32+32]))})
                totals.append(struct.pack('<1024H',*result))
            total=evaluate_stage(plan.stages['reduce'],{'partials':struct.unpack('<1024H',pack_partial_columns(totals))})
            inverse=evaluate_stage(plan.stages['inverse'],{'total':total})
            for ctile in range(plan.width//32):
                result=evaluate_stage(plan.stages['publish'],{'hidden':tuple(map(int,hidden[:,ctile*32:ctile*32+32].flat)),'inverse':inverse,
                    'gamma':tuple(map(int,np.tile(gw[ctile*32:ctile*32+32],(32,1)).flat))})
                for r in range(min(32,len(hw)-first)):
                    for c in range(32):
                        code=int(result[r*32+c]);bits=0 if not code else ((code&128)<<24)|((((code>>3)&15)+120+plan.output_scale_exponent)<<23)|((code&7)<<20)
                        expected[first+r,ctile*32+c]=bits
        self.assertEqual(output.tobytes(),expected.tobytes())
        self.assertEqual(model_before,authenticate_model(Path(os.environ['NPU_MODEL_ROOT'])))
        receipt=dict(status='PASS',model_commit=MODEL_PIN,numerics='rtl',launches=len(device.calls),cycles=cycles,
            image_sha256s=record['device_launch_schedule']['images'],source_request_digest=source.digest(),
            provenance_sha256=policy['provenance_sha256'],all_declared_write_span_guards=True,immutable_inputs=True,
            generated_c_sha256=sha256(files['norm.c']).hexdigest(),
            output_sha256=sha256(output.tobytes()).hexdigest(),comparison='independent primitive stage semantics; never model input')
        (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':unittest.main()
