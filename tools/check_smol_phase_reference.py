"""Prepare native reference-only layer boundaries; no device or RV64 execution.

The exact admitted S2 generated host/raw runtime is retained. Only shared graph
orchestration providers and a named native layer wrapper replace frozen sources.
Native libc/libm and authenticated captured MMIO replies are explicit controls.
"""
import argparse
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import struct
import re
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'tools/host_iss')]
from atlas_path_guard import checked
from atlas_tooling import sha256
from elf_image import Image
from smol_full_plan import calls as graph_calls
from smol_full_phases import segments, live_regions, scratch_regions, check_observations


def provider(directory):
    path = checked(directory/'check_smol_full_host.py')
    spec = importlib.util.spec_from_file_location('smol_native_control',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def wrapper(merlin):
    return '''#include "smol_policy_graph.h"
#include "smol_vision_graph.h"
#include "smol_policy_raw.h"
#include "smol_full_phase.h"
#include <string.h>
smol_full_phase_cursor smol_full_phase_state;
typedef int (*phase_callback)(unsigned,unsigned,unsigned);
static phase_callback phase_observe;
static unsigned phase_enabled;
void smol_phase_configure(unsigned enabled,phase_callback callback) {
    phase_enabled=enabled;phase_observe=callback;
}
typedef int (*predecessor_callback)(unsigned,unsigned,const void*,size_t,size_t,unsigned,unsigned,unsigned);
static predecessor_callback predecessor_observe;
void smol_predecessor_configure(predecessor_callback callback) { predecessor_observe=callback; }
extern int __real_smol_policy_raw_vision(void*,unsigned,unsigned,const uint16_t*,size_t,size_t,uint16_t*,size_t);
extern int __real_smol_policy_raw_graph(void*,unsigned,unsigned,unsigned,unsigned,const void*,size_t,size_t,
    unsigned,void*,size_t);
int __wrap_smol_policy_raw_vision(void *context,unsigned layer,unsigned role,const uint16_t *input,
    size_t rows,size_t k,uint16_t *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,0,input,rows,k,7,layer,role))
        return SMOL_RAW_INPUT;
    return __real_smol_policy_raw_vision(context,layer,role,input,rows,k,output,n);
}
int __wrap_smol_policy_raw_graph(void *context,unsigned site,unsigned layer,unsigned role,unsigned dtype,
    const void *input,size_t rows,size_t k,unsigned output_dtype,void *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,dtype,input,rows,k,site,layer,role))
        return SMOL_RAW_INPUT;
    return __real_smol_policy_raw_graph(context,site,layer,role,dtype,input,rows,k,output_dtype,output,n);
}
extern uint16_t smol_full_image_output[2][61440];
extern float *const smol_step_velocity[10];
extern float smol_full_actual_final_full_state[1600],smol_full_actual_final_actions[300];
static int phased_vision(const uint16_t *image,const smol_vision_parameters *p,
    const smol_vision_frame *w,smol_vision_state *s,smol_vision_project callback,void *context) {
    if (!phase_enabled) return smol_vision_image(image,p,w,s,callback,context);
    unsigned ordinal=((smol_policy_raw_context*)context)->image;
    for(unsigned layer=0;layer<12;++layer) {
        int code=smol_vision_image_layer(image,p,w,s,callback,context,layer);
        if(code) return code;
        if(layer==11) memcpy(smol_full_image_output[ordinal],w->output,61440*sizeof(uint16_t));
        if(phase_observe(0,ordinal,layer)) return 72;
    }
    return 0;
}
static int phased_prefix(const uint16_t *a,const uint16_t *b,const float *state,
    const uint8_t *pad,const uint8_t *attention,const smol_policy_parameters *p,
    const smol_policy_frame *w,smol_policy_state *s,smol_policy_graph_project callback,void *context) {
    if(!phase_enabled) return smol_policy_prefix(a,b,state,pad,attention,p,w,s,callback,context);
    for(unsigned layer=0;layer<16;++layer) {
        int code=smol_policy_prefix_layer(a,b,state,pad,attention,p,w,s,callback,context,layer);
        if(code) return code;
        if(phase_observe(1,0,layer)) return 72;
    }
    return 0;
}
static int phased_step(const float *input,float time,float *output,const uint8_t *pad,
    const smol_policy_parameters *p,const smol_policy_frame *w,smol_policy_state *s,
    smol_policy_graph_project callback,void *context) {
    if(!phase_enabled) return smol_policy_step(input,time,output,pad,p,w,s,callback,context);
    unsigned ordinal=s->steps;
    for(unsigned layer=0;layer<16;++layer) {
        int code=smol_policy_step_layer(input,time,output,pad,p,w,s,callback,context,layer);
        if(code) return code;
        if(layer==15) {
            memcpy(smol_step_velocity[ordinal],w->velocity,1600*sizeof(float));
            if(ordinal==9) {
                memcpy(smol_full_actual_final_full_state,output,1600*sizeof(float));
                for(unsigned row=0;row<50;++row) for(unsigned col=0;col<6;++col)
                    smol_full_actual_final_actions[row*6+col]=output[row*32+col];
            }
        }
        if(phase_observe(2,ordinal,layer)) return 72;
    }
    return 0;
}
'''


def compile_reference(args,native,bundle,data):
    native.validate_static(args,bundle)
    record = native.compile_host(args,bundle,data)
    out = args.out
    pins = record['source_sha256']
    native.pin(pins,__file__);native.pin(pins,ROOT/'tools/smol_full_phases.py')
    native.pin(pins,ROOT/'runtime/smol_full_phase.h')
    native.pin(pins,np.__file__)
    merlin = checked(args.phase_merlin)
    native.pin(pins,merlin/'test_smol_policy_vision_replay.py')
    overrides = {}
    for stem in ('smol_policy_graph','smol_vision_graph'):
        for extension in ('.c','.h'):
            path = native.pin(pins,merlin/(stem+extension))
            overrides[stem+extension] = str(path)
    for relative, expected in bundle.build['compilation_inputs'].items():
        name = Path(relative).name
        if name.endswith('.h') and name not in overrides:
            candidate = checked(merlin/name)
            if candidate.exists(): native.pin(pins,candidate,expected)
    queue = list(overrides.values()); seen = set()
    while queue:
        path = checked(queue.pop())
        if str(path) in seen: continue
        seen.add(str(path));native.pin(pins,path)
        for name in re.findall(r'^\s*#\s*include\s+"([^"\n]+)"',path.read_text(),re.M):
            if Path(name).name != name: raise ValueError('flat graph header closure required')
            queue.append(str(checked(merlin/name)))
    copied = out/'native-source/smol_full_phase.h'
    copied.write_bytes((ROOT/'runtime/smol_full_phase.h').read_bytes());native.pin(pins,copied)
    source = out/'native-source/main.c'
    original = source.read_text()
    text = original
    for name, replacement in (('smol_vision_image','phased_vision'),
            ('smol_policy_prefix','phased_prefix'),('smol_policy_step','phased_step')):
        if text.count(name+'(') != 1: raise ValueError('exact generated graph invocation required')
        text = text.replace(name+'(',replacement+'(')
    observation_check='for(size_t i=0;i<2715;++i)if(!smol_full_cycles[i]){fail(6);return;}'
    if text.count(observation_check)!=1: raise ValueError('exact continuous observation check required')
    text=text.replace(observation_check,'if(!phase_enabled){'+observation_check+'}')
    source.write_text(wrapper(merlin)+text)
    pins[str(source)] = sha256(source)
    command = record['command']
    command[1:1] = ['-I'+str(merlin),'-Wl,--wrap=smol_policy_raw_vision','-Wl,--wrap=smol_policy_raw_graph']
    command = [overrides.get(Path(value).name,value) if value.endswith(('smol_policy_graph.c','smol_vision_graph.c'))
               else value for value in command]
    with (out/'compile-phase.log').open('w') as stream:
        process = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout)
    if process.returncode: raise ValueError('phase reference compilation failed')
    record.update(schema='atlas.smol_phase_native_compile.v1',source_sha256=pins,command=command,
        library_sha256=sha256(out/'host.so'),scope=__doc__,orchestration_overrides=overrides,
        original_adapted_main_sha256=native.digest(original.encode()),
        phase_main_sha256=sha256(source),rv64_resume_admitted=False)
    native.bookends(pins);bundle.verify_bookends()
    (out/'compile.json').write_text(json.dumps(record,indent=2)+'\n')


def native_image(path):
    raw = path.read_bytes()
    header = struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    if header[0][:7] != b'\x7fELF\x02\x01\x01' or header[1:3] != (3,62) or header[11] != 64:
        raise ValueError('native AMD64 shared library required')
    sections = [struct.unpack_from('<IIQQQQIIQQ',raw,header[6]+i*64) for i in range(header[12])]
    symbols = {}
    for section in sections:
        if section[1] != 2: continue
        if section[9] != 24: raise ValueError('native symbol entry size')
        strings = sections[section[6]]
        table = raw[strings[4]:strings[4]+strings[5]]
        for offset in range(section[4],section[4]+section[5],24):
            no,info,other,index,value,size = struct.unpack_from('<IBBHQQ',raw,offset)
            if no:
                name = table[no:table.index(b'\0',no)].decode()
                symbols[name] = dict(address=value,size=size,section=index)
    return SimpleNamespace(symbols=symbols)


def execute_reference(args,native,bundle,data):
    compiled=json.loads((args.out/'compile.json').read_text())
    native.bookends(compiled['source_sha256'])
    native.require(sha256(args.out/'host.so')==compiled['library_sha256'],'native phase library changed')
    host = ct.CDLL(str(args.out/'host.so'))
    image = native_image(args.out/'host.so')
    live = live_regions(data,image)
    scratch = scratch_regions(image,live)
    ranges = segments();boundaries=[];errors=[];observations=[];predecessors=[];predecessor_errors=[]
    arena=bundle.provenance['placement']['arena_base']
    expected_calls=graph_calls()
    replay=json.loads(bundle.read('host/reference-replay.json'))
    native.require(len(replay['calls'])==len(expected_calls)==1421,'complete source call inventory')
    pred_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_size_t,ct.c_size_t,
                          ct.c_uint,ct.c_uint,ct.c_uint)
    @pred_type
    def check_predecessor(index,dtype,pointer,rows,k,site,layer,role):
        try:
            native.require(index==len(predecessors) and index<1421,'original predecessor call order')
            call=expected_calls[index];record=replay['calls'][index];value=record['input']
            native.require(record['index']==index and record['module']==call['module'] and pointer
                and (dtype,rows,k,site,layer,role)==(call['input_dtype'],call['rows'],call['input_width'],
                    call['site'],call['layer'],call['role']),'original predecessor descriptor')
            size=rows*k*(2 if dtype==0 else 4)
            native.require(value['bytes']==size and value['dtype']==('torch.bfloat16' if dtype==0 else 'torch.float32'),
                           'original predecessor type/extent')
            raw=ct.string_at(pointer,size)
            if site==7 and role==0:
                native.require(value['shape']==[1,3,512,512] and (rows,k)==(1024,768),'source patch geometry')
                # Inverse of the independently qualified component gate's CHW
                # reshape(3,32,16,32,16).transpose(1,3,0,2,4) patch mapping.
                raw=np.frombuffer(raw,dtype='<u2').reshape(32,32,3,16,16).transpose(2,0,3,1,4).tobytes()
            else:
                native.require(value['shape']==([1,k] if site==0 else [1,rows,k]),'original predecessor shape')
            actual=native.digest(raw)
            native.require(actual==value['sha256'],'actual BF16/FP32 predecessor bytes differ: '+record['module'])
            predecessors.append(dict(index=index,module=record['module'],dtype=value['dtype'],bytes=size,
                                     sha256=actual,exact=True))
            return 0
        except BaseException as error:
            predecessor_errors.append(str(error));return 1
    host.smol_predecessor_configure.argtypes=[pred_type]
    host.smol_predecessor_configure(check_predecessor)
    blobs = args.out/'boundaries/blobs';blobs.mkdir(parents=True,exist_ok=True)
    def address(name,size): return ct.addressof((ct.c_ubyte*size).in_dll(host,name))
    def capture_regions():
        payloads={}
        for name,size in live.items():
            raw=ct.string_at(address(name,size),size);digest=native.digest(raw)
            target=blobs/(digest+'.bin')
            if not target.exists(): target.write_bytes(raw)
            native.require(sha256(target)==digest,'boundary blob bookend')
            payloads[name]=dict(bytes=size,sha256=digest,file='boundaries/blobs/'+target.name)
        return payloads
    cursor=(ct.c_uint32*10).in_dll(host,'smol_full_phase_state')
    cursor[:]=[1,0,0,0,0,0,0,0,0,0]
    initial_boundary=dict(next_segment=0,regions=capture_regions())
    callback_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_uint)
    @callback_type
    def observe(kind,ordinal,layer):
        try:
            index=len(boundaries);entry=ranges[index]
            native.require((kind,ordinal,layer)==(('vision','prefix','expert').index(entry['kind']),
                entry['ordinal'],entry['layer']),'semantic phase order')
            state=native.actual_state(host)
            native.require(state['panels']==entry['launch_end'] and state['calls']==entry['call_end']
                and state['sequence']==entry['launch_end'] and not any(state[x] for x in
                ('active','buffers','buffer_pointer','acknowledged','failed','result')),'phase drain/cursor')
            cycle_slots=(ct.c_uint32*2715).in_dll(host,'smol_full_cycles')
            values=check_observations(list(cycle_slots),entry['launch_start'],entry['launch_end'])
            observations.append(dict(index=index,launch_start=entry['launch_start'],launch_end=entry['launch_end'],
                native_synthetic_protocol_deltas=values,measured_model_cycles=False))
            ct.memset(ct.addressof(cycle_slots),0,ct.sizeof(cycle_slots))
            cursor=(ct.c_uint32*10).in_dll(host,'smol_full_phase_state')
            cursor[:]=[1,index+1,kind,ordinal,layer,state['panels'],state['calls'],state['sequence'],0,0]
            payloads=capture_regions()
            boundaries.append(dict(**entry,regions=payloads,actual_state=state))
            for name,size in scratch.items(): ct.memset(address(name,size),args.scratch_byte,size)
            ct.memset(arena,args.scratch_byte,data['dram_bytes'])
            if args.phase_mode=='corrupt' and index==0:
                ct.c_ubyte.from_address(address('smol_v_tokens',live['smol_v_tokens'])+1).value=0x7f
            return 0
        except BaseException as error:
            errors.append(str(error));return 1
    host.smol_phase_configure.argtypes=[ct.c_uint,callback_type]
    host.smol_phase_configure(args.phase_mode!='full',observe)
    args.mode='positive'
    try:
        result=native.execute(args,bundle,data)
        native.require(args.phase_mode!='corrupt','corruption unexpectedly passed')
        native.require(not errors and not predecessor_errors and len(predecessors)==1421
            and len(boundaries)==(0 if args.phase_mode=='full' else 200),
                       'complete semantic phase inventory')
        status='PASS_NATIVE_PHASE_REFERENCE' if boundaries else 'PASS_NATIVE_SHARED_FULL_CONTROL'
    except Exception:
        if args.phase_mode!='corrupt': raise
        uart=(args.out/'uart-positive.log').read_text()
        observed=json.loads((args.out/'observed-positive.json').read_text())
        native.require(not errors and len(boundaries)==1 and 'ATLAS_KERNEL_FAIL' in uart
            and 'ATLAS_KERNEL_PASS' not in uart and observed['tohost']!=1,'live-state corruption must reach actual FAIL')
        result=dict(actual_state=observed['actual_state'],device_execution=False)
        status='PASS_NATIVE_PHASE_LIVE_CORRUPTION'
    record=dict(status=status,scope=__doc__,mode=args.phase_mode,boundaries=boundaries,
        initial_boundary=initial_boundary,fixed_live_regions=live,excluded_scratch=scratch,
        scratch_fill_byte=args.scratch_byte,native_protocol_observations=observations,
        arena_scratch=dict(address=arena,bytes=data['dram_bytes'],fill_byte=args.scratch_byte,clear_events=len(boundaries)),
        observation_policy='record current actual deltas, verify prior/future zero, canonicalize all slots before capture',
        real_device_receipt_comparison_required=True,synthetic_counter_is_timing=False,cleared_scratch_each_phase=args.phase_mode!='full',scratch_clear_events=len(boundaries),
        graph_state_abi_bytes=16,vision_state_abi_bytes=16,cursor_abi_bytes=40,
        device_execution=False,guest_execution=False,rv64_resume_admitted=False,
        original_predecessors=predecessors,original_predecessor_errors=predecessor_errors,
        numpy_version=np.__version__,host_control=result,source_sha256=json.loads((args.out/'compile.json').read_text())['source_sha256'])
    native.bookends(record['source_sha256']);bundle.verify_bookends()
    (args.out/'phase-reference.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=status,boundaries=len(boundaries))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','work','out','native-tools','phase-merlin'):
        p.add_argument('--'+name,type=checked,required=True)
    p.add_argument('--validator-tools',type=checked,default=ROOT/'tools')
    p.add_argument('--model-root',type=checked)
    p.add_argument('--cc',type=checked,default=Path('/usr/bin/gcc'))
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--phase-mode',choices=('full','phased','corrupt'),default='phased')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--scratch-byte',type=int,choices=(0,165),default=0)
    p.add_argument('--execution-cpus',choices=('40-47','48-55','56-63'),default='40-47')
    args=p.parse_args();native=provider(args.native_tools)
    if not 0<args.timeout<=1800: raise ValueError('bounded timeout required')
    bundle,data=native.inputs(args)
    if args.execute: execute_reference(args,native,bundle,data);return
    compile_reference(args,native,bundle,data)
    command=['taskset','-c',args.execution_cpus,sys.executable,'-B',str(Path(__file__).absolute()),*sys.argv[1:],'--execute']
    with (args.out/'process-phase.log').open('w') as stream:
        process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout)
    if process.returncode: raise ValueError('phase native control failed; see process-phase.log')
    native.bookends(json.loads((args.out/'compile.json').read_text())['source_sha256']);bundle.verify_bookends()


if __name__=='__main__': main()
