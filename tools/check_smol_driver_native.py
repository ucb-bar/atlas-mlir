"""Qualify the exact dual-mode production driver using captured native MMIO.

Native libc/libm and synthetic protocol counters are explicit substitutions;
this checks arithmetic/predecessor/state boundaries, never Model timing or RV64.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/host_iss')]
from atlas_path_guard import checked
from atlas_tooling import sha256
from smol_full_plan import calls as graph_calls
from smol_full_phases import segments, live_regions, scratch_regions, check_observations
from check_smol_phase_reference import provider, wrapper, native_image


def compile_control(args,native,bundle,data):
    native.STEMS=(*native.STEMS,'smol_full_driver')
    for path in (__file__,ROOT/'tools/check_smol_phase_reference.py',ROOT/'tools/smol_full_phases.py',np.__file__):
        native.pin(args.validator_binding,path)
    record=native.compile_host(args,bundle,data)
    # Only raw projection entry points are wrapped. The production driver and
    # graph bodies are the same compilation units used for the RV64 ELF.
    extra=wrapper(None)
    extra=extra[extra.index('typedef int (*predecessor_callback)'):extra.index('extern uint16_t smol_full_image_output')]
    source=args.out/'native-source/predecessors.c'
    source.write_text('#include "smol_policy_raw.h"\n'+extra)
    native.pin(record['source_sha256'],source)
    command=list(record['command'])
    command[1:1]=['-Wl,--wrap=smol_policy_raw_vision','-Wl,--wrap=smol_policy_raw_graph',str(source)]
    with (args.out/'compile-predecessors.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
    native.require(result.returncode==0,'native predecessor compilation failed')
    native.pin(record['source_sha256'],source)
    native.pin(record['source_sha256'],__file__)
    native.pin(record['source_sha256'],ROOT/'tools/check_smol_phase_reference.py')
    native.pin(record['source_sha256'],ROOT/'tools/smol_full_phases.py')
    native.pin(record['source_sha256'],np.__file__)
    native.bookends(record['source_sha256']);bundle.verify_bookends()
    record.update(command=command,library_sha256=sha256(args.out/'host.so'))
    (args.out/'compile.json').write_text(json.dumps(record,indent=2)+'\n')


def predecessors(host,bundle,native):
    rows=json.loads(bundle.read('host/reference-replay.json'))['calls'];expected=graph_calls()
    results,errors=[],[]
    callback_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_size_t,ct.c_size_t,
                             ct.c_uint,ct.c_uint,ct.c_uint)
    @callback_type
    def check(index,dtype,pointer,m,k,site,layer,role):
        try:
            native.require(index==len(results) and index<1421,'original predecessor order')
            call=expected[index];record=rows[index];value=record['input']
            native.require(record['index']==index and record['module']==call['module'] and pointer
                and (dtype,m,k,site,layer,role)==(call['input_dtype'],call['rows'],call['input_width'],
                    call['site'],call['layer'],call['role']),'original predecessor descriptor')
            size=m*k*(2 if dtype==0 else 4)
            native.require(value['bytes']==size and value['dtype']==('torch.bfloat16' if dtype==0 else 'torch.float32'),
                           'original predecessor type/extent')
            raw=ct.string_at(pointer,size)
            if site==7 and role==0:
                native.require(value['shape']==[1,3,512,512] and (m,k)==(1024,768),'patch geometry')
                raw=np.frombuffer(raw,dtype='<u2').reshape(32,32,3,16,16).transpose(2,0,3,1,4).tobytes()
            else: native.require(value['shape']==([1,k] if site==0 else [1,m,k]),'original predecessor shape')
            native.require(native.digest(raw)==value['sha256'],'original predecessor bytes '+record['module'])
            results.append(dict(index=index,sha256=value['sha256'],dtype=value['dtype'],bytes=size));return 0
        except BaseException as error: errors.append(str(error));return 1
    host.smol_predecessor_configure.argtypes=[callback_type]
    host.smol_predecessor_configure(check)
    return check,results,errors


def execute_phases(args,native,bundle,data):
    require=native.require;bookends=native.bookends;Payloads=native.Payloads
    Raw=native.Raw;Host=native.Host;digest=native.digest;EPILOGUE=native.EPILOGUE
    out = checked(args.out);record = json.loads((out/'compile.json').read_text())
    require(record['bundle_manifest_sha256'] == bundle.checksums_sha256,'compiled bundle binding')
    bookends(record['source_sha256'])
    if record.get('model_root'):
        sys.path[:0] = [str(checked(args.validator_tools))]
        from run_model_tile import configure_model, model_binding
        configure_model(checked(record['model_root']))
        require(model_binding(checked(record['model_root'])) == record['model_binding'],'changed configured model')
    library = out/'host.so';require(sha256(library) == record['library_sha256'],'compiled library changed')
    host = ct.CDLL(str(library))
    callback_type = ct.CFUNCTYPE(ct.c_int,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_size_t,ct.c_uint32,ct.c_void_p)
    host.smol_native_configure.argtypes = [ct.c_size_t,ct.c_size_t,ct.c_size_t,callback_type,ct.c_void_p]
    host.smol_native_map.argtypes = [ct.c_size_t,ct.c_size_t]
    for name in ('smol_native_starts','smol_native_error'):
        getattr(host,name).restype = ct.c_uint32
    host.smol_native_output.restype = ct.c_void_p;host.smol_native_output_bytes.restype = ct.c_size_t
    for name in ('smol_native_imem_base','smol_native_csr_base','smol_native_imem_bytes'):
        getattr(host,name).restype = ct.c_size_t
    memory = ct.CDLL(None);memory.memcmp.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t];memory.memcmp.restype = ct.c_int
    arena = bundle.provenance.get('placement',{}).get('arena_base',0x100000000)
    require(ct.sizeof(Raw) == 160 and ct.sizeof(Host) == 112,'native actual ABI sizes')
    raw_context = Raw.in_dll(host,'smol_full_raw');runtime = Host.in_dll(host,'smol_full_host')
    require(raw_context.dram == arena and raw_context.panel_count == 2715 and raw_context.call_count == 1421
            and raw_context.host == ct.addressof(runtime) and not raw_context.panel_cursor
            and not raw_context.call_cursor,'actual initial raw context')
    require(host.smol_native_map(arena,data['dram_bytes']) == 0,'MAP_FIXED_NOREPLACE arena reservation failed')
    payloads = Payloads(bundle);events,errors = [],[]
    def compare(address,name,size):
        expected,count = payloads.get(name);require(count == size,'captured payload size')
        require(memory.memcmp(address,expected,size) == 0,'actual staged bytes differ: '+name)
    def fill(address,size,value):
        require(ct.string_at(address,size) == bytes([value])*size,'actual initial poison/guard differs')
    @callback_type
    def reply(index,program,words,sequence,context):
        try:
            require(index == len(events) and index < 2715 and sequence == index+1,'ordered native start')
            item = data['kernels'][index];m,k,n = item['physical_mkn']
            require(raw_context.panel_cursor == index and raw_context.call_cursor == item['call']
                    and runtime.active == 1 and runtime.sequence == sequence
                    and runtime.active_buffer_count == 1
                    and runtime.active_buffers == ct.addressof(raw_context.active_buffer)
                    and raw_context.active_buffer.address == arena
                    and raw_context.active_buffer.size == item['total_dram_bytes']
                    and raw_context.active_buffer.access == 2,'actual active raw buffer ownership')
            raw = bundle.read(item['original']);wrapped = raw[:-4]+EPILOGUE
            require(ct.string_at(program,words*4) == wrapped,'actual wrapped IMEM differs')
            compare(arena,item['activation'],m*k);compare(arena+m*k,item['weights'],k*n)
            offset,size = item['output_offset'],2*m*n
            require(offset == m*k+k*n+256 and item['total_dram_bytes'] == offset+size+256,'physical staging extent')
            fill(arena+m*k+k*n,256,0xa5);fill(arena+offset,size,0xcd);fill(arena+offset+size,256,0x5a)
            expected,count = payloads.get(item['expected']);require(count == size,'physical reply extent')
            ct.memmove(arena+offset,expected,size)
            corrupted = args.mode == 'negative' and index == 2714
            if corrupted:
                rows = item['logical_mkn'][0]
                require(rows == 50 and m == 64 and n == 32,'negative requires the final padded action panel')
                word = (rows//32)*(n//32)*1024+(rows%32)*16
                address = arena+offset+2*word
                require(ct.c_uint16.from_address(address).value == 0,'captured padded word must be positive zero')
                ct.c_ubyte.from_address(address).value ^= 1
            events.append(dict(index=index,sequence=sequence,wrapped_program_sha256=digest(wrapped),
                               initial_sha256=item['initial_sha256'],all_staged_bytes_exact=True,
                               captured_output_sha256=bundle.sums[item['expected']],corrupted=corrupted))
            return 0
        except BaseException as error:
            errors.append(dict(index=int(index),error=str(error)));return 1
    require(host.smol_native_configure(host.smol_native_imem_base(),host.smol_native_csr_base(),
                                      host.smol_native_imem_bytes(),reply,None) == 0,'native device configuration')
    pred_callback,pred_records,pred_errors=predecessors(host,bundle,native)
    image=native_image(out/'host.so');live=live_regions(data,image);scratch=scratch_regions(image,live)
    chain=json.loads(bundle.read('segments.json'));plan=segments();observations=[];boundaries=[];hook_errors=[]
    compiled_layout=list((ct.c_uint64*12).in_dll(host,'smol_full_compiled_phase_layout'))
    require(compiled_layout==[1,40,28,16,16,160,144,80,112,2715,1421,200],'compiled phase ABI')
    current=[0];reset_events=[]
    def address(name,size): return ct.addressof((ct.c_ubyte*size).in_dll(host,name))
    ready_type=ct.CFUNCTYPE(None)
    @ready_type
    def ready():
        try:
            index=current[0]
            for name,size in live.items():
                member=chain['boundaries'][index]['regions'][name]
                pointer,count=payloads.get(member);require(count==size,'input boundary size')
                ct.memmove(address(name,size),pointer,size)
            for name,size in scratch.items(): ct.memset(address(name,size),0xa5,size)
            ct.memset(arena,0xa5,data['dram_bytes'])
            reset_events.append(index)
            ct.c_uint64.in_dll(host,'smol_full_segment_request').value=index
            if args.control=='corrupt':
                ct.c_ubyte.from_address(address('smol_v_tokens',live['smol_v_tokens'])+1).value^=1
        except BaseException as error: hook_errors.append(str(error))
    host.smol_full_driver_observe_ready.argtypes=[ready_type]
    host.smol_full_driver_observe_ready(ready)
    start = time.monotonic()
    try:
        for index,entry in enumerate(plan):
            current[0]=index;before=host.smol_native_output_bytes()
            host.atlas_workload_main()
            uart=ct.string_at(host.smol_native_output()+before,host.smol_native_output_bytes()-before).decode('ascii')
            mailbox=list((ct.c_uint64*16).in_dll(host,'atlas_workload_mailbox'))
            if args.control=='corrupt':
                require(index==0 and len(events)==0 and mailbox[1]==3
                    and 'ATLAS_KERNEL_FAIL' in uart and 'check=smol_v_tokens' in uart
                    and 'ATLAS_KERNEL_PASS' not in uart,'input corruption rejection')
                break
            count=entry['launch_end']-entry['launch_start'];calls=entry['call_end']-entry['call_start']
            from result_lines import verify as verify_result_lines
            verify_result_lines(uart.encode('ascii'),'smol-segment',count,progress_launches=[count])
            require(mailbox==[0x534d4f4c46554c4c,2,count,calls,
                int(entry['kind']=='expert' or entry['kind']=='prefix' and entry['layer']==15),
                entry['ordinal']+int(entry['layer']==15) if entry['kind']=='expert' else 0,
                1,0,0,entry['launch_end'],entry['call_end'],entry['launch_end'],0,0,0,0], 'phase mailbox')
            require(ct.c_uint64.in_dll(host,'tohost').value==1,'phase completion mailbox')
            values=list((ct.c_uint32*2715).in_dll(host,'smol_full_segment_observations'))
            own=check_observations(values,entry['launch_start'],entry['launch_end'])
            require(not any((ct.c_uint32*2715).in_dll(host,'smol_full_cycles')),'canonical zero cycles')
            for name,size in live.items(): compare(address(name,size),chain['boundaries'][index+1]['regions'][name],size)
            observations.append(dict(index=index,native_synthetic_protocol_deltas=own,measured_model_cycles=False))
            boundaries.append(index)
        require(not hook_errors and not pred_errors,'native boundary/predecessor callback failure')
        output = ct.string_at(host.smol_native_output(),host.smol_native_output_bytes()).decode('utf-8')
        (out/'uart-phases.log').write_text(output)
        result=dict(status='PASS_NATIVE_DUAL_DRIVER_'+args.control.upper(),mode=args.control,
            boundaries=len(boundaries),starts=len(events),predecessors=pred_records,
            compiled_phase_layout=compiled_layout,scratch_reset_events=len(reset_events),
            arena_reset_events=len(reset_events),scratch_fill_byte=165,
            hook_errors=hook_errors,predecessor_errors=pred_errors,observations=observations,
            synthetic_counter_is_timing=False,guest_execution=False,device_execution=False,
            source_bookends_verified=True,elapsed_seconds=time.monotonic()-start)
        if args.control=='corrupt':
            result.update(negative_tohost=ct.c_uint64.in_dll(host,'tohost').value,
                negative_mailbox=list((ct.c_uint64*16).in_dll(host,'atlas_workload_mailbox')),
                negative_uart_sha256=native.digest(output.encode('ascii')))
        require(not errors and host.smol_native_error()==0,'native device protocol '+str(errors[:1]))
        if args.control!='corrupt':
            require(len(boundaries)==200 and len(events)==2715 and len(pred_records)==1421,'complete traversal')
            for item in data['reference_buffers']: compare(address(item['actual_symbol'],item['bytes']),item['file'],item['bytes'])
        bookends(record['source_sha256']);bundle.verify_bookends()
        (out/'driver-result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        payloads.close();host.smol_native_unmap()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','work','out','native-tools'):
        p.add_argument('--'+name,type=checked,required=True)
    p.add_argument('--validator-tools',type=checked,default=ROOT/'tools')
    p.add_argument('--model-root',type=checked,required=True)
    p.add_argument('--cc',type=checked,default=Path('/usr/bin/gcc'))
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--control',choices=('full','phased','corrupt'),default='phased')
    p.add_argument('--execution-cpus',choices=('40-47','48-55','56-63'),default='40-47')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();native=provider(args.native_tools);args.mode='positive'
    native.require(0<args.timeout<=1200,'bounded native timeout')
    bundle,data=native.inputs(args)
    if args.execute:
        if args.control=='full':
            host=ct.CDLL(str(args.out/'host.so'))
            compiled_layout=list((ct.c_uint64*12).in_dll(host,'smol_full_compiled_phase_layout'))
            native.require(compiled_layout==[1,40,28,16,16,160,144,80,112,2715,1421,200],'compiled phase ABI')
            callback,records,errors=predecessors(host,bundle,native)
            image=native_image(args.out/'host.so');live=live_regions(data,image)
            chain=json.loads(bundle.read('segments.json'));ranges=segments()
            for name,size in live.items():
                native.require(bytes((ct.c_ubyte*size).in_dll(host,name))==bundle.read(
                    chain['boundaries'][0]['regions'][name]),'actual initial full boundary '+name)
            boundary_checks=[];boundary_errors=[]
            observer_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint)
            @observer_type
            def observe(index):
                slots=(ct.c_uint32*2715).in_dll(host,'smol_full_cycles')
                saved=bytes(slots)
                try:
                    native.require(index==len(boundary_checks),'full canonical boundary order')
                    native.require(all(slots[:ranges[index]['launch_end']])
                        and not any(slots[ranges[index]['launch_end']:]),'full actual cycle history')
                    # The original continuous selfcheck sees every actual
                    # observation unchanged. Only this reference comparison
                    # projects timing bytes to the declared canonical zero.
                    ct.memset(ct.addressof(slots),0,ct.sizeof(slots))
                    for name,size in live.items():
                        actual=(ct.c_ubyte*size).in_dll(host,name)
                        expected=bundle.read(chain['boundaries'][index+1]['regions'][name])
                        native.require(bytes(actual)==expected,'full canonical boundary '+name)
                    boundary_checks.append(index);return 0
                except BaseException as error:
                    boundary_errors.append(str(error));return 1
                finally: ct.memmove(ct.addressof(slots),saved,len(saved))
            host.smol_full_driver_observe.argtypes=[observer_type]
            host.smol_full_driver_observe(observe)
            result=native.execute(args,bundle,data)
            native.require(not errors and len(records)==1421,'all original predecessors')
            native.require(not boundary_errors and len(boundary_checks)==200,'full boundary comparison')
            result.update(status='PASS_NATIVE_DUAL_DRIVER_FULL',predecessors=records,
                full_canonical_boundaries_exact=200,full_initial_boundary_exact=True,
                compiled_phase_layout=compiled_layout,full_cycle_observations_restored=True)
            (args.out/'driver-result.json').write_text(json.dumps(result,indent=2)+'\n')
        else: execute_phases(args,native,bundle,data)
        return
    native.validate_static(args,bundle)
    compile_control(args,native,bundle,data)
    command=['taskset','-c',args.execution_cpus,sys.executable,'-B',str(Path(__file__).absolute()),*sys.argv[1:],'--execute']
    with (args.out/'process-driver.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
    native.require(result.returncode==0,'native driver control failed; see process-driver.log')
    native.bookends(json.loads((args.out/'compile.json').read_text())['source_sha256']);bundle.verify_bookends()

if __name__=='__main__': main()
