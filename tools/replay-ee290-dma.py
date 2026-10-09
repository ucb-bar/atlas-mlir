#!/usr/bin/env python3
"""Finite selected EE290 AtlasCore DMA replay; no system qualification."""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

def bootstrap(path):
    from collections import deque
    import os
    import stat
    path = Path(path)
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
    if not path.is_absolute():
        path = Path.cwd() / path
    if denied(path):
        raise ValueError("restricted dependency path")
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".": continue
        if part == "..": current = current.parent; continue
        candidate = current / part
        if denied(candidate): raise ValueError("restricted dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64: raise ValueError("recursive dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target): raise ValueError("restricted dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else: pending.extendleft(reversed(target.parts))
        else: current = candidate
    return current


def load(name):
    path = bootstrap(Path(__file__).parent / name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBS = load("observe-ee290-vls.py")
CAPTURE = load("ee290_build_capture.py")
FINGERPRINT = load("fingerprint-rtl-modules.py")
CHECK = OBS.CHECK



def program(ops):
    lines=['module {','  %s0 = "atlas.start"() : () -> !atlas.state']
    for n,(op,attrs) in enumerate(ops,1):
        lines.append(f'  %s{n} = "atlas.{op}"(%s{n-1}) {{{attrs}}} : (!atlas.state) -> !atlas.state')
    return "\n".join(lines+['}',''])

def add(dst,value,src=0):
    return ('alu_imm',f'kind = "addi", dst = {dst} : i32, src = {src} : i32, immediate = {value} : i32')
def upper(dst,value):
    return ('upper',f'kind = "lui", dst = {dst} : i32, immediate = {value} : i32')
def config(): return ('dma_config','channel = 0 : i32, base_reg = 5 : i32')
def dma(direction,channel):
    return ('dma',f'direction = "{direction}", channel = {channel} : i32, reg = 6 : i32, dram = {1 if direction=="load" else 3} : i32, size = 2 : i32')
def wait(channel): return ('dma_wait',f'channel = {channel} : i32')
def cases():
    end=[add(1,1),('csr','kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32'),('trap','kind = "ecall"')]
    init=[add(5,0),config(),add(6,0),upper(1,589824),add(2,128)]
    store=[upper(3,589824),add(3,1024,3),dma('store',1),wait(1)]
    out={'config_only':(program([add(5,1),config()]+end),0,0)}
    for name,which,pending,repeats in [('capture_after_launch',0,False,1),('before_launch_control',1,False,1),('capture_scalar_and_base_pending',0,True,1),('waited_channel_reuse',0,False,2)]:
        ops=init.copy()
        if which: ops += [upper(1,589825)]
        for n in range(repeats):
            if n: ops += [upper(1,589824)]
            ops += [dma('load',0),upper(1,589825)]
            if pending: ops += [add(6,32),add(2,32),add(5,1),config()]
            ops += [wait(0)]
            if pending: ops += [add(5,0),config(),add(6,0),add(2,128)]
            ops += store
        out[name]=(program(ops+end),which,repeats)
    return out

# A private decoder instance keeps this fixed projection independent of VLS.
DMA_OBS=load('observe-ee290-vls.py')
DMA_SIGNALS={
 'clock':('clock',1),'reset':('reset',1), 'fire':('scalar.s1_fire',1),
 'valid':('scalar.pc_ctrl.io_s1_valid',1),'pc':('scalar.pc_ctrl.io_s1_pc',32),
 'word':('scalar.decoder.io_instr',32),'decoded':('scalar.decoder.io_decoded_dma_cmd',3),
 'channel':('scalar.decoder.io_decoded_funct3',3),'stall':('scalar.stall',1),
 'base':('scalar.dmaBaseReg',32),'rs1':('scalar.regfile.io_rs1_data',32),
 'rs2':('scalar.regfile.io_rs2_data',32),'rd':('scalar.regfile.io_rd_data',32),
 'launch':('scalar.io_dmaCmd_valid',1),'op':('scalar.io_dmaCmd_bits_op',3),
 'vmem':('scalar.io_dmaCmd_bits_vmemAddr',32),'address':('scalar.io_dmaCmd_bits_addr',64),
 'size':('scalar.io_dmaCmd_bits_size',32),'launch_channel':('scalar.io_dmaCmd_bits_channel',3),
 'av':('io_dmaTL_a_valid',1),'ar':('io_dmaTL_a_ready',1),'ao':('io_dmaTL_a_bits_opcode',3),
 'aa':('io_dmaTL_a_bits_address',37),'as':('io_dmaTL_a_bits_source',6),'ad':('io_dmaTL_a_bits_data',256),
 'dv':('io_dmaTL_d_valid',1),'dr':('io_dmaTL_d_ready',1),'ds':('io_dmaTL_d_bits_source',6),'dd':('io_dmaTL_d_bits_data',256),
 'vr':('dma.io_vmemRead_valid',1),'vrg':('dma.io_vmemReadGrant',1),
 'vrb':('dma.io_vmemRead_bits_bankIdx',3),'vra':('dma.io_vmemRead_bits_bankAddr',13),
 'vw':('dma.io_vmemWrite_valid',1),'vwg':('dma.io_vmemWriteGrant',1),
 'vwb':('dma.io_vmemWrite_bits_bankIdx',3),'vwa':('dma.io_vmemWrite_bits_bankAddr',13),'vwd':('dma.io_vmemWrite_bits_data',256),
 'vresp':('dma.io_vmemReadData_valid',1),'vdata':('dma.io_vmemReadData_bits',256),
 'halt':('csrfile.io_csr_halted',1),'ecall':('csrfile.io_csr_set_ecall',1),
 'illegal':('csrfile.io_csr_set_illegal',1),'ebreak':('csrfile.io_csr_set_ebreak',1),
 'csr':('csrfile.io_csr_valid',1),'csr_addr':('csrfile.io_csr_addr',12),
 'csr_op':('csrfile.io_csr_op',3),'csr_data':('csrfile.io_csr_wdata',32),'marker':('csrfile.reg_dbg0',32),
}
for n in range(8):
    DMA_SIGNALS['busy'+str(n)]=('scalar.io_dma_busy_'+str(n),1)
    DMA_SIGNALS['engine'+str(n)]=('dma.io_channelBusy_'+str(n),1)
DMA_OBS.SIGNALS=DMA_SIGNALS

def pattern(which,offset):
    return sum((((53*(offset+n)+79) if which else (37*(offset+n)+11))&255)<<(8*n) for n in range(32))

def validate_dma_edges(samples,words,which,pairs):
    events=[]; instructions=[]; memory=[]; busy=[]; stalls=[]
    regs=[0]*32; base=0; expected_pc=0; pending=None; outstanding={}
    began=False; ended=False; marker=None; prev_busy=None; next_base=None
    active_stall=None; a_hold=None; d_hold=None; last_edge=0
    controls=['fire','valid','stall','launch','av','ar','dv','dr','vr','vrg','vw','vwg','vresp','halt','ecall','illegal','ebreak','csr']+['busy'+str(n) for n in range(8)]+['engine'+str(n) for n in range(8)]
    def reject(msg): raise OBS.ObservationError(msg)
    for sample in samples:
        edge=sample['edge'];last_edge=edge
        def v(key):
            x=sample['values'][key]
            if x is None or type(x)!=int or not 0<=x<(1<<DMA_SIGNALS[key][1]): reject('unknown/out-of-range DMA signal '+key)
            return x
        if v('reset'): continue
        for key in controls: v(key)
        if v('illegal') or v('ebreak'): reject('unexpected scalar termination')
        mask=sum(v('busy'+str(n))<<n for n in range(8))
        engine=sum(v('engine'+str(n))<<n for n in range(8))
        if prev_busy!=(mask,engine): busy.append({'edge':edge,'scalar_mask':mask,'engine_mask':engine});prev_busy=(mask,engine)
        if not began and not v('halt'):
            began=True
            if mask or engine or v('av') or v('dv') or v('vr') or v('vw'): reject('nonquiescent execution entry')
            if v('base')!=0: reject('START did not reset DMA base')
        if next_base is not None:
            if v('base')!=next_base: reject('CONFIG base publication mismatch')
            base=next_base;next_base=None
        if began and not ended and v('base')!=base: reject('unexpected DMA base change')
        if mask and pending is None: reject('busy without pending transfer')
        if pending is not None and mask & ~(1<<pending['channel']): reject('unexpected competing busy channel')
        if engine & ~mask: reject('engine busy omitted from scalar busy')
        if pending is not None and edge>pending['edge'] and pending['d_count']<4 and mask!=(1<<pending['channel']): reject('pending responses lost busy protection')
        decoded_launch=False
        if v('valid') and began and not ended:
            pc=v('pc');word=v('word')
            if pc!=expected_pc or pc>=len(words) or word!=words[pc]: reject('executed instruction differs from selected words')
            opcode=word&127;channel=(word>>12)&7
            cmd=0
            if opcode==0x7b:
                if word>>25 not in (0,1): reject('unsupported DMA transfer encoding')
                cmd=2 if word>>25 else 1
            elif opcode==0x7f:
                if word>>25 not in (0,1): reject('unsupported DMA control encoding')
                cmd=4 if word>>25 else 3
            if v('decoded')!=cmd or (cmd and v('channel')!=channel): reject('independent DMA instruction decode mismatch')
            decoded_launch=cmd in (1,2) and bool(v('fire'))
            if cmd==4:
                if pending is None or channel!=pending['channel']: reject('wait does not match pending transfer')
                if mask:
                    if v('fire') or not v('stall'): reject('WAIT failed to stall dynamically busy transfer')
                    if active_stall is None: active_stall={'channel':channel,'first':edge,'word_index':pc}
            if v('fire'):
                if v('stall'): reject('stalled instruction fired')
                instructions.append({'edge':edge,'word_index':pc,'word_u32':word})
                expected_pc+=1
                rd=(word>>7)&31;rs1=(word>>15)&31;rs2=(word>>20)&31
                if cmd in (1,2):
                    if pending is not None or not v('launch'): reject('overlapping or absent DMA launch')
                    addr=(base<<32)|regs[rs1 if cmd==1 else rd]
                    vm=regs[rd if cmd==1 else rs1];size=regs[rs2]
                    if (v('address'),v('vmem'),v('size'),v('launch_channel'),v('op'))!=(addr,vm,size,channel,cmd): reject('launch operands do not match captured scalar values')
                    if base>31 or addr&31 or vm&7 or vm>=524288 or size!=128: reject('outside finite aligned DMA domain')
                    expected_addr=(0x90001000 if which else 0x90000000) if cmd==1 else 0x90000400
                    if addr!=expected_addr or vm!=0: reject('case launch address differs')
                    pending={'kind':'launch','edge':edge,'channel':channel,'op':'load' if cmd==1 else 'store','vmem_word':vm,'dram_address':addr,'size':size,'word_index':pc,'word_u32':word,'a_count':0,'d_count':0,'vmem_count':0,'response_count':0}
                    events.append({k:x for k,x in pending.items() if k not in ['a_count','d_count','vmem_count','response_count']})
                elif cmd==3:
                    if v('launch') or v('rs1')!=regs[rs1] or regs[rs1]>31: reject('CONFIG launched transfer or scalar base mismatch')
                    next_base=regs[rs1]
                    events.append({'kind':'config','edge':edge,'base':next_base,'word_index':pc,'word_u32':word,'pending':pending is not None})
                elif cmd==4:
                    if mask or outstanding or pending['a_count']!=4 or pending['d_count']!=4 or pending['vmem_count']!=4 or (pending['op']=='store' and pending['response_count']!=4): reject('WAIT retired before transfer/VMEM drain')
                    if active_stall is None: reject('case lacks observed dynamic WAIT stall')
                    active_stall['last']=edge-1;stalls.append(active_stall);active_stall=None
                    events.append({'kind':'wait','edge':edge,'channel':channel,'word_index':pc,'word_u32':word});pending=None
                elif opcode==0x13 and (word>>12)&7==0:
                    imm=word>>20;imm=imm-4096 if imm&2048 else imm
                    if rd: regs[rd]=(regs[rs1]+imm)&0xffffffff
                elif opcode==0x37:
                    if rd: regs[rd]=word&0xfffff000
                elif opcode!=0x73: reject('unsupported scalar instruction in finite replay')
        if bool(v('launch'))!=decoded_launch: reject('launch does not match fired decoded DMA instruction')
        if v('av'):
            record=(v('ao'),v('aa'),v('as'),v('ad'))
            if a_hold is not None and record!=a_hold: reject('A request changed under backpressure')
            if not v('ar'): a_hold=record
            else:
                a_hold=None
                if pending is None: reject('A request without launch')
                op,address,source,data=record
                if op!=(4 if pending['op']=='load' else 0) or address!=pending['dram_address']+32*pending['a_count'] or source in outstanding or pending['a_count']>=4: reject('A beat/address/source mismatch')
                if op==0 and data!=pattern(which,32*pending['a_count']): reject('store data mismatch')
                outstanding[source]={'edge':edge,'op':op,'offset':32*pending['a_count']}
                pending['a_count']+=1
                events.append({'kind':'a','edge':edge,'address':address,'op':op,'source':source,'data_hex':f'{data:064x}'})
        elif a_hold is not None: reject('A valid dropped under backpressure')
        if v('dv'):
            record=(v('ds'),v('dd'))
            if d_hold is not None and record!=d_hold: reject('D response changed under backpressure')
            if not v('dr'): d_hold=record
            else:
                d_hold=None
                source,data=record
                if pending is None or source not in outstanding: reject('unexpected D response')
                request=outstanding.pop(source)
                if edge-request['edge']<43: reject('response was not delayed by fixture contract')
                if request['op']==4 and data!=pattern(which,request['offset']): reject('load response data mismatch')
                pending['d_count']+=1
                events.append({'kind':'d','edge':edge,'source':source,'data_hex':f'{data:064x}'})
        elif d_hold is not None: reject('D valid dropped under backpressure')
        for port,grant,bank,row,data in [('vw','vwg','vwb','vwa','vwd'),('vr','vrg','vrb','vra',None)]:
            if v(port):
                v(bank);v(row)
                if data: v(data)
                if pending is None or (port=='vw')!=(pending['op']=='load'): reject('unbound VMEM request')
            if v(port) and v(grant):
                if pending is None or (port=='vw')!=(pending['op']=='load'): reject('unexpected VMEM direction/request')
                n=pending['vmem_count']
                if n>=4 or v(bank)!=0 or v(row)!=n: reject('VMEM bank/row mismatch')
                if data and v(data)!=pattern(which,32*n): reject('VMEM load write data mismatch')
                memory.append({'kind':port,'edge':edge,'bank':v(bank),'row':v(row)})
                pending['vmem_count']+=1
        if v('vresp'):
            if pending is None or pending['op']!='store' or pending['response_count']>=pending['vmem_count'] or v('vdata')!=pattern(which,32*pending['response_count']): reject('VMEM read response mismatch')
            memory.append({'kind':'response','edge':edge,'data_hex':f'{v("vdata"):064x}'})
            pending['response_count']+=1
        if v('csr'):
            if pending is not None or outstanding or (v('csr_addr'),v('csr_op'),v('csr_data'))!=(0xc10,1,1) or marker is not None: reject('marker before drain or unsupported CSR')
            marker=edge;events.append({'kind':'marker','edge':edge})
        if v('ecall'):
            if ended or not began or pending is not None or outstanding or marker is None or v('marker')!=1 or not v('halt') or v('fire') or expected_pc!=len(words)-1 or v('word')!=words[-1]: reject('incomplete terminal drain')
            events.append({'kind':'halt','edge':edge});ended=True
        if began and not ended and v('halt') and not v('ecall'): reject('unexpected halt')
    if not ended or pending or outstanding or active_stall or next_base is not None or len([x for x in events if x['kind']=='launch'])!=2*pairs: reject('incomplete DMA transcript')
    return {'events':events,'instructions':instructions,'busy_transitions':busy,'wait_stalls':stalls,'memory_events':memory,'sampled_edges':last_edge}

def analyze_dma(trace,words,which,pairs):
    with CHECK.allowed(trace).open() as stream:
        return validate_dma_edges(DMA_OBS.vcd_edges(stream,'TOP.AtlasCore'),words,which,pairs)


def run(args):
    checker=CHECK.Checker()
    report_id=checker.identity(args.selected_core_replay)
    if report_id['sha256']!=args.expected_replay_sha256: raise ValueError('selected replay hash mismatch')
    selected=CHECK.strict_json(CHECK.allowed(args.selected_core_replay).read_bytes())
    if selected['schema']!='atlas.selected_atlascore_replay.v0' or selected['state']!='numerical_and_boundary_replay_passed': raise ValueError('successful selected replay required')
    original=selected['original_inputs']
    witness=CHECK.strict_json(CHECK.allowed(original[0]['path']).read_bytes())
    checker.verify(original[0],args.selected_core_replay.parent)
    # Historical producer sources may evolve; bind executed retained RTL snapshots,
    # not current copies of unrelated historical producers.
    provenance=witness['provenance']
    manifest=checker.verify(provenance['manifest'],args.selected_core_replay.parent)
    hardware=checker.verify(provenance['hardware_ir'],args.selected_core_replay.parent)
    output=OBS._fresh_output(args.output); output.mkdir(parents=True)
    inputs=output/'inputs'; inputs.mkdir()
    rtl=[]
    for item in selected['rtl_snapshots']:
        source=checker.verify(item['snapshot'],args.selected_core_replay.parent)
        dest=inputs/Path(source['path']).name
        dest.write_bytes(CHECK.allowed(source['path']).read_bytes())
        snap=checker.identity(dest)
        if (snap['sha256'],snap['bytes'])!=(source['sha256'],source['bytes']): raise ValueError('snapshot content mismatch')
        rtl.append({'original':source,'snapshot':snap})
    root=CHECK.allowed(Path(__file__).parent.parent)
    harness=checker.identity(root/'test/ee290-dma-replay.cpp')
    copy=inputs/'ee290-dma-replay.cpp';copy.write_bytes(CHECK.allowed(harness['path']).read_bytes())
    dependencies=[]
    for name in ['replay-ee290-dma.py','observe-ee290-vls.py','ee290_build_capture.py','check-ee290-provenance.py','fingerprint-rtl-modules.py','index-retained-hw.py']:
        src=checker.identity(root/'tools'/name);dest=inputs/name;dest.write_bytes(CHECK.allowed(src['path']).read_bytes());dependencies.append({'original':src,'snapshot':checker.identity(dest)})
    tools={name:checker.identity(getattr(args,name)) for name in ('verilator','cxx','make','ar','atlas_emit')}
    runtime=CHECK.allowed(args.verilator_root)
    env={'PATH':str(CHECK.allowed(args.make).parent)+':/bin','LC_ALL':'C','LANG':'C','VERILATOR_ROOT':str(runtime),'CXX':tools['cxx']['path'],'AR':tools['ar']['path']}
    argv=[tools['verilator']['path'],'--cc','--exe','--build','--top-module','AtlasCore','--prefix','VAtlasCore','--Mdir','{output}/obj','--trace','--trace-depth','3','--assert','--output-split','20000','--output-split-cfuncs','500','-Wno-fatal','-j',str(args.jobs),'-CFLAGS','-std=c++17','-MAKEFLAGS','CXX='+tools['cxx']['path']+' LINK='+tools['cxx']['path']+' AR='+tools['ar']['path'],*[r['snapshot']['path'] for r in rtl],str(copy)]
    checker.recheck()
    build_inputs=[{'role':'tool' if n=='verilator' else n,'identity':i} for n,i in tools.items() if n!='atlas_emit']+[{'role':'selected_rtl','identity':r['snapshot']} for r in rtl]+[{'role':'harness','identity':checker.identity(copy)}]
    if args.reuse_compile:
        phase_id=checker.identity(args.reuse_compile)
        phase=CHECK.strict_json(CHECK.allowed(args.reuse_compile).read_bytes())
        if phase['state']!='phase_completed' or phase['kind']!='dma_verilator_compile' or phase['inputs_before']!=phase['inputs_after']: raise ValueError('invalid reused compile')
        contents=lambda members: sorted((x['role'],x['identity']['sha256'],x['identity']['bytes']) for x in members)
        if contents(phase['inputs_before'])!=contents(build_inputs): raise ValueError('reused compile inputs differ')
        for x in phase['inputs_before']: checker.verify(x['identity'],args.reuse_compile.parent)
        def normalized(command,members,phase_output):
            aliases={x['identity']['path']:'INPUT:'+x['role']+':'+x['identity']['sha256'] for x in members}
            return [aliases.get(a,a.replace(phase_output,'{output}')) for a in command]
        if normalized(phase['command']['argv'],phase['inputs_before'],str(args.reuse_compile.parent.resolve()))!=normalized(argv,build_inputs,'{output}') or phase['command']['environment']!=env or phase['command']['returncode']!=0 or phase['command']['timed_out']: raise ValueError('reused compilation recipe differs')
        binary=checker.identity(args.reuse_compile.parent/'obj/VAtlasCore')
        if binary not in [f['identity'] for o in phase['outputs'] for f in o['files']]: raise ValueError('reused model not compile output')
    else:
        phase=CAPTURE.capture_phase(output/'compile','dma_verilator_compile',argv,build_inputs,[{'role':'model','relative_path':'obj','kind':'tree'}],env,args.timeout_seconds)
        if phase['state']!='phase_completed': raise ValueError('DMA compile failed')
        phase_id=checker.identity(output/'compile/phase.json')
        binary=checker.identity(output/'compile/obj/VAtlasCore')
    receipt={'schema':'atlas.ee290_dma_replay.v0','target_config':'EE290SimConfig','state':'prepared','qualification':'conditional','scheduling_qualified':False,'selected_core_replay':report_id,'manifest':manifest,'hardware_ir':hardware,'rtl_snapshots':rtl,'harness':{'original':harness,'snapshot':checker.identity(copy)},'producer_snapshots':dependencies,'tool_identities':tools,'compile_phase':phase_id,'model':binary,'cases':[],'assumptions':['Fresh START with all DMA queues quiescent.','Globally serialized transfers with matching waits; aligned 128-byte finite operands.','Direct host TileLink and deterministic delayed external DMA slave; behavioral SRAM; not CPU/SoC qualification.','Opaque installed Verilator and C++ runtime; two-state clocked simulation.']}
    (output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name,(text,which,count) in cases().items():
        directory=output/name;directory.mkdir()
        source=directory/'program.mlir';source.write_text(text)
        source_id=checker.identity(source)
        python=checker.identity(sys.executable)
        child="import subprocess,sys; subprocess.run([sys.argv[1],sys.argv[2]],stdout=open(sys.argv[3],'wb'),check=True)"
        emission=CAPTURE.capture_phase(directory/'emit','dma_program_emit',[python['path'],'-c',child,tools['atlas_emit']['path'],str(source),'{output}/program.hex'], [{'role':'tool','identity':python},{'role':'atlas_emit','identity':tools['atlas_emit']},{'role':'program','identity':source_id}],[{'role':'emitted_words','relative_path':'program.hex','kind':'file'}],env,args.timeout_seconds)
        if emission['state']!='phase_completed': raise ValueError('DMA emission failed')
        emitted=CHECK.allowed(directory/'emit/program.hex').read_text()
        if not re.fullmatch(r'(?:[0-9a-fA-F]{8}\s+)+',emitted): raise ValueError('invalid emitted words')
        words=directory/'program.hex';words.write_text(emitted);words_id=checker.identity(words)
        execution=CAPTURE.capture_phase(directory/'execution','dma_execution',[binary['path'],str(words),'{output}/trace.vcd','100000',''+str(which),str(count)],[{'role':'tool','identity':binary},{'role':'words','identity':words_id}],[{'role':'trace','relative_path':'trace.vcd','kind':'file'}],env,args.timeout_seconds)
        if execution['state']!='phase_completed': raise ValueError('DMA execution failed '+name)
        log=CHECK.allowed(directory/'execution/command.stdout.log').read_text()
        match=re.search(r'EE290_DMA_PASSED reads=(\d+) writes=(\d+) a_stalls=(\d+) delayed_response_cycles=(\d+) status=5 marker=1 illegal_pc=0 cycles=(\d+)',log)
        if not match or int(match[1])!=4*count or int(match[2])!=4*count: raise ValueError('DMA numerical result mismatch')
        trace_id=checker.identity(directory/'execution/trace.vcd')
        observations=analyze_dma(Path(trace_id['path']),[int(w,16) for w in emitted.split()],which,count)
        transcript={'schema':'atlas.ee290_dma_events.v0','trace':trace_id,'words':words_id,**observations}
        events_path=directory/'events.json';events_path.write_text(json.dumps(transcript,indent=2)+'\n')
        receipt['cases'].append({'name':name,'events':checker.identity(events_path),'boundary_checks':{'config_not_transfer':True,'launch_operands_captured':bool(count),'matching_wait_observed':bool(count),'delayed_progress_observed':True if count else False},'expected_pattern':which,'transfer_pairs':count,'program':source_id,'words':words_id,'emission_phase':checker.identity(directory/'emit/phase.json'),'execution_phase':checker.identity(directory/'execution/phase.json'),'trace':checker.identity(directory/'execution/trace.vcd'),'numerical_checks':{'completion':True,'captured_source':True,'output':True,'guards':True},'observations':dict(zip(['reads','writes','a_stalls','delayed_response_cycles','cycles'],map(int,match.groups())))})
    checker.recheck()
    receipt['state']='finite_dma_cases_passed'
    (output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('selected-core-replay','output','verilator','cxx','make','ar','atlas-emit','verilator-root'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-replay-sha256',required=True)
    p.add_argument('--reuse-compile',type=Path)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--timeout-seconds',type=int,default=1200)
    try: print(json.dumps(run(p.parse_args()),indent=2))
    except (ValueError,KeyError,OBS.ObservationError) as error: p.exit(1,str(error)+'\n')
