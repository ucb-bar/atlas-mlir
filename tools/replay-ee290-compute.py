#!/usr/bin/env python3
"""Selected AtlasCore compute/composition replay, conditional finite evidence only."""
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


DMA=load('replay-ee290-dma.py')
OBS,CAPTURE,CHECK=DMA.OBS,DMA.CAPTURE,DMA.CHECK
BIND=load('check-ee290-compute-export.py')

def vls(direction,mreg,base):
    return (direction,f'{"dst" if direction=="vload" else "src"} = {mreg} : i32, base = {base} : i32, offset = 0 : i32, format = "raw"')
def delay(n): return ('delay',f'cycles = {n} : i32')
def xlu(): return ('xlu_transpose','dst = 1 : i32, src = 0 : i32')
def cases(omit_delays=False):
    end=[DMA.add(1,1),('csr','kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32'),('trap','kind = "ecall"')]
    transpose=[DMA.add(6,0),vls('vload',0,6),delay(40),xlu(),delay(80),DMA.add(8,256),vls('vstore',1,8),delay(40),DMA.add(0,0)]
    mul=[]
    for bank,words in enumerate([0,256,512,768]): mul += [DMA.add(6,words),vls('vload',bank,6),delay(40)]
    mul += [('vpu_binary','kind = "mul", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'),delay(80)]
    for bank,words in [(4,1024),(5,1280)]: mul += [DMA.add(8,words),vls('vstore',bank,8),delay(40)]
    mixed=[DMA.add(5,0),DMA.config(),DMA.add(6,0),DMA.upper(1,589824),DMA.add(2,128),DMA.dma('load',0),DMA.wait(0)]
    mixed += transpose+[DMA.upper(3,589824),DMA.add(3,1024,3),('dma','direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32'),DMA.wait(1)]
    return {name:(DMA.program([op for op in ops+end if not (omit_delays and op[0]=='delay')]),name) for name,ops in [('xlu',transpose),('vmul',mul),('dma_xlu',mixed)]}

COMPUTE_OBS=DMA.load('observe-ee290-vls.py')
COMPUTE_SIGNALS=OBS.SIGNALS.copy()
for key,path,width in [('pc','scalar.pc_ctrl.io_s1_pc',32),('valid','scalar.pc_ctrl.io_s1_valid',1),('word','scalar.decoder.io_instr',32),
 ('xcmd','xlu.io_cmd_valid',1),('xsrc','xlu.io_cmd_bits_srcMregId',6),('xdst','xlu.io_cmd_bits_dstMregId',6),
 ('xr','xlu.io_mregReadReq_valid',1),('xrid','xlu.io_mregReadReq_bits_mregId',6),('xrrow','xlu.io_mregReadReq_bits_row',5),
 ('xresp','xlu.io_mregReadResp_valid',1),('xdata','xlu.io_mregReadResp_bits',256),
 ('xw','xlu.io_mregWriteReq_valid',1),('xwid','xlu.io_mregWriteReq_bits_mregId',6),('xwrow','xlu.io_mregWriteReq_bits_row',5),('xwdata','xlu.io_mregWriteReq_bits_data',256),
 ('xar','xlu.io_activeMregRead_valid',1),('xarid','xlu.io_activeMregRead_bits',6),('xaw','xlu.io_activeMregWrite_valid',1),('xawid','xlu.io_activeMregWrite_bits',6),
 ('pcmd','vpu.io_cmd_valid',1),('pop','vpu.io_cmd_bits_op',5),('pdst','vpu.io_cmd_bits_vd',6),('psrc0','vpu.io_cmd_bits_vs1',6),('psrc1','vpu.io_cmd_bits_vs2',6)]:
    COMPUTE_SIGNALS[key]=(path,width)
for n in range(2):
    for key,path,width in [('r','io_mregReadReq'+str(n)+'_valid',1),('rid','io_mregReadReq'+str(n)+'_bits_mregId',6),('rrow','io_mregReadReq'+str(n)+'_bits_row',5),
                           ('resp','io_mregReadResp'+str(n)+'_valid',1),('data','io_mregReadResp'+str(n)+'_bits',256),
                           ('w','io_mregWriteReq'+str(n)+'_valid',1),('wid','io_mregWriteReq'+str(n)+'_bits_mregId',6),('wrow','io_mregWriteReq'+str(n)+'_bits_row',5),('wdata','io_mregWriteReq'+str(n)+'_bits_data',256)]:
        COMPUTE_SIGNALS['p'+key+str(n)]=('vpu.'+path,width)
for n in range(4):
    COMPUTE_SIGNALS['par'+str(n)]=('vpu.io_activeReads_'+str(n)+'_valid',1)
    COMPUTE_SIGNALS['parid'+str(n)]=('vpu.io_activeReads_'+str(n)+'_bits',6)
    COMPUTE_SIGNALS['paw'+str(n)]=('vpu.io_activeWrites_'+str(n)+'_valid',1)
    COMPUTE_SIGNALS['pawid'+str(n)]=('vpu.io_activeWrites_'+str(n)+'_bits',6)
DMA_KEYS={}
for key,(path,width) in DMA.DMA_SIGNALS.items():
    matches=[k for k,(p,w) in COMPUTE_SIGNALS.items() if p==path]
    if matches: DMA_KEYS[key]=matches[0]
    else:
        DMA_KEYS[key]='dma_'+key;COMPUTE_SIGNALS['dma_'+key]=(path,width)
COMPUTE_OBS.SIGNALS=COMPUTE_SIGNALS

def initial_memory(mode):
    result=bytearray((0xa5^(13*i))&255 for i in range(8192))
    if mode=='vmul':
        for i in range(1024):
            for rhs,start in [(False,0),(True,2048)]:
                exponent=i%3-1 if rhs else i%5-2
                negative=i%11==0 if rhs else i%7==0
                code=(0x8000 if negative else 0)|((127+exponent)<<7)
                result[start+2*i:start+2*i+2]=code.to_bytes(2,'little')
    elif mode in ('xlu','dma_xlu'):
        for i in range(1024): result[i]=(37*i+11+3*(i//32))&255
    else: raise OBS.ObservationError('unsupported finite compute mode')
    return result

def bf16_power_product(lhs,rhs):
    result=bytearray()
    for offset in range(0,32,2):
        a=int.from_bytes(lhs[offset:offset+2],'little');b=int.from_bytes(rhs[offset:offset+2],'little')
        ea=(a>>7)&255;eb=(b>>7)&255
        if a&127 or b&127 or not 1<=ea<=254 or not 1<=eb<=254:
            raise OBS.ObservationError('outside exact finite-normal BF16 power reference')
        e=ea+eb-127
        if not 1<=e<=254: raise OBS.ObservationError('BF16 power product outside normal domain')
        result+=(((a^b)&0x8000)|(e<<7)).to_bytes(2,'little')
    return bytes(result)

def quiescent_state(v,d,busy,engine):
    return {'vls_load_busy':v('load_busy'),'vls_store_busy':v('store_busy'),
            'xlu_read_active':v('xar'),'xlu_write_active':v('xaw'),
            'vpu_read_mask':sum(v('par'+str(n))<<n for n in range(4)),
            'vpu_write_mask':sum(v('paw'+str(n))<<n for n in range(4)),
            'dma_scalar_mask':busy,'dma_engine_mask':engine,
            'dma_a_valid':d('av'),'dma_d_valid':d('dv')}

def validate_compute_edges(samples,words,mode):
    vm=initial_memory(mode);expected_vm=vm.copy();mreg={}
    if mode=='vmul':
        for offset in range(0,2048,32): expected_vm[4096+offset:4096+offset+32]=bf16_power_product(vm[offset:offset+32],vm[2048+offset:2048+offset+32])
    else:
        if mode=='dma_xlu': expected_vm[:128]=bytes((37*i+11)&255 for i in range(128))
        source=bytes(expected_vm[:1024])
        expected_vm[1024:2048]=bytes(source[32*c+r] for r in range(32) for c in range(32))
    commands=[];instructions=[];endpoints=[];busy_events=[]
    regs=[None]*32;regs[0]=0;base=0;next_base=None;pc_expected=0
    active=None;pending=None;outstanding={};read_queue=[];a_hold=None;d_hold=None
    began=False;ended=False;marker=None;publication=None;last_edge=0;prev_busy=None
    controls=list(OBS.CONTROLS)+['valid','xcmd','xr','xresp','xw','xar','xaw','pcmd']
    controls+=['pr'+str(p) for p in range(2)]+['presp'+str(p) for p in range(2)]+['pw'+str(p) for p in range(2)]
    controls+=['par'+str(p) for p in range(4)]+['paw'+str(p) for p in range(4)]
    controls+=list(dict.fromkeys(DMA_KEYS[k] for k in ['launch','stall','av','ar','dv','dr','vr','vrg','vw','vwg','vresp']+['busy'+str(p) for p in range(8)]+['engine'+str(p) for p in range(8)]))
    def reject(message): raise OBS.ObservationError(message)
    def row_bytes(identity,row):
        if (identity,row) not in mreg: reject('read from uninitialized logical MREG row')
        return mreg[identity,row]
    for sample in samples:
        edge=sample['edge'];last_edge=edge
        def v(key):
            value=sample['values'][key]
            if type(value)!=int or not 0<=value<(1<<COMPUTE_SIGNALS[key][1]): reject('unknown/out-of-range compute signal '+key)
            return value
        def d(key): return v(DMA_KEYS[key])
        def data(key): return v(key).to_bytes(32,'little')
        def record(collection,**fields): collection.append({'edge':edge,**fields})
        if v('reset'):
            if began: reject('reset interrupted compute envelope')
            continue
        for key in controls: v(key)
        if v('illegal') or v('ebreak'): reject('unexpected scalar termination')
        busy=sum(d('busy'+str(p))<<p for p in range(8));engine=sum(d('engine'+str(p))<<p for p in range(8))
        if prev_busy!=(busy,engine): record(busy_events,scalar_mask=busy,engine_mask=engine);prev_busy=(busy,engine)
        if not began and not v('halt'):
            began=True
            if busy or engine or active or pending or d('av') or d('dv') or v('cmd') or v('xcmd') or v('pcmd'): reject('nonquiescent observed execution entry')
            if d('base')!=0: reject('START did not reset DMA base')
        if next_base is not None:
            if d('base')!=next_base: reject('CONFIG publication differs from captured scalar base')
            base=next_base;next_base=None
        if began and not ended and d('base')!=base: reject('unexpected global DMA base change')
        if engine&~busy: reject('engine busy omitted from scalar bridge')
        if pending is None and busy: reject('busy without observed pending DMA')
        if pending and (busy&~(1<<pending['channel']) or (edge>pending['edge'] and len(pending['responses'])<4 and busy!=(1<<pending['channel']))): reject('pending DMA lost busy protection')
        if active and edge==active['edge']+active['release_age']:
            active['release_edge']=edge;active=None
        if active and edge>active['edge']+active['release_age']: reject('missing observed compute release')
        expected_cmd={'cmd':False,'xcmd':False,'pcmd':False,DMA_KEYS['launch']:False}
        if v('valid') and began and not ended:
            pc=v('pc');word=v('word')
            if pc!=pc_expected or pc>=len(words) or word!=words[pc]: reject('retired/held instruction differs from selected program words')
            if v('fire'):
                record(instructions,word_index=pc,word_u32=word);pc_expected+=1
                opcode=word&127;rd=(word>>7)&31;rs1=(word>>15)&31;rs2=(word>>20)&31
                common={'edge':edge,'word_index':pc,'word_u32':word,'reads':[],'responses':[],'writes':[]}
                if opcode==0x13 and (word>>12)&7==0:
                    immediate=word>>20;immediate=immediate-4096 if immediate&2048 else immediate
                    if regs[rs1] is None: reject('unknown scalar ADDI operand')
                    if rd: regs[rd]=(regs[rs1]+immediate)&0xffffffff
                elif opcode==0x37:
                    if rd: regs[rd]=word&0xfffff000
                elif opcode==0x67 and (word>>12)&7==1:
                    pass # DELAY is independently bound through its actual held PC/word.
                elif opcode in (0x07,0x6b,0x57):
                    if active or pending: reject('compute command precedes previous observed drain')
                    if opcode==0x07:
                        store=bool(word&(1<<13));identity=(word>>7)&63
                        immediate=word>>20;immediate=immediate-4096 if immediate&2048 else immediate
                        if regs[rs1] is None: reject('unknown VLS scalar base')
                        line=((regs[rs1]+32*immediate)&0xffffffff)>>3 &65535
                        if line%32 or line+32>len(vm)//32: reject('VLS tile outside finite initialized memory')
                        required=(2 if store else 1,identity,line)
                        if (v('op'),v('mreg'),v('line'))!=required: reject('VLS command does not match decoded instruction operands')
                        rows=[row_bytes(identity,r) for r in range(32)] if store else [bytes(vm[32*(line+r):32*(line+r+1)]) for r in range(32)]
                        active={**common,'engine':'vls','op':'store' if store else 'load','mreg':identity,'line':line,'release_age':35,'reference_rows':rows}
                        expected_cmd['cmd']=True
                    elif opcode==0x6b:
                        src=(word>>13)&63;dst=(word>>7)&63
                        if (v('xsrc'),v('xdst'))!=(src,dst): reject('XLU command differs from selected operands')
                        source=b''.join(row_bytes(src,r) for r in range(32))
                        output=bytes(source[32*c+r] for r in range(32) for c in range(32))
                        active={**common,'engine':'xlu','src':src,'dst':dst,'release_age':66,'source_rows':[source[32*r:32*r+32] for r in range(32)],'reference_rows':[output[32*r:32*r+32] for r in range(32)]}
                        expected_cmd['xcmd']=True
                    else:
                        op=word>>25;lhs=(word>>13)&63;rhs=(word>>19)&63;dst=(word>>7)&63
                        if op!=3 or (v('pop'),v('psrc0'),v('psrc1'),v('pdst'))!=(op,lhs,rhs,dst) or any(i%2 for i in (lhs,rhs,dst)): reject('unsupported/mismatched paired VMUL command')
                        sources=[[row_bytes(identity+k//32,k%32) for k in range(64)] for identity in (lhs,rhs)]
                        active={**common,'engine':'vmul','op':'mul.bf16','lhs':lhs,'rhs':rhs,'dst':dst,'release_age':66,'source_rows':sources,'reference_rows':[bf16_power_product(sources[0][k],sources[1][k]) for k in range(64)]}
                        expected_cmd['pcmd']=True
                    commands.append(active)
                elif opcode==0x7b:
                    store=bool(word>>25);channel=(word>>12)&7
                    vmreg=rs1 if store else rd;addressreg=rd if store else rs1
                    if active or pending or any(regs[r] is None for r in (vmreg,addressreg,rs2)): reject('overlapping/unknown DMA launch')
                    address=(base<<32)|regs[addressreg];vmword=regs[vmreg];size=regs[rs2]
                    if (d('address'),d('vmem'),d('size'),d('launch_channel'),d('op'))!=(address,vmword,size,channel,2 if store else 1): reject('DMA launch differs from captured scalar operands')
                    expected_addr=0x90000400 if store else 0x90000000
                    if mode!='dma_xlu' or size!=128 or vmword!=(256 if store else 0) or address!=expected_addr: reject('DMA launch outside finite composition case')
                    rows=[bytes(vm[4*vmword+32*r:4*vmword+32*(r+1)]) for r in range(4)] if store else [DMA.pattern(0,32*r).to_bytes(32,'little') for r in range(4)]
                    pending={**common,'engine':'dma','op':'store' if store else 'load','channel':channel,'vmem_word':vmword,'dram_address':address,'size':size,'reference_rows':rows,'requests':[],'memory':[],'memory_responses':[],'wait_stall_edges':[]}
                    commands.append(pending);expected_cmd[DMA_KEYS['launch']]=True
                elif opcode==0x7f:
                    channel=(word>>12)&7
                    if word>>25:
                        if pending is None or channel!=pending['channel'] or busy or outstanding or read_queue or len(pending['requests'])!=4 or len(pending['responses'])!=4 or len(pending['memory'])!=4 or (pending['op']=='store' and len(pending['memory_responses'])!=4) or not pending['wait_stall_edges']: reject('WAIT precedes matching DMA/data drain')
                        pending['release_edge']=edge;record(endpoints,kind='wait',channel=channel,word_index=pc,word_u32=word);pending=None
                    else:
                        if regs[rs1] is None or regs[rs1]>31 or d('rs1')!=regs[rs1]: reject('unknown/mismatched DMA configuration')
                        next_base=regs[rs1];record(endpoints,kind='config',base=next_base,word_index=pc,word_u32=word)
                elif opcode!=0x73: reject('unsupported scalar instruction in finite compute trace')
            elif (v('word')&127)==0x7f and v('word')>>25 and pending:
                channel=(v('word')>>12)&7
                if channel!=pending['channel'] or not busy or not d('stall'): reject('invalid dynamically held matching WAIT')
                pending['wait_stall_edges'].append(edge)
        for key,expected in expected_cmd.items():
            if bool(v(key))!=expected: reject('unbound or missing decoded command '+key)
        if v('launch')!=v('cmd'): reject('scalar/LSU launch mismatch')
        age=edge-active['edge'] if active else -1
        engine_kind=active['engine'] if active else None
        expected_ports={key:False for key in list(OBS.PORTS)+['xr','xresp','xw','xar','xaw']+['pr'+str(p) for p in range(2)]+['presp'+str(p) for p in range(2)]+['pw'+str(p) for p in range(2)]}
        if engine_kind=='vls':
            store=active['op']=='store';source,response,dest=('mr','mresp','vw') if store else ('vr','vresp','mw')
            expected_ports[source]=1<=age<=32;expected_ports[response]=2<=age<=33;expected_ports[dest]=3<=age<=34
            if expected_ports[source]:
                row=age-1
                actual=(v('mr_id'),v('mr_row')) if store else (v('vr_bank'),v('vr_addr'))
                required=(active['mreg'],row) if store else (active['line']>>13,(active['line']&8191)+row)
                if actual!=required: reject('VLS source bank/row mismatch')
                record(active['reads'],id=active['mreg'] if store else active['line']+row,row=row,data_hex=active['reference_rows'][row].hex())
            if expected_ports[response]:
                row=age-2;observed=data(response+'_data')
                if observed!=active['reference_rows'][row]: reject('VLS source response data mismatch')
                record(active['responses'],id=active['mreg'] if store else active['line']+row,row=row,data_hex=observed.hex())
            if expected_ports[dest]:
                row=age-3;observed=data(dest+'_data')
                actual=(v('vw_bank'),v('vw_addr')) if store else (v('mw_id'),v('mw_row'))
                required=(active['line']>>13,(active['line']&8191)+row) if store else (active['mreg'],row)
                if actual!=required or observed!=active['reference_rows'][row]: reject('VLS destination row/data mismatch')
                if store: vm[32*(active['line']+row):32*(active['line']+row+1)]=observed
                else: mreg[active['mreg'],row]=observed
                record(active['writes'],id=active['line']+row if store else active['mreg'],row=row,data_hex=observed.hex())
        if (v('load_busy'),v('store_busy'))!=((int(engine_kind=='vls' and active['op']=='load' and age>0),int(engine_kind=='vls' and active['op']=='store' and age>0))): reject('LSU busy/release mismatch')
        if engine_kind=='xlu':
            expected_ports.update(xr=1<=age<=32,xresp=2<=age<=33,xw=34<=age<=65,xar=1<=age<=33,xaw=1<=age<=65)
            if expected_ports['xar'] and v('xarid')!=active['src'] or expected_ports['xaw'] and v('xawid')!=active['dst']: reject('XLU active operand mismatch')
            for port,first,count,identity,rows,collection in [('xr',1,32,active['src'],active['source_rows'],'reads'),('xresp',2,32,active['src'],active['source_rows'],'responses'),('xw',34,32,active['dst'],active['reference_rows'],'writes')]:
                if expected_ports[port]:
                    row=age-first
                    if port=='xr':
                        if (v('xrid'),v('xrrow'))!=(identity,row): reject('XLU read row/id mismatch')
                        observed=rows[row]
                    else:
                        observed=data('xdata' if port=='xresp' else 'xwdata')
                        if observed!=rows[row]: reject('XLU response/transpose data mismatch')
                        if port=='xw':
                            if (v('xwid'),v('xwrow'))!=(identity,row): reject('XLU write row/id mismatch')
                            mreg[identity,row]=observed
                    record(active[collection],id=identity,row=row,data_hex=observed.hex())
        if engine_kind=='vmul':
            for port in range(2):
                expected_ports['pr'+str(port)]=0<=age<=63
                expected_ports['presp'+str(port)]=1<=age<=64
                if expected_ports['pr'+str(port)]:
                    k=age;identity=(active['lhs'] if port==0 else active['rhs'])+k//32;row=k%32
                    if (v('prid'+str(port)),v('prrow'+str(port)))!=(identity,row): reject('VMUL paired read id/row mismatch')
                    record(active['reads'],port=port,id=identity,row=row,data_hex=active['source_rows'][port][k].hex())
                if expected_ports['presp'+str(port)]:
                    k=age-1;identity=(active['lhs'] if port==0 else active['rhs'])+k//32;row=k%32;observed=data('pdata'+str(port))
                    if observed!=active['source_rows'][port][k]: reject('VMUL paired read response mismatch')
                    record(active['responses'],port=port,id=identity,row=row,data_hex=observed.hex())
            expected_ports['pw0']=2<=age<=65
            if expected_ports['pw0']:
                k=age-2;identity=active['dst']+k//32;row=k%32;observed=data('pwdata0')
                if (v('pwid0'),v('pwrow0'))!=(identity,row) or observed!=active['reference_rows'][k]: reject('VMUL product/id/row mismatch')
                mreg[identity,row]=observed;record(active['writes'],port=0,id=identity,row=row,data_hex=observed.hex())
        for key,expected in expected_ports.items():
            if bool(v(key))!=expected: reject('unexpected/missing compute boundary event '+key)
        # Wrapper live markers need not cover the launch edge's combinational read.
        for port in range(4):
            for direction in ('par','paw'):
                key=direction+str(port)
                if v(key):
                    if engine_kind!='vmul': reject('unbound VPU live resource marker')
                    identity=v(direction+'id'+str(port));allowed=[active['lhs'],active['lhs']+1,active['rhs'],active['rhs']+1] if direction=='par' else [active['dst'],active['dst']+1]
                    if identity not in allowed: reject('VPU live resource operand mismatch')
        if d('av'):
            request=(d('ao'),d('aa'),d('as'),d('ad'))
            if a_hold is not None and request!=a_hold: reject('DMA A changed under backpressure')
            if not d('ar'): a_hold=request
            else:
                a_hold=None
                if pending is None: reject('unbound DMA A request')
                op,address,source,datum=request;n=len(pending['requests'])
                if n>=4 or op!=(0 if pending['op']=='store' else 4) or address!=pending['dram_address']+32*n or source in outstanding: reject('DMA beat/address/source mismatch')
                if op==0 and datum.to_bytes(32,'little')!=pending['reference_rows'][n]: reject('DMA composition store data mismatch')
                outstanding[source]={'edge':edge,'row':n}
                record(pending['requests'],source=source,address=address,op=op,data_hex=datum.to_bytes(32,'little').hex())
        elif a_hold is not None: reject('DMA A valid dropped under backpressure')
        if d('dv'):
            response=(d('ds'),d('dd'))
            if d_hold is not None and response!=d_hold: reject('DMA D changed under backpressure')
            if not d('dr'): d_hold=response
            else:
                d_hold=None
                source,datum=response
                if pending is None or source not in outstanding: reject('unbound DMA response')
                request=outstanding.pop(source);row=request['row'];observed=datum.to_bytes(32,'little')
                if edge-request['edge']<43 or pending['op']=='load' and observed!=pending['reference_rows'][row]: reject('DMA delayed load response mismatch')
                record(pending['responses'],source=source,row=row,data_hex=observed.hex())
        elif d_hold is not None: reject('DMA D valid dropped under backpressure')
        for port in ('vr','vw'):
            if d(port):
                if pending is None or (port=='vw')!=(pending['op']=='load'): reject('unbound DMA VMEM direction')
                row=len(pending['memory']);line=pending['vmem_word']//8+row
                if row>=4 or (d(port+'b'),d(port+'a'))!=(line>>13,line&8191): reject('DMA VMEM request address mismatch')
                if port=='vw' and d('vwd').to_bytes(32,'little')!=pending['reference_rows'][row]: reject('DMA VMEM write data mismatch')
                if d(port+'g'):
                    observed=pending['reference_rows'][row]
                    if port=='vw': vm[32*line:32*(line+1)]=observed
                    else: read_queue.append((edge+1,row,observed))
                    record(pending['memory'],kind=port,line=line,row=row,data_hex=observed.hex())
        expected_response=bool(read_queue and read_queue[0][0]==edge)
        if bool(d('vresp'))!=expected_response: reject('DMA VMEM read response timing mismatch')
        if expected_response:
            _,row,expected=read_queue.pop(0);observed=d('vdata').to_bytes(32,'little')
            if observed!=expected: reject('DMA VMEM source response mismatch')
            record(pending['memory_responses'],row=row,data_hex=observed.hex())
        if publication is not None:
            if edge!=publication or v('marker')!=1: reject('marker publication did not follow write')
            record(endpoints,kind='marker_publication');publication=None
        if v('csr'):
            if active or pending or outstanding or marker is not None or (v('csr_addr'),v('csr_op'),v('csr_data'))!=(0xc10,1,1): reject('marker precedes observed engine drain')
            marker=edge;publication=edge+1;record(endpoints,kind='marker',word_index=v('pc'),word_u32=v('word'),engine_state=quiescent_state(v,d,busy,engine))
        if v('ecall'):
            if not began or ended or active or pending or outstanding or read_queue or marker is None or v('marker')!=1 or not v('halt') or v('fire') or pc_expected!=len(words)-1 or v('word')!=words[-1]: reject('terminal precedes complete selected-word/engine drain')
            ended=True;record(endpoints,kind='halt',word_index=v('pc'),word_u32=v('word'),engine_state=quiescent_state(v,d,busy,engine))
        elif began and not ended and v('halt'): reject('unexpected halt in compute envelope')
    if not ended or active or pending or outstanding or read_queue or publication is not None: reject('incomplete compute/terminal drain trace')
    if last_edge-next(e['edge'] for e in endpoints if e['kind']=='halt')<2: reject('trace lacks bounded post-halt drain samples')
    expected_counts={'xlu':(2,1,0,0),'vmul':(6,0,1,0),'dma_xlu':(2,1,0,2)}[mode]
    if tuple(sum(c['engine']==kind for c in commands) for kind in ('vls','xlu','vmul','dma'))!=expected_counts: reject('wrong finite compute command coverage')
    if vm!=expected_vm: reject('observed composed VMEM state differs from full reference')
    for command in commands:
        command.pop('reference_rows',None);command.pop('source_rows',None)
        for field in ('reads','responses','writes'):
            for event in command[field]: event['age']=event['edge']-command['edge']
    return {'mode':mode,'instructions':instructions,'commands':commands,'endpoints':endpoints,'busy_transitions':busy_events,'sampled_edges':last_edge,'full_memory_checked_bytes':len(vm),'terminal_drain_edges':last_edge-next(e['edge'] for e in endpoints if e['kind']=='halt')}

def analyze_compute(trace,words,mode):
    with CHECK.allowed(trace).open() as stream:
        return validate_compute_edges(COMPUTE_OBS.vcd_edges(stream,'TOP.AtlasCore'),words,mode)


def run(args):
    if not args.cases or len(set(args.cases.split(',')))!=len(args.cases.split(',')) or not set(args.cases.split(','))<=set(cases()): raise ValueError('unsupported or empty selected cases')
    if args.consumer!='none' and (args.atlas_opt is None or any(getattr(args,key+'_evidence') is None or getattr(args,key+'_evidence_sha256') is None for key in ('vls','dma','xlu','vmul'))): raise ValueError('selected consumer requires optimizer and four explicitly hash-selected receipts')
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
    harness=checker.identity(root/'test/ee290-compute-replay.cpp')
    copy=inputs/'ee290-compute-replay.cpp';copy.write_bytes(CHECK.allowed(harness['path']).read_bytes())
    dependencies=[]
    for name in ['replay-ee290-compute.py','replay-ee290-dma.py','observe-ee290-vls.py','ee290_build_capture.py','check-ee290-provenance.py','fingerprint-rtl-modules.py','index-retained-hw.py','check-ee290-compute-export.py']:
        src=checker.identity(root/'tools'/name);dest=inputs/name;dest.write_bytes(CHECK.allowed(src['path']).read_bytes());dependencies.append({'original':src,'snapshot':checker.identity(dest)})
    tools={name:checker.identity(getattr(args,name)) for name in ('verilator','cxx','make','ar','atlas_emit')}
    if args.consumer!='none': tools['atlas_opt']=checker.identity(args.atlas_opt)
    runtime=CHECK.allowed(args.verilator_root)
    env={'PATH':str(CHECK.allowed(args.make).parent)+':/bin','LC_ALL':'C','LANG':'C','VERILATOR_ROOT':str(runtime),'CXX':tools['cxx']['path'],'AR':tools['ar']['path']}
    argv=[tools['verilator']['path'],'--cc','--exe','--build','--top-module','AtlasCore','--prefix','VAtlasCore','--Mdir','{output}/obj','--trace','--trace-depth','3','--assert','--output-split','20000','--output-split-cfuncs','500','-Wno-fatal','-j',str(args.jobs),'-CFLAGS','-std=c++17','-MAKEFLAGS','CXX='+tools['cxx']['path']+' LINK='+tools['cxx']['path']+' AR='+tools['ar']['path'],*[r['snapshot']['path'] for r in rtl],str(copy)]
    checker.recheck()
    build_inputs=[{'role':'tool' if n=='verilator' else n,'identity':i} for n,i in tools.items() if n in ('verilator','cxx','make','ar')]+[{'role':'selected_rtl','identity':r['snapshot']} for r in rtl]+[{'role':'harness','identity':checker.identity(copy)}]
    if args.reuse_compile:
        phase_id=checker.identity(args.reuse_compile)
        phase=CHECK.strict_json(CHECK.allowed(args.reuse_compile).read_bytes())
        if phase['state']!='phase_completed' or phase['kind']!='compute_verilator_compile' or phase['inputs_before']!=phase['inputs_after']: raise ValueError('invalid reused compile')
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
        phase=CAPTURE.capture_phase(output/'compile','compute_verilator_compile',argv,build_inputs,[{'role':'model','relative_path':'obj','kind':'tree'}],env,args.timeout_seconds)
        if phase['state']!='phase_completed': raise ValueError('compute compile failed')
        phase_id=checker.identity(output/'compile/phase.json')
        binary=checker.identity(output/'compile/obj/VAtlasCore')
    receipt={'schema':'atlas.ee290_compute_replay.v0','target_config':'EE290SimConfig','state':'prepared','qualification':'conditional','scheduling_qualified':False,'selected_core_replay':report_id,'manifest':manifest,'hardware_ir':hardware,'rtl_snapshots':rtl,'harness':{'original':harness,'snapshot':checker.identity(copy)},'producer_snapshots':dependencies,'tool_identities':tools,'compile_phase':phase_id,'model':binary,'cases':[],'assumptions':['Fresh reset followed by START with selected compute engines and all DMA queues quiescent.','Serialized finite VLS, XLU transpose and paired BF16 multiplication with conservative delays; optional aligned128B DMA and matching waits.','Direct host TileLink and deterministic delayed external DMA slave; behavioral SRAM; not CPU/SoC qualification.','Opaque installed Verilator and C++ runtime; two-state clocked simulation and pre-edge fixed boundary sampling.', 'Finite VMUL numerical reference covers signed normal BF16 powers only; other values, rounding and exceptions are not numerically qualified.', 'Only selected fixture command/data streams and the captured post-halt drain are validated; no general concurrency or CPU/system qualification.']}
    evidence={};selection=None;selected_ids=[]
    if args.consumer!='none':
        for key in ('vls','dma','xlu','vmul'):
            path=getattr(args,key+'_evidence');ident=checker.identity(path)
            if ident['sha256']!=getattr(args,key+'_evidence_sha256'): raise ValueError('selected '+key+' evidence hash mismatch')
            selected_ids.append({'kind':key,'identity':ident})
            evidence['evidence_sha256' if key=='vls' else key+'_evidence_sha256']=ident['sha256']
        evidence.update(manifest_sha256=manifest['sha256'],hardware_ir_sha256=hardware['sha256'])
        options={'evidence':selected_ids[0]['identity']['path'],'evidence-sha256':evidence['evidence_sha256'],'manifest-sha256':manifest['sha256'],'hardware-ir-sha256':hardware['sha256'],'allow-conditional':'true'}
        for item in selected_ids[1:]:
            key=item['kind'];options[key+'-evidence']=item['identity']['path'];options[key+'-evidence-sha256']=item['identity']['sha256']
        selection='--select-atlas-rtl-evidence='+' '.join(k+'='+v for k,v in options.items())
        receipt.update(schema='atlas.ee290_scheduled_compute_replay.v0',selected_evidence=selected_ids)
        receipt['assumptions'][1]='Serialized finite VLS, XLU transpose and paired BF16 multiplication; delays come from the selected compiler consumer, optional aligned128B DMA uses matching waits.'
    (output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    consumers=('delay','schedule') if args.consumer=='both' else (args.consumer,)
    selected_cases=cases(args.consumer!='none')
    jobs=[(mode if consumer=='none' else mode+'_'+consumer,text,mode,consumer) for mode,(text,_) in selected_cases.items() if mode in args.cases.split(',') for consumer in consumers]
    receipt['requested_cases']=[name for name,text,mode,consumer in jobs]
    receipt['requested_consumer']=args.consumer
    for name,text,mode,consumer in jobs:
        directory=output/name;directory.mkdir()
        source=directory/'program.mlir';source.write_text(text)
        source_id=checker.identity(source)
        authored_id=source_id;extra={}
        python=checker.identity(sys.executable)
        if consumer!='none':
            child="import subprocess,sys; subprocess.run([sys.argv[1],*sys.argv[4:],sys.argv[2]],stdout=open(sys.argv[3],'wb'),check=True)"
            phase=CAPTURE.capture_phase(directory/'opt','compute_selected_consumer',[python['path'],'-c',child,tools['atlas_opt']['path'],str(source),'{output}/final.mlir',selection,'--insert-atlas-delays' if consumer=='delay' else '--schedule-atlas-stream','--verify-atlas-rtl-timing'],[{'role':'tool','identity':python},{'role':'atlas_opt','identity':tools['atlas_opt']},{'role':'program','identity':source_id}]+[{'role':item['kind']+'_evidence','identity':item['identity']} for item in selected_ids],[{'role':'selected_final_mlir','relative_path':'final.mlir','kind':'file'}],env,args.timeout_seconds)
            if phase['state']!='phase_completed': raise ValueError('selected consumer failed '+name)
            source=directory/'opt/final.mlir';source_id=checker.identity(source)
            extra.update(consumer=consumer,authored_program=authored_id,optimizer_phase=checker.identity(directory/'opt/phase.json'))
        child="import subprocess,sys; subprocess.run([sys.argv[1],sys.argv[2]],stdout=open(sys.argv[3],'wb'),check=True)"
        emission=CAPTURE.capture_phase(directory/'emit','compute_program_emit',[python['path'],'-c',child,tools['atlas_emit']['path'],str(source),'{output}/program.hex'], [{'role':'tool','identity':python},{'role':'atlas_emit','identity':tools['atlas_emit']},{'role':'program','identity':source_id}],[{'role':'emitted_words','relative_path':'program.hex','kind':'file'}],env,args.timeout_seconds)
        if emission['state']!='phase_completed': raise ValueError('DMA emission failed')
        emitted=CHECK.allowed(directory/'emit/program.hex').read_text()
        if not re.fullmatch(r'(?:[0-9a-fA-F]{8}\s+)+',emitted): raise ValueError('invalid emitted words')
        words=directory/'program.hex';words.write_text(emitted);words_id=checker.identity(words)
        execution=CAPTURE.capture_phase(directory/'execution','compute_execution',[binary['path'],str(words),'{output}/trace.vcd','100000',mode],[{'role':'tool','identity':binary},{'role':'words','identity':words_id}],[{'role':'trace','relative_path':'trace.vcd','kind':'file'}],env,args.timeout_seconds)
        if execution['state']!='phase_completed': raise ValueError('DMA execution failed '+name)
        log=CHECK.allowed(directory/'execution/command.stdout.log').read_text()
        match=re.search(r'EE290_COMPUTE_PASSED mode='+mode+r' checked_vmem_bytes=8192 reads=(\d+) writes=(\d+) a_stalls=(\d+) delayed_response_cycles=(\d+) status=5 marker=1 illegal_pc=0 cycles=(\d+)',log)
        count=4 if mode=='dma_xlu' else 0
        if not match or int(match[1])!=count or int(match[2])!=count: raise ValueError('compute numerical mismatch')
        trace_id=checker.identity(directory/'execution/trace.vcd')
        observations=analyze_compute(Path(trace_id['path']),[int(w,16) for w in emitted.split()],mode)
        transcript={'schema':'atlas.ee290_compute_events.v0','validator':'atlas.ee290.compute.fixed.v1','target_config':'EE290SimConfig','data_encoding':'little_endian_bytes_hex','sampling':'values immediately before rising clock timestamp','trace':trace_id,'words':words_id,**observations}
        event_path=directory/'events.json';event_path.write_text(json.dumps(transcript,indent=2)+'\n')
        if consumer!='none':
            child="import subprocess,sys; subprocess.run([sys.argv[1],'--rtl-timing-json',sys.argv[2]],stdout=open(sys.argv[3],'wb'),check=True)"
            phase=CAPTURE.capture_phase(directory/'export','compute_resolved_timing_export',[python['path'],'-c',child,tools['atlas_emit']['path'],str(source),'{output}/timing.json'],[{'role':'tool','identity':python},{'role':'atlas_emit','identity':tools['atlas_emit']},{'role':'program','identity':source_id}],[{'role':'resolved_timing','relative_path':'timing.json','kind':'file'}],env,args.timeout_seconds)
            if phase['state']!='phase_completed': raise ValueError('resolved export failed '+name)
            timing_id=checker.identity(directory/'export/timing.json')
            binding=BIND.bind_export(CHECK.strict_json(CHECK.allowed(timing_id['path']).read_bytes()),transcript,evidence,[int(w,16) for w in emitted.split()],'atlas.vls_dma_xlu_vmul.serialized.v1')
            binding.update(resolved_timing=timing_id,events=checker.identity(event_path),words=words_id)
            path=directory/'export-binding.json';path.write_text(json.dumps(binding,indent=2)+'\n')
            extra.update(resolved_timing=timing_id,export_phase=checker.identity(directory/'export/phase.json'),export_binding=checker.identity(path))
        receipt['cases'].append({**extra,'name':name,'events':checker.identity(event_path),'boundary_checks':{'selected_words':True,'data_streams':True,'observed_release':True,'composed_memory':True,'terminal_drain':True},'program':source_id,'words':words_id,'emission_phase':checker.identity(directory/'emit/phase.json'),'execution_phase':checker.identity(directory/'execution/phase.json'),'trace':trace_id,'numerical_checks':{'completion':True,'output':True,'guards':True},'observations':dict(zip(['reads','writes','a_stalls','delayed_response_cycles','cycles'],map(int,match.groups())))})
    if [item['name'] for item in receipt['cases']]!=receipt['requested_cases'] or not receipt['cases']: raise ValueError('requested replay cases incomplete')
    checker.recheck()
    receipt['state']='finite_compute_cases_passed' if args.consumer=='none' else 'finite_scheduled_compute_cases_passed'
    (output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('selected-core-replay','output','verilator','cxx','make','ar','atlas-emit','verilator-root'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-replay-sha256',required=True)
    p.add_argument('--reuse-compile',type=Path)
    p.add_argument('--consumer',choices=('none','delay','schedule','both'),default='none')
    p.add_argument('--cases',default='xlu,vmul,dma_xlu')
    p.add_argument('--atlas-opt',type=Path)
    for key in ('vls','dma','xlu','vmul'):
        p.add_argument('--'+key+'-evidence',type=Path)
        p.add_argument('--'+key+'-evidence-sha256')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--timeout-seconds',type=int,default=1200)
    try: print(json.dumps(run(p.parse_args()),indent=2))
    except (ValueError,KeyError,OBS.ObservationError) as error: p.exit(1,str(error)+'\n')
