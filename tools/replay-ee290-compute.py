#!/usr/bin/env python3
"""Verilator replay of compiler-scheduled AtlasCore programs against `merlin.op_timing.v1` facts.

Each program is scheduled by atlas-opt (facts selection, dma=wait), emitted by atlas-emit,
run on the Verilated AtlasCore, and its VCD is decoded and bound to `atlas-emit --rtl-timing-json`.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('check_ee290_compute_export', HERE / 'check-ee290-compute-export.py')
BIND = importlib.util.module_from_spec(spec); spec.loader.exec_module(BIND)


class ReplayError(Exception):
    pass


def program(ops):
    lines = ['module {', '  %s0 = "atlas.start"() : () -> !atlas.state']
    for n, (op, attrs) in enumerate(ops, 1):
        lines.append(f'  %s{n} = "atlas.{op}"(%s{n-1}) {{{attrs}}} : (!atlas.state) -> !atlas.state')
    return '\n'.join(lines + ['}', ''])


def add(dst, value, src=0): return ('alu_imm', f'kind = "addi", dst = {dst} : i32, src = {src} : i32, immediate = {value} : i32')
def upper(dst, value): return ('upper', f'kind = "lui", dst = {dst} : i32, immediate = {value} : i32')
def dma(direction, channel, reg=6): return ('dma', f'direction = "{direction}", channel = {channel} : i32, reg = {reg} : i32, dram = {1 if direction == "load" else 3} : i32, size = 2 : i32')
def wait(channel): return ('dma_wait', f'channel = {channel} : i32')
def vls(direction, mreg, base): return (direction, f'{"dst" if direction == "vload" else "src"} = {mreg} : i32, base = {base} : i32, offset = 0 : i32, format = "raw"')
def xlu(): return ('xlu_transpose', 'dst = 1 : i32, src = 0 : i32')


def cases():
    """Delay-free programs; the compiler inserts or schedules every DELAY."""
    end = [add(1, 1), ('csr', 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32'), ('trap', 'kind = "ecall"')]
    transpose = [add(6, 0), vls('vload', 0, 6), xlu(), add(8, 256), vls('vstore', 1, 8), add(0, 0)]
    mul = []
    for bank, words in enumerate([0, 256, 512, 768]): mul += [add(6, words), vls('vload', bank, 6)]
    mul += [('vpu_binary', 'kind = "mul", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32')]
    for bank, words in [(4, 1024), (5, 1280)]: mul += [add(8, words), vls('vstore', bank, 8)]
    mixed = [add(5, 0), ('dma_config', 'channel = 0 : i32, base_reg = 5 : i32'), add(6, 0), upper(1, 589824), add(2, 128), dma('load', 0), wait(0)]
    mixed += transpose + [upper(3, 589824), add(3, 1024, 3), ('dma', 'direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32'), wait(1)]
    return {name: program(ops + end) for name, ops in [('xlu', transpose), ('vmul', mul), ('dma_xlu', mixed)]}


# key: (path below TOP.AtlasCore, width)
SIGNALS = {
    'clock': ('clock', 1), 'reset': ('reset', 1), 'fire': ('scalar.s1_fire', 1), 'launch': ('scalar.is_lsu_launch', 1),
    'valid': ('scalar.pc_ctrl.io_s1_valid', 1), 'pc': ('scalar.pc_ctrl.io_s1_pc', 32), 'word': ('scalar.decoder.io_instr', 32),
    'cmd': ('lsu.io_cmd_valid', 1), 'op': ('lsu.io_cmd_bits_op', 2), 'mreg': ('lsu.io_cmd_bits_mregBank', 6), 'line': ('lsu.io_cmd_bits_vmemLineAddr', 16),
    'load_busy': ('lsu.io_vloadBusy', 1), 'store_busy': ('lsu.io_vstoreBusy', 1),
    'vr': ('lsu.io_vmemVecRead_valid', 1), 'vr_bank': ('lsu.io_vmemVecRead_bits_bankIdx', 3), 'vr_addr': ('lsu.io_vmemVecRead_bits_bankAddr', 13),
    'vresp': ('lsu.io_vmemVecReadData_valid', 1), 'vresp_data': ('lsu.io_vmemVecReadData_bits', 256),
    'vw': ('lsu.io_vmemVecWrite_valid', 1), 'vw_bank': ('lsu.io_vmemVecWrite_bits_bankIdx', 3), 'vw_addr': ('lsu.io_vmemVecWrite_bits_bankAddr', 13), 'vw_data': ('lsu.io_vmemVecWrite_bits_data', 256),
    'mr': ('lsu.io_mregReadReq_valid', 1), 'mr_id': ('lsu.io_mregReadReq_bits_mregId', 6), 'mr_row': ('lsu.io_mregReadReq_bits_row', 5),
    'mresp': ('lsu.io_mregReadResp_valid', 1), 'mresp_data': ('lsu.io_mregReadResp_bits', 256),
    'mw': ('lsu.io_mregWriteReq_valid', 1), 'mw_id': ('lsu.io_mregWriteReq_bits_mregId', 6), 'mw_row': ('lsu.io_mregWriteReq_bits_row', 5), 'mw_data': ('lsu.io_mregWriteReq_bits_data', 256),
    'csr': ('csrfile.io_csr_valid', 1), 'csr_addr': ('csrfile.io_csr_addr', 12), 'csr_op': ('csrfile.io_csr_op', 3), 'csr_data': ('csrfile.io_csr_wdata', 32), 'marker': ('csrfile.reg_dbg0', 32),
    'halt': ('csrfile.io_csr_halted', 1), 'ecall': ('csrfile.io_csr_set_ecall', 1), 'illegal': ('csrfile.io_csr_set_illegal', 1), 'ebreak': ('csrfile.io_csr_set_ebreak', 1),
    'xcmd': ('xlu.io_cmd_valid', 1), 'xsrc': ('xlu.io_cmd_bits_srcMregId', 6), 'xdst': ('xlu.io_cmd_bits_dstMregId', 6),
    'xr': ('xlu.io_mregReadReq_valid', 1), 'xrid': ('xlu.io_mregReadReq_bits_mregId', 6), 'xrrow': ('xlu.io_mregReadReq_bits_row', 5),
    'xresp': ('xlu.io_mregReadResp_valid', 1), 'xdata': ('xlu.io_mregReadResp_bits', 256),
    'xw': ('xlu.io_mregWriteReq_valid', 1), 'xwid': ('xlu.io_mregWriteReq_bits_mregId', 6), 'xwrow': ('xlu.io_mregWriteReq_bits_row', 5), 'xwdata': ('xlu.io_mregWriteReq_bits_data', 256),
    'xar': ('xlu.io_activeMregRead_valid', 1), 'xarid': ('xlu.io_activeMregRead_bits', 6), 'xaw': ('xlu.io_activeMregWrite_valid', 1), 'xawid': ('xlu.io_activeMregWrite_bits', 6),
    'pcmd': ('vpu.io_cmd_valid', 1), 'pop': ('vpu.io_cmd_bits_op', 5), 'pdst': ('vpu.io_cmd_bits_vd', 6), 'psrc0': ('vpu.io_cmd_bits_vs1', 6), 'psrc1': ('vpu.io_cmd_bits_vs2', 6),
}
for n in range(2):
    for key, path, width in [('r', 'ReadReq%d_valid', 1), ('rid', 'ReadReq%d_bits_mregId', 6), ('rrow', 'ReadReq%d_bits_row', 5), ('resp', 'ReadResp%d_valid', 1), ('data', 'ReadResp%d_bits', 256),
                             ('w', 'WriteReq%d_valid', 1), ('wid', 'WriteReq%d_bits_mregId', 6), ('wrow', 'WriteReq%d_bits_row', 5), ('wdata', 'WriteReq%d_bits_data', 256)]:
        SIGNALS[f'p{key}{n}'] = ('vpu.io_mreg' + path % n, width)
for n in range(4):
    SIGNALS.update({f'par{n}': (f'vpu.io_activeReads_{n}_valid', 1), f'parid{n}': (f'vpu.io_activeReads_{n}_bits', 6),
                    f'paw{n}': (f'vpu.io_activeWrites_{n}_valid', 1), f'pawid{n}': (f'vpu.io_activeWrites_{n}_bits', 6)})
for n in range(8):
    SIGNALS.update({f'dma_busy{n}': (f'scalar.io_dma_busy_{n}', 1), f'dma_engine{n}': (f'dma.io_channelBusy_{n}', 1)})
for key, path, width in [
        ('stall', 'scalar.stall', 1), ('base', 'scalar.dmaBaseReg', 32), ('rs1', 'scalar.regfile.io_rs1_data', 32),
        ('launch', 'scalar.io_dmaCmd_valid', 1), ('op', 'scalar.io_dmaCmd_bits_op', 3), ('vmem', 'scalar.io_dmaCmd_bits_vmemAddr', 32),
        ('address', 'scalar.io_dmaCmd_bits_addr', 64), ('size', 'scalar.io_dmaCmd_bits_size', 32), ('launch_channel', 'scalar.io_dmaCmd_bits_channel', 3),
        ('av', 'io_dmaTL_a_valid', 1), ('ar', 'io_dmaTL_a_ready', 1), ('ao', 'io_dmaTL_a_bits_opcode', 3), ('aa', 'io_dmaTL_a_bits_address', 37),
        ('as', 'io_dmaTL_a_bits_source', 6), ('ad', 'io_dmaTL_a_bits_data', 256), ('dv', 'io_dmaTL_d_valid', 1), ('dr', 'io_dmaTL_d_ready', 1),
        ('ds', 'io_dmaTL_d_bits_source', 6), ('dd', 'io_dmaTL_d_bits_data', 256), ('vr', 'dma.io_vmemRead_valid', 1), ('vrg', 'dma.io_vmemReadGrant', 1),
        ('vrb', 'dma.io_vmemRead_bits_bankIdx', 3), ('vra', 'dma.io_vmemRead_bits_bankAddr', 13), ('vw', 'dma.io_vmemWrite_valid', 1),
        ('vwg', 'dma.io_vmemWriteGrant', 1), ('vwb', 'dma.io_vmemWrite_bits_bankIdx', 3), ('vwa', 'dma.io_vmemWrite_bits_bankAddr', 13),
        ('vwd', 'dma.io_vmemWrite_bits_data', 256), ('vresp', 'dma.io_vmemReadData_valid', 1), ('vdata', 'dma.io_vmemReadData_bits', 256)]:
    SIGNALS['dma_' + key] = (path, width)
CONTROLS = ['fire', 'launch', 'cmd', 'load_busy', 'store_busy', 'vr', 'vresp', 'vw', 'mr', 'mresp', 'mw', 'csr', 'halt', 'ecall', 'illegal', 'ebreak',
            'valid', 'xcmd', 'xr', 'xresp', 'xw', 'xar', 'xaw', 'pcmd'] + [f'{p}{n}' for n in range(2) for p in ('pr', 'presp', 'pw')] + [f'{p}{n}' for n in range(4) for p in ('par', 'paw')]
CONTROLS += ['dma_' + k for k in ['launch', 'stall', 'av', 'ar', 'dv', 'dr', 'vr', 'vrg', 'vw', 'vwg', 'vresp']] + [f'dma_{p}{n}' for n in range(8) for p in ('busy', 'engine')]
PORTS = ['vr', 'vresp', 'mw', 'mr', 'mresp', 'vw']


def vcd_edges(stream, scope='TOP.AtlasCore'):
    """Yield {'edge', 'values'} sampled just before each rising clock; unknown or malformed VCD raises."""
    wanted = {}
    for key, (path, _) in SIGNALS.items(): wanted.setdefault(scope + '.' + path, []).append(key)
    codes, scopes, tokens, values, changes = {}, [], [], {}, {}
    stamp, edge, header = None, 0, True
    def flush():
        nonlocal edge
        before = dict(values); values.update(changes)
        if before.get(codes['clock']) == 0 and values.get(codes['clock']) == 1:
            edge += 1
            return {'edge': edge, 'values': {key: before.get(code) for key, code in codes.items()}}
    for line in stream:
        if header:
            tokens += line.split()
            while '$end' in tokens:
                entry, tokens = tokens[:tokens.index('$end')], tokens[tokens.index('$end') + 1:]
                if entry and entry[0] == '$scope': scopes.append(entry[2])
                elif entry and entry[0] == '$upscope': scopes.pop()
                elif entry and entry[0] == '$var':
                    for key in wanted.get('.'.join(scopes + [entry[4]]), []):
                        if key in codes or int(entry[2]) != SIGNALS[key][1]: raise ReplayError('ambiguous or incompatible VCD signal ' + key)
                        codes[key] = entry[3]
                elif entry and entry[0] == '$enddefinitions':
                    if set(codes) != set(SIGNALS): raise ReplayError('VCD lacks signals: ' + ','.join(sorted(set(SIGNALS) - set(codes))))
                    header = False
            continue
        words = line.split(); i = 0
        while i < len(words):
            word = words[i]; i += 1
            if word.startswith('#'):
                new = int(word[1:])
                if stamp is not None:
                    if new <= stamp: raise ReplayError('non-increasing VCD timestamp')
                    sample = flush()
                    if sample: yield sample
                stamp, changes = new, {}
            elif word.startswith('$'): continue
            elif word[0] in 'bB': changes[words[i]] = None if set(word[1:].lower()) & set('xz') else int(word[1:], 2); i += 1
            elif word[0] in '01xz': changes[word[1:]] = None if word[0] in 'xz' else int(word[0])
            else: raise ReplayError('unsupported VCD value')
    if header or stamp is None: raise ReplayError('truncated VCD')
    sample = flush()
    if sample: yield sample


def pattern(offset): return sum(((37 * (offset + n) + 11) & 255) << (8 * n) for n in range(32))


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
    else: raise ReplayError('unsupported finite compute mode')
    return result

def bf16_power_product(lhs,rhs):
    result=bytearray()
    for offset in range(0,32,2):
        a=int.from_bytes(lhs[offset:offset+2],'little');b=int.from_bytes(rhs[offset:offset+2],'little')
        ea=(a>>7)&255;eb=(b>>7)&255
        if a&127 or b&127 or not 1<=ea<=254 or not 1<=eb<=254:
            raise ReplayError('outside exact finite-normal BF16 power reference')
        e=ea+eb-127
        if not 1<=e<=254: raise ReplayError('BF16 power product outside normal domain')
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
    controls=CONTROLS
    def reject(message): raise ReplayError(message)
    def row_bytes(identity,row):
        if (identity,row) not in mreg: reject('read from uninitialized logical MREG row')
        return mreg[identity,row]
    for sample in samples:
        edge=sample['edge'];last_edge=edge
        def v(key):
            value=sample['values'][key]
            if type(value)!=int or not 0<=value<(1<<SIGNALS[key][1]): reject('unknown/out-of-range compute signal '+key)
            return value
        def d(key): return v('dma_'+key)
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
        expected_cmd={'cmd':False,'xcmd':False,'pcmd':False,'dma_launch':False}
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
                    rows=[bytes(vm[4*vmword+32*r:4*vmword+32*(r+1)]) for r in range(4)] if store else [pattern(32*r).to_bytes(32,'little') for r in range(4)]
                    pending={**common,'engine':'dma','op':'store' if store else 'load','channel':channel,'vmem_word':vmword,'dram_address':address,'size':size,'reference_rows':rows,'requests':[],'memory':[],'memory_responses':[],'wait_stall_edges':[]}
                    commands.append(pending);expected_cmd['dma_launch']=True
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
        expected_ports={key:False for key in PORTS+['xr','xresp','xw','xar','xaw']+['pr'+str(p) for p in range(2)]+['presp'+str(p) for p in range(2)]+['pw'+str(p) for p in range(2)]}
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


def sh(argv, timeout, env=None):
    result = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode: raise ReplayError(f'{Path(str(argv[0])).name} failed: {result.stderr[-2000:]}')
    return result.stdout


def build_model(args, work):
    objdir = work / 'obj'; cxx = shutil.which('g++')
    argv = [args.verilator, '--cc', '--exe', '--build', '--top-module', 'AtlasCore', '--prefix', 'VAtlasCore', '--Mdir', str(objdir), '--trace', '--trace-depth', '3',
            '--assert', '--output-split', '20000', '--output-split-cfuncs', '500', '-Wno-fatal', '-j', str(args.jobs), '-CFLAGS', '-std=c++17',
            '-MAKEFLAGS', f"CXX={cxx} LINK={cxx} AR={shutil.which('ar')}",
            *map(str, sorted([*args.rtl.glob('*.sv'), *args.rtl.glob('*.v')])), str(HERE.parent / 'test/ee290-compute-replay.cpp')]
    sh(argv, args.timeout, {**os.environ, **({'VERILATOR_ROOT': str(args.verilator_root)} if args.verilator_root else {})})
    return objdir / 'VAtlasCore'


def run(args):
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    facts = args.facts.resolve(); digest = hashlib.sha256(facts.read_bytes()).hexdigest()
    if args.facts_sha256 and args.facts_sha256 != digest: raise ReplayError('facts hash mismatch')
    model = args.model.resolve() if args.model else build_model(args, work)
    select = f'--select-atlas-rtl-evidence=op-timing={facts} op-timing-sha256={digest} dma=wait'
    opt, emit = args.bin / 'atlas-opt', args.bin / 'atlas-emit'
    available = cases(); results = []
    for mode in args.cases.split(','):
        for consumer in (('delay', 'schedule') if args.consumer == 'both' else (args.consumer,)):
            name = f'{mode}_{consumer}'; out = work / name; out.mkdir(exist_ok=True)
            (out / 'program.mlir').write_text(available[mode])
            sh([opt, out / 'program.mlir', select, '--insert-atlas-delays' if consumer == 'delay' else '--schedule-atlas-stream', '--verify-atlas-rtl-timing', '-o', out / 'final.mlir'], args.timeout)
            text = sh([emit, out / 'final.mlir'], args.timeout)
            if not re.fullmatch(r'(?:[0-9a-fA-F]{8}\s+)+', text): raise ReplayError('invalid emitted words')
            (out / 'program.hex').write_text(text); words = [int(w, 16) for w in text.split()]
            log = sh([model, out / 'program.hex', out / 'trace.vcd', '100000', mode], args.timeout)
            count = 4 if mode == 'dma_xlu' else 0
            match = re.search(r'EE290_COMPUTE_PASSED mode=' + mode + r' checked_vmem_bytes=8192 reads=(\d+) writes=(\d+) .* status=5 marker=1 illegal_pc=0 cycles=(\d+)', log)
            if not match or int(match[1]) != count or int(match[2]) != count: raise ReplayError(name + ': numerical check failed')
            with (out / 'trace.vcd').open() as stream: events = validate_compute_edges(vcd_edges(stream), words, mode)
            (out / 'events.json').write_text(json.dumps(events, indent=1))
            export = json.loads(sh([emit, '--rtl-timing-json', out / 'final.mlir'], args.timeout))
            (out / 'timing.json').write_text(json.dumps(export, indent=1))
            binding = BIND.bind_export(export, events, digest, words)
            results.append({'case': name, 'cycles': int(match[3]), **{k: binding[k] for k in ('instruction_words_bound', 'memory_access_elements_bound')}})
            print(f'PASS {name} words={len(words)} accesses_bound={binding["memory_access_elements_bound"]} dma_intervals={len(binding["dynamic_dma_intervals"])}', flush=True)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--facts', type=Path, required=True, help='merlin.op_timing.v1 facts JSON')
    p.add_argument('--facts-sha256', help='optional expected SHA-256 of --facts')
    p.add_argument('--bin', type=Path, required=True, help='directory holding atlas-opt and atlas-emit')
    p.add_argument('--work', type=Path, required=True, help='output directory')
    p.add_argument('--model', type=Path, help='prebuilt VAtlasCore binary (skips the Verilator build)')
    p.add_argument('--rtl', type=Path, help='directory of AtlasCore *.sv/*.v to Verilate when --model is absent')
    p.add_argument('--verilator', default=shutil.which('verilator') or 'verilator')
    p.add_argument('--verilator-root', type=Path, help='VERILATOR_ROOT override (directory holding include/verilated.h)')
    p.add_argument('--consumer', choices=('delay', 'schedule', 'both'), default='both')
    p.add_argument('--cases', default='vmul,dma_xlu', help='comma list from: xlu,vmul,dma_xlu')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if not a.model and not a.rtl: p.error('--model or --rtl required')
    if not set(a.cases.split(',')) <= set(cases()): p.error('unknown case')
    a.bin = a.bin.resolve()
    try: run(a)
    except (ReplayError, BIND.BindingError, subprocess.SubprocessError, OSError, KeyError) as error: sys.exit(f'FAIL: {error}')
