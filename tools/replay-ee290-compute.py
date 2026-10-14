#!/usr/bin/env python3
"""Schedule AtlasCore cases, replay them on Verilator and bind decoded events to RTL timing exports."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ee290_replay_cases as CASES  # noqa: E402
from ee290_replay_cases import mxu as MXU, vpu_ops as VPU  # noqa: E402

PROVIDER = 'rtl-op-timing-v1'
SCHEMA = 'atlas.resolved_rtl_timing.v1'
BRANCHES = {0: 'beq', 1: 'bne', 4: 'blt', 5: 'bge', 6: 'bltu', 7: 'bgeu'}
# Observed engine -> (selected mnemonic, the engine's own hold unit).
ENGINES = {'vload': ('vload', 'VLOAD path'), 'vstore': ('vstore', 'VSTORE path'), 'xlu': ('vtrpose.xlu', 'XLU'), 'vpu': (None, 'VPU'),
           'mxu0': (None, 'MXU in-flight matmuls'), 'mxu1': (None, 'MXU in-flight matmuls')}  # an MXU owns its own unit index
ENGINE_UNITS = {'VLOAD path', 'VSTORE path', 'XLU', 'VPU'}
MXU_STEMS = ('vmatpush.weight', 'vmatpush.acc.fp8', 'vmatpush.acc.bf16', 'vmatpop.fp8.acc', 'vmatpop.bf16.acc', 'vmatmul', 'vmatmul.acc')
VPU_MNEMONICS = {0: 'vadd.bf16', 2: 'vsub.bf16', 3: 'vmul.bf16', 4: 'vminimum.bf16', 6: 'vmaximum.bf16', 1: 'vredsum.bf16', 5: 'vredmin.bf16',
                 7: 'vredmax.bf16', 0x21: 'vredsum.row.bf16', 0x24: 'vredmin.row.bf16', 0x26: 'vredmax.row.bf16', 0x40: 'vmov',
                 0x41: 'vrecip.bf16', 0x42: 'vexp.bf16', 0x43: 'vexp2.bf16', 0x44: 'vpack.bf16.fp8', 0x45: 'vunpack.fp8.bf16', 0x46: 'vsquare.bf16',
                 0x47: 'vcube.bf16', 0x48: 'vrelu.bf16', 0x49: 'vsin.bf16', 0x4a: 'vcos.bf16', 0x4b: 'vtanh.bf16', 0x4c: 'vlog2.bf16', 0x4d: 'vsqrt.bf16'}
VLI_MNEMONICS = ('vli.all', 'vli.row', 'vli.col', 'vli.one')
# VPU layout -> per-port register bases of (reads/responses, writes) with the allowed pair offsets.
VPU_BASES = {'binary': (('lhs', 'rhs'), ('dst',), (0, 1)), 'unary': (('src',), ('dst',), (0, 1)), 'col': (('src',), ('dst',), (0, 1)),
             'row': (('src', 'src1'), ('dst', 'dst1'), (0,)), 'pack': (('src',), ('dst',), (0, 1)), 'unpack': (('src',), ('dst',), (0, 1)),
             'vli_pair': ((), ('dst',), (0, 1)), 'vli_single': ((), ('dst',), (0,))}

class BindingError(ValueError):
    pass

def require(condition, message):
    if not condition: raise BindingError(message)

def integer(value, label):
    require(type(value) is int and value >= 0, 'invalid ' + label)
    return value

def finite_stream(stream, issue):
    require(stream.get('anywhere') is False and stream.get('at_completion') is False, 'concrete issue-relative access stream required')
    first, count, age, step = (integer(stream[k], k) for k in ('first', 'count', 'age', 'step'))
    require(0 < count <= 256 and step > 0 and type(stream['write']) is bool, 'unsupported finite access count/step')
    return [(stream['resource'], stream['write'], first + n, issue + age + n * step) for n in range(count)]

def branch_target(index, word):
    """Word index reached by a branch or JAL; the PC counts words and the encoded displacement is halved."""
    if word & 127 == 0x63:
        immediate = (word >> 31) << 12 | ((word >> 7) & 1) << 11 | ((word >> 25) & 63) << 5 | ((word >> 8) & 15) << 1; bits = 13
    else:
        immediate = (word >> 31) << 20 | ((word >> 12) & 255) << 12 | ((word >> 20) & 1) << 11 | ((word >> 21) & 1023) << 1; bits = 21
    return index + ((immediate - (1 << bits) if immediate >> (bits - 1) else immediate) >> 1)

COMMANDS = {m for m, _ in ENGINES.values()} | set(VPU_MNEMONICS.values()) | set(VLI_MNEMONICS)

def is_command(mnemonic):
    return mnemonic in COMMANDS or mnemonic.startswith(('dma.load.ch', 'dma.store.ch')) or '.mxu' in mnemonic

def decoded_operands(word):
    opcode = word & 127; rd = (word >> 7) & 31; rs1 = (word >> 15) & 31; rs2 = (word >> 20) & 31
    result = {'rd': 0, 'rs1': 0, 'rs2': 0, 'immediate': 0, 'release': False}
    immediate = word >> 20; signed = immediate - 4096 if immediate & 2048 else immediate
    if opcode == 0x13 and (word >> 12) & 7 == 0:
        mnemonic = 'addi'; result.update(rd=rd, rs1=rs1, immediate=signed)
    elif opcode == 0x37:
        mnemonic = 'lui'; result.update(rd=rd, immediate=word >> 12)
    elif opcode == 0x67 and (word >> 12) & 7 == 1:
        mnemonic = 'delay'; result.update(immediate=immediate)
    elif opcode == 0x63 and (word >> 12) & 7 in BRANCHES:
        # A branch's target is exported only through its block's successors.
        mnemonic = BRANCHES[(word >> 12) & 7]; result.update(rs1=rs1, rs2=rs2)
    elif opcode == 0x6f and rd == 0:
        mnemonic = 'jal'; result.update(immediate=2 * (branch_target(0, word)))
    elif opcode == 0x07:
        mnemonic = 'vstore' if word & (1 << 13) else 'vload'
        result.update(rd=(word >> 7) & 63, rs1=rs1, immediate=signed)
    elif opcode == 0x6b:
        mnemonic = 'vtrpose.xlu'; result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63)
    elif opcode == 0x57 and word >> 25 in VPU_MNEMONICS:
        mnemonic = VPU_MNEMONICS[word >> 25]; result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63, rs2=(word >> 19) & 63)
    elif opcode == 0x5f and (word >> 13) & 7 < 4:
        mnemonic = VLI_MNEMONICS[(word >> 13) & 7]; result.update(rd=(word >> 7) & 63, immediate=word >> 16)
    elif opcode == 0x03 and (word >> 12) & 7 in (2, 6, 7):
        mnemonic = {2: 'lw', 6: 'seld', 7: 'seli'}[(word >> 12) & 7]; result.update(rd=rd, rs1=rs1, immediate=signed)
    elif opcode == 0x23 and (word >> 12) & 7 == 2:
        stored = (word >> 25) << 5 | (word >> 7) & 31
        mnemonic = 'sw'; result.update(rs1=rs1, rs2=rs2, immediate=stored - 4096 if stored & 2048 else stored)
    elif opcode == 0x77 and word >> 26 < len(MXU_STEMS):
        mnemonic = f'{MXU_STEMS[word >> 26]}.mxu{(word >> 25) & 1}'
        result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63, rs2=(word >> 19) & 63)
    elif opcode == 0x7b and word >> 25 in (0, 1):
        mnemonic = 'dma.' + ('store' if word >> 25 else 'load') + '.ch' + str((word >> 12) & 7)
        result.update(rd=rd, rs1=rs1, rs2=rs2)
    elif opcode == 0x7f and word >> 25 in (0, 1):
        mnemonic = 'dma.' + ('wait' if word >> 25 else 'config') + '.ch' + str((word >> 12) & 7)
        if not word >> 25: result.update(rs1=rs1)
    elif opcode == 0x73 and (word >> 12) & 7 == 1:
        mnemonic = 'csrrw'; result.update(rd=rd, rs1=rs1, immediate=immediate, release=immediate == 0xC10)
    elif word == 0x73:
        mnemonic = 'ecall'
    else: raise BindingError('unsupported word in finite export binding')
    return mnemonic, result

def check_header(export, facts_sha256, pairs_sha256):
    require(export.get('schema') == SCHEMA and export.get('provider', {}).get('id') == PROVIDER, 'export schema/provider mismatch')
    require(export['provider'].get('dma_policy') == 'wait', 'DMA wait policy required')
    require(export.get('evidence', {}).get('op_timing_sha256') == facts_sha256, 'export was not resolved from the selected facts')
    require(export['evidence'].get('op_pairs_sha256') == pairs_sha256, 'export was not resolved from the selected pair facts')
    for item in export['instructions']:
        require((item['mnemonic'], item['operands']) == decoded_operands(item['word_u32']), 'exported instruction metadata differs from selected word')

def bind_command(item, command, waits):
    """Bind one observed command to its exported footprint, independently of other commands."""
    mnemonic, operands, footprint, issue, kind = item['mnemonic'], item['operands'], item['footprint'], command['edge'], command['engine']
    require(command['word_u32'] == item['word_u32'], 'command differs from its exported instruction word')
    if kind == 'dma':
        channel = integer(command['channel'], 'DMA channel'); direction = command['op']
        require(mnemonic == f'dma.{direction}.ch{channel}', 'resolved DMA operation/channel mismatch')
        require(footprint.get('dma_async') is True and footprint.get('completion') == 'matching_dma_wait' and all(footprint.get(k) is None for k in ('done_age', 'dma_cycles', 'read_release', 'write_release')), 'DMA completion was assigned a fixed latency')
        vmem = [a for a in footprint['accesses'] if a['resource'] == 'vmem']; dram = [a for a in footprint['accesses'] if a['resource'] == 'dram']
        require(len(vmem) == len(dram) == 1, 'explicit DMA VMEM and conservative DRAM effects required')
        stream = vmem[0]
        require(stream.get('at_completion') is True and stream.get('anywhere') is False and stream.get('age') is None and stream.get('step') is None and stream.get('lifetime') == 'launch_through_matching_wait', 'DMA VMEM effect lacks dynamic completion lifetime')
        # The DRAM span is the captured 32-bit offset in 32-byte units; its completion stays dynamic.
        require(dram[0].get('at_completion') is True and dram[0].get('anywhere') is False and dram[0].get('age') is None and dram[0].get('step') is None and
                (dram[0]['first'], dram[0]['count']) == ((command['dram_address'] & 0xffffffff) // 32, command['size'] // 32), 'DMA DRAM span differs from the captured transfer or lost its dynamic completion')
        require(stream['write'] == (direction == 'load') and dram[0]['write'] == (direction == 'store'), 'DMA effect direction mismatch')
        require(stream['first'] == command['vmem_word'] // 8 and stream['count'] == command['size'] // 32, 'resolved DMA captured VMEM range mismatch')
        memory = command['memory']
        require([e['line'] for e in memory] == list(range(stream['first'], stream['first'] + stream['count'])), 'observed DMA VMEM lines differ from exported range')
        match = [e for e in waits if e['channel'] == channel and e['edge'] == command['release_edge']]
        require(len(match) == 1 and match[0]['word_index'] > command['word_index'] and match[0]['edge'] > issue, 'no matching WAIT after selected launch')
        require(all(issue <= e['edge'] < match[0]['edge'] for e in memory + command['requests'] + command['responses']), 'DMA effects do not drain before actual matching WAIT')
        return len(memory), {'word_index': command['word_index'], 'channel': channel, 'launch_edge': issue, 'matching_wait_edge': match[0]['edge'], 'observed_elapsed_edges': match[0]['edge'] - issue}
    require(kind in ENGINES and mnemonic == (ENGINES[kind][0] or command.get('mnemonic')), 'resolved operation differs from the observed engine')
    mreg = lambda field, write: [('mreg', write, 32 * e['id'] + e['row'], e['edge']) for e in command[field] if e.get('resource', 'mreg') == 'mreg']
    vmem_field = None; mxu = kind.startswith('mxu')
    if mxu:
        unit = int(kind[3:])
        push, matmul = (command.get('acc'), command.get('mreg'), 0), (command.get('acc'), command.get('mreg'), command.get('slot'))
        expected = {0: (command.get('slot'), command.get('mreg'), 0), 1: push, 2: push, 3: (command.get('mreg'), command.get('scale_reg'), command.get('acc')),
                    4: (command.get('mreg'), 0, command.get('acc')), 5: matmul, 6: matmul}.get(command.get('op'))
        require(expected is not None and (operands['rd'], operands['rs1'], operands['rs2']) == expected, 'resolved MXU operands mismatch')
        slots = lambda field, write: [(e['resource'], write, (2 * unit + e['id']) * 32 + e['row'], e['edge']) for e in command[field] if e['resource'] in ('acc', 'weight')]
        points = mreg('reads', False) + mreg('writes', True) + slots('reads', False) + slots('writes', True)
        # A matmul's slot-wide weight ownership (step 0) must cover its observed compute beats.
        owned = [a for a in footprint['accesses'] if a['resource'] == 'weight' and a['step'] == 0]
        beats = [e['edge'] - issue for e in command['compute']]
        require(len(owned) == (2 if beats else 0) and all(not a['write'] and a['count'] == 32 and a['first'] == (2 * unit + command['slot']) * 32 for a in owned) and
                (not beats or min(a['age'] for a in owned) <= min(beats) and max(a['age'] for a in owned) >= max(beats)), 'weight-slot ownership does not cover the observed compute beats')
    elif kind in ('vload', 'vstore'):
        require(operands['rd'] == command['mreg'], 'resolved VLS register mismatch')
        vmem_field = 'reads' if kind == 'vload' else 'writes'
        points = [('vmem', kind == 'vstore', e['id'], e['edge']) for e in command[vmem_field]] + mreg('writes' if kind == 'vload' else 'reads', kind == 'vload')
    elif kind == 'xlu':
        require(operands['rd'] == command['dst'] and operands['rs1'] == command['src'], 'resolved XLU operands mismatch')
        points = mreg('reads', False) + mreg('writes', True)
    else:
        require((operands['rd'], operands['rs1'], operands['rs2']) == (command['dst'], command['lhs'], command['rhs']) and
                (operands['immediate'] == command['imm'] or not mnemonic.startswith('vli')), 'resolved VPU operands mismatch')
        sources, dests, offsets = VPU_BASES[command['layout']]
        registers = {**command, 'src1': command['src'] + 1, 'dst1': command['dst'] + 1}
        for field in ('reads', 'responses', 'writes'):
            bases = [registers[b] for b in (dests if field == 'writes' else sources)]
            for e in command[field]:
                require(type(e.get('port')) is int and 0 <= e['port'] < len(bases) and e['id'] - bases[e['port']] in offsets, 'VPU event lacks its explicit register port')
        points = mreg('reads', False) + mreg('writes', True)
    exported = []
    for stream in footprint['accesses']:
        if stream['resource'] in ('mreg', 'vmem') or stream['resource'] in ('acc', 'weight') and stream['step']: exported += finite_stream(stream, issue)
        else: require(stream['resource'] in ('xreg', 'ereg') or mxu and stream['resource'] == 'weight', 'unbound access resource ' + str(stream['resource']))
    require(sorted(exported) == sorted(points), 'observed access resource/address/age differs from provider footprint')
    # Each port's responses answer its reads in order, one edge later (a column reduction reads its rows twice).
    sources = [e for e in command['reads'] if e.get('resource', 'mreg') in ('mreg', 'vmem')]
    require(len(sources) == len(command['responses']), 'incomplete source stream')
    for port in {e.get('port', 0) for e in sources + command['responses']}:
        reads, responses = ([e for e in field if e.get('port', 0) == port] for field in (sources, command['responses']))
        require(len(reads) == len(responses) and all((r['id'], r['row'], r['edge'] + 1, r['data_hex']) == (s['id'], s['row'], s['edge'], s['data_hex'])
                                                     for r, s in zip(reads, responses)), 'read response differs from observed one-cycle source stream')
    age = lambda event: event['edge'] - issue
    release = integer(command['release_edge'] - issue, 'observed release age')
    activity = max(map(age, command['reads'] + command['responses'] + command['writes'] + command.get('compute', [])))
    own = [h for h in footprint['holds'] if h['unit'] == ENGINES[kind][1] and h['index'] == (int(kind[3]) if mxu else 0)]
    # A VPU admits its next command at next_issue_age, before it releases: its VPU-unit holds may end there,
    # and policy holds may start there, provided they last through the activity.
    require(len(own) == 1 and all(h['from'] == 0 and (release == integer(h['to'], 'hold end') + 1 or kind == 'vpu' and h['to'] < release) for h in own),
            'observed engine release differs from its own-unit hold end + 1')
    issued = own[0]['to'] + 1 if kind == 'vpu' else 0
    if kind == 'vpu':
        clear = integer(command['next_issue_age'], 'observed VPU issue-busy clear age')
        # VectorFSM canIssueOp: a companion can add slot/logic restrictions, delaying this opcode's clear.
        require(clear > 0 and (issued <= clear if command['next_issue_companion'] else issued == clear),
                'observed VPU issue-busy clear differs from its own-unit hold end + 1')
    vmem_ages = [age(e) for e in command[vmem_field]] if vmem_field else []
    for hold in footprint['holds']:
        if any(hold is h for h in own): continue
        if hold['unit'] in ENGINE_UNITS:
            head = hold['unit'] == 'VPU' and hold['from'] == 0 and hold['to'] + 1 == issued
            require(head or hold['from'] in (0, issued) and integer(hold['to'], 'hold end') >= release - 1, 'policy ' + hold['unit'] + ' hold ends before the engine activity')
        elif hold['unit'] == 'VMEM bank': require(vmem_ages and hold['from'] <= min(vmem_ages) and hold['to'] >= max(vmem_ages), 'VMEM bank hold does not cover the observed bank accesses')
        else: raise BindingError('unbound hold unit ' + str(hold['unit']))
    sources = [e for e in command['reads'] + command['responses'] if e.get('resource', 'mreg') == 'mreg'] if kind != 'vload' else []
    dests = [e for e in command['writes'] if e.get('resource', 'mreg') == 'mreg'] if kind != 'vstore' else []
    require(sorted(footprint['mreg_reads']) == sorted({e['id'] for e in sources}) and sorted(footprint['mreg_writes']) == sorted({e['id'] for e in dests}), 'exported MREG lifetime registers differ from observed streams')
    require(footprint['read_release'] >= max(map(age, sources), default=0) and footprint['write_release'] >= max(map(age, dests), default=0) and integer(footprint['done_age'], 'done age') >= max(activity, release - 1), 'exported lifetime ends before an observed access')
    return len(points), None

def bind_scale_load(item, load):
    """A SELD's exported VMEM read and scale-register write must land at its observed LSU command and write ages."""
    require(item['mnemonic'] == 'seld' and item['word_u32'] == load['word_u32'] and item['operands']['rd'] == load['rd'], 'scale load differs from its exported SELD')
    effects = sorted((a['resource'], a['write'], a['first'], a['count'], a['age']) for a in item['footprint']['accesses'] if a['resource'] != 'xreg')
    require(effects == sorted([('vmem', False, load['address'] >> 5, 1, load['command_age']), ('ereg', True, load['rd'], 1, load['write_age'])]),
            'observed SELD VMEM read or scale write differs from its exported footprint')
    return 2

def bind_scalar_memory(item, access):
    """An LW's VMEM read and register write, or an SW's VMEM write, must land at its observed LSU command and write ages."""
    require(item['mnemonic'] == access['kind'] and item['word_u32'] == access['word_u32'], 'scalar memory access differs from its exported instruction')
    effects = sorted((a['resource'], a['write'], a['first'], a['count'], a['age']) for a in item['footprint']['accesses'] if a['resource'] != 'xreg' or a['write'])
    expected = [('vmem', access['kind'] == 'sw', access['address'] >> 5, 1, access['command_age'])]
    if access['kind'] == 'lw' and access['rd']: expected.append(('xreg', True, access['rd'], 1, access['write_age']))
    require(effects == sorted(expected), 'observed scalar VMEM access or register write differs from its exported footprint')
    return len(expected)

def context_items(export, block, registers):
    """The block's instructions with the footprints of each entry-context group the observed entry registers hold."""
    first = block['first_word']; items = export['instructions'][first:first + block['word_count']]
    if 'entry_contexts' not in block:
        require(all(item['footprint'] is not None for item in items), 'instruction lacks its exported footprint')
        return [items]
    groups = [group for group in block['entry_contexts'] if any(all(registers.get(r) == v for r, v in entry.items()) for entry in group['entries'])]
    require(groups and all(len(group['footprints']) == len(items) for group in groups), 'observed block entry matches no exported entry context')
    return [[{**item, 'footprint': footprint} for item, footprint in zip(items, group['footprints'])] for group in groups]

def bind_blocks(export, words):
    blocks, instructions = export['blocks'], export['instructions']
    starts = {block['first_word']: block['index'] for block in blocks}; first = 0
    for number, block in enumerate(blocks):
        require(block['index'] == number and block['first_word'] == first and integer(block['word_count'], 'block size') > 0, 'exported blocks do not partition the program')
        end = first + block['word_count']
        require(end <= len(words) and all(instructions[i]['block'] == number for i in range(first, end)), 'instruction block differs from the block partition')
        if block['exit'] == 'halt':
            require(words[end - 1] == 0x73 and block['successors'] == [] and block['successor_issue_offset'] is None, 'halting block shape mismatch')
        else:
            integer(block['successor_issue_offset'], 'successor issue offset')
            if block['exit'] == 'branch_with_delay_slot':
                redirect = words[end - 2]
                require(end - first >= 2 and redirect & 127 in (0x63, 0x6f), 'block lacks its exported branch')
                successors = [branch_target(end - 2, redirect)] + ([end] if redirect & 127 == 0x63 else [])
            else:
                require(block['exit'] == 'fall_through', 'unknown block exit')
                successors = [end]
            require(all(s in starts for s in successors) and sorted(block['successors']) == sorted({starts[s] for s in successors}), 'exported block successors differ from the words')
        first = end
    require(first == len(words), 'exported blocks do not partition the program')
    return blocks, starts

def bind_export(export, observed, facts_sha256, words, pairs_sha256=None):
    """Check export and observed events agree; returns a small summary."""
    check_header(export, facts_sha256, pairs_sha256)
    require(export.get('program', {}).get('words') == words and export['program'].get('word_count') == len(words), 'export differs from emitted words')
    instructions = export['instructions']
    require(len(instructions) == len(words) and all(item['word_index'] == i and item['word_u32'] == words[i] for i, item in enumerate(instructions)), 'incomplete resolved instruction export')
    blocks, starts = bind_blocks(export, words)
    halts = [item for item in observed['endpoints'] if item['kind'] == 'halt']
    require(len(halts) == 1 and instructions[halts[0]['word_index']].get('event_kind') == 'terminal_acceptance', 'one terminal acceptance required')
    waits = {(e['word_index'], e['edge']): e for e in observed['endpoints'] if e['kind'] == 'wait'}
    require(len(waits) == sum(e['kind'] == 'wait' for e in observed['endpoints']), 'duplicate observed WAIT')
    instances, fired, owner, previous, consumed = [], [], {}, None, 0
    for event in observed['instructions'] + halts:
        word, edge = event['word_index'], integer(event['edge'], 'observed issue edge')
        require(0 <= word < len(words) and event['word_u32'] == words[word], 'resolved/observed instruction word mismatch')
        item = instructions[word]; block = blocks[item['block']]
        if previous is None: require(word == 0, 'execution does not enter at word 0')
        else:
            require(edge > previous[1], 'actual instruction edges do not advance')
            last = blocks[instructions[previous[0]]['block']]
            if previous[0] == last['first_word'] + last['word_count'] - 1:
                require(word in starts and starts[word] in last['successors'], 'dynamic successor is not an exported block successor')
                require(edge - instances[-1]['entry_edge'] >= last['successor_issue_offset'], 'successor issued before the exported drained exit')
            else: require(word == previous[0] + 1, 'execution leaves its block before the block end')
        if word == block['first_word']: instances.append({'block': block['index'], 'entry_edge': edge, 'epoch_origins': [edge]})
        instance = instances[-1]; epoch = len(instance['epoch_origins']) - 1
        current, offset = integer(item['issue_epoch'], 'issue epoch'), integer(item['epoch_offset'], 'epoch offset')
        require(edge - instance['entry_edge'] >= integer(item['minimum_issue_cycle'], 'minimum issue cycle'), 'observed issue precedes exported block-relative lower bound')
        if item.get('event_kind') == 'matching_wait_acceptance':
            require((word, edge) in waits and current == epoch + 1 and offset == 0, 'matching WAIT does not start the next epoch')
            instance['epoch_origins'].append(edge); consumed += 1
        else: require((word, edge) not in waits and current == epoch, 'epoch changes without matching WAIT acceptance')
        require(edge == instance['epoch_origins'][current] + offset, 'observed intra-epoch issue differs from exported block offset')
        fired.append((word, edge)); owner[word, edge] = instance; previous = (word, edge)
    require(consumed == len(waits), 'observed WAIT outside an exported matching wait')
    require(item['mnemonic'] == 'ecall' and block['exit'] == 'halt', 'terminal acceptance convention mismatch')
    entries = observed['block_entries']
    require([(e['word_index'], e['edge']) for e in entries] == [(blocks[i['block']]['first_word'], i['entry_edge']) for i in instances], 'observed block entries differ from exported block instances')
    require(not any(any(e['engine_state'].values()) for e in entries), 'engines not idle at a block entry')
    keys = [(c['word_index'], c['edge']) for c in observed['commands']]
    require(len(set(keys)) == len(keys) and set(keys) == {f for f in fired if is_command(instructions[f[0]]['mnemonic'])}, 'observed commands differ from executed compute instructions')
    # Each dynamic instance binds to the footprints of every entry-context group its observed entry registers hold.
    for instance, entry in zip(instances, entries):
        instance['groups'] = context_items(export, blocks[instance['block']], entry.get('registers', {}))
    def bound(word, edge, bind):
        instance = owner[word, edge]; offset = word - blocks[instance['block']]['first_word']
        results = [bind(group[offset]) for group in instance['groups']]
        return results[0]
    checked, intervals = 0, []
    for command in observed['commands']:
        elements, interval = bound(command['word_index'], command['edge'], lambda item: bind_command(item, command, list(waits.values())))
        checked += elements; intervals += [interval] if interval else []
    loads = observed.get('scale_loads', [])
    require(sorted((l['word_index'], l['edge']) for l in loads) == sorted(f for f in fired if instructions[f[0]]['mnemonic'] == 'seld'), 'observed scale loads differ from executed SELDs')
    checked += sum(bound(l['word_index'], l['edge'], lambda item: bind_scale_load(item, l)) for l in loads)
    accesses = observed.get('scalar_memory', [])
    require(sorted((a['word_index'], a['edge']) for a in accesses) == sorted(f for f in fired if instructions[f[0]]['mnemonic'] in ('lw', 'sw')), 'observed scalar memory accesses differ from executed LW/SW')
    checked += sum(bound(a['word_index'], a['edge'], lambda item: bind_scalar_memory(item, a)) for a in accesses)
    for instance in instances: instance['entry_context_groups'] = len(instance.pop('groups'))
    return {'instruction_words_bound': len(fired), 'block_instances': instances, 'memory_access_elements_bound': checked, 'dynamic_dma_intervals': intervals}

def bind_footprints(export, observed, facts_sha256, pairs_sha256=None):
    """Bind each observed command to the same command of a straight-line reference export, without issue offsets."""
    check_header(export, facts_sha256, pairs_sha256)
    items = [item for item in export['instructions'] if is_command(item['mnemonic'])]
    require(len(export['blocks']) == 1 and len(items) == len(observed['commands']), 'reference export differs from observed commands')
    waits = [e for e in observed['endpoints'] if e['kind'] == 'wait']
    checked, intervals = 0, []
    for item, command in zip(items, observed['commands']):
        elements, interval = bind_command(item, command, waits)
        checked += elements; intervals += [interval] if interval else []
    seld, loads = [i for i in export['instructions'] if i['mnemonic'] == 'seld'], observed.get('scale_loads', [])
    require(len(seld) == len(loads), 'reference export differs from observed scale loads')
    checked += sum(bind_scale_load(item, load) for item, load in zip(seld, loads))
    return {'instruction_words_bound': 0, 'block_instances': [], 'memory_access_elements_bound': checked, 'dynamic_dma_intervals': intervals}

class ReplayError(Exception):
    pass

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
    'pimm': ('vpu.io_cmd_bits_imm', 16), 'pscale': ('vpu.io_cmd_bits_scaleE8M0', 8),
    # Scalar LSU (LW, SW, SELD), the register or scale register write a load ends in, and SELI's scale write.
    'scmd': ('lsu.io_scalarCmd_valid', 1), 'scmd_store': ('lsu.io_scalarCmd_bits_isStore', 1), 'scmd_addr': ('lsu.io_scalarCmd_bits_byteAddr', 21),
    'scmd_wdata': ('lsu.io_scalarCmd_bits_wdata', 32), 'scmd_wmask': ('lsu.io_scalarCmd_bits_wmask', 4),
    'xw_en': ('scalar.regfile.io_wr_en', 1), 'xw_addr': ('scalar.regfile.io_wr_addr', 5), 'xw_data': ('scalar.regfile.io_wr_data', 32),
    'sresp': ('lsu.io_scalarResp_valid', 1), 'sresp_data': ('lsu.io_scalarResp_bits', 32),
    'sew': ('scalar.scaleRF.io_writeEn', 1), 'sew_idx': ('scalar.scaleRF.io_writeIdx', 5), 'sew_data': ('scalar.scaleRF.io_writeData', 8),
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
            'valid', 'xcmd', 'xr', 'xresp', 'xw', 'xar', 'xaw', 'pcmd', 'scmd', 'sresp', 'sew', 'xw_en'] + [f'{p}{n}' for n in range(2) for p in ('pr', 'presp', 'pw')] + [f'{p}{n}' for n in range(4) for p in ('par', 'paw')]
CONTROLS += ['dma_' + k for k in ['launch', 'stall', 'av', 'ar', 'dv', 'dr', 'vr', 'vrg', 'vw', 'vwg', 'vresp']] + [f'dma_{p}{n}' for n in range(8) for p in ('busy', 'engine')]

SIGNALS.update({'vpu_busy': ('vpu.io_issueBusy', 31), 'mreg_read_busy': ('scalar.io_mregReadBusy', 64), 'mreg_write_busy': ('scalar.io_mregWriteBusy', 64)})
CONTROLS += ['vpu_busy', 'mreg_read_busy', 'mreg_write_busy']

# MXU u: command, MREG ports, sequencer weight/compute/accumulator ports and port-engine state.
MXU_INFLIGHT = (3, 2)  # in-flight matmul tracker depth (SystolicArraySequencer / InnerProductTreesSequencer)
for u in range(2):
    m, s = f'mxu{u}.', f'mxu{u}.seq.'
    group = {'cmd': (m + 'io_cmd_valid', 1), 'op': (m + 'io_cmd_bits_op', 3), 'mreg': (m + 'io_cmd_bits_mregId', 6),
             'acc': (m + 'io_cmd_bits_accSel', 1), 'slot': (m + 'io_cmd_bits_weightSlot', 1), 'scale': (m + 'io_cmd_bits_scaleE8M0', 8),
             'al_acc': (s + 'io_accLoadReq_bits_accSel', 1), 'al_row': (s + 'io_accLoadReq_bits_rowIdx', 5),
             'ar_acc': (s + 'io_accComputeReadAddr_accSel', 1), 'ar_row': (s + 'io_accComputeReadAddr_rowIdx', 5),
             'ww': (s + 'io_weightWriteReq_valid', 1), 'ww_slot': (s + 'io_weightWriteReq_bits_weightSlot', 1), 'ww_lane': (s + 'io_weightWriteReq_bits_laneIdx', 5),
             'cv': (s + 'io_compute_valid', 1), 'cacc': (s + 'io_compute_bits_accumulate', 1), 'cslot': (s + 'io_compute_bits_weightBufSel', 1),
             'aw': (s + 'io_accComputeWrite_valid', 1), 'aw_acc': (s + 'io_accComputeWrite_bits_accSel', 1), 'aw_row': (s + 'io_accComputeWrite_bits_rowIdx', 5),
             'ar': (s + 'io_accComputeReadEn', 1), 'al': (s + 'io_accLoadReq_valid', 1),
             'as': (s + 'io_accStoreReadEn', 1), 'as_acc': (s + 'io_accStoreAddr_accSel', 1), 'as_row': (s + 'io_accStoreAddr_rowIdx', 5),
             'p0v': (s + 'p0CmdValid', 1), 'p1v': (s + 'p1CmdValid', 1), 'w0v': (s + 'w0CmdValid', 1), 'w1v': (s + 'w1CmdValid', 1), 'p0op': (s + 'p0Cmd_op', 3)}
    for p in range(2):
        group.update({f'r{p}': (m + f'io_mregReadReq{p}_valid', 1), f'rid{p}': (m + f'io_mregReadReq{p}_bits_mregId', 6), f'rrow{p}': (m + f'io_mregReadReq{p}_bits_row', 5),
                      f'data{p}': (m + f'io_mregReadResp{p}_bits', 256), f'w{p}': (m + f'io_mregWriteReq{p}_valid', 1), f'wid{p}': (m + f'io_mregWriteReq{p}_bits_mregId', 6),
                      f'wrow{p}': (m + f'io_mregWriteReq{p}_bits_row', 5), f'wdata{p}': (m + f'io_mregWriteReq{p}_bits_data', 256),
                      f'actr{p}': (m + f'io_activeReads_{p}_valid', 1), f'actrid{p}': (m + f'io_activeReads_{p}_bits', 6),
                      f'actw{p}': (m + f'io_activeWrites_{p}_valid', 1), f'actwid{p}': (m + f'io_activeWrites_{p}_bits', 6)})
    for k in range(32):
        group.update({f'wwd{k}': (s + f'io_weightWriteReq_bits_data_{k}', 8), f'act{k}': (s + f'io_compute_bits_act_{k}', 8),
                      f'awd{k}': (s + f'io_accComputeWrite_bits_data_{k}', 16), f'ald{k}': (s + f'io_accLoadReq_bits_data_{k}', 16),
                      f'psum{k}': (s + f'io_compute_bits_psum_{k}', 16)})
    group.update({f'inf{i}': (s + f'inflightValid_{i}', 1) for i in range(MXU_INFLIGHT[u])})
    SIGNALS.update({f'm{u}_{k}': value for k, value in group.items()})
    CONTROLS += [f'm{u}_{k}' for k in ('cmd', 'ww', 'cv', 'aw', 'ar', 'al', 'as', 'p0v', 'p1v', 'w0v', 'w1v', 'r0', 'r1', 'w0', 'w1', 'actr0', 'actr1', 'actw0', 'actw1')]
    CONTROLS += [f'm{u}_inf{i}' for i in range(MXU_INFLIGHT[u])]

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

# Engine activity: a command releases at the first edge after its issue where its engine shows none.
BUSY = {'vload': lambda v: v('load_busy'), 'vstore': lambda v: v('store_busy'), 'xlu': lambda v: v('xar') | v('xaw'),
        'vpu': lambda v: int(bool(v('vpu_busy') or any(v(p + str(n)) for p in ('par', 'paw') for n in range(4))))}
# MXU data/compute busy as SystolicArraySequencer drives io_dataBusy | io_computeBusy (not traced): a port engine or a tracked matmul.
for u in range(2):
    BUSY[f'mxu{u}'] = lambda v, u=u: int(any(v(f'm{u}_{k}') for k in ('p0v', 'p1v', 'w0v', 'w1v', *(f'inf{i}' for i in range(MXU_INFLIGHT[u])))))
STAGES = ('reads', 'responses', 'writes')
EVENTS = {'vload': (32, 32, 32), 'vstore': (32, 32, 32), 'xlu': (32, 32, 32)}
# Replayed MXU operations: MxuOp code -> (mnemonic stem, (reads, responses, writes, compute beats)); a matmul.acc also reads its accumulator.
MXU_OPS = dict(enumerate(zip(MXU_STEMS, ((32, 32, 32, 0), (32, 32, 32, 0), (64, 64, 32, 0),
                                             (32, 0, 32, 0), (32, 0, 64, 0), (32, 32, 32, 32), (64, 32, 32, 32)))))
# MXU op -> {(stage, port): events}: ports as `element` numbers them (2: matmul.acc accumulator read, 3: accumulator load).
MXU_PORTS = {0: {('reads', 1): 32, ('responses', 1): 32, ('writes', 0): 32}, 1: {('reads', 1): 32, ('responses', 1): 32, ('writes', 3): 32},
             2: {('reads', 0): 32, ('reads', 1): 32, ('responses', 0): 32, ('responses', 1): 32, ('writes', 3): 32},
             3: {('reads', 0): 32, ('writes', 0): 32}, 4: {('reads', 0): 32, ('writes', 0): 32, ('writes', 1): 32},
             5: {('reads', 0): 32, ('responses', 0): 32, ('writes', 0): 32},
             6: {('reads', 0): 32, ('reads', 2): 32, ('responses', 0): 32, ('writes', 0): 32}}
UNCHECKED = object()  # a write whose value the validator cannot predict (matmul results; checked through pop and final memory)
# VPU funct7 -> (ScalarISA VPU_* code on vpu.io_cmd_bits_op, selected mnemonic, layout); VLI is keyed by its mode.
VPU_OPS = {funct7: (code, VPU_MNEMONICS[funct7], layout) for layout, codes in {
    'binary': {0: 1, 2: 2, 3: 3, 4: 25, 6: 24},
    'unary': {0x40: 26, 0x41: 4, 0x42: 10, 0x43: 11, 0x46: 12, 0x47: 13, 0x48: 19,
              0x49: 6, 0x4a: 7, 0x4b: 8, 0x4c: 9, 0x4d: 5},
    'row': {0x21: 14, 0x24: 21, 0x26: 20}, 'col': {1: 15, 5: 23, 7: 22}, 'pack': {0x44: 17}, 'unpack': {0x45: 18},
}.items() for funct7, code in codes.items()}
VLI_MODES = {mode: (code, VLI_MNEMONICS[mode], 'vli_pair' if mode < 2 else 'vli_single')
             for mode, code in enumerate((30, 29, 28, 27))}
# Layout -> per-port (reads, writes) counts as VectorFSM sequences them; responses equal reads.
VPU_PORTS = {'binary': ((64, 64), (64, 0)), 'unary': ((64, 0), (64, 0)), 'row': ((32, 32), (32, 32)), 'col': ((128, 0), (64, 0)),
             'pack': ((64, 0), (32, 0)), 'unpack': ((32, 0), (64, 0)), 'vli_pair': ((0, 0), (64, 0)), 'vli_single': ((0, 0), (32, 0))}

def event_counts(command):
    return command['counts'] if 'counts' in command else EVENTS[command['engine']] + (0,)
# valid signal -> (engine, stage, port, address signals, data signal); VMEM addresses are (bank, row).
PORT_EVENTS = {'vr': ('vload', 'reads', 0, ('vr_bank', 'vr_addr'), None), 'vresp': ('vload', 'responses', 0, None, 'vresp_data'),
               'mw': ('vload', 'writes', 0, ('mw_id', 'mw_row'), 'mw_data'), 'mr': ('vstore', 'reads', 0, ('mr_id', 'mr_row'), None),
               'mresp': ('vstore', 'responses', 0, None, 'mresp_data'), 'vw': ('vstore', 'writes', 0, ('vw_bank', 'vw_addr'), 'vw_data'),
               'xr': ('xlu', 'reads', 0, ('xrid', 'xrrow'), None), 'xresp': ('xlu', 'responses', 0, None, 'xdata'), 'xw': ('xlu', 'writes', 0, ('xwid', 'xwrow'), 'xwdata')}
for n in range(2):
    PORT_EVENTS.update({f'pr{n}': ('vpu', 'reads', n, (f'prid{n}', f'prrow{n}'), None), f'presp{n}': ('vpu', 'responses', n, None, f'pdata{n}'),
                        f'pw{n}': ('vpu', 'writes', n, (f'pwid{n}', f'pwrow{n}'), f'pwdata{n}')})
# MREG responses are derived one edge after each read. Vector data: (signal prefix, lanes, bytes per lane).
# Accumulator ports 2/3 carry matmul.acc reads / acc-push writes; acc/weight ids are slots.
RESOURCE = {}
for u in range(2):
    m, kind = f'm{u}_', f'mxu{u}'
    for n in range(2):
        PORT_EVENTS.update({m + f'r{n}': (kind, 'reads', n, (m + f'rid{n}', m + f'rrow{n}'), None), m + f'resp{n}': (kind, 'responses', n, None, m + f'data{n}'),
                            m + f'w{n}': (kind, 'writes', n, (m + f'wid{n}', m + f'wrow{n}'), m + f'wdata{n}')})
        RESOURCE.update({m + f'r{n}': 'mreg', m + f'w{n}': 'mreg'})
    PORT_EVENTS.update({m + 'ww': (kind, 'writes', 0, (m + 'ww_slot', m + 'ww_lane'), (m + 'wwd', 32, 1)),
                        m + 'aw': (kind, 'writes', 0, (m + 'aw_acc', m + 'aw_row'), (m + 'awd', 32, 2)),
                        m + 'as': (kind, 'reads', 0, (m + 'as_acc', m + 'as_row'), None),
                        m + 'ar': (kind, 'reads', 2, (m + 'ar_acc', m + 'ar_row'), None),
                        m + 'al': (kind, 'writes', 3, (m + 'al_acc', m + 'al_row'), (m + 'ald', 32, 2))})
    RESOURCE.update({m + 'ww': 'weight', m + 'aw': 'acc', m + 'as': 'acc', m + 'ar': 'acc', m + 'al': 'acc'})

def element(command, stage, port, n):
    """(resource, id, row) of the n-th event of a stage on a port; VMEM ids are lines, acc/weight ids slots."""
    engine = command['engine']
    if engine.startswith('mxu'):
        op, reads = command['op'], stage == 'reads'
        if op == 0: return ('mreg', command['mreg'], n) if reads and port == 1 else ('weight', command['slot'], n) if not reads and port == 0 else None
        if op in (1, 2):  # an FP8 acc push reads on port 1 (preferred when free), a BF16 push reads the pair on both ports
            return ('mreg', command['mreg'] + port * (op == 2), n) if reads and port in ((1,) if op == 1 else (0, 1)) else ('acc', command['acc'], n) if not reads and port == 3 else None
        if op in (5, 6): return ('mreg', command['mreg'], n) if reads and port == 0 else ('acc', command['acc'], n) if port == (2 if reads else 0) and (op == 6 or not reads) else None
        return ('acc', command['acc'], n) if reads and port == 0 else ('mreg', command['mreg'] + port, n) if not reads and port < (2 if op == 4 else 1) else None
    if engine in ('vload', 'vstore'):
        return ('vmem', command['line'] + n, n) if (stage == 'writes') == (engine == 'vstore') else ('mreg', command['mreg'], n)
    if engine == 'xlu': return 'mreg', command['dst' if stage == 'writes' else 'src'], n
    layout, (reads, writes) = command['layout'], VPU_PORTS[command['layout']]
    if n >= (writes if stage == 'writes' else reads)[port]: return None
    if stage == 'writes':
        if layout == 'row': return 'mreg', command['dst'] + port, n
        return ('mreg', command['dst'], n) if layout in ('pack', 'vli_single') else ('mreg', command['dst'] + n // 32, n % 32)
    if layout == 'binary': return 'mreg', (command['lhs'], command['rhs'])[port] + n // 32, n % 32
    if layout == 'row': return 'mreg', command['src'] + port, n
    if layout == 'unpack': return 'mreg', command['src'], n
    return 'mreg', command['src'] + (n // 32) % 2, n % 32

def produced(command, n, port=0):
    """Data the n-th write must carry, from the command's own observed responses; None if not yet available."""
    data = command['_data']
    if command['engine'].startswith('mxu'):
        op = command['op']
        try:
            if op in (5, 6): return UNCHECKED
            if op == 0: return data[1][n] if n < len(data[1]) else None
            if op == 1: return MXU.dequantize(data[1][n]) if n < len(data[1]) else None
            if op == 2: return data[0][n] + data[1][n] if n < min(len(data[0]), len(data[1])) else None
            if op == 3: return MXU.quantize(data[0][n], command['scale']) if n < len(data[0]) else None
            return data[0][n][32 * port:32 * port + 32] if n < len(data[0]) else None
        except ValueError as error: raise ReplayError(f'{command["mnemonic"]} data outside its exact reference: {error}')
    if command['engine'] in ('vload', 'vstore'): return data[0][n] if n < len(data[0]) else None
    if command['engine'] == 'xlu': return bytes(data[0][c][n] for c in range(32)) if len(data[0]) == 32 else None
    layout, name = command['layout'], command['mnemonic']
    try:
        if layout == 'binary':
            if n >= min(len(data[0]), len(data[1])): return None
            return bf16_power_product(data[0][n], data[1][n]) if name == 'vmul.bf16' else VPU.elementwise(name, data[0][n], data[1][n])
        if layout == 'unary': return VPU.elementwise(name, data[0][n]) if n < len(data[0]) else None
        if layout == 'row': return VPU.row_reduce(name, data[0][n], data[1][n]) if n < min(len(data[0]), len(data[1])) else None
        if layout == 'col': return VPU.column_reduce(name, data[0][:64]) if len(data[0]) >= 64 else None
        if layout == 'pack': return VPU.fp8_pack(data[0][2 * n], data[0][2 * n + 1], command['scale']) if 2 * n + 1 < len(data[0]) else None
        if layout == 'unpack': return VPU.fp8_unpack(data[0][n // 2], n % 2, command['scale']) if n // 2 < len(data[0]) else None
        return VPU.vli(name[4:], command['imm'], n)
    except ValueError as error: raise ReplayError(f'{name} data outside its exact reference: {error}')

def leaders(words):
    """Static block starts, as the compiler forms them: entry, branch targets, after delay slots and ECALL."""
    result = {0}
    for index, word in enumerate(words):
        if word & 127 in (0x63, 0x6f): result |= {branch_target(index, word), index + 2}
        elif word == 0x73: result.add(index + 1)
    return result & set(range(len(words)))

def memories(case):
    vm, dram = {}, {}
    for first, data in case.vmem.items():
        if not data or len(data) % 32: raise ReplayError('VMEM region is not whole lines')
        for r in range(len(data) // 32):
            if first + r in vm: raise ReplayError('overlapping VMEM regions')
            vm[first + r] = bytes(data[32 * r:32 * r + 32])
    for base, data in case.dram.items():
        for i, value in enumerate(data):
            if base + i in dram: raise ReplayError('overlapping DRAM regions')
            dram[base + i] = value
    expected_vm, expected_dram = dict(vm), dict(dram)
    case.expect(expected_vm, expected_dram)
    if expected_vm.keys() != vm.keys() or expected_dram.keys() != dram.keys() or any(len(line) != 32 for line in expected_vm.values()):
        raise ReplayError('case expectation escapes its initialized memory')
    return vm, dram, expected_vm, expected_dram

def image(case):
    """Testbench input: initial and expected bytes of every initialized region, and the DMA beat counts."""
    _, _, vm, dram = memories(case)
    lines = [f'dma {case.dma[0]} {case.dma[1]}']
    for first, data in case.vmem.items():
        lines += [f'vmem {first} {data.hex()}', f'expect_vmem {first} {CASES.tile(vm, first, len(data) // 32).hex()}']
    for base, data in case.dram.items():
        lines += [f'dram {base} {data.hex()}', f'expect_dram {base} {bytes(dram[base + i] for i in range(len(data))).hex()}']
    return '\n'.join(lines) + '\n'

def quiescent_state(v,d,busy,engine):
    return {'vls_load_busy':v('load_busy'),'vls_store_busy':v('store_busy'),
            'xlu_read_active':v('xar'),'xlu_write_active':v('xaw'), 'vpu_issue_busy':v('vpu_busy'),
            'vpu_read_mask':sum(v('par'+str(n))<<n for n in range(4)),
            'vpu_write_mask':sum(v('paw'+str(n))<<n for n in range(4)),
            'mreg_read_busy':v('mreg_read_busy'),'mreg_write_busy':v('mreg_write_busy'),
            **{f'mxu{u}_busy':BUSY[f'mxu{u}'](v) for u in range(2)},
            **{f'mxu{u}_active_mask':sum(v(f'm{u}_act{d}{p}')<<(2*(d=='w')+p) for d in 'rw' for p in range(2)) for u in range(2)},
            'dma_scalar_mask':busy,'dma_engine_mask':engine, 'dma_a_valid':d('av'),'dma_d_valid':d('dv')}

def validate_compute_edges(samples,words,case):
    vm,dram,expected_vm,expected_dram=memories(case);mreg={};mxu={};prior_reads={};starts=leaders(words)
    commands=[];instructions=[];endpoints=[];busy_events=[];entries=[];scale_loads=[]
    regs=[None]*32;regs[0]=0;base=0;next_base=None;pc_expected=0;slot=False;redirect=None;scale=[0]*32;scalar_load=None;scalar_store=None;seli=None;scalar_memory=[]
    # DMA: pending maps channel -> launched command; the engine requests commands in launch order (fifo).
    live={};pending={};fifo=[];outstanding={};read_queue=[];a_hold=None;d_hold=None
    began=False;ended=False;marker=None;publication=None;last_edge=0;prev_busy=None;delayed=False
    def reject(message): raise ReplayError(message)
    def row_bytes(identity,row):
        if (identity,row) not in mreg: reject('read from uninitialized logical MREG row')
        return mreg[identity,row]
    def vm_line(line):
        if line not in vm: reject('VMEM access outside the case-initialized lines')
        return vm[line]
    # VectorFSM has two slots; binary/row commands own both port pairs. Events use command-relative ports.
    def live_commands(kind): return live.get(kind+'_drain',[])+([live[kind]] if kind in live else [])
    def vpu_ports(c): return {0,1} if c['layout'] in ('binary','row') else {c['slot']}
    def vpu_left(c,stage,port):
        logical=port-c['slot']
        if not 0<=logical<2: return 0
        return VPU_PORTS[c['layout']][stage=='writes'][logical]-sum(e['port']==logical for e in c[stage])
    # A slot frees on the edge its last write fires (VectorFSM done1/done2).
    def vpu_done(c):
        return not any(vpu_left(c,s,p) for s in STAGES[:2] for p in range(2)) and all(vpu_left(c,'writes',p)<=(1 if v('pw'+str(p)) else 0) for p in range(2))
    def vpu_released(c):
        return not any(vpu_left(c,s,p) for s in STAGES for p in range(2)) and not any(v('paw'+str(n)) and v('pawid'+str(n)) in c['dests'] for n in range(4))
    # The sequencer retires matmuls in order; attribute each port/resource event to its oldest unfinished command.
    def mxu_left(c,stage,port): return MXU_PORTS[c['op']].get((stage,port),0)-sum(e.get('port',0)==port for e in c[stage])
    def mxu_released(c,u):
        others={r for o in live_commands(f'mxu{u}') if o is not c for r in o['sources']|o['dests']}
        done=all(mxu_left(c,s,p)==0 for s,p in MXU_PORTS[c['op']]) and len(c['compute'])==MXU_OPS[c['op']][1][3]
        return done and not any(v(f'm{u}_act{d}{p}') and v(f'm{u}_act{d}id{p}') in c['sources' if d=='r' else 'dests']-others for d in 'rw' for p in range(2))
    for sample in samples:
        edge=sample['edge'];last_edge=edge
        derived={f'm{u}_resp{p}':prior_reads.get((u,p),0) for u in range(2) for p in range(2)}
        prior_reads={(u,p):sample['values'].get(f'm{u}_r{p}') for u in range(2) for p in range(2)}
        sample={'edge':edge,'values':{**sample['values'],**derived}}
        def v(key):
            value=sample['values'][key]
            if type(value)!=int or not 0<=value<(1<<(SIGNALS[key][1] if key in SIGNALS else 1)): reject('unknown/out-of-range compute signal '+key)
            return value
        def d(key): return v('dma_'+key)
        def data(key):
            if isinstance(key,tuple): return b''.join(v(f'{key[0]}{k}').to_bytes(key[2],'little') for k in range(key[1]))
            return v(key).to_bytes(32,'little')
        def record(collection,**fields): collection.append({'edge':edge,**fields})
        def entry(pc):
            state=quiescent_state(v,d,busy,engine)
            if live or pending or outstanding or read_queue or scalar_load or scalar_store or any(state.values()): reject('block entry is not quiescent')
            record(entries,word_index=pc,engine_state=state,registers={f'x{r}':regs[r] for r in range(1,32) if regs[r] is not None})
        if v('reset'):
            if began: reject('reset interrupted compute envelope')
            continue
        for key in CONTROLS: v(key)
        if v('illegal') or v('ebreak'): reject('unexpected scalar termination')
        busy=sum(d('busy'+str(p))<<p for p in range(8));engine=sum(d('engine'+str(p))<<p for p in range(8))
        if prev_busy!=(busy,engine): record(busy_events,scalar_mask=busy,engine_mask=engine);prev_busy=(busy,engine)
        if not began and not v('halt'):
            began=True
            if busy or engine or d('av') or d('dv') or v('cmd') or v('xcmd') or v('pcmd') or v('m0_cmd') or v('m1_cmd') or any(f(v) for f in BUSY.values()): reject('nonquiescent observed execution entry')
            if d('base')!=0: reject('START did not reset DMA base')
        if next_base is not None:
            if d('base')!=next_base: reject('CONFIG publication differs from captured scalar base')
            base=next_base;next_base=None
        if began and not ended and d('base')!=base: reject('unexpected global DMA base change')
        if engine&~busy: reject('engine busy omitted from scalar bridge')
        if busy&~sum(1<<c for c in pending): reject('busy without observed pending DMA')
        if any(edge>c['edge'] and len(c['responses'])<c['beats'] and not busy>>ch&1 for ch,c in pending.items()): reject('pending DMA lost busy protection')
        # Capture the first post-issue clear of each opcode bit, even if its command has already drained.
        for command in commands:
            if command['engine']=='vpu' and command['next_issue_age'] is None and edge>command['edge'] and not v('vpu_busy')>>command['op']&1:
                command['next_issue_age']=edge-command['edge']
        # Commands with company release once their events and exclusive live markers drain.
        for kind in ('vpu','mxu0','mxu1'):
            for drain in list(live.get(kind+'_drain',())):
                if vpu_released(drain) if kind=='vpu' else mxu_released(drain,int(kind[3])):
                    if kind!='vpu' and tuple(len(drain[s]) for s in STAGES)+(len(drain['compute']),)!=event_counts(drain):
                        reject(kind+' released before its events completed')
                    drain['release_edge']=edge;live[kind+'_drain'].remove(drain)
                    if not live[kind+'_drain']: del live[kind+'_drain']
        for kind in [k for k in live if not k.endswith('_drain')]:
            command=live[kind];company=kind+'_drain' in live
            if edge>command['edge'] and (vpu_released(command) if kind=='vpu' and company else mxu_released(command,int(kind[3])) if kind.startswith('mxu') and company else not BUSY[kind](v)):
                if tuple(len(command[s]) for s in STAGES)+(len(command.get('compute',())),)!=event_counts(command): reject(kind+' released before its events completed')
                command['release_edge']=edge;del live[kind]
        for kind,active in BUSY.items():
            if kind not in live and kind+'_drain' not in live and active(v): reject('unbound '+kind+' activity')
        expected_cmd={'cmd':False,'xcmd':False,'pcmd':False,'dma_launch':False,'m0_cmd':False,'m1_cmd':False}
        new_scalar_load=None;new_scalar_store=None;scalar_write=None
        if v('valid') and began and not ended:
            pc=v('pc');word=v('word')
            if pc!=pc_expected or pc>=len(words) or word!=words[pc]: reject('retired/held instruction differs from selected program words')
            if v('fire'):
                if pc in starts: entry(pc)
                record(instructions,word_index=pc,word_u32=word)
                in_slot,target=slot,redirect;slot,redirect=False,None
                opcode=word&127;delayed=opcode==0x67 and (word>>12)&7==1;rd=(word>>7)&31;rs1=(word>>15)&31;rs2=(word>>20)&31;funct3=(word>>12)&7
                common={'edge':edge,'word_index':pc,'word_u32':word,'reads':[],'responses':[],'writes':[]}
                created=None
                if opcode==0x13 and funct3==0:
                    immediate=word>>20;immediate=immediate-4096 if immediate&2048 else immediate
                    if regs[rs1] is None: reject('unknown scalar ADDI operand')
                    if rd: regs[rd]=(regs[rs1]+immediate)&0xffffffff;scalar_write=(rd,regs[rd])
                elif opcode==0x37:
                    if rd: regs[rd]=word&0xfffff000;scalar_write=(rd,regs[rd])
                elif opcode==0x67 and funct3==1:
                    pass # DELAY is independently bound through its actual held PC/word.
                elif opcode==0x63 and funct3 in (0,1,4,5,6,7) or opcode==0x6f and rd==0:
                    if in_slot: reject('control flow in a delay slot')
                    taken=True
                    if opcode==0x63:
                        if regs[rs1] is None or regs[rs2] is None: reject('unknown branch operand')
                        a,b=regs[rs1],regs[rs2]
                        if funct3 in (4,5): a,b=a-(a>>31<<32),b-(b>>31<<32)
                        taken={0:a==b,1:a!=b,4:a<b,5:a>=b,6:a<b,7:a>=b}[funct3]
                    slot=True;redirect=branch_target(pc,word) if taken else None
                elif opcode==0x03 and funct3 in (2,6):
                    # LW/SELD: scalar LSU reads the aligned VMEM word; it becomes register rd, or its low byte scale register rd.
                    immediate=word>>20;immediate=immediate-4096 if immediate&2048 else immediate
                    if regs[rs1] is None: reject('unknown scalar load operand')
                    # Capture operands before old writeback, then install this load after attributing that write.
                    new_scalar_load={'edge':edge,'word_index':pc,'word_u32':word,'address':(regs[rs1]+immediate)&0xffffffff,'rd':rd,'kind':'seld' if funct3==6 else 'lw'}
                    if new_scalar_load['address']%4: reject('unaligned scalar load')
                    vm_line(new_scalar_load['address']>>5)
                elif opcode==0x03 and funct3==7:
                    seli=(rd,(word>>20)&255)
                elif opcode==0x23 and funct3==2:
                    immediate=((word>>25)<<5)|((word>>7)&31);immediate=immediate-4096 if immediate&2048 else immediate
                    if regs[rs1] is None or regs[rs2] is None: reject('unknown scalar store operand')
                    # A previous SW may commit on this edge while the next captures operands for the following edge.
                    new_scalar_store={'edge':edge,'word_index':pc,'word_u32':word,'address':(regs[rs1]+immediate)&0xffffffff,'data':regs[rs2],'kind':'sw'}
                    if new_scalar_store['address']%4: reject('unaligned scalar store')
                    vm_line(new_scalar_store['address']>>5)
                elif opcode==0x07:
                    store=bool(word&(1<<13));identity=(word>>7)&63
                    immediate=word>>20;immediate=immediate-4096 if immediate&2048 else immediate
                    if regs[rs1] is None: reject('unknown VLS scalar base')
                    line=((regs[rs1]+32*immediate)&0xffffffff)>>3 &65535
                    if line%32 or any(line+r not in vm for r in range(32)): reject('VLS tile outside the case-initialized lines')
                    if (v('op'),v('mreg'),v('line'))!=(2 if store else 1,identity,line): reject('VLS command does not match decoded instruction operands')
                    created={**common,'engine':'vstore' if store else 'vload','mreg':identity,'line':line};expected_cmd['cmd']=True
                elif opcode==0x6b:
                    src=(word>>13)&63;dst=(word>>7)&63
                    if (v('xsrc'),v('xdst'))!=(src,dst): reject('XLU command differs from selected operands')
                    created={**common,'engine':'xlu','src':src,'dst':dst};expected_cmd['xcmd']=True
                elif opcode in (0x57,0x5f):
                    vd=(word>>7)&63;vs1=(word>>13)&63;vs2=(word>>19)&63
                    spec=VPU_OPS.get(word>>25) if opcode==0x57 else VLI_MODES.get((word>>13)&7)
                    if spec is None: reject('unsupported VPU operation in the replay')
                    code,mnemonic,layout=spec
                    if (v('pop'),v('psrc0'),v('psrc1'),v('pdst'),v('pimm'),v('pscale'))!=(code,vs1,vs2,vd,word>>16,scale[vs1&31]): reject('VPU command differs from the decoded instruction operands')
                    src=vs2 if layout in ('pack','unpack') else vs1;lhs,rhs=(0,0) if opcode==0x5f else (vs1,vs2)
                    if any(r%2 for r in {'binary':(vs1,vs2,vd),'pack':(src,),'unpack':(vd,),'vli_pair':(vd,),'vli_single':()}.get(layout,(src,vd))): reject('VPU pair operand is not even')
                    (reads,writes),pair=VPU_PORTS[layout],lambda r:{r,r+1}
                    created={**common,'engine':'vpu','mnemonic':mnemonic,'op':code,'layout':layout,'dst':vd,'lhs':lhs,'rhs':rhs,'src':src,
                             'next_issue_age':None,
                             'scale':scale[vs1&31],'imm':word>>16,'counts':(sum(reads),sum(reads),sum(writes),0),
                             'sources':{'binary':pair(vs1)|pair(vs2),'unpack':{src},'vli_pair':set(),'vli_single':set()}.get(layout,pair(src)),
                             'dests':{vd} if layout in ('pack','vli_single') else pair(vd)};expected_cmd['pcmd']=True
                elif opcode==0x77:
                    op,unit=(word>>25)>>1,(word>>25)&1;vd=(word>>7)&63;vs1=(word>>13)&63;vs2=(word>>19)&63;u=f'm{unit}_'
                    if op not in MXU_OPS: reject('unsupported MXU operation in the replay')
                    pop=op in (3,4)
                    fields={'mreg':vd if pop else vs1,'acc':vs2&1 if pop else vd&1,'slot':vd&1 if op==0 else vs2&1,'scale':scale[vs1&31] if op==3 else 0}
                    if op==4 and (vd%2 or vs1): reject('unsupported BF16 pop operands')
                    checked=('mreg','slot') if op==0 else ('mreg','acc','slot') if op in (5,6) else ('mreg','acc')
                    if v(u+'op')!=op or any(v(u+k)!=fields[k] for k in checked+('scale',)): reject('MXU command differs from the decoded instruction operands')
                    stem,counts=MXU_OPS[op]
                    created={**common,'engine':f'mxu{unit}','mnemonic':f'{stem}.mxu{unit}','op':op,**fields,'counts':counts,'compute':[],
                             'sources':set() if pop else {vs1,vs1+1} if op==2 else {vs1},'dests':{vd,vd+1} if op==4 else {vd} if op==3 else set(),
                             **({'scale_reg':vs1} if op==3 else {})};expected_cmd[u+'cmd']=True
                elif opcode==0x7b:
                    store=bool(word>>25);channel=(word>>12)&7
                    vmreg=rs1 if store else rd;addressreg=rd if store else rs1
                    if channel in pending or any(regs[r] is None for r in (vmreg,addressreg,rs2)): reject('overlapping/unknown DMA launch')
                    address=(base<<32)|regs[addressreg];vmword=regs[vmreg];size=regs[rs2];beats=size//32
                    if (d('address'),d('vmem'),d('size'),d('launch_channel'),d('op'))!=(address,vmword,size,channel,2 if store else 1): reject('DMA launch differs from captured scalar operands')
                    if size%32 or not 0<beats<=64 or vmword%8 or any(vmword//8+r not in vm for r in range(beats)) or any(address+i not in dram for i in range(size)): reject('DMA launch outside the case-initialized memory')
                    pending[channel]={**common,'engine':'dma','op':'store' if store else 'load','channel':channel,'vmem_word':vmword,'dram_address':address,'size':size,'beats':beats,'requests':[],'memory':[],'memory_responses':[],'wait_stall_edges':[]}
                    fifo.append(pending[channel]);commands.append(pending[channel]);expected_cmd['dma_launch']=True
                elif opcode==0x7f:
                    channel=(word>>12)&7
                    if word>>25:
                        command=pending.get(channel)
                        if command is None or busy>>channel&1 or any(o['command'] is command for o in outstanding.values()) or any(q[1] is command for q in read_queue) or any(len(command[k])!=command['beats'] for k in ('requests','responses','memory')) or (command['op']=='store' and len(command['memory_responses'])!=command['beats']): reject('WAIT precedes matching DMA/data drain')
                        command['release_edge']=edge;record(endpoints,kind='wait',channel=channel,word_index=pc,word_u32=word);del pending[channel]
                    else:
                        if regs[rs1] is None or regs[rs1]>31 or d('rs1')!=regs[rs1]: reject('unknown/mismatched DMA configuration')
                        next_base=regs[rs1];record(endpoints,kind='config',base=next_base,word_index=pc,word_u32=word)
                elif not (opcode==0x73 and funct3==1): reject('unsupported scalar instruction in finite compute trace')
                if created:
                    previous=live.get(created['engine'])
                    if created['engine']=='vpu':
                        # Beside live VPU work the issue-busy bit must be clear and the command's port pairs free:
                        # held by no command, or by one whose last write fires on this edge.
                        if live_commands('vpu') and v('vpu_busy')>>created['op']&1: reject('VPU command issued while its issue-busy bit is set')
                        held={p for c in live_commands('vpu') if not vpu_done(c) for p in vpu_ports(c)}
                        both=created['layout'] in ('binary','row')
                        created['slot']=next((s for s in (0,1) if not held&({0,1} if both else {s})),None)
                        if created['slot'] is None: reject('VPU command issued while its port pairs are live')
                        if previous: live.setdefault('vpu_drain',[]).append(live.pop('vpu'))
                    elif previous and created['engine'].startswith('mxu'):
                        # The earlier command drains beside the new one; the binder holds every event to its isolated footprint.
                        live.setdefault(created['engine']+'_drain',[]).append(live.pop(created['engine']))
                    elif previous: reject('command issued while its engine is live')
                    created.update(_queue=([],[]),_data=([],[],[]));live[created['engine']]=created;commands.append(created)
                pc_expected=target if in_slot and target is not None else pc+1
            elif (v('word')&127)==0x7f and v('word')>>25 and pending:
                # A WAIT is held by its busy channel, or behind a preceding DELAY once its transfer has already drained.
                channel=(v('word')>>12)&7
                if channel not in pending or not d('stall') or not busy>>channel&1 and not delayed: reject('invalid dynamically held matching WAIT')
                if busy>>channel&1: pending[channel]['wait_stall_edges'].append(edge)
        for key,expected in expected_cmd.items():
            if bool(v(key))!=expected: reject('unbound or missing decoded command '+key)
        if v('launch')!=v('cmd'): reject('scalar/LSU launch mismatch')
        # LW/SELD: command (byte address), response (aligned word), then the register or scale register write; reads at issue
        # see the old value. SW: one masked word write with its command. SELI writes its scale register as it fires.
        if v('scmd'):
            if v('scmd_store'):
                if not scalar_store or v('scmd_addr')!=scalar_store['address']&(1<<21)-1 or (v('scmd_wdata'),v('scmd_wmask'))!=(scalar_store['data'],15): reject('unbound or mismatched scalar store command')
                offset=scalar_store['address']&28;line=vm_line(scalar_store['address']>>5)
                vm[scalar_store['address']>>5]=line[:offset]+scalar_store['data'].to_bytes(4,'little')+line[offset+4:]
                scalar_memory.append({**scalar_store,'command_edge':edge});scalar_store=None
            else:
                if not scalar_load or 'word' in scalar_load or v('scmd_addr')!=scalar_load['address']&(1<<21)-1: reject('unbound or mismatched scalar memory command')
                offset=scalar_load['address']&28;scalar_load.update(word=int.from_bytes(vm_line(scalar_load['address']>>5)[offset:offset+4],'little'),command_edge=edge)
        if v('sresp'):
            if not scalar_load or 'word' not in scalar_load or 'response_edge' in scalar_load or v('sresp_data')!=scalar_load['word']: reject('unbound or mismatched scalar memory response')
            scalar_load['response_edge']=edge
        load_write=False
        if scalar_load and scalar_load['kind']=='lw' and 'response_edge' in scalar_load and (not scalar_load['rd'] or v('xw_en') and v('xw_addr')==scalar_load['rd']):
            if scalar_load['rd'] and v('xw_data')!=scalar_load['word']: reject('scalar load writes a different register value')
            load_write=bool(scalar_load['rd'])
            regs[scalar_load['rd']]=scalar_load['word'] if scalar_load['rd'] else 0
            scalar_memory.append({**scalar_load,'write_edge':edge if scalar_load['rd'] else None});scalar_load=None
        if v('xw_en') and not load_write and (v('xw_addr'),v('xw_data'))!=scalar_write:
            reject('unattributed or mismatched scalar register write')
        if v('sew'):
            if seli:
                if (v('sew_idx'),v('sew_data'))!=seli: reject('mismatched SELI scale register write')
                scale[seli[0]]=seli[1];seli=None
            else:
                if not scalar_load or scalar_load['kind']!='seld' or 'response_edge' not in scalar_load or (v('sew_idx'),v('sew_data'))!=(scalar_load['rd'],scalar_load['word']&255): reject('unbound or mismatched scale register write')
                scale[scalar_load['rd']]=scalar_load['word']&255;scale_loads.append({**scalar_load,'write_edge':edge});scalar_load=None
        if seli: reject('SELI without its scale register write')
        if new_scalar_load:
            if scalar_load: reject('overlapping scalar load before previous writeback')
            scalar_load=new_scalar_load
            if scalar_load['kind']=='lw' and scalar_load['rd']: regs[scalar_load['rd']]=None
        if new_scalar_store:
            if scalar_store: reject('overlapping scalar store before previous command')
            scalar_store=new_scalar_store
        for stage in STAGES:
            for key,(kind,where,port,address,carried) in PORT_EVENTS.items():
                if where!=stage or not v(key): continue
                command=live.get(kind)
                if kind=='vpu': command=next((c for c in live_commands('vpu') if vpu_left(c,stage,port)>0),None)
                elif kind.startswith('mxu') and stage=='responses': command=next((c for c in live_commands(kind) if c['_queue'][port]),None)
                elif kind.startswith('mxu'):
                    command=next((c for c in live_commands(kind) if mxu_left(c,stage,port)>0 and (element(c,stage,port,0) or ('',))[0]==RESOURCE[key]),None)
                if command is None: reject(f'unattributed {key} event: no live {kind} command')
                if kind=='vpu': port-=command['slot']
                n=sum(e.get('port',0)==port for e in command[stage])
                if len(command[stage])>=event_counts(command)[STAGES.index(stage)]: reject(f'excess {kind} {stage} event')
                if stage=='responses':
                    if not command['_queue'][port]: reject(f'{kind} response without an outstanding read')
                    resource,identity,row,observed=command['_queue'][port].pop(0)
                    if data(carried)!=observed: reject(f'{kind} read response data mismatch')
                    command['_data'][port].append(observed)
                else:
                    place=element(command,stage,port,n)
                    if place is None or key in RESOURCE and place[0]!=RESOURCE[key]: reject(f'unsupported {kind} {key} event')
                    resource,identity,row=place
                    if tuple(v(k) for k in address)!=((identity>>13,identity&8191) if resource=='vmem' else (identity,row)): reject(f'{kind} {stage} address mismatch')
                    if stage=='reads' and resource=='acc':
                        observed=mxu.get((kind,'acc',identity,row)) or reject('read from an unwritten accumulator row')
                        command['_data'][port].append(observed)
                    elif stage=='reads':
                        observed=vm_line(identity) if resource=='vmem' else row_bytes(identity,row)
                        command['_queue'][port].append((resource,identity,row,observed))
                    else:
                        observed=data(carried);expected=produced(command,n,port)
                        if expected is not UNCHECKED and observed!=expected: reject(f'{kind} written data differs from its observed sources')
                        if resource=='vmem': vm_line(identity);vm[identity]=observed
                        elif resource=='mreg': mreg[identity,row]=observed
                        else: mxu[kind,resource,identity,row]=observed
                record(command[stage],id=identity,row=row,data_hex=observed.hex(),**({'port':port} if kind=='vpu' or kind.startswith('mxu') else {}),
                       **({'resource':RESOURCE.get(key,'mreg')} if kind.startswith('mxu') else {}))
        for u in range(2):
            kind,m=f'mxu{u}',f'm{u}_';command=next((c for c in live_commands(f'mxu{u}') if c['op'] in (5,6) and len(c['compute'])<32),None)
            if v(m+'cv'):
                if not command or command['op'] not in (5,6) or len(command['compute'])>=32: reject('unattributed or excess MXU compute beat')
                n=len(command['compute']);act=data((m+'act',32,1))
                if n>=len(command['_data'][0]) or act!=command['_data'][0][n] or v(m+'cslot')!=command['slot'] or v(m+'cacc')!=(command['op']==6): reject('MXU compute beat differs from its activation row, slot or mode')
                if command['op']==6 and (n>=len(command['_data'][2]) or data((m+'psum',32,2))!=command['_data'][2][n]): reject('MXU accumulate beat differs from its accumulator row')
                if any((kind,'weight',command['slot'],lane) not in mxu for lane in range(32)): reject('matmul reads an unwritten weight slot')
                record(command['compute'],row=n)
            for p in range(2):
                for direction,allowed in (('r','sources'),('w','dests')):
                    if v(f'{m}act{direction}{p}') and v(f'{m}act{direction}id{p}') not in {r for c in live_commands(f'mxu{u}') for r in c[allowed]}: reject('MXU live resource operand mismatch')
        # Live markers name the active command's operands; they need not cover the launch edge.
        active=live.get('xlu')
        if v('xar') and (not active or v('xarid')!=active['src']) or v('xaw') and (not active or v('xawid')!=active['dst']): reject('XLU active operand mismatch')
        for port in range(4):
            for direction in ('par','paw'):
                if v(direction+str(port)):
                    actives=live_commands('vpu') or reject('unbound VPU live resource marker')
                    if v(direction+'id'+str(port)) not in {r for c in actives for r in c['sources' if direction=='par' else 'dests']}: reject('VPU live resource operand mismatch')
        # DMA: A requests follow launch order; responses carry their source's command; a load response is written to VMEM in the
        # edge it is accepted (D ready is the VMEM write grant), and a store reads VMEM for the command being requested.
        head=next((c for c in fifo if len(c['requests'])<c['beats']),None);accepted=None
        if d('av'):
            request=(d('ao'),d('aa'),d('as'),d('ad'))
            if a_hold is not None and request!=a_hold: reject('DMA A changed under backpressure')
            if not d('ar'): a_hold=request
            else:
                a_hold=None
                if head is None: reject('unbound DMA A request')
                op,address,source,datum=request;n=len(head['requests']);payload=datum.to_bytes(32,'little')
                if op!=(0 if head['op']=='store' else 4) or address!=head['dram_address']+32*n or source in outstanding: reject('DMA beat/address/source mismatch')
                if op==0:
                    if n>=len(head['memory_responses']) or payload.hex()!=head['memory_responses'][n]['data_hex']: reject('DMA store data differs from its VMEM source')
                    for i in range(32): dram[address+i]=payload[i]
                outstanding[source]={'command':head,'edge':edge,'row':n,'data':None if op==0 else bytes(dram[address+i] for i in range(32))}
                record(head['requests'],source=source,address=address,op=op,data_hex=payload.hex())
        elif a_hold is not None: reject('DMA A valid dropped under backpressure')
        if d('dv'):
            response=(d('ds'),d('dd'))
            if d_hold is not None and response!=d_hold: reject('DMA D changed under backpressure')
            source,datum=response
            if source not in outstanding: reject('unbound DMA response')
            request=outstanding[source];owner=request['command']
            if not d('dr'): d_hold=response
            else:
                d_hold=None;del outstanding[source];observed=datum.to_bytes(32,'little');accepted=request
                if edge-request['edge']<43 or owner['op']=='load' and observed!=request['data']: reject('DMA delayed load response mismatch')
                record(owner['responses'],source=source,row=request['row'],data_hex=observed.hex())
        elif d_hold is not None: reject('DMA D valid dropped under backpressure')
        if d('vw'):
            meta=accepted or (outstanding.get(d('ds')) if d('dv') else None)
            if meta is None or meta['command']['op']!='load' or bool(d('vwg'))!=(accepted is not None): reject('unbound DMA VMEM write')
            owner,row=meta['command'],meta['row'];line=owner['vmem_word']//8+row
            if (d('vwb'),d('vwa'))!=(line>>13,line&8191) or d('vwd')!=d('dd'): reject('DMA VMEM write address/data mismatch')
            if accepted: vm_line(line);vm[line]=d('vwd').to_bytes(32,'little');record(owner['memory'],kind='vw',line=line,row=row,data_hex=vm[line].hex())
        elif accepted and accepted['command']['op']=='load': reject('DMA load response without its VMEM write')
        if d('vr'):
            if head is None or head['op']!='store' or len(head['memory'])>=head['beats']: reject('unbound DMA VMEM read')
            row=len(head['memory']);line=head['vmem_word']//8+row
            if (d('vrb'),d('vra'))!=(line>>13,line&8191): reject('DMA VMEM request address mismatch')
            if d('vrg'): read_queue.append((edge+1,head,row,vm_line(line)));record(head['memory'],kind='vr',line=line,row=row,data_hex=vm[line].hex())
        expected_response=bool(read_queue and read_queue[0][0]==edge)
        if bool(d('vresp'))!=expected_response: reject('DMA VMEM read response timing mismatch')
        if expected_response:
            _,owner,row,expected=read_queue.pop(0);observed=d('vdata').to_bytes(32,'little')
            if observed!=expected: reject('DMA VMEM source response mismatch')
            record(owner['memory_responses'],row=row,data_hex=observed.hex())
        if publication is not None:
            if edge!=publication or v('marker')!=1: reject('marker publication did not follow write')
            record(endpoints,kind='marker_publication');publication=None
        if v('csr'):
            if live or pending or outstanding or scalar_load or scalar_store or marker is not None or (v('csr_addr'),v('csr_op'),v('csr_data'))!=(0xc10,1,1): reject('marker precedes observed engine drain')
            marker=edge;publication=edge+1;record(endpoints,kind='marker',word_index=v('pc'),word_u32=v('word'),engine_state=quiescent_state(v,d,busy,engine))
        if v('ecall'):
            if not began or ended or live or pending or outstanding or read_queue or scalar_load or slot or marker is None or v('marker')!=1 or not v('halt') or v('fire') or v('pc')!=pc_expected or pc_expected>=len(words) or v('word')!=words[pc_expected] or words[pc_expected]!=0x73: reject('terminal precedes complete selected-word/engine drain')
            if pc_expected in starts: entry(pc_expected)
            ended=True;record(endpoints,kind='halt',word_index=v('pc'),word_u32=v('word'),engine_state=quiescent_state(v,d,busy,engine))
        elif began and not ended and v('halt'): reject('unexpected halt in compute envelope')
    if not ended or live or pending or outstanding or read_queue or scalar_load or scalar_store or publication is not None: reject('incomplete compute/terminal drain trace')
    halt_edge=next(e['edge'] for e in endpoints if e['kind']=='halt')
    if last_edge-halt_edge<2: reject('trace lacks bounded post-halt drain samples')
    if collections.Counter(c['engine'] for c in commands)!=collections.Counter({k:n for k,n in case.commands.items() if n}): reject('wrong dynamic compute command coverage')
    if vm!=expected_vm or dram!=expected_dram: reject('observed composed VMEM/DRAM state differs from the case expectation')
    vpu_commands=[c for c in commands if c['engine']=='vpu']
    for command in vpu_commands:
        if command['next_issue_age'] is None: reject('VPU command lacks an observed issue-busy clear after issue')
        # Samples precede the rising edge: a successor issued on the clear edge cannot delay that clear.
        command['next_issue_companion']=any(other is not command and other['edge']<command['edge']+command['next_issue_age'] and
                                          command['edge']<other['release_edge'] for other in vpu_commands)
    for command in commands:
        for key in [k for k in command if k.startswith('_')]: del command[key]
        for field in ('reads','responses','writes','compute'):
            for event in command.get(field,()): event['age']=event['edge']-command['edge']
        for key in ('sources','dests'):
            if key in command: command[key]=sorted(command[key])
    for load in scale_loads: load.update({f'{k}_age':load[f'{k}_edge']-load['edge'] for k in ('command','response','write')})
    for access in scalar_memory: access.update({f'{k}_age':access[f'{k}_edge']-access['edge'] for k in ('command','response','write') if access.get(f'{k}_edge') is not None})
    concurrent=[[i,j] for i,a in enumerate(commands) for j,b in enumerate(commands[i+1:],i+1) if b['edge']<a['release_edge'] and a['edge']<b['release_edge']]
    return {'instructions':instructions,'block_entries':entries,'commands':commands,'concurrent_commands':concurrent,'endpoints':endpoints,'busy_transitions':busy_events,
            'scale_loads':scale_loads,'scalar_memory':scalar_memory,'sampled_edges':last_edge,'full_memory_checked_bytes':32*len(vm),'terminal_drain_edges':last_edge-halt_edge}

def sh(argv, timeout, env=None):
    result = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode: raise ReplayError(f'{Path(str(argv[0])).name} failed: {(result.stderr or result.stdout)[-2000:]}')
    return result.stdout

def build_model(args, work):
    objdir = work / 'obj'; cxx = shutil.which('g++')
    argv = [args.verilator, '--cc', '--exe', '--build', '--top-module', 'AtlasCore', '--prefix', 'VAtlasCore', '--Mdir', str(objdir), '--trace', '--trace-depth', '3',
            '--assert', '--output-split', '20000', '--output-split-cfuncs', '500', '-Wno-fatal', '-j', str(args.jobs), '-CFLAGS', '-std=c++17',
            '-MAKEFLAGS', f"CXX={cxx} LINK={cxx} AR={shutil.which('ar')}",
            *map(str, sorted([*args.rtl.glob('*.sv'), *args.rtl.glob('*.v')])), str(HERE.parent / 'test/ee290-compute-replay.cpp')]
    sh(argv, args.timeout, {**os.environ, **({'VERILATOR_ROOT': str(args.verilator_root)} if args.verilator_root else {})})
    return objdir / 'VAtlasCore'

def serialized(ops):
    """Delay-free reference of a final stream; each DMA wait moves up to follow its own launch, so one transfer is pending at a time."""
    channel = lambda op: re.search(r'channel = (\d+)', op[1])[1]
    ops = [op for op in ops if op[0] != 'delay']; result = []
    while ops:
        op = ops.pop(0); result.append(op)
        if op[0] == 'dma':
            wait = next((i for i, o in enumerate(ops) if o[0] == 'dma_wait' and channel(o) == channel(op)), None)
            if wait is not None: result.append(ops.pop(wait))
    return result

def replay(args, case, label, consumer, select, digest, pairs_digest, model):
    """Schedule or take one case stream, run it on the model, decode and bind it; returns (result, PASS line)."""
    opt, emit, out = args.bin / 'atlas-opt', args.bin / 'atlas-emit', args.work.resolve() / label
    flags = {'delay': '--insert-atlas-delays', 'schedule': '--schedule-atlas-stream'}
    out.mkdir(exist_ok=True); (out / 'program.mlir').write_text(case.mlir()); program = out / 'program.mlir'
    if case.pipeline:
        # A virtual program: verified, scheduled, allocated and lowered untimed before the facts are selected.
        sh([opt, program, *case.pipeline, '-o', out / 'lowered.mlir'], args.timeout); program = out / 'lowered.mlir'
    if consumer == 'final': shutil.copyfile(program, out / 'final.mlir')
    else: sh([opt, program, select, flags[consumer], '--verify-atlas-timing', '-o', out / 'final.mlir'], args.timeout)
    text = sh([emit, out / 'final.mlir'], args.timeout)
    if not re.fullmatch(r'(?:[0-9a-fA-F]{8}\s+)+', text): raise ReplayError('invalid emitted words')
    (out / 'program.hex').write_text(text); words = [int(w, 16) for w in text.split()]
    (out / 'image.txt').write_text(image(case))
    log = sh([model, out / 'program.hex', out / 'trace.vcd', '100000', out / 'image.txt'], args.timeout)
    match = re.search(r'EE290_COMPUTE_PASSED checked_vmem_bytes=(\d+) checked_dram_bytes=(\d+) reads=(\d+) writes=(\d+) .* status=5 marker=1 illegal_pc=0 cycles=(\d+)', log)
    if not match or (int(match[3]), int(match[4])) != tuple(case.dma): raise ReplayError(label + ': numerical check failed')
    with (out / 'trace.vcd').open() as stream: events = validate_compute_edges(vcd_edges(stream), words, case)
    (out / 'events.json').write_text(json.dumps(events, indent=1))
    if consumer == 'final':
        # The compiler serializes these engines: bind footprints of a delay-inserted reference, not issue offsets.
        (out / 'reference.mlir').write_text(case.mlir(serialized(case.ops)))
        sh([opt, out / 'reference.mlir', select, flags['delay'], '--verify-atlas-timing', '-o', out / 'reference-final.mlir'], args.timeout)
        export = json.loads(sh([emit, '--rtl-timing-json', out / 'reference-final.mlir'], args.timeout))
        binding = bind_footprints(export, events, digest, pairs_digest)
    else:
        export = json.loads(sh([emit, '--rtl-timing-json', out / 'final.mlir'], args.timeout))
        binding = bind_export(export, events, digest, words, pairs_digest)
    (out / 'timing.json').write_text(json.dumps(export, indent=1))
    if case.check and not case.check(events): raise ReplayError(label + ': case check failed')
    return ({'case': label, 'cycles': int(match[5]), **binding},
            f'PASS {label} words={len(words)} blocks_entered={len(events["block_entries"])} accesses_bound={binding["memory_access_elements_bound"]} '
            f'concurrent={len(events["concurrent_commands"])} dma_intervals={len(binding["dynamic_dma_intervals"])}')

def run(args):
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    facts = args.facts.resolve(); digest = hashlib.sha256(facts.read_bytes()).hexdigest()
    if args.facts_sha256 and args.facts_sha256 != digest: raise ReplayError('facts hash mismatch')
    model = args.model.resolve() if args.model else build_model(args, work)
    select = f'--select-atlas-rtl-evidence=op-timing={facts} op-timing-sha256={digest} dma=wait'
    if args.pairs:
        pairs = args.pairs.resolve()
        pairs_digest = hashlib.sha256(pairs.read_bytes()).hexdigest()
        paired = select + f' op-pairs={pairs} op-pairs-sha256={pairs_digest}'
    available = CASES.load(); results = []
    names = available if args.cases == 'all' else args.cases.split(',')
    if args.cases != 'all' and not args.pairs and any(available[name].pairs for name in names): raise ReplayError('selected cases need --pairs')
    for name in [n for n in names if available[n].pairs and not args.pairs]: print(f'SKIP {name}: needs --pairs', flush=True)
    runs = [(available[name], name if consumer == 'final' else f'{name}_{consumer}', consumer)
            for name in names if args.pairs or not available[name].pairs
            for consumer in available[name].consumers if args.consumer in ('both', consumer) or consumer == 'final']
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for (_, label, _), future in zip(runs, [pool.submit(replay, args, case, label, consumer, *((paired, digest, pairs_digest) if case.pairs else (select, digest, None)), model)
                                                for case, label, consumer in runs]):
            try: result, line = future.result()
            except (ReplayError, BindingError) as error: raise type(error)(f'{label}: {error}') from error
            results.append(result); print(line, flush=True)
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--facts', type=Path, required=True, help='merlin.op_timing.v1 facts JSON')
    p.add_argument('--facts-sha256', help='optional expected SHA-256 of --facts')
    p.add_argument('--pairs', type=Path, help='merlin.op_pairs.v1 pair facts derived from --facts; cases marked `pairs` need them')
    p.add_argument('--bin', type=Path, required=True, help='directory holding atlas-opt and atlas-emit')
    p.add_argument('--work', type=Path, required=True, help='output directory')
    p.add_argument('--model', type=Path, help='prebuilt VAtlasCore binary (skips the Verilator build)')
    p.add_argument('--rtl', type=Path, help='directory of AtlasCore *.sv/*.v to Verilate when --model is absent')
    p.add_argument('--verilator', default=shutil.which('verilator') or 'verilator')
    p.add_argument('--verilator-root', type=Path, help='VERILATOR_ROOT override (directory holding include/verilated.h)')
    p.add_argument('--consumer', choices=('delay', 'schedule', 'both'), default='both', help='compiler consumer; final-stream cases always run')
    p.add_argument('--cases', default='all', help='comma list of case names from tools/ee290_replay_cases, or all')
    p.add_argument('--jobs', type=int, default=8, help='parallel case replays and Verilator build jobs')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if not a.model and not a.rtl: p.error('--model or --rtl required')
    if a.cases != 'all' and not set(a.cases.split(',')) <= set(CASES.load()): p.error('unknown case')
    a.bin = a.bin.resolve()
    try: run(a)
    except (ReplayError, BindingError, subprocess.SubprocessError, OSError, KeyError) as error: sys.exit(f'FAIL: {error}')
