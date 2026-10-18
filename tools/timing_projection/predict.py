#!/usr/bin/env python3
"""Tensor-free Atlas schedule timing prototype; no target traces are inputs.

Only compiler JSON plus a declarative memory profile enter predict(). Scalar
address/control registers are interpreted, tensor/scalar data values are not.
Explicit physical delays preserve the compiler's fixed compute schedule. DMA
beats and LSU bank reservations are simulated causally at predicted issue times.
This intentionally does not establish numerical or resource-hazard correctness.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
# Load npu_model's memory backend by path, without importing the package (and Torch).
BACKEND = Path(importlib.util.find_spec('npu_model').submodule_search_locations[0]) / 'hardware/memory_backend.py'
GEOMETRY = {'rows': 32, 'row_bytes': 32, 'vmem_bytes': 1572864,
            'bank_bytes': 262144, 'dma_beat_bytes': 32, 'dma_channels': 8,
            'dma_credits': 64, 'mxu_units': 2}
_spec = importlib.util.spec_from_file_location('atlas_surrogate_memory_backend', BACKEND)
backend_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = backend_module
_spec.loader.exec_module(backend_module)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend(profile):
    params = dict(profile['params'])
    pattern = profile.get('accept_pattern')
    if pattern:
        params['accept_fn'] = lambda c: c % pattern['period'] < pattern['ready_cycles']
    if profile['kind'] == 'fixed':
        return backend_module.FixedLatencyBackend(**params)
    if profile['kind'] == 'curve':
        return backend_module.CurveMemoryBackend(**params)
    raise ValueError(profile)


@dataclass
class Slot:
    channel: int
    is_store: bool
    address: int
    vmem: int
    beats: int
    pc: int
    outstanding: int = 0
    dispatched: bool = False


class Dma:
    """Timing projection of pinned dma.py, retaining grants/credits/source reuse.

    No tensor data, staged execution, logger, instruction objects or architectural
    state. Can ablate bank contention or source/credit constraints explicitly.
    """
    def __init__(self, profile, claims, use_banks=True, use_credits=True):
        self.mem = backend(profile)
        self.claims = claims
        self.use_banks = use_banks
        self.use_credits = use_credits
        self.slots = []
        self.sources = {}
        self.next_source = 0
        self.request_beat = 0
        self.read_beat = 0
        self.read_complete = False
        self.queue = deque()
        self.events = []
        self.flags = []
        self.peak = 0
        self.bank_denials = {'load_write': 0, 'store_read': 0}

    def blocked(self, cycle, bank):
        return self.use_banks and bank in self.claims.get(cycle, set())

    def tick(self, cycle, launch=None):
        slots = list(self.slots)
        dispatched = [s.dispatched for s in slots]
        old_sources = dict(self.sources)
        head = next((s for s in slots if not s.dispatched), None)
        dslot = None
        write_bank = None
        response = self.mem.response(cycle)
        if response is not None:
            s, index, event = self.sources[response.source]
            bank = (s.vmem + index * 32) // 262144
            if s.is_store or not self.blocked(cycle, bank):
                self.mem.pop_response(cycle)
                del self.sources[response.source]
                dslot = s
                event['response_cycle'] = cycle
                if not s.is_store:
                    write_bank = bank
            else:
                self.bank_denials['load_write'] += 1
        read_granted = read_last = False
        if head and head.is_store and not self.read_complete:
            bank = (head.vmem + self.read_beat * 32) // 262144
            if not self.blocked(cycle, bank) and bank != write_bank:
                self.queue.append((head, self.read_beat, cycle + 2))
                read_granted = True
                read_last = self.read_beat == head.beats - 1
            else:
                self.bank_denials['store_read'] += 1
        aslot = None
        a_last = False
        if head:
            ready = not head.is_store or (self.queue and self.queue[0][2] <= cycle)
            credit = not self.use_credits or (len(old_sources) < 64 and self.next_source not in old_sources)
            if ready and credit and self.mem.can_accept(cycle):
                index = self.request_beat
                if head.is_store:
                    entry = self.queue.popleft()
                    assert entry[:2] == (head, index)
                sid = self.next_source
                self.mem.issue(backend_module.BeatRequest(sid, head.address + index * 32, head.is_store, cycle))
                event = {'source': sid, 'pc': head.pc, 'address': head.address + index*32,
                         'is_store': head.is_store, 'issue_cycle': cycle}
                self.events.append(event)
                self.sources[sid] = (head, index, event)
                self.next_source = (sid + 1) % 64 if self.use_credits else sid + 1
                self.request_beat += 1
                aslot = head
                a_last = index == head.beats - 1
        for s, was_dispatched in zip(slots, dispatched):
            s.outstanding += int(s is aslot) - int(s is dslot)
            if was_dispatched and s.outstanding == 0:
                self.slots.remove(s)
                self.flags.append({'cycle': cycle, 'channel': s.channel, 'busy': False})
        if a_last:
            head.dispatched = True
            self.request_beat = 0
            self.read_beat = 0
            self.read_complete = False
        if read_granted and not a_last:
            self.read_beat += 1
            if read_last:
                self.read_complete = True
        if launch:
            assert not any(s.channel == launch.channel for s in self.slots)
            assert len(self.slots) < 8
            self.slots.append(launch)
            self.flags.append({'cycle': cycle, 'channel': launch.channel, 'busy': True})
        self.peak = max(self.peak, len(self.sources))


def signed32(v):
    return v if v < 2**31 else v - 2**32


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_program(program):
    """Fail closed on unsupported syntax/geometry; this is not a hazard verifier."""
    require(program.get('schema') == 'atlas.physical_program.v1', 'unsupported program schema')
    require(program.get('pc_unit') == 'instruction_word', 'unsupported PC geometry')
    require(program.get('timing_state') == 'timed', 'requires a physical timed schedule')
    require(program.get('word_endianness') == 'little', 'unsupported endianness')
    require(program.get('selected_rtl_revision') == '0079c0541111197741a231c002e3843fa6f545b2',
            'unsupported RTL geometry/revision')
    if 'geometry' in program: require(program['geometry'] == GEOMETRY, 'unsupported geometry')
    fields = {
        'dma_config': 'base_reg channel', 'alu_imm': 'dst immediate kind src',
        'upper': 'dst immediate kind', 'dma': 'channel direction dram reg size',
        'dma_wait': 'channel', 'vload': 'base dst format offset', 'vstore': 'base format offset src',
        'delay': 'cycles', 'mxu_push': 'kind slot src unit',
        'mxu_matmul': 'acc_slot accumulate src unit weight_slot',
        'mxu_pop': 'dst format scale_reg slot unit', 'vpu_unary': 'dst kind src',
        'scalar_load': 'base dst kind offset', 'scalar_store': 'base kind offset src',
        'branch': 'kind lhs offset_bytes rhs', 'csr': 'address dst kind source',
        'trap': 'kind', 'vpu_pack': 'direction dst scale_reg src',
        'vpu_reduce': 'dst kind src', 'vpu_binary': 'dst kind lhs rhs', 'vli': 'dst immediate mode',
        'xlu_transpose': 'dst src'}
    kinds = {'alu_imm': {'addi'}, 'upper': {'lui'}, 'mxu_push': {'weight_fp8'},
             'vpu_unary': {'relu','exp','exp2','recip'}, 'scalar_load': {'lw','seli'},
             'scalar_store': {'sw'}, 'branch': {'blt'}, 'csr': {'rrw'},
             'trap': {'ecall'}, 'vpu_reduce': {'row_max','row_sum'}, 'vpu_binary': {'mul','sub'}}
    insns = program['instructions']
    require(program.get('word_count') == len(insns), 'word count mismatch')
    require(0 <= program['entry_word_index'] < len(insns), 'invalid entry word')
    for pc, insn in enumerate(insns):
        require(insn.get('word_index') == pc, 'noncontiguous instruction words')
        name = insn['operation'].removeprefix('atlas.')
        require(insn['operation'] == 'atlas.'+name and name in fields, 'unsupported operation')
        f = insn['fields']
        require({k for k in f if not k.startswith('atlas.')} == set(fields[name].split()),
                f'unsupported fields for {name}')
        if name in kinds: require(f['kind'] in kinds[name], f'unsupported {name} kind')
        require(type(insn['word_u32']) is int and 0 <= insn['word_u32'] < 2**32,
                'invalid instruction word')
        require(int(insn['word_hex'],16) == insn['word_u32'], 'inconsistent instruction word')
        if 'unit' in f: require(type(f['unit']) is int and 0 <= f['unit'] < 2, 'unsupported MXU unit')
        if 'channel' in f: require(type(f['channel']) is int and 0 <= f['channel'] < 8, 'invalid DMA channel')
        if name == 'dma': require(f['direction'] in {'load','store'}, 'invalid DMA direction')
        if name == 'delay': require(type(f['cycles']) is int and 0 <= f['cycles'] <= 4095, 'invalid delay')
        if name in {'vload','vstore'}: require(f['format']=='raw', 'unsupported VLS format')
        if name == 'mxu_pop': require(f['format']=='bf16', 'unsupported MXU pop format')
        if name in {'mxu_push','mxu_pop'}: require(type(f['slot']) is int and 0 <= f['slot'] < 2, 'unsupported MXU slot')
        if name == 'mxu_matmul':
            require(f['acc_slot'] in (0,1) and f['weight_slot'] in (0,1) and type(f['accumulate']) is bool,
                    'unsupported matmul slots/accumulation field')
        if name == 'vpu_pack': require(f['direction']=='bf16_to_fp8', 'unsupported pack direction')
        if name == 'vli': require(f['mode']=='all', 'unsupported VLI mode')
        if name == 'branch':
            control = insn['control']
            require(control.get('kind')=='conditional_direct' and control.get('delay_slot_word_index')==pc+1,
                    'unsupported branch control/delay slot')
            require(0 <= control['target_word_index'] < len(insns), 'branch target out of range')
        for key in ('base_reg','base','dram','reg','size','scale_reg'):
            if key in f: require(type(f[key]) is int and 0 <= f[key] < 32, f'invalid {key} register')
        if name in {'alu_imm','upper','scalar_load','scalar_store','branch','csr'}:
            for key in ('dst','src','lhs','rhs','source'):
                if key in f: require(type(f[key]) is int and 0 <= f[key] < 32, f'invalid scalar {key}')


def predict(program, profile, use_banks=True, use_credits=True, max_cycles=1000000):
    """Return a prediction using no receipts, traces, or measured target cycles."""
    validate_program(program)
    insns = program['instructions']
    x = [None] * 32
    x[0] = 0
    def known(reg):
        require(x[reg] is not None, f'unknown scalar x{reg} used as address/control')
        return x[reg]

    def local_address(value, vector=False):
        require(type(value) is int and value >= 0, 'invalid VMEM operand')
        address = ((value >> 3) & ((1 << 16)-1))*32 if vector else value & ((1 << 21)-1)
        require(address < GEOMETRY['vmem_bytes'], 'VMEM address outside supported geometry')
        return address
    dma_base = 0
    claims = defaultdict(set)
    dma = Dma(profile, claims, use_banks, use_credits)
    pc = program['entry_word_index']
    redirect = None
    issue_at = 2  # one-cycle fetch, then S1; core.py pinned baseline
    issued = []
    cycle = 0
    while cycle < max_cycles:
        cycle += 1
        launch = None
        if cycle >= issue_at:
            insn = insns[pc]
            op, f = insn['operation'], insn['fields']
            stalled = op == 'atlas.dma_wait' and any(s.channel == f['channel'] for s in dma.slots)
            if not stalled:
                if op == 'atlas.trap':
                    if dma.slots:
                        raise ValueError('ECALL with DMA in flight; outside validated drain contract')
                    return {'cycles': cycle-1, 'drained_ecall_cycle': cycle, 'issued': issued,
                            'dma_beats': dma.events, 'flag_transitions': dma.flags,
                            'peak_outstanding_beats': dma.peak,
                            'bank_denials': dma.bank_denials}
                issued.append({'pc': pc, 'absolute_cycle': cycle,
                               'operation': op, 'word_u32': insn['word_u32']})
                next_pc = pc + 1 if redirect is None else redirect
                redirect = None
                if op == 'atlas.upper':
                    if f['kind'] != 'lui': raise ValueError(f)
                    x[f['dst']] = (f['immediate'] << 12) & 0xffffffff
                elif op == 'atlas.alu_imm':
                    if f['kind'] != 'addi': raise ValueError(f)
                    value = x[f['src']]
                    x[f['dst']] = None if value is None else (value + f['immediate']) & 0xffffffff
                elif op == 'atlas.scalar_load':
                    if f['kind'] != 'seli':
                        address = local_address(known(f['base']) + f['offset'])
                        claims[cycle+1].add((address & ((1 << 21)-1)) // 262144)
                        x[f['dst']] = None  # data never used as control/address in supported programs
                elif op == 'atlas.scalar_store':
                    address = local_address(known(f['base']) + f['offset'])
                    claims[cycle+1].add((address & ((1 << 21)-1)) // 262144)
                elif op in ('atlas.vload', 'atlas.vstore'):
                    operand = known(f['base']) + (f['offset'] << 5)
                    address = local_address(operand, vector=True)
                    require(address % 1024 == 0, 'VLS requires aligned 1KiB tile')
                    require(address+1024 <= GEOMETRY['vmem_bytes'], 'VLS range exceeds VMEM')
                    require(address//262144 == (address+1023)//262144, 'VLS crosses bank')
                    begin = 1 if op == 'atlas.vload' else 3
                    for age in range(begin, begin + 32):
                        claims[cycle+age].add(address // 262144)
                elif op == 'atlas.dma_config':
                    dma_base = known(f['base_reg'])
                elif op == 'atlas.dma':
                    count = known(f['size'])
                    address = (dma_base << 32) | known(f['dram'])
                    vmem = ((known(f['reg']) & ((1 << 19)-1)) >> 3) * 32
                    require(0 < count <= 4096 and count % 32 == 0, 'invalid DMA transfer size')
                    require(address % 32 == 0, 'unaligned DRAM DMA address')
                    require(2147483648 <= address < address+count <= 2147483648+68719476736,
                            'DMA DRAM range outside supported geometry')
                    require(vmem+count <= GEOMETRY['vmem_bytes'], 'DMA range exceeds VMEM')
                    launch = Slot(f['channel'], f['direction'] == 'store', address, vmem, count // 32, pc)
                elif op == 'atlas.branch':
                    lhs, rhs = known(f['lhs']), known(f['rhs'])
                    if f['kind'] != 'blt': raise ValueError(f)
                    if signed32(lhs) < signed32(rhs):
                        redirect = insn['control']['target_word_index']
                elif op == 'atlas.csr':
                    if f['dst']: x[f['dst']] = None
                elif op not in {'atlas.delay', 'atlas.dma_wait', 'atlas.mxu_push', 'atlas.mxu_matmul',
                                 'atlas.mxu_pop', 'atlas.vpu_unary', 'atlas.vpu_pack', 'atlas.vpu_binary',
                                 'atlas.vpu_reduce', 'atlas.vli', 'atlas.xlu_transpose'}:
                    raise ValueError(f'unsupported operation {op}')
                x[0] = 0
                issue_at = cycle + 1 + (f['cycles'] if op == 'atlas.delay' else 0)
                pc = next_pc
        dma.tick(cycle, launch)
    raise ValueError('prediction exceeded cycle cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('program', type=Path)
    parser.add_argument('--latency', type=int, default=43)
    parser.add_argument('--spacing', type=int, default=1)
    parser.add_argument('--no-banks', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = predict(json.loads(args.program.read_text()),
                     {'kind': 'fixed', 'params': {'latency': args.latency, 'cycles_per_beat': args.spacing}},
                     use_banks=not args.no_banks)
    result['prediction_seconds'] = time.perf_counter() - started
    result['inputs'] = {'program_sha256': sha(args.program), 'memory_backend_sha256': sha(BACKEND),
                        'predictor_sha256': sha(__file__)}
    payload = json.dumps(result, indent=2) + '\n'
    if args.output: args.output.write_text(payload)
    else: print(payload)

if __name__ == '__main__':
    main()
