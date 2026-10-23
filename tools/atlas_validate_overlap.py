#!/usr/bin/env python3
"""Gate source-derived overlap facts using bounded, fresh native-unit execution.

The candidate is never fitted to observations. Samples are descriptors, not
admission authority; the final artifact is published only after the gate passes.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import Mock

sys.dont_write_bytecode = True
POLICY = 'source-mechanism-boundaries-v1'
ENGINES = {'mxu0', 'mxu1', 'vpu', 'scalar_lsu'}


def safe_path(value):
    raw = Path(value)
    if any(word in part.lower() for part in raw.parts for word in ('hammer', 'vlsi')):
        raise ValueError('prohibited path')
    path = Path(os.path.abspath(raw))
    if any(p.is_symlink() for p in reversed((path, *path.parents))):
        raise ValueError('symlink paths are not supported')
    return path


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    return json.loads(safe_path(path).read_text(), object_pairs_hook=_object)


def write(path, value):
    path = safe_path(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return digest(path.read_bytes())


def issues(gaps):
    result = [0]
    for gap in gaps:
        if type(gap) is not int or gap < 1:
            raise ValueError('issue gaps must be positive integer cycles')
        result.append(result[-1] + gap)
    return result


def relation_class(relation):
    result = dict(relation)
    banks = result.get('mreg_banks', 'disjoint')
    if banks != 'disjoint':
        result['mreg_banks'] = ('mixed_alias' if '=' in banks and '~' in banks else
                                'register_alias' if '=' in banks else 'bank_alias')
    return result


def validate_candidate(candidate, per_op, per_op_bytes):
    if candidate.get('schema') != 'atlas.model_overlap_derived.v1':
        raise ValueError('unsupported candidate schema')
    if candidate.get('age_convention') != 'issue_edge_zero':
        raise ValueError('unsupported age convention')
    if candidate.get('per_op_sha256') != digest(per_op_bytes):
        raise ValueError('candidate does not bind selected operation facts')
    if candidate.get('model') != per_op.get('model'):
        raise ValueError('candidate model differs from selected operation facts')
    if candidate.get('derivation', {}).get('contract') != 'source-resource-occupancy-v1':
        raise ValueError('unsupported source contract')
    blocks = {b['name']: b for b in per_op['op_timing']}
    seen = set()
    for row in candidate.get('pairs', []):
        if set(row) != {'first', 'second', 'relation', 'minimum', 'high'}:
            raise ValueError('invalid pair fields')
        key = encoded([row['first'], row['second'], row['relation']])
        if key in seen:
            raise ValueError('duplicate pair domain')
        seen.add(key)
        if row['first'] not in blocks or row['second'] not in blocks:
            raise ValueError('pair references unknown operation')
        if row['first'].split('.')[0] != row['second'].split('.')[0]:
            raise ValueError('unsupported cross-engine pair')
        if (type(row['minimum']) is not int or type(row['high']) is not int or
                not 1 <= row['minimum'] <= row['high'] or
                row['high'] != blocks[row['first']]['first_free_age']):
            raise ValueError('invalid bounded pair gap interval')
    if not seen:
        raise ValueError('candidate has no pairs')


def sample_plan(document, candidate, max_samples=128):
    """Select bounded source mechanisms, never catalog rows or benchmark queries."""
    if document.get('schema') != 'atlas.overlap_samples.v1':
        raise ValueError('unsupported samples schema')
    samples = document.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('source mechanism samples are missing')
    normalized = []
    ids = set()
    required_operations = {name for row in candidate['pairs'] for name in (row['first'],row['second'])}
    candidate_keys = {encoded([r['first'], r['second'], r['relation']]): r for r in candidate['pairs']}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('invalid sample')
        required = {'id', 'mechanism', 'engine', 'operations', 'assembly', 'gaps', 'expectation', 'category', 'numerical_equivalent'}
        if not required <= sample.keys() or sample['id'] in ids:
            raise ValueError('missing sample fields or duplicate sample id')
        ids.add(sample['id'])
        if sample['engine'] not in ENGINES or not isinstance(sample['mechanism'], str) or not sample['mechanism']:
            raise ValueError('invalid source mechanism or engine')
        names = sample['operations']
        if (not isinstance(names, list) or len(names) < 2 or len(names) > 16 or
                len(sample['assembly']) != len(names) or len(sample['gaps']) != len(names)-1 or
                any(n.split('.')[0] != sample['engine'] for n in names)):
            raise ValueError('invalid sample operations/assembly/gaps')
        issues(sample['gaps'])
        if sample['expectation'] not in {'positive', 'reject'}:
            raise ValueError('invalid sample expectation')
        if type(sample['numerical_equivalent']) is not bool:
            raise ValueError('missing numerical equivalence assertion')
        if sample['expectation'] == 'positive' and not sample['numerical_equivalent']:
            raise ValueError('positive samples require serial semantic equivalence')
        if sample['category'] == 'pair':
            if len(names) != 2 or sample['expectation']!='positive':
                raise ValueError('pair sample needs exactly two positive instructions')
            row = candidate_keys.get(encoded([*names, sample.get('relation')]))
            if row is None or sample['gaps'] != [row['minimum']]:
                raise ValueError('sample pair does not bind candidate exact relation/minimum')
            sample = dict(sample, minimum=row['minimum'], high=row['high'])
        elif sample['category'] not in {'capacity', 'nonadjacent', 'port_history', 'chain'}:
            raise ValueError('unsupported sample category')
        normalized.append(sample)
    represented = {n for sample in normalized for n in sample['operations']}
    if not required_operations <= represented:
        raise ValueError('source samples omit candidate operations: '+','.join(sorted(required_operations-represented)))

    def labels(sample):
        unit=sample['engine']
        result={('operation',name) for name in sample['operations']}
        result.add(('category',unit,sample['category'],sample['expectation']))
        # Each chain mechanism needs both an accepted boundary and its own
        # rejecting control; another mechanism's positive cannot cover it.
        result.update(('mechanism',unit,name,sample['expectation'])
                      for name in sample['mechanism'].split('+'))
        result.update(('relation',unit,key,value) for key,value in relation_class(sample.get('relation',{})).items())
        if sample['category']=='pair':
            result.add(('threshold',unit,str(sample['minimum'])))
        return result

    # Cover independent source obligations, not their Cartesian product. The
    # selection is completed before any native observation or timing result.
    available=[(sample,labels(sample)) for sample in sorted(normalized,key=lambda r:r['id'])]
    uncovered=set().union(*(coverage for _,coverage in available))
    picked=[]
    while uncovered:
        sample,coverage=min(available,key=lambda entry:(-len(entry[1]&uncovered),entry[0]['id']))
        if not coverage&uncovered:
            raise ValueError('source mechanism coverage is incomplete')
        picked.append(sample)
        uncovered.difference_update(coverage)
        available=[entry for entry in available if entry[0]['id']!=sample['id']]
        if len(picked)>max_samples:
            raise ValueError(f'source mechanism coverage exceeds sample bound {max_samples}')
    picked=sorted(picked,key=lambda r:r['id'])
    represented_engines = {r['engine'] for r in picked if r['expectation'] == 'positive'}
    candidate_engines = {r['first'].split('.')[0] for r in candidate['pairs']}
    if not candidate_engines <= represented_engines:
        raise ValueError('source samples omit candidate engine coverage')
    return picked


@dataclass
class NativeFault(Exception):
    kind: str
    message: str
    age: int
    issued_ages: list

    def record(self):
        return dict(type=self.kind, message=self.message, age=self.age, issued_ages=self.issued_ages)


def resource_fault(fault):
    """Only reviewed native resource conflicts can satisfy a negative control."""
    if fault.kind == 'NativeCapacityError':
        return fault.message == 'VPU two issue slots occupied'
    if fault.kind == 'BankConflictError':
        return fault.message.startswith(('MRF bank conflict:', 'Undefined MRF same-row read/write collision', 'VMEM bank conflict:'))
    if fault.kind != 'RuntimeError':
        return False
    return bool(re.fullmatch(
        r'(?:VPU command .+ issued while its issue-busy bit is set|'
        r'scalar load issued while prior load response pending|'
        r'(?:mxu0|mxu1): (?:command issued before previous compute row-0 writeback|'
        r'compute issued while read port or in-flight tracker is busy|'
        r'BF16 push issued while read ports are busy|pop issued while write ports are busy|'
        r'push issued while read ports are busy|weight push issued before previous matmul on same slot drained|'
        r'weight slot busy|compute reads buffer with an active push|accumulation push target busy|'
        r'(?:weight|accumulator) push issued while another (?:weight|accumulator) push is still writing|'
        r'accumulator read issued the cycle after a push to it|weights changed while a matmul was using them|'
        r'compute and pop read the same accumulator|compute and push write the same accumulator))',
        fault.message))


def nonzero_mreg_bytes(bank, size=1024):
    """Distinct row-varying BF16 bits whose bytes are also small finite FP8.

    Each physical row holds sixteen BF16 elements. The prime-length mantissa
    pattern distinguishes all rows and columns; the high byte separates the
    two logical registers sharing each physical bank. Both bytes are <=0x3e,
    so their FP8 interpretations stay finite with magnitude at most 1.75.
    """
    if type(bank) is not int or not 0 <= bank < 64 or size < 1 or size % 32:
        raise ValueError('invalid nonzero MREG geometry')
    result=bytearray(size)
    for element in range(size//2):
        row,column=divmod(element,16)
        result[element*2]=1+(11*(bank%32)+13*row+7*column)%61
        result[element*2+1]=0x3d+bank//32
    return bytes(result)


def initialize_state(state, nonzero, initial_xrf=None):
    import torch
    if nonzero:
        for bank in range(64):
            data=nonzero_mreg_bytes(bank,state.mrf[bank].numel())
            state.mrf[bank].view(torch.uint8).copy_(torch.tensor(list(data),dtype=torch.uint8))
        coordinates=torch.arange(32,dtype=torch.int64)
        rows,columns=coordinates[:,None],coordinates[None,:]
        for engine_index,engine in enumerate(('mxu0', 'mxu1')):
            for slot in (0, 1):
                # Exact small representable values distinguish slots, engines
                # and row order without provoking numerical overflow.
                acc=(16+(engine_index*11+slot*7+rows*3+columns*5)%32).to(torch.float32)/128
                weight=(8+(engine_index*2+slot+rows*3+columns*5+(rows//8)*(columns//4))%8).to(torch.float32)/64
                state.acc[engine][slot].copy_(acc)
                state.write_wb_fp8(engine,slot,weight.to(torch.float8_e4m3fn))
        state.vmem[:] = torch.arange(len(state.vmem), dtype=torch.int64).remainder(251).to(torch.uint8)
        state.write_xrf(1, 0x12345678)
    state.write_erf(0, 127)
    for reg, value in (initial_xrf or {}).items():
        if not str(reg).isdigit() or not 1 <= int(reg) < 32 or type(value) is not int or not 0 <= value <= 0xffffffff:
            raise ValueError('invalid initial scalar register setting')
        state.write_xrf(int(reg), value)


def state_snapshot(state):
    import torch
    tensors = list(state.mrf) + [t for slots in state.acc.values() for t in slots]
    tensors += [t for slots in state.wb.values() for t in slots] + [state.vmem]
    files = [digest(t.contiguous().view(torch.uint8).numpy().tobytes()) for t in tensors]
    return dict(tensors=files, xrf=copy.deepcopy(state.xrf), erf=copy.deepcopy(state.erf))


def decoded_operand(name, insn):
    """Independently decode the assembler result through the compiler vocabulary."""
    from atlas_model_schema import VOCABULARY
    engine, kind = name.split('.', 1)
    result = {'registers': []}
    if engine == 'scalar_lsu':
        expected = {'load': 'lw', 'store': 'sw'}.get(kind)
        if expected is None or insn.mnemonic != expected:
            raise ValueError('unsupported/mismatched scalar assembly')
        return result
    family = 'mxu' if engine in {'mxu0', 'mxu1'} else 'vpu'
    mnemonic = insn.mnemonic.rsplit('.', 1)[0] if family == 'mxu' else insn.mnemonic
    if family == 'mxu' and not insn.mnemonic.endswith('.'+engine):
        raise ValueError('assembly MXU instance differs from operation')
    entries = [e for e in VOCABULARY[family+'_blocks'] if e['mnemonic'] == mnemonic]
    if len(entries) != 1 or entries[0]['block'] != kind:
        raise ValueError('assembled opcode differs from named operation')
    entry = entries[0]
    roles = VOCABULARY[family+'_operand_roles'][entry['op_class']]
    reads, writes = [], []
    native = {'rd': 'vd', 'rs1': 'vs1', 'rs2': 'vs2'}
    for role in roles:
        if role.get('resource','MReg') != 'MReg' or role['key'] is None:
            continue
        if role['two_input'] and not entry.get('two_input',False):
            continue
        first = int(getattr(insn,native[role['operand']])) + role['offset']
        values = list(range(first,first+role['registers']))
        if any(not 0 <= value < 64 for value in values):
            raise ValueError('sample MREG operand exceeds selected geometry')
        (writes if role['write'] else reads).extend(values)
    result['registers']=reads+writes
    if family == 'mxu':
        if kind == 'push_weight':
            result['weight_slot']=int(insn.vd)
        elif kind.startswith('pop_acc'):
            result['accumulator']=int(insn.vs2)
        elif kind.startswith('push_acc'):
            result['accumulator']=int(insn.vd)
        else:
            result.update(accumulator=int(insn.vd),weight_slot=int(insn.vs2))
    return result


def actual_relation(first,second):
    aliases=[]
    for i,a in enumerate(first['registers']):
        for j,b in enumerate(second['registers']):
            if a%32 == b%32:
                aliases.append(str(i)+('=' if a==b else '~')+str(j))
    result={'mreg_banks':','.join(aliases) if aliases else 'disjoint'}
    for key in ('accumulator','weight_slot'):
        if key in first and key in second:
            result[key]='same' if first[key]==second[key] else 'other'
    return result


def scalar_matches(relation,first,second,initial_xrf):
    required={'first_mnemonic','second_mnemonic','scalar_address_alignment','scalar_registers','vmem_bank'}
    optional={'scalar_destination_file','scalar_other_registers','vmem_window','vmem_word_window'}
    if not required <= relation.keys() or relation.keys()-required-optional-{'mreg_banks'}:
        return False
    if relation['first_mnemonic'] != first.mnemonic or relation['second_mnemonic'] != second.mnemonic:
        return False
    if relation['scalar_address_alignment'] != 'word':
        return False
    def roles(insn):
        base=('xrf',int(insn.rs1))
        destination=('xrf',int(insn.rd)) if insn.mnemonic=='lw' else None
        data=('xrf',int(insn.rs2)) if insn.mnemonic=='sw' else None
        resources={r for r in (base,destination,data) if r is not None}
        return base,destination,data,resources
    a,ad,ax,ar=roles(first);b,bd,bx,br=roles(second)
    common=ar&br
    mode=relation['scalar_registers']
    if mode=='independent':
        okay=not common and 'scalar_destination_file' not in relation
    elif mode=='shared_base_read_only':
        okay=a==b and common=={a} and ad!=a and bd!=a and ax!=a and bx!=a and 'scalar_destination_file' not in relation
    elif mode=='same_load_destination':
        okay=ad is not None and ad==bd and ad!=('xrf',0) and common=={ad} and a!=ad and b!=ad
    elif mode=='store_data_equals_load_destination':
        okay=bd is not None and bd!=('xrf',0) and ax==bd and common=={bd} and a!=bd and b!=bd
    elif mode=='first_address_base_equals_second_load_destination':
        okay=bd is not None and bd!=('xrf',0) and a==bd and common=={bd} and b!=bd and ax!=bd and ad!=bd
    else:
        return False
    if mode!='independent' and relation.get('scalar_other_registers')!='independent':
        return False
    if mode not in {'independent','shared_base_read_only'} and relation.get('scalar_destination_file')!='xrf':
        return False
    def address(insn):
        offset=int(insn.imm)&0xfff
        if offset&0x800:offset-=0x1000
        return initial_xrf.get(str(int(insn.rs1)),0)+offset
    x,y=address(first),address(second)
    if not 0<=x<1572864-3 or not 0<=y<1572864-3 or x%4 or y%4:
        return False
    bank='same' if x//262144==y//262144 else 'other'
    window='same' if x==y else 'overlap' if x<y+4 and y<x+4 else 'disjoint'
    return (okay and relation['vmem_bank']==bank and
            all(relation[k]==window for k in ('vmem_window','vmem_word_window') if k in relation) and
            (mode!='shared_base_read_only' or x==y))


def check_sample_assembly(sample):
    from npu_model.util.converter import stream_to_instrs
    instructions=list(stream_to_instrs(StringIO('\n'.join(sample['assembly']))))
    if len(instructions)!=len(sample['operations']):
        raise ValueError('sample assembly length mismatch')
    operands=[decoded_operand(n,i) for n,i in zip(sample['operations'],instructions)]
    if 'operands' in sample and sample['operands']!=operands:
        raise ValueError('sample concrete operands differ from assembled instructions')
    if sample['category']=='pair':
        relation=sample['relation']
        base={k:v for k,v in relation.items() if k in {'mreg_banks','accumulator','weight_slot'}}
        if actual_relation(*operands)!=base:
            raise ValueError('assembled operand relation differs from candidate relation')
        if sample['engine']=='scalar_lsu':
            if not scalar_matches(relation,*instructions,sample.get('initial_xrf',{})):
                raise ValueError('assembled scalar context does not satisfy candidate qualifiers')
        elif relation!=base:
            raise ValueError('unsupported non-scalar relation qualifier')
    if 'all_pair_relations' in sample:
        expected=[dict(first_index=i,second_index=j,relation=actual_relation(operands[i],operands[j]))
                  for j in range(len(operands)) for i in range(j)]
        if sorted(sample['all_pair_relations'],key=lambda r:(r['first_index'],r['second_index']))!=sorted(expected,key=lambda r:(r['first_index'],r['second_index'])):
            raise ValueError('assembled chain pair-relation matrix differs from descriptor')
    return operands


def native_run(engine, assembly, gaps, nonzero, serial=False, max_cycles=8192, initial_xrf=None):
    """Run unmodified numerical units at exact ages with a fresh state and unit."""
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.lsu import LoadStoreUnit
    from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs
    from atlas_model_timing import configuration
    cfg, _ = configuration()
    state = ArchState(cfg.arch_state_config)
    classes = {'vpu': VectorExecutionUnit, 'mxu0': MatrixExecutionUnitSystolic,
               'mxu1': MatrixExecutionUnitInner, 'scalar_lsu': LoadStoreUnit}
    try:
        initialize_state(state, nonzero, initial_xrf)
        initial = digest(encoded(state_snapshot(state)))
        unit = classes[engine](engine, Mock(spec=Logger), state, config=cfg)
        instructions = list(stream_to_instrs(StringIO('\n'.join(assembly))))
        if len(instructions) != len(assembly):
            raise ValueError('assembly instruction count differs')
        uops = [Uop(insn, pc=i*4) for i, insn in enumerate(instructions)]
        expected = issues(gaps)
        accepted, next_index, completed = [], 0, 0
        for age in range(max_cycles):
            issue = None
            if next_index < len(uops):
                ready = (age == expected[next_index]) if not serial else (next_index == 0 or not unit.has_in_flight)
                if ready:
                    issue = uops[next_index]
            try:
                before = unit.total_instructions
                unit.tick(issue)
                completed += unit.complete_count
                if issue is not None:
                    if unit.total_instructions != before + 1:
                        raise RuntimeError('native unit did not accept exact scheduled issue')
                    accepted.append(age)
                    next_index += 1
            except Exception as exc:
                # The reviewed native VPU uses next(free_slot) without a
                # default. Classify its exact exhausted-slot state; unrelated
                # StopIteration exceptions remain fatal implementation errors.
                if isinstance(exc,StopIteration) and engine=='vpu' and issue is not None:
                    live=[op for op in unit.operations if unit.cycle-op.issued < op.write_last]
                    if {op.slot for op in live}=={0,1}:
                        raise NativeFault('NativeCapacityError','VPU two issue slots occupied',age,accepted) from exc
                raise NativeFault(type(exc).__name__, str(exc), age, accepted) from exc
            if next_index == len(uops) and not unit.has_in_flight:
                # One begin-cycle applies deferred release requests after drain.
                unit.tick(None)
                completed += unit.complete_count
                checker = state.conflict_checker
                if checker._mreg_reservations or checker._mreg_releases:
                    raise NativeFault('RuntimeError', 'MREG reservations remain at drain', age+1, accepted)
                break
        else:
            raise NativeFault('RuntimeError', 'native sequence did not drain within cycle bound', max_cycles, accepted)
        if not serial and accepted != expected:
            raise RuntimeError('observed issue ages differ from exact requested ages')
        if unit.total_instructions != len(uops) or completed != len(uops):
            raise RuntimeError('accepted/completed counts do not equal sequence length')
        final = state_snapshot(state)
        return dict(initial_sha256=initial, final_sha256=digest(encoded(final)), final_state=final,
                    issued_ages=accepted, drained_age=age+1, accepted=unit.total_instructions,
                    completed=completed, instruction_words=[i.to_bytecode() for i in instructions])
    finally:
        state.close()


def run_sample(sample):
    started = time.monotonic()
    check_sample_assembly(sample)
    rows = []
    variants = [('chain', sample['gaps'])]
    if sample['category'] == 'pair':
        minimum, high = sample['minimum'], sample['high']
        variants = [('minimum', [minimum])]
        for tag, gap in [('later', min(minimum+1, high)), ('suffix', high)]:
            if gap != minimum and [gap] not in [g for _, g in variants]:
                variants.append((tag, [gap]))
        variants.append(('one_short', [minimum-1]))
    for nonzero in (False, True):
        serial = None
        if sample['numerical_equivalent']:
            # A failing serial execution is fatal, never a rejected negative.
            serial = native_run(sample['engine'], sample['assembly'], sample['gaps'], nonzero, serial=True, initial_xrf=sample.get('initial_xrf'))
        for variant, gaps in variants:
            row = dict(sample_id=sample['id'], engine=sample['engine'], mechanism=sample['mechanism'],
                       category=sample['category'], variant=variant, nonzero=nonzero, gaps=gaps,
                       expected='reject_or_conservative' if variant == 'one_short' else sample['expectation'])
            if any(gap == 0 for gap in gaps):
                row.update(passed=True, outcome='zero_unencodable', numerical_checked=False)
                rows.append(row)
                continue
            try:
                actual = native_run(sample['engine'], sample['assembly'], gaps, nonzero, initial_xrf=sample.get('initial_xrf'))
            except NativeFault as fault:
                if not resource_fault(fault):
                    raise RuntimeError('unexpected native failure: ' + str(fault.record())) from fault
                row.update(error=fault.record(), outcome='native_rejected', numerical_checked=False,
                           passed=row['expected'] != 'positive')
            else:
                same = serial is not None and actual['initial_sha256'] == serial['initial_sha256'] and actual['final_sha256'] == serial['final_sha256']
                row.update(observation=actual, serial_reference={k:v for k,v in serial.items() if k != 'final_state'} if serial else None,
                           numerical_checked=serial is not None, numerical_equal=same)
                if row['expected'] == 'positive':
                    row.update(passed=same, outcome='accepted_equal' if same else 'numerical_mismatch')
                elif row['expected'] == 'reject_or_conservative':
                    row.update(passed=True, outcome='conservative_one_short' if same else 'one_short_numerical_conflict')
                else:
                    row.update(passed=not same and serial is not None,
                               outcome='numerical_conflict' if not same and serial is not None else 'unexpected_acceptance')
            rows.append(row)
    return dict(id=sample['id'], seconds=time.monotonic()-started, passed=all(r['passed'] for r in rows), runs=rows)


def summarize_runs(flattened):
    engines={}
    for engine in sorted({r['engine'] for r in flattened}):
        runs=[r for r in flattened if r['engine']==engine]
        engines[engine]=dict(positive_runs=sum(r['expected']=='positive' and r['passed'] for r in runs),
                             negative_runs=sum(r['expected']!='positive' for r in runs),
                             negative_rejections=sum(r['expected']!='positive' and r['outcome']=='native_rejected' for r in runs),
                             numerical_checks=sum(r['numerical_checked'] and r.get('numerical_equal',False) for r in runs),
                             chain_positive_runs=sum(r['category']!='pair' and r['expected']=='positive' and r['passed'] for r in runs),
                             chain_negative_runs=sum(r['category']!='pair' and r['expected']!='positive' and r['outcome']=='native_rejected' for r in runs))
    return engines


def worker_init(model):
    sys.path.insert(0, str(safe_path(model)))
    import torch
    import npu_model
    torch.set_num_threads(1)
    if safe_path(Path(npu_model.__file__).parent.parent) != safe_path(model):
        raise ValueError('imported native model differs from --model')


def validate(model, per_op_path, candidate_path, samples_path, output_dir, output_path, workers=4, max_samples=128):
    started = time.monotonic()
    model, output_dir, output_path = map(safe_path, (model, output_dir, output_path))
    if output_dir.exists() or output_path.exists():
        raise ValueError('validation output directory and final artifact must be new')
    if not 1 <= workers <= 4:
        raise ValueError('workers must be in [1,4]')
    per_op_path, candidate_path, samples_path = map(safe_path, (per_op_path, candidate_path, samples_path))
    original = {str(p):digest(p.read_bytes()) for p in (per_op_path, candidate_path, samples_path, Path(__file__))}
    tool_names=('atlas_validate_overlap.py','atlas_model_facts.py','atlas_model_timing.py',
                'atlas_model_schema.py','atlas_model_observer.py','model-timing.lock.json','model-timing-vocabulary.json')
    tool_paths=[safe_path(Path(__file__).with_name(name)) for name in tool_names]
    tool_sources={str(path):digest(path.read_bytes()) for path in tool_paths}
    per_op_bytes=per_op_path.read_bytes()
    per_op, candidate, descriptor = load(per_op_path), load(candidate_path), load(samples_path)
    validate_candidate(candidate, per_op, per_op_bytes)
    planned = sample_plan(descriptor, candidate, max_samples)
    from atlas_model_facts import identity
    before = identity(model)
    for key in ('source_sha256','numerical_data_sha256'):
        if before[0][key] != per_op['model'][key]:
            raise ValueError('native model identity differs from selected facts: ' + key)
    worker_init(str(model))
    from atlas_model_timing import configuration
    _, config = configuration()
    if config != per_op['model']['configuration']:
        raise ValueError('native default configuration differs from selected facts')
    from npu_model.hardware.core import Core
    from npu_model.logging.logger import Logger
    cfg, _ = configuration()
    core = Core(config=cfg, logger=Mock(spec=Logger))
    try:
        if core.ignore_runtime_errors is not False:
            raise ValueError('native model suppresses runtime errors')
    finally:
        core.arch_state.close()
    output_dir.mkdir(parents=True)
    write(output_dir/'sample-plan.json', dict(schema=descriptor['schema'], policy=POLICY, samples=planned, exclusions=descriptor.get('exclusions', [])))
    results=[]
    try:
        if workers == 1:
            for sample in planned:
                result=run_sample(sample);results.append(result)
                print(json.dumps(dict(sample=result['id'],passed=result['passed'],seconds=result['seconds'])),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(str(model),)) as pool:
                for result in pool.map(run_sample, planned):
                    results.append(result)
                    print(json.dumps(dict(sample=result['id'],passed=result['passed'],seconds=result['seconds'])),flush=True)
    except Exception as exc:
        write(output_dir/'fatal-error.json', dict(type=type(exc).__name__,message=str(exc)))
        raise
    flattened=[r for result in results for r in result['runs']]
    counts=Counter(r['outcome'] for r in flattened)
    engines=summarize_runs(flattened)
    unchanged=(before==identity(model) and all(digest(safe_path(p).read_bytes())==h for p,h in original.items())
               and all(digest(safe_path(p).read_bytes())==h for p,h in tool_sources.items()))
    required_engines={r['first'].split('.')[0] for r in candidate['pairs']}
    coverage=all(engines.get(e,{}).get('positive_runs',0)>0 and engines.get(e,{}).get('negative_rejections',0)>0 and engines.get(e,{}).get('numerical_checks',0)>0 for e in required_engines)
    composed_engines=candidate.get('composition',{}).get('engines',{})
    coverage &= all(engines.get(e,{}).get('chain_positive_runs',0)>0 and engines.get(e,{}).get('chain_negative_runs',0)>0 for e in composed_engines)
    passed=unchanged and coverage and all(r['passed'] for r in results)
    report=dict(schema='atlas.model_overlap_validation.v1',passed=passed,sample_policy=POLICY,
                candidate_sha256=original[str(candidate_path)],per_op_sha256=digest(per_op_bytes),
                sample_descriptors_sha256=original[str(samples_path)],tool_sha256=original[str(Path(__file__))],
                tool_sources=tool_sources,serial_reference_runs=sum(2 for s in planned if s['numerical_equivalent']),
                model=before[0],inputs_unchanged=unchanged,engine_coverage=coverage,engines=engines,
                positive_runs=sum(v['positive_runs'] for v in engines.values()),negative_runs=sum(v['negative_runs'] for v in engines.values()),
                negative_rejections=sum(v['negative_rejections'] for v in engines.values()),
                numerical_checks=sum(v['numerical_checks'] for v in engines.values()),outcomes=dict(counts),
                sample_count=len(planned),workers=workers,seconds=time.monotonic()-started,
                exclusions=descriptor.get('exclusions',[]),limitations=['Bounded mechanism samples are not exhaustive pair/operand enumeration or RTL qualification.',
                'One-short native acceptance records conservative bounds; no timing values are fitted.',
                'Zero one-short gaps are ISA-unencodable, not observed model conflicts.'])
    sample_seconds = {result['id']: result.pop('seconds') for result in results}
    write(output_dir/'observations.json',results)
    elapsed = report.pop('seconds')
    report_hash=write(output_dir/'validation-report.json',report)
    write(output_dir/'timing-receipt.json',dict(validation_seconds=elapsed, sample_seconds=sample_seconds, report_sha256=report_hash))
    report['seconds'] = elapsed
    if not passed:
        raise RuntimeError('native overlap validation failed; final artifact was not emitted')
    final=dict(candidate)
    final['validation']=dict(passed=True,report_sha256=report_hash,sample_policy=POLICY,
                             positive_runs=report['positive_runs'],negative_runs=report['negative_runs'],
                             negative_rejections=report['negative_rejections'],
                             numerical_checks=report['numerical_checks'],engines=engines)
    write(output_path,final)
    print(json.dumps(dict(passed=True,report_sha256=report_hash,output=str(output_path),seconds=report['seconds'])),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','per-op','candidate','samples','output-dir','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-samples',type=int,default=128)
    args=parser.parse_args()
    try:
        validate(args.model,args.per_op,args.candidate,args.samples,args.output_dir,args.output,args.workers,args.max_samples)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(1,f'native overlap validation failed: {exc}\n')


if __name__=='__main__':
    main()
