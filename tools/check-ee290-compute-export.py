#!/usr/bin/env python3
"""Bind `atlas-emit --rtl-timing-json` footprints to decoded Verilator observations.

No timing rules live here. Static streams come from the compiler's resolver; every dynamic block
instance binds to its block's offsets, and DMA completion is anchored to the actual matching WAIT.
"""

RESOLVER = 'atlas.op_timing.serialized.v1'
SCHEMA = 'atlas.resolved_rtl_timing.v1'
BRANCHES = {0: 'beq', 1: 'bne', 4: 'blt', 5: 'bge', 6: 'bltu', 7: 'bgeu'}
# Observed engine -> (selected mnemonic, the engine's own hold unit).
ENGINES = {'vload': ('vload', 'VLOAD path'), 'vstore': ('vstore', 'VSTORE path'), 'xlu': ('vtrpose.xlu', 'XLU'), 'vpu': ('vmul.bf16', 'VPU'),
           'mxu0': (None, 'MXU in-flight matmuls'), 'mxu1': (None, 'MXU in-flight matmuls')}  # an MXU owns both MXU units
ENGINE_UNITS = {unit for name, unit in ENGINES.values() if name}
MXU_STEMS = ('vmatpush.weight', 'vmatpush.acc.fp8', 'vmatpush.acc.bf16', 'vmatpop.fp8.acc', 'vmatpop.bf16.acc', 'vmatmul', 'vmatmul.acc')


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


def is_command(mnemonic):
    return mnemonic in {m for m, _ in ENGINES.values()} or mnemonic.startswith(('dma.load.ch', 'dma.store.ch')) or '.mxu' in mnemonic


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
    elif opcode == 0x57 and word >> 25 == 3:
        mnemonic = 'vmul.bf16'; result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63, rs2=(word >> 19) & 63)
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
        mnemonic = 'csrrw'; result.update(rd=rd, rs1=rs1, immediate=immediate)
    elif word == 0x73:
        mnemonic = 'ecall'
    else: raise BindingError('unsupported word in finite export binding')
    return mnemonic, result


def check_header(export, facts_sha256):
    require(export.get('schema') == SCHEMA and export.get('resolver', {}).get('id') == RESOLVER, 'export schema/resolver mismatch')
    require(export['resolver'].get('dma_policy') == 'wait', 'DMA wait policy required')
    require(export.get('evidence', {}).get('op_timing_sha256') == facts_sha256, 'export was not resolved from the selected facts')
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
        require(dram[0].get('at_completion') is True and dram[0].get('anywhere') is True and dram[0].get('age') is None and dram[0].get('step') is None, 'DMA DRAM effect must remain dynamically conservative')
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
        expected = {0: (command.get('slot'), command.get('mreg'), 0), 4: (command.get('mreg'), 0, command.get('acc')),
                    5: (command.get('acc'), command.get('mreg'), command.get('slot'))}.get(command.get('op'))
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
        require((operands['rd'], operands['rs1'], operands['rs2']) == (command['dst'], command['lhs'], command['rhs']), 'resolved VMUL operands mismatch')
        for field in ('reads', 'responses', 'writes'):
            bases = (command['dst'],) if field == 'writes' else (command['lhs'], command['rhs'])
            for e in command[field]:
                require(type(e.get('port')) is int and 0 <= e['port'] < len(bases) and e['id'] - bases[e['port']] in (0, 1), 'VMUL event lacks its explicit pair port')
        points = mreg('reads', False) + mreg('writes', True)
    exported = []
    for stream in footprint['accesses']:
        if stream['resource'] in ('mreg', 'vmem') or stream['resource'] in ('acc', 'weight') and stream['step']: exported += finite_stream(stream, issue)
        else: require(stream['resource'] in ('xreg', 'ereg') or mxu and stream['resource'] == 'weight', 'unbound access resource ' + str(stream['resource']))
    require(sorted(exported) == sorted(points), 'observed access resource/address/age differs from resolver footprint')
    reads = {(e.get('port', 0), e['id'], e['row']): e for e in command['reads'] if e.get('resource', 'mreg') in ('mreg', 'vmem')}
    require(len(reads) == len(command['responses']) == len({(e.get('port', 0), e.get('resource'), e['id'], e['row']) for e in command['reads'] if e.get('resource', 'mreg') in ('mreg', 'vmem')}), 'incomplete or duplicate source stream')
    seen = set()
    for response in command['responses']:
        key = (response.get('port', 0), response['id'], response['row'])
        require(key not in seen and key in reads and response['edge'] == reads[key]['edge'] + 1 and response['data_hex'] == reads[key]['data_hex'], 'read response differs from observed one-cycle source stream')
        seen.add(key)
    age = lambda event: event['edge'] - issue
    release = integer(command['release_edge'] - issue, 'observed release age')
    activity = max(map(age, command['reads'] + command['responses'] + command['writes'] + command.get('compute', [])))
    own = [h for h in footprint['holds'] if h['unit'] == ENGINES[kind][1] and (mxu or h['index'] == 0)]
    require(len(own) == (2 if mxu else 1) and sorted(h['index'] for h in own) == list(range(len(own))) and
            all(h['from'] == 0 and release == integer(h['to'], 'hold end') + 1 for h in own), 'observed engine release differs from its own-unit hold end + 1')
    vmem_ages = [age(e) for e in command[vmem_field]] if vmem_field else []
    for hold in footprint['holds']:
        if any(hold is h for h in own): continue
        if hold['unit'] in ENGINE_UNITS: require(hold['from'] == 0 and integer(hold['to'], 'hold end') >= release - 1, 'policy ' + hold['unit'] + ' hold ends before the engine activity')
        elif hold['unit'] == 'VMEM bank': require(vmem_ages and hold['from'] <= min(vmem_ages) and hold['to'] >= max(vmem_ages), 'VMEM bank hold does not cover the observed bank accesses')
        else: raise BindingError('unbound hold unit ' + str(hold['unit']))
    sources = [e for e in command['reads'] + command['responses'] if e.get('resource', 'mreg') == 'mreg'] if kind != 'vload' else []
    dests = [e for e in command['writes'] if e.get('resource', 'mreg') == 'mreg'] if kind != 'vstore' else []
    require(sorted(footprint['mreg_reads']) == sorted({e['id'] for e in sources}) and sorted(footprint['mreg_writes']) == sorted({e['id'] for e in dests}), 'exported MREG lifetime registers differ from observed streams')
    require(footprint['read_release'] >= max(map(age, sources), default=0) and footprint['write_release'] >= max(map(age, dests), default=0) and integer(footprint['done_age'], 'done age') >= max(activity, release - 1), 'exported lifetime ends before an observed access')
    return len(points), None


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


def bind_export(export, observed, facts_sha256, words):
    """Check export and observed events agree; returns a small summary."""
    check_header(export, facts_sha256)
    require(export.get('program', {}).get('words') == words and export['program'].get('word_count') == len(words), 'export differs from emitted words')
    instructions = export['instructions']
    require(len(instructions) == len(words) and all(item['word_index'] == i and item['word_u32'] == words[i] for i, item in enumerate(instructions)), 'incomplete resolved instruction export')
    blocks, starts = bind_blocks(export, words)
    halts = [item for item in observed['endpoints'] if item['kind'] == 'halt']
    require(len(halts) == 1 and instructions[halts[0]['word_index']].get('event_kind') == 'terminal_acceptance', 'one terminal acceptance required')
    waits = {(e['word_index'], e['edge']): e for e in observed['endpoints'] if e['kind'] == 'wait'}
    require(len(waits) == sum(e['kind'] == 'wait' for e in observed['endpoints']), 'duplicate observed WAIT')
    instances, fired, previous, consumed = [], [], None, 0
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
        fired.append((word, edge)); previous = (word, edge)
    require(consumed == len(waits), 'observed WAIT outside an exported matching wait')
    require(item['mnemonic'] == 'ecall' and block['exit'] == 'halt', 'terminal acceptance convention mismatch')
    entries = observed['block_entries']
    require([(e['word_index'], e['edge']) for e in entries] == [(blocks[i['block']]['first_word'], i['entry_edge']) for i in instances], 'observed block entries differ from exported block instances')
    require(not any(any(e['engine_state'].values()) for e in entries), 'engines not idle at a block entry')
    keys = [(c['word_index'], c['edge']) for c in observed['commands']]
    require(len(set(keys)) == len(keys) and set(keys) == {f for f in fired if is_command(instructions[f[0]]['mnemonic'])}, 'observed commands differ from executed compute instructions')
    checked, intervals = 0, []
    for command in observed['commands']:
        elements, interval = bind_command(instructions[command['word_index']], command, list(waits.values()))
        checked += elements; intervals += [interval] if interval else []
    return {'instruction_words_bound': len(fired), 'block_instances': instances, 'memory_access_elements_bound': checked, 'dynamic_dma_intervals': intervals}


def bind_footprints(export, observed, facts_sha256):
    """Bind each observed command to the same command of a straight-line reference export, without issue offsets."""
    check_header(export, facts_sha256)
    items = [item for item in export['instructions'] if is_command(item['mnemonic'])]
    require(len(export['blocks']) == 1 and len(items) == len(observed['commands']), 'reference export differs from observed commands')
    waits = [e for e in observed['endpoints'] if e['kind'] == 'wait']
    checked, intervals = 0, []
    for item, command in zip(items, observed['commands']):
        elements, interval = bind_command(item, command, waits)
        checked += elements; intervals += [interval] if interval else []
    return {'instruction_words_bound': 0, 'block_instances': [], 'memory_access_elements_bound': checked, 'dynamic_dma_intervals': intervals}
