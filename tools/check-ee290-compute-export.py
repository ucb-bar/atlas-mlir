#!/usr/bin/env python3
"""Bind `atlas-emit --rtl-timing-json` footprints to decoded Verilator observations.

No timing rules live here. Static streams come from the compiler's resolver;
DMA completion is anchored to the actual matching WAIT acceptance.
"""

RESOLVER = 'atlas.op_timing.serialized.v1'
SCHEMA = 'atlas.resolved_rtl_timing.v1'


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
    elif opcode == 0x07:
        mnemonic = 'vstore' if word & (1 << 13) else 'vload'
        result.update(rd=(word >> 7) & 63, rs1=rs1, immediate=signed)
    elif opcode == 0x6b:
        mnemonic = 'vtrpose.xlu'; result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63)
    elif opcode == 0x57 and word >> 25 == 3:
        mnemonic = 'vmul.bf16'; result.update(rd=(word >> 7) & 63, rs1=(word >> 13) & 63, rs2=(word >> 19) & 63)
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


def bind_export(export, observed, facts_sha256, words):
    """Check export and observed events agree; returns a small summary."""
    require(export.get('schema') == SCHEMA and export.get('resolver', {}).get('id') == RESOLVER, 'export schema/resolver mismatch')
    require(export['resolver'].get('dma_policy') == 'wait', 'DMA wait policy required')
    require(export.get('evidence', {}).get('op_timing_sha256') == facts_sha256, 'export was not resolved from the selected facts')
    require(export.get('program', {}).get('words') == words and export['program'].get('word_count') == len(words), 'export differs from emitted words')
    instructions = export['instructions']; require(len(instructions) == len(words), 'incomplete resolved instruction export')
    retired = {item['word_index']: item for item in observed['instructions']}
    require(len(retired) == len(observed['instructions']), 'duplicate observed instruction index')
    halts = [item for item in observed['endpoints'] if item['kind'] == 'halt']
    require(len(halts) == 1, 'one terminal acceptance required')
    retired[halts[0]['word_index']] = halts[0]
    require(set(retired) == set(range(len(words))), 'incomplete actual retirement/terminal words')
    waits = {item['word_index']: item for item in observed['endpoints'] if item['kind'] == 'wait'}
    origins = {}; epoch = 0; first_edge = retired[0]['edge']
    for index, item in enumerate(instructions):
        require(item['word_index'] == index and item['word_u32'] == words[index] == retired[index]['word_u32'], 'resolved/observed instruction word mismatch')
        mnemonic, operands = decoded_operands(words[index])
        require(item['mnemonic'] == mnemonic and item['operands'] == operands, 'exported instruction metadata differs from selected word')
        edge = integer(retired[index]['edge'], 'observed issue edge')
        if index: require(edge > retired[index - 1]['edge'], 'actual instruction edges do not advance')
        current = integer(item['issue_epoch'], 'issue epoch'); offset = integer(item['epoch_offset'], 'epoch offset')
        require(edge - first_edge >= integer(item['minimum_issue_cycle'], 'minimum issue cycle'), 'observed issue precedes exported lower bound')
        if index in waits:
            require(current == epoch + 1 and offset == 0 and item.get('event_kind') == 'matching_wait_acceptance', 'matching WAIT does not start the next epoch')
            epoch = current; origins[epoch] = edge
        else: require(current == epoch, 'epoch changes without matching WAIT acceptance')
        if current not in origins:
            require(current == 0 and index == 0 and offset == 0, 'unsupported epoch origin')
            origins[current] = edge
        require(edge == origins[current] + offset, 'observed intra-epoch issue differs from exported offset')
    require(instructions[-1].get('event_kind') == 'terminal_acceptance' and instructions[-1]['mnemonic'] == 'ecall', 'terminal acceptance convention mismatch')
    commands = {command['word_index']: command for command in observed['commands']}
    require(len(commands) == len(observed['commands']), 'duplicate observed command index')
    resolved = []; checked = 0; intervals = []
    for index, item in enumerate(instructions):
        mnemonic = item['mnemonic']; operands = item['operands']
        if not (mnemonic in ('vload', 'vstore', 'vtrpose.xlu', 'vmul.bf16') or mnemonic.startswith(('dma.load.ch', 'dma.store.ch'))): continue
        require(index in commands, 'exported compute command absent from observations')
        command = commands[index]; footprint = item['footprint']; issue = command['edge']
        require(command['word_u32'] == words[index] and issue == retired[index]['edge'], 'command differs from actual selected instruction')
        kind = command['engine']
        if kind == 'vls':
            require(mnemonic == ('vload' if command['op'] == 'load' else 'vstore') and operands['rd'] == command['mreg'], 'resolved VLS operation/register mismatch')
            points = []
            for field, write in [('reads', False), ('writes', True)]:
                resource = 'vmem' if (command['op'] == 'load') != write else 'mreg'
                points += [(resource, write, e['id'] if resource == 'vmem' else 32 * e['id'] + e['row'], e['edge']) for e in command[field]]
        elif kind == 'xlu':
            require(mnemonic == 'vtrpose.xlu' and operands['rd'] == command['dst'] and operands['rs1'] == command['src'], 'resolved XLU operands mismatch')
            points = [('mreg', w, 32 * e['id'] + e['row'], e['edge']) for f, w in [('reads', False), ('writes', True)] for e in command[f]]
        elif kind == 'vmul':
            require(mnemonic == 'vmul.bf16' and (operands['rd'], operands['rs1'], operands['rs2']) == (command['dst'], command['lhs'], command['rhs']), 'resolved VMUL operands mismatch')
            for field in ('reads', 'responses', 'writes'):
                bases = (command['dst'],) if field == 'writes' else (command['lhs'], command['rhs'])
                for e in command[field]:
                    require(type(e.get('port')) is int and 0 <= e['port'] < len(bases) and e['id'] - bases[e['port']] in (0, 1), 'VMUL event lacks its explicit pair port')
            points = [('mreg', w, 32 * e['id'] + e['row'], e['edge']) for f, w in [('reads', False), ('writes', True)] for e in command[f]]
        elif kind == 'dma':
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
            match = [e for e in waits.values() if e['channel'] == channel and e['edge'] == command['release_edge']]
            require(len(match) == 1 and match[0]['word_index'] > index, 'no matching WAIT after selected launch')
            require(all(issue <= e['edge'] < match[0]['edge'] for e in memory + command['requests'] + command['responses']), 'DMA effects do not drain before actual matching WAIT')
            intervals.append({'word_index': index, 'channel': channel, 'launch_edge': issue, 'matching_wait_edge': match[0]['edge'], 'observed_elapsed_edges': match[0]['edge'] - issue})
            resolved.append(index); checked += len(memory); continue
        else: raise BindingError('unsupported observed command engine')
        exported = []
        for stream in footprint['accesses']:
            if stream['resource'] in ('mreg', 'vmem'): exported += finite_stream(stream, issue)
        require(sorted(exported) == sorted(points), 'observed access resource/address/age differs from resolver footprint')
        reads = {(e.get('port', 0), e['id'], e['row']): e for e in command['reads']}
        require(len(reads) == len(command['reads']) == len(command['responses']), 'incomplete or duplicate source stream')
        seen = set()
        for response in command['responses']:
            key = (response.get('port', 0), response['id'], response['row'])
            require(key not in seen and key in reads and response['edge'] == reads[key]['edge'] + 1 and response['data_hex'] == reads[key]['data_hex'], 'read response differs from observed one-cycle source stream')
            seen.add(key)
        inclusive = [integer(footprint[k], k) for k in ('done_age', 'read_release', 'write_release')] + [integer(h['to'], 'hold end') for h in footprint['holds']]
        require(command['release_edge'] == issue + max(inclusive) + 1, 'observed engine release differs from inclusive resolver holds')
        age = lambda event: event['edge'] - issue
        sources = command['reads'] + command['responses'] if kind != 'vls' or command['op'] == 'store' else []
        dests = command['writes'] if kind != 'vls' or command['op'] == 'load' else []
        require(sorted(footprint['mreg_reads']) == sorted({e['id'] for e in sources}) and sorted(footprint['mreg_writes']) == sorted({e['id'] for e in dests}), 'exported MREG lifetime registers differ from observed streams')
        require(footprint['read_release'] >= max(map(age, sources), default=0) and footprint['write_release'] >= max(map(age, dests), default=0) and footprint['done_age'] >= max(map(age, command['reads'] + command['responses'] + command['writes'])), 'exported lifetime ends before an observed access')
        for unit in {'vls': (), 'xlu': ('XLU',), 'vmul': ('VPU',)}[kind] + ('VLOAD path', 'VSTORE path'):
            require(sum(h['unit'] == unit and h['index'] == 0 and h['from'] == 0 and h['to'] == footprint['done_age'] for h in footprint['holds']) == 1, 'serialized ' + unit + ' hold absent or shortened')
        resolved.append(index); checked += len(points)
    require(set(resolved) == set(commands), 'observed command lacks selected exported footprint')
    return {'instruction_words_bound': len(words), 'memory_access_elements_bound': checked,
            'epoch_origins': [{'epoch': k, 'actual_edge': v} for k, v in sorted(origins.items())], 'dynamic_dma_intervals': intervals}
