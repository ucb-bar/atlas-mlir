#!/usr/bin/env python3
"""Index byte-bound retained HW structure and candidate event cones; no timing rules.

Supported input is firtool custom assembly with one-line module headers,
instances, outputs and operations, two-space top-level module indentation and
four-space module-body indentation. Named ports, private modules and optional
instance inner symbols are supported. Nested-region results are opaque cone
boundaries. This is a structural index, not an MLIR verifier or an interpreter.
Unsupported required syntax/references fail; unrelated top-level operations,
including OM metadata, are outside this index. Opaque memory ports are retained
without assigning depth, implementation behavior or instruction latency.

The small declaration/instance reader follows the emitted grammar also used by
Merlin's historical extract_module.py; it does not import Merlin or ModelIR.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


class IndexError(ValueError):
    """Unsupported or inconsistent selected evidence."""


SSA = r'%[A-Za-z0-9_$.-]+(?:#[0-9]+)?'
SYMBOL = r'@(?:"(?:\\.|[^"\\])*"|[A-Za-z0-9_$.-]+)'
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
REQUIRED_HINTS = ('s1_fire', 'is_lsu_launch', 'is_dma_launch',
                  'is_mxu0_launch', 'is_mxu1_launch', 'is_vpu_launch', 'is_xlu_launch')
COMBINATIONAL = {'hw.constant', 'comb.and', 'comb.or', 'comb.xor', 'comb.mux',
                 'comb.icmp', 'comb.extract', 'comb.concat', 'comb.replicate',
                 'comb.add', 'comb.sub', 'comb.mul', 'comb.shl', 'comb.shr_u',
                 'comb.shr_s', 'comb.div_u', 'comb.div_s', 'comb.mod_u', 'comb.mod_s',
                 'comb.parity', 'hw.bitcast', 'seq.from_clock', 'seq.to_clock'}


def allowed(path):
    path = Path(path).absolute()
    for candidate in (path,):
        if any('hammer' in p.lower() or 'vlsi' in p.lower() for p in candidate.parts):
            raise IndexError('Prohibited path component')
    resolved = path.resolve()
    if any('hammer' in p.lower() or 'vlsi' in p.lower() for p in resolved.parts):
        raise IndexError('Prohibited symlink target component')
    return resolved


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def checked_member(member, directory):
    if not isinstance(member, dict) or not isinstance(member.get('path'), str):
        raise IndexError('Missing artifact identity')
    path = Path(member['path'])
    path = allowed(path if path.is_absolute() else directory / path)
    observed = identity(path)
    if any(observed[k] != member.get(k) for k in ('sha256', 'bytes')):
        raise IndexError('Retained hardware artifact hash/size mismatch')
    return path, observed


def unquote(token):
    try:
        return json.loads(token) if token.startswith('"') else token
    except ValueError as exc:
        raise IndexError('Unsupported quoted identifier escape') from exc


def split_top(text, separator=','):
    """Split outside balanced custom-assembly brackets and quoted strings."""
    parts, stack, start, quoted, escaped = [], [], 0, False, False
    pairs = {')': '(', ']': '[', '}': '{', '>': '<'}
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '([{<':
            stack.append(char)
        elif char in pairs:
            if char == '>' and i and text[i - 1] == '-':
                continue
            if not stack or stack.pop() != pairs[char]:
                raise IndexError('Unbalanced custom assembly')
        elif char == separator and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    if quoted or stack:
        raise IndexError('Multiline/unbalanced custom assembly is unsupported')
    parts.append(text[start:].strip())
    return parts


def balanced_group(text, begin, opening='(', closing=')'):
    if text[begin:begin + 1] != opening:
        raise IndexError('Expected parenthesized port list')
    depth, quoted, escaped = 0, False, False
    for i in range(begin, len(text)):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[begin + 1:i], i + 1
    raise IndexError('Multiline port list is unsupported')


def strip_location(text):
    return re.sub(r'\s+loc\(.*\)$', '', text).strip()


def anchor(line, text, op):
    return {'line': line, 'op': op, 'text': text.strip()}


def parse_ports(text):
    ports = []
    for item in split_top(text) if text.strip() else []:
        m = re.fullmatch(r'(in|out)\s+(%?[A-Za-z0-9_$.-]+|"(?:\\.|[^"\\])*")\s*:\s*(.+)', item)
        if not m or (m[1] == 'in') != m[2].startswith('%'):
            raise IndexError('Unsupported module port declaration: ' + item)
        ports.append({'direction': m[1], 'name': unquote(m[2].lstrip('%')),
                      'ssa': m[2] if m[1] == 'in' else None,
                      'type': strip_location(m[3])})
    if len({p['name'] for p in ports}) != len(ports):
        raise IndexError('Duplicate module port')
    return ports


def module_table(lines):
    modules = {}
    for i, line in enumerate(lines):
        if not line.startswith(('  hw.module', '  sv.verbatim.module', '  sv.verbatim.source')):
            continue
        m = re.match(r'  (hw\.module(?:\.extern|\.generated)?|sv\.verbatim\.(?:module|source))\s+(?:(?:private|public)\s+)?(' + SYMBOL + r')', line)
        if not m:
            raise IndexError(f'Unsupported module declaration at line {i + 1}')
        kind, name = m[1], unquote(m[2][1:])
        if name in modules:
            raise IndexError('Duplicate module symbol: ' + name)
        ports, end, parameters = [], i, None
        if kind.startswith('hw.module'):
            begin = m.end()
            if line[begin:begin + 1] == '<':
                parameters, begin = balanced_group(line, begin, '<', '>')
            content, tail = balanced_group(line, begin)
            ports = parse_ports(content)
            if kind == 'hw.module':
                if not line.rstrip().endswith('{'):
                    raise IndexError('Unsupported internal module header')
                end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith('  }')), None)
                if end is None:
                    raise IndexError('Unterminated module: ' + name)
        modules[name] = {'name': name, 'kind': kind, 'ports': ports,
                         'parameters_text': parameters,
                         'declaration': anchor(i + 1, line, kind), '_start': i, '_end': end}
    if not modules:
        raise IndexError('No supported HW module declarations')
    return modules


def parse_operation(line, number):
    """Read a flat SSA operation; unknown ops remain explicit cone boundaries."""
    text = line.strip()
    m = re.match(r'((?:' + SSA + r')(?:\s*,\s*' + SSA + r')*)\s*=\s*([\w.]+)\b(.*)', text)
    if not m:
        if text.startswith('hw.instance '):
            return {'results': [], 'operands': [], 'namehint': None,
                    **anchor(number, text, 'hw.instance')}
        return None
    results = [x.strip() for x in m[1].split(',')]
    raw = m[3]
    if not (raw.rstrip().endswith('{') and m[2] not in COMBINATIONAL):
        split_top(raw)  # Reject incomplete flat operations instead of losing operands.
    if m[2].startswith('comb.') or m[2] == 'hw.bitcast':
        typed = split_top(raw, ':')
        if len(typed) != 2 or not typed[1] or typed[0].rstrip().endswith(','):
            raise IndexError(f'Incomplete typed combinational operation at line {number}')
    if m[2] in ('seq.to_clock', 'seq.from_clock'):
        if not re.fullmatch(r'\s*' + SSA, strip_location(raw)):
            raise IndexError(f'Unsupported clock conversion at line {number}')
    # Only the operation portion preceding attributes/types contributes operands.
    # Quoted text never contributes an SSA use.
    clean = STRING.sub('""', raw)
    operands = re.findall(SSA, clean.split(' {', 1)[0].split(' : ', 1)[0])
    hint = re.search(r'sv\.namehint\s*=\s*("(?:\\.|[^"\\])*")', raw)
    return {'results': results, 'operands': operands, 'namehint': unquote(hint[1]) if hint else None,
            **anchor(number, text, m[2])}


def parse_instance(op):
    text = op['text']
    tail = text.split('hw.instance ', 1)[1]
    m = re.match(r'("(?:\\.|[^"\\])*"|[A-Za-z0-9_$.-]+)\s+(?:sym\s+' + SYMBOL + r'\s+)?(' + SYMBOL + r')', tail)
    if not m:
        raise IndexError('Unsupported instance at line ' + str(op['line']))
    name, target = unquote(m[1]), unquote(m[2][1:])
    begin, parameters = m.end(), None
    if tail[begin:begin + 1] == '<':
        parameters, begin = balanced_group(tail, begin, '<', '>')
    inputs, end = balanced_group(tail, begin)
    rest = tail[end:].lstrip()
    if not rest.startswith('-> '):
        raise IndexError('Missing instance result port list')
    outputs, _ = balanced_group(rest, 3)
    bindings = []
    for part in split_top(inputs) if inputs.strip() else []:
        fields = split_top(part, ':')
        if len(fields) != 3 or not re.fullmatch(SSA, fields[1]):
            raise IndexError('Unsupported instance input binding: ' + part)
        bindings.append({'port': unquote(fields[0]), 'ssa': fields[1], 'type': fields[2]})
    results = []
    for part in split_top(outputs) if outputs.strip() else []:
        fields = split_top(part, ':')
        if len(fields) != 2:
            raise IndexError('Unsupported instance output binding: ' + part)
        results.append({'port': unquote(fields[0]), 'type': fields[1]})
    if len(results) != len(op['results']):
        raise IndexError('Instance SSA/result arity mismatch')
    for result, ssa in zip(results, op['results']):
        result['ssa'] = ssa
    return {'name': name, 'module': target, 'parameters_text': parameters,
            'inputs': bindings, 'outputs': results,
            'locator': anchor(op['line'], op['text'], op['op'])}


def read_module(module, lines):
    defs, instances, registers, memories, memory_ports, outputs = {}, [], [], [], [], []
    for i in range(module['_start'] + 1, module['_end']):
        line = lines[i]
        if len(line) - len(line.lstrip()) != 4:
            if re.search(r'\bhw\.instance\b|\bseq\.(?:firreg|compreg|firmem)\b', STRING.sub('', line)):
                raise IndexError(f'Nested required structure unsupported at line {i + 1}')
            continue
        op = parse_operation(line, i + 1)
        if op:
            for ssa in op['results']:
                if ssa in defs:
                    raise IndexError('Duplicate SSA definition in ' + module['name'])
                defs[ssa] = op
            if op['op'] == 'hw.instance':
                instances.append(parse_instance(op))
            elif op['op'] in ('seq.firreg', 'seq.compreg'):
                m = re.search(r'\bseq\.(firreg|compreg)\s+(' + SSA + r')\s+clock\s+(' + SSA + r')(?:\s+reset\s+(sync|async)\s+(' + SSA + r'),\s*(' + SSA + r'))?', line)
                if not m or len(op['results']) != 1:
                    raise IndexError(f'Unsupported register at line {i + 1}')
                if not line[m.end():].startswith((' {', ' :')):
                    raise IndexError(f'Unsupported register suffix at line {i + 1}')
                typ = re.search(r'}?\s*:\s*(i\d+)\s*(?:loc\(.*\))?$', line)
                if typ is None:
                    raise IndexError(f'Unsupported register type at line {i + 1}')
                registers.append({'ssa': op['results'][0], 'next': m[2], 'clock': m[3],
                                  'type': typ[1], 'width_bits': int(typ[1][1:]),
                                  'edge': 'rising' if m[1] == 'firreg' else 'unknown',
                                  'reset': {'kind': m[4], 'signal': m[5], 'value': m[6]} if m[4] else None,
                                  'locator': anchor(i + 1, line, op['op'])})
            elif op['op'] == 'seq.firmem':
                m = re.search(r'seq\.firmem\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+).*:\s*<(\d+)\s*x\s*(\d+)>', line)
                if not m:
                    raise IndexError(f'Unsupported memory declaration at line {i + 1}')
                memories.append({'ssa': op['results'][0], 'read_latency_declared': int(m[1]),
                                 'write_latency_declared': int(m[2]), 'read_under_write': m[3],
                                 'write_under_write': m[4], 'depth_declared': int(m[5]),
                                 'width_bits_declared': int(m[6]), 'locator': anchor(i + 1, line, op['op'])})
            elif op['op'].startswith('seq.firmem.'):
                memory_ports.append(anchor(i + 1, line, op['op']))
        elif line.strip().startswith('seq.firmem.'):
            memory_ports.append(anchor(i + 1, line, line.strip().split()[0]))
        elif line.strip().startswith('hw.output'):
            outputs.append((i + 1, line))
        elif line.strip().startswith(('%', 'hw.instance', 'seq.firreg', 'seq.compreg', 'seq.firmem')):
            raise IndexError(f'Unsupported SSA result syntax at line {i + 1}')
    if len({x['name'] for x in instances}) != len(instances):
        raise IndexError('Duplicate instance name in ' + module['name'])
    inputs = {p['ssa'] for p in module['ports'] if p['direction'] == 'in'}
    if inputs.intersection(defs):
        raise IndexError('SSA definition collides with module input in ' + module['name'])
    defined = inputs | defs.keys()
    for instance in instances:
        if any(p['ssa'] not in defined for p in instance['inputs']):
            raise IndexError('Missing instance input SSA definition in ' + module['name'])
    for register in registers:
        uses = [register['next'], register['clock']]
        if register['reset']:
            uses += [register['reset']['signal'], register['reset']['value']]
        if any(ssa not in defined for ssa in uses):
            raise IndexError('Missing register operand SSA definition in ' + module['name'])
    output_ports = [p for p in module['ports'] if p['direction'] == 'out']
    if module['kind'] == 'hw.module':
        if len(outputs) != 1:
            raise IndexError('Missing/ambiguous hw.output in ' + module['name'])
        number, text = outputs[0]
        values = re.findall(SSA, text.split(' : ', 1)[0])
        if len(values) != len(output_ports):
            raise IndexError('Module output arity mismatch in ' + module['name'])
        if any(v not in defined for v in values):
            raise IndexError('Missing event-cone SSA definition in ' + module['name'])
        module['output_locator'] = anchor(number, text, 'hw.output')
        for index, (port, value) in enumerate(zip(output_ports, values)):
            port['ssa'] = value
            port['output_locator'] = {'line': number, 'operand_index': index}
    module.update(instances=instances, registers=registers, internal_memories=memories,
                  memory_ports=memory_ports, _defs=defs)
    return module


def cone(module, value):
    """A textual local use-def DAG, with no evaluation or intermodule inference."""
    inputs = {p['ssa']: p for p in module['ports'] if p['direction'] == 'in'}
    nodes, seen, pending = [], set(), [value]
    while pending:
        ssa = pending.pop()
        if ssa in seen:
            continue
        seen.add(ssa)
        if ssa in inputs:
            nodes.append({'ssa': ssa, 'boundary': 'module_input', 'port': inputs[ssa]['name']})
            continue
        op = module['_defs'].get(ssa)
        if op is None:
            raise IndexError('Missing event-cone SSA definition: ' + module['name'] + '/' + ssa)
        boundary = ('register' if op['op'] in ('seq.firreg', 'seq.compreg') else
                    'instance_output' if op['op'] == 'hw.instance' else
                    'constant' if op['op'] == 'hw.constant' else
                    'unsupported_operation' if op['op'] not in COMBINATIONAL else None)
        nodes.append({'ssa': ssa, 'boundary': boundary, 'operands': op['operands'],
                      'locator': anchor(op['line'], op['text'], op['op'])})
        if boundary is None:
            pending.extend(reversed(op['operands']))
    return nodes


def build_index(text, root='AtlasCore'):
    lines = text.splitlines()
    table = module_table(lines)
    if root not in table:
        raise IndexError('Selected root module is absent: ' + root)
    selected = {}

    def visit(name, ancestors):
        if name in ancestors:
            raise IndexError('Cyclic module instance hierarchy: ' + name)
        if name not in table:
            raise IndexError('Unresolved module reference: ' + name)
        if name in selected:
            return
        module = read_module(table[name], lines)
        selected[name] = module
        for instance in module['instances']:
            visit(instance['module'], ancestors | {name})
            child = selected[instance['module']]
            for direction, bindings in [('in', instance['inputs']), ('out', instance['outputs'])]:
                expected = [(p['name'], p['type']) for p in child['ports'] if p['direction'] == direction]
                actual = [(p['port'], p['type']) for p in bindings]
                if actual != expected:
                    raise IndexError('Instance port correspondence mismatch: ' + name + '/' + instance['name'])

    visit(root, set())
    paths = []

    def expand(name, path):
        paths.append({'path': path, 'module': name})
        for instance in selected[name]['instances']:
            expand(instance['module'], path + [instance['name']])

    expand(root, [root])
    events = []
    scalar = selected.get('ScalarCore')
    if root == 'AtlasCore' and scalar is None:
        raise IndexError('Required ScalarCore anchor is absent')
    if scalar:
        for hint in REQUIRED_HINTS:
            matches = {op['line']: op for op in scalar['_defs'].values() if op['namehint'] == hint}
            if len(matches) != 1:
                raise IndexError('Missing/ambiguous ScalarCore event hint: ' + hint)
            op = next(iter(matches.values()))
            if len(op['results']) != 1:
                raise IndexError('Ambiguous event-hint result: ' + hint)
            events.append({'module': 'ScalarCore', 'selector': {'namehint': hint},
                           'ssa': op['results'][0], 'cone': cone(scalar, op['results'][0])})
    event_modules = {root, *[i['module'] for i in selected[root]['instances']]}
    for name in sorted(event_modules):
        module = selected[name]
        if module['kind'] != 'hw.module':
            continue
        for port in module['ports']:
            if port['direction'] == 'out' and re.search(r'(?:valid|ready|[Gg]rant|[Bb]usy|halted|inst_retire|set_ecall|set_ebreak|set_illegal)(?:_\d+)?$', port['name']):
                events.append({'module': name, 'selector': {'output_port': port['name']},
                               'ssa': port['ssa'], 'cone': cone(module, port['ssa'])})
    for event in events:
        event.update(interpretation='candidate_only', clock_edge='unknown',
                     relation_to_instruction_issue='unknown', numeric_timing=None,
                     qualified_for_scheduling=False)
    public_modules = []
    for name in sorted(selected):
        module = selected[name]
        public_modules.append({k: v for k, v in module.items() if not k.startswith('_')})
    return {'schema': 'atlas.retained_hw_index.v0', 'root': root,
            'qualification': 'structural observations and unevaluated local event cones only',
            'timing_rules_enabled': False, 'resolver_bindings': [],
            'modules': public_modules, 'instance_paths': paths, 'candidate_events': events,
            'selector_scope': {'namehint_module': 'ScalarCore', 'required_namehints': list(REQUIRED_HINTS) if scalar else [],
                               'output_modules': sorted(event_modules),
                               'output_names': 'valid/ready/grant/busy/halted/retire/exception suffixes, optional numeric index'},
            'limitations': ['Not an exhaustive instruction-event mapping or timing qualification.',
                            'Cones stop at registers, module inputs, instance outputs and unsupported operations.',
                            'Top-level OM metadata and blackbox implementation behavior are not interpreted.',
                            'Memory declarations and register counts do not establish operation latency or admission capacity.',
                            'Retention producer/source provenance is recorded, not independently reconstructed.']}


def run(manifest_path, output, root='AtlasCore'):
    manifest_path, output = allowed(manifest_path), allowed(output)
    if any(p in ('.agents', '.codex') for p in output.parts):
        raise IndexError('Generated evidence belongs outside agent notes/configuration')
    if output.exists():
        raise IndexError('Output directory must be new')
    manifest_identity = identity(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'atlas.retained_hw_ir.v0' or manifest.get('state') != 'verified':
        raise IndexError('Expected a verified retained-HW manifest')
    hw, hw_identity = checked_member(manifest.get('hardware_ir'), manifest_path.parent)
    executed = allowed(__file__)
    script_identity, script_bytes = identity(executed), executed.read_bytes()
    report = build_index(hw.read_text(), root)
    report['inputs'] = {'retention_manifest': manifest_identity, 'hardware_ir': hw_identity,
                        'extractor': script_identity,
                        'declared_elaboration_inputs': manifest.get('inputs', {}),
                        'declared_input_snapshots': manifest.get('snapshots', {}),
                        'retention_producer': manifest.get('producer'),
                        'target_config': manifest.get('target_config'),
                        'source_config': manifest.get('source_config')}
    if identity(hw) != hw_identity or identity(manifest_path) != manifest_identity or identity(executed) != script_identity:
        raise IndexError('Selected input/extractor changed during indexing')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'retention-manifest.json').write_bytes(manifest_bytes)
    (output / 'index-retained-hw.executed.py').write_bytes(script_bytes)
    report['snapshots'] = {'retention_manifest': 'retention-manifest.json',
                           'extractor': 'index-retained-hw.executed.py'}
    (output / 'hw-index.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', default='AtlasCore')
    args = parser.parse_args(argv)
    try:
        report = run(args.manifest, args.output, args.root)
    except (IndexError, OSError, ValueError) as exc:
        parser.exit(1, 'index-retained-hw: ' + str(exc) + '\n')
    print(json.dumps({'report': str(args.output / 'hw-index.json'),
                      'modules': len(report['modules']), 'instances': len(report['instance_paths']),
                      'candidate_events': len(report['candidate_events']),
                      'timing_rules_enabled': False}))


if __name__ == '__main__':
    main()
