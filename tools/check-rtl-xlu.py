#!/usr/bin/env python3
"""Replay selected retained XLU hardware under an explicit MREG response model.

This produces conditional component evidence, not integrated EE290 qualification.
"""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def bootstrap_allowed(path):
    path = Path(path).absolute()
    if any('hammer' in part.lower() or 'vlsi' in part.lower() for part in path.parts):
        raise ValueError('Prohibited path component')
    resolved = path.resolve()
    if any('hammer' in part.lower() or 'vlsi' in part.lower() for part in resolved.parts):
        raise ValueError('Prohibited symlink target component')
    return resolved


SCRIPT = bootstrap_allowed(__file__)
REPO = SCRIPT.parent.parent
HELPER_PATH = bootstrap_allowed(SCRIPT.parent / 'check-vls-timing.py')
SPEC = importlib.util.spec_from_file_location('xlu_vls_helpers', HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
INDEX = HELPER.INDEX
allowed, identity, checked_member = INDEX.allowed, INDEX.identity, INDEX.checked_member
CheckError, Runner, json_file = HELPER.CheckError, HELPER.Runner, HELPER.json_file
SELECTED_HW = '49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6'
SOURCE_SHA = {
    'XLU.scala': '9015a665d5bdd11fb94766e32353b4b394fc0f561e1c24415c13003b0277a03a',
    'MregFile.scala': '4b9349ccfff146b5322180ab9e88e7a3adcc777a993beee8b863404ad3ddb79f',
    'MregParams.scala': '006b5190cbd5f457718d699222b1b304a6c5c78981706a6af4d3c8c123a1a93c',
}
CASES = [
    dict(name='distinct', src=3, dst=7, gap=-1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='in-place', src=5, dst=5, gap=-1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='paired-physical-bank', src=3, dst=35, gap=-1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='endpoints', src=0, dst=63, gap=-1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='reverse-endpoints', src=63, dst=0, gap=-1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='busy-unrelated-age1', src=3, dst=7, gap=1, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='busy-unrelated-age65', src=3, dst=7, gap=65, second_src=22, second_dst=24, delay=1, accept_second=False),
    dict(name='reuse-age66', src=3, dst=7, gap=66, second_src=7, second_dst=24, delay=1, accept_second=True),
    dict(name='two-cycle-response', src=3, dst=7, gap=-1, second_src=22, second_dst=24, delay=2, accept_second=False),
]


def selected_slice(hw_bytes):
    lines = hw_bytes.decode('utf-8').splitlines(keepends=True)
    stripped = [line.rstrip('\n') for line in lines]
    table = INDEX.module_table(stripped)
    if 'XluEngine' not in table or table['XluEngine']['kind'] != 'hw.module':
        raise CheckError('Required internal XluEngine module is absent')
    module = INDEX.read_module(table['XluEngine'], stripped)
    if module['instances'] or module['internal_memories'] or module['memory_ports']:
        raise CheckError('Replay requires an instance-free XLU register-buffer boundary')
    wrapper = [i for i, line in enumerate(lines) if line.startswith('module ') or line == 'module {\n']
    if len(wrapper) != 1:
        raise CheckError('Expected one builtin module wrapper')
    first_hw = min(item['_start'] for item in table.values())
    if first_hw <= wrapper[0]:
        raise CheckError('Invalid module preamble order')
    preamble = ''.join(lines[wrapper[0] + 1:first_hw])
    raw = ''.join(lines[module['_start']:module['_end'] + 1])
    standalone = lines[wrapper[0]] + preamble + raw + '}\n'
    standalone = re.sub(r'\s*loc\(#loc[0-9]+\)', '', standalone)
    if 'loc(#' in standalone:
        raise CheckError('Unsupported standalone location reference')
    return raw.encode(), preamble.encode(), standalone.encode(), {
        'module': 'XluEngine', 'first_line': module['_start'] + 1,
        'last_line': module['_end'] + 1, 'ports': module['ports'],
        'transform': 'Preserve builtin wrapper, preamble and complete XluEngine; remove only loc(#locN) references.',
        'instances': 0, 'assertions_preserved': True,
        'selected_module_has_assertions': any(token in raw for token in ('sv.fatal', 'sv.error', 'verif.assert')),
        'assertion_preservation_note': 'No XLU assertion is invented; any selected assertion operations remain in the slice.'}


def check_trace(case, rows, summary):
    """Compare observed event tuples with a separately declared finite case."""
    origins = [(0, case['src'], case['dst'])]
    if case['accept_second']:
        origins.append((case['gap'], case['second_src'], case['second_dst']))
    delay = case['delay']
    reads, writes, responses, read_active, write_active = [], [], [], [], []
    for age, src, dst in origins:
        reads.extend((age + 1 + row, src, row) for row in range(32))
        writes.extend((age + 33 + delay + row, dst, row) for row in range(32))
        responses.extend(range(age + 1 + delay, age + 33 + delay))
        read_active.extend((cycle, src) for cycle in range(age + 1, age + 33 + delay))
        write_active.extend((cycle, dst) for cycle in range(age + 1, age + 65 + delay))
    checks = {
        'contiguous_cycles': [row['cycle'] for row in rows] == list(range((max(case['gap'], 0)) + 75)),
        'source_stream': [(x['cycle'], x['mreg_read_reg'], x['mreg_read_row']) for x in rows if x['mreg_read']] == reads,
        'response_stream': [x['cycle'] for x in rows if x['mreg_response']] == responses,
        'destination_stream': [(x['cycle'], x['mreg_write_reg'], x['mreg_write_row']) for x in rows if x['mreg_write']] == writes,
        'source_lifetime': [(x['cycle'], x['read_active_reg']) for x in rows if x['read_active']] == read_active,
        'destination_and_engine_lifetime': [(x['cycle'], x['write_active_reg']) for x in rows if x['write_active']] == write_active,
        'capture_events': [x['cycle'] for x in rows if x['capture']] == [x[0] for x in origins],
        'launch_events': [x['cycle'] for x in rows if x['launch']] == ([0, case['gap']] if case['gap'] >= 0 else [0]),
        'busy_launch_loss': [x['cycle'] for x in rows if x['busy_launch']] == ([case['gap']] if case['gap'] >= 0 and not case['accept_second'] else []),
        'numerical_and_guard_result': summary.get('complete') is True and summary.get('data_and_guards_pass') is True,
        'access_and_capture_counts': summary.get('reads') == len(reads) and summary.get('writes') == len(writes) and summary.get('captures') == len(origins),
        'busy_launch_count': summary.get('busy_launches') == (1 if case['gap'] >= 0 and not case['accept_second'] else 0),
    }
    return checks


def replay_cases(runner, executable):
    records = []
    for case in CASES:
        result = runner.run('replay-' + case['name'], [executable, case['src'], case['dst'],
            case['gap'], case['second_src'], case['second_dst'], case['delay'], int(case['accept_second'])], required=False)
        records_raw = [json.loads(line) for line in result.stdout.decode().splitlines() if line.startswith('{')]
        rows = [item for item in records_raw if 'cycle' in item]
        summaries = [item for item in records_raw if item.get('complete')]
        if len(summaries) != 1:
            raise CheckError('Missing or duplicate completion: ' + case['name'])
        checks = check_trace(case, rows, summaries[0])
        records.append({**case, 'returncode': result.returncode, 'checks': checks,
                        'expectation_met': result.returncode == 0 and all(checks.values()),
                        'completion': summaries[0], 'trace_log_stage': 'replay-' + case['name']})
    return records


def checked_sources(inventory, inventory_path):
    snapshots = {}
    for entry in inventory.get('sources', []):
        name = Path(entry['path']).name
        if name not in SOURCE_SHA:
            continue
        if name in snapshots:
            raise CheckError('Ambiguous reviewed source: ' + name)
        copy = entry.get('captured_copy', {})
        if entry.get('sha256') != SOURCE_SHA[name] or copy.get('sha256') != SOURCE_SHA[name]:
            raise CheckError('Source is outside the reviewed XLU scope: ' + name)
        path, member = checked_member(copy, inventory_path.parent)
        snapshots[name] = member
    if set(snapshots) != set(SOURCE_SHA):
        raise CheckError('Missing reviewed XLU/MREG source snapshots')
    return snapshots


def run(args):
    manifest_path, output = allowed(args.manifest), allowed(args.output)
    try:
        output.relative_to(allowed(REPO / 'build/rtl-timing'))
    except ValueError as exc:
        raise CheckError('Generated output must be under build/rtl-timing') from exc
    if output.exists():
        raise CheckError('Output directory must be new')
    manifest_id = identity(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'atlas.retained_hw_ir.v0' or manifest.get('state') != 'verified' or manifest.get('target_config') != 'EE290SimConfig':
        raise CheckError('Expected verified EE290SimConfig retained-HW manifest')
    hw, hw_id = checked_member(manifest.get('hardware_ir'), manifest_path.parent)
    if hw_id['sha256'] != SELECTED_HW:
        raise CheckError('Selected HW identity is outside this reviewed XLU scope')
    raw, preamble, standalone, extraction = selected_slice(hw.read_bytes())
    sources = {'checker': SCRIPT, 'replay_harness': allowed(REPO / 'test/rtl-xlu-replay.cpp'),
               'vls_helpers': HELPER_PATH, 'indexer': HELPER.INDEX_PATH}
    source_bytes = {key: path.read_bytes() for key, path in sources.items()}
    source_ids = {key: identity(path) for key, path in sources.items()}
    inventory_path = allowed(args.source_inventory)
    inventory_id = identity(inventory_path)
    inventory = json.loads(inventory_path.read_bytes())
    snapshot_ids = checked_sources(inventory, inventory_path)
    tools = {key: allowed(getattr(args, key)) for key in ('circt_opt', 'verilator', 'cxx', 'make', 'ar')}
    for path in tools.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CheckError('Missing executable: ' + str(path))
    root = allowed(args.verilator_root) if args.verilator_root else allowed(tools['verilator'].parent.parent / 'share/verilator')
    if not allowed(root / 'include/verilated_std.sv').is_file():
        raise CheckError('Verilator root is unavailable')
    backend = allowed(tools['verilator'].parent / 'verilator_bin')
    effective = backend if backend.is_file() and os.access(backend, os.X_OK) else tools['verilator']
    environment = dict(os.environ, VERILATOR_ROOT=str(root))
    output.mkdir(parents=True)
    report = {'schema': 'atlas.conditional_xlu_hw_check.v0', 'state': 'running',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'target_config': 'EE290SimConfig', 'rtl_rules_enabled': False,
              'scheduling_qualified': False, 'integrated_target_execution_qualified': False,
              'inputs': {'manifest': manifest_id, 'hardware_ir': hw_id, 'sources': source_ids,
                         'source_inventory': inventory_id, 'source_snapshots': snapshot_ids},
              'extraction': extraction, 'commands': [], 'tools': {},
              'environment_overrides': {'VERILATOR_ROOT': str(root)},
              'scope': {'execution': 'selected standalone XluEngine', 'cycle_origin': 'capture edge age zero; pre-edge observation',
                        'memory': 'injected ordered synchronous MREG response, one cycle except discriminating delay2 case',
                        'physical_mapping': 'bank=id&31; row={id[5],logicalRow[4:0]}',
                        'other_engines_and_external_traffic': 'absent'},
              'resolved_facts': {'read_age': 1, 'read_count': 32, 'read_step': 1,
                                 'write_age': 34, 'write_count': 32, 'write_step': 1,
                                 'read_release': 33, 'write_release': 65,
                                 'engine_hold': {'from': 0, 'to': 65}, 'first_free_age': 66},
              'limitations': ['The MregFile/SRAM implementation is not executed in this component slice.',
                              'Physical alias cases validate responder mapping and XLU phases, not integrated arbitration.',
                              'Original elaboration/source/build linkage is not reconstructed.',
                              'Tool binaries and generated executable are pinned; runtime dependencies are not fully hermetic.',
                              'Busy-launch loss is an observation, never permission for compiler issue.']}
    snapshot = output / 'snapshot'
    snapshot.mkdir()
    names = {'checker': 'check-rtl-xlu.executed.py', 'replay_harness': 'rtl-xlu-replay.cpp',
             'vls_helpers': 'check-vls-timing.executed.py', 'indexer': 'index-retained-hw.executed.py'}
    for key, name in names.items():
        (snapshot / name).write_bytes(source_bytes[key])
    for name, member in snapshot_ids.items():
        (snapshot / name).write_bytes(allowed(member['path']).read_bytes())
    (output / 'retention-manifest.json').write_bytes(manifest_bytes)
    (output / 'XluEngine.raw.mlir').write_bytes(raw)
    (output / 'preamble.raw.mlir').write_bytes(preamble)
    (output / 'XluEngine.mlir').write_bytes(standalone)
    runner = Runner(output, report, environment)
    try:
        for key, tool in tools.items():
            result = runner.run('version-' + key, [effective if key == 'verilator' else tool, '--version'])
            report['tools'][key] = {**identity(tool), 'version': result.stdout.decode(errors='replace').strip()}
        report['tools']['verilator_executed'] = identity(effective)
        exported = runner.run('export-xlu', [tools['circt_opt'], output / 'XluEngine.mlir',
            '--verify-each', '--lower-seq-to-sv', '--lower-verif-to-sv', '--export-verilog', '-o', output / 'XluEngine.lowered.mlir'])
        (output / 'XluEngine.sv').write_bytes(exported.stdout)
        if extraction['selected_module_has_assertions'] and b'$fatal' not in exported.stdout:
            raise CheckError('Verilog export lost selected assertions')
        runner.run('verilate-xlu', [effective, '--cc', '--exe', '--assert', '--top-module', 'XluEngine',
            '--prefix', 'VXLU', '--Mdir', output / 'obj', '-Wno-fatal', output / 'XluEngine.sv', snapshot / 'rtl-xlu-replay.cpp'])
        runner.run('build-xlu', [tools['make'], '-C', output / 'obj', '-f', 'VXLU.mk',
            'CXX=' + str(tools['cxx']), 'LINK=' + str(tools['cxx']), 'PYTHON3=' + sys.executable,
            'AR=' + str(tools['ar']), '-j4'])
        report['replay_cases'] = replay_cases(runner, output / 'obj/VXLU')
        report['conditional_replay_expectations_met'] = all(item['expectation_met'] for item in report['replay_cases'])
        report['state'] = 'conditional_checks_passed' if report['conditional_replay_expectations_met'] else 'conditional_checks_failed'
        for key, path in sources.items():
            if identity(path) != source_ids[key]:
                raise CheckError('Replay source changed: ' + key)
        if identity(hw) != hw_id or identity(manifest_path) != manifest_id or identity(inventory_path) != inventory_id:
            raise CheckError('Selected input changed during replay')
        for member in snapshot_ids.values():
            checked_member(member, inventory_path.parent)
    except (CheckError, INDEX.IndexError, OSError, ValueError) as exc:
        report['state'], report['failure'] = 'failed', str(exc)
    report['artifacts'] = {}
    for name in ['retention-manifest.json', 'XluEngine.raw.mlir', 'preamble.raw.mlir', 'XluEngine.mlir',
                 'XluEngine.lowered.mlir', 'XluEngine.sv', 'obj/VXLU']:
        path = allowed(output / name)
        if path.is_file():
            report['artifacts'][name] = identity(path)
    report['snapshots'] = {key: identity(snapshot / name) for key, name in names.items()}
    report['source_snapshot_artifacts'] = {name: identity(snapshot / name) for name in snapshot_ids}
    json_file(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'output', 'source-inventory', 'circt-opt', 'verilator', 'cxx', 'make', 'ar'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--verilator-root', type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (CheckError, INDEX.IndexError, OSError, ValueError) as exc:
        parser.exit(1, 'check-rtl-xlu: ' + str(exc) + '\n')
    print(json.dumps({'report': str(args.output / 'report.json'), 'state': report['state'],
                      'scheduling_qualified': False}))
    return 0 if report['state'] == 'conditional_checks_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
