#!/usr/bin/env python3
"""Replay byte-bound retained LSU hardware and compare current compiler APIs.

This is a conditional standalone LSU experiment with one-cycle SRAM responders,
not a compiler profile provider or integrated EE290SimConfig qualification.
Generated artifacts are restricted to a new ignored build/rtl-timing directory.
"""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


def bootstrap_allowed(path):
    path = Path(path).absolute()
    if any('hammer' in p.lower() or 'vlsi' in p.lower() for p in path.parts):
        raise ValueError('Prohibited path component')
    resolved = path.resolve()
    if any('hammer' in p.lower() or 'vlsi' in p.lower() for p in resolved.parts):
        raise ValueError('Prohibited symlink target component')
    return resolved


SCRIPT = bootstrap_allowed(__file__)
REPO = SCRIPT.parent.parent
INDEX_PATH = bootstrap_allowed(SCRIPT.parent / 'index-retained-hw.py')
SPEC = importlib.util.spec_from_file_location('retained_hw_index', INDEX_PATH)
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)
allowed, identity, checked_member = INDEX.allowed, INDEX.identity, INDEX.checked_member


class CheckError(ValueError):
    pass


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def no_cores():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class Runner:
    def __init__(self, output, report, environment):
        self.output, self.report, self.environment = output, report, environment

    def run(self, stage, argv, required=True):
        argv = [str(x) for x in argv]
        start = time.monotonic()
        result = subprocess.run(argv, cwd=self.output, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, preexec_fn=no_cores,
                                env=self.environment)
        stdout = self.output / (stage + '.stdout.log')
        stderr = self.output / (stage + '.stderr.log')
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        item = {'stage': stage, 'argv': argv, 'cwd': str(self.output),
                'returncode': result.returncode,
                'elapsed_seconds': time.monotonic() - start,
                'stdout': identity(stdout), 'stderr': identity(stderr)}
        self.report['commands'].append(item)
        if required and result.returncode:
            raise CheckError(f'{stage} failed ({result.returncode}); see {stderr}')
        return result


def selected_slice(hw_bytes):
    text = hw_bytes.decode('utf-8')
    lines = text.splitlines(keepends=True)
    table = INDEX.module_table([line.rstrip('\n') for line in lines])
    if 'LSU' not in table or table['LSU']['kind'] != 'hw.module':
        raise CheckError('Required internal LSU module is absent')
    lsu = INDEX.read_module(table['LSU'], [line.rstrip('\n') for line in lines])
    if lsu['instances'] or lsu['internal_memories'] or lsu['memory_ports']:
        raise CheckError('Replay supports only an instance-free LSU boundary')
    top = [i for i, line in enumerate(lines) if line.startswith('module ') or line == 'module {\n']
    if len(top) != 1:
        raise CheckError('Expected one builtin module wrapper')
    first_hw = min(m['_start'] for m in table.values())
    if first_hw <= top[0]:
        raise CheckError('Invalid module preamble order')
    preamble = ''.join(lines[top[0] + 1:first_hw])
    raw = ''.join(lines[lsu['_start']:lsu['_end'] + 1])
    # Remove only location alias references: declarations live outside the
    # selected module, while assertion/control/fragment operations stay intact.
    standalone = lines[top[0]] + preamble + raw + '}\n'
    standalone = re.sub(r'\s*loc\(#loc[0-9]+\)', '', standalone)
    if 'loc(#' in standalone:
        raise CheckError('Unsupported standalone location reference')
    if 'sv.fatal' not in raw or 'sv.error' not in raw:
        raise CheckError('Selected LSU lacks required preserved assertions')
    for name in ('STOP_COND_FRAGMENT', 'ASSERT_VERBOSE_COND_FRAGMENT'):
        if 'emit.fragment @' + name not in preamble:
            raise CheckError('Required assertion fragment absent: ' + name)
    return raw.encode(), preamble.encode(), standalone.encode(), {
        'module': 'LSU', 'first_line': lsu['_start'] + 1,
        'last_line': lsu['_end'] + 1,
        'preamble_first_line': top[0] + 2, 'preamble_last_line': first_hw,
        'transform': 'Preserve builtin module opening attributes, preamble and complete LSU; remove only loc(#locN) references.',
        'instances': 0, 'assertions_preserved': True,
        'ports': lsu['ports']}


def stream_events(trace, resource_name, write):
    prefix = ('vmem_' if resource_name == 'Vmem' else 'mreg_') + ('write' if write else 'read')
    result = []
    for row in trace:
        if row.get(prefix):
            element = (row[prefix + '_line'] if resource_name == 'Vmem' else
                       row[prefix + '_reg'] * 32 + row[prefix + '_row'])
            result.append({'cycle': row['cycle'], 'element': element})
    return result


def compare_footprints(probe, traces):
    comparisons = []
    for op, sequence in [('vload', 'L'), ('vstore', 'S')]:
        instances = [x for x in probe['instances'] if x['operation'] == op and
                     x['mreg'] == 0 and x['base_word'] == 0 and x['offset'] == 0]
        if len(instances) != 1:
            raise CheckError('Missing or ambiguous isolated compiler reference: ' + op)
        instance = instances[0]
        footprint = instance['footprint']
        observed = traces[sequence]
        memory_accesses = [a for a in footprint['accesses'] if a['resource'] in ('MReg', 'Vmem')]
        required_roles = [('MReg', sequence == 'L'), ('Vmem', sequence == 'S')]
        roles = sorted((a['resource'], a['write']) for a in memory_accesses)
        comparisons.append({'operation': op, 'comparison': 'stream_coverage',
                            'required_roles': required_roles, 'declared_roles': roles,
                            'matches_required_stream_coverage': roles == sorted(required_roles) and not footprint['error']})
        for access in footprint['accesses']:
            if access['resource'] not in ('MReg', 'Vmem'):
                continue
            actual = stream_events(observed, access['resource'], access['write'])
            # Harness operands are mreg3 and line64; compare normalized element
            # order while retaining observed absolute elements in the report.
            base = 64 if access['resource'] == 'Vmem' else 3 * 32
            normalized = [{'cycle': x['cycle'], 'element': x['element'] - base}
                          for x in actual]
            declared = [{'cycle': access['age'] + i * access['step'], 'element': access['first'] + i}
                        for i in range(access['count'])]
            comparisons.append({'operation': op, 'resource': access['resource'],
                                'write': access['write'], 'compiler_access': access,
                                'observed_absolute': actual,
                                'matches_normalized_stream': normalized == declared and
                                    access['count'] == 32 and not access['anywhere'] and not access['at_completion']})
        busy_name = 'load_busy' if sequence == 'L' else 'store_busy'
        busy = [x['cycle'] for x in observed if x.get(busy_name)]
        last_source = max(x['cycle'] for x in observed if
                          x.get('vmem_read' if sequence == 'L' else 'mreg_read'))
        last_destination = max(x['cycle'] for x in observed if
                               x.get('mreg_write' if sequence == 'L' else 'vmem_write'))
        path_name = 'vload_path' if sequence == 'L' else 'vstore_path'
        # Keep names reported by the API rather than guessing their spelling.
        path_holds = [h for h in footprint['holds'] if re.sub(r'[^a-z0-9]', '', h['unit'].lower()) == path_name.replace('_', '')]
        bank_holds = [h for h in footprint['holds'] if re.sub(r'[^a-z0-9]', '', h['unit'].lower()) == 'vmembank' and h['index'] == 0]
        bank_ages = [x['cycle'] for x in observed if x.get('vmem_read' if sequence == 'L' else 'vmem_write')]
        safety_checks = {
            'write_release_covers_last_destination': footprint['write_release'] >= last_destination,
            'store_read_release_covers_last_source': sequence == 'L' or footprint['read_release'] >= last_source,
            'one_path_hold_covers_capture_and_busy': len(path_holds) == 1 and path_holds[0]['index'] == 0 and
                path_holds[0]['from'] <= 0 and path_holds[0]['to'] >= max(busy) and path_holds[0]['alt'] == -1,
            'one_bank_hold_covers_actual_vmem_accesses': len(bank_holds) == 1 and
                bank_holds[0]['from'] <= min(bank_ages) and bank_holds[0]['to'] >= max(bank_ages) and bank_holds[0]['alt'] == -1}
        comparisons.append({'operation': op, 'comparison': 'releases_and_path',
                            'observed_last_source_access': last_source,
                            'observed_last_destination_access': last_destination,
                            'observed_busy_ages': busy,
                            'observed_first_free_age': max(busy) + 1,
                            'compiler_read_release': footprint['read_release'],
                            'compiler_write_release': footprint['write_release'],
                            'compiler_done_age': footprint['done_age'],
                            'compiler_holds': footprint['holds'], 'safety_checks': safety_checks,
                            'matches_release_and_holds': all(safety_checks.values())})
    return comparisons


def compiler_pair(probe, sequence, gap, same_bank, same_reg):
    ops = {'L': 'vload', 'S': 'vstore'}
    address = 'same_bank_disjoint_range' if same_bank else 'different_bank'
    register = 'same_register' if same_reg else 'different_register'
    pairs = [x for x in probe['pair_matrix'] if
             x['first_operation'] == ops[sequence[0]] and
             x['second_operation'] == ops[sequence[1]] and
             x['address_relation'] == address and x['register_relation'] == register]
    if len(pairs) != 1:
        raise CheckError('Missing or ambiguous compiler pair reference')
    pair = pairs[0]
    choices = [x for x in pair['gaps'] if x['gap'] == gap]
    if len(choices) != 1:
        raise CheckError('Missing or ambiguous compiler gap reference')
    choice = choices[0]
    return {'compiler_api_allows': choice['compiler_api_allows'],
            'single_frontend_allows': gap >= 1,
            'bounded_compiler_allows': gap >= 1 and choice['compiler_api_allows'],
            'dependence_distance': pair['dependence_distance'],
            'dependence_reason': pair['dependence_reason'],
            'reservation_conflict': choice['reservation_conflict']}


def replay_cases(runner, executable, probe):
    cases = [('L', 1, 0, 0, 'pass'), ('S', 1, 0, 0, 'pass')]
    for sequence in ('LL', 'SS'):
        cases.extend((sequence, gap, 0, 0, 'busy' if gap == 34 else 'pass')
                     for gap in (34, 35, 36))
    cases += [('LS', 29, 1, 0, 'bank_assert'), ('LS', 30, 1, 0, 'pass'),
              ('LS', 31, 1, 0, 'pass'), ('SL', 33, 1, 0, 'bank_assert'),
              ('SL', 34, 1, 0, 'pass'), ('SL', 35, 1, 0, 'pass'),
              ('LS', 1, 0, 0, 'pass'), ('SL', 1, 0, 0, 'pass'),
              ('LS', 34, 0, 1, 'logical_hazard'), ('LS', 35, 0, 1, 'pass'),
              ('L', 1, 0, 0, 'delayed_response'), ('S', 1, 0, 0, 'delayed_response')]
    records, traces = [], {}
    for sequence, gap, same_bank, same_reg, expected in cases:
        delay = 2 if expected == 'delayed_response' else 1
        name = f'replay-{sequence}-{gap}-{same_bank}-{same_reg}-delay{delay}'
        result = runner.run(name, [executable, sequence, gap, same_bank, same_reg, delay], required=False)
        stdout = result.stdout.decode(errors='replace')
        stderr = result.stderr.decode(errors='replace')
        rows = []
        for line in stdout.splitlines():
            if line.startswith('{'):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        cycles = [x for x in rows if 'cycle' in x]
        summaries = [x for x in rows if x.get('complete')]
        if expected == 'pass':
            checked = result.returncode == 0 and len(summaries) == 1 and summaries[0].get('data_and_guards_pass') is True
        elif expected == 'busy':
            checked = result.returncode == 3 and bool(cycles) and cycles[-1].get('frontend_busy_violation') == 1
        elif expected == 'logical_hazard':
            checked = result.returncode == 3 and bool(cycles) and cycles[-1].get('frontend_logical_write_hazard') == 1
        elif expected == 'delayed_response':
            checked = result.returncode == 5 and len(summaries) == 1 and summaries[0].get('data_and_guards_pass') is False
        else:
            checked = result.returncode != 0 and 'VLOAD and VSTORE target same VMEM bank' in stdout + stderr
        record = {'sequence': sequence, 'gap': gap, 'same_bank': bool(same_bank),
                  'same_register': bool(same_reg), 'expected': expected,
                  'response_delay_cycles': delay,
                  'rejection_basis': ('audited frontend obligation in replay driver' if expected in ('busy', 'logical_hazard') else
                                      'preserved selected LSU assertion' if expected == 'bank_assert' else None),
                  'returncode': result.returncode, 'expectation_met': checked,
                  'log_stage': name, 'completion': summaries}
        if len(sequence) == 2:
            record['compiler_comparison'] = compiler_pair(probe, sequence, gap, same_bank, same_reg)
            record['compiler_matches_bounded_replay_admission'] = record['compiler_comparison']['bounded_compiler_allows'] == (expected == 'pass')
        elif expected != 'delayed_response':
            traces[sequence] = cycles
            source, destination = ('vmem_read', 'mreg_write') if sequence == 'L' else ('mreg_read', 'vmem_write')
            busy = 'load_busy' if sequence == 'L' else 'store_busy'
            record['conditional_trace_checks'] = {
                'source_ages_1_through_32': [x['cycle'] for x in cycles if x.get(source)] == list(range(1, 33)),
                'destination_ages_3_through_34': [x['cycle'] for x in cycles if x.get(destination)] == list(range(3, 35)),
                'busy_ages_1_through_34': [x['cycle'] for x in cycles if x.get(busy)] == list(range(1, 35))}
            record['expectation_met'] &= all(record['conditional_trace_checks'].values())
        records.append(record)
    return records, traces


def identity_rejection_checks(runner, args, manifest):
    checks = []
    for field in ('sha256', 'bytes'):
        bad = json.loads(json.dumps(manifest))
        bad['hardware_ir'][field] = ('0' * 64 if field == 'sha256' else bad['hardware_ir']['bytes'] + 1)
        path = runner.output / ('mismatched-' + field + '-manifest.json')
        json_file(path, bad)
        rejected_output = runner.output / ('must-not-exist-' + field)
        argv = [sys.executable, SCRIPT, '--manifest', path, '--output', rejected_output]
        for option in ('circt-opt', 'verilator', 'cxx', 'make', 'llvm-source-include', 'llvm-build-include'):
            argv.extend(['--' + option, getattr(args, option.replace('-', '_'))])
        if args.verilator_root:
            argv.extend(['--verilator-root', args.verilator_root])
        if args.ar:
            argv.extend(['--ar', args.ar])
        result = runner.run('reject-' + field + '-mismatch', argv, required=False)
        message = result.stderr.decode(errors='replace')
        checks.append({'field': field, 'returncode': result.returncode,
                       'manifest': identity(path), 'rejected_output': str(rejected_output),
                       'rejected_before_output_creation': result.returncode != 0 and
                           'hash/size mismatch' in message and not rejected_output.exists()})
    return checks


def domain_observations(probe):
    observations = []
    for item in probe['instances']:
        base = item['base_word']
        if base is None:
            admitted, reason, line = False, 'unknown effective address', None
        else:
            immediate = ((item['offset'] & 0xfff) ^ 0x800) - 0x800
            line = (((base + immediate * 32) & 0xffffffff) >> 3) & 0xffff
            admitted = line % 32 == 0 and line < 49152 and line + 31 < 49152 and line // 8192 == (line + 31) // 8192
            reason = 'valid aligned one-bank tile' if admitted else 'misaligned or unimplemented VMEM range'
        compiler_accepts = not item['footprint']['error']
        observations.append({'operation': item['operation'], 'base_word': base,
                             'offset': item['offset'], 'mreg': item['mreg'], 'effective_line': line,
                             'bounded_domain_admits': admitted, 'bounded_domain_reason': reason,
                             'compiler_footprint_accepts': compiler_accepts,
                             'compiler_error': item['footprint']['error'],
                             'comparison_scope': 'supported concrete legal tile' if admitted else 'out-of-scope domain observation',
                             'same_admission': admitted == compiler_accepts})
    return observations


def run(args):
    manifest_path, output = allowed(args.manifest), allowed(args.output)
    try:
        output.relative_to(allowed(REPO / 'build/rtl-timing'))
    except ValueError as exc:
        raise CheckError('Generated output must be under build/rtl-timing') from exc
    if output.exists():
        raise CheckError('Output directory must be new')
    # All identity/domain checks precede creation of even the output directory.
    manifest_identity = identity(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'atlas.retained_hw_ir.v0' or manifest.get('state') != 'verified':
        raise CheckError('Expected a verified retained-HW manifest')
    hw, hw_identity = checked_member(manifest.get('hardware_ir'), manifest_path.parent)
    hw_bytes = hw.read_bytes()
    raw, preamble, standalone, extraction = selected_slice(hw_bytes)
    tools = {key: allowed(getattr(args, key)) for key in ('circt_opt', 'verilator', 'cxx', 'make')}
    includes = {key: allowed(getattr(args, key)) for key in ('llvm_source_include', 'llvm_build_include')}
    for path in tools.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CheckError('Missing executable: ' + str(path))
    for path in includes.values():
        if not path.is_dir():
            raise CheckError('Missing include directory: ' + str(path))
    tools['python'] = allowed(sys.executable)
    tools['ar'] = allowed(args.ar) if args.ar else allowed(tools['cxx'].parent / 'ar')
    if not tools['ar'].is_file() or not os.access(tools['ar'], os.X_OK):
        raise CheckError('Archive tool unavailable; provide --ar')
    sources = {
        'checker': SCRIPT, 'indexer': INDEX_PATH,
        'replay_harness': allowed(REPO / 'test/vls-hw-replay.cpp'),
        'compiler_probe': allowed(REPO / 'test/vls-timing-probe.cpp'),
        'timing_cpp': allowed(REPO / 'lib/AtlasTiming.cpp'),
        'timing_header': allowed(REPO / 'include/Atlas/AtlasTiming.h')}
    source_bytes = {key: path.read_bytes() for key, path in sources.items()}
    source_ids = {key: identity(path) for key, path in sources.items()}
    if identity(hw) != hw_identity or identity(manifest_path) != manifest_identity:
        raise CheckError('Selected input changed during extraction')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'atlas.conditional_vls_hw_check.v0', 'state': 'running',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'target_config': manifest.get('target_config'),
              'target_reference': manifest.get('target_reference'),
              'source_config': manifest.get('source_config'),
              'rtl_rules_enabled': False, 'resolver_bindings': [],
              'inputs': {'manifest': manifest_identity, 'hardware_ir': hw_identity,
                         'sources': source_ids, 'llvm_include_roots': {k: str(v) for k, v in includes.items()}},
              'extraction': extraction, 'commands': [], 'tools': {},
              'scope': {'execution': 'selected standalone LSU hardware with injected one-cycle synchronous SRAM responses',
                        'other_engines': 'quiescent', 'external_traffic': 'none',
                        'addresses': 'known aligned in-range 32-line tiles wholly in one VMEM bank',
                        'mregs': 'distinct logical registers during overlap; same-register load then store only at gap35 after busy release',
                        'cycle_origin': 'command-valid/op sampled on rising edge age zero; pre-edge observation',
                        'opaque_sram_implementation_qualified': False,
                        'integrated_target_execution_qualified': False},
              'limitations': ['No general timing provider is enabled.',
                              'SRAM extern implementation and read-during-write semantics remain unqualified.',
                              'Passing slice tests do not qualify integrated EE290SimConfig execution.',
                              'Tool and compiled dependency identities are recorded; the toolchain is not claimed hermetic.',
                              'Same-register forwarding and cross-engine physical-port alias traces are not replayed.',
                              'Compiler API discrepancies are observations, not accepted RTL-derived compiler rules.']}
    # Relocated Verilator packages may contain a stale compiled installation
    # prefix. Derive the package root from the selected executable, not a
    # workspace-specific path or an inherited stale VERILATOR_ROOT.
    verilator_root = allowed(args.verilator_root) if args.verilator_root else allowed(tools['verilator'].parent.parent / 'share/verilator')
    if not allowed(verilator_root / 'include/verilated_std.sv').is_file():
        raise CheckError('Verilator installation root is unavailable; provide --verilator-root')
    environment = dict(os.environ, VERILATOR_ROOT=str(verilator_root))
    report['environment_overrides'] = {'VERILATOR_ROOT': str(verilator_root)}
    backend = allowed(tools['verilator'].parent / 'verilator_bin')
    effective_verilator = backend if backend.is_file() and os.access(backend, os.X_OK) else tools['verilator']
    report['verilator_driver'] = {'selected': identity(tools['verilator']),
                                'executed': identity(effective_verilator),
                                'reason': 'Use sibling package backend with selected installation-root override when present; wrapper otherwise.'}
    runner = Runner(output, report, environment)
    snapshot = output / 'snapshot'
    (snapshot / 'include/Atlas').mkdir(parents=True)
    names = {'checker': 'check-vls-timing.executed.py', 'indexer': 'index-retained-hw.executed.py',
             'replay_harness': 'vls-hw-replay.cpp', 'compiler_probe': 'vls-timing-probe.cpp',
             'timing_cpp': 'AtlasTiming.cpp', 'timing_header': 'include/Atlas/AtlasTiming.h'}
    for key, name in names.items():
        (snapshot / name).write_bytes(source_bytes[key])
    (output / 'retention-manifest.json').write_bytes(manifest_bytes)
    (output / 'LSU.raw.mlir').write_bytes(raw)
    (output / 'preamble.raw.mlir').write_bytes(preamble)
    (output / 'LSU.mlir').write_bytes(standalone)
    try:
        for key, tool in tools.items():
            result = runner.run('version-' + key, [effective_verilator if key == 'verilator' else tool, '--version'])
            report['tools'][key] = {**identity(tool), 'version': result.stdout.decode(errors='replace').strip()}
        exported = runner.run('export-lsu', [tools['circt_opt'], output / 'LSU.mlir', '--verify-each',
                                           '--lower-seq-to-sv', '--lower-verif-to-sv', '--export-verilog',
                                           '-o', output / 'LSU.lowered.mlir'])
        # circt-opt's export pass emits Verilog on stdout; -o selects the
        # post-pass MLIR output, not the Verilog artifact.
        (output / 'LSU.sv').write_bytes(exported.stdout)
        sv = (output / 'LSU.sv').read_text()
        if '$fatal' not in sv or 'VLOAD and VSTORE target same VMEM bank' not in sv:
            raise CheckError('Verilog export lost required assertions')
        runner.run('verilate-lsu', [effective_verilator, '--cc', '--exe', '--top-module', 'LSU',
                                    '--prefix', 'VLSU', '--Mdir', output / 'obj', '-Wno-fatal',
                                    output / 'LSU.sv', snapshot / 'vls-hw-replay.cpp'])
        runner.run('build-lsu', [tools['make'], '-C', output / 'obj', '-f', 'VLSU.mk',
                               'CXX=' + str(tools['cxx']), 'LINK=' + str(tools['cxx']),
                               'PYTHON3=' + str(tools['python']), 'AR=' + str(tools['ar']), '-j4'])
        probe_exe = output / 'vls-timing-probe'
        runner.run('build-compiler-probe', [tools['cxx'], '-std=c++17', '-O2', '-MMD',
                                          '-I', snapshot / 'include', '-I', includes['llvm_source_include'],
                                          '-I', includes['llvm_build_include'],
                                          snapshot / 'vls-timing-probe.cpp', snapshot / 'AtlasTiming.cpp',
                                          '-o', probe_exe])
        result = runner.run('compiler-probe', [probe_exe, '--suite'])
        probe = json.loads(result.stdout)
        if probe.get('schema') != 'atlas.compiler_vls_probe.v0' or len(probe.get('instances', [])) != 22 or len(probe.get('pair_matrix', [])) != 36:
            raise CheckError('Unexpected compiler probe schema or coverage')
        if any(len(pair['gaps']) != 37 for pair in probe['pair_matrix']):
            raise CheckError('Unexpected compiler pair gap coverage')
        json_file(output / 'compiler-probe.json', probe)
        report['identity_rejection_checks'] = identity_rejection_checks(runner, args, manifest)
        report['instance_domain_observations'] = domain_observations(probe)
        records, traces = replay_cases(runner, output / 'obj/VLSU', probe)
        report['replay_cases'] = records
        report['footprint_comparisons'] = compare_footprints(probe, traces)
        report['gap_zero_guard'] = {'reason': 'one scalar frontend issues at most one instruction per clock edge',
                                   'enforced_by_comparison': True,
                                   'compiler_api_zero_gap_acceptances': sum(
                                       pair['gaps'][0]['compiler_api_allows'] for pair in probe['pair_matrix']),
                                   'bounded_zero_gap_acceptances': 0}
        report['compiler_discrepancies'] = {
            'access_streams': [x for x in report['footprint_comparisons'] if x.get('matches_normalized_stream') is False],
            'coverage_releases_and_holds': [x for x in report['footprint_comparisons'] if
                x.get('matches_required_stream_coverage') is False or x.get('matches_release_and_holds') is False],
            'paired_admission': [x for x in records if x.get('compiler_matches_bounded_replay_admission') is False],
            'supported_instance_admission': [x for x in report['instance_domain_observations'] if x['bounded_domain_admits'] and not x['same_admission']]}
        report['out_of_scope_domain_observations'] = [x for x in report['instance_domain_observations'] if not x['bounded_domain_admits']]
        report['compiler_unknown_address_instances'] = [x for x in probe['instances'] if x['base_word'] is None]
        checks_passed = all(x['expectation_met'] for x in records) and all(x['rejected_before_output_creation'] for x in report['identity_rejection_checks'])
        report['conditional_replay_expectations_met'] = checks_passed
        report['compiler_comparison_matches'] = not any(report['compiler_discrepancies'].values())
        report['state'] = ('conditional_checks_failed' if not checks_passed else
                           'compiler_discrepancies_found' if not report['compiler_comparison_matches'] else
                           'conditional_checks_passed')
        # Capture non-system compiler headers consumed by the copied probe.
        dependencies = set()
        for depfile in output.glob('*.d'):
            text = depfile.read_text().replace('\\\n', ' ')
            dependencies.update(text.split(':', 1)[1].split())
        report['inputs']['compiled_dependency_identities'] = [identity(allowed(x)) for x in sorted(dependencies)]
        for key, path in sources.items():
            if identity(path) != source_ids[key]:
                raise CheckError('Source changed during replay: ' + key)
        if identity(hw) != hw_identity or identity(manifest_path) != manifest_identity:
            raise CheckError('Selected input changed during replay')
    except (CheckError, OSError, ValueError) as exc:
        report['state'], report['failure'] = 'failed', str(exc)
    report['artifacts'] = {}
    for relative in ['retention-manifest.json', 'LSU.raw.mlir', 'preamble.raw.mlir', 'LSU.mlir',
                     'LSU.lowered.mlir', 'LSU.sv', 'obj/VLSU', 'vls-timing-probe', 'compiler-probe.json']:
        path = allowed(output / relative)
        if path.is_file():
            report['artifacts'][relative] = identity(path)
    report['snapshots'] = {key: identity(snapshot / name) for key, name in names.items()}
    json_file(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'output', 'circt-opt', 'verilator', 'cxx', 'make',
                'llvm-source-include', 'llvm-build-include'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--verilator-root', type=Path,
                        help='Optional relocated Verilator installation root; otherwise inferred from executable prefix')
    parser.add_argument('--ar', type=Path, help='Optional archive tool; otherwise sibling ar of selected C++ tool')
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (CheckError, INDEX.IndexError, OSError, ValueError) as exc:
        parser.exit(1, 'check-vls-timing: ' + str(exc) + '\n')
    print(json.dumps({'report': str(args.output / 'report.json'), 'state': report['state'],
                      'rtl_rules_enabled': False}))
    return 0 if report['state'] == 'conditional_checks_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
