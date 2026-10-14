#!/usr/bin/env python3
"""Rebuild, cache and compare RTL timing artifacts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from rtl_extract.regen_cache import StageRunner, output_path, safe_path, source_files, write_json

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / 'tools/rtl_extract/specs/regenerate.json'

def executable(value):
    value = str(value)
    for prefix in ('',) if os.sep in value else os.get_exec_path():
        try:
            path = safe_path(Path(prefix) / value)
        except ValueError:
            if os.sep in value:
                raise
            continue
        if path.is_file() and os.access(path, os.X_OK):
            return path  # Keep a venv Python's lexical path.
    raise ValueError(f'tool not found: {value}')


def version(path):
    return subprocess.check_output([str(path), '--version'], text=True, stderr=subprocess.STDOUT).strip()

def emit_environment(args, config, root):
    paths = {name: str(output_path(root, value)) for name, value in config['environment'].items()}
    paths.update(ATLAS_PAIR_SWEEPS=os.pathsep.join(str(output_path(root, s['output']))
                                                for s in config['sweeps'] if s['kind'] == 'pairs'),
                 ATLAS_OP_PAIRS=os.pathsep.join(str(output_path(root, f'out/pairs-mxu{i}.json')) for i in (0, 1)),
                 ATLAS_TIMING_PROBE=str(args.bin / 'rtl-timing-facts-probe'), ATLAS_EMIT=str(args.bin / 'atlas-emit'),
                 ATLAS_OOT_BIN_DIR=str(args.bin), VPY=str(args.python), PYTHONDONTWRITEBYTECODE='1')
    output_path(root, 'env.sh').write_text(''.join(f'export {name}={shlex.quote(value)}\n' for name, value in paths.items()))

def stages(config, selected):
    names = ['lower', 'exporter', 'facts', 'compiler_facts',
             *[f"{s['kind']}-{s['name']}" for s in config['sweeps']],
             'pair_facts', 'probe', 'model', 'replay', 'replay_pairs', 'environment']
    wanted = set(selected.split(',')) if selected else set(names)
    if 'sweeps' in wanted:
        wanted.remove('sweeps')
        wanted.update(f"{s['kind']}-{s['name']}" for s in config['sweeps'])
    if wanted - set(names):
        raise ValueError(f'unknown stages: {sorted(wanted - set(names))}')
    return [name for name in names if name in wanted]

def artifacts(config, selected):
    mapping = {'facts': {'out/all.json': 'all.json'}, 'compiler_facts': {'out/compiler-facts.json': 'compiler-facts.json'},
               'pair_facts': {'out/op-pairs.json': 'op-pairs.json'}}
    mapping.update({f"{s['kind']}-{s['name']}": {s['output']: s['reference']} for s in config['sweeps']})
    return {path: reference for name in selected for path, reference in mapping.get(name, {}).items()}

def regenerate(args, config, root, runner, selected):
    python, cmake, cxx = executable(args.python), executable(args.cmake), executable(args.cxx)
    circt = args.circt_prefix
    firtool, opt = executable(circt / 'bin/firtool'), executable(circt / 'bin/circt-opt')
    py_versions = {'python': version(python), 'PyYAML': subprocess.check_output(
        [str(python), '-c', 'import yaml; print(yaml.__version__)'], text=True).strip()}
    path = lambda name: output_path(root, name)
    hw, exporter = path('ir-debug/ee290.debug.hw.mlir'), path('exporter/hw_ir_export')
    facts, compiler_facts, pair_facts, model = map(path, ('out/all.json', 'out/compiler-facts.json',
                                                       'out/op-pairs.json', 'model/VAtlasCore'))
    engine_sources = [p for p in source_files(REPO / 'tools/rtl_extract', {'.py', '.yaml', '.json'})
                      if not p.name.startswith('regen_')]
    def run(name, commands, inputs, outputs, versions=None, prepare=None, inventory=None):
        return runner.run(name, commands, [python, Path(__file__), REPO / 'tools/rtl_extract/regen_cache.py',
                          REPO / 'tools/rtl_extract/facts.py', *inputs],
                          outputs, versions or py_versions, prepare, inventory)
    def remove(directory):
        if directory.exists():
            for current, dirs, files in os.walk(directory, followlinks=False):
                for name in dirs + files:
                    safe_path(Path(current) / name)
            shutil.rmtree(directory)
    for stage in selected:
        if stage == 'lower':
            if 'firtool-1.75.0' not in version(firtool):
                raise ValueError('debug lowering requires firtool 1.75.0')
            script = REPO / 'tools/retain-ee290-hw-ir.py'
            run(stage, [[python, script, '--mode', 'debug', '--firrtl', args.firrtl, '--annotations', args.annotations,
                         '--lowering-options', args.lowering_options, '--firtool', firtool, '--circt-opt', opt,
                         '--output', path('ir-debug')]],
                [script, args.firrtl, args.annotations, args.lowering_options, firtool, opt],
                [hw, *[path('ir-debug/' + name) for name in ('retained-hw.debug.anno.json', 'manifest.json',
                                                            'lower.log', 'verify.log')]],
                {**py_versions, 'firtool': version(firtool), 'circt-opt': version(opt)},
                prepare=lambda: remove(path('ir-debug')))
        elif stage == 'exporter':
            source, directory = REPO / 'tools/rtl_extract/export', path('exporter')
            inputs = source_files(source) + source_files(circt / 'lib/cmake')
            configure = [cmake, '-S', source, '-B', directory, '-G', 'Unix Makefiles', '-DCMAKE_BUILD_TYPE=Release',
                         f'-DCIRCT_DIR={circt}/lib/cmake/circt', f'-DMLIR_DIR={circt}/lib/cmake/mlir',
                         f'-DLLVM_DIR={circt}/lib/cmake/llvm', f'-DCMAKE_CXX_COMPILER={cxx}']
            link = directory / 'CMakeFiles/hw_ir_export.dir/link.txt'
            tool_versions = {'cmake': version(cmake), 'cxx': version(cxx)}
            run('exporter-configure', [configure], [cmake, cxx, *inputs], [directory / 'CMakeCache.txt', link],
                tool_versions, prepare=lambda: remove(directory))
            dependencies = subprocess.check_output([str(cxx), '-std=c++17', '-fno-rtti', '-MM',
                '-I', str(circt / 'include'), str(source / 'hw_ir_export.cpp')], text=True)
            inputs += [safe_path(p) for rule in dependencies.replace('\\\n', '').splitlines()
                       for p in shlex.split(rule.split(':', 1)[1])]
            inputs += [safe_path(p) for p in shlex.split(link.read_text())
                       if p.startswith('/') and (p.endswith('.a') or '.so' in Path(p).name)]
            run(stage, [[cmake, '--build', directory, '-j', str(args.jobs), '--clean-first']],
                [cmake, cxx, link, *inputs], [exporter], tool_versions)
        elif stage == 'facts':
            run(stage, [[python, '-m', 'rtl_extract', '--hw-ir', hw, '--exporter', exporter, '--output', facts]],
                [hw, exporter, *engine_sources], [facts])
        elif stage == 'compiler_facts':
            run(stage, [[python, '-c', 'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])', facts, compiler_facts]],
                [facts], [compiler_facts])
        elif stage.startswith(('pairs-', 'sequences-')):
            item = next(s for s in config['sweeps'] if stage == f"{s['kind']}-{s['name']}")
            run(stage, [[python, '-m', 'rtl_extract.regen_sweeps', '--hw-ir', hw, '--exporter', exporter,
                '--kind', item['kind'], '--name', item['name'], '--jobs', str(args.jobs),
                '--shards', str(item.get('shards', 1)), '--output', path(item['output'])]],
                [hw, exporter, REPO / 'tools/rtl_extract/regen_sweeps.py', *engine_sources], [path(item['output'])])
        elif stage == 'pair_facts':
            pairs = [path(name) for name in config['pair_facts']['pairs']]
            sequences = [path(name) for name in config['pair_facts']['sequences']]
            run(stage, [[python, '-m', 'rtl_extract.pairfacts', '--op-timing', compiler_facts, '--pairs', *pairs,
                         '--sequences', *sequences, '--output', pair_facts]],
                [compiler_facts, *pairs, *sequences, *engine_sources], [pair_facts])
        elif stage == 'probe':
            # Reuse the mapped equality, conservative-bound and coverage checks.
            script = ('import os,sys,json; from pathlib import Path; '
                      'os.environ.update(ATLAS_OP_TIMING=sys.argv[1],ATLAS_TIMING_PROBE=sys.argv[2]); '
                      'import test_rtl_extract_hw as t; f=t.load_facts(); '
                      'p=t.run_probe(sys.argv[2],sorted({m for _,m in t.mapped_ops().values()})); r=t.compare(f,p); '
                      'assert all(row[6] for row in r), t.table([row for row in r if not row[6]]); '
                      'assert not t.coverage(f), t.coverage(f); assert set(t.MAP["not_modeled_records"]) <= set(f); '
                      'Path(sys.argv[3]).write_text(json.dumps(r)+"\\n")')
            probe = args.bin / 'rtl-timing-facts-probe'
            run(stage, [[python, '-c', script, compiler_facts, probe, path('probe.json')]],
                [compiler_facts, probe, REPO / 'test/test_rtl_extract_hw.py',
                 REPO / 'test/rtl-timing-facts-map.yaml', *engine_sources], [path('probe.json')])
        elif stage == 'model':
            verilator, runtime = executable(args.verilator), args.verilator_root
            helper, script = REPO / 'tools/rtl_extract/regen_model.py', REPO / 'tools/replay-ee290-compute.py'
            inputs = source_files(args.rtl, {'.sv', '.v', '.svh', '.vh'})
            if not inputs:
                raise ValueError(f'no RTL files in {args.rtl}')
            run(stage, [[python, '-m', 'rtl_extract.regen_model', 'model', '--script', script, '--rtl', args.rtl,
                '--output', model, '--verilator', verilator, '--verilator-root', runtime,
                '--jobs', str(args.jobs), '--build-root', root]],
                [helper, script, REPO / 'test/ee290-compute-replay.cpp', verilator, executable('g++'),
                 executable('make'), executable('ar'), *inputs, *source_files(runtime / 'include'),
                 *source_files(REPO / 'tools/ee290_replay_cases', {'.py'})], [model, model.parent / 'build.json'],
                {**py_versions, 'verilator': version(verilator), 'cxx': version(executable('g++')),
                 'make': version(executable('make'))}, prepare=lambda: remove(model.parent / 'obj'))
        elif stage in ('replay', 'replay_pairs'):
            paired = stage == 'replay_pairs'
            work, helper = path('replay-pairs' if paired else 'replay'), REPO / 'tools/rtl_extract/regen_model.py'
            script = REPO / 'tools/replay-ee290-compute.py'
            command = [python, '-m', 'rtl_extract.regen_model', 'replay', '--script', script,
                '--facts', compiler_facts, '--bin', args.bin, '--work', work, '--model', model,
                '--jobs', str(args.jobs), '--expected-pass', str(config['replay'][stage]), '--build-root', root]
            inputs = [helper, script, compiler_facts, model, args.bin / 'atlas-opt', args.bin / 'atlas-emit',
                      *source_files(REPO / 'tools/ee290_replay_cases', {'.py'}),
                      *source_files(REPO / 'test/examples', {'.mlir'})]
            if paired:
                command += ['--pairs', pair_facts]
                inputs.append(pair_facts)
            run(stage, [command], inputs, [work / 'summary.json'], inventory=work / 'summary.json')
        elif stage == 'environment':
            run(stage, [[python, Path(__file__), '--manifest', args.manifest, '--build-root', root,
                '--bin', args.bin, '--python', python, '--emit-env-only']], [args.manifest], [path('env.sh')])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    p.add_argument('--build-root', type=Path, default=REPO / 'build/regen')
    for name in ('firrtl', 'annotations', 'lowering-options', 'circt-prefix', 'rtl', 'bin', 'verilator-root'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    for name, default in (('verilator', 'verilator'), ('cmake', 'cmake'), ('cxx', 'g++')):
        p.add_argument('--' + name, default=default)
    p.add_argument('--jobs', type=int, default=32)
    p.add_argument('--stages', help='comma-separated stage names; sweeps selects every pair/sequence sweep')
    p.add_argument('--check', type=Path, help='reference out directory')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--reference-hw-ir', type=Path)
    p.add_argument('--emit-env-only', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, safe_path(value))
    config = json.loads(args.manifest.read_text())
    if config.get('schema') != 'rtl_extract.regenerate.v1':
        p.error('unsupported regeneration manifest schema')
    if args.emit_env_only:
        if not args.bin:
            p.error('--bin required')
        emit_environment(args, config, args.build_root)
        return 0
    if args.check_only and not args.check:
        p.error('--check-only requires --check')
    if not args.check_only:
        missing = [name for name in ('firrtl', 'annotations', 'lowering_options', 'circt_prefix', 'rtl', 'bin',
                                    'verilator_root') if not getattr(args, name)]
        if missing:
            p.error('required for regeneration: ' + ', '.join('--' + name.replace('_', '-') for name in missing))
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(REPO / 'tools'), str(REPO / 'test'),
                  os.environ.get('PYTHONPATH', ''))), 'PYTHONDONTWRITEBYTECODE': '1'}
    search = []
    for directory in os.get_exec_path():
        try:
            search.append(str(safe_path(directory)))
        except ValueError:
            continue
    environment['PATH'] = os.pathsep.join(search)
    runner = StageRunner(args.build_root, REPO, environment)
    start = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    receipt = {'command': runner.command([sys.executable, Path(__file__), *sys.argv[1:]]), 'stages': runner.records}
    result = 0
    try:
        selected = stages(config, args.stages)
        if not args.check_only:
            regenerate(args, config, args.build_root, runner, selected)
        if args.check:
            from rtl_extract.regen_check import compare_artifacts
            mapping = artifacts(config, selected)
            differences = compare_artifacts(args.build_root, args.check, mapping,
                generated_hw_ir=output_path(args.build_root, 'ir-debug/ee290.debug.hw.mlir'),
                reference_hw_ir=args.reference_hw_ir)
            receipt['check'] = {'artifacts': mapping, 'differences': differences}
            for difference in differences:
                print('DIFF ' + difference, flush=True)
            print(f'CHECK {len(mapping)} artifacts: {len(differences)} differences', flush=True)
            result = int(bool(differences))
        receipt['state'] = 'different' if result else 'complete'
    except Exception as error:
        receipt.update(state='failed', error=str(error))
        print(f'FAIL {error}', file=sys.stderr, flush=True)
        result = 1
    finally:
        receipt['wall_seconds'] = time.monotonic() - start
        write_json(args.build_root / 'runs' / f'{stamp}.json', receipt)
        write_json(args.build_root / 'run.json', receipt)
        print(f'RUN MANIFEST {args.build_root / "run.json"} ({receipt["wall_seconds"]:.3f}s)', flush=True)
    return result

if __name__ == '__main__':
    raise SystemExit(main())
