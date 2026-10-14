"""Record the existing Verilator builder and fully bound replay results."""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from .regen_cache import safe_path, source_files, sha256, write_json

def relative(path, root):
    return os.path.relpath(safe_path(path), safe_path(root))

def command_record(argv, root):
    result = []
    for arg in map(str, argv):
        if Path(arg).is_absolute(): arg = relative(arg, root)
        elif arg.startswith('CXX='):
            tokens = [token.split('=', 1) for token in shlex.split(arg)]
            arg = shlex.join(name + '=' + relative(value, root) for name, value in tokens)
        result.append(arg)
    return result

def inventory(directory, root, exclude=()):
    excluded = set(map(safe_path, exclude))
    return [{'path': relative(path, root), 'sha256': sha256(path)}
            for path in source_files(directory) if path not in excluded and not path.is_symlink()]

def load_helper(script):
    spec = importlib.util.spec_from_file_location('atlas_regeneration_replay', safe_path(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def model(args):
    helper = load_helper(args.script)
    output = safe_path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = args.build_root or output.parent.parent
    commands, original = [], helper.sh

    class RTLInputs:
        def glob(self, pattern):
            return [safe_path(path) for path in safe_path(args.rtl).glob(pattern)
                    if not any(word in path.name.lower() for word in ('hammer', 'vlsi'))]

    def recorded(argv, timeout, env=None):
        commands.append({'argv': command_record(argv, root), 'environment':
                         {'VERILATOR_ROOT': relative(args.verilator_root, root)} if args.verilator_root else {}})
        return original(argv, timeout, env)

    helper.sh = recorded
    start = time.monotonic()
    build_args = argparse.Namespace(**{**vars(args), 'rtl': RTLInputs()})
    binary = safe_path(helper.build_model(build_args, output.parent))
    if binary != output: shutil.copy2(binary, output)
    version = subprocess.run([args.verilator, '--version'], check=True, capture_output=True, text=True).stdout.strip()
    record = {'schema': 'rtl_extract.model_build.v1', 'commands': commands,
              'tool_versions': {'verilator': version}, 'wall_seconds': time.monotonic() - start,
              'outputs': [{'path': relative(output, root), 'sha256': sha256(output)}]}
    write_json(output.parent / 'build.json', record)
    return record

def replay(args):
    helper = load_helper(args.script)
    root = args.build_root or args.work.parent.parent
    summary = args.summary or args.work / 'summary.json'
    start = time.monotonic()
    # run returns only after numerical, waveform, and export binding checks pass.
    results = helper.run(args)
    if len(results) != args.expected_pass:
        raise ValueError(f'replay PASS count: expected {args.expected_pass}, got {len(results)}')
    if len({result['case'] for result in results}) != len(results):
        raise ValueError('duplicate replay PASS case')
    record = {'schema': 'rtl_extract.replay.v1', 'pass_count': len(results),
              'expected_pass': args.expected_pass, 'results': results,
              'wall_seconds': time.monotonic() - start, 'outputs': inventory(args.work, root, [summary])}
    write_json(summary, record)
    return record

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('model', 'replay'):
        command = sub.add_parser(name)
        command.add_argument('--script', type=Path, required=True)
        command.add_argument('--build-root', type=Path)
        command.add_argument('--jobs', type=int, default=8)
        command.add_argument('--timeout', type=int, default=1800)
    for flag in ('rtl', 'output'):
        sub.choices['model'].add_argument('--' + flag, type=Path, required=True)
    sub.choices['model'].add_argument('--verilator', required=True)
    sub.choices['model'].add_argument('--verilator-root', type=Path)
    for flag in ('facts', 'bin', 'work', 'model', 'pairs', 'summary'):
        sub.choices['replay'].add_argument('--' + flag, type=Path, required=flag not in ('pairs', 'summary'))
    sub.choices['replay'].add_argument('--expected-pass', type=int, required=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    try:
        for value in vars(args).values():
            if isinstance(value, Path): safe_path(value)
        if args.stage == 'model':
            safe_path(shutil.which(args.verilator) or args.verilator)
        else:
            args.facts_sha256, args.cases, args.consumer = None, 'all', 'both'
            for tool in ('atlas-opt', 'atlas-emit'): safe_path(args.bin / tool)
        return model(args) if args.stage == 'model' else replay(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'FAIL: {error}\n')

if __name__ == '__main__':
    main()
