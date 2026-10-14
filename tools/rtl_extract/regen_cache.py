"""Content-addressed stage execution and build-relative provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .facts import sha256 as file_sha256

def safe_path(path):
    path = Path(os.path.abspath(path))
    pending, resolved, links = list(path.parts[1:]), Path(path.anchor), 0
    while pending:
        part = pending.pop(0)
        if any(word in part.lower() for word in ('hammer', 'vlsi')):
            raise ValueError('restricted path')
        if part == '..':
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise ValueError('symlink loop')
            target = Path(os.readlink(candidate))
            if any(any(word in p.lower() for word in ('hammer', 'vlsi')) for p in target.parts):
                raise ValueError('restricted symlink target')
            if target.is_absolute():
                resolved = Path(target.anchor)
            pending = list(target.parts[1:] if target.is_absolute() else target.parts) + pending
        else:
            resolved = candidate
    return path

def source_files(directory, suffixes=None):
    """Walk only a selected source subtree, pruning restricted names before descent."""
    directory = safe_path(directory)
    result = []
    for root, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not any(x in d.lower() for x in ('hammer', 'vlsi'))
                         and d not in ('__pycache__', '.git') and not (Path(root) / d).is_symlink())
        for name in sorted(names):
            if any(x in name.lower() for x in ('hammer', 'vlsi')):
                continue
            p = safe_path(Path(root) / name)
            if suffixes is None or p.suffix in suffixes:
                result.append(p)
    return result

def sha256(path):
    return file_sha256(safe_path(path))

def stage_key(commands, inputs, versions):
    payload = {'commands': commands, 'inputs': inputs, 'versions': versions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

class StageRunner:
    def __init__(self, build_root, cwd, environment=None):
        self.root = safe_path(build_root)
        self.cwd = safe_path(cwd)
        self.env = environment or os.environ.copy()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = safe_path(self.root / 'state.json')
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.records = []

    def relative(self, path):
        return os.path.relpath(safe_path(path), self.root)

    def describe(self, paths):
        return [{'path': self.relative(path), 'sha256': sha256(path)}
                for path in sorted(set(map(safe_path, paths)), key=str)]

    def command(self, argv):
        # Keep flags intact; replace absolute path tokens (including --flag=/path).
        result = []
        for arg in map(str, argv):
            head, sep, tail = arg.partition('=')
            if arg.startswith('/'):
                arg = self.relative(arg)
            elif sep and tail.startswith('/'):
                arg = head + '=' + self.relative(tail)
            result.append(arg)
        return result

    def output_path(self, path):
        path = safe_path(path)
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('stage output escapes build root')
        return path

    def run(self, name, commands, inputs, outputs, versions, prepare=None,
            output_inventory=None):
        start = time.monotonic()
        inputs = self.describe(inputs)
        normalized = [self.command(cmd) for cmd in commands]
        key = stage_key(normalized, inputs, versions)
        previous = self.state.get(name, {})
        output_paths = list(map(self.output_path, outputs))
        if output_inventory and safe_path(output_inventory).is_file():
            try:
                output_paths += [self.output_path(self.root / item['path'])
                                 for item in json.loads(Path(output_inventory).read_text())['outputs']]
            except (ValueError, TypeError, KeyError):
                previous = {}  # A damaged receipt is a cache miss.
        current_outputs = self.describe(output_paths) if all(p.is_file() for p in output_paths) else None
        record = {'stage': name, 'key': key, 'commands': normalized, 'cwd': self.relative(self.cwd),
                  'inputs': inputs, 'tool_versions': versions}
        if previous.get('key') == key and current_outputs == previous.get('outputs'):
            record.update(status='skipped', outputs=current_outputs,
                          wall_seconds=time.monotonic() - start,
                          produced_wall_seconds=previous.get('wall_seconds'))
            self.records.append(record)
            print(f'SKIP {name}', flush=True)
            return record
        print(f'RUN  {name}', flush=True)
        log = safe_path(self.root / 'logs' / f'{name}.log')
        log.parent.mkdir(exist_ok=True)
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if prepare:
                prepare()
            with log.open('w') as stream:
                for command in commands:
                    stream.write(json.dumps(list(map(str, command))) + '\n')
                    stream.flush()
                    subprocess.run(list(map(str, command)), cwd=self.cwd, env=self.env,
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
            if output_inventory:
                output_paths = list(map(self.output_path, outputs)) + [self.output_path(self.root / item['path'])
                    for item in json.loads(Path(output_inventory).read_text())['outputs']]
            record.update(status='produced', outputs=self.describe(output_paths),
                          log=self.relative(log), wall_seconds=time.monotonic() - start)
            self.state[name] = record
            write_json(self.state_path, self.state)
        except Exception:
            record.update(status='failed', log=self.relative(log), wall_seconds=time.monotonic() - start)
            self.records.append(record)
            raise
        self.records.append(record)
        return record
