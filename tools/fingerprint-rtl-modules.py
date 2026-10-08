#!/usr/bin/env python3
"""Print review candidates for versioned retained-HW module compatibility.

Reads explicitly selected hardware and follows the selected root modules'
instance closure. Normalization removes only horizontal whitespace plus
loc(#locN) references outside strings. Other bytes, including newlines and
semantic attributes, are retained. Output never updates compiler trust.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat


def prohibited(path):
    return any('hammer' in part.lower() or 'vlsi' in part.lower()
               for part in path.parts)


def checked_path(value):
    """Check each symlink's spelling before traversing its target."""
    path = Path(value)
    if prohibited(path):
        raise ValueError('Prohibited path component')
    if not path.is_absolute():
        path = Path.cwd() / path
    if prohibited(path):
        raise ValueError('Prohibited path component')
    pending = list(path.parts[1:])
    resolved, links = Path(path.anchor), 0
    while pending:
        part = pending.pop(0)
        if part == '.':
            continue
        if part == '..':
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if prohibited(candidate):
            raise ValueError('Prohibited path component')
        if stat.S_ISLNK(candidate.lstat().st_mode):
            links += 1
            if links > 64:
                raise ValueError('Symlink cycle')
            target = Path(os.readlink(candidate))
            if prohibited(target):
                raise ValueError('Prohibited symlink target')
            if target.is_absolute():
                resolved = Path(target.anchor)
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        else:
            resolved = candidate
    return resolved


def normalize(text):
    # Quoted strings are first in the alternation and are copied unchanged.
    return re.sub(r'"(?:\\.|[^"\\])*"|[ \t]+loc\(#loc[0-9]+\)',
                  lambda match: match[0] if match[0].startswith('"') else '',
                  text)


def fingerprints(hardware, roots):
    indexer = checked_path(Path(__file__).parent / 'index-retained-hw.py')
    spec = importlib.util.spec_from_file_location('atlas_hw_index', indexer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = hardware.splitlines(keepends=True)
    plain = [line.rstrip('\n') for line in lines]
    table = module.module_table(plain)
    selected = {}

    def visit(name, active):
        if name in active:
            raise ValueError('Recursive module instantiation')
        if name in selected:
            return
        if name not in table:
            raise ValueError('Missing selected module: ' + name)
        record = module.read_module(table[name], plain)
        raw = ''.join(lines[record['_start']:record['_end'] + 1])
        normalized = normalize(raw)
        if re.search(r'\bloc\(', normalized):
            raise ValueError('Unsupported inline/nonalias location: ' + name)
        selected[name] = hashlib.sha256(normalized.encode()).hexdigest()
        for instance in record['instances']:
            visit(instance['module'], active | {name})

    for root in roots:
        visit(root, set())
    return dict(sorted(selected.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-ir', required=True)
    parser.add_argument('--root', action='append', default=[])
    parser.add_argument('--format', choices=('json', 'cpp'), default='json')
    args = parser.parse_args()
    with checked_path(args.hardware_ir).open('r', newline='') as source:
        selected = fingerprints(source.read(), args.root or ['AtlasCore'])
    if args.format == 'cpp':
        print('// Reviewed normalized module closure for atlas.vls.conservative.v1.')
        print('// Normalizer: horizontal whitespace plus loc(#locN), outside strings.')
        print('// Regeneration prints candidates; changing trust requires source review.')
        for name, digest in selected.items():
            print('{' + json.dumps(name) + ', ' + json.dumps(digest) + '},')
    else:
        print(json.dumps({'normalizer': 'atlas.module.locations_only.v1',
                          'roots': args.root or ['AtlasCore'], 'modules': selected},
                         indent=2))


if __name__ == '__main__':
    main()
