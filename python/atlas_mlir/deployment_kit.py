"""Package bound on-chip artifacts and authenticate the runner's dry contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from .deploy import Gate, Refusal, digest, safe_path
from .deployment_link import RECORD_TAG

# Board run budget in chip cycles (120 s at 100 MHz). A kit states it in seconds at its carrier's declared clock; a
# runner that measures the board clock converts the cycles. The link clock adds chip cycles but never enters the budget.
BOARD_TIMEOUT_CYCLES = 12_000_000_000


def owned_tools():
    """Return the installed runner sources used by every packaged kit."""
    root = safe_path(Path(__file__).with_name('chiprunner'))
    tools = {path.name: _member(root, path.name) for path in root.glob('*.py')}
    from atlas_mlir_native.soc import DECLARATION
    tools['soc.json'] = DECLARATION
    for name in ('atlas-soc.cfg', 'spike_atlas.cc', 'openocd-builds.json'):
        tools[name] = _member(root, name)
    if not {'run_kit.py', 'check_rung.py', 'kit_manifest.py', 'preflight.py', 'capture_run.py'} <= tools.keys():
        raise Refusal('installed runner tools are incomplete')
    return tools


_RUN_SH = '''#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/bench/capture_run.py" "$@"
'''

_README = '''# Generated deployment kit

This package contains positive and intentional negative ELF images, independent
positive references, and predicted negative outputs from the authenticated,
executed pre-link mutation. Offline checks are not silicon qualification.
Python 3.10 or newer on Linux is required (macOS is not bench-qualified yet); no pip
dependencies are needed.

Export BUNDLE as this directory's absolute path and NOV_OPENOCD as the absolute
path of an OpenOCD build declared in bench/openocd-builds.json (its sha256 and
exact version banner). No executable is bundled. Before every rung,
obtain the bench owner's explicit authorization, fresh reviewer GO, and exclusive bench
ownership; preserve supplies and stop after any unexpected result. Never retry
or recover automatically.

Authenticate and preflight immediately before each authorized attempt:

```sh
cd "$BUNDLE"
sha256sum -c SHA256SUMS
python3 bench/run_kit.py "$BUNDLE" --dry-run
python3 bench/check_rung.py kit --bundle "$BUNDLE" --dry-run
```

Use a fresh UTC RUN_STAMP (YYYYMMDDTHHMMSSZ) for every attempt. Positive commands:

```sh
sh run.sh "$BUNDLE" "$RUN_STAMP"
python3 bench/check_rung.py kit "runs/kit-$RUN_STAMP-terminal.txt" --bundle "$BUNDLE" --logs "runs/kit-$RUN_STAMP" --run-stamp "$RUN_STAMP" --openocd-path "$NOV_OPENOCD"
```

The intentional negative is a separate authorized rung with a fresh stamp.
First repeat inventory validation and both dry checks using
`--elf negative/chip.elf` for run_kit.py and `--negative` for check_rung.py:

```sh
sh run.sh "$BUNDLE" "$RUN_STAMP" negative
python3 bench/check_rung.py kit "runs/kit-$RUN_STAMP-terminal.txt" --bundle "$BUNDLE" --logs "runs/kit-$RUN_STAMP" --run-stamp "$RUN_STAMP" --openocd-path "$NOV_OPENOCD" --negative
```

Expected return codes are runner 0/checker 0 for positive, runner 1/checker 0 for
negative. Negative success requires status zero, complete UART framing and every
output byte equal to its frozen negative prediction. Timeout, missing completion,
or transport errors never qualify. The wrapper records stdout/stderr and rc in a
new terminal capture, refuses existing capture/log paths, and runs once without
tee. It is the one authorized attempt: it passes the runner's transport flag and
stops a runner that outlives its own bounded waits, recorded as a transport
error with rc 2. Run the checker on the same host with the same absolute BUNDLE.
Preserve the full capture and logs.
'''

_BOARD_README = '''# Board DDR kit

`chip.elf` is the whole application: host code, the bound inputs, and the resident weights, placed in the
board's DDR by its linker script. `bench/soc.json` is the SoC declaration it was placed against.
`memory-plan.json` describes the measured host exclusion and device window. `loader-manifest.json` describes
ELF loads and weight chunks with their hashes. Loading over SPI on Puffin is assumed: the loader must write every
PT_LOAD segment at its physical address, verify the chunks, clear zero-fill tails, and jump to the ELF entry.
When the declared board host ends a run on tohost (`board_program` in `bench/soc.json`), `main` returning (0 after
`ATLAS COMPLETE`, 1 after a nonzero STATUS) waits for the UART to drain and writes `(code << 1) | 1` to `tohost`;
a 0 code does not mean the hashes match.
`negative/chip.elf` is the expected-fail mutation of input zero. `reference/output_NNN.bin` and
`negative/negative-evidence/output_NNN.bin` are the bytes each must produce. `atlascc run OUT --simulate` loads
and runs both on the simulated board. `atlascc run OUT` refuses until the board's SPI loader exists, and the kit has
no bench runner.
'''


def _negative(kit, elf, deployment, gate, settings, record):
    """Apply the executed checker mutation to precisely the linked data backing."""
    from .deployment_link import _elf_record, read_only_parameters

    receipt = json.loads(gate.receipt.read_text())
    flip = receipt.get('byte_flip', {})
    window = (kit / 'window.bin').read_bytes()
    offset = flip.get('window_byte_offset', -1)
    inputs = [row for row in deployment.get('inputs', []) if row['index'] == 0]
    if len(inputs) != 1:
        raise Refusal('negative requires bound input zero')
    row = inputs[0]
    begin = row['address'] - deployment['window']['address']
    if (not isinstance(offset, int) or not begin <= offset < begin + row['bytes']
            or not 0 <= offset < len(window) or flip.get('xor_mask') != 128
            or flip.get('input_index') != 0 or flip.get('input_name') != row['id']
            or flip.get('input_byte_offset') != offset - begin
            or flip.get('address') != deployment['window']['address'] + offset):
        raise Refusal('negative mutation differs from bound input zero')
    changed = bytearray(window)
    changed[offset] ^= 128
    sha = lambda data: hashlib.sha256(data).hexdigest()
    if (flip.get('before_byte') != window[offset] or flip.get('after_byte') != changed[offset]
            or flip.get('window_before_sha256') != sha(window)
            or flip.get('window_after_sha256') != sha(changed)
            or flip.get('input_before_sha256') != sha(window[begin:begin + row['bytes']])
            or flip.get('input_after_sha256') != sha(changed[begin:begin + row['bytes']])):
        raise Refusal('negative mutation window authentication differs')
    raw = elf.read_bytes()
    try:
        shoff = struct.unpack_from('<Q', raw, 40)[0]
        shsize, count, names_index = struct.unpack_from('<HHH', raw, 58)
        if shsize != 64 or not count or names_index >= count or shoff + count * shsize > len(raw):
            raise Refusal('negative ELF section table is invalid')
        sections = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * shsize) for i in range(count)]
        names = sections[names_index]
        strings = raw[names[4]:names[4] + names[5]]
        windows = [sec for sec in sections if strings[sec[0]:].split(b'\0')[0] == b'.window']
        if len(windows) != 1:
            raise Refusal('negative ELF requires one window section')
        section = windows[0]
        phoff = struct.unpack_from('<Q', raw, 32)[0]
        phsize, phcount = struct.unpack_from('<HH', raw, 54)
        if phsize != 56 or phoff + phsize * phcount > len(raw):
            raise Refusal('negative ELF load table is invalid')
        backing = [struct.unpack_from('<IIQQQQQQ', raw, phoff + i * phsize)
                   for i in range(phcount)]
        backing = [load for load in backing if load[0] == 1
                   and load[3] == load[4] == section[3] and load[2] == section[4]
                   and load[5] >= section[5]]
        # Read-only parameters, when declared, are the window's tail in their own load; input zero precedes them.
        if (section[1] != 1 or not section[2] & 2 or len(backing) != 1
                or section[3] != deployment['window']['address'] or not begin + row['bytes'] <= section[5] <= len(window)
                or (section[5] < len(window) and not read_only_parameters(deployment))
                or raw[section[4]:section[4] + section[5]] != window[:section[5]]):
            raise Refusal('negative ELF window differs from bound bytes')
    except (IndexError, struct.error) as error:
        raise Refusal('negative ELF section table is invalid') from error
    patched = bytearray(raw)
    patched[section[4] + offset] ^= 128
    negative = kit / 'negative'
    (negative / 'negative-evidence').mkdir(parents=True)
    (negative / 'chip.elf').write_bytes(patched)
    metadata = _elf_record(negative / 'chip.elf', deployment, settings)
    if metadata['entry'] != record['entry'] or metadata['loads'] != record['loads']:
        raise Refusal('negative ELF placement differs from positive')
    members = {'negative/chip.elf', 'negative/manifest.json'}
    (negative / 'manifest.json').write_text(json.dumps(dict(sizes=dict(chip=dict(
        loads=metadata['loads'], sha256=metadata['sha256']))), indent=2) + '\n')
    outputs = receipt.get('negative', {}).get('outputs', [])
    expected_rows = deployment['outputs']
    if len(outputs) != len(expected_rows) or {x['index'] for x in outputs} != {x['index'] for x in expected_rows}:
        raise Refusal('negative predictions must cover every output exactly once')
    changed_output = False
    for expected in expected_rows:
        row = next(x for x in outputs if x['index'] == expected['index'])
        source = _member(gate.receipt.parent / 'exact/negative', row['path'])
        if gate.hashes.get(str(source)) != digest(source):
            raise Refusal('negative prediction is not authenticated by EXACT gate')
        data = source.read_bytes()
        if (len(data) != expected['bytes'] or row['bytes'] != len(data)
                or row['name'] != expected['id'] or row['address'] != expected['address']
                or row['sha256'] != sha(data)):
            raise Refusal('negative prediction differs from executed output')
        changed_output |= data != (kit / ('reference/output_%03d.bin' % expected['index'])).read_bytes()
        name = 'negative/negative-evidence/output_%03d.bin' % expected['index']
        (kit / name).write_bytes(data)
        members.add(name)
    if not changed_output:
        raise Refusal('negative predictions do not change any output')
    return metadata, members


def _member(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise Refusal('invalid kit inventory member')
    path = safe_path(root / relative)
    if not path.is_file():
        raise Refusal('kit member missing: ' + str(relative))
    return path


def _verify_tools(paths):
    sources = owned_tools()
    if paths.keys() != sources.keys():
        raise Refusal('runner offline contract changed: tool inventory')
    for name, source in sources.items():
        if digest(paths[name]) != digest(source):
            raise Refusal('runner offline contract changed: ' + name)


def dry_preflight(bundle, runner, checker=None):
    """Execute only authenticated dry entry points; never supply live options."""
    bundle, runner = safe_path(bundle), safe_path(runner)
    checker = safe_path(checker) if checker is not None else _member(runner.parent, 'check_rung.py')
    paths = {name: _member(runner.parent, name) for name in owned_tools()}
    if {path.name for path in runner.parent.iterdir()} != paths.keys():
        raise Refusal('runner offline contract changed: unexpected tool directory member')
    if paths['run_kit.py'] != runner or paths['check_rung.py'] != checker:
        raise Refusal('runner and checker must share the staged tool directory')
    _verify_tools(paths)
    # Validate every path before the external inventory loader sees it.
    listed = set()
    for line in _member(bundle, 'SHA256SUMS').read_text().splitlines():
        try:
            expected, name = line.split('  ', 1)
        except ValueError:
            raise Refusal('malformed kit inventory') from None
        if name in listed or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise Refusal('malformed or duplicate kit inventory entry')
        listed.add(name)
        if digest(_member(bundle, name)) != expected:
            raise Refusal('kit integrity mismatch: ' + name)
    required = {'manifest.json', 'chip.elf', 'window.bin', 'application/deployment.json',
                'negative/chip.elf', 'negative/manifest.json', 'run.sh'}
    required |= {'bench/' + name for name in paths}
    if not required <= listed:
        raise Refusal('kit inventory omits required artifacts')
    commands = (
        ('runner', [str(runner), str(bundle), '--dry-run'],
         'KIT DRY PASS authenticated preflight; no execution'),
        ('checker', [str(checker), 'kit', '--bundle', str(bundle), '--dry-run'],
         'KIT CHECKER DRY PASS authenticated preflight; no execution evidence checked'),
        ('runner-negative', [str(runner), str(bundle), '--dry-run', '--elf', 'negative/chip.elf'],
         'KIT DRY PASS authenticated preflight; no execution'),
        ('checker-negative', [str(checker), 'kit', '--bundle', str(bundle), '--dry-run', '--negative'],
         'KIT CHECKER DRY PASS authenticated preflight; no execution evidence checked'),
    )
    logs = {}
    for name, arguments, expected in commands:
        try:
            result = subprocess.run([sys.executable, '-E', '-s', '-B', *arguments],
                                    capture_output=True, text=True, timeout=60, cwd=bundle)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Refusal('kit ' + name + ' dry check failed: ' + str(error)) from error
        logs[name] = result.stdout + result.stderr
        if result.returncode or result.stdout.strip() != expected or result.stderr:
            raise Refusal('kit ' + name + ' dry check refused: ' + logs[name].strip())
    _verify_tools(paths)
    return logs


def package(context):
    """Publish a generalized on-chip kit after the bound EXACT gate and link."""
    import numpy as np

    gate = context.get('gate')
    if not isinstance(gate, Gate):
        raise Refusal('packaging requires the bound EXACT gate')
    gate.verify()
    bundle = safe_path(context['bundle'])
    elf = safe_path(context['elf'])
    record = context['elf_record']
    settings = context['settings']
    if digest(elf) != record['sha256'] or any(record.get(key) != settings[key] for key in (
            'clock_hz', 'receiver_baud')):
        raise Refusal('linked ELF identity or clock settings drift')
    # The bench runner and its JTAG rungs serve on-chip kits only; a board DDR kit keeps only the SoC declaration.
    jtag = context.get('transport') == 'jtag'
    tools = {name: _member(source.parent, source.name) for name, source in owned_tools().items()
             if jtag or name == 'soc.json'}
    kit = safe_path(context['out'] / 'kit')
    if kit.exists():
        raise Refusal('kit output must be fresh')
    kit.mkdir()
    copied = set()
    for source, expected in gate.hashes.items():
        source = safe_path(source)
        if source.is_relative_to(bundle):
            name = source.relative_to(bundle)
            if name.as_posix() == 'SHA256SUMS':
                continue
            if digest(source) != expected:
                raise Refusal('bound kit source drift')
            destination = kit / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if digest(destination) != expected:
                raise Refusal('bound kit copy drift')
            copied.add(name.as_posix())
    required = {'manifest.json', 'window.bin', 'reference/expected.npz', 'application/deployment.json'}
    if not required <= copied:
        raise Refusal('gate inventory omits kit inputs: ' + ', '.join(sorted(required - copied)))
    shutil.copyfile(kit / 'manifest.json', kit / 'gate-manifest.json')
    manifest = json.loads((kit / 'manifest.json').read_text())
    deployment = json.loads((kit / 'application/deployment.json').read_text())
    if manifest['deployment'] != deployment or 'paging' in deployment:
        raise Refusal('bound deployment mismatch or unsupported paging')
    expected = {(row['name'], row['role']): row for row in manifest['expected_hashes']}
    ranges = deployment['hash_ranges']
    keys = [(row['id'], row['role']) for row in ranges]
    if (not keys or len(set(keys)) != len(keys) or set(keys) != set(expected)
            or len(expected) != len(manifest['expected_hashes']) or ranges != deployment['outputs']):
        raise Refusal('expected hashes must cover every output exactly once')
    nonzero = False
    with np.load(kit / 'reference/expected.npz', allow_pickle=False) as archive:
        for row in ranges:
            data = archive['output_' + str(row['index'])].tobytes(order='C')
            frozen = expected[row['id'], row['role']]
            if (len(data) != row['bytes'] or len(data) != frozen['bytes']
                    or hashlib.sha256(data).hexdigest() != frozen['sha256']):
                raise Refusal('independent output bytes differ from bound hashes')
            nonzero |= any(data)
            (kit / 'reference' / ('output_%03d.bin' % row['index'])).write_bytes(data)
    if not nonzero:
        raise Refusal('all-zero output reference cannot qualify packaging')
    shutil.copyfile(elf, kit / 'chip.elf')
    if digest(kit / 'chip.elf') != record['sha256']:
        raise Refusal('linked ELF copy drift')
    negative_record, negative_members = _negative(kit, elf, deployment, gate, settings, record)
    manifest['bench'] = dict(entry_address=record['entry'])
    manifest['runner'] = dict(schema='atlas.kit.v1', record_tag=RECORD_TAG)
    manifest.setdefault('sizes', {})['chip'] = dict(loads=record['loads'], sha256=record['sha256'])
    manifest['bench_facts'] = dict(core_clock_hz=settings['clock_hz'], receiver_baud=settings['receiver_baud'],
                                 uart_divisor=settings['uart_divisor'], requested_uart_baud=settings['receiver_baud'],
                                 uart_divisor_basis='predicted from configured clock and requested baud; not observed',
                                 mtime_hz=settings['mtime_hz'], source=settings['clock_basis'],
                                 carriers=settings['carriers'])
    manifest['platform_assumptions'] = dict(timeout_cycles=BOARD_TIMEOUT_CYCLES,
                                            timeout_seconds=BOARD_TIMEOUT_CYCLES / settings['clock_hz'])
    (kit / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (kit / 'bench').mkdir(exist_ok=True)
    for name, source in tools.items():
        shutil.copyfile(source, kit / 'bench' / name)
    if jtag:
        (kit / 'run.sh').write_text(_RUN_SH)
    (kit / 'README.md').write_text(_README if jtag else _BOARD_README)
    (kit / 'qualification').mkdir(exist_ok=True)
    shutil.copyfile(gate.receipt, kit / 'qualification/check.json')
    result = dict(settings=settings, elf_sha256=record['sha256'],
                  negative_elf_sha256=negative_record['sha256'],
                  exact_receipt_sha256=digest(gate.receipt), dry_inventory=jtag,
                  dry_static_preflight=jtag, qualification='diagnostic', checker_dry_run=jtag,
                  transport_executed=False, scope='offline packaging; no silicon execution evidence')
    (kit / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    members = copied | {'gate-manifest.json', 'chip.elf', 'qualification/check.json', 'result.json'}
    members |= {'bench/' + name for name in tools}
    members |= negative_members | {'README.md'} | ({'run.sh'} if jtag else set())
    members |= {'reference/output_%03d.bin' % row['index'] for row in ranges}
    if not jtag:
        from atlas_mlir_native.soc import DECLARATION_SHA256
        from .deployment_loader import describe
        if digest(kit / 'bench/soc.json') != DECLARATION_SHA256:
            raise Refusal('kit SoC declaration differs from the declaration the deployment was placed against')
        plan, loader = describe(kit / 'chip.elf', deployment, record)
        for name, data in (('memory-plan.json', plan), ('loader-manifest.json', loader)):
            (kit / name).write_text(json.dumps(data, indent=2) + '\n')
            members.add(name)
    (kit / 'SHA256SUMS').write_text(''.join(digest(_member(kit, name)) + '  ' + name + '\n' for name in sorted(members)))
    logs = dry_preflight(kit, kit / 'bench/run_kit.py', kit / 'bench/check_rung.py') if jtag else {}
    for name, log in logs.items():
        (context['out'] / ('kit-' + name + '-dry.log')).write_text(log)
    gate.verify()
    if digest(elf) != record['sha256']:
        raise Refusal('linked ELF identity drift during packaging')
    context['kit'] = kit
    (context['out'] / 'kit-preflight.json').write_text(json.dumps(result, indent=2) + '\n')


def package_staged(context):
    """Publish a staged kit: the gated composed plan with each linked step ELF in place of its application.

    A blocked step keeps its generated step application, so the kit simulates but
    refuses live execution. Members are named explicitly; no directory is listed.
    """
    from .chiprunner import composed

    gate = context.get('gate')
    if not isinstance(gate, Gate):
        raise Refusal('packaging requires the bound EXACT gate')
    gate.verify()
    bundle, settings = safe_path(context['bundle']), context['settings']
    kit = safe_path(context['out'] / 'kit')
    if kit.exists():
        raise Refusal('kit output must be fresh')
    manifest = json.loads(_member(bundle, 'manifest.json').read_text())
    if gate.hashes.get(str(bundle / 'manifest.json')) != digest(bundle / 'manifest.json'):
        raise Refusal('composed plan is not authenticated by the EXACT gate')
    sources, added = {}, {}
    for region in manifest['composed']['regions']:
        package = Path(region['deployment']).parts[2]
        step = context['steps'][package]
        if 'elf' not in step:
            continue
        record = step['record']
        if digest(step['elf']) != record['sha256'] or record.get('clock_hz') != settings['clock_hz']:
            raise Refusal('linked step ELF identity or clock settings drift: ' + package)
        added['steps/%s/chip.elf' % package] = step['elf']
        added['steps/%s/link.json' % package] = context['out'] / 'steps' / package / 'link.json'
        region.update(kind='elf', elf='steps/%s/chip.elf' % package, entry=record['entry'])
        region.pop('sources')
    for region in manifest['composed']['regions']:
        names = [region['deployment'], *(image[key] for image in region['images'] for key in ('program', 'plan')),
                 *region.get('sources', []), *(row['file'] for row in region['weights'])]
        sources.update({name: bundle / name for name in names})
    for row in manifest['composed']['values']:
        sources.update({row[key]: bundle / row[key] for key in ('file', 'reference') if key in row})
    sources[manifest['composed']['negative']['file']] = bundle / manifest['composed']['negative']['file']
    for name, source in sources.items():
        if name in added or gate.hashes.get(str(source)) != digest(source):
            raise Refusal('kit member is not authenticated by the EXACT gate: ' + name)
    sources.update(added)
    sources['qualification/composed-check.json'] = gate.receipt
    sources.update({'bench/' + name: path for name, path in owned_tools().items()})
    kit.mkdir()
    for name, source in sources.items():
        target = safe_path(kit / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(source), target)
    blocked = context.get('blocked', {})
    manifest['composed']['receiver_baud'] = settings['receiver_baud']
    manifest['staged'].update(blocked=blocked, gate=dict(receipt='qualification/composed-check.json',
                                                         receipt_sha256=digest(gate.receipt)))
    (kit / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    names = sorted([*sources, 'manifest.json'])
    (kit / 'SHA256SUMS').write_text(''.join(digest(kit / name) + '  ' + name + '\n' for name in names))
    try:
        composed.load_plan(kit)
    except (ValueError, KeyError) as error:
        raise Refusal('staged kit plan refused: ' + str(error)) from error
    gate.verify()
    context['kit'] = kit
