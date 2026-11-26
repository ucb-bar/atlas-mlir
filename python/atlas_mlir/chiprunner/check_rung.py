#!/usr/bin/env python3
"""Read-only replay checker for inventoried kits and recorded transport receipts."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys

if __package__:
    from . import kit_manifest as kits
    from .preflight import openocd_builds
else:
    import kit_manifest as kits
    from preflight import openocd_builds

OPENOCD_PATH = os.environ.get('NOV_OPENOCD', '')
BYTE_SEMANTICS = ('requested logical payload bytes; observed_bytes records dump file size; '
                  'ELF PT_LOAD file bytes exclude zero-fill and metadata; not USB/JTAG wire bytes')
SUMMARY_KEYS = {'verdict', 'started_utc', 'wall_seconds', 'operations', 'transfer_totals',
                'byte_semantics', 'error', 'cleanup_error', 'bundle', 'elf', 'baud',
                'timeout_seconds', 'core_clock_hz', 'clock_source'}
safe_path = kits.safe_path
read_bytes = kits.read_bytes
sha = kits.sha


class Checker:
    def __init__(self):
        self.checks = []

    def add(self, status, name, lines, detail):
        self.checks.append((status, name, lines, detail))

    def test(self, condition, name, lines, detail):
        self.add('PASS' if condition else 'FAIL', name, lines, detail)

    def read(self, path, name, lines, kind='text'):
        try:
            data = read_bytes(path)
            if kind == 'bytes':
                return data
            value = data.decode('utf-8')
            if kind == 'json':
                value = json.loads(value)
                if not isinstance(value, (dict, list)):
                    raise ValueError('expected JSON object or array')
            return value
        except FileNotFoundError:
            self.add('INCOMPLETE', name, lines, 'required evidence missing')
        except (OSError, ValueError, UnicodeError) as error:
            self.add('FAIL', name, lines, 'unreadable or malformed evidence: ' + str(error))
        return None

    @property
    def verdict(self):
        statuses = [x[0] for x in self.checks]
        return 'FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses else 'PASS'


def check(capture, logs=None, *, bundle, negative=False, openocd_path=OPENOCD_PATH,
          run_stamp=None, openocd_banner=None):
    c = Checker()
    bundle = Path(bundle)
    try:
        loaded = kits.load(bundle, 'negative/chip.elf' if negative else 'chip.elf')
        bundle = kits.safe_path(bundle)
        c.add('PASS', 'fresh kit preflight', '63,76,80', 'inventory, binding, ELF, and references revalidated')
    except FileNotFoundError as error:
        c.add('INCOMPLETE', 'fresh kit preflight', '63,76,80', str(error))
        return c
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        c.add('FAIL', 'fresh kit preflight', '63,76,80', str(error))
        return c
    negative_hashes = {(x['row']['id'], x['row']['role']): x['sha256'] for x in loaded['negative']}
    changed = [x['row']['id'] for x,y in zip(loaded['references'], loaded['negative']) if x['data'] != y['data']]
    negative_reason = ('UART hash mismatch: ' + changed[0] + '; ' + '; '.join('dump differs from frozen reference: ' + x for x in changed)) if negative else None
    terminal = c.read(capture, 'terminal capture', '52,63,65')
    expected_rc = 1 if negative else 0
    expected_verdict = 'FAIL' if negative else 'PASS'
    if terminal is not None:
        lines = terminal.splitlines()
        expected_kit_line = ('KIT FAIL ' + negative_reason if negative else
                             'KIT PASS UART and dumped bytes match frozen reference')
        exact = lines == [expected_kit_line, 'rc=%d' % expected_rc]
        c.add('PASS' if exact else 'INCOMPLETE', 'terminal capture shape', '52,63,65,74',
              'expected KIT line and rc=%d' % expected_rc if exact else
              'unexpected terminal output; human review')
        if re.search(r'KIT TRANSPORT_ERROR|Traceback|KeyboardInterrupt', terminal):
            c.add('FAIL', 'transport/interruption', '74,76,80', 'stop; preserve evidence, no automatic retry')
    manifest = c.read(bundle / 'manifest.json', 'frozen manifest', '63,76', 'json')
    inventory = c.read(bundle / 'SHA256SUMS', 'reference inventory', '7,80')
    if manifest is not None and inventory is not None:
        listed = {}
        for line in inventory.splitlines():
            if '  ' in line:
                digest, name = line.split('  ', 1)
                listed[name] = digest
        c.test(sha(read_bytes(bundle / 'manifest.json')) == listed.get('manifest.json'),
               'manifest inventory', '7,80', 'manifest matches retained kit inventory')
    else:
        listed = {}
    if logs is None:
        c.add('INCOMPLETE', 'run logs', '63,65,80', 'supply --logs to verify UART, dumps, and receipts')
        return c
    try:
        logs = safe_path(logs)
    except ValueError as error:
        c.add('FAIL', 'logs path', '80', str(error))
        return c
    uart = c.read(logs / 'uart.log', 'UART', '63,65,76')
    console = c.read(logs / 'console-check.txt', 'console checker receipt', '63,74')
    if console is not None:
        expected = ['CONSOLE FAIL'] + ['hash mismatch or missing ' + x for x in changed] if negative else ['CONSOLE PASS']
        c.test(console.splitlines() == expected, 'console checker result', '63,74', 'expected ' + repr(expected))
    if manifest is None:
        return c
    try:
        hashes = {(r['name'], r['role']): r['sha256'] for r in manifest['expected_hashes']}
        ranges = manifest['deployment']['hash_ranges']
        if uart is not None:
            tag = loaded['settings']['record_tag']
            normalized = uart.replace('\r\n', '\n')
            records = normalized.splitlines()
            statuses = [x for x in records if x.startswith(tag + ' STATUS')]
            completes = [x for x in records if x.startswith(tag + ' COMPLETE')]
            hash_lines = [x for x in records if x.startswith(tag + ' HASH')]
            c.test(len(statuses) == 1 and bool(re.fullmatch(re.escape(tag) + r' STATUS 0 CYCLES [0-9]+', statuses[0])),
                   'successful status', '63,65,74', 'exactly one status zero with decimal cycle count')
            c.test(completes == [tag + ' COMPLETE'], 'exact completion', '63,65,74', 'exactly one completion record')
            expected_hashes = dict(hashes)
            if negative:
                expected_hashes.update(negative_hashes)
            wanted = ['%s HASH %s %s %s' % (tag, name, role, digest)
                      for (name, role), digest in expected_hashes.items()]
            c.test(sorted(hash_lines) == sorted(wanted), 'UART hashes', '63,65,68,76', 'exact expected name/role/digest records, no duplicates')
            for line in records:
                match = re.fullmatch(re.escape(tag) + r' HASH (\S+) (\S+) ([0-9a-f]{64})', line)
                mismatch = match and hashes.get((match[1], match[2])) != match[3]
                predicted = negative and match and negative_hashes.get((match[1], match[2])) == match[3]
                if re.match('^' + re.escape(tag) + r' STATUS -?[1-9][0-9]* CYCLES ', line) or (mismatch and not predicted):
                    c.add('FAIL', 'numerical evidence', '76', line + '; takes precedence over transport faults')
            allowed = statuses + completes + hash_lines
            extra = [x for x in records if x not in allowed and
                     not re.fullmatch(r'WARNING: address 0x[0-9a-f]+ is not aligned to the request size of 64', x)]
            c.test(not extra and normalized.endswith('\n'), 'UART framing', '63,65,78', 'no unexplained bytes or unterminated record')
            if len(statuses) == len(completes) == 1 and len(hash_lines) == len(wanted):
                c.test(records.index(statuses[0]) < min(records.index(x) for x in hash_lines) and
                       max(records.index(x) for x in hash_lines) < records.index(completes[0]),
                       'record order', '63,65', 'STATUS, HASH, COMPLETE order from kit fixtures')
        for i, row in enumerate(ranges):
            name = str(loaded['references'][i]['path'].relative_to(bundle))
            reference = c.read(loaded['references'][i]['path'], 'frozen reference', '63', 'bytes')
            dump = c.read(logs / ('output-%d.bin' % i), 'output dump %d' % i, '63,70,71,74', 'bytes')
            if reference is not None:
                c.test(len(reference) == row['bytes'] and sha(reference) == hashes[(row['id'], row['role'])] and
                       sha(reference) == listed.get(name), 'reference integrity %d' % i, '63,76', 'reference bytes match manifest and inventory')
            if dump is not None:
                c.test(len(dump) == row['bytes'], 'dump size %d' % i, '63,74', 'expected %d bytes; got %d' % (row['bytes'], len(dump)))
                if negative:
                    name = str(loaded['negative'][i]['path'].relative_to(bundle))
                    predicted = c.read(loaded['negative'][i]['path'], 'negative frozen evidence', '70,71,74', 'bytes')
                    if predicted is not None:
                        c.test(sha(predicted) == negative_hashes[(row['id'], row['role'])] and sha(predicted) == listed.get(name) and dump == predicted,
                               'predicted negative bytes', '65,70,71,74', 'exact frozen negative bytes and predicted SHA256; differs from positive')
                elif reference is not None:
                    c.test(dump == reference, 'dump bytes %d' % i, '63', 'every published byte equals frozen reference')
        receipts(c, logs, bundle, manifest, expected_verdict, openocd_path, run_stamp, loaded, negative_reason, openocd_banner)
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError) as error:
        c.add('FAIL', 'evidence structure', '63,80,119', str(error))
    return c


def receipts(c, logs, bundle, manifest, verdict, openocd_path, run_stamp, loaded, negative_reason, openocd_banner):
    summary = c.read(logs / 'summary.json', 'summary', '80,119', 'json')
    transport = c.read(logs / 'transport.jsonl', 'transport events', '80,119')
    human = c.read(logs / 'run.log', 'human transfer log', '80,119')
    openocd_log = c.read(logs / 'openocd.log', 'OpenOCD log', '27,80')
    if openocd_log is not None:
        # The banner of a declared build; an explicit expected banner must also be declared.
        banners = [row['banner'] for row in openocd_builds() if openocd_banner in (None, row['banner'])]
        c.test(any(banner in openocd_log.splitlines() for banner in banners), 'OpenOCD version banner', '27,37',
               'requires the exact banner line of a declared build: ' + ' or '.join(banners or [str(openocd_banner)]))
    command = c.read(logs / 'openocd-command.json', 'OpenOCD command receipt', '27,37,80', 'json')
    if command is not None:
        c.test(isinstance(command, list) and len(command) > 1 and command[0] == openocd_path,
               'explicit OpenOCD path', '27,30,37', 'expected recorded executable: ' + openocd_path)
        c.test(isinstance(command, list) and len(command) == 11 and
               command[1:8] == ['-c', 'gdb_port disabled', '-c', 'telnet_port disabled', '-c', 'tcl_port 29666', '-f'] and
               isinstance(command[8], str) and command[8].endswith('/' + bundle.name + '/' + loaded['paths']['config']) and
               command[9:] == ['-c', 'adapter speed 10000'],
               'OpenOCD configuration receipt', '56,58,60', 'expected disabled ports, Tcl port, bound cfg and adapter speed')
    if summary is None or transport is None:
        return
    c.test(isinstance(summary, dict) and set(summary) == SUMMARY_KEYS,
           'summary schema', '63,80,119', 'exact 14-key run_kit/transport_log summary')
    if not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS:
        return
    match = re.fullmatch(r'.+-(\d{8}T\d{6}Z)', logs.name)
    stamp = match[1] if match else None
    valid_start = False
    try:
        stamp_time = datetime.strptime(stamp, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
        started = datetime.fromisoformat(summary['started_utc'].replace('Z', '+00:00'))
        valid_start = started.tzinfo is not None and started >= stamp_time
    except (ValueError, TypeError, AttributeError):
        pass
    c.test(bool(match) and valid_start and (run_stamp is None or run_stamp == stamp),
           'run directory and start time', '7,15,56,58,60',
           'path=%s stamp=%s started_utc=%s expected_stamp=%s' %
           (logs, stamp, summary['started_utc'], run_stamp))
    events = [json.loads(line) for line in transport.splitlines()]
    elf = str(loaded['elf'].relative_to(bundle))
    c.test(summary['verdict'] == verdict and summary['elf'] == elf and
           Path(summary['bundle']).name == bundle.name and summary['baud'] == loaded['settings']['baud'] and
           summary['core_clock_hz'] == loaded['settings']['core_clock_hz'] and
           summary['clock_source'] == loaded['settings']['clock_source'] and
           summary.get('cleanup_error') is None,
           'summary identity and verdict', '7,63,74,80', 'rung, baud, supplied clock and verdict agree')
    c.test(events == summary['operations'], 'receipt agreement', '80,119', 'JSONL equals summary operations')
    c.test(isinstance(summary['wall_seconds'], (int, float)) and
           math.isfinite(summary['wall_seconds']) and summary['wall_seconds'] >= 0 and
           bool(summary['started_utc']) and
           summary['byte_semantics'] == BYTE_SEMANTICS and
           isinstance(summary['timeout_seconds'], (int, float)) and
           math.isfinite(summary['timeout_seconds']) and summary['timeout_seconds'] == loaded['settings']['timeout'] and
           summary.get('error') == negative_reason,
           'summary timing and semantics', '63,119', 'finite duration, bound timeout, timestamp, exact logical payload semantics and expected error')
    elf_bytes = loaded['elf_bytes']
    window_bytes = len(kits.read_bytes(loaded['window']))
    counts = [None, window_bytes, elf_bytes, elf_bytes, None, None] + [r['bytes'] for r in manifest['deployment']['hash_ranges']]
    ops = ['reset', 'load_image', 'load_image', 'verify_image', 'resume', 'halt'] + ['dump_image'] * len(manifest['deployment']['hash_ranges'])
    patch = loaded.get('negative_patch')
    if patch:
        counts[4:4] = [1, 1]
        ops[4:4] = ['dump_image', 'load_image']
        c.test(c.read(logs/'negative-before.bin', 'negative patch preimage', '74', 'bytes') == bytes([patch['before_byte']])
               and c.read(logs/'negative-patch.bin', 'negative patch byte', '74', 'bytes') == bytes([patch['after_byte']]),
               'qualified mutation bytes', '74', 'target preimage and applied byte equal bound qualification')
    c.test([(e['operation'], e['bytes']) for e in events] == list(zip(ops, counts)),
           'transfer operations and counts', '63,119', 'single execution; window/ELF loads, verify, halt and all dumps')
    valid = True
    for e in events:
        seconds = e['wall_seconds']
        valid = valid and e['success'] is True and isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0 and bool(e['started_utc'])
        if e['bytes'] is not None:
            count = e.get('observed_bytes', e['bytes'])
            rate = e['bytes_per_second']
            valid = valid and seconds > 0 and isinstance(rate, (int, float)) and math.isfinite(rate) and math.isclose(rate * seconds, count, rel_tol=1e-8, abs_tol=1e-6)
            if e['operation'] == 'dump_image':
                valid = valid and count == e['bytes']
        if human is not None:
            valid = valid and ('> ' + e['command'] + '\n') in human and json.dumps(e, sort_keys=True) in human
    c.test(valid, 'transfer success and timing', '80,119', 'successful RPCs, full observed dumps, finite payload rates and retained human records')
    commands = []
    for e in events:
        commands.append([a or b or d for a, b, d in re.findall(r'\{([^{}]*)\}|"([^"]*)"|(\S+)', e['command'])])
    command_valid = len(commands) == len(ops)
    if command_valid:
        expected_commands = [
            ['reset', 'halt'],
            ['load_image', str(loaded['window']), hex(manifest['deployment']['window']['address']), 'bin'],
            ['load_image', str(loaded['elf'])], ['verify_image', str(loaded['elf'])]]
        if patch:
            expected_commands += [
                ['dump_image', str(logs/'negative-before.bin'), hex(patch['address']), '1'],
                ['load_image', str(logs/'negative-patch.bin'), hex(patch['address']), 'bin']]
        expected_commands += [['resume', hex(manifest['bench']['entry_address'])], ['halt']]
        expected_commands += [['dump_image', str(logs/('output-%d.bin' % i)), hex(row['address']), str(row['bytes'])]
                              for i, row in enumerate(manifest['deployment']['hash_ranges'])]
        command_valid = commands == expected_commands
    c.test(command_valid, 'recorded target commands', '63,119', 'manifest entry, selected image and published output addresses agree')
    totals = {}
    for e in events:
        if e['bytes'] is not None:
            row = totals.setdefault(e['operation'], dict(attempted_bytes=0, completed_bytes=0, wall_seconds=0))
            row['attempted_bytes'] += e['bytes']
            row['completed_bytes'] += e.get('observed_bytes', e['bytes']) if e['success'] else 0
            row['wall_seconds'] += e['wall_seconds']
    c.test(summary['transfer_totals'] == totals, 'transfer totals', '119', 'totals recomputed from retained per-command receipts')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['kit'], help='kept for existing command lines; every check is generic')
    parser.add_argument('capture', nargs='?', help='terminal capture ending in rc=<decimal>')
    parser.add_argument('--dry-run', action='store_true',
                        help='authenticate bundle and physical fit only; no execution evidence or transport')
    parser.add_argument('--bundle', required=True, help='inventoried kit directory')
    parser.add_argument('--negative', action='store_true', help='replay the declared output-changing negative')
    parser.add_argument('--openocd-banner', help='expected recorded banner of a declared build (default: any declared build); executable is never opened')
    parser.add_argument('--logs', help='existing run_kit.py log directory (never created by checker)')
    parser.add_argument('--openocd-path', default=OPENOCD_PATH,
                        help='expected recorded November executable path (never opened)')
    parser.add_argument('--run-stamp', help='expected PREBENCH RUN_STAMP, YYYYMMDDTHHMMSSZ')
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.capture is not None or args.logs is not None or args.run_stamp is not None:
            parser.error('--dry-run does not accept execution evidence')
        try:
            kits.load(args.bundle, 'negative/chip.elf' if args.negative else 'chip.elf')
        except (ValueError, OSError, TypeError, KeyError, IndexError, OverflowError) as error:
            print('KIT CHECKER DRY FAIL ' + str(error))
            return 1
        print('KIT CHECKER DRY PASS authenticated preflight; no execution evidence checked')
        return 0
    if args.capture is None:
        parser.error('execution check requires capture')
    try:
        result = check(args.capture, args.logs, bundle=args.bundle, negative=args.negative,
                       openocd_path=args.openocd_path, run_stamp=args.run_stamp,
                       openocd_banner=args.openocd_banner)
    except (ValueError, OSError, TypeError) as error:
        result = Checker()
        result.add('FAIL', 'input access', '80', str(error))
    print(result.verdict)
    for status, name, lines, detail in result.checks:
        print('%s %s [PREBENCH.md:%s] %s' % (status, name, lines, detail))
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result.verdict]


if __name__ == '__main__':
    sys.exit(main())
