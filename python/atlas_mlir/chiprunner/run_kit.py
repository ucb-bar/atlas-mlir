# Bring-up tooling: one attempt, published addresses, UART and byte verification.
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import math
if __package__:
    from .kit_manifest import load, console_errors, safe_path
else:
    from kit_manifest import load, console_errors, safe_path
if __package__:
    from .tcl_rpc import TclRPC, TransportError, tcl_path
else:
    from tcl_rpc import TclRPC, TransportError, tcl_path
if __package__:
    from .transport_log import TransferLog
else:
    from transport_log import TransferLog


class NumericalError(RuntimeError):
    pass


class UART:
    def __init__(self, path, baud):
        self.fd = os.open(path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            a = termios.tcgetattr(self.fd)
            a[0] = a[1] = a[3] = 0
            a[2] = termios.CS8 | termios.CSTOPB | termios.CREAD | termios.CLOCAL
            speed = getattr(termios, 'B%d' % baud, termios.B38400)
            a[4] = a[5] = speed
            termios.tcsetattr(self.fd, termios.TCSANOW, a)
            if not hasattr(termios, 'B%d' % baud):
                # Linux termios2 BOTHER supports the proven nonstandard receiver baud.
                import fcntl
                raw = bytearray(44)
                fcntl.ioctl(self.fd, 0x802c542a, raw)
                flags = struct.unpack_from('I', raw, 8)[0]
                struct.pack_into('I', raw, 8, (flags & ~0x100f) | 0x1000)
                struct.pack_into('II', raw, 36, baud, baud)
                fcntl.ioctl(self.fd, 0x402c542b, raw)
        except BaseException:
            os.close(self.fd)
            raise

    def clear(self):
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def records(self, timeout, console, tag):
        deadline, data, tag = time.monotonic() + timeout, b'', re.escape(tag.encode())
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.fd], [], [], min(.2, max(0, deadline-time.monotonic())))
            if not ready:
                continue
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise TransportError('UART disconnected')
            console.write(chunk); console.flush()
            data += chunk
            lines = data.replace(b'\r\n', b'\n').splitlines()
            if any(re.match(b'^' + tag + rb' STATUS -?[1-9][0-9]* CYCLES ', line) for line in lines):
                raise NumericalError('UART reported nonzero status')
            # Preserve all records, including malformed/duplicate ones, for the final checker.
            if any(re.match(tag + b' COMPLETE', line) for line in lines) and data.endswith(b'\n'):
                return data
        raise TransportError('UART completion timeout')

    def close(self):
        os.close(self.fd)


def numerical_evidence(manifest, uart_log, tag):
    if not uart_log.exists():
        return None
    lines, tag = uart_log.read_text(errors='replace').splitlines(), re.escape(tag)
    if any(re.match('^' + tag + r' STATUS -?[1-9][0-9]* CYCLES ', line) for line in lines):
        return 'UART reported nonzero status'
    expected = {(row['name'], row['role']): row['sha256'] for row in manifest['expected_hashes']}
    for line in lines:
        match = re.fullmatch(tag + r' HASH (\S+) (\S+) ([0-9a-f]{64})', line)
        if match and expected.get((match[1], match[2])) != match[3]:
            return 'UART hash mismatch: ' + match[1]
    return None


def check_console(kit, logs):
    manifest, tag = kit['manifest'], kit['settings']['record_tag']
    errors = console_errors(manifest, (logs/'uart.log').read_bytes(), tag)
    lines = ['CONSOLE FAIL' if errors else 'CONSOLE PASS']
    actual = (logs/'uart.log').read_text().splitlines()
    for row in manifest['expected_hashes']:
        if '%s HASH %s %s %s' % (tag,row['name'],row['role'],row['sha256']) not in actual:
            lines.append('hash mismatch or missing '+row['name'])
    (logs/'console-check.txt').write_text('\n'.join(lines)+'\n')
    return subprocess.CompletedProcess([], bool(errors))


def step_hashes(rows, records, tag):
    """Runtime digests of `rows` from one completed call's UART records; no frozen oracle applies."""
    records = records.replace('\r\n', '\n')
    hashes = {}
    expected = {(r['id'], r['role']) for r in rows}
    lines = records.splitlines()
    if (not records.endswith('\n') or len(lines) != len(rows)+2
            or not re.fullmatch(re.escape(tag) + r' STATUS 0 CYCLES [0-9]+', lines[0])
            or lines[-1] != tag + ' COMPLETE'):
        raise NumericalError('missing or malformed step completion')
    for line in lines[1:-1]:
        match = re.fullmatch(re.escape(tag) + r' HASH (\S+) (\S+) ([0-9a-f]{64})', line)
        if not match or (match[1], match[2]) not in expected or match[1] in hashes:
            raise NumericalError('malformed step output hash')
        hashes[match[1]] = match[3]
    if set(hashes) != {r['id'] for r in rows}:
        raise NumericalError('missing step output hash')
    return hashes


class Session:
    """One exclusive OpenOCD session holding one loaded window and ELF.

    `call` keeps the ELF and immutable parameters resident. It restores every
    other declared allocation from the packaged window, poisons the outputs,
    stages the inputs, restarts at the entry, and dumps every output.
    """

    def __init__(self, a, kit, logs):
        self.a, self.kit, self.logs = a, kit, logs
        self.process = self.uart = self.rpc = None
        self.running = False
        self.transport = TransferLog(logs, None)
        self.command = self.transport.command
        self.openocd_log = (logs/'openocd.log').open('w')

    def open(self):
        a, kit = self.a, self.kit
        self.uart = UART(a.serial, a.baud)
        self.process = launch_openocd(a, kit['config'], self.logs, self.openocd_log)
        self.rpc = connect(a, self.process)
        self.transport.session = self.rpc
        self.command('reset halt')
        self.uart.clear()
        window = kit['window']
        self.window = window.read_bytes()
        self.command('load_image %s %#x bin' % (tcl_path(window), kit['dep']['window']['address']), len(self.window))
        self.command('load_image '+tcl_path(kit['elf']), kit['elf_bytes'])
        self.command('verify_image '+tcl_path(kit['elf']), kit['elf_bytes'])

    def restart(self, logs):
        """Resume at the bound entry and wait for one completion record set."""
        self.uart.clear()
        self.running = True
        self.command(f'resume {self.kit["manifest"]["bench"]["entry_address"]:#x}')
        with (logs/'uart.log').open('wb') as console:
            self.uart.records(self.a.timeout, console, self.kit['settings']['record_tag'])
        self.command('halt')
        self.running = False

    def load(self, path, address, data):
        path.write_bytes(data)
        self.command('load_image %s %#x bin' % (tcl_path(path), address), len(data))

    def write(self, path, address, data):
        """Load `data`, then dump the range and require it byte for byte; the readback is kept only when it differs."""
        self.load(path, address, data)
        readback = path.with_name(path.stem + '-readback.bin')
        readback.unlink(missing_ok=True)
        self.command('dump_image %s %#x %d' % (tcl_path(readback), address, len(data)), len(data), output_path=readback)
        if readback.read_bytes() != data:
            raise TransportError(f'write readback differs at {address:#x}')
        readback.unlink()

    def call(self, logs, staged, references=None):
        """Execute one call; with `references`, every output must equal them."""
        kit, dep = self.kit, self.kit['dep']
        if not dep.get('allocations'):
            raise ValueError('resident calls require declared allocations')
        logs = safe_path(logs); logs.mkdir(exist_ok=False)
        window, base = self.window, dep['window']['address']
        outputs = {row['address'] for row in dep['outputs']}
        inputs = {row['address'] for row in dep.get('inputs', [])}
        for row in dep['allocations']:
            if row['role'] != 'immutable_parameters' and row['address'] not in outputs | inputs:
                offset = row['address'] - base
                data = window[offset:offset+row['bytes']].ljust(row['bytes'], b'\0')
                self.write(logs/('restore-%s.bin' % row['name']), row['address'], data)
        for row in dep['outputs']:
            self.write(logs/('poison-%d.bin' % row['index']), row['address'], b'\xa5'*row['bytes'])
        for item in staged:
            self.write(logs/('input-%d.bin' % item['row']['index']), item['row']['address'], item['data'])
        self.restart(logs)
        hashes = step_hashes(dep['outputs'], (logs/'uart.log').read_text(), kit['settings']['record_tag'])
        result = []
        for i, row in enumerate(dep['outputs']):
            path = logs/f'output-{i}.bin'
            self.command(f'dump_image {tcl_path(path)} {row["address"]:#x} {row["bytes"]}', row['bytes'], output_path=path)
            data = path.read_bytes()
            if len(data) != row['bytes']:
                raise TransportError('short output dump')
            if hashlib.sha256(data).hexdigest() != hashes[row['id']]:
                raise TransportError('dump differs from execution hash: '+row['id'])
            if data == b'\xa5'*row['bytes']:
                raise NumericalError('output remained poisoned: '+row['id'])
            if references is not None and data != references[i]['data']:
                raise NumericalError('resident reference call differs: '+row['id'])
            result.append(data)
        return result

    def close(self, verdict, failure=None):
        # Every cleanup step runs even if an earlier one raises; the summary records each failure.
        errors = []
        def attempt(step):
            try: step()
            except Exception as error: errors.append(str(error))
        def stop():
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: raise TransportError('OpenOCD did not exit after kill')
        if self.running and self.rpc is not None and not self.rpc.broken: attempt(lambda: self.command('halt'))
        for step in [self.rpc and self.rpc.close, self.uart and self.uart.close, self.process and stop, self.openocd_log.close]:
            if step: attempt(step)
        cleanup_error = '; '.join(errors) or None
        return self.transport.finish(verdict, error=failure, cleanup_error=cleanup_error,
            bundle=str(self.a.bundle), elf=self.a.elf, baud=self.a.baud,
            timeout_seconds=self.a.timeout, core_clock_hz=self.kit['manifest']['bench_facts']['core_clock_hz'],
            clock_source=self.kit['settings']['clock_source'])


def launch_openocd(a, config, logs, log):
    """Start the one OpenOCD process on the bound config; the caller owns and stops it."""
    cmd = [a.openocd, '-c', 'gdb_port disabled', '-c', 'telnet_port disabled', '-c',
           f'tcl_port {a.tcl_port}', '-f', str(config), '-c', 'adapter speed 10000']
    (logs/'openocd-command.json').write_text(json.dumps(cmd)+'\n')
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def connect(a, process):
    deadline = time.monotonic() + a.start_timeout
    while True:
        if process.poll() is not None:
            raise TransportError('OpenOCD exited during startup')
        try:
            return TclRPC(port=a.tcl_port, timeout=a.rpc_timeout)
        except OSError:
            if time.monotonic() >= deadline:
                raise TransportError('OpenOCD startup timeout')
            time.sleep(.3)


def prepare(a):
    """Fresh authenticated preflight and bound settings; opens no transport."""
    bundle = a.bundle = safe_path(a.bundle)
    kit = load(bundle, a.elf)
    for name, default in [('baud',kit['settings']['baud']), ('timeout',kit['settings']['timeout'])]:
        value = getattr(a,name)
        if value is not None and value != default:
            raise NumericalError(name+' differs from bound manifest')
        setattr(a,name,default)
    for name in ('timeout','rpc_timeout','start_timeout'):
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<=0:
            raise NumericalError('waits must be finite and positive')
    a.elf = str(kit['elf'].relative_to(bundle))
    return kit


def live(a, kit):
    if not getattr(a, 'allow_transport', False):
        raise ValueError('--allow-transport is required to open transport')
    if a.logs is None:
        raise NumericalError('--logs is required for execution')
    if __package__:
        from .preflight import live as check
    else:
        from preflight import live as check
    check(kit, a.openocd, a.serial)
    logs = safe_path(a.logs); logs.mkdir(parents=True, exist_ok=False)
    return logs


def execute(a):
    kit = prepare(a)
    manifest, dep = kit['manifest'], kit['dep']
    tag = kit['settings']['record_tag']
    if a.replay is not None:
        if not a.dry_run: raise NumericalError('--replay requires --dry-run')
        logs = safe_path(a.replay)
        errors = console_errors(manifest, (logs/'uart.log').read_bytes(), tag)
        for i, ref in enumerate(kit['references']):
            data = (logs/f'output-{i}.bin').read_bytes()
            if data != ref['data']: errors.append('dump differs from frozen reference: '+ref['row']['id'])
        if errors: raise NumericalError('; '.join(errors))
    if a.dry_run:
        print('KIT DRY PASS authenticated preflight; no execution')
        return
    logs = live(a, kit)
    session = Session(a, kit, logs)
    verdict, failure, check = 'TRANSPORT_ERROR', None, None
    try:
        session.open()
        command = session.command
        patch = kit.get('negative_patch')
        if patch:
            before_path = logs/'negative-before.bin'
            command('dump_image %s %#x 1' % (tcl_path(before_path), patch['address']), 1, output_path=before_path)
            if before_path.read_bytes() != bytes([patch['before_byte']]):
                raise TransportError('negative patch target preimage differs')
            session.load(logs/'negative-patch.bin', patch['address'], bytes([patch['after_byte']]))
        session.restart(logs)
        check = check_console(kit, logs)
        errors = []
        for i, row in enumerate(dep['hash_ranges']):
            path = logs/f'output-{i}.bin'
            command(f'dump_image {tcl_path(path)} {row["address"]:#x} {row["bytes"]}', row['bytes'], output_path=path)
            data = path.read_bytes()
            if len(data) != row['bytes']:
                raise TransportError('short output dump')
            expected = kit['references'][i]['data']
            frozen = next(x for x in manifest['expected_hashes'] if (x['name'],x['role']) == (row['id'],row['role']))
            if hashlib.sha256(expected).hexdigest() != frozen['sha256'] or data != expected:
                errors.append('dump differs from frozen reference: '+row['id'])
        if check.returncode or errors:
            evidence = numerical_evidence(manifest, logs/'uart.log', tag)
            raise NumericalError('; '.join(([evidence] if evidence else []) + errors) or 'UART records failed frozen reference check')
        verdict = 'PASS'
    except Exception as error:
        checked_before_error = check is not None
        if (logs/'uart.log').exists() and check is None:
            check = check_console(kit, logs)
        evidence = numerical_evidence(manifest, logs/'uart.log', tag)
        if not evidence and checked_before_error and check.returncode:
            evidence = 'UART records failed frozen reference check'
        if not isinstance(error, NumericalError) and evidence:
            failure = evidence + ' (then transport: ' + str(error) + ')'
            verdict = 'FAIL'
            raise NumericalError(failure) from error
        verdict = 'FAIL' if isinstance(error, NumericalError) else 'TRANSPORT_ERROR'
        failure = str(error)
        raise
    finally:
        session.close(verdict, failure)


def arguments(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--allow-transport', action='store_true', help='explicit authorization to open transport')
    p.add_argument('--composed', action='store_true', help='execute ordered regions offline')
    p.add_argument('--backend', choices=['sim'], default='sim')
    p.add_argument('--resume', action='store_true', help='resume completed composed checkpoints')
    p.add_argument('--recheck', action='store_true', help='recheck a finished composed run in --logs from its saved outputs')
    p.add_argument('--serial', default='/dev/serial/by-id/usb-FTDI_Dual_RS232-HS-if01-port0')
    p.add_argument('--baud', type=int, default=None)
    p.add_argument('--openocd', default=os.environ.get('NOV_OPENOCD', ''))
    p.add_argument('--tcl-port', type=int, default=29666)
    p.add_argument('--timeout', type=float, default=None, help='UART run deadline in seconds (default two minutes; measured cycles are about 51 ms at 100 MHz)')
    p.add_argument('--rpc-timeout', type=float, default=120)
    p.add_argument('--start-timeout', type=float, default=180)
    p.add_argument('--logs', type=Path, help='new attempt directory')
    p.add_argument('--replay', type=Path, help='read recorded UART and complete output dumps during dry run')
    p.add_argument('--dry-run', action='store_true', help='fresh authenticated preflight, without opening any transport')
    p.add_argument('--elf', choices=['chip.elf', 'negative/chip.elf'], default='chip.elf')
    return p.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    try:
        if a.composed:
            if __package__:
                from .composed import execute as execute_composed, load_plan, recheck, BoundaryError
            else:
                from composed import execute as execute_composed, load_plan, recheck, BoundaryError
            if a.dry_run:
                load_plan(a.bundle)
                print('KIT DRY PASS authenticated composed plan; no execution')
                return 0
            if a.logs is None: raise ValueError('--logs is required for composed execution')
            try:
                if a.recheck:
                    print('KIT PASS recheck '+json.dumps(recheck(a.bundle, a.logs), sort_keys=True))
                    return 0
                execute_composed(a.bundle, a.logs, backend=a.backend, resume=a.resume)
            except BoundaryError as error:
                raise NumericalError(str(error)) from error
            print('KIT PASS simulated composed boundaries')
            return 0
        if a.recheck: raise ValueError('--recheck requires --composed')
        if a.resume: raise ValueError('--resume requires --composed')
        execute(a)
    except NumericalError as error:
        print('KIT FAIL '+str(error)); return 1
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, StopIteration, TypeError, IndexError, OverflowError, termios.error, TransportError) as error:
        print('KIT TRANSPORT_ERROR '+str(error)); return 2
    if a.dry_run: return 0
    print('KIT PASS UART and dumped bytes match frozen reference'); return 0


def _terminated(signum, frame):
    raise TransportError('runner terminated')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminated)
    sys.exit(main())
