"""Physical fit and live bench prerequisites. No transport is opened here."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
if __package__:
    from .kit_manifest import DATA_WINDOW, HOST_WINDOW, inside, read_bytes, safe_path
else:
    from kit_manifest import DATA_WINDOW, HOST_WINDOW, inside, read_bytes, safe_path

# Each command prints what it finds and exits 1 when it finds nothing. macOS pgrep prints command lines only with -l
# and skips its own ancestors unless given -a (Linux pgrep never skips them); macOS lists a device's users with lsof -t.
PROCESS_CHECKS = {'linux': (['pgrep', '-fa'], ['fuser']), 'darwin': (['pgrep', '-afl'], ['lsof', '-t', '--'])}
# macOS names each serial port twice, /dev/cu.NAME and /dev/tty.NAME; a process holding either one owns the port.
SERIAL_TWINS = {'darwin': {'/dev/cu.': '/dev/tty.', '/dev/tty.': '/dev/cu.'}}


def listing(command):
    """One process or UART check; a missing tool refuses by name."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise ValueError(command[0] + ' not found; the bench checks need it (on Linux, pgrep is in procps and fuser in psmisc)') from None


def serial_devices(serial):
    """The UART device and, where the host names it twice, its twin."""
    for prefix, other in SERIAL_TWINS.get(sys.platform, {}).items():
        if serial.startswith(prefix):
            return [serial, other + serial[len(prefix):]]
    return [serial]


def openocd_builds():
    """The declared OpenOCD builds, each an executable digest and its exact version banner."""
    rows = json.loads(read_bytes(Path(__file__).with_name('openocd-builds.json')))
    # A banner identifies its build in a recorded log, so each is a distinct OpenOCD version line.
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or set(row) != {'sha256', 'banner', 'package'}
            or not all(isinstance(value, str) and value for value in row.values())
            or not re.fullmatch('[0-9a-f]{64}', row['sha256']) or not row['banner'].startswith('Open On-Chip Debugger ')
            for row in rows)
            or len({row['sha256'] for row in rows}) != len(rows) or len({row['banner'] for row in rows}) != len(rows)):
        raise ValueError('malformed OpenOCD build table')
    return rows


def physical(kit):
    raw = read_bytes(kit['elf'])
    if struct.unpack_from('<H', raw, 18)[0] != 243:
        raise ValueError('ELF must target RISC-V')
    off = struct.unpack_from('<Q', raw, 32)[0]
    size, count = struct.unpack_from('<HH', raw, 54)
    if size != 56 or count == 0 or off + size * count > len(raw):
        raise ValueError('invalid ELF program headers')
    loads = []
    for i in range(count):
        typ, flags, pos, va, pa, files, mem, align = struct.unpack_from('<IIQQQQQQ', raw, off+i*size)
        if typ != 1:
            continue
        if files > mem or pos + files > len(raw):
            raise ValueError('invalid ELF segment')
        host = (HOST_WINDOW[0], min(HOST_WINDOW[1], kit['manifest']['bench'].get('usable_host_end', HOST_WINDOW[1])))
        if not (inside(host, pa, mem) or inside(DATA_WINDOW, pa, mem)):
            raise ValueError('ELF exceeds on-chip host or VMEM capacity')
        if any(pa < row['address']+row['memory_bytes'] and row['address'] < pa+mem for row in loads):
            raise ValueError('overlapping ELF segments')
        loads.append(dict(address=pa, file_bytes=files, memory_bytes=mem))
    meta = json.loads(read_bytes(kit['elf'].parent/'manifest.json'))['sizes']['chip']['loads']
    if not loads or loads != [{k: row[k] for k in ('address','file_bytes','memory_bytes')} for row in meta]:
        raise ValueError('ELF PT_LOAD differs from metadata')
    start, n = kit['dep']['window']['address'], kit['dep']['window']['bytes']
    if n <= 0 or not inside(DATA_WINDOW, start, n):
        raise ValueError('data exceeds banks 3 to 5')
    for row in kit['dep'].get('allocations', []):
        if row['bytes'] < 0 or row['address'] < start or row['address']+row['bytes'] > start+n:
            raise ValueError('allocation exceeds declared window')


def live(kit, executable, serial):
    physical(kit)
    bench(executable, serial)


def bench(executable, serial):
    """A declared OpenOCD build, no other bench process, and a free UART."""
    if not Path(executable).is_absolute():
        raise ValueError('absolute November OpenOCD path required')
    p = safe_path(executable)
    if not os.access(p, os.X_OK):
        raise ValueError('explicit executable November OpenOCD path required')
    digest = hashlib.sha256(read_bytes(p)).hexdigest()
    build = next((row for row in openocd_builds() if row['sha256'] == digest), None)
    if build is None:
        raise ValueError('OpenOCD build is not declared: sha256 ' + digest)
    r = subprocess.run([str(p), '--version'], capture_output=True, text=True, timeout=10)
    if r.returncode or build['banner'] not in r.stdout.splitlines() + r.stderr.splitlines():
        raise ValueError('OpenOCD version banner differs from its declared build')
    processes, owners = PROCESS_CHECKS.get(sys.platform, PROCESS_CHECKS['linux'])
    r = listing(processes + ['[r]un_full|[c]hain22|[o]penocd'])
    others = [line for line in r.stdout.splitlines() if line.split()[0] != str(os.getpid())]
    if r.returncode not in (0, 1) or others:
        raise ValueError('bench process check is not empty: '+'; '.join(others))
    for device in serial_devices(serial):
        r = listing(owners + [device])
        if r.returncode != 1 or r.stdout.strip():
            raise ValueError('UART ownership check is not empty: ' + device)
