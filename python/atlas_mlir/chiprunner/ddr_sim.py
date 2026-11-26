"""Simulated board for DDR kits: SophiaLake's DDR as a file, loaded the way FireSim's loadmem writes DRAM.

Writes and reads go straight to a sparse file the size of board DDR. Starting the hart runs Spike from the entry with
that file mapped as DDR and the Atlas devices of spike_atlas.cc: IMEM and CSR words reach the model checker's
functional runtime, which executes each launch in the pinned model (numerics="rtl") over the same file, and the UART
records go to a log. The file and its session persist in the board directory across runs, as a powered board's DDR
does. No tensor arithmetic is implemented here.
"""
from __future__ import annotations

import fcntl
import json
import mmap
import os
from pathlib import Path
import secrets
import select
import shutil
import signal
import struct
import subprocess
import sys
import time

if __package__:
    from .composed_sim import (VMEM_BASE, VMEM_BYTES, _application_runtime, _bind_sources, _build, _model_root,
                               _payload_elf, _step_runtime, _tool)
    from .ddr import Transport
    from .kit_manifest import HOST_WINDOW, PAGE, safe_path
    from .run_kit import NumericalError
    from .tcl_rpc import TransportError
else:
    from composed_sim import (VMEM_BASE, VMEM_BYTES, _application_runtime, _bind_sources, _build, _model_root,
                              _payload_elf, _step_runtime, _tool)
    from ddr import Transport
    from kit_manifest import HOST_WINDOW, PAGE, safe_path
    from run_kit import NumericalError
    from tcl_rpc import TransportError


class SimulatedBoard(Transport):
    requires_tohost = False  # Spike has no cache: DDR holds every store when the UART reports completion

    def __init__(self, root, bundle, board, ddr_window):
        self.root, self.bundle, self.board = safe_path(root), safe_path(bundle), board
        self.base, self.bytes = ddr_window['address'], ddr_window['bytes']
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker = self.work = None
        self.lock = os.open(self.root / 'lock', os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one run at a time owns the board
        except BlockingIOError:
            os.close(self.lock)
            self.lock = None
            raise TransportError('simulated board is in use by another run') from None
        self.path, session = self.root / 'ddr.bin', self.root / 'session'
        if not self.path.is_file() or not session.is_file():
            # Power on: DDR holds nothing the protocol may trust, and not zeros, up to the device span's end.
            window, span = _device_span(self.bundle)
            with self.path.open('wb') as stream:
                stream.truncate(self.bytes)
                stream.write(b'\xa5' * (window + span - self.base))
            session.write_text(secrets.token_hex(16) + '\n')
        if self.path.stat().st_size != self.bytes:
            raise TransportError('simulated DDR file has the wrong size')
        self.session = session.read_text().strip()
        self.journal = self.root / 'load-journal.json'
        self.runs = 0

    def identify(self):
        return dict(board=self.board, link='simulated', ddr=dict(address=self.base, bytes=self.bytes),
                    session=self.session)

    def _offset(self, address, count):
        if not self.base <= address <= address + count <= self.base + self.bytes:
            raise TransportError(f'access of {count} bytes at {address:#x} is outside board DDR')
        return address - self.base

    def write(self, address, data):
        with self.path.open('r+b') as stream:
            stream.seek(self._offset(address, len(data)))
            stream.write(data)

    def read(self, address, count):
        with self.path.open('rb') as stream:
            stream.seek(self._offset(address, count))
            return stream.read(count)

    def start(self, entry, tohost):
        deployment = json.loads((self.bundle / 'manifest.json').read_text())['deployment']
        window, span = _device_span(self.bundle)
        application = self.bundle / 'application'
        images = [dict(program=str(safe_path(application / row['path'])),
                       plan=str(safe_path(application / Path(row['path']).parent / 'execution_plan.json')))
                  for row in deployment['provider']['images']]
        runtime, _ = _application_runtime()
        self.runs += 1
        self.work = self.root / f'start-{self.runs:03d}-{os.getpid()}'
        self.work.mkdir()
        tag = json.loads((self.bundle / 'manifest.json').read_text())['runner']['record_tag']
        request = dict(ddr=str(self.path), ddr_base=self.base, ddr_bytes=self.bytes, entry=entry, tohost=tohost,
                       window=dict(address=window, bytes=span), images=images, record_tag=tag,
                       manifest=str(safe_path(self.bundle / 'manifest.json')),
                       spike=_tool('SPIKE', 'spike'), atlas_root=str(safe_path(os.environ['ATLAS_ROOT'])),
                       model_root=str(_model_root()), runtime=str(runtime), cache=str(self.root / 'libraries'))
        (self.work / 'request.json').write_text(json.dumps(request))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1')
        with (self.work / 'model.log').open('wb') as log:
            self.worker = subprocess.Popen([_tool('COMPOSED_MODEL_PYTHON', 'python3.14'), str(Path(__file__).absolute()),
                                            str(self.work / 'request.json')], cwd=self.work, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        (self.work / 'worker.pid').write_text(f'{self.worker.pid}\n')

    def records(self, cycles, console, tag):
        """CPU_SECONDS_PER_CYCLE times the kit's cycle budget bounds the simulator's CPU seconds (its whole process
        group), not wall time, so a loaded host cannot fail a run; WALL_FACTOR times that catches a simulator that blocks."""
        limit = CPU_SECONDS_PER_CYCLE * cycles
        deadline, cpu = time.monotonic() + WALL_FACTOR * limit, 0.0
        try:
            while (code := self.worker.poll()) is None:
                cpu = max(cpu, _group_cpu(self.worker.pid))
                if cpu > limit:
                    raise TransportError(f'UART completion timeout: the simulator used {limit:g} s of CPU')
                if time.monotonic() > deadline:
                    raise TransportError(f'UART completion timeout: no completion in {WALL_FACTOR * limit:g} s of '
                                         'wall time')
                time.sleep(.5)
        finally:
            (self.work / 'cpu-seconds.txt').write_text(f'{cpu:.2f}\n')
            self._stop()
        data = (self.work / 'uart.log').read_bytes() if (self.work / 'uart.log').is_file() else b''
        console.write(data); console.flush()
        if code == 3:
            raise NumericalError('UART reported nonzero status')
        if code:
            lines = (self.work / 'model.log').read_text(errors='replace').strip().splitlines()
            raise TransportError('simulated hart failed: ' + (lines[-1] if lines else f'worker exit {code}'))
        _model_root()  # the pinned model must not change during a run
        return data

    def _stop(self):
        """Kill the worker's whole process group, so no Spike outlives it."""
        if self.worker is not None:
            try:
                os.killpg(self.worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.worker.wait()

    def close(self):
        self._stop()
        if self.lock is not None:
            os.close(self.lock)  # releases the board
            self.lock = None


# Spike and the Python model run far slower than the chip: one DroNet 200 run used 108.4 and 119.8 CPU seconds (positive
# and negative) against the board's 120 s at 100 MHz, so the simulated board allows 4 CPU seconds per 100,000,000 budget
# cycles (480 s for the 12,000,000,000-cycle budget), whatever clock the kit's board supplies.
CPU_SECONDS_PER_CYCLE = 4 / 100_000_000
WALL_FACTOR = 10  # times the CPU limit: the backstop for a simulator that blocks without using CPU


def _group_cpu(group):
    """CPU seconds of every live process in `group`, including the children they have reaped (Linux /proc)."""
    ticks = 0
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except OSError:
            continue  # the process exited while being read
        if int(fields[2]) == group:
            ticks += sum(int(value) for value in fields[11:15])  # utime, stime, cutime, cstime
    return ticks / os.sysconf('SC_CLK_TCK')


def _device_span(bundle):
    """The device window's address and the bytes of it up to the last byte any declared buffer uses."""
    deployment = json.loads((bundle / 'manifest.json').read_text())['deployment']
    window = deployment['window']
    # The device also reads a resident arena in place; it starts where the window ends.
    arena = deployment.get('resident')
    rows = [*deployment['allocations'], *deployment['inputs'], *deployment.get('state_inputs', []), *deployment['outputs'],
            *deployment['provider'].get('reservations', []), *filter(None, [deployment.get('parameters'), arena])]
    span = -(-(max(row['address'] + row['bytes'] for row in rows) - window['address']) // PAGE) * PAGE
    if (window['address'] % PAGE or not 0 < span <= -(-(window['bytes'] + (arena['bytes'] if arena else 0)) // PAGE) * PAGE
            or (arena and arena['address'] != window['address'] + window['bytes'])):
        raise TransportError('device window is not page aligned inside board DDR')
    return window['address'], span


def _worker(path):
    """Run Spike from the entry over the DDR file; exit 0 at completion, 3 at a failing status, else an error.

    Without `tohost`, the UART device stops the hart at the COMPLETE or failing STATUS record. With it, the device is
    given a tag the program never prints, and the hart stops once the program writes its exit word, as FireSim's
    host does.
    """
    request = json.loads(safe_path(path).read_text())
    work = Path(path).parent
    _bind_sources(request)
    import torch
    torch.set_num_threads(1)
    base, size = request['window']['address'], request['window']['bytes']
    vmem = work / 'vmem.bin'
    with vmem.open('wb') as stream:
        stream.truncate(VMEM_BYTES)
    include = os.environ.get('SPIKE_INCLUDE') or str(Path(os.path.realpath(request['spike'])).parents[1] / 'include')
    plugin, _ = _build('g++', ['-std=c++17', '-O2', '-shared', '-fPIC', '-Wall', '-I', include],
                       [Path(__file__).with_name('spike_atlas.cc')], safe_path(request['cache']), '.so')
    if shutil.which('dtc') is None:
        raise ValueError('dtc on PATH is required to attach the simulated Atlas devices')
    from functional_runtime import resident_schedule
    # A kit that declares resident images is checked start by start against its published schedule.
    schedule = resident_schedule(json.loads(safe_path(request['manifest']).read_text())['deployment'])
    runtime = _step_runtime(request, work, base, size, schedule)
    descriptor = os.open(request['ddr'], os.O_RDWR)
    runtime.memory = mmap.mmap(descriptor, size, offset=base - request['ddr_base'])  # the device sees DDR itself
    os.close(descriptor)
    # Spike needs one ELF to start; this one loads nothing, so DDR holds only what the transport wrote.
    header = bytearray(64)
    header[:7] = b'\x7fELF\x02\x01\x01'
    struct.pack_into('<HHIQ', header, 16, 2, 243, 1, request['entry'])
    (work / 'empty.elf').write_bytes(_payload_elf(header, []))
    requests, plugin_requests = os.pipe()
    plugin_responses, responses = os.pipe()
    uart = work / 'uart.log'
    tag = request['record_tag'] if request['tohost'] is None else 'tohost exit'
    scratch = HOST_WINDOW[1] - HOST_WINDOW[0]
    argv = [request['spike'], '--isa=rv64gcv_zicntr', '--priv=m', f'-m0x{HOST_WINDOW[0]:x}:0x{scratch:x}',
            f'--pc=0x{request["entry"]:x}', f'--extlib={plugin}',
            f'--device=atlas_vmem,0x{VMEM_BASE:x},0x{VMEM_BYTES:x},{vmem}',
            f'--device=atlas_vmem,0x{request["ddr_base"]:x},0x{request["ddr_bytes"]:x},{request["ddr"]}',
            f'--device=atlas_mmio,0x20000,{plugin_requests},{plugin_responses}',
            f'--device=atlas_uart,0x10020000,{uart},{tag}', str(work / 'empty.elf')]
    if any(',' in str(name) for name in (vmem, uart, request['ddr'])):
        raise ValueError('simulator paths must not contain commas')
    with (work / 'spike.log').open('wb') as log:
        process = subprocess.Popen(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=(plugin_requests, plugin_responses))
    os.close(plugin_requests); os.close(plugin_responses)
    started = time.monotonic()
    exit_word, stopped = None, False
    if request['tohost'] is not None:
        page = (request['tohost'] - request['ddr_base']) // PAGE * PAGE
        descriptor = os.open(request['ddr'], os.O_RDONLY)
        exit_word = mmap.mmap(descriptor, PAGE, prot=mmap.PROT_READ, offset=page)
        os.close(descriptor)
        at = request['tohost'] - request['ddr_base'] - page
    try:
        pending = b''
        while True:
            if exit_word is not None and int.from_bytes(exit_word[at:at + 8], 'little'):
                process.kill(); process.wait()
                stopped = True
                break
            if not select.select([requests], [], [], .05)[0]:
                continue
            chunk = os.read(requests, 4096)
            if not chunk:
                break
            pending += chunk
            while len(pending) >= 13:
                operation, address, word = pending[:1], *struct.unpack_from('<QI', pending, 1)
                pending = pending[13:]
                if operation == b'R':
                    os.write(responses, struct.pack('<I', runtime.read(None, address) & 0xffffffff))
                elif operation == b'W':
                    runtime.write(None, address, word)
                else:
                    raise ValueError('invalid device request')
        code = 0 if stopped else process.wait()
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        os.close(requests); os.close(responses)
    execution = runtime.finish()
    (work / 'result.json').write_text(json.dumps(dict(exit=code, launches=len(execution['launches']),
        instructions=execution['total_instructions'], seconds=round(time.monotonic() - started, 3))) + '\n')
    sys.exit(code if code in (0, 3) else 1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('internal worker requires a request file')
    _worker(sys.argv[1])
