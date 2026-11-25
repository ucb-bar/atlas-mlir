"""A board DDR kit runs through `atlascc run`: resumable, hash-verified loading, the UART records, and the outputs.

ATLAS_DDR_DEPLOYMENT names a packaged plain-deploy output (a board DDR kit with its loader manifest); a missing one
fails. Protocol tests use a scripted in-memory board. Simulated-board tests run the kit's unchanged ELFs in Spike with
the Atlas devices and the pinned model, so they need the simulator environment of test_chiprunner_spike.py. Neither
qualifies a board or its SPI loader.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from atlas_mlir import deployment_run
from atlas_mlir.chiprunner import ddr, ddr_sim
from atlas_mlir.chiprunner.run_kit import NumericalError
from atlas_mlir.chiprunner.tcl_rpc import TransportError
from atlas_mlir.deploy import Refusal
from atlas_mlir_native.soc import SOC


def deployment(root):
    """A fresh copy of the packaged deployment: its status and kit."""
    source = Path(os.environ['ATLAS_DDR_DEPLOYMENT'])
    out = root / 'out'
    out.mkdir()
    shutil.copy(source / 'deploy-status.json', out)
    shutil.copytree(source / 'kit', out / 'kit', symlinks=True)
    return out


class ScriptedBoard(ddr.Transport):
    """Board DDR in memory; its hart writes the given output bytes and reports their hashes, or `exit` to tohost."""
    requires_tohost = False  # its scripted hart stores straight to DDR; one test sets it

    def __init__(self, run, outputs, *, session='power-1', fail_write=None, corrupt=None, exit=1, report=None,
                 journal=None):
        self.ddr, self.memory, self.session, self.journal = run['ddr'], {}, session, journal
        self.run, self.outputs, self.exit, self.report = run, outputs, exit, report
        self.fail_write, self.corrupt, self.writes, self.started = fail_write, corrupt, 0, []

    def identify(self):
        return dict(board=self.run['board'], link='scripted', ddr=self.ddr, session=self.session)

    def write(self, address, data):
        self.writes += 1
        if self.writes == self.fail_write:
            raise TransportError('link dropped')
        if address == self.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        self.memory.pop(address, None)
        self.memory[address] = bytes(data)

    def read(self, address, count):
        for start, data in reversed(self.memory.items()):  # the newest write wins
            if start <= address and address + count <= start + len(data):
                return data[address - start:address - start + count]
        return bytes(count)

    def start(self, entry, tohost):
        self.started.append((entry, tohost))
        lines = [f'{self.run["tag"]} STATUS 0 CYCLES 1']
        for row, data in zip(self.run['hash_ranges'], self.outputs):
            self.write(row['address'], data)
            digest = self.report or hashlib.sha256(data).hexdigest()
            lines.append(f'{self.run["tag"]} HASH {row["id"]} {row["role"]} {digest}')
        lines.append(f'{self.run["tag"]} COMPLETE')
        self.uart = ('\n'.join(lines) + '\n').encode()
        if tohost is not None:
            self.write(tohost, self.exit.to_bytes(8, 'little'))

    def records(self, cycles, console, tag):
        self.cycles = cycles
        console.write(self.uart)
        return self.uart


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = deployment(self.root)
        self.bundle = self.out / 'kit'
        self.run = ddr.plan(self.bundle)
        self.reference = [(self.bundle / f'reference/output_{row["index"]:03d}.bin').read_bytes()
                          for row in self.run['hash_ranges']]
        self.journal = self.out / 'load-journal.json'

    def execute(self, board, name, **options):
        board.journal = board.journal or self.journal
        return ddr.execute(self.bundle, board, self.root / name, **options)

    def test_the_transport_gets_the_kit_budget_in_cycles_not_seconds_at_the_compiled_clock(self):
        # A board converts the budget at the clock it measures; the simulated board allows the same CPU at any kit clock.
        board = ScriptedBoard(self.run, self.reference)
        self.execute(board, 'first')
        self.assertEqual(board.cycles, 12000000000)
        self.assertEqual(ddr_sim.CPU_SECONDS_PER_CYCLE * board.cycles, 480)

    def test_a_kit_that_records_only_seconds_gets_its_budget_at_its_own_clock(self):
        # Kits packaged before the budget was stated in cycles record seconds at their compiled clock.
        record = json.loads((self.bundle / 'manifest.json').read_text())
        for clock_hz, seconds in ((100000000, 120.0), (500000000, 24.0)):
            record['bench_facts']['core_clock_hz'] = clock_hz
            record['platform_assumptions'] = dict(timeout_seconds=seconds)
            (self.bundle / 'manifest.json').write_text(json.dumps(record))
            sums = [line if not line.endswith('  manifest.json') else
                    hashlib.sha256((self.bundle / 'manifest.json').read_bytes()).hexdigest() + '  manifest.json'
                    for line in (self.bundle / 'SHA256SUMS').read_text().splitlines()]
            (self.bundle / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
            with self.subTest(clock_hz=clock_hz), contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(ddr.plan(self.bundle)['timeout_cycles'], 12000000000)
                deployment_run.execute(self.out, dry_run=True)
                self.assertIn('RUN DRY PASS', printed.getvalue())

    def test_a_write_over_a_journaled_chunk_forgets_it(self):
        board = ScriptedBoard(self.run, self.reference, journal=self.journal)
        journal = ddr.Journal(self.journal, board.identify())
        rows = lambda data, writable: dict(raw=data, rows=[dict(address=self.run['ddr']['address'] + 4096,
            bytes=len(data), file_offset=0, sha256=hashlib.sha256(data).hexdigest(), writable=writable)])
        first, other = b'A' * 4096, b'B' * 4096
        for data, writable in ((first, False), (other, True)):
            ddr.load(board, rows(data, writable), journal, io.StringIO()); journal.started()
        # The read-only chunk was journaled before another ELF wrote over it, so it must be written again.
        self.assertEqual(ddr.load(board, rows(first, False), journal, io.StringIO())['resumed'], 0)
        self.assertEqual(board.read(self.run['ddr']['address'] + 4096, 4096), first)

    def test_the_journal_belongs_to_the_board_not_the_deployment(self):
        board = ScriptedBoard(self.run, self.reference, journal=self.root / 'board-journal.json')
        self.execute(board, 'first')
        second = self.root / 'second'
        second.mkdir()
        shutil.copytree(self.bundle, second / 'kit', symlinks=True)
        read_only = sum(not row['writable'] for row in self.run['rows'])
        record = ddr.execute(second / 'kit', board, self.root / 'second-run')
        self.assertEqual(record['load']['resumed'], read_only)
        self.assertFalse(self.journal.exists())

    def test_interrupted_load_resumes_from_the_first_unverified_chunk(self):
        rows = len(self.run['rows'])
        self.assertGreater(rows, 2)
        board = ScriptedBoard(self.run, self.reference, fail_write=3)
        with self.assertRaisesRegex(TransportError, 'link dropped'):
            self.execute(board, 'first')
        self.assertEqual(len(json.loads(self.journal.read_text())['verified']), 2)
        self.assertEqual(board.started, [])
        board.fail_write = None
        record = self.execute(board, 'second')
        self.assertEqual((record['verdict'], record['load']['resumed'], record['load']['written']), ('EXACT', 2, rows - 2))
        self.assertEqual(board.started, [(self.run['entry'], self.run['tohost'])])

    def test_a_chunk_that_reads_back_wrong_fails_and_stays_unverified(self):
        row = self.run['rows'][1]
        board = ScriptedBoard(self.run, self.reference, corrupt=row['address'])
        message = f'load verification failed: {row["bytes"]} bytes at {row["address"]:#x}'
        with self.assertRaisesRegex(TransportError, message):
            self.execute(board, 'corrupt')
        self.assertEqual(board.started, [])
        self.assertNotIn(ddr.Journal.key(row), json.loads(self.journal.read_text())['verified'])

    def test_a_run_keeps_only_read_only_chunks_and_a_new_session_reloads_all(self):
        board = ScriptedBoard(self.run, self.reference)
        self.execute(board, 'first')
        read_only = sum(not row['writable'] for row in self.run['rows'])
        self.assertGreater(read_only, 0)
        self.assertEqual(self.execute(board, 'again')['load']['resumed'], read_only)
        board.session = 'power-2'
        self.assertEqual(self.execute(board, 'power-cycled')['load']['resumed'], 0)

    def test_without_a_session_journaled_chunks_are_read_back(self):
        board = ScriptedBoard(self.run, self.reference, session=None)
        self.execute(board, 'first')
        read_only = [row for row in self.run['rows'] if not row['writable']]
        self.assertEqual(self.execute(board, 'kept')['load']['resumed'], len(read_only))
        board.memory = {}  # the board lost its DDR
        self.assertEqual(self.execute(board, 'lost')['load']['resumed'], 0)

    def test_outputs_must_match_the_uart_hash_and_the_reference(self):
        with self.assertRaisesRegex(TransportError, 'DDR bytes differ from the UART hash'):
            self.execute(ScriptedBoard(self.run, self.reference, report='0' * 64), 'hash')
        wrong = [bytes([data[0] ^ 1]) + data[1:] for data in self.reference]
        with self.assertRaisesRegex(NumericalError, 'output differs from the kit reference'):
            self.execute(ScriptedBoard(self.run, wrong), 'wrong')

    def test_a_linked_exit_word_must_be_status_zero(self):
        # A board DDR kit is built with the tohost exit, so its ELF links the exit word.
        exit_word = self.run['tohost']
        self.assertIsNotNone(exit_word)
        board = ScriptedBoard(self.run, self.reference)
        self.assertEqual(self.execute(board, 'zero')['tohost'], 1)
        self.assertEqual(board.started[-1][1], exit_word)
        board.exit = (3 << 1) | 1
        with self.assertRaisesRegex(NumericalError, 'exited with status 3'):
            self.execute(board, 'three')
        board.exit = 2
        with self.assertRaisesRegex(TransportError, 'tohost is 0x2, not an exit word'):
            self.execute(board, 'even')

    def test_a_cached_hart_needs_a_linked_exit_word_before_any_write(self):
        board = ScriptedBoard(self.run, self.reference)
        board.requires_tohost = True
        with patch.object(ddr, 'tohost', return_value=None):  # an ELF built without the exit
            with self.assertRaisesRegex(ValueError, 'need the tohost exit build'):
                self.execute(board, 'uncached')
        self.assertEqual(board.writes, 0)

    def test_another_board_is_refused_before_any_write(self):
        board = ScriptedBoard(self.run, self.reference)
        board.ddr = dict(address=self.run['ddr']['address'], bytes=self.run['ddr']['bytes'] // 2)
        with self.assertRaisesRegex(TransportError, 'is not the kit target'):
            self.execute(board, 'other')
        self.assertEqual(board.writes, 0)

    def test_front_door_refuses_an_undeclared_transport(self):
        for status in (dict(status='PACKAGED'), dict(status='PACKAGED', transport='uart')):
            (self.out / 'deploy-status.json').write_text(json.dumps(status))
            with self.subTest(status=status), self.assertRaisesRegex(Refusal, 'declares transport .*knows jtag and spi'):
                deployment_run.execute(self.out, dry_run=True)

    def test_front_door_chooses_by_the_declared_transport(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaisesRegex(Refusal, 'no SPI loader'):
            deployment_run.execute(self.out)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            deployment_run.execute(self.out, dry_run=True)
        self.assertIn('RUN DRY PASS', printed.getvalue())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, 'packaged inputs'):
            deployment_run.execute(self.out, simulate=True, inputs=['x.npz'])


class DeviceSpanTests(unittest.TestCase):
    def span(self, **deployment):
        with tempfile.TemporaryDirectory() as root:
            rows = dict(allocations=[dict(address=0x80002000, bytes=64)], inputs=[], outputs=[], provider={})
            (Path(root) / 'manifest.json').write_text(json.dumps(dict(deployment=dict(rows, **deployment))))
            return ddr_sim._device_span(Path(root))

    def test_a_resident_span_covers_the_arena_that_starts_where_the_window_ends(self):
        window = dict(address=0x80002000, bytes=0x10000)
        self.assertEqual(self.span(window=window), (0x80002000, 0x1000))
        self.assertEqual(self.span(window=window, resident=dict(address=0x80012000, bytes=0x800)), (0x80002000, 0x11000))
        with self.assertRaisesRegex(TransportError, 'not page aligned'):
            self.span(window=window, resident=dict(address=0x80013000, bytes=0x800))
        with self.assertRaisesRegex(TransportError, 'not page aligned'):
            self.span(window=window, allocations=[dict(address=0x80012000, bytes=64)])


class FlakyBoard(ddr_sim.SimulatedBoard):
    """The simulated board with a link that drops on one write or flips a byte of one chunk."""
    fail_write = corrupt = None

    def write(self, address, data):
        self.writes = getattr(self, 'writes', 0) + 1
        if self.writes == self.fail_write:
            raise TransportError('link dropped')
        if address == self.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        super().write(address, data)


class SimulatedBoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = deployment(self.root)
        self.bundle = self.out / 'kit'
        self.run = ddr.plan(self.bundle)

    def board(self, **faults):
        board = FlakyBoard(self.out / 'simulated-board', self.bundle, self.run['board'], self.run['ddr'])
        for name, value in faults.items():
            setattr(board, name, value)
        self.addCleanup(board.close)
        return board

    def test_front_door_runs_the_kit_and_its_negative(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            logs = deployment_run.execute(self.out, simulate=True)
        self.assertIn('RUN SIM PASS', printed.getvalue())
        self.assertIn('POSITIVE EXACT (simulated board)', printed.getvalue())
        positive = json.loads((logs / 'positive/run.json').read_text())
        negative = json.loads((logs / 'negative/run.json').read_text())
        self.assertEqual({positive['board']['link'], negative['board']['link']}, {'simulated'})
        self.assertEqual((positive['verdict'], negative['verdict']), ('EXACT', 'MISMATCH'))
        self.assertEqual(positive['load']['written'], len(self.run['rows']))
        # The negative changes only a writable input, so the read-only chunks (the weights) stay verified.
        self.assertEqual(negative['load']['resumed'], sum(not row['writable'] for row in self.run['rows']))
        for row in self.run['hash_ranges']:
            self.assertEqual((logs / f'positive/output-{row["index"]}.bin').read_bytes(),
                             (self.bundle / f'reference/output_{row["index"]:03d}.bin').read_bytes())
        self.assertTrue((logs / 'positive/uart.log').read_text().endswith(self.run['tag'] + ' COMPLETE\n'))

    def test_one_run_owns_the_board(self):
        board = self.board()
        with self.assertRaisesRegex(TransportError, 'in use by another run'):
            self.board()
        board.close()
        self.board().close()

    def test_sigterm_stops_the_run_and_its_simulator(self):
        code = 'import sys; from pathlib import Path; from atlas_mlir import deployment_run; ' \
               'deployment_run.execute(Path(sys.argv[1]), simulate=True)'
        run = subprocess.Popen([sys.executable, '-c', code, str(self.out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: run.poll() is None and run.kill())
        deadline = time.monotonic() + 120
        while not list(self.out.glob('simulated-board/start-*/worker.pid')):
            self.assertLess(time.monotonic(), deadline, 'the simulator did not start')
            self.assertIsNone(run.poll(), 'the run ended before its simulator started')
            time.sleep(.2)
        worker = int(next(self.out.glob('simulated-board/start-*/worker.pid')).read_text())
        run.send_signal(signal.SIGTERM)
        output = run.communicate(timeout=60)[0].decode()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('run stopped by SIGTERM', output)
        with self.assertRaises(ProcessLookupError):
            os.killpg(worker, 0)  # neither the worker nor its Spike survives
        self.board().close()  # and the board is free again

    def test_power_on_fills_ddr_and_takes_the_board_from_the_kit(self):
        board = self.board()
        self.assertEqual(board.identify()['board'], self.run['board'])
        self.assertEqual(board.read(self.run['entry'], 16), b'\xa5' * 16)  # not zeros until the protocol loads
        other = ddr_sim.SimulatedBoard(self.root / 'other', self.bundle, 'another-board', self.run['ddr'])
        self.assertEqual(other.identify()['board'], 'another-board')

    def test_a_program_that_writes_a_resumed_read_only_chunk_fails_its_next_run(self):
        board = self.board()
        ddr.execute(self.bundle, board, self.root / 'positive')
        weights = json.loads((self.bundle / 'loader-manifest.json').read_text())['weights']
        # A stray store into the weights, which nothing on the chip forbids; the journal still trusts them.
        board.write(weights['address'], bytes(b ^ 0xff for b in board.read(weights['address'], weights['bytes'])))
        board.close()
        with contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaises(Refusal):
            deployment_run.execute(self.out, simulate=True)
        self.assertIn('RUN FAIL', printed.getvalue())
        logs = max(self.out.glob('run-*'))
        self.assertNotIn(f'LOAD {weights["address"]:#x} ', (logs / 'positive/load.log').read_text())  # resumed

    def test_interrupted_load_resumes_and_runs_exact(self):
        first = self.board(fail_write=4)
        with self.assertRaisesRegex(TransportError, 'link dropped'):
            ddr.execute(self.bundle, first, self.root / 'first')
        first.close()
        record = ddr.execute(self.bundle, self.board(), self.root / 'second')
        self.assertEqual((record['verdict'], record['load']['resumed'], record['load']['written']),
                         ('EXACT', 3, len(self.run['rows']) - 3))

    def test_hash_mismatch_fails_before_the_hart_starts(self):
        row = self.run['rows'][-1]
        board = self.board(corrupt=row['address'])
        with patch.object(board, 'start') as start, self.assertRaisesRegex(TransportError, 'load verification failed'):
            ddr.execute(self.bundle, board, self.root / 'corrupt')
        start.assert_not_called()
        data = self.run['raw'][row['file_offset']:row['file_offset'] + row['bytes']]
        self.assertNotEqual(board.read(row['address'], row['bytes']), data)


EXIT = r'''
#define UART ((volatile unsigned int *)0x10020000u)
volatile unsigned long tohost __attribute__((section(".tohost"))) = 0;
volatile unsigned long after __attribute__((section(".tohost"))) = 0;
static void put(const char *s) { while (*s) { while (UART[0] & 0x80000000u) {} UART[0] = (unsigned char)*s++; } }
void main(void) {
  put("ATLAS STATUS 0 CYCLES 1\nATLAS COMPLETE\n");
  after = 0x5a;  /* the hart runs past COMPLETE until it writes its exit word */
  tohost = 1;
  for (;;) {}
}
'''


HANG = r'''
void main(void) { for (;;) {} }  /* never reports completion */
'''


class SimulatedExitWordTests(unittest.TestCase):
    def program(self, source, tohost=True):
        """A small ELF linked at the DDR base, loaded onto a fresh simulated board and started."""
        root = Path(tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))); self.addCleanup(shutil.rmtree, root)
        compiler = shutil.which('riscv64-unknown-elf-gcc')
        if compiler is None:
            raise AssertionError('RISC-V GCC is required')
        (root / 'exit.c').write_text(source)
        from atlas_mlir_native.soc import SOC
        ddr_region = SOC.memory_region('ddr')
        base = ddr_region.base
        (root / 'start.S').write_text(f'.section .text.init\n.global _start\n_start:\n  li sp, {base + 0x100000:#x}\n  call main\n')
        (root / 'exit.ld').write_text(f'ENTRY(_start) SECTIONS {{ . = {base:#x}; .text : {{ *(.text.init) *(.text*) '
                                      f'*(.rodata*) }} . = {base + 0x2000:#x}; .data : {{ *(.tohost) }} /DISCARD/ : {{ *(.comment) }} }}')
        subprocess.run([compiler, '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-O2', '-nostdlib', '-nostartfiles',
                        '-T', str(root / 'exit.ld'), '-Wl,--no-warn-rwx-segments', str(root / 'start.S'),
                        str(root / 'exit.c'), '-o', str(root / 'exit.elf')], check=True)
        raw = (root / 'exit.elf').read_bytes()
        bundle = root / 'bundle'; bundle.mkdir()
        window = dict(address=base + 0x10000, bytes=0x1000)
        (bundle / 'manifest.json').write_text(json.dumps(dict(runner=dict(record_tag='ATLAS'), deployment=dict(
            window=window, allocations=[dict(window, name='scratch')], inputs=[], outputs=[], provider=dict(images=[])))))
        board = ddr_sim.SimulatedBoard(root / 'board', bundle, 'synthetic-board',
                                       dict(address=base, bytes=ddr_region.capacity_bytes))
        self.addCleanup(board.close)
        for row in ddr._loads(raw):
            data = raw[row['file_offset']:row['file_offset'] + row['file_bytes']]
            board.write(row['address'], data.ljust(row['memory_bytes'], b'\0'))
        board.start(int.from_bytes(raw[24:32], 'little'), ddr.tohost(raw) if tohost else None)
        return board, raw

    def test_a_linked_tohost_ends_the_run_after_complete(self):
        board, raw = self.program(EXIT)
        exit_word = ddr.tohost(raw)
        base = SOC.memory_region('ddr').base
        self.assertIn(exit_word, (base + 0x2000, base + 0x2008))  # the linker orders the two words
        self.assertEqual(board.records(240 / ddr_sim.CPU_SECONDS_PER_CYCLE, io.BytesIO(), 'ATLAS'), b'ATLAS STATUS 0 CYCLES 1\nATLAS COMPLETE\n')
        word = lambda address: int.from_bytes(board.read(address, 8), 'little')
        self.assertEqual((word(exit_word), word(exit_word ^ 8)), (1, 0x5a))

    def test_a_slow_wall_clock_with_little_cpu_does_not_time_out(self):
        board, _ = self.program(EXIT)
        os.killpg(board.worker.pid, signal.SIGSTOP)  # a host so loaded that the simulator gets no CPU for a while
        resume = threading.Timer(25, os.killpg, (board.worker.pid, signal.SIGCONT))
        resume.start(); self.addCleanup(resume.cancel)
        started = time.monotonic()
        self.assertTrue(board.records(80 / ddr_sim.CPU_SECONDS_PER_CYCLE, io.BytesIO(), 'ATLAS').endswith(b'ATLAS COMPLETE\n'))
        self.assertGreater(time.monotonic() - started, 20)

    def test_a_true_hang_still_times_out_on_cpu(self):
        board, _ = self.program(HANG, tohost=False)
        started = time.monotonic()
        with self.assertRaisesRegex(TransportError, 'used 10 s of CPU'):
            board.records(10 / ddr_sim.CPU_SECONDS_PER_CYCLE, io.BytesIO(), 'ATLAS')
        self.assertLess(time.monotonic() - started, 100)
        # SIGKILL reached the whole group; its children vanish once reaped, which a loaded host may delay.
        with self.assertRaises(ProcessLookupError):
            for _ in range(60):
                os.killpg(board.worker.pid, 0)
                time.sleep(0.5)


if __name__ == '__main__':
    unittest.main()
