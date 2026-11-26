"""Live staged steps over in-memory OpenOCD, UART, and process peers. Protocol only, not ELF math or hardware."""
import contextlib
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_mlir.chiprunner import composed, composed_sim, preflight, run_kit
from atlas_mlir.chiprunner.composed_tcl import TclTransport
from atlas_mlir.chiprunner.tcl_rpc import TransportError
from test_chiprunner_composed import elf

BASE, ENTRY = 0x200c0000, 0x08000000
BENCH = SimpleNamespace(openocd='/unused/openocd', serial='/unused/serial',
                        tcl_port=1, rpc_timeout=1, start_timeout=1)


def write(root, manifest):
    (root / 'manifest.json').write_text(json.dumps(manifest))
    names = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(hashlib.sha256((root / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))


def step_plan(root):
    """x -> s0 (+1) -> m -> s1 (+1) -> f, each step a linked ELF with its own deployment hash ranges."""
    (root / 'bench').mkdir(exist_ok=True)
    if not (root / 'bench/atlas-soc.cfg').exists(): (root / 'bench/atlas-soc.cfg').write_text('# unused by the peers\n')
    files = {'x.bin': b'\x01\x02\x03\x04', 'x-negative.bin': b'\x01\x02\x83\x04', 'm.bin': b'\x02\x03\x04\x05', 'f.bin': b'\x03\x04\x05\x06'}
    for name, data in files.items(): (root / name).write_bytes(data)
    regions = []
    for i, (source, target) in enumerate((('x', 'm'), ('m', 'f'))):
        rid = 's%d' % i; step = root / rid; step.mkdir()
        (step / 'chip.elf').write_bytes(elf(ENTRY, [(ENTRY, 5, rid.encode() * 8), (BASE + 256, 6, bytes(64))]))
        output = dict(id='o', address=BASE + 64, bytes=4)
        (step / 'deployment.json').write_text(json.dumps(dict(hash_ranges=[dict(output, role='public_output')])))
        regions.append(dict(id=rid, kind='elf', elf=rid + '/chip.elf', entry=ENTRY, deployment=rid + '/deployment.json', images=[], launches=1,
                            window=dict(address=BASE, bytes=4096), inputs=[dict(id='i', address=BASE, bytes=4, source=dict(value=source))],
                            outputs=[dict(output, value=dict(id=target))]))
    values = [dict(id='x', bytes=4, file='x.bin'), dict(id='m', bytes=4, reference='m.bin'), dict(id='f', bytes=4, reference='f.bin')]
    manifest = dict(runner=dict(schema='atlas.kit.v1', record_tag='ATLAS'),
                    composed=dict(values=values, regions=regions, negative=dict(value='x', file='x-negative.bin'), receiver_baud=115200))
    write(root, manifest)
    return manifest


def openocd_crc(data):
    """OpenOCD's image_calculate_checksum: CRC-32, polynomial 0x04C11DB7, initial 0xFFFFFFFF, MSB first, no final XOR."""
    crc = 0xffffffff
    for byte in data:
        crc ^= byte << 24
        for _ in range(8): crc = ((crc << 1) ^ 0x04c11db7 if crc & 0x80000000 else crc << 1) & 0xffffffff
    return crc


def peers(manifest, root, *, status=0, wrong_hash=False, complete=True, reset_fails=False, close_fails=False, during_records=None,
          corrupt=None, traffic=None, stale=None, idle=False, starts=None):
    """OpenOCD Tcl, UART, and process peers. On resume the loaded step adds one to each input byte.

    `close_fails` makes closing the Tcl connection, the first cleanup step, raise; `during_records` runs while the
    UART is awaited. `verify_image` compares OpenOCD's CRC-32 of each section and the bytes only on a CRC mismatch, as
    OpenOCD does; a binary `load_image` at an address in `corrupt` leaves those bytes instead. `traffic` collects each
    command's (operation, bytes) over JTAG. `stale` is memory left from earlier work; an `idle` step writes no output.
    `starts` collects (region, memory) at each start.
    """
    memory, log, uart, processes = dict(stale or {}), [], {}, []
    regions = {str(root / r['elf']): r for r in manifest['composed']['regions']}
    loaded = {}

    def step_of(sections):
        """The region whose step ELF holds every section of a sparse load image, byte for byte."""
        def within(region, address, data):
            return any(a <= address and address + len(data) <= a + len(d) and d[address - a:address - a + len(data)] == data
                       for a, d, _, _ in composed_sim._elf_loads((root / region['elf']).read_bytes())[1])
        found = [r for r in regions.values() if all(within(r, a, d) for a, d in sections)]
        assert len(found) == 1, 'a sparse load image must come from one step ELF'
        return found[0]

    class RPC:
        broken = False
        timeout = 1
        def __init__(self, **kwargs): pass
        def command(self, text):
            log.append(text)
            if reset_fails and text == 'reset halt': raise TransportError('reset halt failed')
            match = re.fullmatch(r'(load_image|verify_image|dump_image) \{([^}]+)\}(?: (0x[0-9a-f]+) (bin|[0-9]+))?', text)
            if match and match[1] != 'dump_image':
                raw = Path(match[2]).read_bytes()
                sections = [(int(match[3], 16), raw)] if match[3] else [(a, d) for a, d, _, _ in composed_sim._elf_loads(raw)[1]]
            if match and traffic is not None:
                traffic.append((match[1], int(match[4]) if match[1] == 'dump_image' else sum(len(d) for _, d in sections)))
            if match and match[1] == 'load_image' and match[3] is None:
                loaded['region'] = regions.get(match[2]) or step_of(sections)
                for address, data in sections:
                    for i, byte in enumerate(data): memory[address + i] = byte
            elif match and match[1] == 'load_image':
                address = int(match[3], 16)
                for i, byte in enumerate((corrupt or {}).get(address, raw)): memory[address + i] = byte
            elif match and match[1] == 'verify_image':
                for address, data in sections:
                    held = bytes(memory.get(address + i, 0) for i in range(len(data)))
                    if openocd_crc(held) != openocd_crc(data) and held != data:
                        raise TransportError('OpenOCD command failed: checksum mismatch - attempting binary compare')
            elif match and match[1] == 'dump_image':
                address = int(match[3], 16)
                Path(match[2]).write_bytes(bytes(memory[address + i] for i in range(int(match[4]))))
            elif text.startswith('resume '):
                region = loaded['region']
                assert int(text.split()[1], 16) == region['entry']
                if starts is not None: starts.append((region['id'], dict(memory)))
                data = bytes(memory[region['inputs'][0]['address'] + i] for i in range(4))
                row = region['outputs'][0]
                if not idle:
                    for i, byte in enumerate(data): memory[row['address'] + i] = (byte + 1) & 0xff
                produced = bytes(memory[row['address'] + i] for i in range(4))
                digest = hashlib.sha256(produced[::-1] if wrong_hash else produced).hexdigest()
                uart['text'] = 'ATLAS STATUS %d CYCLES 9\nATLAS HASH o public_output %s\n' % (status, digest) + ('ATLAS COMPLETE\n' if complete else '')
            return ''
        def close(self):
            log.append('rpc closed')
            if close_fails: raise TransportError('Tcl close failed')

    class UART:
        def __init__(self, path, baud): log.append('uart %s %d' % (path, baud))
        def clear(self): uart.pop('text', None)
        def close(self): log.append('uart closed')
        def records(self, timeout, console, tag):
            if during_records: during_records()
            text = uart['text']; console.write(text.encode())
            if status: raise run_kit.NumericalError('UART reported nonzero status')
            if not complete: raise TransportError('UART completion timeout')
            return text.encode()

    class Process:
        def __init__(self, command, **kwargs): processes.append(self); self.command = command; self.stopped = False
        def poll(self): return None
        def terminate(self): self.stopped = True
        def wait(self, **kwargs): return 0

    patches = [patch.object(preflight, 'bench', side_effect=lambda *a: log.append('bench preflight')),
               patch.object(run_kit, 'UART', UART), patch.object(run_kit, 'TclRPC', RPC), patch.object(run_kit.subprocess, 'Popen', Process)]
    return patches, log, processes


class LiveStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'bundle'; self.root.mkdir(); self.manifest = step_plan(self.root)
        self.logs = Path(self.tmp.name) / 'logs'

    def run_live(self, mutation=False, bench=BENCH, **behavior):
        patches, self.log, self.processes = peers(self.manifest, self.root, **behavior)
        with contextlib.ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            return composed.execute(self.root, self.logs, backend='jtag', mutation=mutation, bench=bench)

    def test_each_step_loads_its_elf_runs_to_uart_completion_and_feeds_the_next(self):
        journal = self.run_live()
        self.assertEqual([r['id'] for r in journal['completed']], ['s0', 's1'])
        self.assertEqual((self.logs / 's1-output-0.bin').read_bytes(), b'\x03\x04\x05\x06')
        self.assertEqual(journal['completed'][1]['inputs']['inputs:i'], composed.sha(b'\x02\x03\x04\x05'))
        self.assertEqual(self.log[:2], ['bench preflight', 'uart /unused/serial 115200'])
        self.assertEqual(self.log.count('reset halt'), 1)
        for rid in ('s0', 's1'):
            elf_load = self.log.index('load_image {%s}' % (self.root / rid / 'chip.elf'))
            resume = [i for i, text in enumerate(self.log) if text == 'resume 0x8000000'][int(rid[1])]
            self.assertLess(elf_load, resume)
        # Per step the output is dumped twice: the poison write's readback, then the hash check; the runner reuses the hashed bytes.
        self.assertEqual(sum(text.startswith('dump_image') and text.endswith('0x200c0040 4') for text in self.log), 4)
        self.assertEqual([re.sub('[0-9]+-', '', p.name) for p in sorted(self.logs.glob('uart-*.log'), key=lambda p: p.stat().st_mtime_ns)], ['uart-s0.log', 'uart-s1.log'])
        self.assertTrue(self.processes[0].stopped)
        self.assertEqual(self.log[-2:], ['rpc closed', 'uart closed'])

    def test_each_write_is_read_back_once_and_the_poison_is_not_read_again(self):
        traffic = []
        self.run_live(traffic=traffic)
        resumes = [i for i, text in enumerate(self.log) if text.startswith('resume')]
        for rid, resume in zip(('s0', 's1'), resumes):
            staged = self.log[self.log.index('load_image {%s}' % (self.root / rid / 'chip.elf')):resume]
            poison = [i for i, text in enumerate(staged) if text.startswith('load_image') and text.endswith(' 0x200c0040 bin')]
            self.assertEqual(len(poison), 1)
            self.assertTrue(staged[poison[0] + 1].startswith('dump_image') and staged[poison[0] + 1].endswith(' 0x200c0040 4'))
            self.assertEqual(sum(text.endswith(' 0x200c0040 4') for text in staged), 1)
            # A matching readback leaves no file; the hash-checked output dumps after the start are kept.
            dumped = [Path(re.search(r'\{([^}]+)\}', text)[1]) for text in staged if text.startswith('dump_image')]
            self.assertEqual(len(dumped), 2)
            self.assertFalse(any(path.exists() for path in dumped))
        kept = [text for i, text in enumerate(self.log) if text.startswith('dump_image') and not self.log[i - 1].startswith('load_image')]
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(Path(re.search(r'\{([^}]+)\}', text)[1]).exists() for text in kept))
        # Per step the ELF is loaded and verified, the poison and the input are loaded and read back, and the output is
        # dumped once. The removed second poison read was one more dump of every output range, 4 bytes per step here.
        code = sum(len(data) for _, data, _, _ in composed_sim._elf_loads((self.root / 's0/chip.elf').read_bytes())[1])
        step = [('load_image', code), ('verify_image', code), ('load_image', 4), ('dump_image', 4),
                ('load_image', 4), ('dump_image', 4), ('dump_image', 4)]
        self.assertEqual(traffic, step * 2)

    def test_a_poison_write_that_reads_back_differently_fails_the_run_before_the_step_starts(self):
        with self.assertRaisesRegex(TransportError, 'write readback differs at 0x200c0040') as caught:
            self.run_live(corrupt={BASE + 64: b'\x00\xa5\xa5\xa5'})
        self.assertNotIsInstance(caught.exception, composed.BoundaryError)
        transfers = [text for text in self.log if text.startswith(('load_image', 'verify_image', 'dump_image'))]
        self.assertTrue(transfers[-2].startswith('load_image') and transfers[-2].endswith(' 0x200c0040 bin'))
        self.assertTrue(transfers[-1].startswith('dump_image') and transfers[-1].endswith(' 0x200c0040 4'))
        self.assertEqual(Path(re.search(r'\{([^}]+)\}', transfers[-1])[1]).read_bytes(), b'\x00\xa5\xa5\xa5')  # kept as evidence
        self.assertFalse(any(text.startswith('resume') for text in self.log))
        journal = json.loads((self.logs / 'journal.json').read_text())
        self.assertEqual((journal['completed'], journal['active']), ([], 's0'))
        self.assertTrue(self.processes[0].stopped)

    def test_a_crc_preserving_write_error_passes_verify_image_but_not_the_readback(self):
        self.assertEqual(openocd_crc(b'123456789'), 0x0376e6e7)
        poison, held = b'\xa5' * 16, bytes.fromhex('27c52b7e25') + b'\xa5' * 11
        self.assertEqual(openocd_crc(held), openocd_crc(poison))
        patches, self.log, _ = peers(self.manifest, self.root, corrupt={BASE + 256: held})
        with patches[2]: rpc = run_kit.TclRPC()
        work = Path(self.tmp.name) / 'work'; work.mkdir()
        transport = TclTransport(self.root, work, self.manifest, rpc=rpc)
        # The previous write path, load_image then verify_image, accepts these bytes.
        (work / 'poison.bin').write_bytes(poison)
        for command in ('load_image', 'verify_image'): rpc.command('%s {%s} %#x bin' % (command, work / 'poison.bin', BASE + 256))
        with self.assertRaisesRegex(TransportError, 'write readback differs at 0x200c0100'):
            transport.write(BASE + 256, poison)

    def test_a_step_that_writes_no_output_fails_as_poisoned_although_its_elf_no_longer_carries_the_poison(self):
        """The step ELF's window image holds the binder's poison over the output; the loader leaves it out, so only
        the runner's exact poison write stands between stale bytes and a step that wrote nothing."""
        image = bytes(64) + b'\xa5' * 4 + bytes(60)
        (self.root / 's0/chip.elf').write_bytes(elf(ENTRY, [(ENTRY, 5, b's0' * 8), (BASE, 6, image)]))
        write(self.root, self.manifest)
        traffic = []
        with self.assertRaisesRegex(ValueError, 'output remained poisoned: s0/o') as caught:
            self.run_live(stale={BASE + 64 + i: byte for i, byte in enumerate(b'\x02\x03\x04\x05')}, idle=True, traffic=traffic)
        self.assertNotIsInstance(caught.exception, composed.BoundaryError)
        # The ELF load sends the window image without the input and output ranges: 128 - 4 - 4 bytes of it.
        self.assertEqual(traffic[0], ('load_image', 16 + 120))
        self.assertEqual(json.loads((self.logs / 'journal.json').read_text())['active'], 's0')

    def test_a_weights_row_inside_a_step_elf_load_is_rewritten_before_every_start(self):
        """The loader leaves a weights row inside a step ELF load out of that load, so the row must not be resident: the
        runner writes it after every load. Here s1's ELF loads its own bytes over s0's weights between s0's starts."""
        weights = b'\x11\x22\x33\x44'
        (self.root / 'w.bin').write_bytes(weights)
        images = {'s0': bytes(64) + b'\xa5' * 4 + bytes(60) + weights + bytes(124), 's1': bytes(64) + b'\xa5' * 4 + bytes(188)}
        for rid, image in images.items():
            (self.root / rid / 'chip.elf').write_bytes(elf(ENTRY, [(ENTRY, 5, rid.encode() * 8), (BASE, 6, image)]))
        self.manifest['composed']['regions'][0]['weights'] = [dict(id='w', address=BASE + 128, bytes=4, file='w.bin')]
        write(self.root, self.manifest)
        traffic, starts = [], []
        patches, self.log, _ = peers(self.manifest, self.root, traffic=traffic, starts=starts)
        with contextlib.ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            work = Path(self.tmp.name) / 'work'; work.mkdir()
            session = composed.Session(self.root, work, 'jtag', bench=BENCH)
            try:
                for call in range(2): session.call(Path(self.tmp.name) / ('logs-%d' % call))
            finally:
                session.close()
        # At every start of s0 the window holds its full image with the runner's writes: input, poison, and weights.
        full = bytearray(images['s0']); full[0:4] = b'\x01\x02\x03\x04'
        s0 = [memory for rid, memory in starts if rid == 's0']
        self.assertEqual(len(s0), 2)
        for memory in s0:
            self.assertEqual(bytes(memory[BASE + i] for i in range(256)), bytes(full))
        self.assertNotIn(('s0', 'weights', 'w'), session.resident)
        # The s0 ELF load sends its window image without the input, output, and weights ranges: 256 - 12 bytes of it.
        self.assertEqual(traffic[0], ('load_image', 16 + 244))

    def test_every_cleanup_step_runs_when_the_first_raises(self):
        with self.assertRaisesRegex(TransportError, 'Tcl close failed'):
            self.run_live(close_fails=True)
        self.assertTrue(self.processes[0].stopped)
        self.assertEqual(self.log[-2:], ['rpc closed', 'uart closed'])

    def test_cleanup_failure_surfaces_after_a_boundary_mismatch(self):
        with self.assertRaisesRegex(TransportError, 'Tcl close failed') as caught:
            self.run_live(mutation=True, close_fails=True)
        self.assertIsInstance(caught.exception.__context__, composed.BoundaryError)
        self.assertTrue(self.processes[0].stopped)

    def test_declared_negative_fails_at_its_first_consumer_only_numerically(self):
        with self.assertRaisesRegex(composed.BoundaryError, r'boundary mismatch: s0/o \(value m\)'):
            self.run_live(mutation=True)
        self.assertEqual(sum(text.startswith('resume') for text in self.log), 1)

    def test_runtime_failures_are_never_numerical(self):
        cases = {'dump differs from execution hash: o': dict(wrong_hash=True), 'nonzero status': dict(status=3),
                 'completion timeout': dict(complete=False)}
        for message, behavior in cases.items():
            with self.subTest(message=message):
                self.logs = Path(self.tmp.name) / ('logs-' + message.split()[0])
                with self.assertRaisesRegex(Exception, message) as caught:
                    self.run_live(**behavior)
                self.assertNotIsInstance(caught.exception, composed.BoundaryError)
                self.assertNotIn('resume 0x8000000', self.log[self.log.index('resume 0x8000000') + 1:])
                self.assertTrue(self.processes[0].stopped)

    def test_a_failed_initial_reset_closes_openocd_uart_and_rpc_without_retry(self):
        with self.assertRaisesRegex(TransportError, 'reset halt failed'):
            self.run_live(reset_fails=True)
        self.assertEqual(self.log, ['bench preflight', 'uart /unused/serial 115200', 'reset halt', 'rpc closed', 'uart closed'])
        self.assertEqual(len(self.processes), 1)
        self.assertTrue(self.processes[0].stopped)

    def test_refusals_before_any_transport_opens(self):
        patches, log, processes = peers(self.manifest, self.root)
        patches[0] = patch.object(preflight, 'bench', side_effect=ValueError('November OpenOCD digest differs'))
        with contextlib.ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            with self.assertRaisesRegex(ValueError, 'OpenOCD digest differs'):
                composed.execute(self.root, Path(self.tmp.name) / 'logs-openocd', backend='jtag', bench=BENCH)
        self.assertEqual((log, processes), ([], []))
        del self.manifest['composed']['receiver_baud']; write(self.root, self.manifest)
        self.logs = Path(self.tmp.name) / 'logs-baud'
        with self.assertRaisesRegex(ValueError, 'receiver_baud'):
            self.run_live()
        self.assertEqual(self.processes, [])

    def test_a_step_output_must_be_a_uart_hash_range(self):
        (self.root / 's1/deployment.json').write_text(json.dumps(dict(hash_ranges=[])))
        write(self.root, self.manifest)
        with self.assertRaisesRegex(ValueError, 'not a UART hash range'):
            composed.load_plan(self.root)


if __name__ == '__main__':
    unittest.main()
