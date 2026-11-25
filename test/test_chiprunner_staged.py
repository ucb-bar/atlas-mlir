"""Staged plans: whole values bound by byte ranges, step windows, and a declared negative."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir.chiprunner import composed, run_kit, step_inputs

BASE = 0x200c0000


class CopyTransport:
    """Each region writes its inputs, concatenated and incremented, to its outputs."""
    def __init__(self, *args): self.mem = {}
    def write(self, address, data):
        for i, b in enumerate(data): self.mem[address + i] = b
    def read(self, address, size): return bytes(self.mem.get(address + i, 0) for i in range(size))
    def launch(self, region):
        data = b''.join(self.read(row['address'], row['bytes']) for row in region['inputs'])
        data = bytes((b + 1) & 0xff for b in data)
        for row in region['outputs']:
            self.write(row['address'], data[:row['bytes']]); data = data[row['bytes']:]
    def wait(self, *args): pass
    def close(self): pass


def whole(offset, size): return dict(offset=offset, bytes=size, count=1, stride=0)


def staged(root, *, middle=b'\x02\x03\x04\x05', final=b'\x03\x04\x05\x06'):
    """x (4 bytes) -> two column slices into m -> one strided read of m -> f."""
    files = {'x.bin': b'\x01\x02\x03\x04', 'x-negative.bin': b'\x01\x02\x83\x04', 'm.bin': middle, 'f.bin': final}
    for name, data in files.items(): (root / name).write_bytes(data)
    values = [dict(id='x', bytes=4, file='x.bin'), dict(id='m', bytes=4, reference='m.bin'), dict(id='f', bytes=4, reference='f.bin')]
    window = dict(address=BASE, bytes=4096)
    def step(name, source, target, size=2):
        return dict(id=name, kind='device', program='x.bin', window=window,
                    inputs=[dict(id='i', address=BASE, bytes=size, source=dict(value=source[0], range=source[1]))],
                    outputs=[dict(id='o', address=BASE + 64, bytes=size, value=dict(id=target[0], range=target[1]))])
    columns = lambda first: dict(offset=first, bytes=1, count=2, stride=2)
    regions = [step('s0', ('x', columns(0)), ('m', columns(0))), step('s1', ('x', columns(1)), ('m', columns(1))),
               step('s2', ('m', None), ('f', None), size=4)]
    plan = dict(values=values, regions=regions, negative=dict(value='x', file='x-negative.bin'))
    write(root, dict(composed=plan))


def write(root, manifest):
    (root / 'manifest.json').write_text(json.dumps(manifest))
    names = sorted(p.name for p in root.iterdir() if p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(hashlib.sha256((root / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))


class StagedValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'bundle'; self.root.mkdir(); staged(self.root)
        self.logs = Path(self.tmp.name) / 'logs'

    def run_plan(self, **kw):
        return composed.execute(self.root, self.logs, backend='protocol-test', transport_factory=CopyTransport, **kw)

    def edit(self, change):
        manifest = json.loads((self.root / 'manifest.json').read_text()); change(manifest['composed']); write(self.root, manifest)

    def test_slices_bind_into_whole_values_and_strided_reads_follow(self):
        journal = self.run_plan()
        self.assertEqual([r['id'] for r in journal['completed']], ['s0', 's1', 's2'])
        self.assertEqual((self.logs / 's2-output-0.bin').read_bytes(), b'\x03\x04\x05\x06')
        self.assertEqual(journal['completed'][1]['inputs']['inputs:i'], composed.sha(b'\x02\x04'))

    def test_a_wrong_slice_is_a_boundary_mismatch_naming_its_value(self):
        staged(self.root, middle=b'\x02\x03\x04\x06')
        with self.assertRaisesRegex(composed.BoundaryError, r'boundary mismatch: s1/o \(value m\)'): self.run_plan()

    def test_resume_restores_values_from_completed_steps(self):
        original = composed.atomic_json
        def stop(path, value):
            original(path, value)
            if len(value['completed']) == 2 and value['active'] is None: raise RuntimeError('interrupted')
        with patch.object(composed, 'atomic_json', side_effect=stop):
            with self.assertRaises(RuntimeError): self.run_plan()
        journal = self.run_plan(resume=True)
        self.assertEqual([r['id'] for r in journal['completed']], ['s0', 's1', 's2'])
        self.assertEqual((self.logs / 's2-output-0.bin').read_bytes(), b'\x03\x04\x05\x06')

    def test_declared_negative_qualifies_only_as_a_numerical_failure(self):
        result = step_inputs.qualify(self.root, self.logs, backend='protocol-test', transport_factory=CopyTransport)
        self.assertEqual(result['negative'], 'expected numerical failure')
        rejected = json.loads((self.logs / 'negative/journal.json').read_text())['rejected']
        self.assertEqual(rejected['id'], 's0')

    def test_plan_refusals_before_any_transport(self):
        cases = {
            'omitted byte range must cover the whole value': lambda p: p['regions'][2]['inputs'][0].update(bytes=2),
            'value bytes written twice': lambda p: p['regions'][1]['outputs'][0]['value'].update(range=dict(offset=0, bytes=1, count=2, stride=2)),
            'missing producer': lambda p: p['regions'].insert(0, p['regions'].pop(2)),
            'referenced value is never completed': lambda p: p['regions'].pop(1) and p['regions'].pop(1),
            'unaligned': lambda p: [r['window'].update(address=BASE + 64) for r in p['regions']],
            'buffer exceeds step window': lambda p: p['regions'][0]['outputs'][0].update(address=BASE + 4096),
            'negative value must change': lambda p: p['negative'].update(file='x.bin'),
            'negative must replace one file-backed value': lambda p: p['negative'].update(value='m'),
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                staged(self.root); self.edit(change)
                with self.assertRaisesRegex(ValueError, message): composed.load_plan(self.root)

    def test_step_regions_bind_their_images_and_launch_count(self):
        for kind in ('elf', 'application'):
            with self.subTest(kind=kind):
                staged(self.root); self.edit(lambda p: p['regions'][0].update(kind=kind, deployment='x.bin', images=[], elf='x.bin', sources=['y.bin']))
                with self.assertRaisesRegex(ValueError, 'unbound member|launch count|ELF'): composed.load_plan(self.root)

    def test_an_application_window_clear_reloads_its_weights(self):
        """Clearing an application's window erases weights inside it, so they are rewritten after the clear."""
        (self.root / 'w.bin').write_bytes(b'\x05')
        region = dict(id='a', kind='application', sources=['x.bin'], deployment='x.bin', images=[], launches=0,
                      window=dict(address=BASE, bytes=4096), weights=[dict(id='w', address=BASE + 128, bytes=1, file='w.bin')],
                      inputs=[dict(id='i', address=BASE, bytes=1, file='x.bin', range=whole(0, 1))],
                      outputs=[dict(id='o', address=BASE + 64, bytes=1)])
        write(self.root, dict(composed=dict(regions=[region])))
        class Weighted(CopyTransport):
            def launch(self, region):
                self.weight = self.read(BASE + 128, 1); super().launch(region)
        session = composed.Session(self.root, self.logs, backend='protocol-test', transport_factory=Weighted)
        self.addCleanup(session.close)
        self.assertEqual(session.resident, set())
        session.call(self.logs)
        self.assertEqual(session.transport.weight, b'\x05')

    def test_a_window_clear_restores_weights_inside_it_on_every_call(self):
        """Weights inside a step application's window, its own or another region's, are rewritten on every call; others stay resident."""
        for name, data in {'w.bin': b'\x05', 'v.bin': b'\x06', 'u.bin': b'\x07'}.items(): (self.root / name).write_bytes(data)
        x = dict(id='i', address=BASE, bytes=1, file='x.bin', range=whole(0, 1))
        a = dict(id='a', kind='application', sources=['x.bin'], deployment='x.bin', images=[], launches=0,
                 window=dict(address=BASE, bytes=4096), inputs=[x], outputs=[dict(id='o', address=BASE + 64, bytes=1)],
                 weights=[dict(id='w', address=BASE + 128, bytes=1, file='w.bin')])
        b = dict(id='b', kind='device', program='x.bin', inputs=[dict(x, address=BASE + 512)],
                 outputs=[dict(id='o', address=BASE + 576, bytes=1)], weights=[dict(id='v', address=BASE + 640, bytes=1, file='v.bin')])
        c = dict(id='c', kind='device', program='x.bin', inputs=[dict(x, address=BASE + 4096)],
                 outputs=[dict(id='o', address=BASE + 4160, bytes=1)], weights=[dict(id='u', address=BASE + 4224, bytes=1, file='u.bin')])
        class Watched(CopyTransport):
            def __init__(self, *args): super().__init__(); self.seen, self.writes = [], []
            def write(self, address, data): self.writes.append(address); super().write(address, data)
            def launch(self, region):
                self.seen += [(region['id'], self.read(row['address'], 1)) for row in region.get('weights', [])]; super().launch(region)
        packaged = {'a': b'\x05', 'b': b'\x06', 'c': b'\x07'}
        for order in ([a, b, c], [b, a, c]):
            with self.subTest(order=[r['id'] for r in order]):
                write(self.root, dict(composed=dict(regions=order)))
                session = composed.Session(self.root, self.logs, backend='protocol-test', transport_factory=Watched)
                self.addCleanup(session.close)
                self.assertEqual(session.resident, {('c', 'weights', 'u')})
                for call in range(3): session.call(Path(self.tmp.name) / f'{order[0]["id"]}-call-{call}')
                self.assertEqual(session.transport.seen, [(r['id'], packaged[r['id']]) for r in order] * 3)
                self.assertEqual(session.transport.writes.count(BASE + 4224), 1)

    def ranged(self):
        """One region reading bytes 4 and 5 of an eight-byte file, with its output directly after that input."""
        root = Path(self.tmp.name) / 'ranged'; root.mkdir()
        (root / 'r.bin').write_bytes(bytes(range(8))); (root / 'y.bin').write_bytes(b'\x05\x06')
        region = dict(id='s', kind='device', program='r.bin', window=dict(address=BASE, bytes=4096),
                      inputs=[dict(id='i', address=BASE, bytes=2, file='r.bin', range=whole(4, 2))],
                      outputs=[dict(id='o', address=BASE + 2, bytes=2, reference='y.bin')])
        write(root, dict(composed=dict(regions=[region])))
        return root

    def test_a_fallback_negative_flips_only_the_selected_input_bytes(self):
        """Without a declared negative, mutation flips the first gathered byte and writes only the declared input range."""
        root = self.ranged()
        writes, poisoned = [], []
        class Recorded(CopyTransport):
            def write(self, address, data): writes.append((address, bytes(data))); super().write(address, data)
            def launch(self, region): poisoned.append(self.read(BASE + 2, 2)); super().launch(region)
        composed.execute(root, Path(self.tmp.name) / 'positive', backend='protocol-test', transport_factory=Recorded)
        with self.assertRaisesRegex(composed.BoundaryError, 'boundary mismatch: s/o'):
            composed.execute(root, self.logs, backend='protocol-test', transport_factory=Recorded, mutation=True)
        self.assertEqual([data for address, data in writes if address == BASE], [b'\x04\x05', b'\x05\x05'])
        self.assertTrue(all(address + len(data) <= BASE + 4 for address, data in writes))
        self.assertEqual(poisoned, [b'\xa5\xa5'] * 2)
        self.assertEqual((self.logs / 's-output-0.bin').read_bytes(), b'\x06\x06')

    def test_recheck_rebuilds_values_from_saved_outputs_and_refuses_changes(self):
        self.run_plan()
        self.assertEqual(composed.recheck(self.root, self.logs), dict(
            regions=3, completed=3, output_slices=3, referenced_values=2, compared_nonzero=2, differing=[], all_zero_references=[]))
        recheck = [str(self.root), '--composed', '--recheck', '--logs', str(self.logs)]
        with patch('sys.stdout'): self.assertEqual(run_kit.main(recheck), 0)
        journal_path = self.logs / 'journal.json'; journal = json.loads(journal_path.read_text())
        # A saved output whose bytes differ from its journal hash is refused, not compared.
        (self.logs / 's1-output-0.bin').write_bytes(b'\x03\x07')
        with self.assertRaisesRegex(ValueError, 'saved output differs from its journal hash: s1/o'): composed.recheck(self.root, self.logs)
        # A consistent journal and output that differ from the reference are a numerical mismatch of the rebuilt value.
        journal['completed'][1]['outputs']['o']['after'] = composed.sha(b'\x03\x07'); journal_path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(composed.BoundaryError, 'recheck mismatch: value m'): composed.recheck(self.root, self.logs)
        with patch('sys.stdout'): self.assertEqual(run_kit.main(recheck), 1)

    def test_recheck_refuses_a_negative_or_incomplete_run(self):
        step_inputs.qualify(self.root, self.logs, backend='protocol-test', transport_factory=CopyTransport)
        with self.assertRaisesRegex(ValueError, 'not a packaged-input run'): composed.recheck(self.root, self.logs / 'negative')
        journal_path = self.logs / 'positive' / 'journal.json'; journal = json.loads(journal_path.read_text())
        journal['completed'].pop(); journal_path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, 'every region in plan order'): composed.recheck(self.root, self.logs / 'positive')

    def test_a_wrongly_sized_step_input_is_refused_before_any_transport(self):
        root, opened = self.ranged(), []
        oversized = [dict(row=dict(address=BASE, bytes=2), data=bytes(8))]
        with patch.object(step_inputs, 'load_inputs', return_value=oversized):
            with self.assertRaisesRegex(ValueError, 'input bytes differ from its declared range: s/i'):
                composed.execute(root, self.logs, backend='protocol-test', transport_factory=lambda *args: opened.append(args), inputs=root)
        self.assertEqual(opened, [])


class ApplicationSourceTests(unittest.TestCase):
    """A step application is built with the Atlas host runtime only when no device region embeds it."""
    def sources(self, placements):
        from atlas_mlir.chiprunner.composed_sim import _application_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'deployment.json').write_text(json.dumps(dict(calls=[dict(placement=p) for p in placements])))
            (root / 'step.c').write_text('')
            request = dict(sources=[str(root / 'step.c'), str(root / 'step.h')], deployment=str(root / 'deployment.json'),
                           atlas_root=str(root / 'atlas'))
            return [Path(path).relative_to(root).as_posix() for path in _application_sources(request, root / 'shim.c')]

    def test_a_host_only_step_builds_the_atlas_host_runtime(self):
        self.assertEqual(self.sources(['host']), ['step.c', 'shim.c', 'atlas/runtime/atlas_host.c'])

    def test_a_device_step_uses_the_runtime_its_source_embeds(self):
        self.assertEqual(self.sources(['host', 'accelerator']), ['step.c', 'shim.c'])


if __name__ == '__main__':
    unittest.main()
