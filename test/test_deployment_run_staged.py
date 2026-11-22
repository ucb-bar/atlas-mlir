"""`atlasc run --inputs` and `atlasc reference` on staged kits, whose manifests declare only untyped outputs.

The tensor rows come from the step deployment records the kit already inventories; existing kits keep their staged
runner contract. Offline: simulated plans over a test transport, never hardware.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir import deployment_run as runner
from atlas_mlir import deployment_stages
from atlas_mlir.chiprunner import composed, composed_sim
from atlas_mlir.deploy import Refusal

BASE = 0x200c0000
X = b'\x01\x02\x03\x04'
U8 = dict(dtype='u8', numerical_policy='raw-bit-pattern')


class AddTransport:
    """Each step writes its input bytes plus one to its output."""
    def __init__(self, *args): self.mem = {}
    def write(self, address, data):
        for i, b in enumerate(data): self.mem[address + i] = b
    def read(self, address, size): return bytes(self.mem.get(address + i, 0) for i in range(size))
    def launch(self, region):
        row, = region['inputs']
        data = bytes((b + 1) & 0xff for b in self.read(row['address'], row['bytes']))
        for out in region['outputs']: self.write(out['address'], data[:out['bytes']])
    def wait(self, *args): pass
    def close(self): pass


def plus(data, n): return bytes((b + n) & 0xff for b in data)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(kit):
    files = sorted(p for p in kit.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (kit / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + str(p.relative_to(kit)) + '\n' for p in files))


def record(out, **change):
    """An evaluator-shaped record whose decision inputs are named by absolute paths under `out`, as the evaluator names them."""
    identities = {str(out / 'compiled/application/manifest.json'): '1' * 64, str(out / 'capture/capture_receipt.json'): '2' * 64}
    return dict(dict(status='EXPECTED', mode='source_capture', source_trace_sha256='3' * 64,
                     compiler_decision_inputs=dict(schema='atlas.source_selected_decisions.v1', identities=identities)), **change)


def staged_out(out, *, second_reader=False):
    """A staged OUT as bind packages it: x (u8[4], file-backed) -> s0 -> m -> s1 -> f (deployment output 0, u8[2, 2]).

    The manifest's deployment declares only f's untyped range; each step's deployment record types its ranges; the
    deployment's own reference/reference.json is named by digest in the manifest, and the kit packages no record.
    """
    kit = out / 'kit'
    from atlas_mlir.deployment_kit import owned_tools
    (kit / 'bench').mkdir(parents=True)
    for name, source in owned_tools().items():
        (kit / 'bench' / name).write_bytes(source.read_bytes())
    window = dict(address=BASE, bytes=4096)
    files = {'inputs/x.bin': X, 'negative/x.bin': b'\x01\x02\x83\x04', 'reference/f.bin': plus(X, 2), 'program.bin': b'stub'}
    for name, data in files.items():
        (kit / name).parent.mkdir(parents=True, exist_ok=True)
        (kit / name).write_bytes(data)
    def step(name, source, target, typed=True):
        row_in = dict(id='i', address=BASE, bytes=4, source=dict(value=source, range=dict(offset=0, bytes=4, count=1, stride=0)))
        row_out = dict(id='o', address=BASE + 64, bytes=4, value=dict(id=target, range=dict(offset=0, bytes=4, count=1, stride=0)))
        deployment = 'application/steps/%s/deployment.json' % name
        rows = dict(inputs=[dict(id='i', index=0, address=BASE, bytes=4, type=dict(U8, shape=[4]))] if typed else [],
                    outputs=[dict(id='o', index=0, address=BASE + 64, bytes=4, type=dict(U8, shape=[2, 2]))])
        (kit / deployment).parent.mkdir(parents=True)
        (kit / deployment).write_text(json.dumps(rows))
        return dict(id=name, kind='device', program='program.bin', window=window, deployment=deployment, inputs=[row_in], outputs=[row_out])
    regions = [step('s0', 'x', 'm'), step('s1', 'm', 'f')]
    regions[1]['outputs'][0]['reference'] = 'reference/f.bin'
    values = [dict(id='x', bytes=4, file='inputs/x.bin'), dict(id='m', bytes=4), dict(id='f', bytes=4)]
    if second_reader:
        regions.append(step('s2', 'x', 'g')); values.append(dict(id='g', bytes=4))
    (out / 'reference').mkdir()
    (out / 'reference/reference.json').write_text(json.dumps(record(out)))
    manifest = dict(runner=dict(schema='atlas.kit.v1', record_tag='ATLAS'),
                    composed=dict(values=values, regions=regions, negative=dict(value='x', file='negative/x.bin')),
                    deployment=dict(outputs=[dict(index=0, id='f', address=BASE + 64, bytes=4, source=dict(region='s1', output='o'))]),
                    staged=dict(reference_sha256=digest(out / 'reference/reference.json')))
    (kit / 'manifest.json').write_text(json.dumps(manifest))
    inventory(kit)
    (out / 'deploy-status.json').write_text(json.dumps(dict(status='PACKAGED', transport='jtag')))
    return manifest


def supplied(out, directory, x, *, made_in=None, **change):
    """The reference `atlasc reference` makes for input bytes `x` (from an OUT at `made_in`, default `out`)."""
    directory.mkdir()
    np.savez(directory / 'expected.npz', output_0=np.frombuffer(plus(x, 2), np.uint8).reshape(2, 2))
    body = record(made_in or out, inputs=dict(input_000=hashlib.sha256(x).hexdigest()),
                  expected_sha256=digest(directory / 'expected.npz'), **change)
    (directory / 'reference.json').write_text(json.dumps(body))
    return directory


class StagedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.out = self.work / 'out'
        staged_out(self.out)
        stack = contextlib.ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(composed_sim, 'SimTransport', AddTransport))
        stack.enter_context(patch.object(composed_sim, 'runtime_identity', return_value={}))

    def frame(self, data, dtype=np.uint8, shape=(4,)):
        path = self.work / ('frame-%d.npz' % len(list(self.work.glob('frame-*.npz'))))
        np.savez(path, input_000=np.frombuffer(data, np.uint8).astype(dtype).reshape(shape))
        return path

    def run_front_door(self, *args):
        from atlas_mlir import cli
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rc = cli.main(['run', str(self.out), *args])
        return rc, output.getvalue()

    def test_simulated_frames_run_and_are_checked_against_supplied_and_packaged_references(self):
        f1, f2 = self.frame(b'\x05\x06\x07\x08'), self.frame(X)
        r1 = supplied(self.out, self.work / 'R1', b'\x05\x06\x07\x08')
        r2 = supplied(self.out, self.work / 'R2', X)
        rc, output = self.run_front_door('--simulate', '--inputs', str(f1), str(f2), '--reference', str(r1), str(r2))
        lines = output.splitlines()
        self.assertEqual(rc, 0, output)
        self.assertEqual([line.split(' outputs=')[0] for line in lines],
                         [f'STEP 000 EXACT ({r1} reference)', f'STEP 001 EXACT ({r2} reference)', lines[2]])
        self.assertTrue(lines[2].startswith('RUN SIM PASS logs='))
        logs = Path(lines[2].split('logs=')[1])
        self.assertEqual((logs / 'step-000/output-0.bin').read_bytes(), b'\x07\x08\x09\x0a')
        self.assertEqual((logs / 'step-001/output-0.bin').read_bytes(), plus(X, 2))
        with np.load(logs / 'step-000/outputs.npz') as saved:
            self.assertEqual(saved['output_000'].tolist(), [[7, 8], [9, 10]])
        journal = json.loads((logs / 'step-000/journal.json').read_text())
        self.assertEqual(journal['identity']['step_inputs'], {'s0/i': hashlib.sha256(b'\x05\x06\x07\x08').hexdigest()})
        self.assertEqual(journal['completed'][0]['inputs']['inputs:i'], hashlib.sha256(b'\x05\x06\x07\x08').hexdigest())
        # One pass per frame after the session's reference-open, the positive that compares every boundary; the
        # negative is the kit's own run (deploy ran it), and checked frames need no closing repeat.
        self.assertEqual(sorted(path.parent.name for path in logs.glob('**/journal.json')), ['reference-open', 'step-000', 'step-001'])
        self.assertEqual((logs / 'reference-open/output-0.bin').read_bytes(), plus(X, 2))

    def test_one_session_per_frame_is_two_passes_each(self):
        for data in (b'\x05\x06\x07\x08', b'\x09\x09\x09\x09'):
            reference = supplied(self.out, self.work / ('R-%d' % data[0]), data)
            rc, output = self.run_front_door('--simulate', '--inputs', str(self.frame(data)), '--reference', str(reference))
            self.assertEqual(rc, 0, output)
            logs = Path(output.splitlines()[-1].split('logs=')[1])
            self.assertEqual(sorted(path.parent.name for path in logs.glob('**/journal.json')), ['reference-open', 'step-000'])

    def test_a_kit_whose_packaged_check_fails_runs_no_frame(self):
        (self.out / 'kit/reference/f.bin').write_bytes(plus(X, 3)); inventory(self.out / 'kit')
        frame = self.frame(b'\x05\x06\x07\x08')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, 'boundary mismatch: s1/o'):
            runner.execute(self.out, simulate=True, inputs=[frame], references=[supplied(self.out, self.work / 'R1', b'\x05\x06\x07\x08')])
        logs, = self.out.glob('run-*')
        self.assertFalse((logs / 'step-000').exists())

    def test_without_frames_simulation_still_runs_the_plan_and_its_negative(self):
        rc, output = self.run_front_door('--simulate')
        self.assertEqual(rc, 0, output)
        logs = Path(output.splitlines()[-1].split('logs=')[1])
        self.assertEqual(sorted(path.parent.name for path in logs.glob('**/journal.json')), ['negative', 'positive'])

    def test_packaged_frame_is_exact_without_a_supplied_reference_and_others_are_unchecked(self):
        rc, output = self.run_front_door('--simulate', '--inputs', str(self.frame(b'\x09\x09\x09\x09')), str(self.frame(X)))
        self.assertEqual(rc, 0, output)
        self.assertEqual([line.split(' outputs=')[0] for line in output.splitlines()[:2]],
                         ['STEP 000 UNCHECKED (no independent reference)', 'STEP 001 EXACT (packaged reference)'])

    def test_dry_run_validates_staged_frames_and_references(self):
        rc, output = self.run_front_door('--dry-run', '--inputs', str(self.frame(b'\x05\x06\x07\x08')),
                                         '--reference', str(supplied(self.out, self.work / 'R1', b'\x05\x06\x07\x08')))
        self.assertEqual(rc, 0, output)
        self.assertIn('RUN DRY PASS', output)

    def test_wrong_frames_and_references_refuse_before_any_session(self):
        cases = ((self.frame(X[:2], shape=(2,)), None, 'shape or dtype mismatch: input_000'),
                 (self.frame(X, dtype=np.int8), None, 'shape or dtype mismatch'),
                 (self.frame(b'\x05\x06\x07\x08'), supplied(self.out, self.work / 'other', X), 'made from other inputs'),
                 (self.frame(X), supplied(self.out, self.work / 'source', X, source_trace_sha256='4' * 64), "not made from this kit's source"))
        for frame, reference, message in cases:
            with self.subTest(message=message), patch.object(composed, 'Session') as session:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, message):
                    runner.execute(self.out, simulate=True, inputs=[frame], references=[reference] if reference else ())
                session.assert_not_called()

    def test_reference_made_against_a_copied_out_is_accepted_and_relocation_keeps_digests(self):
        copy = self.work / 'bench' / 'kit-out'
        rc, output = self.run_front_door('--dry-run', '--inputs', str(self.frame(b'\x05\x06\x07\x08')), '--reference',
                                         str(supplied(self.out, self.work / 'R1', b'\x05\x06\x07\x08', made_in=copy)))
        self.assertEqual(rc, 0, output)
        changed = record(copy)['compiler_decision_inputs']
        changed['identities'][str(copy / 'capture/capture_receipt.json')] = '5' * 64
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, "not made from this kit's source"):
            runner.execute(self.out, dry_run=True, inputs=[self.frame(X)],
                           references=[supplied(self.out, self.work / 'R2', X, compiler_decision_inputs=changed)])

    def test_staged_reference_record_must_be_the_one_the_manifest_names(self):
        (self.out / 'reference/reference.json').write_text(json.dumps(record(self.out, mode='other')))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, 'differs from the one the staged kit names'):
            runner.execute(self.out, dry_run=True, inputs=[self.frame(X)], references=[supplied(self.out, self.work / 'R1', X)])

    def test_an_input_read_by_two_steps_or_untyped_refuses(self):
        for name, build, message in (('two', dict(second_reader=True), 'read by 2 steps'),):
            out = self.work / name
            staged_out(out, **build)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, message):
                runner.execute(out, dry_run=True, inputs=[self.frame(X)])
        out = self.work / 'untyped'
        manifest = staged_out(out)
        (out / 'kit' / manifest['composed']['regions'][0]['deployment']).write_text(json.dumps(dict(inputs=[], outputs=[])))
        inventory(out / 'kit')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, 'has no type in the deployment record of step s0'):
            runner.execute(out, dry_run=True, inputs=[self.frame(X)])

    def test_staged_plan_view_types_rows_and_changes_only_the_reader(self):
        from atlas_mlir.chiprunner import composed
        manifest, _, bound = composed.load_plan(self.out / 'kit')
        view = runner.staged_view(manifest, bound)
        self.assertEqual(view['deployment']['inputs'], [dict(index=0, id='x', address=BASE, bytes=4, type=dict(U8, shape=[4]))])
        self.assertEqual(view['deployment']['outputs'][0]['type'], dict(U8, shape=[2, 2]))
        self.assertEqual(view['composed']['regions'][0]['inputs'], [dict(id='i', address=BASE, bytes=4, file='inputs/x.bin')])
        self.assertEqual(view['composed']['regions'][1:], manifest['composed']['regions'][1:])
        self.assertEqual(manifest['deployment'], json.loads((self.out / 'kit/manifest.json').read_text())['deployment'])
        self.assertIs(composed.load_plan.__module__, composed.__name__)  # the view is never left installed


class StagedReferenceTests(unittest.TestCase):
    """`atlasc reference OUT --inputs FILE -o DIR` on a staged kit, with the evaluator stood in."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.out = self.work / 'out'
        staged_out(self.out)
        (self.out / 'capture').mkdir()
        (self.out / 'capture/capture_receipt.json').write_text('{}')
        source_io = 'atlas_mlir_native.reference_eval.source_io.'
        tensor = SimpleNamespace(dtype='u8', raw=np.frombuffer(X, np.uint8))
        for target, value in ((source_io + 'capture_sources', lambda *a, **k: (None, [], {})),
                              (source_io + 'capture_inputs', lambda *a: {'x': tensor}),
                              ('atlas_mlir.deployment_stages._evaluate', self.evaluate)):
            patcher = patch(target, value); patcher.start(); self.addCleanup(patcher.stop)

    def evaluate(self, out, destination, log, inputs=None):
        x = np.load(inputs)['input_000'].tobytes()
        supplied(out, destination, x)

    def frame(self, data):
        path = self.work / 'frame.npz'
        np.savez(path, input_000=np.frombuffer(data, np.uint8))
        return path

    def test_reference_for_a_staged_kit_is_made_and_accepted_by_run(self):
        destination = deployment_stages.new_inputs(self.out, self.frame(b'\x05\x06\x07\x08'), self.work / 'R1')
        self.assertEqual(json.loads((destination / 'reference.json').read_text())['inputs'],
                         dict(input_000=hashlib.sha256(b'\x05\x06\x07\x08').hexdigest()))

    def test_reference_refuses_a_kit_whose_packaged_input_is_not_the_source_input(self):
        tensor = SimpleNamespace(dtype='u8', raw=np.frombuffer(b'\x09\x09\x09\x09', np.uint8))
        with patch('atlas_mlir_native.reference_eval.source_io.capture_inputs', lambda *a: {'x': tensor}), \
                self.assertRaisesRegex(Refusal, 'not the source user inputs in order'):
            deployment_stages.new_inputs(self.out, self.frame(X), self.work / 'R1')


if __name__ == '__main__':
    unittest.main()
