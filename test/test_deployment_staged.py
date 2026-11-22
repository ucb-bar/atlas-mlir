"""Staged deployment on a synthetic one-step plan: composed plan, composed EXACT gate, and staged kit.

A fake transport stands in for execution (y = x + w bytewise); these tests check plan, gate, and
packaging evidence only and make no numerical claim.
"""
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir import deploy, deployment_gate as gate, deployment_kit as kit
from atlas_mlir.chiprunner import composed, composed_sim
from atlas_mlir.deploy import Refusal, digest

BASE = 0x200c0000
X, W = bytes([3, 0x41, 5, 7]), bytes([10, 20, 30, 40])
Y = bytes(a + b for a, b in zip(X, W))


def whole(offset, size, **extra):
    return dict(offset=offset, bytes=size, count=1, stride=0, **extra)


class AddTransport:
    """y = x + w, read and written at the step's declared addresses."""
    def __init__(self, *args): self.mem = {}
    def write(self, address, data):
        for i, b in enumerate(data): self.mem[address + i] = b
    def read(self, address, size): return bytes(self.mem.get(address + i, 0) for i in range(size))
    def launch(self, region):
        x, w = self.read(BASE, 4), self.read(BASE + 64, 4)
        self.write(BASE + 128, bytes((a + b) & 0xff for a, b in zip(x, w)))
    def wait(self, *args): pass
    def close(self): pass


def elf(entry=0x08000000):
    header = bytearray(64)
    header[:7] = b'\x7fELF\x02\x01\x01'
    struct.pack_into('<HHIQ', header, 16, 2, 243, 1, entry)
    return bytes(composed_sim._payload_elf(header, [(entry, b'\x13\x00\x00\x00', 4, 5)]))


class StagedDeployTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = out = Path(self.tmp.name) / 'out'
        app = out / 'compiled' / 'application'
        files = {
            'deployment.json': dict(schema='merlin.onchip_deployment.v3', mode='bench_staged',
                application_outputs=[dict(value='y', producer_steps=[0])],
                steps=[dict(step=0, package='region_000', placement='host', launches=0,
                    loads=[dict(port='x', address=BASE, bytes=4, source=whole(0, 4, kind='application_input', index=0, value='x')),
                           dict(port='w', address=BASE + 64, bytes=4, source=whole(4, 4, kind='parameter', value='w',
                                path='handoff/parameters.bin', sha256=hashlib.sha256(W).hexdigest()))],
                    reads=[dict(port='y', address=BASE + 128, bytes=4, destination=whole(0, 4, value='y'))])]),
            'source-request.json': dict(nodes=[dict(id=k, type=dict(dtype='u8', shape=[4])) for k in ('x', 'y')], outputs=['y']),
            'steps/region_000/deployment.json': dict(window=dict(address=BASE, bytes=4096), provider=dict(images=[]),
                calls=[dict(placement='host')], hash_ranges=[dict(id='y', address=BASE + 128, bytes=4, role='public_output')]),
            'steps/region_000/handoff/abi.json': dict(source='step.c', header='step.h'),
        }
        for name, value in files.items():
            (app / name).parent.mkdir(parents=True, exist_ok=True)
            (app / name).write_text(json.dumps(value))
        for name in ('step.c', 'step.h', 'deployment_adapter.h'):
            (app / 'steps/region_000/handoff' / name).write_text('/* generated */\n')
        (app / 'handoff').mkdir()
        (app / 'handoff/parameters.bin').write_bytes(bytes(4) + W)
        names = [*files, 'handoff/parameters.bin', *('steps/region_000/handoff/' + n for n in ('step.c', 'step.h', 'deployment_adapter.h'))]
        (app / 'manifest.json').write_text(json.dumps(dict(stage_artifacts={n: digest(app / n) for n in names})))
        (out / 'reference').mkdir()
        np.savez(out / 'reference/expected.npz', output_0=np.frombuffer(Y, dtype=np.uint8))
        (out / 'reference/reference.json').write_text('{}')
        (out / 'capture').mkdir()
        for name in ('capture_receipt.json', 'frontend-trace.json', 'inputs.npz', 'weights.safetensors'):
            (out / 'capture' / name).write_text('{}')
        for name in ('target-rules.json', 'precision-policy.json'):
            (out / name).write_text('{}')
        self.context = dict(out=out, reference=out / 'reference', settings=deploy.clock_settings('100'),
                            objects=dict(hashes={}))
        tensor = SimpleNamespace(dtype='u8', raw=np.frombuffer(X, dtype=np.uint8))
        source_io = 'atlas_mlir_native.reference_eval.source_io.'
        for target, value in ((source_io + 'capture_sources', lambda *a, **k: (out / 'capture', [], {})),
                              (source_io + 'capture_inputs', lambda *a: {'x': tensor}),
                              ('atlas_mlir.deployment_gate._reference', lambda *a: ({}, [])),
                              ('atlas_mlir.deployment_gate._mutated_reference', self.reference)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    absorbed = ()

    def reference(self, context, inputs, destination):
        """Stand-in independent reference, y = x + w, except that changes at `absorbed` byte offsets are absorbed."""
        x = bytearray(np.load(inputs)['input_000'].tobytes())
        for offset in self.absorbed:
            x[offset] = X[offset]
        destination.mkdir()
        np.savez(destination / 'expected.npz', output_0=np.frombuffer(bytes((a + b) & 0xff for a, b in zip(x, W)), dtype=np.uint8))
        (destination / 'reference.json').write_text('{}')
        return gate._archive(destination / 'expected.npz'), destination

    def exact(self, transport=AddTransport):
        with patch.object(gate, '_qualify', partial(gate._qualify, backend='test', transport_factory=transport)):
            return gate.exact_staged(self.context)

    def test_plan_binds_inputs_weights_reference_and_one_bit_negative(self):
        gate.bind_staged(self.context)
        manifest, _, bound = composed.load_plan(self.out / 'bundle')
        plan = manifest['composed']
        self.assertEqual([r['kind'] for r in plan['regions']], ['application'])
        values = {row['id']: row for row in plan['values']}
        self.assertEqual(bound(values['x']['file']).read_bytes(), X)
        self.assertEqual(bound(values['y']['reference']).read_bytes(), Y)
        weight = plan['regions'][0]['weights'][0]
        self.assertEqual(composed.gather(bound(weight['file']).read_bytes(), weight['range'], 4), W)
        # u8 elements are one byte: the first nonzero element is byte 0 (3), and bit 6 of it flips.
        negative = manifest['staged']['negative']
        self.assertEqual({k: negative[k] for k in ('value', 'byte_offset', 'before', 'after', 'xor_mask', 'candidate')},
                         dict(value='x', byte_offset=0, before=3, after=0x43, xor_mask=0x40, candidate=0))
        self.assertEqual([row['result'] for row in negative['tried']], ['changed'])
        self.assertEqual(manifest['deployment']['outputs'][0]['source'], dict(region='step_000_region_000', output='y'))

    def test_negative_absorbed_by_the_reference_takes_the_next_candidate(self):
        # Byte 0 is absorbed, as a trained model absorbs a saturated corner pixel: both of its masks leave the
        # reference unchanged, so the negative is the middle element (byte 2, value 5) with bit 6 flipped.
        self.absorbed = (0,)
        gate.bind_staged(self.context)
        manifest, _, bound = composed.load_plan(self.out / 'bundle')
        negative = manifest['staged']['negative']
        self.assertEqual({k: negative[k] for k in ('byte_offset', 'before', 'after', 'xor_mask', 'candidate')},
                         dict(byte_offset=2, before=5, after=0x45, xor_mask=0x40, candidate=2))
        self.assertEqual([(row['byte_offset'], row['xor_mask'], row['result']) for row in negative['tried']],
                         [(0, 0x40, 'unchanged'), (0, 0x80, 'unchanged'), (2, 0x40, 'changed')])
        self.assertEqual(negative['reference']['changed_outputs'], ['output_0'])
        executed = bound(manifest['composed']['negative']['file']).read_bytes()
        self.assertEqual(executed, bytes([3, 0x41, 0x45, 7]))
        receipt = json.loads(self.exact().receipt.read_text())
        self.assertEqual((receipt['positive']['result'], receipt['negative']['result']), ('EXACT', 'MISMATCH'))

    def test_no_output_changing_candidate_refuses_bind(self):
        self.absorbed = (0, 1, 2, 3)
        with self.assertRaisesRegex(Refusal, 'no candidate input mutation changes an application output in the '
                                    'independent reference; tried byte 0 xor 0x40: unchanged; byte 0 xor 0x80: unchanged; '
                                    'byte 2 xor 0x40: unchanged; byte 2 xor 0x80: unchanged; byte 3 xor 0x40: unchanged; '
                                    'byte 3 xor 0x80: unchanged'):
            gate.bind_staged(self.context)

    def test_staged_outputs_must_follow_the_source_outputs(self):
        request = self.out / 'compiled/application/source-request.json'
        record = json.loads(request.read_text()); record['outputs'] = ['x']
        request.write_text(json.dumps(record))
        with self.assertRaisesRegex(Refusal, 'hash drift'):
            gate.bind_staged(self.context)
        manifest = self.out / 'compiled/application/manifest.json'
        artifacts = json.loads(manifest.read_text())
        artifacts['stage_artifacts']['source-request.json'] = digest(request)
        manifest.write_text(json.dumps(artifacts))
        with self.assertRaisesRegex(Refusal, 'outputs differ from the source outputs'):
            gate.bind_staged(self.context)

    def unused_source_input(self, captured_args=('nx:v0',), request_inputs=('x',)):
        """Source inputs x (loaded) and t (never loaded); y consumes the given captured values and request ids."""
        app = self.out / 'compiled/application'
        request = app / 'source-request.json'
        record = json.loads(request.read_text())
        record['nodes'] = [dict(id='x', op='input', inputs=[], type=dict(dtype='u8', shape=[4])),
                           dict(id='t', op='input', inputs=[], type=dict(dtype='u8', shape=[1])),
                           dict(id='y', op='add', inputs=list(request_inputs), type=dict(dtype='u8', shape=[4]))]
        record['input_order'] = ['x', 't']
        request.write_text(json.dumps(record))
        manifest = json.loads((app / 'manifest.json').read_text())
        manifest['stage_artifacts']['source-request.json'] = digest(request)
        (app / 'manifest.json').write_text(json.dumps(manifest))
        nodes = [dict(id='nx', op='placeholder', args=[]), dict(id='nt', op='placeholder', args=[]),
                 dict(id='ny', op='call_function', args=[list(captured_args)], kwargs={})]
        x = SimpleNamespace(dtype='u8', raw=np.frombuffer(X, dtype=np.uint8))
        t = SimpleNamespace(dtype='u8', raw=np.array([9], dtype=np.uint8))
        source_io = 'atlas_mlir_native.reference_eval.source_io.'
        with patch(source_io + 'capture_sources', lambda *a, **k: (self.out / 'capture', nodes, {})), \
                patch(source_io + 'capture_inputs', lambda *a: {'nx': x, 'nt': t}):
            gate.bind_staged(self.context)

    def test_source_input_unconsumed_in_both_graphs_may_be_left_unloaded(self):
        self.unused_source_input()
        manifest, _, _ = composed.load_plan(self.out / 'bundle')
        self.assertEqual({row['id'] for row in manifest['composed']['values'] if 'file' in row}, {'x'})
        self.assertEqual(manifest['staged']['negative']['value'], 'x')

    def test_unloaded_source_input_consumed_in_either_graph_refuses(self):
        for options in (dict(captured_args=('nx:v0', 'nt:v0')), dict(request_inputs=('x', 't'))):
            with self.subTest(**{key: str(value) for key, value in options.items()}):
                shutil.rmtree(self.out / 'bundle', ignore_errors=True)
                with self.assertRaisesRegex(Refusal, 'source and staged application input indices differ'):
                    self.unused_source_input(**options)

    def test_unloaded_input_proof_checks_the_argument_pairing(self):
        nodes = [dict(id='nx', op='placeholder'), dict(id='nt', op='placeholder'), dict(id='ny', op='call', args=['nx:v0'])]
        request = dict(input_order=['x', 't'], outputs=['y'], nodes=[
            dict(id='x', op='input', inputs=[]), dict(id='t', op='input', inputs=[]), dict(id='y', op='add', inputs=['x'])])
        gate._unconsumed_inputs(nodes, ['nx', 'nt'], request, [(0, 'x')])
        for loaded, edit in (([(0, 't')], None), ([(0, 'x')], lambda r: r.update(input_order=['x'])),
                             ([(0, 'x')], lambda r: r.update(outputs=['y', 't'])),
                             ([(0, 'x')], lambda r: r['nodes'].__setitem__(1, dict(id='t', op='constant', inputs=[])))):
            with self.subTest(loaded=loaded, edited=edit is not None):
                changed = json.loads(json.dumps(request))
                if edit:
                    edit(changed)
                with self.assertRaisesRegex(Refusal, 'indices differ'):
                    gate._unconsumed_inputs(nodes, ['nx', 'nt'], changed, loaded)

    def test_unloaded_input_named_by_a_result_id_or_any_request_field_refuses(self):
        # t's captured result id 'tv' has no 'nt:' prefix, so only the result id itself names it.
        nodes = [dict(id='nx', op='placeholder', results=[dict(id='nx:v0')]),
                 dict(id='nt', op='placeholder', results=[dict(id='tv')]),
                 dict(id='ny', op='call', args=['nx:v0'], kwargs={})]
        request = dict(input_order=['x', 't'], outputs=['y'], nodes=[
            dict(id='x', op='input', inputs=[], attrs={}, index_maps=[], effect='input'),
            dict(id='t', op='input', inputs=[], attrs={}, index_maps=[], effect='input'),
            dict(id='y', op='add', inputs=['x'], attrs={}, index_maps=[], effect='pure')])
        gate._unconsumed_inputs(nodes, ['nx', 'nt'], request, [(0, 'x')])
        edits = dict(
            bare_value_id=lambda n, r: n[2]['args'].append('tv'),
            kwargs_value_id=lambda n, r: n[2]['kwargs'].update(scale=dict(value_id='tv')),
            kwargs_node_and_value_id=lambda n, r: n[2]['kwargs'].update(scale=dict(node_id='nt', value_id='tv')),
            request_attrs=lambda n, r: r['nodes'][2]['attrs'].update(scale=['t']),
            request_index_maps=lambda n, r: r['nodes'][2]['index_maps'].append(dict(value='t')),
            request_effect=lambda n, r: r['nodes'][2].update(effect='t'))
        for label, edit in edits.items():
            with self.subTest(label):
                changed_nodes, changed = json.loads(json.dumps(nodes)), json.loads(json.dumps(request))
                edit(changed_nodes, changed)
                with self.assertRaisesRegex(Refusal, 'indices differ'):
                    gate._unconsumed_inputs(changed_nodes, ['nx', 'nt'], changed, [(0, 'x')])

    def test_unloaded_input_with_an_unprovable_reference_refuses(self):
        # Each edit leaves t named somewhere the old scan missed, or makes its ids unmatchable, so the proof fails.
        nodes = [dict(id='nx', op='placeholder', results=[dict(id='nx:v0')]),
                 dict(id='nt', op='placeholder', results=[dict(id='tv')]),
                 dict(id='ny', op='call', args=['nx:v0'], kwargs={})]
        request = dict(input_order=['x', 't'], outputs=['y'], input_storages={'x': {}, 't': {}}, nodes=[
            dict(id='x', op='input', inputs=[], attrs={}, index_maps=[], effect='input'),
            dict(id='t', op='input', inputs=[], attrs={}, index_maps=[], effect='input'),
            dict(id='y', op='add', inputs=['x'], attrs={}, index_maps=[], effect='pure')])
        gate._unconsumed_inputs(nodes, ['nx', 'nt'], request, [(0, 'x')])

        def int_result_id(n, r):
            n[1]['results'][0]['id'] = 7
            n[2]['kwargs'].update(scale=dict(value_id=7))

        def int_node_id(n, r):
            n[1]['id'] = 7
            n[2]['args'].append(7)

        def int_request_input(n, r):
            r['input_order'][1] = r['nodes'][1]['id'] = 7
            r['nodes'][2]['inputs'].append(7)

        def duplicate_captured_id(n, r):
            # A second node with t's id hides t's real result id tv from a by-id lookup.
            n.insert(2, dict(id='nt', op='call', args=['nx:v0'], kwargs={}, results=[dict(id='other')]))
            n[3]['args'].append('tv')

        edits = dict(
            int_result_id=int_result_id, int_node_id=int_node_id, int_request_input=int_request_input,
            result_not_a_dict=lambda n, r: (n[1].update(results=['tv']), n[2]['args'].append('tv')),
            duplicate_captured_id=duplicate_captured_id,
            duplicate_request_id=lambda n, r: r['nodes'].append(dict(id='t', op='add', inputs=['x'], attrs={})),
            captured_kwargs_key=lambda n, r: n[2]['kwargs'].update(table={'tv': 1}),
            captured_other_field=lambda n, r: n[2].update(operands=['tv']),
            request_attrs_key=lambda n, r: r['nodes'][2]['attrs'].update(t=1),
            request_output_storages=lambda n, r: r.update(output_storages={'t': {}}),
            request_boundary_layouts=lambda n, r: r.update(boundary_layouts=[dict(value='t')]),
            request_constants=lambda n, r: r.update(constants=[dict(node_id='t')]))
        for label, edit in edits.items():
            with self.subTest(label):
                changed_nodes, changed = json.loads(json.dumps(nodes)), json.loads(json.dumps(request))
                edit(changed_nodes, changed)
                source_ids = [node['id'] for node in changed_nodes if node['op'] == 'placeholder']
                with self.assertRaisesRegex(Refusal, 'indices differ'):
                    gate._unconsumed_inputs(changed_nodes, source_ids, changed, [(0, 'x')])

    def test_every_output_reference_must_be_nonzero(self):
        # A second application output z whose reference is all zero, beside the nonzero y.
        app = self.out / 'compiled/application'
        edits = {'deployment.json': lambda d: (d['application_outputs'].append(dict(value='z', producer_steps=[0])),
                                               d['steps'][0]['reads'].append(dict(port='z', address=BASE + 192, bytes=4,
                                                                                  destination=whole(0, 4, value='z')))),
                 'source-request.json': lambda d: (d['nodes'].append(dict(id='z', type=dict(dtype='u8', shape=[4]))),
                                                   d['outputs'].append('z'))}
        manifest = json.loads((app / 'manifest.json').read_text())
        for name, edit in edits.items():
            record = json.loads((app / name).read_text())
            edit(record)
            (app / name).write_text(json.dumps(record))
            manifest['stage_artifacts'][name] = digest(app / name)
        (app / 'manifest.json').write_text(json.dumps(manifest))
        np.savez(self.out / 'reference/expected.npz', output_0=np.frombuffer(Y, dtype=np.uint8), output_1=np.zeros(4, np.uint8))
        with self.assertRaisesRegex(Refusal, 'all-zero application output reference refuses coverage: z'):
            gate.bind_staged(self.context)

    def test_gate_passes_only_with_exact_positive_and_rejected_negative(self):
        gate.bind_staged(self.context)
        result = self.exact()
        self.assertIsInstance(result, deploy.Gate)
        receipt = json.loads(result.receipt.read_text())
        self.assertEqual((receipt['positive']['result'], receipt['negative']['result']), ('EXACT', 'MISMATCH'))
        self.assertTrue(receipt['negative']['errors'][0].startswith('boundary mismatch: step_000_region_000/y'))
        result.receipt.write_text(json.dumps(dict(receipt, status='FAIL')))
        with self.assertRaisesRegex(Refusal, 'hash drift'):
            result.verify()

    def test_wrong_output_refuses_the_gate(self):
        gate.bind_staged(self.context)

        class Wrong(AddTransport):
            def launch(self, region): self.write(BASE + 128, bytes(4))
        with self.assertRaisesRegex(Refusal, 'composed EXACT refused'):
            self.exact(Wrong)

    def test_negative_that_changes_no_output_refuses_the_gate(self):
        gate.bind_staged(self.context)

        class Constant(AddTransport):
            def launch(self, region): self.write(BASE + 128, Y)
        with self.assertRaisesRegex(Refusal, 'composed EXACT refused: input mutation'):
            self.exact(Constant)

    def package(self, steps):
        gate.bind_staged(self.context)
        self.context.update(gate=self.exact(), steps=steps)
        kit.package_staged(self.context)
        return composed.load_plan(self.out / 'kit')

    def test_kit_replaces_the_step_application_with_its_linked_elf(self):
        path = self.out / 'steps/region_000/chip.elf'
        path.parent.mkdir(parents=True)
        path.write_bytes(elf())
        (path.parent / 'link.json').write_text('{}')
        manifest, members, _ = self.package({'region_000': dict(elf=path, record=dict(
            sha256=digest(path), entry=0x08000000, clock_hz=100000000))})
        region = manifest['composed']['regions'][0]
        self.assertEqual((region['kind'], region['elf'], 'sources' in region), ('elf', 'steps/region_000/chip.elf', False))
        self.assertEqual(manifest['composed']['receiver_baud'], 115200)
        self.assertIn('qualification/composed-check.json', members)
        self.assertNotIn('application/steps/region_000/handoff/step.c', members)

    def test_a_blocked_step_keeps_its_application(self):
        manifest, members, _ = self.package({'region_000': dict(blocked='code reservation exceeded')})
        self.assertEqual(manifest['composed']['regions'][0]['kind'], 'application')
        self.assertIn('application/steps/region_000/handoff/step.c', members)

    def test_changed_plan_member_refuses_packaging(self):
        gate.bind_staged(self.context)
        self.context.update(gate=self.exact(), steps={'region_000': dict(blocked='reason')})
        (self.out / 'bundle/reference/y.bin').write_bytes(bytes(4))
        with self.assertRaisesRegex(Refusal, 'hash drift'):
            kit.package_staged(self.context)


if __name__ == '__main__':
    unittest.main()
