"""Per-step byte routing tests use explicit test transports, never hardware."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from atlas_mlir.chiprunner import composed, step_inputs, run_kit, preflight, check_rung
from atlas_mlir.chiprunner.kit_manifest import load
from atlas_mlir.chiprunner.tcl_rpc import TransportError
from test_chiprunner import synthetic, inventory
from test_chiprunner_composed import fixture, MemoryTransport, inventory as composed_inventory


def composed_kit(root):
    """Composed fixture with one i8 observation and one i8 action."""
    fixture(root)
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['deployment'] = dict(window=dict(address=0x200c0000, bytes=64), inputs=[
        dict(id='observation', index=0, address=0x200c0000, bytes=1, type=dict(dtype='i8', shape=[1]))],
        outputs=[dict(id='action', index=0, address=0x200c0002, bytes=1, type=dict(dtype='i8', shape=[1]),
                      source=dict(region='b', output='y'))])
    (root/'manifest.json').write_text(json.dumps(manifest))
    composed_inventory(root)
    return manifest


class CountingTransport(MemoryTransport):
    def __init__(self, *args):
        super().__init__(*args); self.writes = []
    def write(self, address, data):
        self.writes.append(address); super().write(address, data)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.root = self.work/'bundle'; self.root.mkdir()
        self.manifest = composed_kit(self.root)
        self.inputs = self.work/'inputs.npz'
        MemoryTransport.instances = []

    def save(self, value, dtype=np.int8):
        np.savez(self.inputs, input_000=np.array([value], dtype=dtype))

    def test_session_keeps_weights_resident_and_returns_new_outputs(self):
        session = composed.Session(self.root, self.work/'session', backend='protocol-test',
                                   transport_factory=CountingTransport)
        self.addCleanup(session.close)
        for index, value in enumerate((-3, 7)):
            self.save(value)
            logs = self.work/('step'+str(index))
            journal = session.call(logs, inputs=self.inputs)
            expected = bytes([(value % 256 * 3 + 1) % 256])
            self.assertEqual((logs/'output-0.bin').read_bytes(), expected)
            self.assertEqual((logs/'b-output-0.bin').read_bytes(), expected)
            self.assertEqual(journal['completed'][0]['inputs']['inputs:x'], composed.sha(bytes([value % 256])))
        transport, = MemoryTransport.instances
        self.assertEqual(transport.seen, [b'\xfd', b'\xf7', b'\x07', b'\x15'])
        self.assertEqual(transport.writes.count(0x200c0001), 1)

    def test_shared_weight_ranges_are_rewritten_each_call(self):
        self.manifest['composed']['regions'][1]['weights'] = [dict(id='w', address=0x200c0001, bytes=1, file='weight.bin')]
        (self.root/'manifest.json').write_text(json.dumps(self.manifest)); composed_inventory(self.root)
        session = composed.Session(self.root, self.work/'session', backend='protocol-test',
                                   transport_factory=CountingTransport)
        self.addCleanup(session.close)
        for index in range(2):
            self.save(4); session.call(self.work/('step'+str(index)), inputs=self.inputs)
        self.assertEqual(MemoryTransport.instances[0].writes.count(0x200c0001), 4)

    def test_bundle_change_during_session_refuses(self):
        session = composed.Session(self.root, self.work/'session', backend='protocol-test',
                                   transport_factory=MemoryTransport)
        self.addCleanup(session.close)
        self.save(4); session.call(self.work/'step0', inputs=self.inputs)
        (self.root/'input.bin').write_bytes(b'\x05'); composed_inventory(self.root)
        with self.assertRaisesRegex(ValueError, 'bundle changed'):
            session.call(self.work/'step1', inputs=self.inputs)

    def test_deployment_outputs_must_name_region_output_ranges(self):
        for change in (dict(address=0x200c0003), dict(source=dict(region='b', output='z')), dict(index=1)):
            row = dict(self.manifest['deployment']['outputs'][0], **change)
            manifest = json.loads(json.dumps(self.manifest)); manifest['deployment']['outputs'] = [row]
            (self.root/'manifest.json').write_text(json.dumps(manifest)); composed_inventory(self.root)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'deployment output'):
                composed.load_plan(self.root)

    def test_shapes_dtypes_sizes_extra_members_and_missing_inputs_fail(self):
        dep = self.manifest['deployment']
        for array in (np.array([4], dtype=np.uint8), np.array([4], dtype=np.int32), np.array([[4]], dtype=np.int8)):
            np.savez(self.inputs, input_000=array)
            with self.assertRaisesRegex(ValueError, 'shape or dtype'):
                step_inputs.load_inputs(dep, self.inputs)
        self.save(-4)
        self.assertEqual(step_inputs.load_inputs(dep, self.inputs)[0]['data'], b'\xfc')
        dep['inputs'][0]['bytes'] = 2
        with self.assertRaisesRegex(ValueError, 'byte range'):
            step_inputs.load_inputs(dep, self.inputs)
        dep['inputs'][0]['bytes'] = 1
        for members in (dict(input_000=np.int8([4]), input_001=np.int8([4])), dict(observation=np.int8([4])), {}):
            np.savez(self.inputs, **members)
            with self.subTest(members=sorted(members)), self.assertRaisesRegex(ValueError, 'unexpected or missing'):
                step_inputs.load_inputs(dep, self.inputs)
        np.save(self.work/'input_000.npy', np.int8([4]))
        with self.assertRaisesRegex(ValueError, 'one .npz archive'):
            step_inputs.load_inputs(dep, self.work/'input_000.npy')
        self.inputs.unlink()
        with self.assertRaises(FileNotFoundError): step_inputs.load_inputs(dep, self.inputs)

    def test_bf16_storage_bits_and_float32_dtype(self):
        dep = dict(window=dict(address=0x200c0000, bytes=64), inputs=[
            dict(id='x', index=0, address=0x200c0000, bytes=4,
                 type=dict(dtype='bf16', shape=[2]))])
        np.savez(self.inputs, input_000=np.array([0x3f80, 0x4000], dtype=np.uint16))
        self.assertEqual(step_inputs.load_inputs(dep, self.inputs)[0]['data'], b'\x80\x3f\x00\x40')
        dep['inputs'][0].update(bytes=8, type=dict(dtype='f32', shape=[2]))
        np.savez(self.inputs, input_000=np.array([1, 2], dtype=np.float32))
        self.assertEqual(len(step_inputs.load_inputs(dep, self.inputs)[0]['data']), 8)

    def test_resume_binds_inputs(self):
        logs = self.work/'step'
        self.save(4)
        composed.execute(self.root, logs, inputs=self.inputs, backend='protocol-test', transport_factory=MemoryTransport)
        self.save(5)
        with self.assertRaisesRegex(ValueError, 'stale'):
            composed.execute(self.root, logs, inputs=self.inputs, resume=True,
                             backend='protocol-test', transport_factory=MemoryTransport)

    def test_simulation_qualifies_executed_mutation_and_transport_failures_refuse(self):
        result = step_inputs.qualify(self.root, self.work/'ladder',
            backend='protocol-test', transport_factory=MemoryTransport)
        self.assertEqual(result['negative'], 'expected numerical failure')
        self.assertTrue((self.work/'ladder/negative/a-output-0.bin').is_file())
        with patch.object(MemoryTransport, 'wait', side_effect=TimeoutError('deadline')):
            with self.assertRaises(TimeoutError):
                step_inputs.qualify(self.root, self.work/'timeout',
                    backend='protocol-test', transport_factory=MemoryTransport)

    def test_missing_simulator_dependency_is_failure(self):
        from atlas_mlir.chiprunner.composed_sim import runtime_identity
        with patch.dict(os.environ, {'ATLAS_NPU_MODEL_ROOT': '', 'NPU_MODEL_ROOT': ''}):
            with self.assertRaisesRegex(ValueError, 'required'): runtime_identity()


def live_kit(root):
    """Synthetic packaged kit with one i8 input and a qualified byte patch."""
    manifest = synthetic(root)
    dep = manifest['deployment']
    dep['inputs'] = [dict(id='x', index=0, address=0x200c0020, bytes=1, type=dict(dtype='i8', shape=[1]))]
    for row in dep['outputs']: row['type'] = dict(dtype='u8', shape=[2, 2])
    dep['allocations'].append(dict(name='input:x', role='public_input', address=0x200c0020, bytes=1))
    manifest['runner']['negative_patch'] = dict(address=0x200c0020, before_byte=1, after_byte=129)
    rewrite(root, manifest)
    return manifest


def rewrite(root, manifest):
    (root/'qualification').mkdir(exist_ok=True)
    (root/'qualification/check.json').write_text(json.dumps(dict(byte_flip=manifest['runner']['negative_patch'])))
    for name in ['manifest.json','images/manifest.json','mutation/manifest.json']:
        (root/name).write_text(json.dumps(manifest))
    (root/'bound/layout.json').write_text(json.dumps(manifest['deployment']))
    inventory(root)


def peers(manifest, *, status=0, write_outputs=True, fail_dump_after=None, corrupt=None):
    """In-memory Tcl, UART, and OpenOCD peers. Not hardware or ELF evidence.

    On resume the peer computes outputs from the input byte at 0x200c0020 and
    the resident parameter byte at 0x200c0030, so a reload or a lost window
    shows up in the bytes. A `load_image` at an address in `corrupt` leaves those bytes instead.
    """
    memory, log = {}, []

    class RPC:
        broken = False
        timeout = 1
        def __init__(self, **kwargs): pass
        def command(self, text):
            log.append(text)
            match = re.fullmatch(r'(load_image|dump_image) \{([^}]+)\}(?: (0x[0-9a-f]+) (bin|[0-9]+))?', text)
            if match and match[3]:
                op, name, address, tail = match.groups(); address = int(address, 16)
                if op == 'load_image':
                    for i, byte in enumerate((corrupt or {}).get(address, Path(name).read_bytes())): memory[address+i] = byte
                else:
                    if fail_dump_after is not None and sum(x.startswith('resume') for x in log) > fail_dump_after:
                        self.broken = True
                        raise TransportError('dump connection lost')
                    Path(name).write_bytes(bytes(memory[address+i] for i in range(int(tail))))
            if text.startswith('resume ') and write_outputs:
                start = memory[0x200c0020]
                first = 9 if start == 129 else start * memory[0x200c0030] % 256
                for i, byte in enumerate(bytes([first, 2, 3, 4, 5, 6, 7, 8])):
                    memory[0x200c0000 + (i//4)*8 + i%4] = byte
            return ''
        def close(self): pass

    class UART:
        def __init__(self, *args): pass
        def clear(self): pass
        def close(self): pass
        def records(self, timeout, console, tag):
            data = '%s STATUS %d CYCLES 9\n' % (tag, status)
            if status:
                console.write(data.encode()); raise run_kit.NumericalError('UART reported nonzero status')
            for row in manifest['deployment']['outputs']:
                raw = bytes(memory[row['address']+i] for i in range(row['bytes']))
                data += '%s HASH %s %s %s\n' % (tag, row['id'], row['role'], hashlib.sha256(raw).hexdigest())
            console.write((data + tag + ' COMPLETE\n').encode())

    class Process:
        def __init__(self, *args, **kwargs): kwargs['stdout'].write(preflight.openocd_builds()[0]['banner']+'\n')
        def poll(self): return None
        def terminate(self): pass
        def wait(self, **kwargs): return 0

    patches = [patch.object(preflight, 'live'), patch.object(run_kit, 'UART', UART),
               patch.object(run_kit, 'TclRPC', RPC), patch.object(run_kit.subprocess, 'Popen', Process)]
    return patches, log


class LiveTests(unittest.TestCase):
    """Synthetic packaged kit over in-memory peers; protocol, not ELF math."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.manifest = live_kit(self.root)

    def test_patch_is_selected_only_for_negative_and_bound_to_qualification(self):
        self.assertIsNone(load(self.root)['negative_patch'])
        self.assertEqual(load(self.root, 'negative/chip.elf')['negative_patch']['after_byte'], 129)
        (self.root/'qualification/check.json').write_text(json.dumps(dict(byte_flip={})))
        inventory(self.root)
        with self.assertRaisesRegex(ValueError, 'qualification'): load(self.root)

    def test_patch_outside_inputs_and_wrong_preimage_fail(self):
        for change in (dict(address=0x200c0010), dict(before_byte=2)):
            before = self.manifest['runner']['negative_patch'].copy()
            self.manifest['runner']['negative_patch'].update(change); rewrite(self.root, self.manifest)
            with self.assertRaises(ValueError): load(self.root, 'negative/chip.elf')
            self.manifest['runner']['negative_patch'] = before

    def test_transport_requires_explicit_gate(self):
        with patch.object(run_kit, 'UART', side_effect=AssertionError('transport opened')):
            self.assertEqual(run_kit.main([str(self.root), '--logs', str(self.root/'logs')]), 2)

    def test_negative_patch_executes_after_elf_verify_and_checker_accepts(self):
        import io
        from contextlib import ExitStack, redirect_stdout
        logs = self.root/'negative-20261117T000000Z'
        patches, _ = peers(self.manifest)
        output = io.StringIO()
        with ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            stack.enter_context(redirect_stdout(output))
            rc = run_kit.main([str(self.root), '--logs', str(logs), '--allow-transport', '--elf', 'negative/chip.elf'])
        capture = logs/'terminal.txt'; capture.write_text(output.getvalue() + f'rc={rc}\n')
        self.assertEqual(rc, 1)
        self.assertEqual((logs/'negative-before.bin').read_bytes(), b'\x01')
        self.assertEqual((logs/'negative-patch.bin').read_bytes(), b'\x81')
        operations = json.loads((logs/'summary.json').read_text())['operations']
        self.assertEqual([r['operation'] for r in operations][:8],
            ['reset', 'load_image', 'load_image', 'verify_image', 'dump_image', 'load_image', 'resume', 'halt'])
        checked = check_rung.check(capture, logs, bundle=self.root, negative=True)
        self.assertEqual(checked.verdict, 'PASS', checked.checks)

    def session(self, **behavior):
        from contextlib import ExitStack
        patches, log = peers(self.manifest, **behavior)
        stack = ExitStack(); self.addCleanup(stack.close)
        for item in patches: stack.enter_context(item)
        a = run_kit.arguments([str(self.root), '--allow-transport', '--logs', str(self.root/'session')])
        kit = run_kit.prepare(a)
        session = run_kit.Session(a, kit, run_kit.live(a, kit))
        session.open()
        return session, kit, log

    def staged(self, value):
        return [dict(row=self.manifest['deployment']['inputs'][0], data=np.int8(value).tobytes())]

    def test_resident_session_loads_once_and_restages_each_call(self):
        session, kit, log = self.session()
        reference = step_inputs.window_inputs(kit['dep'], session.window)
        self.assertEqual(session.call(self.root/'reference', reference, kit['references']),
                         [b'\x01\x02\x03\x04', b'\x05\x06\x07\x08'])
        for index, value in enumerate((-3, 7)):
            outputs = session.call(self.root/('step'+str(index)), self.staged(value))
            self.assertEqual(outputs[0], bytes([value % 256, 2, 3, 4]))
            self.assertEqual((self.root/('step%d/output-0.bin' % index)).read_bytes(), outputs[0])
        session.close('PASS')
        self.assertEqual(log.count('reset halt'), 1)
        self.assertEqual(sum(x.startswith('load_image {'+str(kit['window'])) for x in log), 1)
        self.assertEqual(sum(x.startswith('load_image {'+str(kit['elf'])) for x in log), 1)
        self.assertEqual(sum(x.startswith('resume') for x in log), 3)
        loaded = [x.split()[2] for x in log if x.startswith('load_image') and len(x.split()) == 4]
        self.assertNotIn('0x200c0030', loaded)
        self.assertEqual(loaded.count('0x200c0010'), 3)
        self.assertEqual(json.loads((self.root/'session/summary.json').read_text())['verdict'], 'PASS')
        # Each call reads back its one restore, two poisons, and one input; a matching readback leaves no file.
        self.assertEqual(sum(x.startswith('dump_image') and '-readback.bin}' in x for x in log), 3 * 4)
        self.assertEqual(list(self.root.glob('*/*-readback.bin')), [])

    def test_a_poison_write_that_reads_back_differently_fails_before_the_restart(self):
        session, kit, log = self.session(corrupt={0x200c0000: b'\x00\xa5\xa5\xa5'})
        with self.assertRaisesRegex(TransportError, 'write readback differs at 0x200c0000'):
            session.call(self.root/'step', self.staged(7))
        self.assertFalse(any(x.startswith('resume') for x in log))
        self.assertEqual((self.root/'step/poison-0-readback.bin').read_bytes(), b'\x00\xa5\xa5\xa5')
        session.close('TRANSPORT_ERROR', 'write readback differs')

    def test_dump_failure_after_valid_completion_is_transport_error(self):
        session, _, _ = self.session(fail_dump_after=0)
        with self.assertRaisesRegex(TransportError, 'dump connection lost'):
            session.call(self.root/'step', self.staged(7))
        self.assertTrue((self.root/'step/uart.log').read_text().endswith('COMPLETE\n'))
        session.close('TRANSPORT_ERROR', 'dump connection lost')

    def test_step_status_and_unwritten_outputs_are_numerical_failures(self):
        for behavior, message in ((dict(status=3), 'nonzero status'), (dict(write_outputs=False), 'poisoned')):
            with self.subTest(**behavior):
                session, _, _ = self.session(**behavior)
                with self.assertRaisesRegex(run_kit.NumericalError, message):
                    session.call(self.root/'step', self.staged(7))
                session.close('FAIL')
                import shutil
                shutil.rmtree(self.root/'session'); shutil.rmtree(self.root/'step')

    def test_reference_call_must_match_frozen_outputs(self):
        session, kit, _ = self.session()
        with self.assertRaisesRegex(run_kit.NumericalError, 'resident reference'):
            session.call(self.root/'reference', self.staged(7), kit['references'])
        session.close('FAIL')


if __name__ == '__main__':
    unittest.main()
