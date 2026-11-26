"""Adapter checks exercise evidence validation, without arithmetic claims."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest

import numpy as np

from atlas_mlir.deploy import Refusal
from atlas_mlir.deployment_gate import _case, _member, _reference


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GateAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.reference = self.out / 'reference'
        self.reference.mkdir()
        capture = self.out / 'capture/capture_receipt.json'
        write(capture, {})
        application = self.out / 'compiled/application/manifest.json'
        write(application, {})
        write(self.out / 'target-rules.json', {})
        self.values = np.array([1, 2], dtype=np.uint16)
        np.savez(self.reference / 'expected.npz', output_0=self.values)
        self.record = dict(target_rules_sha256=sha(self.out / 'target-rules.json'), status='EXPECTED', ordinary_framework_kernels_called=False,
            compiler_arithmetic_or_emitted_program_inputs=False,
            expected_sha256=sha(self.reference / 'expected.npz'),
            authentication={'receipt_sha256': sha(capture)},
            compiler_decision_inputs={'identities': {str(application): sha(application)}},
            outputs={'output_0': {'shape': [2], 'dtype': 'uint16', 'nonzero': 2}})
        write(self.reference / 'reference.json', self.record)
        self.row = dict(index=0, id='value', role='public_output', address=4096,
                        bytes=4, type={'shape': [2], 'dtype': 'bf16'})
        self.deployment = dict(outputs=[self.row], hash_ranges=[self.row])
        self.context = dict(out=self.out, reference=self.reference)

    def test_reference_preserves_every_raw_word(self):
        _, rows = _reference(self.context, self.deployment)
        self.assertEqual(rows[0]['sha256'], hashlib.sha256(self.values.tobytes()).hexdigest())

    def test_reference_refuses_identity_or_storage_drift(self):
        for field, value in [('status', 'REFUSED'), ('ordinary_framework_kernels_called', True),
                             ('expected_sha256', 'wrong')]:
            with self.subTest(field=field):
                write(self.reference / 'reference.json', dict(self.record, **{field: value}))
                with self.assertRaises(Refusal):
                    _reference(self.context, self.deployment)
        write(self.reference / 'reference.json', self.record)
        self.deployment['outputs'][0]['type']['dtype'] = 'f32'
        with self.assertRaisesRegex(Refusal, 'type or size'):
            _reference(self.context, self.deployment)

    def test_reference_consumers_refuse_an_audit_record_even_with_a_valid_archive(self):
        """An audit reports COVERED or GAPS and writes no archive; neither status stands in for EXPECTED."""
        from subprocess import CompletedProcess
        from atlas_mlir import deployment_stages
        for status in ('COVERED', 'GAPS'):
            with self.subTest(consumer='gate', status=status), self.assertRaisesRegex(Refusal, 'authentication missing'):
                write(self.reference / 'reference.json', dict(self.record, status=status))
                _reference(self.context, self.deployment)
            destination = self.out / ('stage-' + status)

            def fabricated(command, **kwargs):
                destination.mkdir()
                np.savez(destination / 'expected.npz', output_0=self.values)
                write(destination / 'reference.json', dict(self.record, status=status,
                                                           expected_sha256=sha(destination / 'expected.npz')))
                return CompletedProcess(command, 0)
            with self.subTest(consumer='stage', status=status), \
                    patch.object(deployment_stages.subprocess, 'run', side_effect=fabricated), \
                    self.assertRaisesRegex(Refusal, 'identity or nonzero coverage missing'):
                deployment_stages._evaluate(self.out, destination, self.out / ('stage-' + status + '.log'))

    def test_reference_refuses_missing_output(self):
        self.deployment['outputs'].append(dict(self.row, index=1))
        with self.assertRaisesRegex(Refusal, 'every deployment output'):
            _reference(self.context, self.deployment)

    def test_reference_refuses_all_zero(self):
        np.savez(self.reference / 'expected.npz', output_0=np.zeros(2, dtype=np.uint16))
        write(self.reference / 'reference.json', dict(self.record, expected_sha256=sha(self.reference / 'expected.npz')))
        with self.assertRaisesRegex(Refusal, 'all-zero'):
            _reference(self.context, self.deployment)

    def test_unsafe_members_refuse(self):
        for member in ('../file', '/file'):
            with self.subTest(member=member), self.assertRaises(Refusal):
                _member(self.out, member)

    def _case_record(self):
        root = self.out / 'exact/positive'
        root.mkdir(parents=True)
        write(root / 'execution/runtime.json', {'launches': [{}]})
        (root / 'output_000.bin').write_bytes(self.values.tobytes())
        record = dict(case='positive', bind_status=0, run_status=0, launch_count=1,
            total_instructions=1, public_buffers_hashed_at_published_addresses=True,
            immutable_buffers_unchanged=True, result='EXACT',
            dma={'bounds_checked_before_execution': True, 'commands': {'load': 1, 'store': 1}},
            runtime_path='execution/runtime.json', runtime_sha256=sha(root / 'execution/runtime.json'),
            outputs=[dict(index=0, name='value', address=4096, bytes=4,
                path='output_000.bin', sha256=sha(root / 'output_000.bin'),
                expected_sha256=hashlib.sha256(self.values.tobytes()).hexdigest(),
                mismatched_bytes=0, exact=True)])
        write(root / 'result.json', record)
        return record

    def test_case_requires_actual_output_bytes(self):
        record = self._case_record()
        files = _case(self.out, self.out / 'exact', 'positive', record, self.deployment)
        self.assertEqual(len(files), 3)
        (self.out / 'exact/positive/output_000.bin').write_bytes(b'bad!')
        with self.assertRaisesRegex(Refusal, 'output evidence'):
            _case(self.out, self.out / 'exact', 'positive', record, self.deployment)

    def test_case_refuses_success_label_over_changed_bytes(self):
        record = self._case_record()
        root = self.out / 'exact/positive'
        raw = bytearray(self.values.tobytes())
        raw[0] ^= 1
        (root / 'output_000.bin').write_bytes(raw)
        record['outputs'][0].update(sha256=sha(root / 'output_000.bin'), mismatched_bytes=1, exact=False)
        write(root / 'result.json', record)
        with self.assertRaisesRegex(Refusal, 'case result'):
            _case(self.out, self.out / 'exact', 'positive', record, self.deployment)

    def test_case_checks_concrete_negative_mismatch(self):
        record = self._case_record()
        source = self.out / 'exact/positive'
        target = self.out / 'exact/negative'
        source.rename(target)
        raw = bytearray(self.values.tobytes())
        raw[0] ^= 1
        (target / 'output_000.bin').write_bytes(raw)
        record.update(case='negative', result='MISMATCH', first_mismatch=dict(
            output_index=0, byte_offset=0, address=4096, actual_byte=raw[0], expected_byte=1))
        record['outputs'][0].update(sha256=sha(target / 'output_000.bin'), mismatched_bytes=1, exact=False)
        write(target / 'result.json', record)
        _case(self.out, self.out / 'exact', 'negative', record, self.deployment)
        record['first_mismatch']['address'] += 1
        write(target / 'result.json', record)
        with self.assertRaisesRegex(Refusal, 'mismatch location'):
            _case(self.out, self.out / 'exact', 'negative', record, self.deployment)

    def test_case_requires_executed_dma(self):
        record = self._case_record()
        record['dma']['commands']['store'] = 0
        with self.assertRaisesRegex(Refusal, 'DMA'):
            _case(self.out, self.out / 'exact', 'positive', record, self.deployment)

    def test_case_refuses_runtime_hash_drift(self):
        record = self._case_record()
        write(self.out / 'exact/positive/execution/runtime.json', {'launches': []})
        with self.assertRaisesRegex(Refusal, 'runtime evidence'):
            _case(self.out, self.out / 'exact', 'positive', record, self.deployment)


class ReadOnlyStateBindingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.application = self.out / 'compiled/application'
        self.payload_path = 'handoff/state/weight.bin'
        self.raw = bytes.fromhex('803f0040')
        payload = self.application / self.payload_path
        payload.parent.mkdir(parents=True)
        payload.write_bytes(self.raw)
        write(self.out / 'capture/capture_receipt.json', {})
        self.row = dict(role='persistent_state_input', state_mode='read_only', initialization='captured',
                        id='weight', state='weight', index=1, source_input_index=1,
                        path=self.payload_path, bytes=len(self.raw), sha256=sha(payload))
        self.deployment = dict(schema='merlin.onchip_deployment.v1', inputs=[],
                               state_inputs=[self.row], provider=dict(sys_clk_freq_hz=100_000_000), parameters=None)
        self.context = dict(out=self.out, settings=dict(clock_hz=100_000_000))
        self.publish()

    def publish(self, *, include_state=True):
        write(self.application / 'deployment.json', self.deployment)
        members = ['deployment.json'] + ([self.payload_path] if include_state else [])
        write(self.application / 'manifest.json', dict(stage_artifacts={
            member: sha(self.application / member) for member in members}))

    def bind(self):
        from atlas_mlir.deployment_gate import prepare_binding
        with patch('atlas_mlir_native.reference_eval.source_io.capture_sources', return_value=(self.out, (), None)), \
                patch('atlas_mlir_native.reference_eval.source_io.capture_inputs', return_value={}), \
                patch('atlas_mlir_native.deployment_platform.bind_deployment_payloads', return_value={}) as binder:
            prepared = prepare_binding(self.context)
        self.assertEqual(binder.call_args.args[1], {self.payload_path: self.raw})
        return prepared

    def test_read_only_state_uses_exact_authenticated_raw_payload(self):
        prepared = self.bind()
        self.assertEqual(prepared['payloads'], {self.payload_path: self.raw})
        self.assertEqual(prepared['deployment']['state_inputs'], [self.row])
        self.assertEqual(prepared['deployment']['inputs'], [])

    def test_compiled_clock_must_be_the_deploy_clock(self):
        self.bind()
        self.context['settings'] = dict(clock_hz=500_000_000)
        with self.assertRaisesRegex(Refusal, 'compiled host clock differs from the deploy clock'):
            self.bind()

    def test_mutable_uninitialized_or_drifted_state_refuses(self):
        original = dict(self.row)
        for changed in (dict(state_mode='updated'), dict(initialization='external'),
                        dict(role='public_input'), dict(bytes=len(self.raw) + 1),
                        dict(sha256='a' * 64)):
            with self.subTest(changed=changed):
                self.row.clear()
                self.row.update(original, **changed)
                self.publish()
                with self.assertRaises(Refusal):
                    self.bind()
        self.row.clear()
        self.row.update(original)
        self.publish(include_state=False)
        with self.assertRaisesRegex(Refusal, 'state payload inventory'):
            self.bind()
        self.publish()
        (self.application / self.payload_path).write_bytes(b'bad!')
        with self.assertRaises(Refusal):
            self.bind()


if __name__ == '__main__':
    unittest.main()


class SignOffsetTest(unittest.TestCase):
    def test_sign_byte_of_the_first_nonzero_element_of_input_zero(self):
        from atlas_mlir.deployment_gate import _sign_offset
        with tempfile.TemporaryDirectory() as name:
            bundle = Path(name)
            row = dict(index=0, address=0x80000010, bytes=8, type=dict(dtype='bf16', shape=[4]))
            (bundle / 'manifest.json').write_text(json.dumps(dict(deployment=dict(
                window=dict(address=0x80000000), inputs=[row]))))
            (bundle / 'window.bin').write_bytes(b'\x01' * 16 + b'\0\0\0\0\x7a\x3e\0\0')
            self.assertEqual(_sign_offset(bundle), 5)
            (bundle / 'window.bin').write_bytes(b'\x01' * 16 + b'\0' * 8)
            self.assertIsNone(_sign_offset(bundle))
