"""Packaging contract tests with synthetic tools; never silicon evidence."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir.deploy import Gate, Refusal, clock_settings, digest
from atlas_mlir import deployment_kit
from atlas_mlir.deployment_kit import dry_preflight, package
from atlas_mlir.deployment_link import _HARNESS, RECORD_TAG


class KitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / 'bundle'
        (self.bundle / 'application').mkdir(parents=True)
        (self.bundle / 'reference').mkdir()
        data = np.array([7, 2], dtype=np.uint8)
        np.savez(self.bundle / 'reference/expected.npz', output_0=data)
        ranges = [dict(id='output', index=0, role='public_output', address=0x200c0004, bytes=2)]
        self.dep = dict(window=dict(address=0x200c0000, bytes=64), outputs=ranges, hash_ranges=ranges,
                        inputs=[dict(id='input', index=0, address=0x200c0000, bytes=2,
                                     type=dict(dtype='u8', shape=[2]))])
        (self.bundle / 'application/deployment.json').write_text(json.dumps(self.dep))
        (self.bundle / 'window.bin').write_bytes(b'\0' * 16)
        manifest = dict(deployment=self.dep, expected_hashes=[dict(name='output', role='public_output',
            sha256=hashlib.sha256(data.tobytes()).hexdigest(), bytes=2)])
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        self.receipt = self.root / 'gate.json'
        negative = self.root / 'exact/negative/output-0.bin'
        negative.parent.mkdir(parents=True)
        negative.write_bytes(b'\x08\x02')
        window = b'\0' * 16
        changed = b'\x80' + window[1:]
        sha = lambda data: hashlib.sha256(data).hexdigest()
        self.receipt.write_text(json.dumps(dict(schema='atlas.model_check.v1', status='PASS',
            positive=dict(result='EXACT'), negative=dict(result='MISMATCH', first_mismatch={},
                outputs=[dict(index=0, name='output', path='output-0.bin', bytes=2,
                              address=0x200c0004, sha256=sha(negative.read_bytes()))]),
            byte_flip=dict(changed_byte_count=1, window_byte_offset=0, xor_mask=128,
                input_index=0, input_name='input', input_byte_offset=0, address=0x200c0000,
                before_byte=0, after_byte=128, window_before_sha256=sha(window),
                window_after_sha256=sha(changed), input_before_sha256=sha(window[:2]),
                input_after_sha256=sha(changed[:2])), same_application_library=True,
            same_instruction_images=True, independent_expected_unchanged=True)))
        names = ('manifest.json', 'window.bin', 'application/deployment.json', 'reference/expected.npz')
        hashes = {str(self.bundle / name): digest(self.bundle / name) for name in names}
        hashes[str(self.receipt)] = digest(self.receipt)
        hashes[str(negative)] = digest(negative)
        self.gate = Gate(self.receipt, hashes)
        self.elf = self.root / 'model.elf'
        raw = bytearray(512)
        raw[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<H', raw, 18, 243)
        struct.pack_into('<Q', raw, 24, 0x08000000)
        struct.pack_into('<Q', raw, 32, 64)
        struct.pack_into('<HH', raw, 54, 56, 2)
        struct.pack_into('<IIQQQQQQ', raw, 64, 1, 7, 120, 0x08000000, 0x08000000, 64, 64, 8)
        struct.pack_into('<IIQQQQQQ', raw, 120, 1, 6, 224, 0x200c0000, 0x200c0000, 16, 16, 8)
        struct.pack_into('<Q', raw, 40, 256)
        struct.pack_into('<HHH', raw, 58, 64, 3, 1)
        names = b'\0.shstrtab\0.window\0'
        raw[184:184 + len(names)] = names
        struct.pack_into('<IIQQQQIIQQ', raw, 320, 1, 3, 0, 0, 184, len(names), 0, 0, 1, 0)
        struct.pack_into('<IIQQQQIIQQ', raw, 384, 11, 1, 3, 0x200c0000, 224, 16, 0, 0, 8, 0)
        self.elf.write_bytes(raw)
        self.tool_root = self.root / 'tools'
        self.tools = self.tool_root / 'scripts'
        self.tools.mkdir(parents=True)
        self.tool_hashes = {}
        # Synthetic entry points prove argument selection and failure handling.
        # Actual runner/checker qualification is a separate integration check.
        messages = dict(run_kit='KIT DRY PASS authenticated preflight; no execution',
                        check_rung='KIT CHECKER DRY PASS authenticated preflight; no execution evidence checked')
        for name in deployment_kit.owned_tools():
            destination = self.tools / name
            content = "raise AssertionError('dependency must not execute')\n"
            if name == 'capture_run.py':
                content = Path(deployment_kit.__file__).with_name('chiprunner').joinpath(name).read_text()
            elif Path(name).stem in messages:
                content = ("import sys\nassert '--dry-run' in sys.argv\nassert '--logs' not in sys.argv\n"
                           + 'print(' + repr(messages[Path(name).stem]) + ')\n')
            destination.write_text(content)
            self.tool_hashes[name] = digest(destination)
        self.out = self.root / 'out'
        self.out.mkdir()
        self.settings = clock_settings('100')
        self.context = dict(out=self.out, bundle=self.bundle, elf=self.elf, gate=self.gate, transport='jtag',
            settings=self.settings, elf_record=dict(sha256=digest(self.elf), entry=0x08000000,
            loads=[dict(address=0x08000000, file_bytes=64, memory_bytes=64, flags=7),
                   dict(address=0x200c0000, file_bytes=16, memory_bytes=16, flags=6)],
            **{key: self.settings[key] for key in ('clock_hz', 'uart_divisor', 'receiver_baud')}))
        self.env = dict(ATLAS_KIT_RUNNER=str(self.tools / 'run_kit.py'),
                        ATLAS_KIT_CHECKER=str(self.tools / 'check_rung.py'))
        self.pins = patch.object(deployment_kit, 'owned_tools', return_value={
            name: self.tools / name for name in self.tool_hashes})
        self.pins.start()
        self.addCleanup(self.pins.stop)
        def negative_record(path, deployment, settings):
            return dict(self.context['elf_record'], sha256=digest(path))
        self.measure = patch('atlas_mlir.deployment_link._elf_record', side_effect=negative_record)
        self.measure.start()
        self.addCleanup(self.measure.stop)

    def stage(self):
        with patch.dict(os.environ, self.env, clear=True):
            package(self.context)
        return self.out / 'kit'

    def test_generalized_layout_and_both_dry_commands(self):
        manifest_before = (self.bundle / 'manifest.json').read_bytes()
        kit = self.stage()
        logs = dry_preflight(kit, kit / 'bench/run_kit.py')
        self.assertFalse((kit / 'memory-plan.json').exists() or (kit / 'loader-manifest.json').exists())
        self.assertEqual(set(logs), {'runner', 'checker', 'runner-negative', 'checker-negative'})
        self.assertEqual((kit / 'reference/output_000.bin').read_bytes(), b'\x07\x02')
        self.assertEqual((kit / 'gate-manifest.json').read_bytes(), manifest_before)
        self.assertEqual((self.bundle / 'manifest.json').read_bytes(), manifest_before)
        self.assertEqual((kit / 'qualification/check.json').read_bytes(), self.receipt.read_bytes())
        manifest = json.loads((kit / 'manifest.json').read_text())
        self.assertEqual(manifest['bench_facts']['source'], self.settings['clock_basis'])
        self.assertEqual(manifest['bench_facts']['uart_divisor'], 867)
        result = json.loads((self.out / 'kit-preflight.json').read_text())
        self.assertTrue(result['checker_dry_run'])
        self.assertFalse(result['transport_executed'])
        self.assertEqual(self.context['kit'], kit)
        for name in self.tool_hashes:
            self.assertTrue((kit / 'bench' / name).is_file())
        original = self.elf.read_bytes()
        patched = (kit / 'negative/chip.elf').read_bytes()
        self.assertEqual((kit / 'chip.elf').read_bytes(), original)
        self.assertEqual([i for i, pair in enumerate(zip(original, patched)) if pair[0] != pair[1]], [224])
        self.assertEqual(patched[224], original[224] ^ 128)
        self.assertEqual((kit / 'negative/negative-evidence/output_000.bin').read_bytes(), b'\x08\x02')
        negative = json.loads((kit / 'negative/manifest.json').read_text())
        self.assertEqual(negative['sizes']['chip']['sha256'], digest(kit / 'negative/chip.elf'))
        self.assertTrue((kit / 'run.sh').is_file())
        self.assertTrue((kit / 'bench/capture_run.py').is_file())

    def test_board_ddr_kit_keeps_only_the_soc_declaration_and_makes_no_dry_claims(self):
        self.context['transport'] = 'spi'
        declared = patch('atlas_mlir_native.soc.DECLARATION_SHA256', self.tool_hashes['soc.json'])
        with declared, patch('atlas_mlir.deployment_loader.describe',
                             return_value=({'plan': True}, {'loader': True})) as describe:
            kit = self.stage()
        deployment = json.loads((kit / 'application/deployment.json').read_text())
        describe.assert_called_once_with(kit / 'chip.elf', deployment, self.context['elf_record'])
        self.assertEqual(json.loads((kit / 'memory-plan.json').read_text()), {'plan': True})
        self.assertIn('loader-manifest.json', (kit / 'SHA256SUMS').read_text())
        self.assertFalse((kit / 'run.sh').exists())
        self.assertEqual(sorted(path.name for path in (kit / 'bench').iterdir()), ['soc.json'])
        self.assertIn("refuses until the board's SPI loader exists", (kit / 'README.md').read_text())
        self.assertIn('writes `(code << 1) | 1` to `tohost`', (kit / 'README.md').read_text())
        result = json.loads((kit / 'result.json').read_text())
        self.assertEqual([result[key] for key in ('dry_inventory', 'dry_static_preflight', 'checker_dry_run')], [False] * 3)
        listed = {line.split('  ', 1)[1] for line in (kit / 'SHA256SUMS').read_text().splitlines()}
        self.assertEqual(listed, {str(path.relative_to(kit)) for path in kit.rglob('*')
                                  if path.is_file() and path.name != 'SHA256SUMS'})
        self.assertTrue({'chip.elf', 'negative/chip.elf', 'reference/output_000.bin', 'bench/soc.json'} <= listed)
        self.assertFalse(list(self.out.glob('kit-*-dry.log')))
        # The kit's declaration must be the one the deployment was placed against.
        self.out = self.root / 'other'
        self.out.mkdir()
        self.context['out'] = self.out
        with self.assertRaisesRegex(Refusal, 'kit SoC declaration differs'):
            self.stage()

    def test_manifest_declares_the_tag_the_harness_emits(self):
        # The runner refuses a bundle without runner.record_tag and parses UART records by it.
        runner = json.loads((self.stage() / 'manifest.json').read_text())['runner']
        self.assertEqual(runner, dict(schema='atlas.kit.v1', record_tag=RECORD_TAG))
        self.assertTrue(_HARNESS.startswith('#define RECORD_TAG "%s"\n' % RECORD_TAG))
        for record in ('STATUS ', 'HASH ', 'COMPLETE\\n'):
            self.assertEqual(_HARNESS.count('emit(RECORD_TAG " ' + record + '")'), 1)
        self.assertNotIn(RECORD_TAG + ' ', _HARNESS)

    def test_multiple_outputs_preserve_every_reference_byte(self):
        second = np.array([11, 17, 19], dtype=np.uint8)
        np.savez(self.bundle / 'reference/expected.npz',
                 output_0=np.array([7, 2], dtype=np.uint8), output_1=second)
        row = dict(id='second', index=1, role='public_output', address=0x200c0010, bytes=3)
        self.dep['hash_ranges'].append(row)
        (self.bundle / 'application/deployment.json').write_text(json.dumps(self.dep))
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        manifest['deployment'] = self.dep
        manifest['expected_hashes'].append(dict(name='second', role='public_output', bytes=3,
                                                sha256=hashlib.sha256(second.tobytes()).hexdigest()))
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        for name in ('reference/expected.npz', 'application/deployment.json', 'manifest.json'):
            self.gate.hashes[str(self.bundle / name)] = digest(self.bundle / name)
        path = self.root / 'exact/negative/output-1.bin'
        path.write_bytes(second.tobytes())
        receipt = json.loads(self.receipt.read_text())
        receipt['negative']['outputs'].append(dict(index=1, name='second', path=path.name,
            address=row['address'], bytes=3, sha256=digest(path)))
        self.receipt.write_text(json.dumps(receipt))
        for source in (path, self.receipt):
            self.gate.hashes[str(source)] = digest(source)
        kit = self.stage()
        self.assertEqual((kit / 'reference/output_001.bin').read_bytes(), second.tobytes())

    def test_all_zero_references_refuse(self):
        np.savez(self.bundle / 'reference/expected.npz', output_0=np.zeros(2, dtype=np.uint8))
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        manifest['expected_hashes'][0]['sha256'] = hashlib.sha256(b'\0\0').hexdigest()
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        for name in ('reference/expected.npz', 'manifest.json'):
            self.gate.hashes[str(self.bundle / name)] = digest(self.bundle / name)
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'all-zero'):
                package(self.context)

    def test_environment_cannot_supply_packaging_tools(self):
        supplied = self.root / 'untrusted.py'
        supplied.write_text('raise RuntimeError("must not execute")\n')
        self.env = dict(ATLAS_KIT_RUNNER=str(supplied), ATLAS_KIT_CHECKER=str(supplied))
        kit = self.stage()
        self.assertEqual((kit / 'bench/run_kit.py').read_bytes(),
                         (self.tools / 'run_kit.py').read_bytes())

    def test_changed_staged_tool_refuses(self):
        kit = self.stage()
        (kit / 'bench/run_kit.py').write_text('raise RuntimeError("must not execute")\n')
        with self.assertRaisesRegex(Refusal, 'offline contract changed'):
            dry_preflight(kit, kit / 'bench/run_kit.py')

    def test_unowned_staged_module_refuses(self):
        kit = self.stage()
        (kit / 'bench/json.py').write_text('raise RuntimeError("must not execute")\n')
        with self.assertRaisesRegex(Refusal, 'unexpected tool directory member'):
            dry_preflight(kit, kit / 'bench/run_kit.py')

    def test_inventory_rejects_symlink_and_parent_traversal(self):
        kit = self.stage()
        target = kit / 'window.bin'
        target.unlink()
        target.symlink_to(self.bundle / 'window.bin')
        with self.assertRaisesRegex(Refusal, 'symlink'):
            dry_preflight(kit, kit / 'bench/run_kit.py')
        target.unlink()
        target.write_bytes(b'\0' * 16)
        with (kit / 'SHA256SUMS').open('a') as stream:
            stream.write('0' * 64 + '  ../outside\n')
        with self.assertRaisesRegex(Refusal, 'invalid kit inventory member'):
            dry_preflight(kit, kit / 'bench/run_kit.py')

    def test_inventory_refuses_drift_duplicate_and_missing_member(self):
        kit = self.stage()
        inventory = kit / 'SHA256SUMS'
        original = inventory.read_text()
        for modified, message in [(original + original.splitlines()[0] + '\n', 'duplicate'),
                ('\n'.join(x for x in original.splitlines() if not x.endswith('  chip.elf')) + '\n', 'omits')]:
            inventory.write_text(modified)
            with self.assertRaisesRegex(Refusal, message):
                dry_preflight(kit, kit / 'bench/run_kit.py')
        inventory.write_text(original)
        (kit / 'reference/output_000.bin').write_bytes(b'bad')
        with self.assertRaisesRegex(Refusal, 'integrity mismatch'):
            dry_preflight(kit, kit / 'bench/run_kit.py')

    def test_changed_elf_refuses_before_staging(self):
        self.elf.write_bytes(b'changed')
        with self.assertRaisesRegex(Refusal, 'ELF identity'):
            package(self.context)
        self.assertFalse((self.out / 'kit').exists())

    def test_clock_drift_refuses_before_staging(self):
        self.context['elf_record']['clock_hz'] = 0
        with self.assertRaisesRegex(Refusal, 'clock settings drift'):
            package(self.context)

    def test_checker_failure_blocks_completion(self):
        checker = self.tools / 'check_rung.py'
        checker.write_text("print('KIT CHECKER DRY FAIL synthetic failure')\nraise SystemExit(1)\n")
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'checker dry check refused'):
                package(self.context)
        self.assertNotIn('kit', self.context)
        self.assertFalse((self.out / 'kit-preflight.json').exists())

    def test_gate_drift_blocks_packaging(self):
        (self.bundle / 'window.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(Refusal, 'hash drift'):
            package(self.context)

    def test_output_hash_mismatch_refuses(self):
        np.savez(self.bundle / 'reference/expected.npz', output_0=np.array([9, 2], dtype=np.uint8))
        self.gate.hashes[str(self.bundle / 'reference/expected.npz')] = digest(self.bundle / 'reference/expected.npz')
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'output bytes differ'):
                package(self.context)

    def test_missing_executed_mutation_refuses_before_staging(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['negative']['result'] = 'SKIPPED'
        self.receipt.write_text(json.dumps(receipt))
        self.gate.hashes[str(self.receipt)] = digest(self.receipt)
        with self.assertRaisesRegex(Refusal, 'executed output-changing mutation'):
            package(self.context)
        self.assertFalse((self.out / 'kit').exists())

    def test_missing_owned_checker_refuses_before_staging(self):
        (self.tools / 'check_rung.py').unlink()
        with self.assertRaisesRegex(Refusal, 'kit member missing'):
            package(self.context)
        self.assertFalse((self.out / 'kit').exists())

    def test_negative_prediction_gate_authentication_required(self):
        self.gate.hashes.pop(str(self.root / 'exact/negative/output-0.bin'))
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'not authenticated'):
                package(self.context)

    def test_negative_requires_executed_changed_output(self):
        path = self.root / 'exact/negative/output-0.bin'
        path.write_bytes(b'\x07\x02')
        receipt = json.loads(self.receipt.read_text())
        receipt['negative']['outputs'][0]['sha256'] = digest(path)
        self.receipt.write_text(json.dumps(receipt))
        for source in (path, self.receipt):
            self.gate.hashes[str(source)] = digest(source)
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'do not change'):
                package(self.context)

    def test_mutation_identity_must_match_bound_window(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['byte_flip']['after_byte'] = 7
        self.receipt.write_text(json.dumps(receipt))
        self.gate.hashes[str(self.receipt)] = digest(self.receipt)
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(Refusal, 'window authentication'):
                package(self.context)

    def wrapper(self, kit, runner, arguments):
        module = types.SimpleNamespace(safe_path=lambda value: Path(value).absolute(),
                                       load=lambda bundle, elf: dict(settings=dict(timeout=120), dep=dict(hash_ranges=[{}])))
        with patch.dict(sys.modules, {'kit_manifest': module}), patch.object(sys, 'argv', ['capture_run.py', str(kit), *arguments]), \
                patch.dict(os.environ, {'NOV_OPENOCD': '/explicit/unused-openocd'}), \
                patch('subprocess.Popen', side_effect=runner), patch('sys.stdout'):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(kit / 'bench/capture_run.py'), run_name='__main__')
        return result.exception.code

    def test_capture_records_return_code_and_refuses_reuse(self):
        kit = self.stage()
        attempts = []
        def run(command, **kwargs):
            attempts.append((command, kwargs))
            kwargs['stdout'].write('KIT FAIL synthetic output-changing negative\n')
            kwargs['stdout'].flush()
            return types.SimpleNamespace(wait=lambda timeout: 1)
        for expected in (1, 2):
            self.assertEqual(self.wrapper(kit, run, ['20261118T120000Z', 'negative']), expected)
        self.assertEqual(len(attempts), 1)
        command, options = attempts[0]
        self.assertIn('negative/chip.elf', command)
        self.assertEqual(command[-1], '--allow-transport')
        self.assertNotIn('--openocd', command)
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertEqual((kit / 'runs/kit-20261118T120000Z-terminal.txt').read_text(),
                         'KIT FAIL synthetic output-changing negative\nrc=1\n')

    def test_capture_bound_stops_runner_as_transport_failure(self):
        kit = self.stage()
        events = []
        def wait(timeout):
            events.append(('wait', timeout))
            if len(events) == 1:
                raise subprocess.TimeoutExpired('run_kit.py', timeout)
            return -15
        runner = types.SimpleNamespace(wait=wait, terminate=lambda: events.append('terminate'),
                                       kill=lambda: events.append('kill'))
        self.assertEqual(self.wrapper(kit, lambda command, **kwargs: runner, ['20261118T120000Z']), 2)
        self.assertEqual(events, [('wait', 1500), 'terminate', ('wait', 150)])
        self.assertEqual((kit / 'runs/kit-20261118T120000Z-terminal.txt').read_text(),
                         'KIT TRANSPORT_ERROR runner exceeded 1500 s and was stopped\nrc=2\n')

    def test_wrapper_reaches_the_real_runner_live_preflight(self):
        # Real run.sh wrapper into the real runner: authorization passes, then the staged build table refuses the digest.
        self.pins.stop()
        with patch.dict(os.environ, {}, clear=True):
            package(self.context)
        kit = self.context['kit']
        wrong = self.root / 'openocd'
        wrong.write_text('#!/bin/sh\nexit 1\n')
        wrong.chmod(0o755)
        result = subprocess.run([sys.executable, str(kit / 'bench/capture_run.py'), str(kit), '20261118T120000Z'],
                                env=dict(PATH=os.environ.get('PATH', ''), NOV_OPENOCD=str(wrong)),
                                capture_output=True, text=True, timeout=120)
        terminal = (kit / 'runs/kit-20261118T120000Z-terminal.txt').read_text()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(terminal, 'KIT TRANSPORT_ERROR OpenOCD build is not declared: sha256 %s\nrc=2\n' % digest(wrong))
        self.assertFalse((kit / 'runs/kit-20261118T120000Z').exists())

    def test_owned_runner_stages_standalone_without_environment(self):
        self.pins.stop()
        with patch.dict(os.environ, {}, clear=True):
            package(self.context)
        kit = self.context['kit']
        self.assertEqual(set(dry_preflight(kit, kit / 'bench/run_kit.py')),
                         {'runner', 'checker', 'runner-negative', 'checker-negative'})


if __name__ == '__main__':
    unittest.main()
