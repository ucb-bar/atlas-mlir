"""Offline ladder control tests. These do not qualify hardware."""
import contextlib
import dis
import io
import json
import numpy as np
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from atlas_mlir import deployment_run as runner
from atlas_mlir.deploy import Refusal
from atlas_mlir.chiprunner import composed, run_kit, check_rung


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'kit').mkdir()
        from atlas_mlir.deployment_kit import owned_tools
        (self.root / 'kit/bench').mkdir()
        for name, source in owned_tools().items():
            (self.root / 'kit/bench' / name).write_bytes(source.read_bytes())
        from test_chiprunner_inputs import live_kit
        live_kit(self.root / 'kit')
        (self.root / 'deploy-status.json').write_text(json.dumps(dict(status='PACKAGED', transport='jtag')))

    def front_door(self, *args, **behavior):
        # `atlascc run OUT ARGS` over the in-memory Tcl, UART, and OpenOCD peers; never a real process or board.
        from atlas_mlir import cli
        from test_chiprunner_inputs import peers
        patches, log = peers(json.loads((self.root / 'kit/manifest.json').read_text()))
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
            mocks = {item.attribute: stack.enter_context(item) for item in patches}
            for name, item in behavior.items():
                stack.enter_context(patch.object(run_kit if name != 'Popen' else run_kit.subprocess, name, item))
            rc = cli.main(['run', str(self.root), *args])
        return rc, output.getvalue(), log, mocks['live']

    def test_dry_run_never_opens_transport(self):
        opened = AssertionError('transport opened')
        rc, output, log, preflight = self.front_door('--dry-run', UART=Mock(side_effect=opened), TclRPC=Mock(side_effect=opened), Popen=Mock(side_effect=opened))
        self.assertEqual((rc, log), (0, []))
        self.assertIn('RUN DRY PASS logs=', output)
        preflight.assert_not_called()
        self.assertFalse(list(self.root.glob('run-*/*-terminal.txt')))

    def test_live_run_needs_no_flag(self):
        rc, output, log, preflight = self.front_door()
        self.assertEqual(rc, 0)
        self.assertIn('RUN PASS logs=', output)
        preflight.assert_called_once()
        self.assertEqual(log[0], 'reset halt')
        terminal, = self.root.glob('run-*/positive-*-terminal.txt')
        self.assertEqual(terminal.read_text(), 'KIT PASS UART and dumped bytes match frozen reference\nrc=0\n')
        summary = terminal.parent / terminal.name.replace('-terminal.txt', '') / 'summary.json'
        self.assertEqual(json.loads(summary.read_text())['verdict'], 'PASS')

    def test_each_invocation_runs_one_selected_rung(self):
        for negative in (False, True):
            calls = []
            def raw(args):
                selected = '--elf' in args
                dry = '--dry-run' in args
                calls.append(('dry' if dry else 'run', selected))
                return int(selected and not dry)
            def check(*args, **kw):
                calls.append(('check', kw['negative']))
                return type('Result', (), dict(verdict='PASS', checks=[]))()
            with patch.object(run_kit, 'main', side_effect=raw), patch.object(check_rung, 'check', side_effect=check), contextlib.redirect_stdout(io.StringIO()) as output:
                logs = runner.execute(self.root, negative=negative)
            self.assertEqual(calls, [('dry', negative), ('run', negative), ('check', negative)])
            verdict = 'RUN PASS (negative rejected as expected) logs=' if negative else 'RUN PASS logs='
            self.assertIn(verdict, output.getvalue())
            self.assertTrue(logs.is_dir())

    def test_serial_reaches_the_runner_and_its_default_is_unchanged(self):
        for serial in (None, '/dev/cu.usbserial-1'):
            with self.subTest(serial=serial), patch.object(run_kit, 'main', return_value=0) as raw, \
                    patch.object(check_rung, 'check', return_value=type('Result', (), dict(verdict='PASS', checks=[]))()), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.execute(self.root, serial=serial)
            live = raw.call_args_list[-1].args[0]
            if serial:
                self.assertEqual(live[live.index('--serial') + 1], serial)
            else:
                self.assertNotIn('--serial', live)
        default = run_kit.arguments([str(self.root / 'kit')]).serial
        self.assertEqual(default, '/dev/serial/by-id/usb-FTDI_Dual_RS232-HS-if01-port0')
        self.assertEqual(runner._bench(self.root / 'kit').serial, default)
        self.assertEqual(runner._bench(self.root / 'kit', ('--serial', '/dev/cu.usbserial-1')).serial, '/dev/cu.usbserial-1')

    def test_negative_selects_only_a_live_rung(self):
        for mode in ('dry_run', 'simulate'):
            with patch.object(run_kit, 'main') as raw, self.assertRaisesRegex(Refusal, 'one live rung'):
                runner.execute(self.root, negative=True, **{mode: True})
            raw.assert_not_called()

    def test_a_kit_with_blocked_steps_only_simulates(self):
        (self.root / 'deploy-status.json').write_text(json.dumps(dict(status='PACKAGED_BLOCKED', transport='jtag',
                                                                      blocked={'region_083': 'host window exceeded'})))
        with patch.object(run_kit, 'main') as raw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'only simulates: region_083'):
                runner.execute(self.root)
        raw.assert_not_called()
        with patch.object(run_kit, 'main', return_value=0) as raw, contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.root, dry_run=True)
        self.assertEqual(raw.call_count, 2)

    def test_first_surprise_stops(self):
        for failure in (1, 2):
            with patch.object(run_kit, 'main', side_effect=[0, failure]) as raw, patch.object(check_rung, 'check') as checker, contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(Refusal):
                    runner.execute(self.root)
            self.assertEqual(raw.call_count, 2)
            checker.assert_not_called()
            self.assertIn('RUN FAIL logs=', output.getvalue())

    def test_negative_success_is_a_surprise(self):
        result = type('Result', (), dict(verdict='PASS', checks=[]))()
        with patch.object(run_kit, 'main', side_effect=[0, 0]), patch.object(check_rung, 'check', return_value=result) as checker, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'expected 1'):
                runner.execute(self.root, negative=True)
        checker.assert_not_called()

    def test_dry_never_requests_transport(self):
        with patch.object(run_kit, 'main', return_value=0) as raw, patch.object(check_rung, 'check') as checker, contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.root, dry_run=True)
        self.assertEqual(raw.call_count, 2)
        self.assertTrue(all('--dry-run' in call.args[0] for call in raw.call_args_list))
        checker.assert_not_called()

    def test_simulation_never_falls_back_to_transport(self):
        with patch.object(run_kit, 'main') as raw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'composed plan'):
                runner.execute(self.root, simulate=True)
        raw.assert_not_called()

    def test_live_inputs_run_steps_and_say_which_are_unchecked(self):
        self.live()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, inputs=self.inputs(-3, 7, 1))
        lines = output.getvalue().splitlines()
        self.assertEqual([line.split(' outputs=')[0] for line in lines[:3]],
                         ['STEP 000 UNCHECKED (no independent reference)', 'STEP 001 UNCHECKED (no independent reference)',
                          'STEP 002 EXACT (packaged reference)'])
        self.assertEqual(lines[3], f'RUN PASS (2 of 3 steps unchecked: no independent reference) logs={logs}')
        self.assertEqual((logs / 'session-terminal.txt').read_text(), '\n'.join(lines[:3] + ['SESSION PASS', 'rc=0', '']))
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        self.assertEqual([call.kwargs['negative'] for call in self.rungs.call_args_list], [False])
        with np.load(logs / 'step-001/outputs.npz') as outputs:
            self.assertEqual(outputs['output_000'].tolist(), [[7, 2], [3, 4]])
            self.assertEqual(outputs['output_001'].tolist(), [[5, 6], [7, 8]])

    def test_invalid_inputs_refuse_before_qualification(self):
        self.live()
        path, = self.inputs(7)
        np.savez(path, input_000=np.int8([7]), input_001=np.int8([7]))
        with patch.object(run_kit, 'main') as raw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'unexpected or missing'):
                runner.execute(self.root, inputs=[path])
        raw.assert_not_called()
        self.rungs.assert_not_called()

    def reference(self, value, outputs=None, **change):
        """An evaluator-shaped reference for input `value`, made as the kit's packaged record was (unless `change`d)."""
        import hashlib
        record = dict(status='EXPECTED', mode='source_capture', source_trace_sha256='1' * 64, implementation_sha256={'a.py': '2' * 64})
        kit = self.root / 'kit'
        if not (kit / 'reference/reference.json').exists():
            (kit / 'reference').mkdir(exist_ok=True)
            (kit / 'reference/reference.json').write_text(json.dumps(dict(record, evaluator_members={})))
            from test_chiprunner import inventory
            inventory(kit)
        directory = self.root / ('reference-%d-%d' % (value, len(list(self.root.glob('reference-*')))))
        directory.mkdir()
        outputs = outputs or [bytes([value % 256, 2, 3, 4]), b'\x05\x06\x07\x08']
        np.savez(directory / 'expected.npz', **{f'output_{i}': np.frombuffer(data, np.uint8).reshape(2, 2) for i, data in enumerate(outputs)})
        record.update(inputs=dict(input_000=hashlib.sha256(np.int8(value).tobytes()).hexdigest()),
                      expected_sha256=hashlib.sha256((directory / 'expected.npz').read_bytes()).hexdigest(), **change)
        (directory / 'reference.json').write_text(json.dumps(record))
        return directory

    def test_supplied_reference_checks_a_new_input(self):
        self.live()
        reference = self.reference(7)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, inputs=self.inputs(7), references=[reference])
        self.assertEqual(output.getvalue().splitlines(),
                         [f'STEP 000 EXACT ({reference} reference) outputs={logs}/step-000/outputs.npz', f'RUN PASS logs={logs}'])

    def test_references_for_other_inputs_sources_or_outputs_refuse_before_qualification(self):
        self.live()
        for reference, message in ((self.reference(6), 'made from other inputs'),
                                   (self.reference(7, source_trace_sha256='3' * 64), "not made from this kit's source"),
                                   (self.reference(7, status='REFUSED'), 'not an EXPECTED evaluator reference'),
                                   (self.reference(7, outputs=[b'\x07\x02\x03\x04']), 'outputs differ from the deployment')):
            with self.subTest(message=message), patch.object(run_kit, 'main') as raw, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(Refusal, message):
                    runner.execute(self.root, inputs=self.inputs(7), references=[reference])
            raw.assert_not_called()
        self.rungs.assert_not_called()

    def test_reference_made_against_a_copied_out_is_accepted_with_the_same_digests(self):
        # The evaluator names decision inputs by absolute path under the OUT it read; the bench's OUT is a copy.
        self.live()
        decisions = lambda root, receipt='1' * 64: dict(schema='atlas.source_selected_decisions.v1', identities={
            str(root / 'capture/capture_receipt.json'): receipt, str(root / 'compiled/application/manifest.json'): '2' * 64})
        kit = self.root / 'kit'
        (kit / 'reference').mkdir()
        (kit / 'reference/reference.json').write_text(json.dumps(dict(
            status='EXPECTED', mode='source_capture', source_trace_sha256='1' * 64, implementation_sha256={'a.py': '2' * 64},
            evaluator_members={}, compiler_decision_inputs=decisions(self.root))))
        from test_chiprunner import inventory
        inventory(kit)
        copied = Path('/bench/root/kit-out')
        reference = self.reference(7, compiler_decision_inputs=decisions(copied))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, inputs=self.inputs(7), references=[reference])
        self.assertIn(f'STEP 000 EXACT ({reference} reference)', output.getvalue())
        other = self.reference(7, compiler_decision_inputs=decisions(copied, receipt='3' * 64))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, "not made from this kit's source"):
            runner.execute(self.root, inputs=self.inputs(7), references=[other])

    def test_output_that_differs_from_its_reference_fails_the_run(self):
        self.live()
        reference = self.reference(7, outputs=[b'\x07\x02\x03\x05', b'\x05\x06\x07\x08'])
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaisesRegex(Refusal, 'step-000: output result0 differs'):
            runner.execute(self.root, inputs=self.inputs(7), references=[reference])
        self.assertIn('RUN FAIL logs=', output.getvalue())
        terminal, = self.root.glob('run-*/session-terminal.txt')
        self.assertEqual(terminal.read_text(), f'SESSION FAIL step-000: output result0 differs from the {reference} reference\nrc=1\n')
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_stop_queued_as_the_session_exits_still_fails(self):
        # SIGTERM as the run's exit starts is queued, then lands in the closing reference call or once the session closed.
        self.live()
        marker = self.root / 'signalled'
        def unhandled(*args):
            raise AssertionError('SIGTERM reached the test process handler')
        previous = signal.signal(signal.SIGTERM, unhandled); self.addCleanup(signal.signal, signal.SIGTERM, previous)
        with signal_at(runner.Run.__exit__.__code__, lambda frame, event: event == 'call', marker), \
                contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaisesRegex(Refusal, 'resident session stopped by SIGTERM'):
            runner.execute(self.root, inputs=self.inputs(7))
        self.assertTrue(marker.exists())
        self.assertIn('RUN FAIL logs=', output.getvalue())
        terminal, = self.root.glob('run-*/session-terminal.txt')
        self.assertTrue(terminal.read_text().endswith('\nSESSION TRANSPORT_ERROR resident session stopped by SIGTERM\nrc=2\n'))
        self.assertEqual(json.loads((terminal.parent / 'session/summary.json').read_text())['verdict'], 'TRANSPORT_ERROR')
        self.assertIs(signal.getsignal(signal.SIGTERM), unhandled)

    def test_stop_as_the_session_closes_fails_the_summary_and_the_terminal(self):
        # A stop queued inside the session's close, after it wrote its summary, or as the records are made.
        self.live()
        def unhandled(*args):
            raise AssertionError('SIGTERM reached the test process handler')
        previous = signal.signal(signal.SIGTERM, unhandled); self.addCleanup(signal.signal, signal.SIGTERM, previous)
        for name, code, event in (('close', run_kit.Session.close.__code__, 'call'), ('summary', run_kit.Session.close.__code__, 'return'),
                                  ('finish', runner.Run._finish.__code__, 'call')):
            with self.subTest(boundary=name):
                marker = self.fresh()
                with signal_at(code, lambda frame, kind: kind == event, marker), contextlib.redirect_stdout(io.StringIO()) as output, \
                        self.assertRaisesRegex(Refusal, 'resident session stopped by SIGTERM'):
                    runner.execute(self.root, inputs=self.inputs(7))
                self.assertTrue(marker.exists())
                self.assertIn('RUN FAIL logs=', output.getvalue())
                terminal, = self.root.glob('run-*/session-terminal.txt')
                self.assertTrue(terminal.read_text().endswith('\nSESSION TRANSPORT_ERROR resident session stopped by SIGTERM\nrc=2\n'))
                summary = json.loads((terminal.parent / 'session/summary.json').read_text())
                self.assertEqual((summary['verdict'], summary['error']), ('TRANSPORT_ERROR', 'resident session stopped by SIGTERM'))
                self.assertIs(signal.getsignal(signal.SIGTERM), unhandled)

    def test_signal_after_the_records_are_made_reaches_the_restored_handler(self):
        # From the seal on no stop is accepted; the signal is raised again once the records are complete and the handlers restored.
        writing = terminal_line()
        self.live()
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        for name, code, when in (('terminal', runner.Run._finish.__code__, lambda frame, kind: kind == 'line' and frame.f_lineno == writing),
                                 ('restoring', runner._Stop.close.__code__, lambda frame, kind: kind == 'call')):
            with self.subTest(boundary=name):
                marker = self.fresh()
                received = []
                def restored(number, frame):
                    received.append((number, next(self.root.glob('run-*/session-terminal.txt')).read_text()))
                signal.signal(signal.SIGTERM, restored)
                with signal_at(code, when, marker), contextlib.redirect_stdout(io.StringIO()) as output:
                    logs = runner.execute(self.root, inputs=self.inputs(7))
                self.assertTrue(marker.exists())
                record = (logs / 'session-terminal.txt').read_text()
                self.assertTrue(record.endswith('\nSESSION PASS\nrc=0\n'), record)
                self.assertEqual(received, [(signal.SIGTERM, record)])
                self.assertIn('RUN PASS', output.getvalue())
                self.assertEqual(json.loads((logs / 'session/summary.json').read_text())['verdict'], 'PASS')

    def fresh(self):
        """Remove earlier runs and the signal marker; return the marker."""
        (self.root / 'signalled').unlink(missing_ok=True)
        for path in self.root.glob('run-*'):
            shutil.rmtree(path)
        return self.root / 'signalled'

    def test_close_that_raises_still_ends_the_records_and_restores_handlers(self):
        self.live()
        close = run_kit.Session.close
        def failing(session, *args):
            close(session, *args)
            raise OSError('close failed')
        with patch.object(run_kit.Session, 'close', failing), contextlib.redirect_stdout(io.StringIO()) as output, \
                self.assertRaisesRegex(Refusal, 'close failed'):
            runner.execute(self.root, inputs=self.inputs(7))
        self.assertIn('RUN FAIL logs=', output.getvalue())
        terminal, = self.root.glob('run-*/session-terminal.txt')
        self.assertTrue(terminal.read_text().endswith('\nSESSION TRANSPORT_ERROR close failed\nrc=2\n'))
        summary = json.loads((terminal.parent / 'session/summary.json').read_text())
        self.assertEqual((summary['verdict'], summary['error']), ('TRANSPORT_ERROR', 'close failed'))
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    def test_stdin_loop_runs_one_step_per_line(self):
        self.live()
        first, second = self.inputs(7, -3)
        reference = self.reference(7)
        with patch.object(sys, 'stdin', io.StringIO(f'{first} {reference}\n{second}\n')), contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, inputs=['-'])
        self.assertEqual([line.split(' outputs=')[0] for line in output.getvalue().splitlines()],
                         [f'STEP 000 EXACT ({reference} reference)', 'STEP 001 UNCHECKED (no independent reference)',
                          f'RUN PASS (1 of 2 steps unchecked: no independent reference) logs={logs}'])
        for text, message in ((f'{first} {reference} extra\n', 'INPUTS \\[REFERENCE\\]'),):
            with patch.object(sys, 'stdin', io.StringIO(text)), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refusal, message):
                runner.execute(self.root, inputs=['-'])
        with self.assertRaisesRegex(Refusal, 'name each reference on its stdin line'):
            runner.execute(self.root, inputs=['-'], references=[reference])
        with self.assertRaisesRegex(Refusal, 'one --reference per --inputs file'):
            runner.execute(self.root, inputs=self.inputs(7, 7), references=[reference])

    def test_staged_runner_drift_refuses(self):
        (self.root / 'kit/bench/run_kit.py').write_text('raise RuntimeError()')
        with patch.object(run_kit, 'main') as raw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'contract changed'):
                runner.execute(self.root, dry_run=True)
        raw.assert_not_called()

    def test_actual_dry_front_door_with_inventoried_fixture(self):
        from test_chiprunner import synthetic
        synthetic(self.root / 'kit')
        with patch.object(run_kit, 'UART', side_effect=AssertionError('transport')), patch.object(run_kit.subprocess, 'Popen', side_effect=AssertionError('process')), contextlib.redirect_stdout(io.StringIO()) as capture:
            logs = runner.execute(self.root, dry_run=True)
        self.assertIn('RUN DRY PASS', capture.getvalue())
        self.assertTrue((logs / 'negative-preflight.log').exists())

    def inputs(self, *values, dtype=np.int8):
        result = []
        for value in values:
            path = self.root / ('inputs-%d.npz' % len(list(self.root.glob('inputs-*.npz'))))
            np.savez(path, input_000=np.array([value], dtype=dtype))
            result.append(path)
        return result

    def live(self, **behavior):
        from test_chiprunner_inputs import live_kit, peers
        manifest = live_kit(self.root / 'kit')
        patches, log = peers(manifest, **behavior)
        stack = contextlib.ExitStack(); self.addCleanup(stack.close)
        for item in patches: stack.enter_context(item)
        self.rungs = stack.enter_context(patch.object(runner, '_rung'))
        return log

    def steps(self, *paths):
        with contextlib.redirect_stdout(io.StringIO()), runner.Run(self.root) as run:
            self.run_logs = run.logs
            for path in paths:
                run.step(path)
        return run.logs

    def test_explicit_steps_share_one_qualified_resident_session(self):
        log = self.live()
        logs = self.steps(*self.inputs(-3, 7))
        self.assertEqual([call.kwargs['negative'] for call in self.rungs.call_args_list], [False])
        self.assertEqual((logs / 'step-000/output-0.bin').read_bytes(), b'\xfd\x02\x03\x04')
        self.assertEqual((logs / 'step-001/output-0.bin').read_bytes(), b'\x07\x02\x03\x04')
        self.assertEqual((logs / 'reference-close/output-0.bin').read_bytes(), b'\x01\x02\x03\x04')
        self.assertEqual(log.count('reset halt'), 1)
        self.assertEqual(json.loads((logs / 'session/summary.json').read_text())['verdict'], 'PASS')

    def test_rung_and_resident_session_use_the_selected_serial(self):
        self.live()
        seen, peer = [], run_kit.UART
        class Recording(peer):
            def __init__(self, path, *args): seen.append(path); super().__init__(path, *args)
        with patch.object(run_kit, 'UART', Recording), contextlib.redirect_stdout(io.StringIO()), \
                runner.Run(self.root, serial='/dev/cu.usbserial-1') as run:
            run.step(self.inputs(-3)[0])
        self.assertEqual(seen, ['/dev/cu.usbserial-1'])
        self.assertEqual(self.rungs.call_args.kwargs['serial'], ('--serial', '/dev/cu.usbserial-1'))

    def test_live_step_dump_failure_is_transport_error_and_stops(self):
        from atlas_mlir.chiprunner.tcl_rpc import TransportError
        self.live(fail_dump_after=1)
        with self.assertRaisesRegex(TransportError, 'dump connection lost'):
            self.steps(*self.inputs(-3, 7))
        logs = self.run_logs
        self.assertFalse((logs / 'step-001').exists())
        summary = json.loads((logs / 'session/summary.json').read_text())
        self.assertEqual((summary['verdict'], summary['error']), ('TRANSPORT_ERROR', 'dump connection lost'))

    def test_dry_inputs_are_validated_without_steps(self):
        self.live()
        with patch.object(run_kit, 'UART', side_effect=AssertionError('transport')), contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, dry_run=True, inputs=self.inputs(-3, 7))
        self.assertIn('RUN DRY PASS', output.getvalue())
        self.assertEqual(sorted(path.name for path in logs.iterdir()), ['negative-preflight.log', 'preflight.log'])
        self.rungs.assert_not_called()

    def test_wrong_input_dtype_refuses_before_any_transport(self):
        self.live()
        with patch.object(run_kit, 'UART', side_effect=AssertionError('transport')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Refusal, 'shape or dtype'):
                runner.execute(self.root, dry_run=True, inputs=self.inputs(7, dtype=np.uint8))
        self.rungs.assert_not_called()

    def test_dry_step_elf_plan_never_opens_transport(self):
        from test_chiprunner_live_steps import step_plan, peers
        patches, log, processes = peers(step_plan(self.root / 'kit'), self.root / 'kit')
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
            for item in patches: stack.enter_context(item)
            runner.execute(self.root, dry_run=True)
        self.assertIn('RUN DRY PASS', output.getvalue())
        self.assertEqual((log, processes), ([], []))

    def test_live_step_elf_plan_runs_one_rung_per_command(self):
        from test_chiprunner_live_steps import step_plan, peers
        manifest = step_plan(self.root / 'kit')
        for negative in (False, True):
            patches, log, processes = peers(manifest, self.root / 'kit')
            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
                for item in patches: stack.enter_context(item)
                logs = runner.execute(self.root, negative=negative)
            self.assertIn('RUN PASS (negative rejected as expected) logs=' if negative else 'RUN PASS logs=', output.getvalue())
            self.assertEqual(sum(text.startswith('resume') for text in log), 1 if negative else 2)
            self.assertEqual(json.loads((logs / ('negative' if negative else 'positive') / 'journal.json').read_text())['identity']['backend'], 'jtag')
            self.assertEqual(len(processes), 1)
            terminal = 'KIT FAIL boundary mismatch: s0/o (value m)\nrc=1\n' if negative else 'KIT PASS every step boundary matches its reference\nrc=0\n'
            self.assertEqual((logs / (('negative' if negative else 'positive') + '-terminal.txt')).read_text(), terminal)

    def live_step_rung(self, manifest, message, negative=False, **behavior):
        # One failing live step ELF rung through the front door over the in-memory peers; returns its terminal file.
        from test_chiprunner_live_steps import peers
        patches, log, processes = peers(manifest, self.root / 'kit', **behavior)
        handlers = [signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)]
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
            for item in patches: stack.enter_context(item)
            with self.assertRaisesRegex(Refusal, message):
                runner.execute(self.root, negative=negative)
        logs = Path(output.getvalue().split('RUN FAIL logs=')[1].strip())
        self.assertEqual(handlers, [signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
        self.assertTrue(processes[0].stopped)
        self.assertEqual(log[-2:], ['rpc closed', 'uart closed'])
        return (logs / (('negative' if negative else 'positive') + '-terminal.txt')).read_text()

    def test_live_step_elf_rung_stops_on_the_cap_and_signals(self):
        # Two steps: 300 s, two 120 s plan timeouts, and 18 Tcl commands at 120 s each.
        from test_chiprunner_live_steps import step_plan
        manifest, armed = step_plan(self.root / 'kit'), []
        for number, reason in ((signal.SIGALRM, 'runner exceeded 2700 s and was stopped'), (signal.SIGTERM, 'runner stopped by SIGTERM'),
                               (signal.SIGINT, 'runner stopped by SIGINT')):
            with self.subTest(signal=number.name):
                def deliver():
                    armed.append(signal.getitimer(signal.ITIMER_REAL)[0])
                    os.kill(os.getpid(), number)
                terminal = self.live_step_rung(manifest, 'positive: ' + reason, during_records=deliver)
                self.assertEqual(terminal, 'KIT TRANSPORT_ERROR %s\nrc=2\n' % reason)
        self.assertTrue(all(2699 < seconds <= 2700 for seconds in armed), armed)

    def test_live_step_elf_rung_cleanup_runs_every_step_when_the_first_raises(self):
        from test_chiprunner_live_steps import step_plan
        terminal = self.live_step_rung(step_plan(self.root / 'kit'), 'positive: Tcl close failed', close_fails=True)
        self.assertEqual(terminal, 'KIT TRANSPORT_ERROR Tcl close failed\nrc=2\n')

    def test_live_step_elf_nonzero_uart_status_is_a_named_transport_error(self):
        from test_chiprunner_live_steps import step_plan
        terminal = self.live_step_rung(step_plan(self.root / 'kit'), 'positive: UART reported nonzero status', status=3)
        self.assertEqual(terminal, 'KIT TRANSPORT_ERROR UART reported nonzero status\nrc=2\n')

    def test_live_step_elf_negative_cleanup_failure_is_a_transport_error(self):
        # The expected boundary mismatch never hides a cleanup failure: rc=2 and RUN FAIL, naming both.
        from test_chiprunner_live_steps import step_plan
        terminal = self.live_step_rung(step_plan(self.root / 'kit'), 'negative: Tcl close failed', negative=True, close_fails=True)
        self.assertEqual(terminal, 'KIT TRANSPORT_ERROR Tcl close failed (after boundary mismatch: s0/o (value m))\nrc=2\n')

    def test_live_step_elf_stop_before_the_session_stores_its_transport_is_queued(self):
        # SIGTERM after the transport (which owns OpenOCD) is built and before composed.Session._open stores it.
        from atlas_mlir.chiprunner import composed
        from test_chiprunner_live_steps import step_plan
        manifest, code = step_plan(self.root / 'kit'), composed.Session._open.__code__
        offsets = {item.offset for item in dis.get_instructions(code) if item.opname == 'STORE_ATTR' and item.argval == 'transport'}
        self.assertEqual(len(offsets), 1)
        marker = self.root / 'signalled'
        with signal_at(code, lambda frame, event: event == 'opcode' and frame.f_lasti in offsets, marker):
            terminal = self.live_step_rung(manifest, 'positive: runner stopped by SIGTERM')
        self.assertTrue(marker.exists())
        self.assertRegex(terminal, '(^|\n)KIT TRANSPORT_ERROR runner stopped by SIGTERM( after the runner returned)?\nrc=2\n$')

    def test_live_step_elf_stop_as_the_session_closes_is_queued(self):
        # SIGTERM as composed.Session.close starts, before it reaches the transport's own close.
        from atlas_mlir.chiprunner import composed
        from test_chiprunner_live_steps import step_plan
        manifest, marker = step_plan(self.root / 'kit'), self.root / 'signalled'
        with signal_at(composed.Session.close.__code__, lambda frame, event: event == 'call', marker):
            terminal = self.live_step_rung(manifest, 'positive: runner stopped by SIGTERM')
        self.assertTrue(marker.exists())
        self.assertRegex(terminal, '(^|\n)KIT TRANSPORT_ERROR runner stopped by SIGTERM( after the runner returned)?\nrc=2\n$')

    def stops_entering_cleanup(self, code, rung, behaviors):
        """SIGTERM at each bytecode `code` runs after its outermost try body on each path that `rung(**behavior)`
        takes: the normal return, the exception unwinding, and the cleanup call itself. Every rung must end rc=2
        with its OpenOCD stopped and the handlers restored. `rung` returns (stdout, terminal text, peers stopped)."""
        end = try_body_end(code)
        handlers, delivered = lambda: [signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)], 0
        before = handlers()
        for behavior in behaviors:
            seen, marker = [], self.root / 'signalled'
            marker.unlink(missing_ok=True)
            with signal_at(code, lambda frame, event: event == 'opcode' and seen.append(frame.f_lasti), marker):
                rung(**behavior)
            offsets = sorted({offset for offset in seen if offset >= end})
            self.assertTrue(offsets, behavior)
            for offset in offsets:
                with self.subTest(behavior=behavior, offset=offset):
                    marker.unlink(missing_ok=True)
                    with signal_at(code, lambda frame, event: event == 'opcode' and frame.f_lasti == offset, marker):
                        output, terminal, stopped = rung(**behavior)
                    self.assertTrue(marker.exists())
                    self.assertIn('RUN FAIL logs=', output)
                    self.assertRegex(terminal, '(^|\n)KIT TRANSPORT_ERROR runner stopped by SIGTERM( after the runner returned)?\nrc=2\n$')
                    self.assertTrue(stopped)
                    self.assertEqual((handlers(), signal.getitimer(signal.ITIMER_REAL)[0]), (before, 0))
                    delivered += 1
        return delivered

    def step_elf_rung(self, **behavior):
        from test_chiprunner_live_steps import peers
        patches, log, processes = peers(json.loads((self.root / 'kit/manifest.json').read_text()), self.root / 'kit', **behavior)
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
            for item in patches: stack.enter_context(item)
            with contextlib.suppress(Refusal):
                runner.execute(self.root)
        logs = Path(output.getvalue().split(' logs=')[1].strip())
        stopped = len(processes) == 1 and processes[0].stopped and log[-2:] == ['rpc closed', 'uart closed']
        return output.getvalue(), (logs / 'positive-terminal.txt').read_text(), stopped

    def test_live_step_elf_stop_entering_cleanup_is_queued(self):
        from atlas_mlir.chiprunner import composed
        from test_chiprunner_live_steps import step_plan
        step_plan(self.root / 'kit')
        self.assertGreater(self.stops_entering_cleanup(composed.execute.__code__, self.step_elf_rung, ({}, dict(wrong_hash=True))), 6)

    def test_live_stop_entering_cleanup_is_queued(self):
        from test_chiprunner_inputs import peers
        def rung(**behavior):
            patches, log = peers(json.loads((self.root / 'kit/manifest.json').read_text()), **behavior)
            processes, Process = [], next(item.new for item in patches if item.attribute == 'Popen')
            class Recorded(Process):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs); self.stopped = False; processes.append(self)
                def terminate(self): self.stopped = True
            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
                for item in patches: stack.enter_context(item)
                stack.enter_context(patch.object(run_kit.subprocess, 'Popen', Recorded))
                with contextlib.suppress(Refusal):
                    runner.execute(self.root)
            logs = Path(output.getvalue().split(' logs=')[1].strip())
            terminal = next(logs.glob('positive-*-terminal.txt'))
            summary = json.loads((logs / terminal.name.replace('-terminal.txt', '') / 'summary.json').read_text())
            return output.getvalue(), terminal.read_text(), len(processes) == 1 and processes[0].stopped and 'verdict' in summary
        self.assertGreater(self.stops_entering_cleanup(run_kit.execute.__code__, rung, ({}, dict(status=3))), 6)

    def test_live_step_elf_stop_before_the_openocd_handle_is_stored_is_queued(self):
        from atlas_mlir.chiprunner.composed_tcl import TclTransport
        from test_chiprunner_live_steps import step_plan
        manifest, code = step_plan(self.root / 'kit'), TclTransport.__init__.__code__
        offsets = {item.offset for item in dis.get_instructions(code) if item.opname == 'STORE_ATTR' and item.argval == 'process'}
        marker = self.root / 'signalled'
        # The first STORE_ATTR sets process to None; the stop lands at the second, after OpenOCD started and before it is stored.
        late = max(offsets)
        with signal_at(code, lambda frame, event: event == 'opcode' and frame.f_lasti == late, marker):
            terminal = self.live_step_rung(manifest, 'positive: runner stopped by SIGTERM')
        self.assertTrue(marker.exists())
        self.assertRegex(terminal, '(^|\n)KIT TRANSPORT_ERROR runner stopped by SIGTERM( after the runner returned)?\nrc=2\n$')

    def test_live_step_surprises_stop_and_transport_errors_never_qualify_the_negative(self):
        from test_chiprunner_live_steps import step_plan, peers, write
        manifest = step_plan(self.root / 'kit')
        (self.root / 'kit/f.bin').write_bytes(b'\x03\x04\x05\x07'); write(self.root / 'kit', manifest)
        for negative, behavior, message in ((False, {}, 'positive: boundary mismatch: s1/o'), (True, dict(wrong_hash=True), 'dump differs')):
            patches, log, processes = peers(manifest, self.root / 'kit', **behavior)
            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as output:
                for item in patches: stack.enter_context(item)
                with self.assertRaisesRegex(Refusal, message):
                    runner.execute(self.root, negative=negative)
            self.assertIn('RUN FAIL logs=', output.getvalue())

    def test_live_composed_plans_without_step_elves_refuse(self):
        from test_chiprunner_inputs import composed_kit
        composed_kit(self.root / 'kit')
        with patch.object(run_kit, 'UART', side_effect=AssertionError('transport')):
            with self.assertRaisesRegex(Refusal, 'needs a linked step ELF for every region'):
                runner.execute(self.root)

    def test_simulated_steps_through_front_door(self):
        from atlas_mlir.chiprunner import composed_sim
        from test_chiprunner_inputs import composed_kit, CountingTransport
        composed_kit(self.root / 'kit')
        CountingTransport.instances = []
        with patch.object(composed_sim, 'SimTransport', CountingTransport), \
                patch.object(composed_sim, 'runtime_identity', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            logs = runner.execute(self.root, simulate=True, inputs=self.inputs(-3, 7, 2))
        self.assertEqual([line.split(' outputs=')[0] for line in output.getvalue().splitlines()],
                         ['STEP 000 UNCHECKED (no independent reference)', 'STEP 001 UNCHECKED (no independent reference)',
                          'STEP 002 EXACT (packaged reference)', f'RUN SIM PASS (2 of 3 steps unchecked: no independent reference) logs={logs}'])
        self.assertEqual((logs / 'step-000/output-0.bin').read_bytes(), bytes([(253 * 3 + 1) % 256]))
        self.assertEqual((logs / 'step-001/output-0.bin').read_bytes(), bytes([22]))
        session, = CountingTransport.instances  # the resident reference-open is the positive; no qualification session
        self.assertEqual(session.writes.count(0x200c0001), 1)
        self.assertFalse((logs / 'qualification').exists())
        self.assertTrue((logs / 'reference-close/b-output-0.bin').is_file())  # steps were unchecked


class Hung:
    """UART peer that never completes; records the armed cap, writes one stderr line, and marks readiness."""
    marker = None
    def __init__(self, *args): pass
    def clear(self): pass
    def close(self): pass
    def records(self, timeout, console, tag):
        from atlas_mlir.chiprunner.tcl_rpc import TransportError
        print('peer waiting on UART', file=sys.stderr)
        Hung.marker.write_text(str(signal.getitimer(signal.ITIMER_REAL)[0]))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(.05)
        raise TransportError('UART completion timeout')


def live_rung(root, scenario):
    """One live positive rung over in-memory Tcl and the stub OpenOCD at NOV_OPENOCD (a real child process).

    `hang` uses a UART that never completes. The others signal SIGTERM
    themselves: `assign` after OpenOCD has started but before the session
    stores its process, `open` as the terminal file is opened, `cleanup` (a
    completed rung) as the runner leaves its try body for cleanup, `close` (a
    completed rung) while the runner closes the Tcl connection, and
    `restored` (a completed rung) once the handlers are restored. `double`
    sends SIGINT while the runner closes after a first stop, and `evidence`
    turns a stop into a numerical failure, as a UART hash mismatch would.
    """
    from test_chiprunner_inputs import peers
    patches, _ = peers(json.loads((root / 'kit/manifest.json').read_text()))
    keep = ('live', 'UART') if scenario == 'close' else ('live', 'UART', 'TclRPC') if scenario in ('restored', 'cleanup') else ('live', 'TclRPC')
    rpc = next(item.new for item in patches if item.attribute == 'TclRPC')
    class Closing(rpc):
        def close(self):
            os.kill(os.getpid(), signal.SIGINT if scenario == 'double' else signal.SIGTERM)
            time.sleep(.2)
    Hung.marker = root / 'waiting'
    extra = [patch.object(run_kit, 'UART', Hung)] if 'UART' not in keep else []
    extra += [patch.object(run_kit, 'TclRPC', Closing)] if scenario in ('close', 'double') else []
    extra += [patch.object(run_kit, 'numerical_evidence', lambda *args: 'output differs')] if scenario == 'evidence' else []
    marker = root / 'signalled'
    if scenario == 'assign':
        code = run_kit.Session.open.__code__
        offsets = {item.offset for item in dis.get_instructions(code) if item.opname == 'STORE_ATTR' and item.argval == 'process'}
        assert len(offsets) == 1, offsets
        extra.append(signal_at(code, lambda frame, event: event == 'opcode' and frame.f_lasti in offsets, marker))
    if scenario == 'open':
        extra.append(signal_at(Path.open.__code__, lambda frame, event: frame.f_locals['self'].name.endswith('-terminal.txt'), marker))
    if scenario == 'restored':
        extra.append(signal_at(runner._Stop.close.__code__, lambda frame, event: event == 'return', marker))
    if scenario == 'cleanup':
        end = try_body_end(run_kit.execute.__code__)
        extra.append(signal_at(run_kit.execute.__code__, lambda frame, event: event == 'opcode' and frame.f_lasti >= end, marker))
    with contextlib.ExitStack() as stack:
        for item in [item for item in patches if item.attribute in keep] + extra:
            stack.enter_context(item)
        return runner.execute(root)


def terminal_line():
    """The line of Run._finish that writes the session's verdict and rc= to its terminal record."""
    import inspect
    lines, start = inspect.getsourcelines(runner.Run._finish)
    return start + next(index for index, text in enumerate(lines) if 'self.terminal.write(' in text)


def try_body_end(code):
    """The bytecode offset where `code`'s outermost try body ends; later offsets return, unwind, and clean up."""
    entries = dis.Bytecode(code).exception_entries
    return max(entry.end for entry in entries if entry.target == entries[0].target)


@contextlib.contextmanager
def signal_at(code, when, marker):
    """Deliver SIGTERM once, at the first trace event in `code` for which `when(frame, event)` holds.

    An `opcode` event fires before its bytecode executes; `return` fires as the function returns.
    """
    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        frame.f_trace_opcodes = True
        if not marker.exists() and when(frame, event):
            marker.write_text(event)
            os.kill(os.getpid(), signal.SIGTERM)
        return trace
    sys._getframe().f_trace_opcodes = True  # Python 3.12 enables opcode events only when set before settrace
    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(None)


class StopTests(unittest.TestCase):
    """Wall-clock cap and signal cleanup in a child process with a stub OpenOCD; never the real binary or a board."""
    setUp = RunTests.setUp

    def stub(self):
        stub = self.root / 'openocd-stub'
        stub.write_text('#!/bin/sh\necho $$ > "$0.pid"\nexec sleep 60\n')
        stub.chmod(0o755)
        return stub

    def openocd_stopped(self):
        deadline, written = time.monotonic() + 10, lambda: (self.root / 'openocd-stub.pid').exists() and (self.root / 'openocd-stub.pid').read_text()
        while not written() and time.monotonic() < deadline - 9:
            time.sleep(.05)
        # The child has exited; a stub it started and leaked would have written its pid by now. A stub stopped after its
        # shell created the pid file but before it wrote the pid leaves the file empty.
        if not written():
            return True
        pid = int((self.root / 'openocd-stub.pid').read_text())
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            stat = Path('/proc/%d/stat' % pid)  # an exited, not yet reaped stub is stopped
            if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return True
            time.sleep(.05)
        os.kill(pid, signal.SIGKILL)
        return False

    def child(self, scenario, signum=None, returncode=1):
        for path in [self.root / 'waiting', self.root / 'signalled', self.root / 'openocd-stub.pid', *self.root.glob('run-*')]:
            subprocess.run(['rm', '-rf', str(path)], check=True)
        code = ('import sys, signal; sys.path.insert(0, sys.argv[1]); signal.signal(signal.SIGINT, signal.default_int_handler); '
                'import test_deployment_run as t; from pathlib import Path; t.live_rung(Path(sys.argv[2]), sys.argv[3])')
        process = subprocess.Popen([sys.executable, '-B', '-c', code, str(Path(__file__).parent), str(self.root), scenario],
                                   env=dict(os.environ, NOV_OPENOCD=str(self.stub())), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if signum is not None:
            deadline = time.monotonic() + 60
            while not ((self.root / 'waiting').exists() and (self.root / 'openocd-stub.pid').exists()) and time.monotonic() < deadline:
                time.sleep(.05)
            process.send_signal(signum)
        output, _ = process.communicate(timeout=120)
        self.assertTrue(self.openocd_stopped(), output)
        self.assertEqual(process.returncode, returncode, output)
        capture = next(self.root.glob('run-*/positive-*-terminal.txt'), None)
        self.assertIsNotNone(capture, output)
        summary = capture.parent / capture.name.replace('-terminal.txt', '') / 'summary.json'
        if returncode == 1:
            self.assertIn('RUN FAIL logs=', output)
        return capture.read_text(), json.loads(summary.read_text())['verdict'] if summary.exists() else None

    def test_stub_stopped_before_writing_its_pid_counts_as_stopped(self):
        (self.root / 'openocd-stub.pid').write_text('')  # stopped between the shell's redirection and its echo
        self.assertTrue(self.openocd_stopped())

    def test_cap_stops_runner_and_openocd(self):
        # The child delivers the armed cap's SIGALRM early; the cap itself equals the run.sh bound for this kit.
        terminal, verdict = self.child('hang', signal.SIGALRM)
        self.assertTrue(1500 < float((self.root / 'waiting').read_text()) <= 1505)
        self.assertEqual(terminal, 'peer waiting on UART\nKIT TRANSPORT_ERROR runner exceeded 1505 s and was stopped\nrc=2\n')
        self.assertEqual(verdict, 'TRANSPORT_ERROR')

    def test_signals_stop_runner_and_openocd_and_keep_terminal(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum.name):
                terminal, verdict = self.child('hang', signum)
                self.assertEqual(terminal, 'peer waiting on UART\nKIT TRANSPORT_ERROR runner stopped by %s\nrc=2\n' % signum.name)
                self.assertEqual(verdict, 'TRANSPORT_ERROR')

    def test_signal_before_openocd_handle_is_stored_is_queued(self):
        # SIGTERM lands after Popen returned and before Session.open stores the process; cleanup must still own and stop it.
        terminal, verdict = self.child('assign')
        self.assertTrue((self.root / 'signalled').exists())
        self.assertTrue(terminal.endswith('KIT TRANSPORT_ERROR runner stopped by SIGTERM\nrc=2\n'), terminal)
        self.assertEqual(verdict, 'TRANSPORT_ERROR')

    def test_signal_as_terminal_opens_is_queued(self):
        terminal, _ = self.child('open')
        self.assertTrue((self.root / 'signalled').exists())
        self.assertRegex(terminal, '(^|\n)KIT TRANSPORT_ERROR runner stopped by SIGTERM\nrc=2\n$')

    def test_signal_entering_cleanup_is_queued(self):
        terminal, verdict = self.child('cleanup')
        self.assertTrue((self.root / 'signalled').exists())
        self.assertRegex(terminal, '^(KIT PASS UART and dumped bytes match frozen reference\n)?KIT TRANSPORT_ERROR runner stopped by SIGTERM( after the runner returned)?\nrc=2\n$')
        self.assertEqual(verdict, 'PASS')

    def test_second_interrupt_during_cleanup_is_ignored(self):
        # A user's double Ctrl-C: the second SIGINT lands while the runner closes after the first stop.
        terminal, verdict = self.child('double', signal.SIGINT)
        self.assertEqual(terminal, 'peer waiting on UART\nKIT TRANSPORT_ERROR runner stopped by SIGINT\nrc=2\n')
        self.assertEqual(verdict, 'TRANSPORT_ERROR')

    def test_stop_after_numerical_failure_is_rc_2(self):
        terminal, verdict = self.child('evidence', signal.SIGTERM)
        self.assertEqual(terminal, 'peer waiting on UART\nKIT FAIL output differs (then transport: runner stopped by SIGTERM)\nrc=2\n')
        self.assertEqual(verdict, 'FAIL')

    def test_terminal_is_complete_before_handlers_are_restored(self):
        # A SIGTERM after the handlers are restored ends the process as usual; the terminal file is already complete.
        terminal, verdict = self.child('restored', returncode=-signal.SIGTERM)
        self.assertTrue((self.root / 'signalled').exists())
        self.assertEqual(terminal, 'KIT PASS UART and dumped bytes match frozen reference\nrc=0\n')
        self.assertEqual(verdict, 'PASS')

    def test_cap_never_replaces_a_stop_queued_during_install(self):
        install = signal.signal
        def installing(number, handler):
            previous = install(number, handler)
            if number == signal.SIGTERM and handler != previous:
                os.kill(os.getpid(), signal.SIGTERM)
            return previous
        with patch.object(signal, 'signal', installing):
            stop = runner._Stop(1505)
        try:
            self.assertEqual(stop.reason, 'runner stopped by SIGTERM')
            self.assertLessEqual(signal.getitimer(signal.ITIMER_REAL)[0], .05)
        finally:
            stop.close()

    def test_close_never_rearms_the_timer(self):
        stray = signal.signal(signal.SIGALRM, lambda *args: None)  # a stray alarm must not end this test process
        try:
            stop, arm = runner._Stop(1505), signal.setitimer
            def cancelling(which, seconds, *rest):
                result = arm(which, seconds, *rest)
                if seconds == 0:
                    os.kill(os.getpid(), signal.SIGTERM)  # pending as the timer is cancelled
                return result
            with patch.object(signal, 'setitimer', cancelling):
                stop.close()
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertIsNone(stop.reason)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, stray)

    def test_external_alarm_stops_a_resident_session_with_a_named_reason(self):
        from atlas_mlir.chiprunner.tcl_rpc import TransportError
        stop = runner._Resident(); stop.install()
        try:
            with self.assertRaisesRegex(TransportError, '^resident session stopped by SIGALRM$'):
                signal.raise_signal(signal.SIGALRM)
            self.assertEqual(stop.seal(), 'resident session stopped by SIGALRM')
        finally:
            stop.close()

    def test_signal_during_cleanup_waits_for_openocd_stop(self):
        terminal, verdict = self.child('close')
        # The deferred stop lands after cleanup, inside or just after the runner; both end as a transport error.
        lines = terminal.splitlines()
        self.assertEqual((lines[0], lines[-1], len(lines)), ('KIT PASS UART and dumped bytes match frozen reference', 'rc=2', 3))
        self.assertRegex(lines[1], '^KIT TRANSPORT_ERROR runner stopped by SIGTERM')
        self.assertEqual(verdict, 'PASS')


def resident_session(root, scenario):
    """A live run with resident steps over in-memory Tcl and the stub OpenOCD at NOV_OPENOCD (a real child process).

    The qualification rung is stubbed out (StopTests covers it). `open`, `call`, and `close` run one listed step on a UART
    that never completes the reference call at open, the step, or the reference call at close; `stdin` reads steps from
    stdin on a UART that completes, so the parent can stop the run while it waits for the next line; `sealed` completes
    one step and sends itself SIGTERM as the session's verdict is written to its terminal record.
    """
    from test_chiprunner_inputs import peers
    from atlas_mlir.chiprunner.tcl_rpc import TransportError
    patches, _ = peers(json.loads((root / 'kit/manifest.json').read_text()))
    uart = next(item.new for item in patches if item.attribute == 'UART')
    hang = dict(open=1, call=2, close=3).get(scenario)
    class Hanging(uart):
        calls = 0
        def records(self, timeout, console, tag):
            Hanging.calls += 1
            if Hanging.calls != hang:
                return super().records(timeout, console, tag)
            (root / 'waiting').write_text('')
            time.sleep(30)
            raise TransportError('UART completion timeout')
    path = root / 'inputs.npz'
    np.savez(path, input_000=np.int8([-3]))
    with contextlib.ExitStack() as stack:
        for item in [item for item in patches if item.attribute in ('live', 'TclRPC')]:
            stack.enter_context(item)
        stack.enter_context(patch.object(run_kit, 'UART', Hanging))
        stack.enter_context(patch.object(runner, '_rung'))
        if scenario == 'sealed':
            writing = terminal_line()
            stack.enter_context(signal_at(runner.Run._finish.__code__, lambda frame, event: event == 'line' and frame.f_lineno == writing,
                                          root / 'signalled'))
        return runner.execute(root, inputs=['-'] if scenario == 'stdin' else [path])


def step_kit(root):
    """The live step-ELF plan with a deployment: the first step's input as one u8[4] tensor, the last step's output as another."""
    from test_chiprunner_live_steps import BASE, step_plan, write
    manifest = step_plan(root)
    first, last = manifest['composed']['regions']
    first['inputs'] = [dict(id='i', address=BASE, bytes=4, file='x.bin')]
    tensor = dict(index=0, bytes=4, type=dict(dtype='u8', shape=[4]))
    manifest['deployment'] = dict(window=dict(address=BASE, bytes=4096), inputs=[dict(tensor, id='x', address=BASE)],
                                  outputs=[dict(tensor, id='f', address=last['outputs'][0]['address'], source=dict(region='s1', output='o'))])
    write(root, manifest)


def composed_session(root, scenario):
    """A live resident session of linked step ELFs over in-memory Tcl and UART and the stub OpenOCD at NOV_OPENOCD (a real child process).

    The qualification rung is stubbed out. `pass` runs one listed step; `stored` sends itself SIGTERM as the Tcl transport
    returns, before the session stores it; `stdin` reads steps from stdin, so the parent can stop the run while it waits.
    """
    from test_chiprunner_live_steps import peers
    from atlas_mlir.chiprunner import composed_tcl
    patches, _, _ = peers(json.loads((root / 'kit/manifest.json').read_text()), root / 'kit')
    with contextlib.ExitStack() as stack:
        for item in patches:
            if item.attribute != 'Popen':
                stack.enter_context(item)
        stack.enter_context(patch.object(runner, '_composed_rung'))
        if scenario == 'stored':
            stack.enter_context(signal_at(composed_tcl.TclTransport.__init__.__code__, lambda frame, event: event == 'return',
                                          root / 'signalled'))
        return runner.execute(root, inputs=['-'] if scenario == 'stdin' else [root / 'packaged.npz' if scenario == 'packaged'
                                                                              else root / 'inputs.npz'])


class ResidentStopTests(unittest.TestCase):
    """SIGTERM and SIGINT during a resident session, in a child process with a stub OpenOCD; never the real binary or a board."""
    setUp = RunTests.setUp
    stub, openocd_stopped = StopTests.stub, StopTests.openocd_stopped
    session = 'resident_session'

    def spawn(self, scenario):
        for path in [self.root / 'waiting', self.root / 'signalled', self.root / 'openocd-stub.pid', *self.root.glob('run-*')]:
            subprocess.run(['rm', '-rf', str(path)], check=True)
        code = ('import sys, signal; sys.path.insert(0, sys.argv[1]); signal.signal(signal.SIGINT, signal.default_int_handler); '
                'import test_deployment_run as t; from pathlib import Path; t.%s(Path(sys.argv[2]), sys.argv[3])' % self.session)
        return subprocess.Popen([sys.executable, '-B', '-c', code, str(Path(__file__).parent), str(self.root), scenario],
                                env=dict(os.environ, NOV_OPENOCD=str(self.stub())), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def child(self, scenario, signum):
        process = self.spawn(scenario)
        lines = []
        if scenario == 'stdin':
            process.stdin.write(f'{self.root / "inputs.npz"}\n'); process.stdin.flush()
            while not (lines and lines[-1].startswith('STEP 000 ')):
                lines.append(process.stdout.readline())
                self.assertTrue(lines[-1], ''.join(lines))
            time.sleep(.5)  # the child is now waiting for the next stdin line
        else:
            deadline = time.monotonic() + 60
            while not ((self.root / 'waiting').exists() and (self.root / 'openocd-stub.pid').exists()) and time.monotonic() < deadline:
                time.sleep(.05)
        process.send_signal(signum)
        output, _ = process.communicate(timeout=120)
        output = ''.join(lines) + output
        self.assertTrue(self.openocd_stopped(), output)
        self.assertEqual(process.returncode, 1, output)
        self.assertIn('RUN FAIL logs=', output)
        logs, = self.root.glob('run-*')
        reason = 'resident session stopped by %s' % signum.name
        self.assertIn(reason, (logs / 'failure.txt').read_text())
        summary = json.loads((logs / 'session/summary.json').read_text())
        self.assertEqual((summary['verdict'], summary['error']), ('TRANSPORT_ERROR', reason))
        self.assertTrue((logs / 'reference-open/uart.log').is_file())
        return logs, (logs / 'session-terminal.txt').read_text()

    def test_stop_during_a_resident_call_stops_openocd_and_fails(self):
        for scenario, signum in (('open', signal.SIGTERM), ('call', signal.SIGTERM), ('call', signal.SIGINT), ('close', signal.SIGTERM)):
            with self.subTest(scenario=scenario, signal=signum.name):
                logs, terminal = self.child(scenario, signum)
                steps = 'STEP 000 UNCHECKED (no independent reference) outputs=%s\n' % (logs / 'step-000/outputs.npz')
                self.assertEqual(terminal, (steps if scenario == 'close' else '') +
                                 'SESSION TRANSPORT_ERROR resident session stopped by %s\nrc=2\n' % signum.name)
                self.assertEqual((logs / 'step-000/uart.log').is_file(), scenario != 'open')

    def test_stop_between_stdin_steps_stops_openocd_and_fails(self):
        logs, terminal = self.child('stdin', signal.SIGTERM)
        self.assertEqual(terminal, 'STEP 000 UNCHECKED (no independent reference) outputs=%s\n' % (logs / 'step-000/outputs.npz') +
                         'SESSION TRANSPORT_ERROR resident session stopped by SIGTERM\nrc=2\n')
        self.assertTrue((logs / 'step-000/outputs.npz').is_file())
        self.assertFalse((logs / 'reference-close').exists())

    def test_signal_after_the_records_are_made_ends_the_process_after_them(self):
        # The session accepts no stop once sealed; the restored default handler then ends the process with the records complete.
        process = self.spawn('sealed')
        output, _ = process.communicate(timeout=120)
        self.assertTrue((self.root / 'signalled').exists(), output)
        self.assertTrue(self.openocd_stopped(), output)
        self.assertEqual(process.returncode, -signal.SIGTERM, output)
        self.assertNotIn('RUN ', output)
        logs, = self.root.glob('run-*')
        self.assertEqual((logs / 'session-terminal.txt').read_text(), 'STEP 000 UNCHECKED (no independent reference) outputs=%s\n'
                         % (logs / 'step-000/outputs.npz') + 'SESSION PASS\nrc=0\n')
        self.assertEqual(json.loads((logs / 'session/summary.json').read_text())['verdict'], 'PASS')


class ComposedResidentTests(unittest.TestCase):
    """A live resident session of linked step ELFs in a child process with a stub OpenOCD; never the real binary or a board."""
    stub, openocd_stopped, spawn = StopTests.stub, StopTests.openocd_stopped, ResidentStopTests.spawn
    session = 'composed_session'

    def setUp(self):
        RunTests.setUp(self)
        step_kit(self.root / 'kit')
        np.savez(self.root / 'inputs.npz', input_000=np.uint8([10, 20, 30, 40]))
        np.savez(self.root / 'packaged.npz', input_000=np.uint8([1, 2, 3, 4]))

    def run_child(self, scenario, returncode):
        process = self.spawn(scenario)
        lines = []
        if scenario == 'stdin':
            process.stdin.write(f'{self.root / "inputs.npz"}\n'); process.stdin.flush()
            while not (lines and lines[-1].startswith('STEP 000 ')):
                lines.append(process.stdout.readline())
                self.assertTrue(lines[-1], ''.join(lines))
            time.sleep(.5)  # the child is now waiting for the next stdin line
            process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=120)
        output = ''.join(lines) + output
        self.assertTrue((self.root / 'openocd-stub.pid').exists(), output)  # the stub OpenOCD ran
        self.assertTrue(self.openocd_stopped(), output)
        self.assertEqual(process.returncode, returncode, output)
        logs, = self.root.glob('run-*')
        self.assertTrue((logs / 'session/openocd.log').is_file(), output)
        return output, logs, (logs / 'session-terminal.txt').read_text()

    def test_session_opens_runs_a_step_and_stops_openocd(self):
        output, logs, terminal = self.run_child('pass', 0)
        step = 'STEP 000 UNCHECKED (no independent reference) outputs=%s\n' % (logs / 'step-000/outputs.npz')
        self.assertIn(step + f'RUN PASS (1 of 1 steps unchecked: no independent reference) logs={logs}', output)
        self.assertEqual(terminal, step + 'SESSION PASS\nrc=0\n')
        with np.load(logs / 'step-000/outputs.npz') as outputs:
            self.assertEqual(outputs['output_000'].tolist(), [12, 22, 32, 42])
        for name in ('reference-open', 'step-000', 'reference-close'):
            self.assertEqual([row['id'] for row in json.loads((logs / name / 'journal.json').read_text())['completed']], ['s0', 's1'])

    def test_stop_before_the_session_stores_its_transport_stops_openocd(self):
        output, logs, terminal = self.run_child('stored', 1)
        self.assertTrue((self.root / 'signalled').exists(), output)
        self.assertIn('RUN FAIL logs=', output)
        self.assertTrue(terminal.endswith('SESSION TRANSPORT_ERROR resident session stopped by SIGTERM\nrc=2\n'), terminal)
        self.assertIn('resident session stopped by SIGTERM', (logs / 'failure.txt').read_text())

    def test_stop_between_stdin_steps_stops_openocd(self):
        output, logs, terminal = self.run_child('stdin', 1)
        self.assertIn('RUN FAIL logs=', output)
        self.assertEqual(terminal, 'STEP 000 UNCHECKED (no independent reference) outputs=%s\n' % (logs / 'step-000/outputs.npz') +
                         'SESSION TRANSPORT_ERROR resident session stopped by SIGTERM\nrc=2\n')
        self.assertFalse((logs / 'reference-close').exists())

    def test_a_checked_step_runs_two_passes_in_its_session(self):
        # The session's reference-open is the positive and the step is checked, so no rung and no closing repeat.
        from test_chiprunner_live_steps import write
        manifest = json.loads((self.root / 'kit/manifest.json').read_text())
        manifest['composed']['regions'][-1]['outputs'][0]['reference'] = 'f.bin'  # the packaged output, so steps are checked
        write(self.root / 'kit', manifest)
        output, logs, terminal = self.run_child('packaged', 0)
        self.assertIn(f'STEP 000 EXACT (packaged reference) outputs={logs / "step-000/outputs.npz"}\nRUN PASS logs={logs}', output)
        self.assertEqual(sorted(path.parent.name for path in logs.glob('*/journal.json')), ['reference-open', 'step-000'])
        self.assertTrue(terminal.endswith('SESSION PASS\nrc=0\n'), terminal)


class ComposedQualificationTests(unittest.TestCase):
    """Which passes a live composed run makes before its steps; no transport opens."""
    def setUp(self):
        RunTests.setUp(self)
        step_kit(self.root / 'kit')
        stack = contextlib.ExitStack(); self.addCleanup(stack.close)
        self.rung = stack.enter_context(patch.object(runner, '_composed_rung'))
        stack.enter_context(patch.object(composed, 'Session', side_effect=AssertionError('transport')))

    def test_without_steps_the_run_is_the_rung(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.root)
        self.assertEqual(self.rung.call_count, 1)

    def test_with_steps_the_session_qualifies_and_a_run_without_any_step_refuses(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, patch('sys.stdin', io.StringIO('')):
            with self.assertRaisesRegex(Refusal, 'no step ran'):
                runner.execute(self.root, inputs=['-'])
        self.rung.assert_not_called()
        self.assertIn('RUN FAIL', output.getvalue())
