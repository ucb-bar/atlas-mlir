"""Synthetic control-flow evidence only, never numerical qualification."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from atlas_mlir import deploy as b
from atlas_mlir import cli


class Stub(b.Stages):
    def __init__(self, fail=None, mutate=False):
        self.calls = []
        self.fail = fail
        self.mutate = mutate

    def hit(self, name):
        self.calls.append(name)
        if name == self.fail:
            raise b.Refusal(name + ' unavailable')

    def admit(self, export, policy, out):
        self.hit('admit')
        return ['--capture', export, '-o', out / 'compile', '--precision-policy', policy or 'derived.json',
                '--application-memory-bytes', '1', '--application-deployment', 'onchip', '--board', 'sophialake']

    def build(self, arguments):
        self.hit('build')
        self.arguments = arguments

    def measure(self, context, arguments):
        # A board host binds a second compile; an on-chip host that fits needs none.
        self.hit('measure')
        return arguments if context['transport'] == 'spi' else None

    def reference(self, context): self.hit('reference')
    def objects(self, context): self.hit('objects')
    def bind(self, context): self.hit('bind')

    def exact(self, context):
        self.hit('exact')
        receipt = context['out'] / 'check.json'
        record = dict(schema='atlas.model_check.v1', status='PASS', positive=dict(result='EXACT'),
                      negative=dict(result='MISMATCH', first_mismatch=0),
                      byte_flip=dict(changed_byte_count=1), same_application_library=True,
                      same_instruction_images=True, independent_expected_unchanged=True)
        if self.mutate:
            record['negative']['result'] = 'EXACT'
        receipt.write_text(json.dumps(record))
        return b.Gate(receipt, {str(receipt): b.digest(receipt)})

    def link(self, context):
        self.hit('link')
        if self.fail == 'drift':
            context['gate'].receipt.write_text('{}')

    def package(self, context): self.hit('package')


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = self.root / 'export'
        self.export.mkdir()
        self.out = self.root / 'out'

    def test_full_stub_sequence_and_transport_refusal(self):
        for jtag in (False, True):
            out = self.root / str(jtag)
            stub = Stub()
            b.deploy(self.export, out=out, jtag=jtag, stages=stub)
            self.assertEqual(stub.calls, ['admit', 'build', 'measure', *([] if jtag else ['build']), 'reference', 'objects',
                                          'bind', 'exact', 'link', 'package'])
            self.assertIn('--capture', stub.arguments)
            # SPI refuses; a live JTAG run refuses on the stub's missing staged tools. Neither starts a process.
            refusal = (FileNotFoundError, 'kit/bench') if jtag else (b.Refusal, 'no SPI loader')
            with patch('subprocess.run') as transport, patch('subprocess.Popen') as process:
                with self.assertRaisesRegex(*refusal):
                    b.run(out)
                transport.assert_not_called()
                process.assert_not_called()
            status = json.loads((out / 'deploy-status.json').read_text())
            self.assertEqual(status['settings']['uart_divisor'], 867)
            self.assertEqual(status['settings']['mtime_hz'], 100000)

    def test_an_onchip_host_that_does_not_fit_compiles_again_with_its_measured_size(self):
        class Over(Stub):
            def measure(self, context, arguments):
                self.hit('measure')
                return [*map(str, arguments), '--measured-host-bytes', '113552']
        stub = Over()
        b.deploy(self.export, out=self.out, jtag=True, stages=stub)
        self.assertEqual(stub.calls, ['admit', 'build', 'measure', 'build', 'reference', 'objects', 'bind', 'exact',
                                      'link', 'package'])
        self.assertEqual(stub.arguments[-2:], ['--measured-host-bytes', '113552'])

    def test_blocked_steps_package_with_their_own_status(self):
        class Blocked(Stub):
            def objects(self, context):
                self.hit('objects')
                context['blocked'] = {'region_083': 'host window exceeded'}
        b.deploy(self.export, out=self.out, jtag=True, stages=Blocked())
        status = json.loads((self.out / 'deploy-status.json').read_text())
        self.assertEqual((status['status'], status['blocked']), ('PACKAGED_BLOCKED', {'region_083': 'host window exceeded'}))

    def test_every_missing_handoff_stops(self):
        stages = ['admit', 'build', 'measure', 'build', 'reference', 'objects', 'bind', 'exact', 'link', 'package']
        for stage in dict.fromkeys(stages):
            stub = Stub(fail=stage)
            with self.assertRaises(b.Refusal):
                b.deploy(self.export, out=self.root / stage, stages=stub)
            self.assertEqual(stub.calls, stages[:stages.index(stage) + 1])

    def test_a_clock_other_than_the_declared_one_reaches_the_compile(self):
        for mhz, expected in (('100', None), ('500', '500000000')):
            stub = Stub()
            b.deploy(self.export, out=self.root / mhz, jtag=True, clock_mhz=mhz, stages=stub)
            supplied = stub.arguments[stub.arguments.index('--host-clock-hz') + 1] if '--host-clock-hz' in stub.arguments else None
            self.assertEqual(supplied, expected)

    def test_board_deploy_refuses_a_bench_staged_plan(self):
        application = self.out / 'compiled' / 'application'
        application.mkdir(parents=True)
        (application / 'deployment.json').write_text(json.dumps(dict(mode='bench_staged')))
        with patch('atlas_mlir.deployment_link.measure_host') as measure:
            with self.assertRaisesRegex(b.Refusal, 'bench-staged plan has no board DDR deployment'):
                b.Stages().measure(dict(out=self.out), [])
        measure.assert_not_called()

    def test_mutation_must_change_output_before_link(self):
        stub = Stub(mutate=True)
        with self.assertRaisesRegex(b.Refusal, 'mutation'):
            b.deploy(self.export, out=self.out, stages=stub)
        self.assertNotIn('link', stub.calls)

    def test_drift_blocks_package(self):
        stub = Stub(fail='drift')
        with self.assertRaisesRegex(b.Refusal, 'hash drift'):
            b.deploy(self.export, out=self.out, stages=stub)
        self.assertNotIn('package', stub.calls)

    def test_real_admission_refuses(self):
        with self.assertRaisesRegex(b.Refusal, 'missing inventory member'):
            b.deploy(self.export, out=self.out)
        with self.assertRaisesRegex(b.Refusal, 'incomplete'):
            b.run(self.out)

    def test_cli_real_refusals_return_failure(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(['deploy', str(self.export), '-o', str(self.out)]), 1)
            self.assertIn('missing inventory member', error.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(['run', str(self.out)]), 1)
            self.assertIn('incomplete', error.getvalue())

    def test_fresh_output(self):
        self.out.mkdir()
        with self.assertRaisesRegex(b.Refusal, 'fresh'):
            b.deploy(self.export, out=self.out)

    def test_invalid_clock_no_output(self):
        for clock in ('0', '-1', 'NaN', 'Infinity', 'abc', '5000', '0.0000001', '100.000001'):
            with self.assertRaises(b.Refusal):
                b.deploy(self.export, out=self.out, clock_mhz=clock)
            self.assertFalse(self.out.exists())

    def test_clock_boundaries(self):
        for clock in ('0.115', '4294.968'):
            with self.assertRaisesRegex(b.Refusal, 'integral kHz from 116 to 4294967 kHz'):
                b.clock_settings(clock)
        self.assertEqual(b.clock_settings('0.116')['clock_hz'], 116000)
        self.assertEqual(b.clock_settings('4294.967')['clock_hz'], 4294967000)

    def test_uart_baud_is_the_declared_soc_fact(self):
        from atlas_mlir_native import soc

        declared = soc.SOC
        self.assertEqual(b.clock_settings('100')['receiver_baud'], declared.resource('penguin.uart0').baud)
        uart = replace(declared.resource('penguin.uart0'), baud=57600)
        other = replace(declared, resources=tuple(uart if r.id == uart.id else r for r in declared.resources))
        with patch.object(soc, 'SOC', other):
            settings = b.clock_settings('100')
            with self.assertRaisesRegex(b.Refusal, 'integral kHz from 58 to'):
                b.clock_settings('0.057')
        self.assertEqual((settings['receiver_baud'], settings['uart_divisor']), (57600, 1735))

    def test_fractional_mhz_with_integral_khz(self):
        stub = Stub()
        b.deploy(self.export, out=self.out, clock_mhz='100.001', stages=stub)
        self.assertIn('package', stub.calls)

    def test_console_entrypoint(self):
        metadata = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
        scripts = metadata['project']['scripts']
        self.assertEqual(scripts['atlasc'], 'atlas_mlir.cli:main')
        self.assertNotIn('atlas-button', scripts)
        self.assertFalse(hasattr(b, 'main'))

    def test_cli_flags(self):
        policy = self.root / 'policy.json'
        policy.write_text('{}')
        args = ['deploy', str(self.export), '--jtag', '--clock-mhz', '80', '-o', str(self.out), '--precision-policy', str(policy)]
        with patch.object(b, 'deploy') as deploy, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        self.assertEqual(deploy.call_args.kwargs, dict(export=str(self.export), jtag=True, clock_mhz='80', out=str(self.out), precision_policy=str(policy)))

    def test_public_cli_help(self):
        for args, expected in ((['--help'], ('deploy', 'run')),
                               (['deploy', '--help'], ('--jtag', '--clock-mhz', '--precision-policy')),
                               (['run', '--help'], ('run',))):
            with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    cli.main(args)
                self.assertEqual(result.exception.code, 0)
                for option in expected:
                    self.assertIn(option, output.getvalue())

    def test_cli_run_flags(self):
        with patch.object(b, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['run', str(self.out)]), 0)
            self.assertEqual(run.call_args.kwargs, dict(out=str(self.out)))
            self.assertEqual(cli.main(['run', str(self.out), '--dry-run']), 0)
            self.assertEqual(run.call_args.kwargs, dict(out=str(self.out), dry_run=True))
            self.assertEqual(cli.main(['run', str(self.out), '--inputs', 'a.npz', 'b.npz', '--reference', 'r']), 0)
            self.assertEqual(run.call_args.kwargs, dict(out=str(self.out), inputs=['a.npz', 'b.npz'], references=['r']))
            self.assertEqual(cli.main(['run', str(self.out), '--serial', '/dev/cu.usbserial-1']), 0)
            self.assertEqual(run.call_args.kwargs, dict(out=str(self.out), serial='/dev/cu.usbserial-1'))
        # The live run is `atlasc run OUT`; the removed authorization flag has no alias.
        with patch.object(b, 'run') as run, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main(['run', str(self.out), '--allow-transport'])
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()

    def test_public_cli_stub_deploy_and_run_refusal(self):
        stub = Stub()
        with patch.object(b, 'Stages', return_value=stub), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['deploy', str(self.export), '-o', str(self.out)]), 0)
        self.assertEqual(stub.calls, ['admit', 'build', 'measure', 'build', 'reference', 'objects', 'bind', 'exact',
                                      'link', 'package'])
        with patch('subprocess.run') as transport, contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(['run', str(self.out)]), 1)
            self.assertIn('no SPI loader', error.getvalue())
            transport.assert_not_called()

    def test_cli_reference_needs_a_packaged_deployment_and_a_fresh_outside_directory(self):
        self.out.mkdir()
        for status, destination, message in (('INCOMPLETE', self.root / 'ref', 'deployment incomplete'),
                                             ('PACKAGED', self.out / 'ref', 'fresh and outside OUT'),
                                             ('PACKAGED', self.export, 'fresh and outside OUT')):
            (self.out / 'deploy-status.json').write_text(json.dumps(dict(status=status)))
            with self.subTest(message=message), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main(['reference', str(self.out), '--inputs', 'x.npz', '-o', str(destination)]), 1)
                self.assertIn(message, error.getvalue())

    def test_symlinks_refuse_without_following_targets(self):
        target = self.root / 'missing'
        self.out.symlink_to(target)
        with self.assertRaisesRegex(b.Refusal, 'symlink'):
            b.deploy(self.export, out=self.out)
        self.assertFalse(target.exists())
        parent = self.root / 'parent'
        parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(b.Refusal, 'symlink'):
            b.safe_path(parent / 'child')

    def test_symlink_before_parent_component_refuses(self):
        link = self.root / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        lexical_out = link / '..' / 'out'
        with self.assertRaisesRegex(b.Refusal, 'parent path components'):
            b.safe_path(lexical_out)
        with self.assertRaisesRegex(b.Refusal, 'parent path components'):
            b.deploy(self.export, out=lexical_out, stages=Stub())
        self.assertFalse(self.out.exists())

    def test_status_symlink_refuses(self):
        self.out.mkdir()
        (self.out / 'deploy-status.json').symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(b.Refusal, 'symlink'):
            b.run(self.out)

    def test_build_uses_atlasc(self):
        with patch('subprocess.run') as invoke:
            b.Stages().build(['--capture', self.export])
        self.assertEqual(invoke.call_args.args[0][1:4], ['-m', 'atlas_mlir.cli', 'build'])
        self.assertTrue(invoke.call_args.kwargs['check'])

    def test_gate_requires_authenticated_receipt(self):
        with self.assertRaisesRegex(b.Refusal, 'hashes'):
            b.Gate(self.out, {}).verify()


if __name__ == '__main__':
    unittest.main()
