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

    def admit(self, export, out):
        self.hit('admit')
        return ['--capture', export, '--export', export, '-o', out / 'compile',
                '--application-memory-bytes', '1', '--application-deployment', 'onchip']

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
            # The on-chip bench board supplies 100 MHz; both DDR boards supply 500 MHz.
            self.assertEqual((status['settings']['uart_divisor'], status['settings']['mtime_hz']),
                             (867, 100000) if jtag else (4339, 500000))

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

    def test_plain_deploy_takes_the_clock_of_its_declared_carriers(self):
        for jtag, hz, divisor, carriers in ((True, 100000000, 867, ['albatross']),
                                            (False, 500000000, 4339, ['firesim-tethered', 'puffin'])):
            stub = Stub()
            out = self.root / str(jtag)
            b.deploy(self.export, out=out, jtag=jtag, stages=stub)
            # The compile's provider takes the same declared clock; the gate refuses a deployment that disagrees.
            self.assertNotIn('--host-clock-hz', stub.arguments)
            settings = json.loads((out / 'deploy-status.json').read_text())['settings']
            self.assertEqual((settings['clock_hz'], settings['uart_divisor'], settings['carriers']), (hz, divisor, carriers))
            self.assertEqual(settings['clock_basis'],
                             f'declared by penguin_sophialake carriers {", ".join(carriers)}, unmeasured')

    def test_an_unknown_missing_or_conflicting_carrier_refuses_before_output(self):
        from atlas_mlir_native import soc

        row = lambda name, placement, hz: dict(id=name, placement=placement, clock_hz=hz, evidence='e')
        for rows, message in (([row('a', 'onchip', 100000000), row('p', 'ddr', None)],
                               'carrier p declares no chip clock'),
                              ([row('a', 'onchip', 100000000)], 'no carrier in the SoC declaration runs ddr'),
                              ([row('t', 'ddr', 500000000), row('p', 'ddr', 400000000)],
                               'ddr carriers supply different chip clocks: t 500000000 Hz, p 400000000 Hz'),
                              ([row('p', 'ddr', 500000001)], 'declared clock 500000001 Hz must be integral kHz'),
                              ([row('p', 'ddr', 115000)], 'integral kHz from 116 to 4294967 kHz')):
            with self.subTest(message=message), patch.object(soc, 'CARRIERS', tuple(rows)):
                with self.assertRaisesRegex(b.Refusal, message):
                    b.deploy(self.export, out=self.out, stages=Stub())
                self.assertFalse(self.out.exists())

    def test_declared_carriers_are_validated_and_carry_no_link_clock(self):
        from atlas_mlir_native import soc

        self.assertEqual([(row['id'], row['placement'], row['clock_hz']) for row in soc.CARRIERS],
                         [('albatross', 'onchip', 100000000), ('firesim-tethered', 'ddr', 500000000),
                          ('puffin', 'ddr', 500000000)])
        # The chip declares no host clock of its own, so no chip fact restates one board's clock.
        self.assertIsNone(next(clock.frequency_hz for clock in soc.SOC.clocks if clock.id == 'host'))
        good = dict(id='p', placement='ddr', clock_hz=None, evidence='e')
        self.assertEqual(soc.carriers({'carriers': [good]}), (good,))
        for rows, message in (([{**good, 'link_clock_hz': 50000000}], 'link clock belongs to the bitstream'),
                              ([{**good, 'placement': 'spi'}], 'placement'), ([{**good, 'clock_hz': 0}], 'clock_hz'),
                              ([{**good, 'clock_hz': True}], 'clock_hz'), ([good, good], 'unique'), ([], 'carriers')):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                soc.carriers({'carriers': rows})

    def test_the_developer_object_compile_defaults_to_the_onchip_carriers_clock(self):
        from atlas_mlir_native.elf_build import __main__ as elf_build

        source = self.root / 'a.c'
        source.write_text('int a;\n')
        with patch('sys.argv', ['elf_build', 'compile', str(source), '--cc', str(source), '-o', str(self.out / 'objects')]), \
                patch.object(elf_build.subprocess, 'run') as run, patch.object(elf_build, 'measure', return_value={}):
            elf_build.main()
        self.assertIn('-DSYS_CLK_FREQ=100000000', run.call_args.args[0])
        self.assertEqual(json.loads((self.out / 'objects' / 'sections.json').read_text())['clock_mhz'], 100)
        with patch.object(elf_build, 'carrier_clock', side_effect=ValueError('carrier albatross declares no chip clock')), \
                patch('sys.argv', ['elf_build', 'compile', str(source), '--cc', str(source), '-o', str(self.out / 'o2')]), \
                self.assertRaisesRegex(ValueError, 'declares no chip clock'):
            elf_build.main()

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

    def test_uart_baud_is_the_declared_soc_fact(self):
        from atlas_mlir_native import soc

        declared = soc.SOC
        self.assertEqual(b.board_settings('onchip')['receiver_baud'], declared.resource('penguin.uart0').baud)
        uart = replace(declared.resource('penguin.uart0'), baud=57600)
        other = replace(declared, resources=tuple(uart if r.id == uart.id else r for r in declared.resources))
        with patch.object(soc, 'SOC', other):
            settings = b.board_settings('onchip')
        self.assertEqual((settings['receiver_baud'], settings['uart_divisor']), (57600, 1735))

    def test_console_entrypoint(self):
        metadata = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
        scripts = metadata['project']['scripts']
        self.assertEqual(scripts['atlascc'], 'atlas_mlir.cli:main')
        self.assertNotIn('atlas-button', scripts)
        self.assertFalse(hasattr(b, 'main'))

    def test_cli_flags(self):
        args = ['deploy', str(self.export), '--jtag', '-o', str(self.out)]
        with patch.object(b, 'deploy') as deploy, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        self.assertEqual(deploy.call_args.kwargs, dict(export=str(self.export), jtag=True, out=str(self.out)))
        for option in ('--precision-policy', '--clock-mhz'):
            with patch.object(b, 'deploy') as deploy, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                cli.main([*args, option, '500'])
            self.assertEqual(result.exception.code, 2)
            deploy.assert_not_called()

    def test_public_cli_help(self):
        for args, expected in ((['--help'], ('deploy', 'run')),
                               (['deploy', '--help'], ('--jtag',)),
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
        # The live run is `atlascc run OUT`; the removed authorization flag has no alias.
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

    def test_build_uses_atlascc(self):
        with patch('subprocess.run') as invoke:
            b.Stages().build(['--capture', self.export])
        self.assertEqual(invoke.call_args.args[0][1:4], ['-m', 'atlas_mlir.cli', 'build'])
        self.assertTrue(invoke.call_args.kwargs['check'])

    def test_gate_requires_authenticated_receipt(self):
        with self.assertRaisesRegex(b.Refusal, 'hashes'):
            b.Gate(self.out, {}).verify()


if __name__ == '__main__':
    unittest.main()
