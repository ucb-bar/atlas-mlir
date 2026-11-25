"""Deploy's EXACT gate runs the checker atlas-mlir ships and refuses a missing, retired, broken, or changed one."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir import deployment_gate as gate
from atlas_mlir.chiprunner import composed_sim
from atlas_mlir.deploy import Refusal
from atlas_mlir.toolchain import data_root

QUALIFICATION = ('qualification.json', 'memory-qualification.json')


class ExactnessGateTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.qualification = self.root / 'qualification'
        self.qualification.mkdir()
        for name in QUALIFICATION:
            (self.qualification / name).write_text('{}\n')
        environment = patch.dict(os.environ, {'ATLAS_CHECKER_QUALIFICATION': str(self.qualification),
                                              'ATLAS_MODEL_CHECKER_PYTHON': sys.executable})
        environment.start()
        self.addCleanup(environment.stop)
        for name in ('ATLAS_MODEL_CHECKER', 'COMPOSED_RUNTIME_PATH'):
            os.environ.pop(name, None)
        bundle = self.root / 'bundle'
        bundle.mkdir()
        row = dict(index=0, address=0x80000010, bytes=4, type=dict(dtype='bf16', shape=[2]))
        (bundle / 'manifest.json').write_text(json.dumps(dict(deployment=dict(window=dict(address=0x80000000),
                                                                              inputs=[row]))))
        (bundle / 'window.bin').write_bytes(b'\0' * 16 + b'\x7a\x3e\0\0')
        out = self.root / 'out'
        out.mkdir()
        window = bundle / 'window.bin'
        self.context = dict(out=out, bundle=bundle,
                            bound_hashes={str(window): hashlib.sha256(window.read_bytes()).hexdigest()})

    def copy(self):
        """A copy of the shipped checker that the gate runs instead, for the broken and changed cases."""
        copy = self.root / 'exactness'
        shutil.copytree(gate.CHECKER, copy)
        replaced = patch.object(gate, 'CHECKER', copy)
        replaced.start()
        self.addCleanup(replaced.stop)
        return copy

    def test_inputs_are_the_shipped_checker_and_its_declared_interfaces(self):
        checker, atlas, qualification, hashes = gate.checker_inputs()
        shipped = Path(gate.__file__).resolve().parent / 'exactness'
        self.assertEqual(checker, shipped / 'check_model.py')
        self.assertEqual((atlas, qualification), (data_root(), self.qualification))
        files = ([shipped / name for name in gate.CHECKER_FILES] + [atlas / name for name in gate.CHECKER_INTERFACES]
                 + [self.qualification / name for name in QUALIFICATION])
        self.assertEqual(hashes, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})

    def test_step_runtime_is_the_shipped_runtime(self):
        runtime, hashes = composed_sim._application_runtime()
        self.assertEqual(runtime, gate.CHECKER)
        self.assertEqual(set(hashes), {'functional_runtime.py', 'model_runtime.py', 'sources.py'})
        for value in (str(self.root), ''):
            os.environ['COMPOSED_RUNTIME_PATH'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'COMPOSED_RUNTIME_PATH is retired'):
                composed_sim._application_runtime()

    def test_retired_external_checker_refuses_even_when_empty(self):
        for value in (str(self.root / 'check-model.py'), ''):
            os.environ['ATLAS_MODEL_CHECKER'] = value
            with self.subTest(value=value), self.assertRaisesRegex(Refusal, 'ATLAS_MODEL_CHECKER is retired'):
                gate.exact(self.context)

    def test_missing_qualification_or_checker_file_refuses(self):
        del os.environ['ATLAS_CHECKER_QUALIFICATION']
        with self.assertRaisesRegex(Refusal, 'set ATLAS_CHECKER_QUALIFICATION'):
            gate.checker_inputs()
        os.environ['ATLAS_CHECKER_QUALIFICATION'] = str(self.qualification)
        (self.qualification / 'memory-qualification.json').unlink()
        with self.assertRaisesRegex(Refusal, 'input missing: .*memory-qualification.json'):
            gate.checker_inputs()
        (self.qualification / 'memory-qualification.json').write_text('{}\n')
        (self.copy() / 'qualified_native_dot.c').unlink()
        with self.assertRaisesRegex(Refusal, 'input missing: .*qualified_native_dot.c'):
            gate.checker_inputs()

    def test_exact_runs_the_shipped_checker_with_its_inputs(self):
        commands = []

        def run(command, **unused):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1)
        with patch.object(gate.subprocess, 'run', run), self.assertRaisesRegex(Refusal, 'bound model execution refused'):
            gate.exact(self.context)
        command = commands[0]
        self.assertEqual(command[:2], [sys.executable, str(Path(gate.__file__).resolve().parent / 'exactness/check_model.py')])
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual((options['--atlas-root'], options['--qualification']), (str(data_root()), str(self.qualification)))

    def test_broken_checker_refuses(self):
        (self.copy() / 'check_model.py').write_text('raise SystemExit("broken checker")\n')
        with self.assertRaisesRegex(Refusal, 'bound model execution refused'):
            gate.exact(self.context)
        self.assertIn('broken checker', (self.context['out'] / 'exact.log').read_text())

    def test_checker_changed_during_the_run_refuses(self):
        copy = self.copy()

        def run(command, **unused):
            (copy / 'functional_runtime.py').write_text('# changed\n')
            return subprocess.CompletedProcess(command, 0)
        with patch.object(gate.subprocess, 'run', run), self.assertRaisesRegex(Refusal, 'hash drift: .*functional_runtime.py'):
            gate.exact(self.context)


if __name__ == '__main__':
    unittest.main()
