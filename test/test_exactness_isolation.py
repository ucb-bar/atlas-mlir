"""The checker and the staged step worker load only declared code, from the files the gate hashes.

Each case runs in its own interpreter, which must import npu_model, with an audit hook that records every module
import, dynamic ones included. It binds the sources as the checker (check_model.bind_sources) or a staged worker
(composed_sim._bind_sources) does, then executes: a functional runtime over the pinned model, the native dot build,
and the sparse memory. A compiler module loaded at any depth, or a declared module loaded from another file, is a
violation. The negative cases show the check fails for a compiler import inside a declared interface and for a
runtime in the Atlas root under the workers' old import order, and that binding refuses a runtime already loaded
from another file.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from atlas_mlir.deployment_gate import CHECKER
from atlas_mlir.toolchain import data_root

COMPILER = ('atlas_mlir', 'atlas_mlir_native', 'atlas_m2m_adapter', 'm2m', 'xdsl')
MERLIN = ('merlin', 'merlin.runtime', 'merlin.runtime.uniform_sparse_memory')
# The staged worker's runner (chiprunner/kit_manifest.py) reads its memory windows from the SoC declaration when it
# runs outside a kit; that declaration is its one other interface. The checker loads none of it.
SOC = ('atlas_mlir_native', 'atlas_mlir_native.soc', 'merlin.target', 'merlin.target.soc')
SHIPPED = ('check_model', 'model_runtime', 'functional_runtime', 'qualified_native_dot')
INTERFACES = ('tools.check_variant_inventory', 'tools.check_program_decode', 'tools.model_instructions')

DRIVER = textwrap.dedent('''
    import atexit, importlib.util, json, sys
    from pathlib import Path
    mode, exactness, atlas, model, work, out = sys.argv[1:7]
    imported = []
    sys.addaudithook(lambda event, args: imported.append(args[0]) if event == "import" else None)
    atexit.register(lambda: Path(out).write_text(json.dumps(dict(imported=imported, files={
        name: getattr(module, "__file__", None) for name, module in list(sys.modules.items())}))))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if mode in ("checker", "preloaded"):
        if mode == "preloaded":
            import functional_runtime  # a colliding file from PYTHONPATH, loaded before the checker binds
        sys.path.insert(0, exactness)  # as when the gate runs check_model.py by path
        load("check_model", Path(exactness) / "check_model.py").bind_sources(
            Path(atlas), ("model_runtime", "qualified_native_dot", "functional_runtime"))
    elif mode == "worker":
        sys.path.insert(0, str(Path(exactness).parent / "chiprunner"))  # as when composed_sim.py runs by path
        load("composed_sim", Path(exactness).parent / "chiprunner/composed_sim.py")._bind_sources(
            dict(runtime=exactness, atlas_root=atlas, model_root=model))
    else:  # "old-worker": the step worker's import path before binding, then imports by name
        for root in (exactness, atlas, model):  # its runtime, atlas_root, and model_root, each inserted at index zero
            sys.path.insert(0, root)
    from functional_runtime import FunctionalModelRuntime
    FunctionalModelRuntime(Path(atlas), Path(model), [], Path(work) / "execution", memory_base=0x90000000,
                           memory_bytes=4096).finish()
    if mode == "checker":
        import qualified_native_dot
        from npu_model.configs import numerics, rtl_math
        from npu_model.functional import state
        from merlin.runtime.uniform_sparse_memory import UniformSparseByteMemory, verify_qualification
        qualified_native_dot.candidate_for_qualification(rtl_math.product, Path(work) / "native")
        import torch
        memory = UniformSparseByteMemory(8192)
        memory[4096:4100] = torch.ones(4, dtype=torch.uint8)
        assert memory[4096:4100].tolist() == [1, 1, 1, 1]
''')


def violations(record, atlas, model, allowed=()):
    """Compiler modules loaded at any depth, and declared modules loaded from other files than the bound ones."""
    found = [f'compiler module {name}' for name in sorted(set(record['imported']) | set(record['files']))
             if name not in allowed
             and (name.split('.')[0] in COMPILER or (name.split('.')[0] == 'merlin' and name not in MERLIN))]
    expected = {**{name: CHECKER / f'{name}.py' for name in SHIPPED},
                **{name: atlas / 'tools' / f'{name.split(".")[1]}.py' for name in INTERFACES}}
    for name, path in record['files'].items():
        if name in expected and Path(path).resolve() != expected[name].resolve():
            found.append(f'{name} from {path}')
        if name.split('.')[0] == 'npu_model' and path and not Path(path).resolve().is_relative_to(model.resolve()):
            found.append(f'{name} from {path}')
    return found


class ExactnessIsolationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.atlas, self.model = data_root(), Path(os.environ['ATLAS_NPU_MODEL_ROOT'])
        # Same-named runtime and a regular tools package with the decoder, ahead of everything on the import path.
        self.collider = self.root / 'collider'
        (self.collider / 'tools').mkdir(parents=True)
        for name in ('functional_runtime.py', 'model_runtime.py'):
            shutil.copy(CHECKER / name, self.collider)
        (self.collider / 'tools/__init__.py').write_text('')
        for name in INTERFACES:
            shutil.copy(self.atlas / 'tools' / f'{name.split(".")[1]}.py', self.collider / 'tools')

    def copy_atlas(self):
        """A support root with copies of the decoder's Python files and the variant inventory."""
        atlas = self.root / f'atlas-{len(list(self.root.iterdir()))}'
        shutil.copytree(self.atlas / 'tools', atlas / 'tools',
                        ignore=lambda folder, names: [name for name in names if not name.endswith('.py')])
        (atlas / 'docs').mkdir()
        shutil.copy(self.atlas / 'docs/selected-variant-inventory.json', atlas / 'docs')
        return atlas

    def run_case(self, mode, atlas=None, collide=False):
        work = self.root / f'{mode}-{len(list(self.root.iterdir()))}'
        work.mkdir()
        environment = dict(os.environ)
        if collide:
            environment['PYTHONPATH'] = os.pathsep.join([str(self.collider), environment.get('PYTHONPATH', '')])
        done = subprocess.run([sys.executable, '-c', DRIVER, mode, str(CHECKER), str(atlas or self.atlas),
                               str(self.model), str(work), str(work / 'record.json')],
                              env=environment, capture_output=True, text=True, timeout=600)
        return done, json.loads((work / 'record.json').read_text())

    def assert_clean(self, mode, collide):
        done, record = self.run_case(mode, collide=collide)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(violations(record, self.atlas, self.model, SOC if mode == 'worker' else ()), [])
        loaded = {'functional_runtime', 'model_runtime', *INTERFACES} | ({'qualified_native_dot'} if mode == 'checker' else set())
        self.assertLessEqual(loaded, set(record['files']))

    def test_checker_loads_only_declared_code_from_bound_files(self):
        self.assert_clean('checker', collide=False)

    def test_staged_worker_loads_only_declared_code_from_bound_files(self):
        self.assert_clean('worker', collide=False)

    def test_colliding_runtime_and_tools_package_do_not_replace_bound_files(self):
        for mode in ('checker', 'worker'):
            with self.subTest(mode=mode):
                self.assert_clean(mode, collide=True)

    def test_runtime_in_the_atlas_root_wins_the_old_worker_order_and_not_the_binding(self):
        atlas = self.copy_atlas()
        for name in ('functional_runtime.py', 'model_runtime.py'):
            shutil.copy(CHECKER / name, atlas)
        done, record = self.run_case('old-worker', atlas=atlas)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn(f'functional_runtime from {atlas / "functional_runtime.py"}', violations(record, atlas, self.model))
        done, record = self.run_case('worker', atlas=atlas)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(violations(record, atlas, self.model, SOC), [])

    def test_runtime_already_loaded_from_another_file_is_refused(self):
        done, _ = self.run_case('preloaded', collide=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertRegex(done.stderr, f'runtime is already loaded from {self.collider}')

    def test_compiler_import_inside_a_declared_interface_is_a_violation(self):
        atlas = self.copy_atlas()
        with (atlas / 'tools/model_instructions.py').open('a') as stream:
            stream.write('\nimport importlib\ntry:\n    importlib.import_module("atlas_mlir_native.options")\n'
                         'except Exception:\n    pass\n')
        done, record = self.run_case('checker', atlas=atlas)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn('compiler module atlas_mlir_native', violations(record, atlas, self.model))


if __name__ == '__main__':
    unittest.main()
