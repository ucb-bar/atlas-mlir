"""ModeLIR driver compatibility checks; fake protocol/core, no ARC execution."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from selected_core_runtime import HALT_SIGNAL, _allowed_path, run_selected_program


ROOT = Path(__file__).resolve().parents[1]


class SelectedCoreRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name) / "state.json"
        self.model = Path(self.temporary.name) / "not-a-native-model.so"
        self.manifest = [{"name": "AtlasCore", "numStateBytes": 8,
                          "states": [{"name": HALT_SIGNAL, "numBits": 1,
                                      "type": "wire", "offset": 0}]}]
        self.write_manifest()

    def write_manifest(self):
        self.state.write_text(json.dumps(self.manifest))

    def test_explicit_halt_passed_and_arguments_preserved(self):
        seen = {}
        core_type = type("ExistingSelectedCore", (), {})

        def runner(model, state, words, preload=None, *, halt_signal="io_halted",
                   max_cycles=20000, on_cycle=None):
            seen.update(model=model, state=state, words=words, preload=preload,
                        halt_signal=halt_signal, max_cycles=max_cycles, on_cycle=on_cycle)
            return "result"

        module = SimpleNamespace(run_program=runner, CosimCore=core_type)
        callback = lambda core: None
        words, preload = (0x13, 0x73), [(0x90000000, b"input")]
        result = run_selected_program(module, self.model, self.state, words,
                                      preload=preload, max_cycles=17, on_cycle=callback)
        self.assertEqual(result, "result")
        self.assertEqual(seen, dict(model=self.model, state=self.state, words=words,
                                   preload=preload, halt_signal=HALT_SIGNAL,
                                   max_cycles=17, on_cycle=callback))
        self.assertIs(module.CosimCore, core_type)

    def test_historical_alias_preserves_existing_poke_guard_and_restores(self):
        seen = []

        class ExistingSelectedCore:
            def peek(self, name):
                seen.append(name)
                if name == HALT_SIGNAL:
                    return 1
                raise KeyError(name)

            def poke(self, name, value):
                if name == "io_dmaTL_d_bits_opcode":
                    seen.append("guarded optimized-away opcode")
                    return
                raise KeyError(name)

        module = SimpleNamespace(CosimCore=ExistingSelectedCore)

        def historical_runner(model, state, words, preload=None, *, max_cycles=20000,
                              on_cycle=None):
            core = module.CosimCore()
            core.poke("io_dmaTL_d_bits_opcode", 1)
            if on_cycle:
                on_cycle(core)
            return core.peek("io_halted")

        module.run_program = historical_runner
        self.assertEqual(run_selected_program(module, self.model, self.state, (0x73,)), 1)
        self.assertEqual(seen, ["guarded optimized-away opcode", HALT_SIGNAL])
        self.assertIs(module.CosimCore, ExistingSelectedCore)

        def fail(core):
            raise RuntimeError("runner failure")

        with self.assertRaisesRegex(RuntimeError, "runner failure"):
            run_selected_program(module, self.model, self.state, (), on_cycle=fail)
        self.assertIs(module.CosimCore, ExistingSelectedCore)

    def test_invalid_manifests_fail_before_driver_invocation(self):
        called = []

        def runner(*args, halt_signal="io_halted", **kwargs):
            called.append(True)

        module = SimpleNamespace(run_program=runner)
        cases = (
            [], self.manifest * 2,
            [{"name": "OtherCore", "states": self.manifest[0]["states"]}],
            [{"name": "AtlasCore", "states": []}],
            [{"name": "AtlasCore", "states": self.manifest[0]["states"] * 2}],
            [{"name": "AtlasCore", "states": [{"name": HALT_SIGNAL, "numBits": 8}]}],
            [{"name": "AtlasCore", "states": [{"name": "io_halted", "numBits": 1}]}],
            [{"name": "AtlasCore", "states": [{"name": HALT_SIGNAL, "numBits": True}]}],
        )
        for manifest in cases:
            with self.subTest(manifest=manifest):
                self.state.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    run_selected_program(module, self.model, self.state, ())
        self.assertEqual(called, [])

    def test_no_wrong_halt_override_or_typeerror_retry(self):
        count = []

        def runner(*args, halt_signal="io_halted", **kwargs):
            count.append(halt_signal)
            raise TypeError("native driver failed")

        module = SimpleNamespace(run_program=runner)
        with self.assertRaises(ValueError):
            run_selected_program(module, self.model, self.state, (), halt_signal="io_halted")
        self.assertEqual(count, [])
        with self.assertRaisesRegex(TypeError, "native driver failed"):
            run_selected_program(module, self.model, self.state, ())
        self.assertEqual(count, [HALT_SIGNAL])

    def test_restricted_paths_rejected_before_resolution(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve")):
            for path in ("/tmp/HaMmEr/nonexistent", "/tmp/VLSI/nonexistent"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    _allowed_path(path)

    def test_restricted_symlink_target_rejected_before_target_stat(self):
        alias = Path(self.temporary.name) / "allowed-alias"
        original_lstat = Path.lstat
        visited = []

        def guarded_lstat(path, *args, **kwargs):
            self.assertFalse(any("hammer" in part.lower() or "vlsi" in part.lower()
                                 for part in path.parts), "prohibited target was inspected")
            visited.append(path)
            return original_lstat(path, *args, **kwargs)

        for target in ("/tmp/HaMmEr/nonexistent", "../VLSI/nonexistent"):
            with self.subTest(target=target):
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to(target)
                with patch.object(Path, "lstat", guarded_lstat), self.assertRaises(ValueError):
                    _allowed_path(alias / "state.json")
        self.assertIn(alias, visited)

    def test_allowed_symlink_and_nonexistent_tail(self):
        alias = Path(self.temporary.name) / "allowed-alias"
        alias.symlink_to(self.state.name)
        self.assertEqual(_allowed_path(alias), self.state)
        self.assertEqual(_allowed_path(self.model), self.model)

    def test_current_real_modelir_runner_with_fake_core_and_protocol(self):
        # Load only the runner source into an isolated package. Stub dependencies
        # prevent imports/builds/ctypes and make this a driver-contract check.
        modelir = _allowed_path(os.environ.get("ATLAS_MODELIR_ROOT",
                                               ROOT.parents[2] / "ModeLIR"))
        source = _allowed_path(modelir / "mlc/backends/cosim_atlas.py")
        if not source.is_file():
            self.skipTest("ModeLIR source absent; portable adapter contracts still run")
        source_text = source.read_text()
        tree = ast.parse(source_text)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "run_program")
        if not any(a.arg == "halt_signal" for a in run.args.kwonlyargs):
            self.skipTest("historical ModeLIR checkout; current real-runner contract unavailable")
        created, loads, observed = [], [], []

        class FakeCore:
            def __init__(core, model_path, manifest_path):
                core.manifest = json.loads(Path(manifest_path).read_text())[0]
                core._S = {s["name"]: s for s in core.manifest["states"]}
                core.cycle = 0
                created.append(core)

            def reset(core):
                core.cycle = 0

            def peek(core, name):
                # Represent the old local adapter: this alias alone still
                # cannot satisfy the real runner's manifest validation.
                if name == "io_halted":
                    name = HALT_SIGNAL
                if name not in core._S:
                    raise KeyError(name)
                return int(core.cycle >= 2)

            def tick(core):
                core.cycle += 1

            def has(core, name):
                return name in core._S

        class FakeAdapter:
            def __init__(adapter, core, prefix):
                adapter.prefix = prefix

            def put(adapter, addr, value, size):
                loads.append((adapter.prefix, addr, value, size))

        class FakeSlave:
            def __init__(slave, core, prefix, **kwargs):
                slave.reads = slave.writes = 0
                slave.preloaded = []

            def preload(slave, addr, data):
                slave.preloaded.append((addr, data))

            def step(slave):
                pass

        package = ModuleType("_selected_modelir_contract")
        package.__path__ = []
        core_module = ModuleType(package.__name__ + ".cosim_core")
        core_module.CosimCore = FakeCore
        core_module.large_stack_call = lambda fn: fn()
        protocols = ModuleType(package.__name__ + ".protocols")
        protocols.TileLinkAdapter, protocols.TileLinkSlave = FakeAdapter, FakeSlave
        module_name = package.__name__ + ".cosim_atlas"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {package.__name__: package,
                                      core_module.__name__: core_module,
                                      protocols.__name__: protocols, module_name: module}):
            spec.loader.exec_module(module)
            # The old peek alias cannot pass the current manifest validation.
            with self.assertRaisesRegex(ValueError, "halt signal"):
                module.run_program(self.model, self.state, (0x13, 0x73), max_cycles=5)
            self.assertEqual(loads, [])
            result = run_selected_program(module, self.model, self.state, (0x13, 0x73),
                                          preload=[(0x90000000, b"input")], max_cycles=5,
                                          on_cycle=lambda core: observed.append(core.cycle))
        self.assertTrue(result.halted)
        self.assertEqual(result.cycles, 2)
        self.assertEqual(observed, [0, 1, 2])
        self.assertEqual(loads, [("imemTL", 0x20000, 0x13, 2),
                                 ("imemTL", 0x20004, 0x73, 2), ("csrTL", 0x18, 1, 2)])
        self.assertEqual(result.slave.preloaded, [(0x90000000, b"input")])


if __name__ == "__main__":
    unittest.main()
