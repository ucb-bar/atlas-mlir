"""Halt-state adapter checks against a fake ModeLIR runner."""
import unittest
from types import SimpleNamespace

from selected_core_runtime import HALT_SIGNAL, run_selected_program


class SelectedCoreRuntimeTest(unittest.TestCase):
    def test_modern_runner_gets_explicit_halt_and_arguments(self):
        seen = {}

        def runner(model, state, words, preload=None, *, halt_signal="io_halted", max_cycles=20000):
            seen.update(model=model, state=state, words=words, preload=preload, halt_signal=halt_signal, max_cycles=max_cycles)
            return "result"

        module = SimpleNamespace(run_program=runner, CosimCore=type("Core", (), {}))
        self.assertEqual(run_selected_program(module, "m", "s", (0x13,), preload=[1], max_cycles=17), "result")
        self.assertEqual(seen, dict(model="m", state="s", words=(0x13,), preload=[1], halt_signal=HALT_SIGNAL, max_cycles=17))

    def test_historical_runner_aliases_io_halted_and_restores_core(self):
        peeked = []

        class Core:
            def peek(self, name):
                peeked.append(name)
                return 1

        module = SimpleNamespace(CosimCore=Core)

        def runner(model, state, words, max_cycles=20000):
            peeked.append(module.CosimCore().peek("io_halted"))
            module.CosimCore().peek("other")
            return "legacy"

        module.run_program = runner
        self.assertEqual(run_selected_program(module, "m", "s", ()), "legacy")
        self.assertEqual(peeked, [HALT_SIGNAL, 1, "other"])
        self.assertIs(module.CosimCore, Core)

    def test_runner_failure_restores_core(self):
        class Core:
            def peek(self, name): return 0

        def runner(*args, **kwargs): raise RuntimeError("boom")

        module = SimpleNamespace(CosimCore=Core, run_program=runner)
        with self.assertRaises(RuntimeError): run_selected_program(module, "m", "s", ())
        self.assertIs(module.CosimCore, Core)


if __name__ == "__main__":
    unittest.main()
