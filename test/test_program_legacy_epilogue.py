"""Archived C hook reproduction has narrow admission and unchanged new defaults."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program import assembly, legacy_epilogue_assembly
from atlas_weight_source import WeightSource


class LegacyEpilogueTests(unittest.TestCase):
    def test_legacy_hook_uses_original_registers_and_new_default_stays_free(self):
        m, k, n = 3136, 64, 64
        offset = m * k + k * n + 256
        hook = Epilogue(m, n, offset, (Step('add', TileInput(offset + 2*m*n + 256, 'channel')),))
        current = assembly(m, k, n, offset, -3, traffic_reuse=True, epilogue=hook)
        old = legacy_epilogue_assembly(m, k, n, offset, -3, epilogue=hook)
        self.assertNotEqual(current, old)
        self.assertIn('li x31,', old)
        self.assertNotIn('li x31,', current)
        self.assertEqual(current, assembly(m, k, n, offset, -3, traffic_reuse=True, epilogue=hook))

    def test_legacy_hook_passes_weight_source_through(self):
        m, k, n = 3136, 64, 64
        offset = m * k + k * n + 256
        hook = Epilogue(m, n, offset, (Step('add', TileInput(offset + 2*m*n + 256, 'channel')),))
        source = WeightSource(0x84000000, k * n, 0x90000000)
        old = legacy_epilogue_assembly(m, k, n, offset, -3, epilogue=hook, weight_source=source)
        current = assembly(m, k, n, offset, -3, traffic_reuse=True, epilogue=hook, weight_source=source)
        self.assertEqual(old, re.sub(r'\bx1([01])\b', r'x3\1', current))
        self.assertNotEqual(old, legacy_epilogue_assembly(m, k, n, offset, -3, epilogue=hook))

    def test_reject_tiny_schedules_with_and_without_hook(self):
        for m, k, n in ((128, 2048, 2048), (32, 2048, 1024)):
            offset = m*k+k*n+256
            for hook in (None, Epilogue(m, n, offset)):
                with self.subTest(shape=(m,k,n), hook=hook), self.assertRaisesRegex(ValueError, 'excludes Tiny'):
                    legacy_epilogue_assembly(m,k,n,offset,0,epilogue=hook)

    def test_reject_modern_contracts(self):
        m, k, n = 3136, 64, 64
        offset = m*k+k*n+256
        class Derived(Epilogue):
            pass
        cases = [Epilogue(m,n,offset,(Step('silu'),)), Epilogue(m,n,offset,scratch_registers=(10,11)),
                 Epilogue(m,n,offset,output_dtype='fp8',output_row_stride=n*64), Derived(m,n,offset)]
        for hook in cases:
            with self.subTest(hook=hook), self.assertRaisesRegex(ValueError, 'original hook contract'):
                legacy_epilogue_assembly(m,k,n,offset,0,epilogue=hook)


if __name__ == '__main__':
    unittest.main()
