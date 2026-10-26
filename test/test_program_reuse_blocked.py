"""Direct blocked-generator admission; shared proofs cover every dispatch shape."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_blocked as producer


class BlockedReuseTests(unittest.TestCase):
    def test_strict_admission(self):
        for shape in ((224., 2304, 256), (225, 2304, 256), (32, 32, 32),
                      (32, 2048, 1024), (128., 768, 3072), (129, 768, 3072),
                      (64, 12288, 960), (128, 768, 768), (128, 3072, 768), (64, 736, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                producer.layout(*shape)
        for m, k, n in ((224, 2304, 256), (64, 320, 320)):
            offset = m * k + k * n + 256
            for bad_offset in (0, offset + 1, float(offset), True):
                with self.subTest(offset=bad_offset), self.assertRaises(ValueError):
                    producer.assembly(m, k, n, bad_offset, -13)
            for exponent in (None, -127, 128, True, 0.0):
                with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                    producer.assembly(m, k, n, offset, exponent)


if __name__ == '__main__':
    unittest.main()
