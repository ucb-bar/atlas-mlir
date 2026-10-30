"""Direct connector checks; shared proofs cover the emitted program."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_smol_connector as producer


class ConnectorTest(unittest.TestCase):

    def test_rejected_inputs(self):
        for shape in ((64., 12288, 960), (32, 12288, 960), (64, 12256, 960)):
            with self.assertRaises(ValueError):
                producer.layout(*shape)
        args = (64, 12288, 960, 12583168)
        for exponent in (None, True, -127, 128):
            with self.assertRaises(ValueError):
                producer.assembly(*args, exponent)


if __name__ == '__main__':
    unittest.main()
