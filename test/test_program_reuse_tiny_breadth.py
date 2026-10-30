"""Direct prefetch-generator admission checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_tiny_breadth as producer


class BreadthTests(unittest.TestCase):
    def test_admission(self):
        for shape in [(32.,2048,1024), (32,2048,1024.), (32,2048,1056)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                producer.allocation(*shape)
        for args in [(128,2048,2048,4456704,-14),(128,5632,1024,0,-18),
                     (32,2048,2048,4260096,False),(32,2048,2048,4260096,-127)]:
            with self.assertRaises(ValueError):producer.assembly(*args)


if __name__ == '__main__':
    unittest.main()
