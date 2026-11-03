"""Admission for matching recipe and appended-post-pass scalar scratch."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atlas_norm_postpass import ResidentNormEpilogue, postpass


class NormScratchTests(unittest.TestCase):
    def spec(self, kind, width, **changes):
        args = dict(m=128, n=width, real_n=width, norm_kind=kind, output_offset=4096,
                    norm_mask_offset=1048576, norm_weight_offset=1114112,
                    norm_bias_offset=1114112 + 64 * width if kind == 'layer' else None)
        return ResidentNormEpilogue(**(args | changes))

    def test_reject_default_sentinel_before_composing_postpass(self):
        for kind, width in (('layer', 768), ('rms', 960)):
            for pair in ((30, 31), (10, 31), (30, 11)):
                with self.subTest(kind=kind, pair=pair), self.assertRaisesRegex(ValueError, 'explicit recipe'):
                    self.spec(kind, width, scratch_registers=pair)

    def test_explicit_recipe_pairs_emit_no_hook_use_of_reserved_registers(self):
        for kind, width, pair in (('layer', 768, (27, 28)), ('rms', 960, (10, 11))):
            with self.subTest(kind=kind):
                source = '\n'.join(postpass(self.spec(kind, width, scratch_registers=pair)))
                self.assertNotIn('x31', source)
                self.assertNotIn('x30', source)


if __name__ == '__main__':
    unittest.main()
