"""Source-derived PE windows and whole-row projection."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_mxu as mxu
from model_evidence import NativeModelFixture


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native weight checks")
class NativeWeightTest(NativeModelFixture, unittest.TestCase):
    def test_changed_weight_masks_mapping_and_read_order_refuse(self):
        source, checker = mxu._sources(self.cfg)
        mutations = (
            ("1 <= age <= 32", "1 <= age < 32"),
            ("row <= 31", "row < 31"),
            ("age - 1 - _GRID[:, None] - _GRID[None, :]", "age - 1 - _GRID[:, None]"),
            ("used & (age == 1)", "used & (age == 2)"),
            ("read_wb_u8(self.mxu, weight_push.weight)[row]", "read_wb_u8(self.mxu, weight_push.weight)[:, row]"),
            ("self._sample_weights(op, self.cycle - op.issued)", "self._sample_weights(op, self.cycle - op.issued - 1)"),
        )
        for old, new in mutations:
            self.assertIn(old, source)
            with self.subTest(mutation=old), patch.object(mxu, "_sources", return_value=(source.replace(old, new), checker)):
                with self.assertRaises(mxu.UnsupportedMXU):
                    mxu.derive_weight_timing(self.cfg)

if __name__ == "__main__":
    unittest.main()
