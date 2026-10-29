"""Source-only frontend timing, operand access and compiler capabilities."""
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_frontend as frontend
from model_evidence import ModelSourceFixture


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native source contracts")
class SourceContractTest(ModelSourceFixture, unittest.TestCase):
    source_shapes = frontend._METHODS

    def derive(self, sources=None, **kwargs):
        return frontend.derive_frontend_sources(sources or self.sources, self.scalar,
                                                fetch_width=kwargs.get("fetch_width", self.cfg.fetch_width))

    def test_delay_immediate_mask_is_extracted_and_propagates(self):
        source = self.mutated("core.py", "self.delay_counter = int(insn.imm) & 0xFFF",
                              "self.delay_counter = int(insn.imm) & 0x7FF")
        expected = self.derive()
        expected["delay"]["immediate_mask"] = 2047
        self.assertEqual(self.derive(source), expected)

    def test_delay_countdown_change_has_precise_unsupported_error(self):
        source = self.mutated("core.py", "self.delay_counter -= 1", "self.delay_counter -= 2")
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "only unit countdown reaches zero"):
            self.derive(source)

    def test_fetch_width_change_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "one S1 issue slot"):
            self.derive(fetch_width=2)

if __name__ == "__main__":
    unittest.main()
