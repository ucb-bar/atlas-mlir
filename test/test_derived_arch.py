"""Consumed model geometry and representation limits."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_arch import derive_arch
from atlas_model_capabilities import check_configuration
from atlas_model_diagnostics import UnsupportedModelTiming
from model_evidence import NativeModelFixture


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT")
class ArchitectureTest(NativeModelFixture, unittest.TestCase):
    def test_consumed_geometry(self):
        facts = derive_arch(self.cfg)
        self.assertEqual(len(facts), 16)
        self.assertEqual((facts["mreg_count"], facts["mreg_bytes"], facts["bf16_registers_per_tile"]), (64, 1024, 2))
        self.assertEqual((facts["vmem_bank_count"], facts["scalar_word_bytes"]), (6, 4))

    def test_unsupported_rows_refuse(self):
        self.cfg.arch_state_config = replace(self.cfg.arch_state_config, mrf_depth=16)
        with self.assertRaises(UnsupportedModelTiming):
            check_configuration(self.cfg)
