"""External model evidence skips only when no pins were configured."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_evidence import model_evidence_available


class ModelEvidenceTest(unittest.TestCase):
    def test_unconfigured_evidence_skips(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(model_evidence_available())

    def test_partial_and_missing_evidence_fail(self):
        for values in ({"ATLAS_TEST_OP_TIMING": "/missing/atlas-op.json"},
                       {"ATLAS_TEST_OP_TIMING": "", "ATLAS_TEST_PAIR_TIMING": ""},
                       {"ATLAS_TEST_OP_TIMING": "/missing/atlas-op.json",
                        "ATLAS_TEST_PAIR_TIMING": "/missing/atlas-pairs.json"}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(FileNotFoundError):
                    model_evidence_available()

    def test_existing_pins_enable_tests_and_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            facts, pairs = Path(directory) / "facts.json", Path(directory) / "pairs.json"
            facts.write_text("{}")
            pairs.write_text("{}")
            with patch.dict(os.environ, {"ATLAS_TEST_OP_TIMING": str(facts),
                                         "ATLAS_TEST_PAIR_TIMING": str(pairs)}, clear=True):
                self.assertTrue(model_evidence_available())
                pairs.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "ATLAS_TEST_PAIR_TIMING"):
                    model_evidence_available()
