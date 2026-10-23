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
        for values in ({"ATLAS_TEST_MODEL_FACTS": "/missing/atlas-bundle"},
                       {"ATLAS_TEST_MODEL_FACTS": ""}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(FileNotFoundError):
                    model_evidence_available()

    def test_existing_pins_enable_tests_and_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ("facts.lock.json", "model-timing.json", "model-overlap.json"):
                (bundle / name).write_text("{}")
            with patch.dict(os.environ, {"ATLAS_TEST_MODEL_FACTS": str(bundle)}, clear=True):
                self.assertTrue(model_evidence_available())
                (bundle / "model-overlap.json").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "ATLAS_TEST_MODEL_FACTS"):
                    model_evidence_available()
