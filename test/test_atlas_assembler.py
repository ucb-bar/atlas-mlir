"""Selected assembler dependencies fail explicitly."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import atlas_assembler


class AtlasAssemblerTest(unittest.TestCase):
    def test_unselected_reference_skips(self) -> None:
        with patch.dict(os.environ, {"ATLAS_ASSEMBLER_ROOT": ""}):
            with self.assertRaises(unittest.SkipTest):
                atlas_assembler.assemble("nop")

    def test_missing_selected_dependency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "atlas_asm.py"
            with self.assertRaisesRegex(AssertionError, str(path)):
                atlas_assembler.assemble("nop", temp)


if __name__ == "__main__":
    unittest.main()
