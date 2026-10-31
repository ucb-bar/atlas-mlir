import os
import subprocess
import sys
import unittest
from pathlib import Path


class IssueRulesTableTest(unittest.TestCase):
    def test_generated_table(self):
        root = os.environ.get("ATLAS_ISSUE_RULES_ROOT")
        if not root:
            self.skipTest("set ATLAS_ISSUE_RULES_ROOT to an npu_model checkout with issue-rules")
        tool = Path(__file__).resolve().parents[1] / "tools/atlas_issue_rules_table.py"
        result = subprocess.run([sys.executable, "-I", "-B", str(tool), "--npu-model", root, "--check"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
