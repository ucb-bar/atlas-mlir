"""Source pins ignore comments and docstrings while retaining module structure."""
from collections import namedtuple
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import atlas_model_semantics as semantics
from atlas_model_source import UnsupportedModelTiming


class SourcePinsTest(unittest.TestCase):
    def test_unsupported_python_refuses_before_source_access(self):
        version = namedtuple("Version", "major minor micro releaselevel serial")
        for minor in (12, 13, 15):
            with (self.subTest(minor=minor),
                  patch.object(semantics.sys, "version_info", version(3, minor, 0, "final", 0)),
                  patch.object(semantics, "read_source", side_effect=AssertionError("source accessed")),
                  self.assertRaisesRegex(UnsupportedModelTiming, "requires Python 3.14")):
                semantics.check_model_semantics(Path("unused"))

    def test_comments_and_leading_docstrings_keep_the_pin(self):
        before = ('"""module docs"""\nclass Core:\n    """class docs"""\n'
                  '    def reset(self):\n        """method docs"""\n        self.cycle = 0\n')
        after = '# a comment\nclass Core:\n    def reset(self):\n        self.cycle = 0  # another comment\n'
        self.assertEqual(semantics.module_contract(before), semantics.module_contract(after))

    def test_structure_changes_require_acceptance(self):
        local = 'def helper(value):\n    temporary = value + 1\n    return temporary\n'
        first = '    def first(self):\n        return 1\n'
        second = '    def second(self):\n        return 2\n'
        expression = 'def helper():\n    value = 1\n    "a real expression"\n    return value\n'
        for before, after in ((local, local.replace("temporary", "result")),
                              ('class Core:\n' + first + second, 'class Core:\n' + second + first),
                              (expression, expression.replace("a real expression", "changed expression"))):
            with self.subTest(after=after):
                self.assertNotEqual(semantics.module_contract(before), semantics.module_contract(after))


if __name__ == "__main__":
    unittest.main()
