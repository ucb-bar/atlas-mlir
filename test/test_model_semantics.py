"""Source dependency closure and conservative semantic normalization."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import atlas_model_semantics as semantics
from atlas_model_source import UnsupportedModelTiming


class NormalizationTest(unittest.TestCase):
    def test_unsupported_python_refuses_before_source_access(self):
        from collections import namedtuple
        version = namedtuple("Version", "major minor micro releaselevel serial")
        for minor in (12, 13, 15):
            with (self.subTest(minor=minor),
                  patch.object(semantics.sys, "version_info", version(3, minor, 0, "final", 0)),
                  patch.object(semantics, "read_source", side_effect=AssertionError("source accessed"))):
                with self.assertRaisesRegex(UnsupportedModelTiming, "requires Python 3.14"):
                    semantics.check_model_semantics(Path("unused"))

    def same(self, before, after):
        self.assertEqual(semantics.module_contract(before), semantics.module_contract(after))

    def different(self, before, after):
        self.assertNotEqual(semantics.module_contract(before), semantics.module_contract(after))

    def test_comments_and_genuine_docstrings_are_observational(self):
        before = ('"""module docs"""\nclass Core:\n    """class docs"""\n'
                  '    def reset(self):\n        """method docs"""\n        self.cycle = 0\n')
        after = '# a comment\nclass Core:\n    def reset(self):\n        self.cycle = 0  # another comment\n'
        self.same(before, after)

    def test_only_leading_string_expressions_are_docstrings(self):
        before = 'def helper():\n    value = 1\n    "a real expression"\n    return value\n'
        self.different(before, before.replace('"a real expression"', '"changed expression"'))

    def test_simple_local_renames_preserve_the_contract(self):
        before = 'def helper(value):\n    temporary = value + 1\n    return temporary * 2\n'
        self.same(before, before.replace("temporary", "result"))

    def test_argument_names_and_annotations_are_preserved(self):
        before = 'def helper(value: int) -> int:\n    temporary = value + 1\n    return temporary\n'
        self.different(before, before.replace("value", "operand"))
        self.different(before, before.replace("value: int", "value: float"))

    def test_complex_scopes_do_not_alpha_rename_locals(self):
        bodies = (
            '    temporary = [item for item in value]\n    return temporary\n',
            '    temporary = value\n    def nested():\n        return temporary\n    return nested()\n',
            '    try:\n        temporary = value()\n    except Exception as error:\n'
            '        temporary = error\n    return temporary\n',
            '    temporary = value\n    match value:\n        case 0:\n            return temporary\n    return None\n',
            '    temporary = value\n    return locals()["temporary"]\n',
            '    temporary = value\n    return sys._getframe().f_locals["temporary"]\n',
        )
        for body in bodies:
            with self.subTest(body=body):
                before = 'def helper(value):\n' + body
                # Preserve string keys used by reflective code; renaming them
                # too would mask an unsafe normalization of local variables.
                tree = ast.parse(before)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == "temporary":
                        node.id = "result"
                self.different(before, ast.unparse(tree))

    def test_plain_known_class_method_order_is_observational(self):
        first = '    def reset(self):\n        self.cycle = 0\n'
        second = '    @property\n    def done(self):\n        return self.cycle == 0\n'
        self.same('class Core:\n' + first + second, 'class Core:\n' + second + first)

    def test_unknown_classes_and_executable_defaults_preserve_order(self):
        first = '    def first(self):\n        return 1\n'
        second = '    def second(self):\n        return 2\n'
        self.different('class Arbitrary:\n' + first + second, 'class Arbitrary:\n' + second + first)
        dynamic = '    def second(self, value=register()):\n        return value\n'
        self.different('class Core:\n' + first + dynamic, 'class Core:\n' + dynamic + first)

    def test_evaluated_annotations_preserve_method_definition_order(self):
        first = '    def first(self) -> register("first"):\n        return 1\n'
        second = '    def second(self) -> register("second"):\n        return 2\n'
        self.different('class Core:\n' + first + second, 'class Core:\n' + second + first)

    def test_duplicate_methods_and_runtime_assignments_are_preserved(self):
        before = 'class Core:\n    def reset(self):\n        self.cycle = 0\n'
        with self.assertRaises(UnsupportedModelTiming):
            semantics.module_contract(before + '    def reset(self):\n        self.cycle = 1\n')
        self.different(before, before + 'Core.reset = replacement\n')

    def test_import_bindings_constants_and_logging_arguments_are_preserved(self):
        before = ('from .helper import operation\nLIMIT = 32\ndef helper(value):\n'
                  '    logger.log_event(operation(value))\n    return LIMIT\n')
        for old, new in ((".helper", ".other"), ("LIMIT = 32", "LIMIT = 64"),
                         ("operation(value)", "other(value)")):
            with self.subTest(old=old):
                self.different(before, before.replace(old, new))


if __name__ == "__main__":
    unittest.main()
