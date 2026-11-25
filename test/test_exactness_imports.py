"""The exactness checker imports only its siblings and its declared interfaces, never the compiler it checks."""
import ast
import sys
import unittest

from atlas_mlir.deployment_gate import CHECKER

# The interfaces python/atlas_mlir/exactness/README.md declares, besides the standard library and its siblings.
DECLARED = ('numpy', 'torch', 'npu_model', 'merlin.runtime.uniform_sparse_memory', 'tools.check_program_decode',
            'tools.model_instructions')


def imported(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield '.' * node.level + (node.module or '')


class ExactnessImportsTest(unittest.TestCase):
    def test_checker_is_not_a_package_of_the_compiler(self):
        self.assertFalse((CHECKER / '__init__.py').exists())

    def test_only_siblings_and_declared_interfaces_are_imported(self):
        sources = sorted(CHECKER.glob('*.py'))
        siblings = {path.stem for path in sources}
        self.assertTrue({'check_model', 'functional_runtime', 'model_runtime', 'qualified_native_dot'} <= siblings)
        for path in sources:
            for name in imported(path):
                with self.subTest(file=path.name, module=name):
                    self.assertTrue(name.split('.')[0] in sys.stdlib_module_names or name in siblings
                                    or any(name == item or name.startswith(item + '.') for item in DECLARED),
                                    f'{path.name} imports {name}')


if __name__ == '__main__':
    unittest.main()
