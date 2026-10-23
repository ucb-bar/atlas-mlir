"""Source dependency closure and conservative semantic normalization."""
import ast
import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import atlas_model_semantics as semantics
from atlas_model_diagnostics import UnsupportedModelTiming


class NormalizationTest(unittest.TestCase):
    def test_unsupported_python_refuses_before_source_access(self):
        from collections import namedtuple
        version = namedtuple("Version", "major minor micro releaselevel serial")
        for minor in (12, 13, 15):
            with self.subTest(minor=minor), patch.object(semantics.sys, "version_info", version(3, minor, 0, "final", 0)), patch.object(semantics, "read_source", side_effect=AssertionError("source accessed")):
                with self.assertRaisesRegex(UnsupportedModelTiming, "requires Python 3.14"):
                    semantics.check_model_semantics(Path("unused"))

    def same(self, before, after):
        self.assertEqual(semantics.module_contract(before), semantics.module_contract(after))

    def different(self, before, after):
        self.assertNotEqual(semantics.module_contract(before), semantics.module_contract(after))

    def test_comments_and_genuine_docstrings_are_observational(self):
        before = '"""module docs"""\nclass Core:\n    """class docs"""\n    def reset(self):\n        """method docs"""\n        self.cycle = 0\n'
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
            '    try:\n        temporary = value()\n    except Exception as error:\n        temporary = error\n    return temporary\n',
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
        before = 'from .helper import operation\nLIMIT = 32\ndef helper(value):\n    logger.log_event(operation(value))\n    return LIMIT\n'
        for old, new in ((".helper", ".other"), ("LIMIT = 32", "LIMIT = 64"),
                         ("operation(value)", "other(value)")):
            with self.subTest(old=old):
                self.different(before, before.replace(old, new))


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"),
                     "set ATLAS_DERIVE_MODEL_ROOT for selected source dependency contracts")
class DependencyClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        manifest = json.loads((TOOLS / "model-semantics.json").read_text())
        cls.sources = {}
        for relative in manifest["files"]:
            path = cls.model / "npu_model" / relative
            if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
                raise AssertionError("prohibited source path")
            if ".." in path.parts:
                raise AssertionError("source parent traversal")
            for parent in reversed((path, *path.parents)):
                if parent.is_symlink():
                    raise AssertionError("source symlink")
            cls.sources[relative] = path.read_text()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.copy = Path(self.directory.name)
        for relative, source in self.sources.items():
            path = self.copy / "npu_model" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def reject_mutation(self, relative, old, new):
        original = self.sources[relative]
        self.assertIn(old, original, "mutation must alter the selected source")
        path = self.copy / "npu_model" / relative
        path.write_text(original.replace(old, new, 1))
        try:
            with self.assertRaises(UnsupportedModelTiming):
                semantics.check_model_semantics(self.copy)
        finally:
            path.write_text(original)

    def test_manifest_covers_the_explicit_transitive_dependencies(self):
        expected = {"hardware/" + name + ".py" for name in
                    ("core", "arch_state", "exu", "ifu", "stage_data", "hardware", "sparse_memory")}
        expected.update({"software/instruction.py", "software/program.py", "isa.py", "isa_types.py",
                         "isa_patterns.py", "configs/isa_definition.py"})
        self.assertTrue(expected <= self.sources.keys(), expected - self.sources.keys())
        # A partial source-only copy has neither a package initializer nor its
        # numerical dependencies: acceptance cannot depend on importing it.
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "npu_model" or name.startswith("npu_model."):
                raise AssertionError("model import during source checking")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import):
            inventory = semantics.check_model_semantics(self.copy)
        self.assertTrue(inventory)

    def test_fatal_policy_dispatch_and_cold_initialization_mutations_are_rejected(self):
        cases = (
            ("hardware/core.py", "if not self.ignore_runtime_errors:\n            return False",
             "if not self.ignore_runtime_errors:\n            return True"),
            ("hardware/core.py", "EXU(type(exu).__name__): exu", "EXU(type(exu).__name__): self.exus[0]"),
            ("hardware/core.py", "self.cycle_count = 0", "self.cycle_count = 1"),
            ("hardware/core.py", "self.delay_counter = 0", "self.delay_counter = 1"),
            ("hardware/core.py", "self.ifu.load_program(program)", "self.ifu.reset()"),
            ("hardware/exu.py", "self._pending_completion_uop: Uop | None = None",
             "self._pending_completion_uop: Uop | None = stale_uop"),
            ("hardware/arch_state.py", "self.initialize_buffers()", "self.initialize_buffers = None"),
            ("hardware/arch_state.py", "self.pc = 0", "self.pc = 1"),
        )
        for relative, old, new in cases:
            with self.subTest(relative=relative, old=old):
                self.reject_mutation(relative, old, new)

    def test_scalar_flags_csr_and_bf16_storage_mutations_are_rejected(self):
        cases = (
            ("if rd == 0:", "if rd == 1:"),
            ("return self.xrf[rs]", "return 0"),
            ("self.erf[rd] = value & 0xFF", "self.erf[rd] = value & 0x7F"),
            ("self.flags[flag] = True", "self.flags[flag] = False"),
            ("self.flags[flag] = False", "self.flags[flag] = True"),
            ("return self.flags[flag]", "return False"),
            ("self._csr_written.clear()", "self._csr_written = {0xC00}"),
            ("self._csr_values[address] + increment", "self._csr_values[address] + increment + 1"),
            ("self.mrf[vd].view(torch.bfloat16)[:] = value.flatten()",
             "self.mrf[vd + 1].view(torch.bfloat16)[:] = value.flatten()"),
            ("self.read_mrf_bf16(vs + 1)", "self.read_mrf_bf16(vs + 2)"),
        )
        for old, new in cases:
            with self.subTest(old=old):
                self.reject_mutation("hardware/arch_state.py", old, new)

    def test_imem_stage_handoff_and_instruction_pc_mutations_are_rejected(self):
        cases = (
            ("hardware/ifu.py", "WORDS = 32768", "WORDS = 16384"),
            ("hardware/ifu.py", "read_address = fetch_address & (self.WORDS - 1)",
             "read_address = (fetch_address + 1) & (self.WORDS - 1)"),
            ("hardware/ifu.py", "self.words[:len(program)] = deepcopy(program.instructions)",
             "self.words[1:len(program) + 1] = deepcopy(program.instructions)"),
            ("hardware/stage_data.py", "self.valid = self._is_non_empty(data)", "self.valid = True"),
            ("hardware/stage_data.py", "self.valid = False\n            return data",
             "self.valid = True\n            return data"),
            ("software/instruction.py", "self.pc = pc", "self.pc = pc + 1"),
            ("software/program.py", "self.instructions = instructions", "self.instructions = instructions[1:]"),
        )
        for relative, old, new in cases:
            with self.subTest(relative=relative, old=old):
                self.reject_mutation(relative, old, new)

    def test_isa_helper_bindings_and_sparse_memory_mutations_are_rejected(self):
        cases = (
            ("configs/isa_definition.py", "_MASK32 = 0xFFFFFFFF", "_MASK32 = 0x7FFFFFFF"),
            ("configs/isa_definition.py", "value -= 1 << length", "value += 1 << length"),
            ("isa.py", "IsaSpec.operations[cls.mnemonic] = cls", "IsaSpec.operations[cls.mnemonic] = Instruction"),
            ("isa_types.py", 'SCALAR = "ScalarExecutionUnit"', 'SCALAR = "LoadStoreUnit"'),
            ("isa_patterns.py", 'Named(ScalarReg, "rs1")', 'Named(ScalarReg, "rs2")'),
            ("hardware/sparse_memory.py", "_PAGE_BITS = 16", "_PAGE_BITS = 15"),
            ("hardware/sparse_memory.py", "out[address - start:address - start + run].zero_()",
             "out[address - start:address - start + run].fill_(1)"),
            ("hardware/sparse_memory.py", "return start, max(start, stop)", "return start + 1, max(start, stop)"),
        )
        for relative, old, new in cases:
            with self.subTest(relative=relative, old=old):
                self.reject_mutation(relative, old, new)

    def test_added_override_and_changed_import_binding_are_rejected(self):
        self.reject_mutation("hardware/core.py", "from .ifu import InstructionFetch",
                             "from .other_ifu import InstructionFetch")
        self.reject_mutation("hardware/core.py", "class Core(Module):",
                             "class Core(Module):\n    def _handle_runtime_error(self, stage, exc):\n        return True\n")
        relative = "hardware/core.py"
        path = self.copy / "npu_model" / relative
        path.write_text(self.sources[relative] + "\nCore.tick = lambda self: None\n")
        with self.assertRaises(UnsupportedModelTiming):
            semantics.check_model_semantics(self.copy)

    def test_selected_source_comment_and_docstring_changes_are_accepted(self):
        baseline = semantics.check_model_semantics(self.copy)
        path = self.copy / "npu_model" / "hardware/core.py"
        tree = ast.parse(self.sources["hardware/core.py"])
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Core")
        self.assertIsInstance(owner.body[0], ast.Expr)
        owner.body[0].value = ast.Constant(value="Updated explanatory documentation.")
        path.write_text("# editorial comment\n" + ast.unparse(tree) + "\n")
        self.assertEqual(semantics.check_model_semantics(self.copy), baseline)


if __name__ == "__main__":
    unittest.main()
