"""Shared timing vocabulary and identity pin plumbing."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import atlas_model_schema as schema
from atlas_model_schema import generate


class ModelSchemaTest(unittest.TestCase):
    def load_changed(self, loader, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text(json.dumps(value))
            return loader(path)

    def test_python_consumers_share_vocabulary_and_pins(self):
        import atlas_model_pairs as pairs
        import atlas_model_timing as timing
        self.assertIs(pairs.RELATION_VALUES, schema.RELATION_VALUES)
        self.assertIs(pairs.QUALIFIED_CONSTRAINT_HASHES, schema.QUALIFIED_CONSTRAINT_HASHES)
        self.assertIs(timing.VPU_BLOCKS, schema.VPU_BLOCKS)
        self.assertIs(timing.MXU_BLOCKS, schema.MXU_BLOCKS)
        self.assertIs(timing.encoded, schema.canonical)
        self.assertIs(pairs.canonical, schema.canonical)
        self.assertIs(timing.sha256, pairs.sha256)
        for name in schema.LOCK:
            if name in {"schema", "source_shapes", "qualified_constraint_hashes"}:
                continue
            for consumer in (pairs, timing):
                if hasattr(consumer, name.upper()):
                    self.assertEqual(getattr(consumer, name.upper()), schema.LOCK[name])

    def test_lock_rejects_missing_unknown_and_malformed_pins(self):
        for name, value in (("model_revision", "a" * 39), ("source_sha256", "A" * 64),
                            ("per_op_sha256", None), ("unknown_sha256", "a" * 64)):
            with self.subTest(name=name):
                changed = dict(schema.LOCK, **{name: value})
                with self.assertRaises(ValueError):
                    self.load_changed(schema.load_lock, changed)
        changed = dict(schema.LOCK)
        del changed["master_sha256"]
        with self.assertRaises(ValueError):
            self.load_changed(schema.load_lock, changed)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text('{"schema": "first", "schema": "second"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                schema.load_lock(path)

    def test_vocabulary_rejects_invalid_roles_and_duplicate_mnemonics(self):
        for field, value in (("operand", "unknown"), ("registers", 0), ("passes", True),
                             ("write", 1), ("offset", -1), ("two_input", "yes")):
            with self.subTest(field=field):
                changed = deepcopy(schema.VOCABULARY)
                changed["vpu_operand_roles"]["VpuElementwise"][0][field] = value
                with self.assertRaises(ValueError):
                    self.load_changed(schema.load_vocabulary, changed)
        changed = deepcopy(schema.VOCABULARY)
        changed["vpu_blocks"].append(changed["vpu_blocks"][0])
        with self.assertRaises(ValueError):
            self.load_changed(schema.load_vocabulary, changed)

    def test_generation_is_deterministic_and_uses_modified_source(self):
        original = generate(schema.VOCABULARY)
        self.assertEqual(original, generate(deepcopy(schema.VOCABULARY)))
        changed = deepcopy(schema.VOCABULARY)
        changed["vpu_blocks"][0]["block"] = "modified"
        changed["relation_values"]["vmem_bank"].append("new_value")
        changed["vpu_operand_roles"]["VpuElementwise"][0]["passes"] = 3
        output = generate(changed)
        self.assertIn('"vpu.modified"', output)
        self.assertIn('"new_value"', output)
        self.assertIn('{"read0", &Instr::rs1, false, 2, 3, 0, false}', output)
        self.assertNotEqual(original, output)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_generated_cpp_tables_compile_and_preserve_operand_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "AtlasModelVocabulary.h").write_text(generate(schema.VOCABULARY))
            source = directory / "check.cpp"
            source.write_text('''#include "AtlasModelVocabulary.h"
#include <cassert>
#include <string>
using namespace mlir::atlas::timing;
using namespace mlir::atlas::timing::model_schema;
int main() {
  assert(std::string(vpuBlocks().at("vadd.bf16")) == "vpu.add");
  assert(std::string(mxuBlocks().at(OpClass::MatMulAcc)) == "matmul_acc");
  assert(relationValues().at("scalar_registers").count("shared_base_read_only"));
  const auto &elementwise = vpuEvents().at(OpClass::VpuElementwise);
  assert(elementwise.size() == 3 && elementwise[1].twoInput && elementwise[1].operand == &Instr::rs2);
  const auto &pack = vpuEvents().at(OpClass::VpuPack);
  assert(pack[0].registers == 2 && pack[1].key == nullptr && pack[1].operand == &Instr::rs1);
  const auto &row = vpuEvents().at(OpClass::VpuRowReduce);
  assert(row.size() == 4 && row[1].offset == 1 && row[3].offset == 1 && row[3].write);
  const auto &col = vpuEvents().at(OpClass::VpuColReduce);
  assert(col[0].passes == 2 && col[0].registers == 2 && col[1].registers == 2);
}
''')
            executable = directory / "check"
            subprocess.run([shutil.which("c++"), "-std=c++17", "-I" + str(ROOT / "include"), str(source),
                            "-o", str(executable)], check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
