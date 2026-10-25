"""Shared timing vocabulary and identity pin plumbing."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import atlas_model_schema as schema


class ModelSchemaTest(unittest.TestCase):
    def load_changed(self, loader, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text(json.dumps(value))
            return loader(path)

    def test_lock_rejects_missing_unknown_and_malformed_pins(self):
        for name, value in (("model_revision", "a" * 39), ("source_sha256", "A" * 64),
                            ("numerical_data_sha256", None), ("unknown_sha256", "a" * 64)):
            with self.subTest(name=name):
                changed = dict(schema.LOCK, **{name: value})
                with self.assertRaises(ValueError):
                    self.load_changed(schema.load_lock, changed)
        changed = dict(schema.LOCK)
        del changed["source_sha256"]
        with self.assertRaises(ValueError):
            self.load_changed(schema.load_lock, changed)

    def test_old_lock_without_dma_contract_remains_loadable(self):
        value = {key: item for key, item in schema.LOCK.items()
                 if key not in {"curve_sha256", "dma_source_sha256"}}
        self.assertEqual(self.load_changed(schema.load_lock, value), value)
        with self.assertRaisesRegex(ValueError, "curve_sha256"):
            self.load_changed(schema.load_lock, dict(value, curve_sha256="bad"))

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

if __name__ == "__main__":
    unittest.main()
