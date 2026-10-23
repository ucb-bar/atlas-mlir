"""Controls for exact-byte shared assets and early capture admission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("tiny_full_export", Path(__file__).absolute().parents[1]
                                            / "tools/tiny_full/export_capture.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class FullExportTest(unittest.TestCase):
    def test_identical_bytes_share_asset_distinct_rescale_words_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            store = exporter.Assets(Path(directory))
            first = np.array([0x00000013, 0x3f800037, 0x73], dtype="<u4").tobytes()
            second = np.array([0x00000013, 0x40000037, 0x73], dtype="<u4").tobytes()
            path = store.put("programs", first)
            self.assertEqual(store.put("programs", bytes(first)), path)
            self.assertNotEqual(store.put("programs", second), path)
            self.assertEqual(len(store.sums), 2)

    def test_array_payload_is_bound_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.array([[0x3f80, 0x8000]], dtype="<u2").tobytes()
            (root / "words.bin").write_bytes(raw)
            descriptor = {"file": "words.bin", "shape": [1, 2], "dtype": "uint16", "bytes": 4,
                          "sha256": hashlib.sha256(raw).hexdigest()}
            words, _ = exporter.read_array(root, descriptor)
            self.assertEqual(words.tobytes(), raw)
            (root / "words.bin").write_bytes(raw[:-1] + b"\x01")
            with self.assertRaises(ValueError):
                exporter.read_array(root, descriptor)

    def test_partial_capture_refused_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"status": "failed"}))
            args = argparse.Namespace(capture=receipt, capture_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                      model_root=root / "model", out=root / "bundle")
            with mock.patch.object(exporter, "configure_model") as model:
                with self.assertRaises(ValueError):
                    exporter.export(args)
                model.assert_not_called()
            self.assertFalse(args.out.exists())

    def test_restricted_payload_rejected_before_filesystem_probe(self):
        with mock.patch.object(exporter, "physical_checked") as physical:
            with self.assertRaises(ValueError):
                exporter.checked("/source/old-VLSI/payload")
            physical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
