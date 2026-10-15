"""Portable selected-tile binding controls; no model or RTL invocation."""
import copy
import os
import hashlib
import stat
import importlib.util
from pathlib import Path
import unittest
import tempfile
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


HERE = Path(__file__).absolute().parent
prepare = module(HERE / "prepare_cold_tile.py", "test_cold_preparer")
# Use the local authenticated oracle; an explicit path can select a frozen copy.
oracle = Path(os.environ.get("ATLAS_COLD_TILE_ORACLE", HERE.parent / "atlas_tile_arithmetic.py")).absolute()
if ".." in oracle.parts or any(bad in part.lower() for part in oracle.parts for bad in ("hammer", "vlsi")):
    raise ValueError("Unadmitted arithmetic dependency path")
current = Path(oracle.anchor)
for part in oracle.parts[1:]:
    current /= part
    if stat.S_ISLNK(current.lstat().st_mode):
        raise ValueError("Arithmetic dependency symlink refused")
if hashlib.sha256(oracle.read_bytes()).hexdigest() != prepare.ORACLE:
    raise ValueError("Arithmetic dependency hash differs")
arithmetic = module(oracle, "test_cold_arithmetic")


class Bindings(unittest.TestCase):
    def setUp(self):
        self.source = {name+"_bf16": np.full((32, 32), 0x3F80, dtype=np.uint16) for name in ("A", "B")}
        self.encoded = {name: np.full((32, 32), arithmetic.pack_bf16(0x3F80, 127), dtype=np.uint8) for name in ("A", "B")}
        self.tile = {"row_range": [0,32], "column_range": [0,32], "k_range": [0,32]}
        self.original = {"shape": [32,32,32], "tiles": [self.tile], "output_rescale_exponent": 0,
            "conversion": {name: {"scale_code": 127} for name in ("A", "B")},
            "recipe": "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1", "output_rescale_execution": "VLI_ALL_and_VMUL_BF16"}
        self.descriptor = {"tile_index": 0, "tile": copy.deepcopy(self.tile), "scale_codes": {"A":127,"B":127},
                           "output_rescale_exponent": 0, "factor_bf16": 0x3F80}
        self.target = np.full((32,32), 32, dtype=np.float32)
        self.initial = self.encoded["A"].tobytes()+self.encoded["B"].tobytes()+b"\xa5"*256+b"\xcd"*2048+b"\x5a"*256

    def run_binding(self):
        return prepare.selected_bytes(arithmetic, self.descriptor, self.original, self.source,
                                      self.encoded, self.target, self.initial)

    def test_positive(self):
        expected, memory = self.run_binding()
        self.assertEqual(expected, arithmetic.output_halves(np.full((32,32), 0x4200, dtype=np.uint16)))
        self.assertEqual(memory["total_dram_bytes"], 4608)

    def test_changed_source(self):
        self.source["A_bf16"][0,0] = 0x4000
        with self.assertRaisesRegex(ValueError, "source-to-global-encoded"):
            self.run_binding()

    def test_changed_global_scale(self):
        self.descriptor["scale_codes"]["A"] = 126
        with self.assertRaisesRegex(ValueError, "Global scale/factor"):
            self.run_binding()

    def test_changed_selected_tile(self):
        self.descriptor["tile"]["column_range"] = [32,64]
        with self.assertRaisesRegex(ValueError, "tile descriptor"):
            self.run_binding()

    def test_changed_payload(self):
        self.initial = b"\0"+self.initial[1:]
        with self.assertRaisesRegex(ValueError, "DRAM payload/guards"):
            self.run_binding()

    def test_changed_aggregate_actual(self):
        self.target[0,0] = 31
        with self.assertRaisesRegex(ValueError, "parent aggregate actual"):
            self.run_binding()

    def test_changed_factor(self):
        self.descriptor["factor_bf16"] ^= 128
        with self.assertRaisesRegex(ValueError, "Global scale/factor"):
            self.run_binding()

    def test_descriptor_types(self):
        valid = {**self.descriptor, "schema_version": 1, "source_kind": "cold_operator_tile"}
        prepare.validate_descriptor(valid)
        for key, bad in (("schema_version", True), ("tile_index", True), ("tile_index", -1),
                         ("scale_codes", {"A": True, "B": 127}), ("output_rescale_exponent", False),
                         ("output_rescale_exponent", 128), ("factor_bf16", 1.0), ("source_kind", "whole_program")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "descriptor types"):
                prepare.validate_descriptor({**valid, key: bad})

    def test_refuse_controls_and_reuse(self):
        for controls, reuse, model in ((True, False, Path("model")), (False, True, Path("model")), (False, False, None)):
            with self.subTest(controls=controls, reuse=reuse), self.assertRaisesRegex(ValueError, "fresh selected arithmetic"):
                prepare.prepare(None, None, None, None, None, controls, reuse_model_arithmetic=reuse, model_root=model)

    def test_archive_preflight(self):
        support = SimpleNamespace(safe_path=Path)
        with tempfile.TemporaryDirectory(prefix="atlas-cold-controls-") as directory:
            path = Path(directory) / "test.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("A.npy", b"fixture")
            self.assertEqual(prepare.preflight(support, path), path)
            for name in ("../A.npy", "/A.npy", "a\\b.npy", "forbidden-"+"ham"+"mer/A.npy", "V"+"LSI.npy"):
                metadata = MagicMock()
                metadata.__enter__.return_value.infolist.return_value = [SimpleNamespace(filename=name, external_attr=0)]
                with self.subTest(name=name), patch.object(prepare.zipfile, "ZipFile", return_value=metadata):
                    with self.assertRaisesRegex(ValueError, "archive member path"):
                        prepare.preflight(support, path)


if __name__ == "__main__":
    unittest.main()
