"""Opt-in independent span96 arithmetic qualification, without emitted execution.

From the repository root, with NumPy, Torch and Merlin installed:
    PYTHONPATH=python:$MERLIN_NATIVE_ROOT/src python test/test_span96_model_qualification.py --model-root /path/to/npu-model

The model must contain commit b97b2f5 and byte-identical arithmetic sources and
EXP/RCP tables. Ordinary unittest discovery skips this expensive comparison.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

MODEL_ROOT = None
MODEL_REVISION = "b97b2f5"
MODEL_FILES = (
    "npu_model/__init__.py", "npu_model/configs/__init__.py",
    "npu_model/configs/numerics.py", "npu_model/configs/rtl_math.py",
    "npu_model/configs/data/exp.bin.gz", "npu_model/configs/data/rcp.bin.gz",
)


def authenticate_model(root):
    """Check original Git blobs before importing any model code or LUT."""
    root = Path(root).resolve(strict=True)

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        if result.returncode:
            raise ValueError("model authentication failed: " + result.stderr.decode().strip())
        return result.stdout

    revision = git("rev-parse", "--verify", MODEL_REVISION + "^{commit}").decode().strip()
    git("cat-file", "-e", revision + "^{commit}")
    names = set(MODEL_FILES)
    for name, module in tuple(sys.modules.items()):
        if name == "npu_model" or name.startswith("npu_model."):
            filename = getattr(module, "__file__", None)
            if filename:
                try:
                    relative = Path(filename).resolve(strict=True).relative_to(root)
                except ValueError as error:
                    raise ValueError("imported model module is outside authenticated root: " + name) from error
                names.add(relative.as_posix())
    sources = {}
    for name in sorted(names):
        expected = git("cat-file", "blob", revision + ":" + name)
        path = root / name
        if path.resolve(strict=True) != path or path.read_bytes() != expected:
            raise ValueError("model source or LUT differs from pinned Git blob: " + name)
        sources[name] = hashlib.sha256(expected).hexdigest()
    return {"revision": revision, "sha256": sources}


@unittest.skipUnless(MODEL_ROOT, "run this file with explicit --model-root for pinned-model qualification")
class Span96ModelQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = authenticate_model(MODEL_ROOT)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(MODEL_ROOT))
        import numpy as np
        import torch
        model = importlib.import_module("npu_model.configs.numerics")
        if Path(model.__file__).resolve() != MODEL_ROOT / "npu_model/configs/numerics.py":
            raise ValueError("model module was imported from a different root")
        from atlas_mlir_native.bf16_domains import (
            vexp_bf16_nonpositive_bits, mul_bf16_finite_bits,
            build_bf16_primitive_request,
        )
        from atlas_mlir_native.recip_lane import vrecip_bf16_bits
        from atlas_mlir_native.reference import (
            evaluate_bf16_row_sum_nonnegative, evaluate_bf16_add_nonnegative,
        )
        from merlin.compiler.semantic.reference import TensorValue
        cls.provenance = authenticate_model(MODEL_ROOT)
        torch.set_flush_denormal(False)
        cls.np, cls.torch, cls.rtl = np, torch, model.RtlNumerics
        cls.exp_port = staticmethod(vexp_bf16_nonpositive_bits)
        cls.mul_port = staticmethod(mul_bf16_finite_bits)
        cls.rcp_port = staticmethod(vrecip_bf16_bits)
        cls.request = staticmethod(build_bf16_primitive_request)
        cls.row_port = staticmethod(evaluate_bf16_row_sum_nonnegative)
        cls.add_port = staticmethod(evaluate_bf16_add_nonnegative)
        cls.value = TensorValue
        cls.results = {}
        cls.shifted = np.array([0, *range(0x8000, 0xc2c1)], dtype=np.uint16)
        cls.exp_expected = cls.bits(cls.rtl.unary("exp", cls.tensor(cls.shifted)))
        cls.exps = np.unique(np.concatenate((cls.exp_expected, np.arange(128, dtype=np.uint16))))
        cls.denominators = np.arange(0x3f80, 0x4481, dtype=np.uint16)
        cls.reciprocals = cls.bits(cls.rtl.unary("rcp", cls.tensor(cls.denominators)))
        print(json.dumps({"model": cls.provenance, "numpy": np.__version__,
                          "torch": torch.__version__}), flush=True)

    @classmethod
    def tensor(cls, codes):
        return cls.torch.from_numpy(cls.np.asarray(codes, dtype=cls.np.uint16).copy()).view(cls.torch.bfloat16)

    @classmethod
    def bits(cls, value):
        return value.contiguous().view(cls.torch.int16).numpy().view(cls.np.uint16)

    def compare(self, name, expected, actual):
        np = self.np
        expected, actual = (np.asarray(x, dtype=np.uint16).reshape(-1) for x in (expected, actual))
        self.assertEqual(expected.shape, actual.shape, name)
        bad = np.flatnonzero(expected != actual)
        result = {"cases": int(expected.size), "mismatches": int(bad.size),
                  "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                  "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        self.results[name] = result
        print(name, json.dumps(result), flush=True)
        self.assertEqual(int(bad.size), 0, f"{name}: first mismatches {bad[:16].tolist()}")

    def test_exhaustive_exp_and_masks(self):
        self.compare("exp_all_shifted_codes", self.exp_expected,
                     [self.exp_port(int(x)) for x in self.shifted])
        for mask in (0, 0x3f80):
            masks = self.np.full(len(self.exps), mask, dtype=self.np.uint16)
            self.compare("mask_" + hex(mask), self.bits(self.rtl.mul(self.tensor(self.exps), self.tensor(masks))),
                         [self.mul_port(int(x), mask) for x in self.exps])

    def test_all_denominators_and_normalization_cartesian(self):
        self.compare("reciprocal_denominators", self.reciprocals,
                     [self.rcp_port(int(x), wide=True) for x in self.denominators])
        expected, actual = [], []
        for reciprocal in self.np.unique(self.reciprocals):
            rhs = self.np.full(len(self.exps), reciprocal, dtype=self.np.uint16)
            expected.append(self.bits(self.rtl.mul(self.tensor(self.exps), self.tensor(rhs))))
            actual.extend(self.mul_port(int(x), int(reciprocal)) for x in self.exps)
        self.compare("normalization_cartesian", self.np.concatenate(expected), actual)

    def check_callback(self, name, op, callback, inputs, expected):
        np = self.np
        node = self.request(op).nodes[-1]
        inputs = [np.asarray(x, dtype=np.uint16).reshape(-1) for x in inputs]
        expected = np.asarray(expected, dtype=np.uint16).reshape(-1)
        actual = []
        for start in range(0, len(expected), 1024):
            length = min(1024, len(expected) - start)
            operands = tuple(self.value(node.type, tuple(int(x) for x in
                             np.pad(values[start:start + length], (0, 1024 - length)))) for values in inputs)
            actual.extend(callback(node, operands).elements[:length])
        self.compare(name, expected, actual)

    def test_directed_row_sum_subnormals(self):
        np = self.np
        rows = [np.full(32, x, np.uint16) for x in self.exps]
        for subnormal in range(1, 128):
            for position in range(32):
                row = np.full(32, subnormal, np.uint16)
                row[position] = 0x3f80
                rows.append(row)
            row = np.zeros(32, np.uint16)
            row[0] = subnormal
            rows.append(row)
        rows.extend(np.random.default_rng(7319).choice(self.exps, (1024, 32)))
        rows = np.asarray(rows, np.uint16)
        expected = np.repeat(self.bits(self.rtl.row_sum(self.tensor(rows))), 32, axis=-1)
        self.check_callback("row_sum_subnormal_trees", "row_sum_bf16_nonnegative",
                            self.row_port, [rows], expected)

    def test_directed_add_partial_sums(self):
        np = self.np
        lhs = np.tile(np.arange(0x4481, dtype=np.uint16), 5)
        rhs = np.repeat(np.array([0, 1, 128, 0x3f80, 0x4200], np.uint16), 0x4481)
        rng = np.random.default_rng(7319)
        lhs = np.concatenate((lhs, rng.integers(0, 0x4481, 8192, dtype=np.uint16)))
        rhs = np.concatenate((rhs, rng.integers(0, 0x4481, 8192, dtype=np.uint16)))
        self.check_callback("add_partial_sums", "add_bf16_nonnegative", self.add_port,
                            [lhs, rhs], self.bits(self.rtl.add(self.tensor(lhs), self.tensor(rhs))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    MODEL_ROOT = args.model_root.resolve()
    # unittest's class decorator ran before CLI parsing.
    Span96ModelQualification.__unittest_skip__ = False
    unittest.main(argv=[sys.argv[0]], verbosity=2)
