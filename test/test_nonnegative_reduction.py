"""Explicit positive-normal reduction admission with unchanged native arithmetic."""
from __future__ import annotations

import ctypes
from dataclasses import replace
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

import numpy as np

from atlas_mlir_native.reduction import (
    NONNEGATIVE_NORMAL_REDUCTION_POLICY as POSITIVE,
    REDUCTION_POLICY,
    body_request,
    input_spec,
    prepare_plan,
    prepare_source,
    request_for,
)
from atlas_mlir_native.reduction_c import render_nonnegative_guard_c, render_nonnegative_encoder_c
from atlas_mlir_native.reference import evaluate_seeded_bf16_reduction
from merlin.semantic_compiler.reference import TensorValue


def source_request(shape=(1, 33), axes=(1,)):
    request = request_for(shape, axes)
    nodes = tuple(replace(node, type=replace(node.type, dtype="f32", numerical_policy="raw-bit-pattern"), attrs=())
                  for node in request.nodes[:2])
    result = request.nodes[-1]
    result = replace(result, type=replace(result.type, dtype="f32", numerical_policy="raw-bit-pattern"),
                     op="source_f32_reduce_add_unqualified", attrs=(
                         ("axes", ",".join(map(str, axes))), ("reduction_order", "unresolved"),
                         ("numeric_semantics", "source-f32-add-unqualified-v1"),
                         ("scalar_add_order", "input-accumulator")))
    return replace(request, nodes=(*nodes, result))


def evaluate(request, data, seed):
    return evaluate_seeded_bf16_reduction(request.nodes[-1], (
        TensorValue(request.nodes[0].type, tuple(data)), TensorValue(request.nodes[1].type, tuple(seed)))).elements


class NonnegativeReductionTest(unittest.TestCase):
    def test_explicit_source_policy_preserves_other_domains_and_native_body(self):
        source = source_request()
        old = prepare_source(source)
        new = prepare_source(source, reduction_policy=POSITIVE)
        self.assertEqual(old.record()["arithmetic"], REDUCTION_POLICY)
        self.assertEqual(new.record()["arithmetic"], POSITIVE)
        self.assertNotEqual(new.request.digest(), old.request.digest())
        old_body, new_body = body_request(prepare_plan(old.request)), body_request(prepare_plan(new.request))
        self.assertEqual(old_body.nodes, new_body.nodes)
        self.assertEqual(old_body.outputs, new_body.outputs)
        with self.assertRaises(ValueError):
            prepare_source(source, reduction_policy="unknown")
        witness = np.array([0x1ff5 << 16], dtype="<u4").view("<f4")
        self.assertEqual(input_spec(reduction_policy=POSITIVE).encode(witness).tolist(), [0x1ff5])
        with self.assertRaises(ValueError):
            input_spec().encode(witness)
        from atlas_mlir_native.pointwise import input_spec as pointwise_spec
        with self.assertRaises(ValueError):
            pointwise_spec().encode(witness)

    def test_application_selects_new_domain_explicitly(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from test_application_c_package import ApplicationLayoutPolicyTest

        policy = {**ApplicationLayoutPolicyTest().policy(),
                  "schema": "atlas.diagnostic_application_precision.v6", "reduction": POSITIVE}
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        self.assertEqual(adapter.reduction_policy, POSITIVE)
        self.assertEqual(len(adapter.plan(source_request()).selected), 1)
        old = AtlasApplicationAdapter({**policy, "reduction": REDUCTION_POLICY},
                                      original_source_sha256=policy["source_sha256"])
        self.assertEqual(old.reduction_policy, REDUCTION_POLICY)

    def test_boundary_entire_positive_normal_domain_and_invalid_f32(self):
        spec = input_spec(reduction_policy=POSITIVE)
        codes = np.array([0, *range(0x80, 0x7f80)], dtype="<u4")
        self.assertTrue(np.array_equal(spec.encode((codes << 16).view("<f4")), codes))
        for bits in (0x80000000, 0x00010000, 0x007f0000, 0xbf800000, 0x7f800000, 0x7fc00000,
                     0x7f7fffff, 0x00008001):
            with self.subTest(bits=hex(bits)), self.assertRaises(ValueError):
                spec.encode(np.array([bits], dtype="<u4").view("<f4"))
        # A positive f32 small enough to round to +0 is admitted as +0.
        self.assertEqual(spec.encode(np.array([1], dtype="<u4").view("<f4")).tolist(), [0])
        self.assertEqual(input_spec().encode(np.array([-0.0], dtype="<f4")).tolist(), [0])

    def test_reference_full_positive_normal_range_and_overflow(self):
        request = request_for((32513, 1), (1,), reduction_policy=POSITIVE)
        codes = (0, *range(0x80, 0x7f80))
        self.assertEqual(evaluate(request, codes, [0] * len(codes)), codes)
        request = request_for((1, 33), (1,), reduction_policy=POSITIVE)
        self.assertEqual(evaluate(request, [0x1ff5] + [0] * 32, [0]), (0x1ff5,))
        for data, seed in (([0x7f7f] * 33, [0]),
                           ([0x7f7f] + [0] * 32, [0x7f7f]),
                           ([0x7f7f, 0x7b00] + [0] * 31, [0]),
                           ([0x8000] + [0] * 32, [0]),
                           ([0] * 33, [0x8000])):
            with self.assertRaises(ValueError):
                evaluate(request, data, seed)

    def test_generated_guard_matches_reference_axes_tails_and_extremes(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            directory = Path(directory)
            for index, (shape, axes) in enumerate((((2, 3, 35), (0, 2)), ((1, 33), (1,)))):
                request = request_for(shape, axes, reduction_policy=POSITIVE)
                c = render_nonnegative_guard_c(prepare_plan(request), "probe")
                c += '\nint guard(const uint8_t *data,const uint8_t *seed) { return probe_domain(data,seed); }\n'
                path = directory / "guard.c"
                path.write_text(c)
                subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                                str(path), "-o", str(directory / f"guard{index}.so")], check=True, capture_output=True)
                library = ctypes.CDLL(str(directory / f"guard{index}.so"))
                library.guard.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
                rng = random.Random(901)
                count, rows = np.prod(shape), np.prod(request.nodes[1].type.shape)
                for case in range(120):
                    pool = [0, 0x80, 0xff, 0x100, 0x1ff5, 0x3f80, 0x3bc0, 0x7b00, 0x7f7e, 0x7f7f]
                    if 10 <= case < 70:
                        pool = pool[:8]
                    data = [rng.choice(pool) for _ in range(count)]
                    seed = [rng.choice(pool) for _ in range(rows)]
                    if case < 10:
                        data = [pool[case]] + [0] * (count - 1)
                        seed = [0] * rows
                    try:
                        evaluate(request, data, seed)
                        expected = 0
                    except ValueError:
                        expected = 1
                    actual = library.guard(struct.pack(f"<{count}H", *data), struct.pack(f"<{rows}H", *seed))
                    self.assertEqual(actual, expected, (shape, axes, case))
    def test_generated_encoder_matches_exhaustive_code_domain_and_rounding(self):
        codes = np.array([0, *range(0x80, 0x7f80)], dtype="<u4")
        values = (codes << 16).view("<f4")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            directory = Path(directory)
            text = render_nonnegative_encoder_c(len(codes), "encode_all")
            text += render_nonnegative_encoder_c(1, "encode_one")
            text += "int all(const float *p,uint8_t *q) { return encode_all(p,q); }\n"
            text += "int one(const float *p,uint8_t *q) { return encode_one(p,q); }\n"
            (directory / "encoder.c").write_text(text)
            subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            str(directory / "encoder.c"), "-o", str(directory / "encoder.so")],
                           check=True, capture_output=True)
            lib = ctypes.CDLL(str(directory / "encoder.so"))
            lib.all.argtypes = lib.one.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
            output = np.zeros(len(codes), dtype="<u2")
            self.assertEqual(lib.all(values.ctypes.data, output.ctypes.data), 0)
            self.assertTrue(np.array_equal(codes, output))
            bits = [0, 1, 0x8000, 0x8001, 0x007f8000, 0x007f7fff, 0x3f808000,
                    0x3f818000, 0x7f7f7fff, 0x7f7f8000, 0x7f7fffff, 0x80000000, 0x7f800000, 0x7fc00000]
            for raw in bits:
                source = np.array([raw], dtype="<u4").view("<f4")
                output = np.array([0xa5a5], dtype="<u2")
                status = lib.one(source.ctypes.data, output.ctypes.data)
                try:
                    expected = input_spec(reduction_policy=POSITIVE).encode(source)
                except ValueError:
                    self.assertNotEqual(status, 0, hex(raw))
                    self.assertEqual(int(output[0]), 0xa5a5)
                else:
                    self.assertEqual(status, 0, hex(raw))
                    self.assertEqual(int(output[0]), int(expected[0]))


if __name__ == "__main__":
    unittest.main()
