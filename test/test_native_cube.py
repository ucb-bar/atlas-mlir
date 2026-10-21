"""VCUBE arithmetic, table provenance and native instruction selection."""

import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from atlas_mlir_native.cube_lane import vcube_bf16_bits


def source_request():
    from atlas_mlir_native.binding import BF16_CUBE_ATTRS
    from test_vpu_tanh_reference import source_request as tanh_request

    source = tanh_request()
    arg, output = source.nodes
    return dataclasses.replace(source, nodes=(
        dataclasses.replace(arg, attrs=(BF16_CUBE_ATTRS[0],)),
        dataclasses.replace(output, op="cube_bf16", attrs=BF16_CUBE_ATTRS),
    ))


class CubeReferenceTest(unittest.TestCase):
    def test_entire_finite_domain_against_rational_chopping(self):
        for exponent in range(255):
            for fraction in range(128):
                value = Fraction(128 + fraction) * Fraction(2) ** (exponent - 134)
                cube = value ** 3
                power = cube.numerator.bit_length() - cube.denominator.bit_length()
                if cube < Fraction(2) ** power:
                    power -= 1
                expected = (
                    0 if not exponent or power < -126 else 0x7F80 if power > 127
                    else ((power + 127) << 7) | (int(cube / Fraction(2) ** (power - 7)) - 128)
                )
                for sign in (0, 0x8000):
                    self.assertEqual(vcube_bf16_bits(sign | exponent << 7 | fraction), sign | expected)
        # Chopping the cube differs from a sequence of rounded products.
        self.assertEqual(vcube_bf16_bits(0x3F81), 0x3F83)
        for code in (-1, 65536, 0x7F80, 0xFF80, 0x7FC1, True):
            with self.subTest(code=code), self.assertRaises(ValueError):
                vcube_bf16_bits(code)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the pinned model table")
    def test_all_finite_codes_match_the_pinned_model_table(self):
        from npu_model.configs import rtl_math

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json"))
        self.assertIn("337e9f249f938a1dff007ddf8a763e5cd4b3aa2b", origin["url"])
        data = gzip.decompress((Path(rtl_math.__file__).with_name("data") / "cube.bin.gz").read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "8c1b2ee7b0ff3607f0f8beeea65a102f8dc7c58982fab8090ff291016833625d")
        table = struct.unpack("<65536H", data)
        for code in range(65536):
            if code >> 7 & 255 != 255:
                self.assertEqual(vcube_bf16_bits(code), table[code])

    @unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                         "select Merlin and Atlas tools")
    def test_native_rule_emits_cube_and_refuses_a_changed_policy(self):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from test_looped_mnk import ROOT, build_snapshot

        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            source = source_request()
            compiled = root / "compiled"
            manifest = AtlasNativeBinding.compile(
                snapshot, source, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=compiled, limits=SearchLimits(),
            )
            verify_native_publication(compiled, manifest, engine="merlin_native",
                                      request_digest=source.digest(), target_identity=source.target_identity)
            program = json.loads((compiled / "physical-program.json").read_text())
            self.assertEqual([row["fields"]["kind"] for row in program["instructions"]
                              if row["operation"] == "atlas.vpu_unary"], ["cube"])
            changed = dataclasses.replace(source.nodes[-1], attrs=())
            with self.assertRaises(NativeCompilationError):
                AtlasNativeBinding.compile(
                    snapshot, dataclasses.replace(source, nodes=(source.nodes[0], changed)),
                    fixed_inputs={"x": 3}, fixed_outputs=(35,), target_source=ROOT,
                    destination=root / "invalid", limits=SearchLimits(),
                )
