"""Finite VTANH arithmetic and generated native-rule checks."""

import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
import struct
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

from atlas_mlir_native.tanh_lane import TANH_LUT_Q1_16, vtanh_bf16_bits


def source_request():
    from atlas_mlir_native.binding import BF16_TANH_ATTRS, TARGET_IDENTITY

    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

    tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
    return KernelRequest(
        (SemanticNode("x", "input", (), tensor, (BF16_TANH_ATTRS[0],), "input"),
         SemanticNode("y", "tanh_bf16", ("x",), tensor, BF16_TANH_ATTRS,
                      index_maps=pointwise_identity_maps(2, 1))),
        ("y",), ("external",), (("x", "external"),), TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )


def panel(phase):
    boundary = (0, 1, 127, 128, 0x377F, 0x3780, 0x3781, 0x3C3F, 0x3F00,
                0x3F80, 0x407F, 0x4080, 0x7F7F)
    values = tuple(sign | code for sign in (0, 0x8000) for code in boundary)
    return tuple(values[(i * 7 + phase * 3) % len(values)] for i in range(1024))


class TanhReferenceTest(unittest.TestCase):
    def test_lut_is_high_precision_positive_tanh_rounded_to_q16(self):
        with localcontext() as context:
            context.prec = 80
            expected = []
            for index in range(32):
                exponential = (Decimal(index) / 4).exp()
                value = (exponential - 1) / (exponential + 1)
                expected.append(int((value * 65536).quantize(Decimal(1), rounding=ROUND_HALF_UP)))
        self.assertEqual(tuple(expected), TANH_LUT_Q1_16)

    def test_zero_fixed_point_threshold_saturation_and_rejections(self):
        self.assertEqual(vtanh_bf16_bits(0), 0)
        self.assertEqual(vtanh_bf16_bits(0x8000), 0x8000)
        self.assertEqual(vtanh_bf16_bits(0x377F), 0)
        self.assertEqual(vtanh_bf16_bits(0x3780), 0)  # First Q16 bin still interpolates to zero.
        for code in (1, 127, 128, 0x3C3F, 0x3F80, 0x4080, 0x7F7F):
            self.assertEqual(vtanh_bf16_bits(code | 0x8000), vtanh_bf16_bits(code) | 0x8000)
        self.assertEqual(vtanh_bf16_bits(0x4080), 0x3F80)
        for code in (-1, 65536, 0x7F80, 0x7FC1, 0xFFFF, True):
            with self.subTest(code=code), self.assertRaises(ValueError):
                vtanh_bf16_bits(code)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the pinned model table")
    def test_all_finite_codes_match_the_selected_model_rtl_table(self):
        from npu_model.configs import rtl_math

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json"))
        self.assertIn("337e9f249f938a1dff007ddf8a763e5cd4b3aa2b", origin["url"])
        data = gzip.decompress((Path(rtl_math.__file__).with_name("data") / "tanh.bin.gz").read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "b3aa936ac8b5591f3cf38ded8b7d383b9c6729a3ab304d0b53f029af55accbc6")
        expected = struct.unpack("<65536H", data)
        codes = tuple(code for code in range(65536) if (code >> 7) & 255 != 255)
        self.assertEqual(len(codes), 65280)
        self.assertEqual(tuple(vtanh_bf16_bits(code) for code in codes), tuple(expected[code] for code in codes))


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                     "select Merlin and Atlas tools")
class TanhNativeTest(unittest.TestCase):
    def test_native_selection_emits_tanh_and_preserves_its_numerical_policy(self):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from test_looped_mnk import ROOT, build_snapshot

        from merlin.semantic_compiler.model import ConstantBinding
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
                              if row["operation"] == "atlas.vpu_unary"], ["tanh"])
            constant = dataclasses.replace(source.nodes[0], id="c", op="constant", effect="constant")
            result = dataclasses.replace(source.nodes[-1], id="z", inputs=("c",))
            payload = struct.pack("<1024H", *panel(0))
            with_constant = dataclasses.replace(
                source, nodes=(source.nodes[0], constant, source.nodes[-1], result), outputs=("y", "z"),
                output_storages=("external", "external"),
                constants=(ConstantBinding("c", "bf16-le-raw", payload.hex()),),
            )
            initialized = AtlasNativeBinding.compile(
                snapshot, with_constant, fixed_inputs={"x": 3, "c": 4}, fixed_outputs=(35, 36),
                target_source=ROOT, destination=root / "constant", limits=SearchLimits(),
            )
            self.assertEqual(initialized["constant_inputs"][0]["node_id"], "c")
            invalid = dataclasses.replace(with_constant, constants=(ConstantBinding(
                "c", "bf16-le-raw", (bytes.fromhex("807f") + payload[2:]).hex(),
            ),))
            with self.assertRaisesRegex(NativeCompilationError, "input domain"):
                AtlasNativeBinding.compile(
                    snapshot, invalid, fixed_inputs={"x": 3, "c": 4}, fixed_outputs=(35, 36),
                    target_source=ROOT, destination=root / "nonfinite", limits=SearchLimits(),
                )
            changed = dataclasses.replace(source.nodes[-1], attrs=())
            with self.assertRaises(NativeCompilationError):
                AtlasNativeBinding.compile(
                    snapshot, dataclasses.replace(source, nodes=(*source.nodes[:-1], changed)),
                    fixed_inputs={"x": 3}, fixed_outputs=(35,), target_source=ROOT,
                    destination=root / "invalid", limits=SearchLimits(),
                )


if __name__ == "__main__":
    unittest.main()
