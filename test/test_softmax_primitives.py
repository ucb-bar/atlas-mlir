"""Public native/RTL fixtures for the primitives needed by tiled softmax."""

import argparse
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

from test_looped_mnk import ROOT, build_snapshot


def request_for(operation):
    from atlas_mlir_native.binding import (
        BF16_BINARY_ATTRS,
        BF16_UNARY_ATTRS,
        TARGET_IDENTITY,
    )

    from merlin.semantic_compiler.model import (
        KernelRequest,
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    attributes = (BF16_BINARY_ATTRS | BF16_UNARY_ATTRS)[operation]
    names = ("x", "y") if operation in BF16_BINARY_ATTRS else ("x",)
    tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
    return KernelRequest(
        (
            *[SemanticNode(name, "input", (), tensor, (attributes[0],), "input") for name in names],
            SemanticNode(
                "result",
                operation,
                names,
                tensor,
                attributes,
                index_maps=pointwise_identity_maps(2, len(names)),
            ),
        ),
        ("result",),
        ("external",),
        tuple((name, "external") for name in names),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )


def _float(code):
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def _rounded_sum(left, right, subtract=False):
    # Two BF16 operands rounded to binary32. Cancellation and the binary32
    # halfway cases are exactly represented by the binary64 computation here.
    value = _float(left) + (-_float(right) if subtract else _float(right))
    try:
        result = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    except OverflowError:
        return None
    return result if not result & 0x7FFF or 0 < result >> 7 & 255 < 255 else None


def _reciprocal(code):
    # Independent Decimal transcription of the selected RcpLUT declaration,
    # including Math.round before BF16 truncation. Not real-reciprocal RNE.
    with localcontext() as context:
        context.prec = 80
        value = Decimal(128 * 65536) / Decimal(128 + (code & 127))
        fixed = int(value.to_integral_value(rounding=ROUND_HALF_UP))
    high = fixed.bit_length() - 1
    return ((254 + high - 16 - (code >> 7)) << 7) | ((fixed >> (high - 7)) & 127)


def panels():
    """Every wide reciprocal input, directed max cases and seeded add/sub pairs."""
    values = list(range(0x3F80, 0x4781))
    for start in range(0, len(values), 1024):
        row = values[start : start + 1024]
        row += [0x3F80] * (1024 - len(row))
        yield (
            f"reciprocal_{start // 1024}",
            "recip_bf16_wide",
            (row,),
            [_reciprocal(code) for code in row],
        )
    randomizer = random.Random(7319)
    for operation in ("max_bf16", "add_bf16_rne", "sub_bf16_rne"):
        left, right, expected = [], [], []
        directed = [
            (0, 0x8000),
            (0x8000, 0),
            (0x3F80, 0x3380),
            (0x3F80, 0xB380),
            (0x3F81, 0x3380),
            (0x3F81, 0xB380),
            (0x3F80, 0xBF80),
            (0x0080, 0x0080),
        ]
        if operation == "max_bf16":
            directed += [(0x7FC1, 0x3F80), (0xFFC1, 0x3F80), (0x7F80, 0x7FC0)]
        while len(expected) < 1024:
            if directed:
                a, b = directed.pop(0)
            elif operation == "max_bf16":
                a, b = randomizer.randrange(65536), randomizer.randrange(65536)
            else:
                exponent = randomizer.randrange(1, 253)
                gap = randomizer.choice((0, 1, 7, 8, 15, 16, 23, 24, 25, 31, 63))
                a = randomizer.randrange(2) * 32768 + exponent * 128 + randomizer.randrange(128)
                b = randomizer.randrange(2) * 32768 + max(1, exponent - gap) * 128 + randomizer.randrange(128)
            if operation == "max_bf16":
                key = lambda code: (~code & 65535) if code & 32768 else code ^ 32768
                result = a if key(a) > key(b) else b
            else:
                result = _rounded_sum(a, b, operation.startswith("sub_"))
                if result is None:
                    continue
            left.append(a)
            right.append(b)
            expected.append(result)
        yield operation, operation, (left, right), expected


def build_cases(root):
    from atlas_mlir_native.binding import AtlasNativeBinding
    from atlas_mlir_native.layout import pack_bf16_pair_halves

    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.target_binding import verify_native_publication

    snapshot = build_snapshot(root)
    compiled = {}
    rows = []
    for name, operation, inputs, expected in panels():
        request = request_for(operation)
        if operation not in compiled:
            destination = root / operation
            abi = AtlasNativeBinding.plan_boundary_abi(request)
            manifest = AtlasNativeBinding.compile(
                snapshot,
                request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=ROOT,
                destination=destination,
                limits=SearchLimits(),
            )
            verify_native_publication(
                destination,
                manifest,
                engine="merlin_native",
                request_digest=request.digest(),
                target_identity=request.target_identity,
            )
            compiled[operation] = destination
        directory = root / ("fixture_" + name)
        directory.mkdir()
        destination = compiled[operation]
        execution = json.loads((destination / "execution_plan.json").read_text())
        binary = (destination / "program.bin").read_bytes()
        (directory / "program.bin").write_bytes(binary)
        properties = [
            "schema=atlas.rtl_tile_fixture.v1",
            "name=softmax_" + name,
            f"input_count={len(inputs)}",
            f"program_sha256={hashlib.sha256(binary).hexdigest()}",
            f"output.address={execution['output_addresses'][0]:#x}",
        ]
        for index, codes in enumerate(inputs):
            payload = pack_bf16_pair_halves(struct.pack("<1024H", *codes))
            (directory / f"input-{index}.bin").write_bytes(payload)
            properties.append(f"input.{index}.address={execution['input_addresses'][index]:#x}")
        (directory / "expected.bin").write_bytes(pack_bf16_pair_halves(struct.pack("<1024H", *expected)))
        (directory / "manifest.properties").write_text("\n".join(properties) + "\n")
        rows.append(
            {
                "name": name,
                "operation": operation,
                "directory": str(directory),
                "words": len(binary) // 4,
                "input_domain": execution["input_domain"],
            }
        )
    (root / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "select Merlin and Atlas tools",
)
class SoftmaxPrimitivesTest(unittest.TestCase):
    def test_native_compilation_and_independent_lane_references(self):
        from atlas_mlir_native.reference import (
            evaluate_bf16_binary_rne,
            evaluate_bf16_max,
            evaluate_bf16_recip,
        )

        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

        callbacks = {
            ("max_bf16", "raw-bit-pattern"): evaluate_bf16_max,
            ("recip_bf16_wide", "raw-bit-pattern"): evaluate_bf16_recip,
            **{(op, "raw-bit-pattern"): evaluate_bf16_binary_rne for op in ("add_bf16_rne", "sub_bf16_rne")},
        }
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            build_cases(Path(directory))
        for name, operation, inputs, expected in panels():
            with self.subTest(name=name):
                request = request_for(operation)
                (result,) = evaluate_graph(
                    request,
                    {node.id: TensorValue(node.type, tuple(codes)) for node, codes in zip(request.nodes, inputs)},
                    primitives=callbacks,
                )
                self.assertEqual(result.elements, tuple(expected))

    def test_old_reciprocal_range_is_preserved(self):
        from atlas_mlir_native.recip_lane import vrecip_bf16_bits

        with self.assertRaises(ValueError):
            vrecip_bf16_bits(0x4280)
        self.assertEqual(vrecip_bf16_bits(0x4280, wide=True), 0x3C80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fixtures":
        parser = argparse.ArgumentParser()
        parser.add_argument("--fixtures", type=Path, required=True)
        args = parser.parse_args()
        args.fixtures.mkdir(parents=True, exist_ok=False)
        print(json.dumps(build_cases(args.fixtures)))
    else:
        unittest.main()
