"""Connected device buffers, pack loops and initialized temporary reads."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from test_looped_mnk import source_request
from test_virtual_ssa import run


def pack_request(shape=(32, 288), code=127):
    from atlas_mlir_native.binding import (
        ROUNDED_PACK_DOMAIN,
        ROUNDED_PACK_POLICY,
        TARGET_IDENTITY,
    )
    from merlin.semantic_compiler.model import (
        KernelRequest,
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    source = SemanticNode(
        "x",
        "input",
        (),
        TensorType(shape, "bf16", "raw-bit-pattern"),
        (("input_domain", ROUNDED_PACK_DOMAIN),),
        "input",
    )
    output = SemanticNode(
        "y",
        "pack_fp8",
        ("x",),
        TensorType(shape, "fp8_e4m3", ROUNDED_PACK_POLICY),
        (("input_domain", ROUNDED_PACK_DOMAIN), ("scale_code", code)),
        index_maps=pointwise_identity_maps(2, 1),
    )
    return KernelRequest(
        (source, output),
        ("y",),
        ("external",),
        (("x", "external"),),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )


def connected_request(hidden):
    from atlas_mlir_native.binding import BF16_RELU_ATTRS
    from merlin.semantic_compiler.model import (
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    first = source_request(32, hidden, 32)
    last = source_request(32, 32, hidden)
    a, w1, dot = first.nodes
    w2 = replace(last.nodes[1], id="w2")
    activation = SemanticNode(
        "hidden",
        "relu",
        (dot.id,),
        TensorType((32, hidden), "bf16", "raw-bit-pattern"),
        BF16_RELU_ATTRS,
        index_maps=pointwise_identity_maps(2, 1),
    )
    pack = replace(
        pack_request((32, hidden)).nodes[1], id="packed", inputs=(activation.id,)
    )
    result = replace(last.nodes[-1], id="final", inputs=(pack.id, w2.id))
    return replace(
        first,
        nodes=(a, w1, w2, dot, activation, pack, result),
        input_storages=((a.id, "external"), (w1.id, "external"), (w2.id, "external")),
        outputs=(activation.id, result.id),
        output_storages=("external", "external"),
    )


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "select Merlin and Atlas tools",
)
class ResidentGraphTest(unittest.TestCase):
    def test_cache_exhaustion_retries_entire_resident_graph_without_stale_regions(self):
        from unittest.mock import patch
        from atlas_mlir_native import resident_graph
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import open_native_snapshot
        from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication
        from test_looped_mnk import build_snapshot
        from test_virtual_ssa import ROOT

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = (open_native_snapshot(Path(os.environ["MERLIN_NATIVE_SNAPSHOT"]))
                        if os.environ.get("MERLIN_NATIVE_SNAPSHOT") else build_snapshot(root))
            request = connected_request(64)
            abi = AtlasNativeBinding.plan_boundary_abi(request)
            target = root / "resident-retry"
            attempts = []
            run = resident_graph._run
            def limited(command, *, input_text):
                if "--atlas-compile-unscheduled" in command or "--atlas-compile" in command:
                    attempts.append('"atlas.virtual_slot"' in input_text)
                    if len(attempts) == 1:
                        self.assertTrue((target / "regions/0/manifest.json").exists())
                        (target / "stale").write_text("discard with published child regions")
                        raise NativeCompilationError(
                            "compile_error", "generated program exceeds the 32768-word policy limit after branch relaxation"
                        )
                    self.assertFalse((target / "stale").exists())
                return run(command, input_text=input_text)
            with codegen_options(CodegenOptions(reduction_unroll="4")):
                with patch.object(resident_graph, "_run", side_effect=limited):
                    manifest = resident_graph.compile_resident(
                        snapshot, request, fixed_inputs=abi["fixed_inputs"], fixed_outputs=tuple(abi["fixed_outputs"]),
                        target_source=ROOT, destination=target, limits=SearchLimits(wall_timeout_s=180),
                    )
            verify_native_publication(
                target, manifest, engine="merlin_native", request_digest=request.digest(),
                target_identity=request.target_identity,
            )
            self.assertEqual(attempts, [True, False])
            execution = json.loads((target / "execution_plan.json").read_text())
            self.assertFalse(execution["codegen"]["operand_reuse_enabled"])
            self.assertTrue(all(region.get("operand_reuse") is None for region in execution["regions"]))

    def test_large_scale_constant_stays_in_device_memory(self):
        from atlas_mlir_native.binding import BF16_BINARY_ATTRS
        from atlas_mlir_native.layout import pack_bf16_matrix
        from atlas_mlir_native.pointwise import prepare_plan
        from atlas_mlir_native.resident_graph import boundary_request, memory_plan
        from merlin.semantic_compiler.model import (
            ConstantBinding,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )

        request = connected_request(64)
        a, w1, w2, dot, activation, packed, result = request.nodes
        attrs = BF16_BINARY_ATTRS["mul_bf16_zero"]
        constant = SemanticNode(
            "scale",
            "constant",
            (),
            TensorType((32, 64), "bf16", "raw-bit-pattern"),
            attrs=(attrs[0],),
            effect="constant",
        )
        multiply = SemanticNode(
            "scaled",
            "mul_bf16_zero",
            (dot.id, constant.id),
            constant.type,
            attrs,
            index_maps=pointwise_identity_maps(2, 2),
        )
        request = replace(
            request,
            nodes=(
                a,
                w1,
                w2,
                constant,
                dot,
                multiply,
                replace(activation, inputs=(multiply.id,)),
                packed,
                result,
            ),
            constants=(ConstantBinding(constant.id, "bf16-le-raw", "003f" * 2048),),
        )
        cuts, _, sizes = memory_plan(request)
        pointwise = boundary_request(cuts[1])
        self.assertEqual(pointwise.constants, ())
        self.assertEqual(prepare_plan(pointwise).packed_bytes, 4096)
        self.assertEqual(sizes[constant.id], 4096)
        # Distinct neighboring columns expose tile and half-register boundaries.
        logical = b"".join(i.to_bytes(2, "little") for i in range(2048))
        physical = pack_bf16_matrix(logical, (32, 64))
        for r, c in ((0, 0), (3, 16), (31, 31), (0, 32), (31, 63)):
            at = (c // 32) * 2048 + ((c % 32) // 16) * 1024 + (r * 16 + c % 16) * 2
            self.assertEqual(
                int.from_bytes(physical[at : at + 2], "little"), r * 64 + c
            )

    def test_plan_accounts_for_shared_output_and_unequal_element_widths(self):
        from atlas_mlir_native.resident_graph import memory_plan, plan_abi

        for width in (64, 288):
            request = connected_request(width)
            cuts, slots, sizes = memory_plan(request)
            self.assertEqual(
                [tuple(n.op for n in r.nodes if n.effect == "pure") for r in cuts],
                [("matmul", "relu"), ("pack_fp8",), ("matmul",)],
            )
            self.assertEqual(sizes["hidden"], 32 * width * 2)
            self.assertEqual(sizes["packed"], 32 * width)
            self.assertEqual(
                plan_abi(request)["fixed_outputs"], [slots["hidden"], slots["final"]]
            )
            spans = sorted(
                (slot * 2048, slot * 2048 + sizes[key]) for key, slot in slots.items()
            )
            self.assertTrue(all(a[1] <= b[0] for a, b in pairwise(spans)))

    def test_internal_tails_and_excess_memory_are_explicit_refusals(self):
        from atlas_mlir_native.resident_graph import memory_plan
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        for width, status in ((65, "unsupported_semantics"), (2048, "resource_limit")):
            with self.assertRaises(NativeCompilationError) as caught:
                memory_plan(connected_request(width))
            self.assertEqual(caught.exception.status, status)

    def test_pack_layout_size_and_signature_mutations(self):
        from atlas_mlir_native.pack_tiling import prepare_plan, virtual_program
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        request = pack_request()
        plan = prepare_plan(request)
        self.assertEqual(plan.buffer_bytes, (18432, 9216))
        self.assertEqual(plan.tiles, 9)
        self.assertEqual(plan.body_request().nodes[0].type.shape, (32, 32))
        for code in (122, 127, 128):
            current = prepare_plan(pack_request(code=code))
            self.assertEqual(
                run(
                    "atlas-opt",
                    virtual_program(current, 0x90000000, 0x90010000),
                    "--verify-atlas-virtual-stream",
                ).returncode,
                0,
            )
        with self.assertRaises(NativeCompilationError):
            prepare_plan(pack_request(code=129))
        wrong = replace(
            request.nodes[-1],
            type=replace(request.nodes[-1].type, numerical_policy="raw-bit-pattern"),
        )
        with self.assertRaises(NativeCompilationError):
            prepare_plan(replace(request, nodes=(request.nodes[0], wrong)))
        with self.assertRaises(NativeCompilationError):
            virtual_program(plan, 0x90000000, 0x90000800)

    def test_pack_body_uses_the_native_selector(self):
        from atlas_mlir_native.pack_tiling import prepare_plan, select_body
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import open_native_snapshot
        from test_looped_mnk import build_snapshot

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            snapshot = (
                open_native_snapshot(Path(os.environ["MERLIN_NATIVE_SNAPSHOT"]))
                if os.environ.get("MERLIN_NATIVE_SNAPSHOT")
                else build_snapshot(Path(directory))
            )
            evidence = select_body(
                snapshot, prepare_plan(pack_request()), limits=SearchLimits()
            )
            self.assertEqual(evidence["engine"], "merlin_native")
            self.assertIn("atlas.vpu_pack_rounded_fp8_tile", evidence["instructions"])
            self.assertTrue(evidence["check_fingerprint"])
