"""Native tile selection inside compact whole-contraction M/N/K loops."""

import dataclasses
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_virtual_ssa import ROOT

SHAPES = ((64, 64, 64), (64, 96, 32), (31, 33, 32), (33, 65, 31), (1, 7, 65), (7, 33, 1), (65, 7, 65))


def source_request(m, n, k, *, relu=False):
    from atlas_mlir_native.binding import (
        BF16_RELU_ATTRS,
        MXU0_INPUT_DOMAIN,
        MXU0_POLICY,
        TARGET_IDENTITY,
        _mxu0_maps,
    )

    from merlin.semantic_compiler.model import (
        KernelRequest,
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    request = KernelRequest(
        (
            SemanticNode(
                "a",
                "input",
                (),
                TensorType((m, k), "fp8_e4m3", "raw-bit-pattern"),
                (("input_domain", MXU0_INPUT_DOMAIN),),
                "input",
            ),
            SemanticNode(
                "w",
                "input",
                (),
                TensorType((n, k), "fp8_e4m3", "raw-bit-pattern"),
                (("input_domain", MXU0_INPUT_DOMAIN),),
                "input",
            ),
            SemanticNode("y", "matmul", ("a", "w"), TensorType((m, n), "bf16", MXU0_POLICY), index_maps=_mxu0_maps()),
        ),
        ("y",),
        ("external",),
        (("a", "external"), ("w", "external")),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )
    if relu:
        result = request.nodes[-1]
        activation = SemanticNode(
            "activated",
            "relu",
            (result.id,),
            TensorType(result.type.shape, "bf16", "raw-bit-pattern"),
            BF16_RELU_ATTRS,
            index_maps=pointwise_identity_maps(2, 1),
        )
        request = dataclasses.replace(request, nodes=(*request.nodes, activation), outputs=(activation.id,))
    return request


def case_data(shape):
    """Independent integer-valued arithmetic and physical output packing.

    K is at most 65 in this public set: all partial integer sums are exactly
    BF16-representable. Dense nonzero terms span K tiles and the head pad.
    """
    from atlas_mlir_native.looped_tiling import prepare_plan

    from merlin.semantic_compiler.ordered_mnk_tiling import (
        materialize_looped_mnk_inputs,
    )

    m, n, k = shape
    request = source_request(m, n, k)
    plan = prepare_plan(request)
    codes = (0x38, 0x40, 0x44)
    a = bytes(codes[(r + i) % 3] for r in range(m) for i in range(k))
    w = bytes((0xB8 if c % 2 else 0x38) if i % 11 == c % 11 else 0 for c in range(n) for i in range(k))
    expected = bytearray()
    for mr in range(0, m, 32):
        for nc in range(0, n, 32):
            tile = bytearray(2048)
            for r in range(32):
                for c in range(32):
                    value = (
                        (
                            sum(1 + (mr + r + i) % 3 for i in range(k) if i % 11 == (nc + c) % 11)
                            * (-1 if (nc + c) % 2 else 1)
                        )
                        if mr + r < m and nc + c < n
                        else 0
                    )
                    bits = struct.unpack("<I", struct.pack("<f", value))[0]
                    assert bits & 0xFFFF == 0
                    offset = (c // 16) * 1024 + (r * 16 + c % 16) * 2
                    struct.pack_into("<H", tile, offset, bits >> 16)
            expected.extend(tile)
    return request, plan, materialize_looped_mnk_inputs(plan, {"a": a, "w": w}), bytes(expected)


def seeded_source_request(shape, relu=False):
    from atlas_mlir_native.binding import (
        BF16_RELU_ATTRS,
        MXU0_SEED_ATTRS,
        _mxu0_accumulate_maps,
    )

    from merlin.semantic_compiler.model import (
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    base = source_request(*shape)
    a, w, result = base.nodes
    seed = SemanticNode("seed", "input", (), result.type, MXU0_SEED_ATTRS, "input")
    result = dataclasses.replace(
        result, op="matmul_accumulate", inputs=(a.id, w.id, seed.id), index_maps=_mxu0_accumulate_maps()
    )
    request = dataclasses.replace(
        base, nodes=(a, w, seed, result), input_storages=(*base.input_storages, (seed.id, "external"))
    )
    if relu:
        activation = SemanticNode(
            "activated",
            "relu",
            (result.id,),
            TensorType(result.type.shape, "bf16", "raw-bit-pattern"),
            BF16_RELU_ATTRS,
            index_maps=pointwise_identity_maps(2, 1),
        )
        request = dataclasses.replace(request, nodes=(*request.nodes, activation), outputs=(activation.id,))
    return request


def seeded_case_data(shape, shift=0, relu=False):
    """Dense source bytes and an independent per-step BF16 rounding reference."""
    from rtl.make_ordered_tile_fixture import _bf16, _fp8, _round_bf16

    m, n, k = shape
    a = bytes(0x18 if (r + shift) % 2 == 0 else 0x98 for r in range(m) for _ in range(k))
    w = bytes(0x18 if c % 3 else 0x98 for c in range(n) for _ in range(k))
    codes = (0x3F81, 0xBF81, 0, 0x4000, 0xC000)
    seed, expected = bytearray(), bytearray()
    for r in range(m):
        for c in range(n):
            code = codes[(r * 7 + c * 3 + shift) % len(codes)]
            seed.extend(code.to_bytes(2, "little"))
            for i in range(k):
                code = _round_bf16(_bf16(code) + _fp8(a[r * k + i]) * _fp8(w[c * k + i]))
            if relu and code & 0x8000:
                code = 0
            expected.extend(code.to_bytes(2, "little"))
    return a, w, bytes(seed), bytes(expected)


def chunked_case_data(shape, shift=0, relu=False):
    """A negative, non-FP8 partial must survive until the final positive term."""
    from rtl.make_ordered_tile_fixture import _bf16, _fp8, _round_bf16

    m, n, k = shape
    a, w = bytearray(), bytearray()
    for row in range(m):
        scale = (row + shift) % 2
        values = bytearray([0x38 + scale * 8]) * k
        values[1] = 0x18 + scale * 8
        a.extend(values)
    for column in range(n):
        values = bytearray(k)
        values[0], values[1], values[-1] = (0xB8, 0xA0, 0x40) if column % 2 == 0 else (0x38, 0x20, 0xC0)
        w.extend(values)
    expected = bytearray()
    for row in range(m):
        for column in range(n):
            value = 0
            for i in range(k):
                value = _round_bf16(_bf16(value) + _fp8(a[row * k + i]) * _fp8(w[column * k + i]))
            if relu and value & 0x8000:
                value = 0
            expected.extend(value.to_bytes(2, "little"))
    return bytes(a), bytes(w), bytes(expected)


def rounding_case_data(panel):
    """Two discriminating ordered reductions spanning the K-loop backedges."""
    from atlas_mlir_native.looped_tiling import prepare_plan
    from rtl.make_ordered_tile_fixture import _bf16, _fp8, _round_bf16

    from merlin.semantic_compiler.ordered_mnk_tiling import (
        materialize_looped_mnk_inputs,
    )

    request = source_request(1, 7, 65)
    plan = prepare_plan(request)
    a = bytearray([0x38] + [0x18] * 64)
    row = bytearray(a)
    if panel == "cancellation":
        a[32], row[32] = 0xB8, 0x38
    elif panel != "ties":
        raise ValueError("unknown public rounding panel")
    code = 0
    for left, right in zip(a, row, strict=True):
        code = _round_bf16(_bf16(code) + _fp8(left) * _fp8(right))
    # A deferred sum would retain the half-ULP terms; ordered BF16 loses them.
    assert code == (0x3E00 if panel == "cancellation" else 0x3F80)
    expected = bytearray(2048)
    for column in range(7):
        struct.pack_into("<H", expected, column * 2, code)
    return materialize_looped_mnk_inputs(plan, {"a": bytes(a), "w": bytes(row) * 7}), bytes(expected)


def build_snapshot(destination):
    from atlas_mlir_native.binding import AtlasNativeBinding

    from merlin.semantic_compiler.snapshot import (
        build_native_snapshot,
        packaged_egg_bridge_path,
    )

    return build_native_snapshot(
        AtlasNativeBinding.profile(),
        destination=destination / "snapshot",
        crate=packaged_egg_bridge_path(),
        cargo_target_dir=destination / "cargo",
        source_revision="looped-mnk-public-test",
    )


def compile_case(snapshot, shape, destination, *, source=None):
    from atlas_mlir_native.binding import AtlasNativeBinding
    from atlas_mlir_native.looped_tiling import prepare_plan

    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.target_binding import verify_native_publication

    request = source if source is not None else source_request(*shape)
    plan = prepare_plan(request)
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
    execution = json.loads((destination / "execution_plan.json").read_text())
    _, _, inputs, expected = case_data(shape)
    return request, plan, inputs, expected, execution


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR",
)
class LoopedMNKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        cls.snapshot = build_snapshot(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_tuning_identity_tracks_provider_and_shared_codegen_sources(self):
        from unittest.mock import patch

        from atlas_mlir_native import looped_tiling
        from merlin.semantic_compiler import codegen
        from merlin.semantic_compiler.search import SearchLimits

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            provider = root / "provider.py"
            helper = root / "helper.py"
            shared = root / "shared" / "codegen.py"
            shared.parent.mkdir()
            provider.write_text("provider")
            helper.write_text("initial helper")
            shared.write_text("shared options")
            request = source_request(2, 3, 257)
            def key():
                return looped_tiling.tuning_key(self.snapshot, request, {"a": 0, "w": 5}, (10,), SearchLimits())
            with patch.object(looped_tiling, "__file__", str(provider)), patch.object(codegen, "__file__", str(shared)):
                initial = key()
                helper.write_text("changed helper")
                provider_changed = key()
                self.assertNotEqual(initial["provider_sources"], provider_changed["provider_sources"])
                self.assertEqual(initial["codegen_options_sha256"], provider_changed["codegen_options_sha256"])
                shared.write_text("changed shared options")
                self.assertNotEqual(provider_changed["codegen_options_sha256"], key()["codegen_options_sha256"])

    def test_diagnostic_k_unroll_keeps_default_bytes_and_checks_cleanup_control_flow(self):
        from atlas_mlir_native.binding import NATIVE_CHECKS, NATIVE_PIPELINES, _run
        from atlas_mlir_native.looped_tiling import _virtual, prepare_plan

        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        for shape in ((3, 2, 257), (2, 3, 545)):
            request = source_request(*shape)
            plan = prepare_plan(request)
            for factor in (1, 2, 4, 8, 16):
                with self.subTest(shape=shape, factor=factor):
                    virtual = _virtual(plan, (0x90000000, 0x90010000), 0x90020000, k_unroll=factor)
                    machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES["mandatory"], "-"], input_text=virtual)
                    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
                    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
                    self.assertTrue(program["instructions"])
                    if factor == 1:
                        self.assertEqual(virtual, _virtual(plan, (0x90000000, 0x90010000), 0x90020000))
        for bad in (0, -1, 65, True, 2.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _virtual(plan, (0x90000000, 0x90010000), 0x90020000, k_unroll=bad)

    def test_operand_cache_scope_capacity_and_compact_fallback(self):
        from atlas_mlir_native.binding import NATIVE_CHECKS, NATIVE_PIPELINES, _run
        from atlas_mlir_native.looped_tiling import _virtual, prepare_plan

        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        addresses, output = (0x90000000, 0x90040000), 0x90080000
        for shape in ((65, 97, 65), (65, 7, 65), (1, 65, 1), (65, 1, 1), (1, 1, 65),
                      (32, 64, 3840), (32, 64, 3841)):
            plan = prepare_plan(source_request(*shape))
            mc, nc, kc = plan.counts
            cached = (mc > 1 or nc > 1) and kc <= 120
            for seed in (None, 0x90070000):
                with self.subTest(shape=shape, seed=seed):
                    virtual = _virtual(plan, addresses, output, seed_address=seed, k_unroll=4)
                    compact = _virtual(plan, addresses, output, seed_address=seed, k_unroll=4, operand_reuse=False)
                    self.assertEqual(virtual.count('"atlas.virtual_slot"'), kc if cached else 0)
                    self.assertEqual(virtual.count('"atlas.virtual_restore"'), kc if cached else 0)
                    if cached:
                        self.assertEqual(virtual.count('"atlas.virtual_dma_load_fp8"'), 2 * kc)
                        self.assertEqual(virtual.count('"atlas.virtual_mxu_readout_bf16"'), kc)
                        self.assertEqual(virtual.count('"atlas.virtual_mxu_load_acc_bf16"'), kc - (seed is None))
                        entry, row = virtual.split("^m_body(", 1)
                        # A is refreshed for each row; a single W column is
                        # initialized once before any output-row iteration.
                        self.assertEqual('"atlas.virtual_save"' in entry, nc == 1)
                        self.assertEqual('"atlas.virtual_save"' in row, nc > 1)
                    else:
                        self.assertEqual(virtual, compact)
                    for pipeline in ("mandatory", "optimized"):
                        machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
                        _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)

    def test_operand_cache_keeps_explicit_code_size_budgets_compact(self):
        from atlas_mlir_native.looped_c import prepare_c_package
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        with codegen_options(CodegenOptions(io_memory_bytes=1048576, reduction_unroll="4", program_size_limit_bytes=2048)):
            snapshot = build_snapshot(self.root / "compact-snapshot")
            target = self.root / "compact-cache-fallback"
            _, _, _, _, execution = compile_case(snapshot, (64, 128, 512), target)
            self.assertIsNone(execution["operand_reuse"])
            self.assertNotIn('"atlas.virtual_slot"', (target / "atlas-virtual.mlir").read_text())
            self.assertLessEqual((target / "program.bin").stat().st_size, 2048)
            prepare_c_package(target / "source-request.json", target, "layer", target_source=ROOT)

    def source_bound_contraction(self):
        import hashlib

        from atlas_mlir_native.binding import (
            AtlasNativeBinding,
            diagnostic_mxu0_f32_profile,
        )

        from merlin.frontends.compile_inputs import StagedCompileInputs

        source = (ROOT / "test/examples/source_bound_mn_k72_matmul.mlir").read_text()
        digest = hashlib.sha256(source.encode()).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": digest,
            "matmul_regions": {"large": dict(profile.selected_decision)},
        }
        raw = json.dumps(policy, sort_keys=True).encode()
        staged = StagedCompileInputs(source, "forward", policy, raw, digest, digest, hashlib.sha256(raw).hexdigest())
        plan = AtlasNativeBinding.prepare_precision_tile(staged, "large", whole_contraction=True)
        with self.assertRaisesRegex(RuntimeError, "tile or batch selection"):
            AtlasNativeBinding.prepare_precision_tile(staged, "large", whole_contraction=True, tile_index=0)
        return plan

    def test_seeded_accumulator_selection_and_disjoint_storage(self):
        from unittest.mock import patch

        from atlas_mlir_native.binding import AtlasNativeBinding

        from merlin.semantic_compiler.buffer_allocation import BufferAllocation
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        for shape, relu in (((32, 32, 32), False), ((33, 35, 65), False), ((1, 7, 65), True)):
            source = seeded_source_request(shape, relu)
            destination = self.root / ("seeded_" + "_".join(map(str, shape)))
            _, _, _, _, execution = compile_case(self.snapshot, shape, destination, source=source)
            body = json.loads((destination / "tile-body-selection.json").read_text())
            self.assertEqual(body["selection_scope"], "representative_seeded_body")
            self.assertIn("atlas.load_mxu0_acc_bf16_tile", body["instructions"])
            self.assertIn("atlas.mxu0_accumulate_tile", body["instructions"])
            self.assertFalse(any("reset" in name for name in body["instructions"]))
            self.assertEqual(execution["accumulator_seed"]["conversion"], "none")
            abi = AtlasNativeBinding.plan_boundary_abi(source)
            with self.assertRaisesRegex(NativeCompilationError, "overlap"):
                AtlasNativeBinding.compile(
                    self.snapshot,
                    source,
                    fixed_inputs=abi["fixed_inputs"],
                    fixed_outputs=(abi["fixed_inputs"]["seed"],),
                    target_source=ROOT,
                    destination=self.root / "invalid-alias",
                    limits=SearchLimits(),
                )
        source = seeded_source_request((33, 35, 65))
        a, w, seed, result = source.nodes
        for altered in (
            dataclasses.replace(seed, attrs=()),
            dataclasses.replace(seed, type=dataclasses.replace(seed.type, numerical_policy="raw-bit-pattern")),
        ):
            with self.assertRaisesRegex(NativeCompilationError, "completed BF16"):
                AtlasNativeBinding.plan_boundary_abi(dataclasses.replace(source, nodes=(a, w, altered, result)))
        with patch(
            "merlin.semantic_compiler.buffer_allocation.allocate_buffers",
            return_value=BufferAllocation("unknown", reason="test work limit"),
        ):
            with self.assertRaises(NativeCompilationError) as error:
                AtlasNativeBinding.plan_boundary_abi(source)
            self.assertEqual(error.exception.status, "search_timeout")

    def test_source_bound_contraction_preserves_windows_and_orients_weights(self):
        import numpy as np
        from atlas_mlir_native.looped_tiling import prepare_plan

        from merlin.semantic_compiler.ordered_mnk_tiling import (
            materialize_looped_mnk_inputs,
        )

        plan = self.source_bound_contraction()
        self.assertTrue(plan.whole_contraction)
        self.assertEqual([node.type.shape for node in plan.request.nodes], [(3, 72), (35, 72), (3, 35)])
        a = np.array([[1 + (r + i) % 3 for i in range(72)] for r in range(3)], dtype=np.float32)
        rhs = np.array(
            [[(-1 if c % 2 else 1) if i % 11 == c % 11 else 0 for c in range(35)] for i in range(72)], dtype=np.float32
        )
        payloads = {
            item.source_role: item.convert_values(value).tobytes()
            for item, value in zip(plan.inputs, (a, rhs), strict=True)
        }
        packed = materialize_looped_mnk_inputs(prepare_plan(plan.request), payloads)
        self.assertEqual(packed, case_data((3, 35, 72))[2])

    def test_looped_shapes_and_tails_use_checked_native_bodies(self):
        from atlas_mlir_native.layout import unpack_bf16_pair_halves

        from merlin.semantic_compiler.ordered_mnk_tiling import (
            assemble_looped_mnk_outputs,
        )

        for shape in SHAPES:
            with self.subTest(shape=shape):
                destination = self.root / ("shape_" + "_".join(map(str, shape)))
                _, plan, inputs, expected, execution = compile_case(self.snapshot, shape, destination)
                selected = json.loads((destination / "tile-body-selection.json").read_text())
                self.assertEqual(selected["engine"], "merlin_native")
                self.assertTrue(selected["check_fingerprint"])
                self.assertEqual([len(x) for x in inputs], list(plan.buffer_bytes[:2]))
                self.assertEqual(execution["output_bytes"], len(expected))
                logical = b"".join(
                    unpack_bf16_pair_halves(expected[i : i + 2048]) for i in range(0, len(expected), 2048)
                )
                self.assertEqual(len(assemble_looped_mnk_outputs(plan, logical)), shape[0] * shape[1] * 2)

    def test_installed_cli_compiles_and_verifies_the_whole_contraction(self):
        source = ROOT / "test/examples/native_mxu0_looped_request.json"
        destination = self.root / "cli-compiled"
        common = [
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            str(self.snapshot.root),
            "--request",
            str(source),
            "--mode",
            "diagnostic",
        ]
        compiled = subprocess.run(
            [
                sys.executable,
                "-m",
                "merlin.targetgen.cli",
                "native-compile",
                *common,
                "--abi",
                "auto",
                "--target-source",
                str(ROOT),
                "--out",
                str(destination),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
        verified = subprocess.run(
            [
                sys.executable,
                "-m",
                "merlin.targetgen.cli",
                "native-verify",
                *common,
                "--artifact",
                str(destination),
                "--out",
                str(self.root / "cli-verify.json"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "set ATLAS_RUN_NPU_MODEL=1")
    def test_full_output_buffers_inputs_and_guards_on_model(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import MODEL_REVISION, model_instruction
        from tools.check_program_decode import decode_binary

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        self.assertTrue(
            MODEL_REVISION in origin.get("url", "") or origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION
        )
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        torch.set_num_threads(1)
        cases = [(shape, None) for shape in SHAPES] + [((1, 7, 65), panel) for panel in ("ties", "cancellation")]
        cases.append(((3, 35, 72), "source"))
        cases.append(((1, 7, 65), "relu_after_reduction"))
        for shape, panel in cases:
            with self.subTest(shape=shape, panel=panel):
                destination = self.root / ("model_" + "_".join(map(str, shape)) + ("_" + panel if panel else ""))
                source = self.source_bound_contraction().request if panel == "source" else None
                if panel == "relu_after_reduction":
                    from atlas_mlir_native.binding import BF16_RELU_ATTRS

                    from merlin.semantic_compiler.model import (
                        SemanticNode,
                        TensorType,
                        pointwise_identity_maps,
                    )

                    base = source_request(*shape)
                    activation = SemanticNode(
                        "activated",
                        "relu",
                        ("y",),
                        TensorType((1, 7), "bf16", "raw-bit-pattern"),
                        BF16_RELU_ATTRS,
                        index_maps=pointwise_identity_maps(2, 1),
                    )
                    source = dataclasses.replace(base, nodes=(*base.nodes, activation), outputs=("activated",))
                _, _, inputs, expected, execution = compile_case(self.snapshot, shape, destination, source=source)
                if panel in ("ties", "cancellation"):
                    inputs, expected = rounding_case_data(panel)
                if panel == "relu_after_reduction":
                    from atlas_mlir_native.looped_tiling import prepare_plan

                    from merlin.semantic_compiler.ordered_mnk_tiling import (
                        materialize_looped_mnk_inputs,
                    )

                    # Partial sums are negative until the final K tile. Applying
                    # ReLU to each partial sum would produce 32 instead of 1.
                    activation = bytes([0x38]) * 65
                    weights = b"".join(
                        bytes([0xB8]) * 32 + bytes([0x38]) * 33
                        if column % 2 == 0
                        else bytes([0x38]) * 32 + bytes([0xB8]) * 33
                        for column in range(7)
                    )
                    inputs = materialize_looped_mnk_inputs(prepare_plan(source), {"a": activation, "w": weights})
                    expected = bytearray(2048)
                    for column in range(0, 7, 2):
                        struct.pack_into("<H", expected, column * 2, 0x3F80)
                    expected = bytes(expected)
                    self.assertEqual(execution["epilogue"]["op"], "relu")
                binary = (destination / "program.bin").read_bytes()
                decoded = decode_binary(inventory, binary)
                program = InstantiableProgram(
                    [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
                )
                program.dram_base = 0x90000000
                output = execution["output_addresses"][0]
                length = (
                    max(output + len(expected), *(a + len(x) for a, x in zip(execution["input_addresses"], inputs)))
                    - program.dram_base
                    + 1024
                )
                initial = bytearray((i * 19 + 31) % 256 for i in range(length))
                for address, payload in zip(execution["input_addresses"], inputs):
                    start = address - program.dram_base
                    initial[start : start + len(payload)] = payload
                program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                config = DefaultHardwareConfig()
                config.arch_state_config = dataclasses.replace(
                    config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=213
                )
                simulation = Simulation(
                    config, LoggerConfig(filename=str(destination / "model.json")), program, verbose=False
                )
                state = simulation.core.arch_state
                try:
                    state.vmem.fill_(0xCD)
                    simulation.run(max_cycles=500000)
                    self.assertTrue(state.halted)
                    self.assertEqual(simulation.get_stats().runtime_errors, 0)
                    start = output - program.dram_base
                    initial[start : start + len(expected)] = expected
                    actual = bytes(state.read_dram(program.dram_base, len(initial)).tolist())
                    self.assertEqual(actual, bytes(initial))
                finally:
                    state.close()


if __name__ == "__main__":
    unittest.main()
