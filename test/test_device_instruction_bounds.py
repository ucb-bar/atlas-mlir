"""Device publications authenticate their recomputed per-launch bounds."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT
from merlin.semantic_compiler.search import SearchLimits


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class DeviceInstructionBoundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="atlas-device-bounds-")
        cls.root = Path(cls.temporary.name)
        cls.snapshot = None

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        selected = os.environ.get("ATLAS_TEST_NATIVE_SNAPSHOT")
        if selected:
            from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
            snapshot = self.native_snapshot()
            capacity = next(bank.capacity for bank in snapshot.profile.banks if bank.name == "external")
            options = codegen_options(CodegenOptions(io_memory_bytes=capacity * 2048))
            options.__enter__()
            self.addCleanup(options.__exit__, None, None, None)

    def native_snapshot(self):
        if self.__class__.snapshot is None:
            selected = os.environ.get("ATLAS_TEST_NATIVE_SNAPSHOT")
            if selected:
                from merlin.semantic_compiler.snapshot import open_native_snapshot
                self.__class__.snapshot = open_native_snapshot(Path(selected))
            else:
                from test_looped_mnk import build_snapshot
                self.__class__.snapshot = build_snapshot(self.root / "native")
        return self.__class__.snapshot

    def check_publication(self, compiled, artifact, export):
        from atlas_mlir_native.restoration_budget import instruction_bound
        program = json.loads((compiled / "physical-program.json").read_text())
        expected = instruction_bound((compiled / "atlas-virtual.mlir").read_text(),
                                     (compiled / "atlas-machine.mlir").read_text(), program)
        record = json.loads((compiled / artifact).read_text())
        self.assertEqual(record, expected)
        self.assertTrue(record["admitted"])
        self.assertEqual(record["max_instructions"], DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT)
        self.assertLessEqual(record["instruction_bound"], DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT)
        manifest = json.loads((compiled / "manifest.json").read_text())
        self.assertEqual(manifest["stage_artifacts"][artifact], sha256((compiled / artifact).read_bytes()).hexdigest())
        binary = (compiled / "program.bin").read_bytes()
        export()
        self.assertEqual((compiled / "program.bin").read_bytes(), binary)
        # Rehash the changed record so ordinary publication hashes still pass.
        # Replay must independently recover and reject the changed proof.
        record["instruction_bound"] -= 1
        (compiled / artifact).write_text(json.dumps(record))
        manifest["stage_artifacts"][artifact] = sha256((compiled / artifact).read_bytes()).hexdigest()
        (compiled / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "instruction bound differs"):
            export()

    @staticmethod
    def low_instruction_limits():
        class Limited(SearchLimits):
            functional_instruction_limit = 1
        return Limited()

    def test_quantizer_reuse_includes_bound_aware_binding(self):
        from types import SimpleNamespace
        from atlas_mlir_native.application import AtlasApplicationAdapter
        alternative = SimpleNamespace(metadata=(("kind", "activation_quantize_device"),))
        self.assertTrue(AtlasApplicationAdapter.supports_region_materialization(None, alternative))

    def test_quantizer_reuse_matches_fresh_publication(self):
        from atlas_mlir_native import activation_quantize_application as quantizer
        from atlas_mlir_native.application_materialization import bind_quantizer
        from merlin.semantic_compiler.materialization import canonical_request
        from test_already_quantized import quantizer_source
        original = quantizer_source(count=64)
        canonical = canonical_request(original, source_identity_policy="provenance")
        fresh, cached, rebound = (self.root / name for name in ("reuse-fresh", "reuse-canonical", "reuse-bound"))
        quantizer.compile_source(None, original, target_source=ROOT, destination=fresh, limits=SearchLimits())
        quantizer.compile_source(None, canonical, target_source=ROOT, destination=cached, limits=SearchLimits())
        bind_quantizer(original, canonical, cached, rebound)
        def members(root):
            return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.assertEqual(members(fresh), members(rebound))

    def test_quantizer_honors_caller_instruction_limit(self):
        from types import SimpleNamespace
        from atlas_mlir_native.activation_quantize_device import compile_source
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from test_already_quantized import quantizer_source
        with self.assertRaisesRegex(NativeCompilationError, "exceeds 1:"):
            compile_source(None, quantizer_source(count=64), target_source=ROOT,
                           destination=self.root / "low-cap", limits=SimpleNamespace(functional_instruction_limit=1))

    def test_quantizer_static_and_dynamic(self):
        from atlas_mlir_native.activation_quantize_device import compile_source, prepare_c_package
        from test_already_quantized import quantizer_source
        from test_activation_quantize_device import dynamic_source
        for index, source in enumerate((quantizer_source(count=64), dynamic_source(shape=(1, 64), reduce="tensor"))):
            with self.subTest(mode=index):
                compiled = self.root / f"quantizer-{index}"
                compile_source(None, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
                self.check_publication(compiled, "quantizer-instruction-bound.json",
                                       lambda: prepare_c_package(source, compiled, "quantizer", target_source=ROOT))

    def test_pointwise_and_math(self):
        from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY, compile_pointwise, plan_abi, prepare_source
        from atlas_mlir_native.pointwise_c import prepare_c_package
        from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
        from test_pointwise_streaming import request
        bf16 = request((32, 32))
        a, b, result = bf16.nodes
        bf16 = replace(bf16, nodes=(replace(a, attrs=()), replace(b, attrs=()),
                                   replace(result, op="bf16_mul", attrs=(("numeric_semantics", "ieee754-bf16-mul-rne-v1"),))))
        type_ = TensorType((32, 32), "f32", "raw-bit-pattern")
        a = SemanticNode("a", "input", (), type_, effect="input")
        b = SemanticNode("b", "input", (), type_, effect="input")
        result = SemanticNode("result", "f32_div", ("a", "b"), type_,
                              (("numeric_semantics", "ieee754-f32-div-rne-v1"),),
                              index_maps=pointwise_identity_maps(2, 2))
        math = KernelRequest((a, b, result), ("result",), ("external",),
                             (("a", "external"), ("b", "external")), TARGET_IDENTITY, "diagnostic")
        for kind, source, options in (("pointwise", bf16, {"bf16_policy": BF16_POINTWISE_POLICY}),
                                      ("pointwise_math", math, {"allow_math": True})):
            with self.subTest(kind=kind):
                precision = prepare_source(source, **options)
                abi = plan_abi(precision.request)
                compiled = self.root / kind
                compile_pointwise(self.native_snapshot(), precision.request, **abi,
                                  target_source=ROOT, destination=compiled, limits=SearchLimits())
                self.check_publication(compiled, "pointwise-instruction-bound.json",
                                       lambda: prepare_c_package(source, compiled, kind, target_source=ROOT, **options))
                from merlin.semantic_compiler.target_binding import NativeCompilationError
                with self.assertRaisesRegex(NativeCompilationError, "exceeds 1:"):
                    compile_pointwise(self.native_snapshot(), precision.request, **abi,
                                      target_source=ROOT, destination=self.root / (kind + "-low-cap"),
                                      limits=self.low_instruction_limits())

    def test_reduction(self):
        from atlas_mlir_native.reduction import compile_reduction, plan_abi, prepare_source, request_for
        from atlas_mlir_native.reduction_c import prepare_c_package
        from merlin.semantic_compiler.model import TensorType
        request = request_for((32, 32), (1,))
        data, seed, result = request.nodes
        nodes = tuple(replace(node, type=TensorType(node.type.shape, "f32", "raw-bit-pattern"), attrs=())
                      for node in (data, seed))
        result = replace(result, op="f32_reduce_add", type=TensorType(result.type.shape, "f32", "raw-bit-pattern"),
                         attrs=(("axes", "1"), ("numeric_semantics", "ieee754-f32-add-rne-v1"),
                                ("reduction_order", "axis-ascending")))
        source = replace(request, nodes=(*nodes, result))
        precision = prepare_source(source)
        compiled = self.root / "reduction"
        compile_reduction(self.native_snapshot(), precision.request, **plan_abi(precision.request),
                          target_source=ROOT, destination=compiled, limits=SearchLimits())
        self.check_publication(compiled, "reduction-instruction-bound.json",
                               lambda: prepare_c_package(source, compiled, "reduction", target_source=ROOT))
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        with self.assertRaisesRegex(NativeCompilationError, "exceeds 1:"):
            compile_reduction(self.native_snapshot(), precision.request, **plan_abi(precision.request),
                              target_source=ROOT, destination=self.root / "reduction-low-cap",
                              limits=self.low_instruction_limits())

    def test_masked_softmax(self):
        from atlas_mlir_native.row_loop import compile_row_loop, verify_row_loop
        from test_tiled_softmax import source_request
        source = source_request((32, 32), masked=True)
        compiled = self.root / "masked-softmax"
        compile_row_loop(self.native_snapshot(), source, target_source=ROOT, destination=compiled, limits=SearchLimits())
        self.check_publication(compiled, "row-loop-instruction-bound.json",
                               lambda: verify_row_loop(source, compiled, target_source=ROOT))
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        with self.assertRaisesRegex(NativeCompilationError, "exceeds 1:"):
            compile_row_loop(self.native_snapshot(), source, target_source=ROOT,
                             destination=self.root / "masked-softmax-low-cap",
                             limits=self.low_instruction_limits())


if __name__ == "__main__":
    unittest.main()
