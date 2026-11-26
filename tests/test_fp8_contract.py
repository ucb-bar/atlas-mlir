"""Small nonzero checks for the admission-only Atlas FP8 adapter."""
import unittest
from dataclasses import replace
import json

import numpy as np

from atlas_mlir_native.fp8_contract import EXPORT_ENCODING, admit_rank2_nt, atlas_numerics, OPS
from atlas_mlir_native.static_quantize import pack_bf16
from merlin.compiler.quantization.contraction.scaled_fp8 import (
    ContractionNumerics, ScaledFP8Spec, require_same_numerics,
)


def codes(values):
    return np.array(values, dtype=np.uint8).view(np.int8)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.lhs = codes([[0x38, 0xc0], [0x40, 0x38]])
        self.rhs = codes([[0x38, 0x40], [0xb8, 0x38], [0x7e, 0xfe]])
        self.le = np.array([[1], [-1]], dtype=np.int32)
        self.re = np.array(-1, dtype=np.int32)

    def admit(self, **changes):
        args = dict(op_name="atlas.qlinear", lhs_codes=self.lhs,
                    lhs_exponents=self.le, rhs_codes=self.rhs, rhs_exponents=self.re)
        args.update(changes)
        return admit_rank2_nt(**args)

    def test_nonzero_nt_and_unchanged_bytes(self):
        before = self.lhs.tobytes(), self.rhs.tobytes(), self.le.tobytes()
        result = self.admit()
        np.testing.assert_array_equal(result.lhs, [[2, -4], [1, .5]])
        np.testing.assert_array_equal(result.rhs, [[.5, 1], [-.5, .5], [224, -224]])
        np.testing.assert_array_equal(result.lhs @ result.rhs.T,
                                      [[-3, -3, 1344], [1, -.25, 112]])
        self.assertEqual(before, (self.lhs.tobytes(), self.rhs.tobytes(), self.le.tobytes()))
        self.assertEqual(result.lhs.dtype, np.float64)

    def test_observed_names_and_numerical_mismatch(self):
        for name in OPS:
            require_same_numerics(atlas_numerics(name), self.admit(op_name=name).numerics)
        bias = atlas_numerics(OPS[0], bias=True)
        self.assertEqual(bias.bias_rounding, "bf16-rne")
        self.assertEqual(bias.activation, "none")
        self.assertEqual(bias.stages[-2:], ("output-bf16", "source-bias-after-output-cast"))
        with self.assertRaises(ValueError):
            atlas_numerics("matmul")
        with self.assertRaises(ValueError):
            require_same_numerics(atlas_numerics(OPS[0]), ContractionNumerics(
                accumulation="fp32", readout="fp32", stages=("matmul",)))

    def test_invalid_codes(self):
        for code in (1, 7, 0x7f, 0x80, 0x81, 0x87, 0xff):
            bad = self.lhs.copy()
            bad.view(np.uint8)[0, 0] = code
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.admit(lhs_codes=bad)
        # Every byte decodes as the exporter's unpacker does: exponent field 0, 0x7f, and 0xff are signed zero.
        raw = np.arange(256, dtype=np.uint8)
        field, fraction = (raw >> 3) & 15, raw & 7
        expected = np.where((field == 0) | ((raw & 0x7f) == 0x7f), 0., (1 + fraction / 8) * 2.0 ** (field - 7.))
        expected = np.where(raw & 0x80, -expected, expected)
        decoded = ScaledFP8Spec((), EXPORT_ENCODING).decode(raw.view(np.int8), np.array(0, dtype=np.int32))
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(np.signbit(decoded), raw >= 0x80)

    def test_exponent_bounds_and_sums(self):
        for lhs, rhs in ((128, -1), (-128, 0), (100, 28), (-100, -34)):
            with self.subTest(lhs=lhs, rhs=rhs), self.assertRaises(ValueError):
                self.admit(lhs_exponents=np.array(lhs, np.int32),
                           rhs_exponents=np.array(rhs, np.int32))
        self.admit(lhs_exponents=np.array(-127, np.int32), rhs_exponents=np.array(-6, np.int32))
        self.admit(lhs_exponents=np.array(127, np.int32), rhs_exponents=np.array(0, np.int32))

    def test_storage_shape_and_per_k_rejections(self):
        for changes in (
                {"lhs_codes": self.lhs.view(np.uint8)},
                {"lhs_exponents": self.le.astype(np.int64)},
                {"lhs_exponents": np.zeros((1, 2), np.int32)},
                {"lhs_exponents": np.zeros((2, 2), np.int32)},
                {"rhs_codes": self.rhs[:, :1]},
                {"lhs_codes": self.lhs.reshape(1, 2, 2)},
        ):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                self.admit(**changes)

    def test_authored_bf16_pack_witnesses(self):
        # Two halfway cases choose the even significand; a carry can lift an
        # underflowing value into minimum normal before the flush decision.
        raw = np.array([0x3f88, 0x3f98, 0x3c78, 0x3c77, 0x43e0, 0x43f0,
                        0xc3f0, 0x7f80, 0xff80, 0x7fc0, 0x8000, 0x0001], np.uint16)
        np.testing.assert_array_equal(pack_bf16(raw, 0),
                                     [0x38, 0x3a, 8, 0, 0x7e, 0x7e,
                                      0xfe, 0x7e, 0xfe, 0, 0, 0])
        # Only the encoding shift clamps. This API returns codes; it does not
        # rewrite the authored exponent attached to them by the source op.
        np.testing.assert_array_equal(pack_bf16(np.array([0x7f00], np.uint16), 128), [0x38])
        np.testing.assert_array_equal(pack_bf16(np.array([0x0080], np.uint16), -128), [0x40])


class SourceAdmissionTests(unittest.TestCase):
    """Exercise the real provider translator and its existing closed intake."""

    setUp = AdapterTests.setUp

    def source(self, *, op_name="atlas.qlinear", constants=(0, 1, 2, 3), output_dtype="bf16", **changes):
        from xdsl.dialects.builtin import BFloat16Type, Float32Type, IntegerType, StringAttr, TensorType
        from xdsl.ir import Block

        from atlas_mlir_native.source_ops import QLinearOp, QKMatmulOp, PVMatmulOp
        from atlas_mlir_native.source_translation import translate_prequantized_contraction
        from merlin.compiler.semantic.model import ConstantBinding, KernelRequest, SemanticNode

        arrays = [changes.get("lhs_codes", self.lhs), changes.get("lhs_exponents", self.le),
                  changes.get("rhs_codes", self.rhs), changes.get("rhs_exponents", self.re)]
        types = [TensorType(IntegerType(8 if index % 2 == 0 else 32), value.shape)
                 for index, value in enumerate(arrays)]
        block = Block(arg_types=types)
        attrs = {key: StringAttr(value) for key, value in {
            "code_dtype": "uint8", "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
            "dequantization": "e4m3(code)*2^exponent", "rhs_transposed": "true",
            "output_rescale": "2^(lhs_exp+rhs_exp)", "bias_order": "after_output_cast",
            "output_dtype": "bfloat16" if output_dtype == "bf16" else "float32",
        }.items()}
        classes = {"atlas.qlinear": QLinearOp, "atlas.qk_matmul": QKMatmulOp, "atlas.pv_matmul": PVMatmulOp}
        operands = [*block.args, []] if op_name == "atlas.qlinear" else list(block.args)
        operation = classes[op_name].build(operands=operands, attributes=attrs,
            result_types=[TensorType(BFloat16Type() if output_dtype == "bf16" else Float32Type(), [2, 3])])
        ids = ("lc", "le", "rc", "re")
        output, = translate_prequantized_contraction(operation, ids, ("out",))
        from merlin.compiler.semantic.linalg.linalg_types import _type
        inputs = tuple(SemanticNode(key, "constant" if index in constants else "input", (), _type(typ),
                                   effect="constant" if index in constants else "input")
                       for index, (key, typ) in enumerate(zip(ids, types)))
        bindings = tuple(ConstantBinding(ids[index], "i8" if index % 2 == 0 else "i32-le",
                                        value.tobytes().hex())
                         for index, value in enumerate(arrays) if index in constants)
        return KernelRequest((*inputs, output), ("out",), ("external",),
                             tuple((key, "external") for index, key in enumerate(ids) if index not in constants),
                             "atlas-test", "diagnostic", "authored-nonzero-export", bindings)

    def test_source_intake_admits_nonzero_constants_and_preserves_custody(self):
        from atlas_mlir_native.source_translation import lower_already_quantized_contraction
        from merlin.compiler.semantic.model import KernelRequest
        for name in OPS:
            for dtype in ("bf16", "f32"):
                with self.subTest(name=name, dtype=dtype):
                    source = self.source(op_name=name, output_dtype=dtype)
                    before = source.record()
                    intake = lower_already_quantized_contraction(source)
                    self.assertEqual(intake.value_admission, "all-constant-operands-checked")
                    require_same_numerics(intake.numerics, atlas_numerics(name, output_dtype=dtype))
                    self.assertEqual(source.record(), before)
                    replay = lower_already_quantized_contraction(KernelRequest.from_record(before))
                    self.assertEqual(intake.record(), replay.record())
                    self.assertEqual(intake.record()["source_request_digest"], source.digest())
                    self.assertEqual(intake.record()["constant_bindings"][0]["node_id"], "lc")
                    self.assertIn("source arithmetic equivalence unproven", intake.record()["numerical_authorization"])

    def test_manifest_normalization_admits_nonzero_constants_with_custody(self):
        from hashlib import sha256
        from xdsl.dialects.builtin import (ArrayAttr, BFloat16Type, DictionaryAttr, IntegerType,
                                           ModuleOp, StringAttr, TensorType)
        from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
        from xdsl.ir import Block, Region

        from atlas_mlir_native.source_normalize import normalize
        from atlas_mlir_native.source_ops import QLinearOp
        from atlas_mlir_native.source_translation import (NUMERICS_ATTRIBUTE, lower_already_quantized_contraction,
                                                         translate_prequantized_contraction)

        source = self.source()
        inputs = source.nodes[:-1]
        types = [TensorType(IntegerType(8 if index % 2 == 0 else 32), node.type.shape)
                 for index, node in enumerate(inputs)]
        result_type = TensorType(BFloat16Type(), [2, 3])
        block = Block(arg_types=types)
        schema = ("atlas::qlinear(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, "
                  "Tensor? bias, str out_dtype, str weight_format) -> Tensor")
        literals = dict(bias=None, out_dtype="bfloat16", weight_format="atlas_e4m3")
        call = CallOp("opaque_export_call", block.args, [result_type])
        call.attributes.update({"prov.source_node_ids": ArrayAttr([StringAttr("source:contraction")]),
                                "m2m.schema": StringAttr(schema)})
        call.attributes.update({"m2m.literal." + key: StringAttr(json.dumps(value))
                                for key, value in literals.items()})
        block.add_ops([call, ReturnOp(call.results[0])])
        arg_attrs, operands, buffers, authored = [], [], {}, {}
        for index, (node, binding, role) in enumerate(zip(inputs, source.constants,
                ("x_codes", "x_exp", "w_codes", "w_exp"), strict=True)):
            dtype = "uint8" if index % 2 == 0 else "int32"
            data = bytes.fromhex(binding.data_hex)
            self.assertNotEqual(data, bytes(len(data)))
            record = dict(dtype=dtype, shape=list(node.type.shape), bytes=len(data), sha256=sha256(data).hexdigest())
            authored["source." + node.id] = record
            buffers[node.id] = dict(source_name="source." + node.id, **record)
            operands.append(dict(role=role, node_id="source:" + node.id, result_index=0,
                                 buffer=node.id, dtype=dtype, shape=list(node.type.shape)))
            arg_attrs.append(DictionaryAttr({"prov.source_node_ids": ArrayAttr([StringAttr("source:" + node.id)]),
                                              "prov.get_attr": StringAttr(node.id)}))
        module = ModuleOp([FuncOp.external("opaque_export_call", types, [result_type]),
                           FuncOp("work", (types, [result_type]), Region(block), arg_attrs=ArrayAttr(arg_attrs))])
        manifest = dict(schema="m2m.quantization_manifest.v2", adapter_id="atlas_fp8",
            sites=[dict(site_id="site", status="quantized", execution_route="atlas.native_fp8")],
            atlas=dict(schema="atlas.m2m_adapter.v1", storage="uint8", buffers=buffers, source_tensors=authored,
                calls=[dict(node_id="source:contraction", site_id="site", invocation=0,
                            target="atlas.qlinear.default", schema=schema, literals=literals, operands=operands,
                            results=[dict(index=0, dtype="bfloat16", shape=[2, 3])])]))
        before = json.dumps(manifest, sort_keys=True)
        normalize(module, manifest)
        normalized = next(op for op in module.walk() if isinstance(op, QLinearOp))
        self.assertEqual(tuple(normalized.operands), tuple(block.args))
        declaration = json.loads(normalized.attributes[NUMERICS_ATTRIBUTE].data)
        self.assertEqual(declaration, json.loads(json.dumps(atlas_numerics("atlas.qlinear").record())))
        output, = translate_prequantized_contraction(normalized, tuple(node.id for node in inputs), ("out",))
        admitted_source = replace(source, nodes=(*inputs, output))
        intake = lower_already_quantized_contraction(admitted_source)
        self.assertEqual(intake.value_admission, "all-constant-operands-checked")
        self.assertEqual(admitted_source.constants, source.constants)
        self.assertEqual(json.dumps(manifest, sort_keys=True), before)
        self.assertIn("prov.quantization_manifest_sha256", module.attributes)
        normalized.attributes[NUMERICS_ATTRIBUTE] = StringAttr(json.dumps({**declaration, "accumulation": "fp32"}))
        with self.assertRaisesRegex(ValueError, "numerical semantics"):
            translate_prequantized_contraction(normalized, tuple(node.id for node in inputs), ("out",))
        binding = admitted_source.constants[0]
        changed = replace(admitted_source, constants=(replace(binding, data_hex="40" + binding.data_hex[2:]),
                                                     *admitted_source.constants[1:]))
        with self.assertRaisesRegex(ValueError, "identity or constant bytes changed"):
            intake.verify_source(changed)

    def test_source_bias_is_declared_but_refused_by_narrow_value_admission(self):
        from atlas_mlir_native.source_translation import (NUMERICS_ATTRIBUTE, contraction_numerics_attribute,
                                                         lower_already_quantized_contraction)
        from merlin.compiler.semantic.model import SemanticNode, TensorType
        source = self.source()
        bias = SemanticNode("bias", "input", (), TensorType((3,), "bf16", "raw-bit-pattern"), effect="input")
        original = source.nodes[-1]
        attrs = dict(original.attrs)
        attrs[NUMERICS_ATTRIBUTE] = contraction_numerics_attribute(original.op, "bfloat16", bias=True)
        biased = replace(original, inputs=(*original.inputs, "bias"), attrs=tuple(sorted(attrs.items())))
        source = replace(source, nodes=(*source.nodes[:-1], bias, biased), input_storages=(("bias", "external"),))
        with self.assertRaisesRegex(ValueError, "requires unbiased source operands"):
            lower_already_quantized_contraction(source)

    def test_k32_k64_source_bias_preparation_and_readout_admission(self):
        from merlin.compiler.semantic.source_preparation import prepare_source, verify_source_preparation
        from merlin.compiler.semantic.prequantized_bias import is_prequantized_bias_add
        from merlin.compiler.semantic.model import ConstantBinding, SemanticNode, TensorType
        from atlas_mlir_native.source_translation import (
            NUMERICS_ATTRIBUTE, contraction_numerics_attribute, lower_already_quantized_contraction)
        for k in (32, 64):
            with self.subTest(k=k):
                source = self.source(lhs_codes=np.tile(self.lhs, (1, k // 2)),
                                     rhs_codes=np.tile(self.rhs, (1, k // 2)))
                # BF16 3/512, -1/2 and +1: unchanged nonzero bias bytes.
                raw_bias = np.array([0x3bc0, 0xbf00, 0x3f80], dtype="<u2")
                bias = SemanticNode("bias", "constant", (),
                                    TensorType((3,), "bf16", "raw-bit-pattern"), effect="constant")
                original = source.nodes[-1]
                attrs = dict(original.attrs)
                attrs[NUMERICS_ATTRIBUTE] = contraction_numerics_attribute(
                    original.op, "bfloat16", bias=True)
                biased = replace(original, inputs=(*original.inputs, "bias"), attrs=tuple(sorted(attrs.items())))
                source = replace(source, nodes=(*source.nodes[:-1], bias, biased),
                                 constants=(*source.constants, ConstantBinding("bias", "bf16-le-raw", raw_bias.tobytes().hex())))
                before = source.record()
                prepared, receipt = prepare_source(source, include_tensor_constants=False)
                self.assertEqual(verify_source_preparation(prepared, receipt).record(), before)
                self.assertEqual(prepared.constants, source.constants)
                addition = prepared.node("out")
                self.assertTrue(is_prequantized_bias_add(prepared, addition))
                self.assertEqual(addition.inputs, ("out.readout", "out.bias"))
                self.assertEqual(dict(addition.attrs)["numeric_semantics"], "ieee754-bf16-add-rne-v1")
                readout = prepared.node("out.readout")
                declaration = json.loads(dict(readout.attrs)[NUMERICS_ATTRIBUTE])
                self.assertEqual(declaration["bias_rounding"], "none")
                self.assertNotIn("source-bias-after-output-cast", declaration["stages"])
                closed = replace(source, nodes=(*source.nodes[:4], readout),
                                 outputs=(readout.id,), constants=source.constants[:4])
                admitted = lower_already_quantized_contraction(closed)
                self.assertEqual(admitted.k, k)
                self.assertEqual(admitted.value_admission, "all-constant-operands-checked")
                # A device chop add and an MXU seed cannot replace this source stage.
                with self.assertRaisesRegex(ValueError, "numerical semantics"):
                    lower_already_quantized_contraction(closed, candidate_numerics=atlas_numerics(
                        original.op, bias=True))
                changed = replace(addition, attrs=(("numeric_semantics", "bf16-chop"),))
                tampered = replace(prepared, nodes=tuple(changed if n.id == addition.id else n for n in prepared.nodes))
                self.assertFalse(is_prequantized_bias_add(tampered, changed))
                with self.assertRaises(ValueError):
                    verify_source_preparation(tampered, receipt)
                # Stale biased metadata on the four-input readout refuses admission.
                stale = replace(readout, attrs=biased.attrs)
                with self.assertRaisesRegex(ValueError, "numerical semantics"):
                    lower_already_quantized_contraction(replace(closed, nodes=(*closed.nodes[:-1], stale)))
                self.assertEqual(source.record(), before)

    def test_batched_source_checks_each_slice_and_refuses_other_broadcasts(self):
        from atlas_mlir_native.source_translation import lower_already_quantized_contraction
        source = self.source()
        # Singleton batch axes preserve every constant byte; each slice is one rank-two contraction.
        nodes = tuple(replace(node, type=replace(node.type, shape=(1, *node.type.shape)))
                      if node.type.shape else node for node in source.nodes)
        batched = lower_already_quantized_contraction(replace(source, nodes=nodes))
        self.assertEqual(batched.batch_shape, (1,))
        self.assertEqual(batched.value_admission, lower_already_quantized_contraction(source).value_admission)
        # A batched right operand against a rank-two left operand is a broadcast, not a shared weight.
        broadcast = tuple(replace(node, type=replace(node.type, shape=(1, *node.type.shape)))
                          if node.id in {"rc", "out"} else node for node in source.nodes)
        with self.assertRaisesRegex(ValueError, "identical batch prefixes or one shared weight"):
            lower_already_quantized_contraction(replace(source, nodes=broadcast))

    def test_source_intake_refuses_invalid_nonzero_constant_values(self):
        from atlas_mlir_native.source_translation import lower_already_quantized_contraction
        bad = self.lhs.copy()
        bad.view(np.uint8)[0, 0] = 0x80
        for changes in ({"lhs_codes": bad}, {"lhs_exponents": np.array(128, np.int32)},
                        {"lhs_exponents": np.array(100, np.int32), "rhs_exponents": np.array(28, np.int32)}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                lower_already_quantized_contraction(self.source(**changes))
        # Validate known weights even when activation inputs arrive at runtime.
        with self.assertRaisesRegex(ValueError, "cap448"):
            lower_already_quantized_contraction(self.source(constants=(0,), lhs_codes=bad))
        with self.assertRaisesRegex(ValueError, "exponent sum"):
            lower_already_quantized_contraction(self.source(constants=(1, 3),
                lhs_exponents=np.array(100, np.int32), rhs_exponents=np.array(28, np.int32)))

    def test_source_numerical_mutations_and_candidate_routes_are_refused(self):
        from atlas_mlir_native.source_translation import NUMERICS_ATTRIBUTE, lower_already_quantized_contraction
        source = self.source()
        original = source.nodes[-1]
        declaration = json.loads(dict(original.attrs)[NUMERICS_ATTRIBUTE])
        for changes in ({"accumulation": "fp32"}, {"readout": "fp32"}, {"bias_rounding": "bf16-chop"},
                        {"activation": "bf16-sign-clamp"}, {"stages": list(reversed(declaration["stages"]))}):
            attrs = dict(original.attrs)
            attrs[NUMERICS_ATTRIBUTE] = json.dumps({**declaration, **changes}, sort_keys=True, separators=(",", ":"))
            mutated = replace(source, nodes=(*source.nodes[:-1], replace(original, attrs=tuple(sorted(attrs.items())))))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "numerical semantics"):
                lower_already_quantized_contraction(mutated)
        # Retired target-specific numerical fields cannot bypass the shared owner.
        stale_attrs = dict(original.attrs)
        stale_attrs["atlas.diagnostic_contraction_numerics"] = stale_attrs.pop(NUMERICS_ATTRIBUTE)
        stale = replace(source, nodes=(*source.nodes[:-1], replace(original, attrs=tuple(sorted(stale_attrs.items())))))
        with self.assertRaises(ValueError):
            lower_already_quantized_contraction(stale)
        candidate = replace(atlas_numerics(OPS[0]), accumulation="fp32")
        with self.assertRaisesRegex(ValueError, "numerical semantics"):
            lower_already_quantized_contraction(source, candidate_numerics=candidate)

    def test_selected_family_recipe_binds_numerical_contract(self):
        from types import SimpleNamespace
        from atlas_mlir_native.compiler_interface import lowering_recipe, validate_recipe
        from atlas_mlir_native.source_translation import lower_already_quantized_contraction
        for dtype in ("bf16", "f32"):
            source = self.source(output_dtype=dtype)
            numerics = lower_already_quantized_contraction(source).numerics
            recipe = lowering_recipe(SimpleNamespace(metadata=(("kind", "already_quantized"),)), numerics=numerics)
            validate_recipe(recipe)
            raw = dict(recipe.guards)["numerical.contraction"]
            self.assertEqual(json.loads(raw), json.loads(json.dumps(numerics.record())))
            for key, value in (("accumulation", "fp32"), ("bias_rounding", "bf16-chop"),
                               ("activation", "bf16-sign-clamp")):
                changed = json.dumps({**json.loads(raw), key: value}, sort_keys=True, separators=(",", ":"))
                forged = replace(recipe, guards=tuple((k, changed if k == "numerical.contraction" else v)
                                                     for k, v in recipe.guards))
                with self.subTest(dtype=dtype, field=key), self.assertRaises(ValueError):
                    validate_recipe(forged)
            missing = replace(recipe, guards=tuple((k, v) for k, v in recipe.guards if k != "numerical.contraction"))
            with self.assertRaises(ValueError):
                validate_recipe(missing)

    def test_runtime_values_are_explicitly_pending_and_source_changes_are_detected(self):
        from atlas_mlir_native.source_translation import lower_already_quantized_contraction
        source = self.source(constants=(2, 3))
        intake = lower_already_quantized_contraction(source)
        self.assertIn("runtime-input-validation-required", intake.record()["value_admission"])
        self.assertEqual(len(intake.record()["runtime_obligations"]), 3)
        binding = source.constants[0]
        changed = replace(source, constants=(replace(binding, data_hex="40" + binding.data_hex[2:]), source.constants[1]))
        with self.assertRaisesRegex(ValueError, "identity or constant bytes changed"):
            intake.verify_source(changed)


    def test_ordered_relu_host_execution_and_family(self):
        import ctypes
        from pathlib import Path
        import subprocess
        import tempfile

        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.binding import TARGET_IDENTITY
        from merlin.compiler.lower.c.float_arithmetic import emit_float_arithmetic_c
        from merlin.compiler.lower.c.host_ops.relu import SELECT_ATTRS
        from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c
        from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

        def request(op, attrs, count, inputs=("x",)):
            type_ = TensorType((count,), "bf16", "raw-bit-pattern")
            operands = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in inputs)
            output = SemanticNode("y", op, inputs, type_, tuple(sorted(attrs.items())),
                                  index_maps=pointwise_identity_maps(1, len(inputs)))
            return KernelRequest((*operands, output), ("y",), ("external",),
                                 tuple((key, "external") for key in inputs), TARGET_IDENTITY, "diagnostic")

        def execute(files, name, values):
            with tempfile.TemporaryDirectory(prefix="fp8-host-execution-") as directory:
                root = Path(directory)
                for member, content in files.items():
                    destination = root / member
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content.encode() if isinstance(content, str) else content)
                library = root / "host.so"
                subprocess.run(["nice", "-n", "10", "cc", "-std=c11", "-shared", "-fPIC", "-O2",
                                *[str(root / member) for member in files if member.endswith(".c")],
                                "-o", str(library)], check=True, capture_output=True)
                function = getattr(ctypes.CDLL(str(library)), name)
                function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
                function.restype = ctypes.c_int
                output = np.full(values[0].shape, 0xdead, dtype="<u2")
                pointers = (ctypes.c_void_p * len(values))(*(value.ctypes.data for value in values))
                outputs = (ctypes.c_void_p * 1)(output.ctypes.data)
                self.assertEqual(function(None, pointers, outputs), 0)
                return output

        ordered = request("relu", SELECT_ATTRS, 65536)
        with self.subTest(check="all-bf16-encodings"):
            record, files = emit_host_pointwise_c(ordered, "ordered_relu")
            self.assertEqual(record["numeric_semantics"], SELECT_ATTRS["numeric_semantics"])
            self.assertEqual(record["input_domain"], "all-bf16-bit-patterns")
            self.assertIn("preserve signed zero and NaN bits", record["operation_order"])
            raw = np.arange(65536, dtype="<u2")
            magnitude = raw & 0x7fff
            expected = np.where(((raw & 0x8000) != 0) & (magnitude > 0)
                                & (magnitude <= 0x7f80), 0, raw).astype("<u2")
            np.testing.assert_array_equal(execute(files, "ordered_relu", (raw,)), expected)

        with self.subTest(check="distinct-compiler-family"):
            policy = json.loads((Path(__file__).resolve().parents[1]
                                 / "test/fixtures/refactor-k32/precision-policy.json").read_text())
            adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
            ordered_alternative, = adapter.plan(ordered).selected
            self.assertEqual(dict(ordered_alternative.metadata)["kind"], "host_ordered_relu")
            self.assertEqual(ordered_alternative.implementation, "source_host_ordered_relu")
            with tempfile.TemporaryDirectory(prefix="fp8-host-family-") as directory:
                compiled = adapter.export_region(ordered, ordered_alternative, "ordered_family", Path(directory),
                                                 Path(__file__).resolve().parents[1])
                ordered_recipe = compiled.description["compiler_recipe"]
                self.assertEqual(dict(ordered_recipe["guards"])["family"], "host.ordered-select")
                self.assertEqual(compiled.description["source_request_digest"], ordered.digest())
                self.assertEqual(compiled.description["numeric_semantics"], SELECT_ATTRS["numeric_semantics"])
                clamp = request("relu", {"numeric_semantics": "bf16-sign-clamp-v1",
                                         "input_domain": "finite-normal-or-zero-bf16"}, 65536)
                from merlin.compiler.semantic.target_binding import NativeCompilationError
                with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
                    adapter.plan(clamp)

        with self.subTest(check="source-bias-rne-witness"):
            addition = request("bf16_add", {"numeric_semantics": "ieee754-bf16-add-rne-v1"}, 1, ("x", "bias"))
            record, files = emit_float_arithmetic_c(addition, "source_bias_rne")
            self.assertEqual(record["numeric_semantics"], "ieee754-bf16-add-rne-v1")
            result = execute(files, "source_bias_rne", (np.array([0x3f80], dtype="<u2"),
                                                       np.array([0x3bc0], dtype="<u2")))
            self.assertEqual(int(result[0]), 0x3f81)
            self.assertNotEqual(int(result[0]), 0x3f80)  # Primitive BF16 chop witness.


if __name__ == "__main__":
    unittest.main()
