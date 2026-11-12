"""Source admission and independent grouped-tree RMSNorm policy checks."""
from __future__ import annotations

from dataclasses import replace
import struct
import unittest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.rmsnorm import RMSNORM_POLICY, prepare_rmsnorm, prepare_source, request_for
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.qualified_rmsnorm_source import recognize_rmsnorm
from test_vpu_row_sum_reference import _decode, _encode_bf16, _exponent, _pow2, _round_nearest_even, _tree_row

def source_for(shape=(2, 64)):
    m, n = shape
    matrix, row, column = (TensorType(s, "f32", "raw-bit-pattern") for s in (shape, (m,), (m, 1)))
    data = SemanticNode("data", "input", (), matrix, effect="input")
    gamma = SemanticNode("gamma", "input", (), TensorType((n,), "f32", "raw-bit-pattern"), effect="input")
    seed = SemanticNode("seed", "constant", (), row, effect="constant")
    divisor = SemanticNode("divisor", "constant", (), row, effect="constant")
    epsilon = SemanticNode("epsilon", "constant", (), column, effect="constant")
    identity = pointwise_identity_maps(2, 1)[0]
    row_map = IndexMap(2, ((1, 0),), (0,))
    gamma_map = IndexMap(2, ((0, 1),), (0,))
    broadcast = IndexMap(2, ((1, 0), (0, 0)), (0, 0))
    numeric = lambda operation: (("numeric_semantics", f"ieee754-f32-{operation}-rne-v1"),)
    square = SemanticNode("square", "f32_mul", (data.id, data.id), matrix, numeric("mul"), index_maps=pointwise_identity_maps(2, 2))
    reduced = SemanticNode("sum", "source_f32_reduce_add_unqualified", (square.id, seed.id), row,
                           (("axis", 1), ("reduction_order", "unresolved"), ("numeric_semantics", "source-f32-add-unqualified-v1"),
                            ("scalar_add_order", "input-accumulator")), index_maps=(identity, row_map, row_map))
    mean = SemanticNode("mean", "f32_div", (reduced.id, divisor.id), row, numeric("div"), index_maps=pointwise_identity_maps(1, 2))
    view = SemanticNode("view", "reshape", (mean.id,), column)
    shifted = SemanticNode("shifted", "f32_add", (view.id, epsilon.id), column, numeric("add"), index_maps=pointwise_identity_maps(2, 2))
    inverse = SemanticNode("inverse", "source_f32_rsqrt_unqualified", (shifted.id,), column,
                           (("numeric_semantics", "source-f32-rsqrt-unqualified-v1"),), index_maps=pointwise_identity_maps(2, 1))
    normalized = SemanticNode("normalized", "f32_mul", (data.id, inverse.id), matrix, numeric("mul"), index_maps=(identity, broadcast, identity))
    out = SemanticNode("result", "f32_mul", (gamma.id, normalized.id), matrix, numeric("mul"), index_maps=(gamma_map, identity, identity))
    constants = tuple(ConstantBinding(key, "f32-le-raw", struct.pack("<f", value).hex() * m)
                      for key, value in ((seed.id, 0.0), (divisor.id, float(n)), (epsilon.id, 1e-5)))
    return KernelRequest((data, gamma, seed, divisor, epsilon, square, reduced, mean, view, shifted, inverse, normalized, out),
                          (out.id,), ("external",), ((data.id, "external"), (gamma.id, "external")), TARGET_IDENTITY,
                          "diagnostic", constants=constants)


def change_node(source, key, **fields):
    return replace(source, nodes=tuple(replace(node, **fields) if node.id == key else node for node in source.nodes))


def change_literal(source, key, code):
    return replace(source, constants=tuple(replace(binding, data_hex=code.to_bytes(4, "little").hex() *
                                                      (len(binding.data_hex) // 8)) if binding.node_id == key else binding
                                          for binding in source.constants))


def chop_bf16(value):
    if not value:
        return 0
    step = _pow2(_exponent(value) - 7)
    truncated = (abs(value) // step) * step
    return _encode_bf16(-truncated if value < 0 else truncated)


class RMSNormTests(unittest.TestCase):
    def test_normalization_policy_keeps_other_pointwise_operations_available(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.pointwise import POINTWISE_POLICY
        from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from test_qualified_rne_add import source_request

        source = source_request("f32", (2, 64))
        left, right, addition = source.nodes
        product = replace(addition, id="product", op="f32_mul", inputs=(left.id, left.id),
                          attrs=(("numeric_semantics", "ieee754-f32-mul-rne-v1"),))
        source = replace(source, nodes=(left, right, product, replace(addition, inputs=(product.id, right.id))))
        source_sha = "1" * 64
        base = dict(schema="atlas.diagnostic_application_precision.v4", source_sha256=source_sha,
                    target_identity=TARGET_IDENTITY,
                    contraction=dict(policy=SCALED_CONTRACTION_POLICY, operand_scale_exponents=[0, 0]),
                    row_softmax=SOFTMAX_POLICY, host_support=["transpose", "reshape", "copy"],
                    epilogues=["relu"], pointwise=POINTWISE_POLICY)
        policy = dict(schema="atlas.diagnostic_normalization_application_precision.v1", source_sha256=source_sha,
                      base_policy=base, rne_add=RNE_ADD_POLICY, rmsnorm=RMSNORM_POLICY)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=source_sha)
        selected = adapter.plan(source).selected
        self.assertEqual([dict(row.metadata)["kind"] for row in selected], ["pointwise", "pointwise"])
        self.assertEqual([row.nodes for row in selected], [(0,), (1,)])
        for changed in (policy | {"rne_add": "plain-chop"}, policy | {"source_sha256": "2" * 64},
                        policy | {"unrecognized_choice": True}, policy | {"base_policy": base | {"source_sha256": "2" * 64}}):
            with self.subTest(policy=changed), self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(changed, original_source_sha256=source_sha)

    def test_conflicting_nested_addition_and_normalization_policies_refuse(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.hierarchical_rmsnorm import POLICY as hierarchical_policy
        from atlas_mlir_native.pointwise import POINTWISE_POLICY
        from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY as qualified_add
        from atlas_mlir_native.rne_add import RNE_ADD_POLICY as paired_add
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        source_sha = "1" * 64
        base = dict(schema="atlas.diagnostic_application_precision.v4", source_sha256=source_sha,
                    target_identity=TARGET_IDENTITY,
                    contraction=dict(policy=SCALED_CONTRACTION_POLICY, operand_scale_exponents=[0, 0]),
                    row_softmax=SOFTMAX_POLICY, host_support=["transpose", "reshape", "copy"],
                    epilogues=["relu"], pointwise=POINTWISE_POLICY)

        def qualified(inner, **selected):
            return dict(schema="atlas.diagnostic_normalization_application_precision.v1",
                        source_sha256=source_sha, base_policy=inner, **selected)

        def paired(inner):
            return dict(schema="atlas.diagnostic_application_precision.v23", source_sha256=source_sha,
                        base_policy=inner, rne_add=paired_add, softmax_value_schedules=[],
                        mxu1_contraction_schedules=[])

        def hierarchical(inner):
            return dict(schema="atlas.diagnostic_application_precision.v24", source_sha256=source_sha,
                        base_policy=inner, rmsnorm_schedules=[dict(source_output="norm",
                            source_request_digest="2" * 64, provenance_sha256="3" * 64,
                            policy=hierarchical_policy, output_scale_exponent=0)])

        for policy in (qualified(paired(base), rne_add=qualified_add),
                       paired(qualified(base, rne_add=qualified_add)),
                       qualified(hierarchical(base), rmsnorm=RMSNORM_POLICY),
                       hierarchical(qualified(base, rmsnorm=RMSNORM_POLICY))):
            with self.subTest(schema=policy["schema"]), self.assertRaisesRegex(
                    NativeCompilationError, "conflicting nested"):
                AtlasApplicationAdapter(policy, original_source_sha256=source_sha)
        for policy in (qualified(paired(base), rmsnorm=RMSNORM_POLICY),
                       hierarchical(qualified(base, rne_add=qualified_add))):
            AtlasApplicationAdapter(policy, original_source_sha256=source_sha)

    def test_source_precision_has_complete_provenance_and_named_policy(self):
        source = source_for()
        precision = prepare_source(source)
        self.assertEqual(precision.input_ids, ("data", "gamma"))
        self.assertEqual(precision.request.source_identity, source.digest())
        self.assertEqual(precision.request.nodes[-1].type.numerical_policy, RMSNORM_POLICY)
        self.assertEqual(precision.record()["source_request_digest"], source.digest())
        self.assertIn("diagnostic_precision_change", precision.record()["qualification"])
        changed = change_literal(source, "epsilon", 0x3727C5AD)
        alternate = prepare_source(changed)
        self.assertNotEqual(alternate.request.digest(), precision.request.digest())
        self.assertEqual(alternate.request.source_identity, changed.digest())

    def test_wrong_seed_divisor_broadcast_and_escaping_value_refuse(self):
        source = source_for()
        identity = pointwise_identity_maps(2, 1)[0]
        wrong_broadcast = IndexMap(2, ((0, 0), (0, 0)), (0, 0))
        mutations = {
            "seed": change_literal(source, "seed", 0x3F800000),
            "divisor": change_literal(source, "divisor", 0x42000000),
            "gamma broadcast": change_node(source, "result", index_maps=(IndexMap(2, ((1, 0),), (0,)), identity, identity)),
            "row broadcast": change_node(source, "normalized", index_maps=(identity, wrong_broadcast, identity)),
            "escaping intermediate": replace(source, outputs=("result", "square"), output_storages=("external", "external")),
            "negative epsilon": change_literal(source, "epsilon", 0xB727C5AC),
            "infinite epsilon": change_literal(source, "epsilon", 0x7F800000),
            "square operand": change_node(source, "normalized", inputs=("square", "inverse")),
        }
        for label, mutated in mutations.items():
            with self.subTest(mutation=label), self.assertRaises(ValueError):
                prepare_source(mutated)
        user = SemanticNode("other_user", "f32_mul", ("square", "data"), source.node("data").type,
                            source.node("square").attrs, index_maps=pointwise_identity_maps(2, 2))
        escaped = replace(source, nodes=(*source.nodes, user))
        with self.assertRaisesRegex(ValueError, "escapes"):
            recognize_rmsnorm(escaped, "result")

    def test_shape_policy_and_target_mutations_refuse(self):
        request = request_for((33, 64))
        tiling = prepare_rmsnorm(request)
        self.assertEqual([(r.row_start, r.valid_rows) for r in tiling.regions], [(0, 32), (32, 1)])
        self.assertEqual(len(tiling.regions[0].request.outputs), 2)
        self.assertEqual(tiling.record()["source_request_digest"], request.digest())
        for mutated in (replace(request, target_identity="other-target"),
                        change_node(request, "result", attrs=(("axis", 0), ("epsilon_bf16_bits", 0x3728))),
                        change_node(request, "result", type=replace(request.nodes[-1].type, numerical_policy="other-policy"))):
            with self.subTest(mutation=mutated.digest()), self.assertRaises(ValueError):
                prepare_rmsnorm(mutated)
        with self.assertRaises(ValueError):
            prepare_rmsnorm(request_for((2, 33)))

    def test_source_physical_representation_cannot_be_silently_replaced(self):
        source = source_for()
        boundaries = tuple(node.id for node in source.nodes if node.effect in {"input", "constant"})
        physical = replace(source, physical_extents=tuple((key, 1) for key in boundaries))
        layouts = replace(physical, boundary_layouts=tuple((key, "selected-layout") for key in boundaries),
                          output_layouts=("selected-layout",))
        for mutated in (physical, layouts):
            with self.subTest(layouts=bool(mutated.boundary_layouts)), self.assertRaises(ValueError):
                prepare_source(mutated)

    def test_macro_boundary_contracts_and_epsilon_bounds_refuse(self):
        request = request_for((2, 64))
        mutations = {}
        for key in ("data", "gamma"):
            mutations[key + " policy"] = change_node(request, key, type=replace(request.node(key).type, numerical_policy="other"))
            mutations[key + " attrs"] = change_node(request, key, attrs=(("unknown", "other-contract"),))
            mutations[key + " maps"] = change_node(request, key, index_maps=(pointwise_identity_maps(len(request.node(key).type.shape), 1)[0],))
        identity = pointwise_identity_maps(2, 1)[0]
        mutations["result maps"] = change_node(request, "result", index_maps=(identity, IndexMap(2, ((0, 1),), (0,)), identity))
        for code in (0x0001, 0x2f7f, 0x4781, 0x7f7f):
            mutations["epsilon " + hex(code)] = request_for((2, 64), epsilon_bf16_bits=code)
        for label, mutated in mutations.items():
            with self.subTest(mutation=label), self.assertRaises(ValueError):
                prepare_rmsnorm(mutated)
        for code in (0x2f80, 0x4780):
            with self.subTest(admitted_epsilon=hex(code)):
                self.assertEqual(prepare_rmsnorm(request_for((2, 64), epsilon_bf16_bits=code)).shape, (2, 64))

    def test_frozen_source_parameters_are_checked_before_precision_selection(self):
        source = source_for()
        for key in ("data", "gamma"):
            count = 1
            for extent in source.node(key).type.shape:
                count *= extent
            constant_source = replace(source, nodes=tuple(replace(node, op="constant", effect="constant") if node.id == key else node for node in source.nodes),
                                      input_storages=tuple((node, storage) for node, storage in source.input_storages if node != key),
                                      constants=(*source.constants, ConstantBinding(key, "f32-le-raw", struct.pack("<f", 1.0).hex() * count)))
            with self.subTest(valid_parameter=key):
                self.assertIn(key, prepare_source(constant_source).input_ids)
            for bits in (0x7fc00000, 0x2b800000, 0x48000000):
                with self.subTest(parameter=key, bits=hex(bits)), self.assertRaises(ValueError):
                    prepare_source(change_literal(constant_source, key, bits))

    def test_two_1024_groups_match_independent_tree_and_differ_from_ltr(self):
        # Every input is exactly129/128. Squaring rounds to65/64 and each
        # 32-column partial is32.5. The hierarchy must preserve its explicit
        # BF16 boundaries; an LTR chopped accumulator loses low bits.
        request = request_for((1, 2048))
        tiling = prepare_rmsnorm(request)
        body = tiling.regions[0].request
        vectors = {}
        for node in body.nodes:
            if node.id.startswith("data_"):
                vectors[node.id] = [0x3F81] * 32
            elif node.id.startswith("gamma_"):
                vectors[node.id] = [0x3F80] * 32
        for literal in body.constants:
            vectors[literal.node_id] = list(struct.unpack("<32H", bytes.fromhex(literal.data_hex)[:64]))
        observed_total = None
        for node in body.nodes:
            if node.id in vectors:
                continue
            args = [vectors[key] for key in node.inputs]
            if "inverse_width" in node.inputs:
                observed_total = args[0][0]
                break
            if node.op == "mul_bf16_zero":
                vectors[node.id] = [_encode_bf16(_decode(a) * _decode(b)) for a, b in zip(*args)]
            elif node.op == "add_bf16_zero":
                vectors[node.id] = [chop_bf16(_round_nearest_even(_decode(a) + _decode(b), 23)) for a, b in zip(*args)]
            elif node.op == "row_sum_bf16":
                vectors[node.id] = [_tree_row(args[0])] * 32
            else:
                self.fail("unexpected primitive before mean: " + node.op)
        squared = _encode_bf16(_decode(0x3F81) ** 2)
        partials = [_tree_row([squared] * 32)] * 64
        group_totals = [_tree_row(partials[:32]), _tree_row(partials[32:])]
        expected = _tree_row([*group_totals, *([0] * 30)])
        sequential = 0
        for partial in partials:
            sequential = chop_bf16(_round_nearest_even(_decode(sequential) + _decode(partial), 23))
        self.assertEqual(expected, 0x4502)
        self.assertEqual(observed_total, expected)
        self.assertNotEqual(sequential, expected)



if __name__ == "__main__":
    unittest.main()
