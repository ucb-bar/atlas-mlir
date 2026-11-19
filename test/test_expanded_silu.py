"""A closed BF16 SiLU expansion retains selected arithmetic and rejection."""
import unittest
from dataclasses import replace

import numpy as np
from atlas_mlir_native import expanded_silu as native
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_adjacent_pointwise import case as exp_case
from test_pointwise_narrowing import adapter as base_adapter


def case(shape=(3, 35)):
    source, _ = exp_case(shape=shape)
    x, neg, literal, ones, exp, add, divide = source.nodes
    x = replace(x, type=replace(x.type, dtype='bf16'))
    widen = SemanticNode('wide', 'bf16_to_f32', ('x',), divide.type,
        (('numeric_semantics', 'bf16-to-f32-exact-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    neg = replace(neg, op='f32_neg', inputs=('wide',), effect='pure',
        attrs=(('numeric_semantics', 'f32-sign-bit-flip-v1'),), index_maps=widen.index_maps)
    divide = replace(divide, inputs=('wide', add.id))
    out = SemanticNode('out', 'f32_to_bf16', (divide.id,), x.type,
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=widen.index_maps)
    return replace(source, nodes=(x, literal, widen, ones, neg, exp, add, divide, out),
        outputs=('out',), input_storages=(('x', 'external'),))


def adapter():
    from atlas_mlir_native.source_exp import POLICY
    return base_adapter(source_exp=POLICY, host_float_arithmetic=['f32_neg'])


def expanded_reference(codes):
    from atlas_mlir_native.exp_lane import vexp_bf16_bits
    from atlas_mlir_native.recip_lane import vrecip_bf16_bounded_bits
    from atlas_mlir_native.reference import _f32_add_rne, _round_bf16_product

    result = []
    for code in codes:
        # Sign flip, unlike the macro's zero subtraction, reverses both zeros.
        exp = vexp_bf16_bits(int(code) ^ 0x8000)
        denominator = _f32_add_rne(0x3f800000, exp << 16) >> 16
        reciprocal = vrecip_bf16_bounded_bits(denominator)
        value = _round_bf16_product(int(code), reciprocal) if code & 0x7fff else int(code)
        result.append(value)
    return np.array(result, dtype='<u2')


class ExpandedSiluTest(unittest.TestCase):
    def setUp(self):
        options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_selector_uses_one_closed_region_and_preserves_source(self):
        source = case()
        before = source.record()
        wrapper = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing', 'expanded_silu'))
        selected, = wrapper.plan(source).selected
        self.assertEqual(selected.nodes, tuple(range(7)))
        self.assertEqual(dict(selected.metadata)['kind'], 'expanded_silu')
        self.assertEqual(tuple(p.buffer for p in selected.inputs), ('x', 'literal'))
        self.assertEqual(tuple(p.buffer for p in selected.outputs), ('out',))
        self.assertEqual(source.record(), before)
        escaped = replace(source, outputs=('exp', 'out'), output_storages=('external',) * 2)
        self.assertFalse(any(len(row.nodes) == 7 for row in wrapper.plan(escaped).selected))

    def test_changed_policies_literal_edges_rounding_and_maps_decline(self):
        source = case()
        base = adapter()
        for field, value in (('pointwise', False), ('pointwise_math', False), ('sigmoid', False),
                             ('source_exp_policy', None), ('qualified_rne_add', 'another-policy')):
            changed = adapter()
            setattr(changed, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                native.prepare(source, changed)
        for index, node in enumerate(source.nodes):
            if node.effect == 'pure' and node.op != 'gather':
                for changed in (replace(node, attrs=()), replace(node, index_maps=())):
                    with self.subTest(node=node.id), self.assertRaises(ValueError):
                        native.prepare(replace(source, nodes=(*source.nodes[:index], changed,
                                                             *source.nodes[index + 1:])), base)
        for changed in (
            replace(source, constants=(replace(source.constants[0], data_hex='00000040'),)),
            replace(source, nodes=(*source.nodes[:-2], replace(source.nodes[-2],
                inputs=('neg', 'add')), source.nodes[-1])),
            replace(source, outputs=('wide', 'out'), output_storages=('external',) * 2),
        ):
            with self.assertRaises(ValueError):
                native.prepare(changed, base)

    def test_exhaustive_bf16_admission_and_selected_arithmetic(self):
        from atlas_mlir_native.pointwise_math import input_specs
        from atlas_mlir_native.reference import evaluate_silu
        from atlas_mlir_native.source_exp import input_spec as exp_spec
        from merlin.semantic_compiler.reference import TensorValue

        group = native.prepare(case(), adapter())
        admitted = lambda code, spec: any(lo <= code <= hi for lo, hi in spec.admitted_code_ranges)
        codes = []
        for code in range(65536):
            original = admitted(code ^ 0x8000, exp_spec()) and admitted(code, input_specs('divide_bf16_lut')[0])
            self.assertEqual(admitted(code, group.operand_specs[0]), original, hex(code))
            if original:
                codes.append(code)
        self.assertEqual(len(codes), 17412)
        group = native.prepare(case((1, len(codes))), adapter())
        result = evaluate_silu(group.request.nodes[-1], (TensorValue(group.request.nodes[0].type, tuple(codes)),))
        np.testing.assert_array_equal(result.elements, expanded_reference(codes))


if __name__ == '__main__':
    unittest.main()
