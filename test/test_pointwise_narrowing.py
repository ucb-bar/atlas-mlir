"""Keep pointwise arithmetic and guards while eliminating its private cast."""
import unittest
from dataclasses import replace

import numpy as np
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise_narrowing import prepare
from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from test_application_float_arithmetic import policy


def case(op='mul', shape=(3, 35)):
    dtype = TensorType(shape, 'f32', 'raw-bit-pattern')
    a, b = (SemanticNode(name, 'input', (), dtype, effect='input') for name in ('a', 'b'))
    value = SemanticNode('y', 'f32_' + op, ('a', 'b'), dtype,
        (('numeric_semantics', 'ieee754-f32-' + op + '-rne-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 2))
    cast = SemanticNode('out', 'f32_to_bf16', ('y',), replace(dtype, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((a, b, value, cast), ('out',), ('external',),
        (('a', 'external'), ('b', 'external')), TARGET_IDENTITY, 'diagnostic')


def adapter(**options):
    from atlas_mlir_native.already_quantized import POLICY
    selected = {**policy(), "already_quantized": POLICY, **options}
    return AtlasApplicationAdapter(selected, original_source_sha256=selected['source_sha256'])


class PointwiseNarrowingTest(unittest.TestCase):
    def test_real_selector_and_unchanged_selected_operation(self):
        for operation in ('mul', 'div'):
            source = case(operation)
            base = adapter()
            ordinary = base.plan(source).selected
            self.assertEqual(len(ordinary), 2)
            wrapper = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',))
            selected, = wrapper.plan(source).selected
            self.assertEqual(dict(selected.metadata)['kind'], 'pointwise_narrowing')
            group = prepare(source, base)
            from atlas_mlir_native.pointwise import prepare_source
            from merlin.compile.semantic_regions import extract_semantic_region
            original = extract_semantic_region(source, ('y',))
            expected = prepare_source(original, allow_math=True)
            self.assertEqual(group.precision.request, expected.request)
            self.assertEqual(group.precision.operand_specs, expected.operand_specs)

    def test_declines_escaping_result_changed_rounding_and_unsupported_policy(self):
        source = case()
        base = adapter()
        wrapper = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',))
        escaped = replace(source, outputs=('y', 'out'), output_storages=('external', 'external'))
        self.assertEqual(len(wrapper.plan(escaped).selected), 2)
        cast = source.nodes[-1]
        for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                        replace(cast, inputs=('a',)), replace(cast, type=source.nodes[0].type)):
            with self.assertRaises(ValueError):
                prepare(replace(source, nodes=(*source.nodes[:-1], changed)), base)
        with self.assertRaises(ValueError):
            prepare(escaped, base)
        for patterns in (('unknown',), ('pointwise_narrowing', 'pointwise_narrowing')):
            with self.assertRaises(ValueError):
                AdjacentDeviceAdapter(base, patterns=patterns)
        self.assertEqual(len(AdjacentDeviceAdapter(base, patterns=()).plan(source).selected), 2)

    def test_non_nan_bf16_identity_includes_subnormals_infinities_and_signed_zero(self):
        from test_exact_float_device import host_reference, request
        codes = np.arange(65536, dtype='<u2')
        codes = codes[(codes & 0x7fff) <= 0x7f80]
        widened = (codes.astype('<u4') << 16).tobytes()
        actual, status = host_reference(request('f32_to_bf16', len(codes)), [widened])
        self.assertEqual(status, 0)
        self.assertEqual(actual, codes.tobytes())


if __name__ == '__main__':
    unittest.main()
