"""A fused pointwise chain retains its private result cast and all guards."""
import unittest
from dataclasses import replace
from pathlib import Path

from atlas_mlir_native import adjacent_pointwise_chain as native
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_adjacent_pointwise_chain import case
from test_pointwise_narrowing import adapter

ROOT = Path(__file__).resolve().parents[1]


def narrowed_case(*, rows=3, columns=35, divide=False):
    source, _, _ = case(rows=rows, columns=columns, divide=divide)
    result = source.nodes[-1]
    cast = SemanticNode('out', 'f32_to_bf16', (result.id,), replace(result.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(result.type.shape), 1))
    return replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,))


class PointwiseChainNarrowingTest(unittest.TestCase):
    def setUp(self):
        options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_selector_keeps_cast_with_the_complete_chain(self):
        source = narrowed_case()
        base = adapter()
        wrapper = AdjacentDeviceAdapter(base, patterns=('adjacent_pointwise_chain', 'pointwise_narrowing'))
        selected, = wrapper.plan(source).selected
        self.assertEqual(selected.nodes, (0, 1, 2, 3))
        self.assertEqual(dict(selected.metadata)['kind'], 'adjacent_pointwise_chain')
        self.assertEqual(tuple(port.buffer for port in selected.outputs), ('out',))
        group = native.prepare(source, base)
        ordinary, _, _ = case(columns=35)
        original = native.prepare(ordinary, base)
        for actual, expected in zip(group['plans'], original['plans'], strict=True):
            self.assertEqual(replace(actual, source=replace(actual.source, source_identity='')),
                             replace(expected, source=replace(expected.source, source_identity='')))
        for actual, expected in zip(group['precisions'], original['precisions'], strict=True):
            self.assertEqual(actual.operand_specs, expected.operand_specs)
            self.assertEqual(actual.source.nodes, expected.source.nodes)
        self.assertIn('output_narrowing', group['proof'])

    def test_escaping_or_changed_cast_cannot_be_removed(self):
        source = narrowed_case()
        escaped = replace(source, outputs=('multiply', 'out'), output_storages=('external', 'external'))
        wrapper = AdjacentDeviceAdapter(adapter(), patterns=('adjacent_pointwise_chain',))
        self.assertFalse(any(len(row.nodes) == 4 for row in wrapper.plan(escaped).selected))
        cast = source.nodes[-1]
        for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                        replace(cast, inputs=('x',)), replace(cast, type=source.nodes[-2].type)):
            with self.assertRaises(ValueError):
                native.prepare(replace(source, nodes=(*source.nodes[:-1], changed)), adapter())


if __name__ == '__main__':
    unittest.main()
