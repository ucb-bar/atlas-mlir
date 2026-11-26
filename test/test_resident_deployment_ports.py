"""C boundary references may repeat and include compiler-owned constants."""
from dataclasses import replace

import pytest

from atlas_mlir_native.resident_deployment import _checked_source_ports
from merlin.compiler.semantic.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def source():
    tensor = TensorType((1,), 'f32', 'raw-bit-pattern')
    nodes = (SemanticNode('x', 'input', (), tensor, effect='input'),
             SemanticNode('literal', 'constant', (), tensor, effect='constant'),
             SemanticNode('intermediate', 'f32_mul', ('x', 'x'), tensor,
                 (('numeric_semantics', 'ieee754-f32-mul-rne-v1'),), index_maps=pointwise_identity_maps(1, 2)),
             SemanticNode('out', 'f32_add', ('intermediate', 'literal'), tensor,
                 (('numeric_semantics', 'ieee754-f32-add-rne-v1'),), index_maps=pointwise_identity_maps(1, 2)))
    return KernelRequest(nodes, ('out',), ('external',), (('x', 'external'),), 'synthetic',
                         constants=(ConstantBinding('literal', 'f32-le-raw', '0000803f'),))


@pytest.mark.parametrize('inputs', [['x'], ['x', 'x'], ['x', 'literal'], ['literal', 'x', 'x', 'literal']])
def test_runtime_coverage_accepts_duplicate_and_constant_boundary_references(inputs):
    _checked_source_ports(source(), inputs, ['out'])


@pytest.mark.parametrize('inputs', [[], ['literal'], ['x', 'unknown'], ['x', 'intermediate'], ['x', 'out']])
def test_missing_runtime_unknown_and_pure_references_refuse(inputs):
    with pytest.raises(ValueError, match='ports differ'):
        _checked_source_ports(source(), inputs, ['out'])


def test_outputs_retain_exact_order_and_count():
    current = replace(source(), outputs=('out', 'intermediate'), output_storages=('external', 'external'))
    _checked_source_ports(current, ['x', 'literal'], ['out', 'intermediate'])
    for outputs in ([], ['out'], ['intermediate', 'out'], ['out', 'out']):
        with pytest.raises(ValueError, match='ports differ'):
            _checked_source_ports(current, ['x'], outputs)
