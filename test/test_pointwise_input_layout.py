"""Pointwise affine-read folding preserves code validation and selected results."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atlas_mlir_native import pointwise_input_layout as fused
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from merlin.semantic_compiler.model import SemanticNode, IndexMap, pointwise_identity_maps
from test_pointwise_quantize import case as quantized_case
from test_bf16_pointwise import request as pointwise_case
from test_pointwise_narrowing import adapter as base_adapter

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ('pointwise_quantize', 'pointwise_input_layout')


def adapter():
    from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
    return base_adapter(bf16_pointwise=BF16_POINTWISE_POLICY, host_support=['transpose','reshape','copy','gather'])


def case(shape=(33, 35), kind='broadcast', packed=False):
    source = quantized_case(shape) if packed else pointwise_case(shape)
    a, b, *operations = source.nodes
    m,n = int(np.prod(shape[:-1])),shape[-1]
    rank = len(shape)
    physical, mapping = {
        'broadcast': ((n,), IndexMap(rank, (tuple([0]*(rank-1)+[1]),), (0,))),
        'transpose': ((n,m), IndexMap(2, ((0,1),(1,0)), (0,0))),
        'reverse': (shape, IndexMap(2, ((1,0),(0,-1)), (0,n-1))),
        'slice': ((m,n+1), IndexMap(2, ((1,0),(0,1)), (0,0))),
    }[kind]
    base = replace(b, id='base', type=replace(b.type, shape=physical))
    view = SemanticNode(b.id, 'gather', ('base',), b.type,
                        index_maps=(mapping, pointwise_identity_maps(rank,1)[-1]))
    return replace(source, nodes=(a,base,view,*operations), input_storages=(('a','external'),('base','external')))


@pytest.mark.parametrize('kind', ['broadcast','transpose','reverse'])
@pytest.mark.parametrize('packed', [False, True])
def test_real_selector_preserves_affine_semantics(kind, packed):
    source = case(kind=kind, packed=packed)
    group = fused.prepare(source, adapter(), 'pointwise_quantize' if packed else 'pointwise')
    selected, = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_input_layout'
    assert group.consumer.outputs == source.outputs
    assert group.consumer.node('result').inputs == ('a','base')
    assert len(group.consumer.nodes) == len(source.nodes)-1


def test_slices_and_shared_views_retain_materialization():
    source = case()
    public = replace(source, outputs=(*source.outputs,'b'), output_storages=('external',)*2)
    for value in (case(kind='slice'), public):
        with pytest.raises(ValueError):
            fused.prepare(value, adapter(), 'pointwise')
        assert all(dict(r.metadata)['kind'] != 'pointwise_input_layout'
                   for r in AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(value).selected)


def test_selected_rounding_policy_is_never_replaced_to_accept_a_layout():
    from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY
    source = case()
    nodes = tuple(replace(node, type=replace(node.type, dtype='f32')) for node in source.nodes)
    addition = replace(nodes[-1], op='f32_add', attrs=(('numeric_semantics','ieee754-f32-add-rne-v1'),))
    source = replace(source, nodes=(*nodes[:-1],addition))
    base = adapter()
    base.qualified_rne_add = RNE_ADD_POLICY
    # This qualified producer admits only identity maps. The unqualified
    # chopped-add implementation can accept broadcasts, but is not a fallback.
    with pytest.raises(ValueError, match='RNE add'):
        fused.prepare(source, base, 'pointwise')


def test_source_indexing_is_replayed_before_compilation_or_export(tmp_path):
    group = fused.prepare(case(), adapter(), 'pointwise')
    changed = replace(group.consumer, outputs=('base',))
    forged = replace(group, consumer=changed)
    with pytest.raises(ValueError, match='composed source indexing'):
        fused.compile_group(forged, None, target_source=ROOT, destination=tmp_path/'compiled', limits=None)
    assert not (tmp_path/'compiled').exists()
    with pytest.raises(ValueError, match='composed source indexing'):
        fused.export_group(forged, tmp_path/'absent', 'pointwise', target_source=ROOT)
