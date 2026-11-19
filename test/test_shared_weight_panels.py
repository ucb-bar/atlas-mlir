"""Tile-aligned independent batches share weights without changing source math."""
from atlas_mlir_native.options import codegen_options, current_codegen_options
from dataclasses import replace
from math import prod

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_panel_geometry import panel_geometry, shared_weight_panel_plan

from test_already_quantized import source_request


def source_for(batch, m=64, n=35, k=257, *, scales='row'):
    source = source_request(m=m, n=n, k=k, batch=batch)
    shape = {'scalar': (), 'row': (*batch, m, 1), 'batch': (*batch, 1, 1)}[scales]
    return replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=shape))
                   if node.id == 'sx' else node for node in source.nodes))


@pytest.mark.parametrize('batch', [(2,), (4,), (2, 4)])
@pytest.mark.parametrize('scales', ['scalar', 'row'])
def test_only_physical_traversal_changes(batch, scales):
    plan = prepare_source(source_for(batch, scales=scales))
    physical = shared_weight_panel_plan(plan)
    assert physical.source is plan.source
    assert physical.m == prod(batch) * plan.m and physical.batch_shape == ()
    assert physical.operand_ids == plan.operand_ids and physical.exponent_ids == plan.exponent_ids
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
        layout = execution_layout(plan, operand_reuse='panel-cache')
        assert layout['operand_reuse_flattened_batches'] == prod(batch)
        with pytest.raises(ValueError, match='schedule differs'):
            virtual_source(plan, {**layout, 'operand_reuse_flattened_batches': 1})


def test_tail_batch_scales_and_batched_rhs_keep_original_traversal():
    for source in (source_for((2,), m=63), source_for((2,), scales='batch'), source_for(())):
        plan = prepare_source(source)
        assert shared_weight_panel_plan(plan) is plan
    source = source_for((2,))
    source = replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=(2, *node.type.shape)))
                     if node.id in ('w', 'sw') else node for node in source.nodes))
    # The checked source matcher permits a batched RHS for attention-style contractions.
    source = replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], op='atlas.qk_matmul')))
    plan = prepare_source(source)
    assert shared_weight_panel_plan(plan) is plan and panel_geometry(plan) is None
