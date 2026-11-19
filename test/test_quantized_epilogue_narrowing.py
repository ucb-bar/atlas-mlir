"""Keep ordered contraction and product semantics across a private result cast."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.already_quantized import prepare_source as prepare_dispatch
from atlas_mlir_native.quantized_epilogue import alternatives as epilogues, prepare_source as prepare_epilogue
from atlas_mlir_native.quantized_epilogue_narrowing import alternatives, prepare_source
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, ValueRepresentation
from test_quantized_epilogue import case as epilogue_case


def case(operand=0, **geometry):
    producer, _, choices = epilogue_case('f32', operand, **geometry)
    product = producer.nodes[-1]
    cast = SemanticNode('narrowed', 'f32_to_bf16', (product.id,), replace(product.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(product.type.shape), 1))
    source = replace(producer, nodes=(*producer.nodes, cast), outputs=(cast.id,))
    last = replace(choices[-1], id='cast', nodes=(2,), inputs=choices[-1].outputs,
        outputs=(BufferRepresentation(cast.id, ValueRepresentation('host_memory', 'dense', 'bf16')),),
        metadata=(('kind', 'exact_float_device'),))
    return source, producer, (*choices, last)


@pytest.mark.parametrize('operand', [0, 1])
def test_preserve_product_order_and_intake(operand):
    source, producer, _ = case(operand)
    old, plan = prepare_epilogue(producer), prepare_dispatch(source)
    assert plan.epilogue_contract_operand == old.epilogue_contract_operand == operand
    assert plan.input_ids == old.input_ids
    assert plan.reset.nodes == old.reset.nodes and plan.continuation.nodes == old.continuation.nodes
    assert plan.epilogue_precision.request.nodes == old.epilogue_precision.request.nodes
    assert plan.source == source and plan.output_id == 'narrowed' and plan.output_dtype == 'bf16'
    assert plan.narrowing_input_id == producer.outputs[0]
    assert 'epilogue product' in plan.record()['narrowing']['admission']


def test_selection_refusals_and_tool_errors_remain_visible():
    source, _, choices = case()
    adapter = SimpleNamespace()
    admitted = SimpleNamespace(record=lambda: dict(route='counted', counted_frame_bytes=65536))
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted', return_value=admitted):
        choices = epilogues(adapter, source, choices)
        selected = alternatives(adapter, source, choices)[-1]
        assert selected.nodes == (0, 1, 2)
        assert dict(selected.metadata)['kind'] == 'quantized_epilogue_narrowing'
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted',
               return_value=SimpleNamespace(record=lambda: dict(route='k32_tiles'))):
        assert alternatives(adapter, source, choices) == choices
    with patch('atlas_mlir_native.quantized_loop_sizing.size_counted',
               side_effect=NativeCompilationError('compile_error', 'test tool failure')):
        with pytest.raises(NativeCompilationError, match='test tool failure'):
            alternatives(adapter, source, choices)
    for changed in (replace(source, outputs=('product', 'narrowed'), output_storages=('external',)*2),
                    replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], attrs=())))):
        with pytest.raises(ValueError):
            prepare_source(changed)
