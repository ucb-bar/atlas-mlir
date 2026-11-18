"""Publish restored BF16 directly through an adjacent exact narrowing cast.

Successful quantized readout admits only finite normal BF16 values and signed
zero. Its FP32 publication is an exact widening of those bits. Narrowing that
private result back to BF16 therefore needs no additional arithmetic or guard.
"""
from dataclasses import replace
from functools import partial

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import pointwise_identity_maps

from .quantized_epilogue import adjacent_alternatives, compile_group, export_group as export_counted


def prepare_source(source):
    from .already_quantized import OPS, prepare_source as prepare_contraction
    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) != 2:
        raise ValueError('quantized narrowing requires exactly two source operations')
    contraction, cast = pure
    if (contraction.op not in OPS or contraction.type.dtype != 'f32'
            or cast.op != 'f32_to_bf16' or cast.inputs != (contraction.id,)
            or cast.type != replace(contraction.type, dtype='bf16')
            or dict(cast.attrs) != {'numeric_semantics': 'f32-to-bf16-rne-v1'}
            or cast.index_maps != pointwise_identity_maps(len(cast.type.shape), 1)
            or source.outputs != (cast.id,)):
        raise ValueError('quantized narrowing requires a private restored FP32 result and exact dense BF16 cast')
    plan = prepare_contraction(extract_semantic_region(source, (contraction.id,)))
    inputs = tuple(node.id for node in source.nodes if node.effect in ('input', 'constant'))
    if (len(inputs) != 4 or set(inputs) != set(plan.input_ids)
            or any(node.effect not in ('pure', 'input', 'constant') for node in source.nodes)):
        raise ValueError('quantized narrowing requires unchanged code/exponent ports')
    return replace(plan, source=source, region_id=cast.id, input_ids=inputs, output_dtype='bf16',
                   narrowing_input_id=contraction.id)


def alternatives(adapter, source, choices):
    return adjacent_alternatives(adapter, source, choices,
                                 kind='quantized_narrowing', consumer_kind='exact_float_device', prepare=prepare_source)


export_group = partial(export_counted, kind='quantized_narrowing')
