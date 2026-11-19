"""Consume private exact input casts through unchanged pointwise encoders."""
from dataclasses import dataclass

from merlin.compile.input_widening import split_private_bf16_widening

from .composition_adapter import contiguous, device_casts, fuse, kind_of, neighbor_pairs
from .pointwise_narrowing import compile_group, export_precision


@dataclass(frozen=True)
class Widening:
    source: object
    precision: object
    input_id: str
    output_dtype: str


def prepare(source, adapter, *, output_dtype='f32'):
    from .pointwise import prepare_source, plan_abi
    from .pointwise_narrowing import prepare as prepare_narrowing

    split = split_private_bf16_widening(source)
    if output_dtype == 'bf16':
        precision = prepare_narrowing(split.consumer, adapter).precision
    elif output_dtype == 'f32':
        result = split.consumer.node(split.consumer.outputs[0])
        precision = prepare_source(split.consumer, allow_math=True,
            bf16_policy=adapter.bf16_pointwise,
            rne_add_policy=adapter.qualified_rne_add if result.op == 'f32_add' else None,
            exp_policy=adapter.source_exp_policy)
    else:
        raise ValueError('unsupported pointwise output dtype')
    inputs = tuple(node.id for node in precision.source.nodes if node.effect in ('input', 'constant'))
    if inputs != (split.widened_id,):
        raise ValueError('direct pointwise codes require one private widened input boundary')
    # Existing BF16 encoders check every raw code before device I/O, preserving
    # the same ranges and zero policy as exact widening followed by f32 encode.
    # Their renderer refuses rounding, clipping and rescaling flags for raw
    # transport. Native arithmetic and output checks remain unchanged.
    from merlin.compile.c_conversions import render_bf16_codes_c
    from merlin.runtime.fp8_formats import float_format_params
    for spec in precision.operand_specs:
        exponent_bits, _, _, _ = float_format_params(spec.format)
        if spec.max_normal_exponent_code not in (None, (1 << exponent_bits)-2):
            raise ValueError('direct codes cannot replace a clamping f32 encoder')
        render_bf16_codes_c(spec, 1, 'proof')
    plan_abi(precision.request)
    return Widening(source, precision, split.input_id, output_dtype)


def alternatives(adapter, source, choices):
    accepts = lambda row: kind_of(row) in ('pointwise', 'pointwise_math', 'pointwise_narrowing') and contiguous(row)
    prepared = lambda region, rows: prepare(region, adapter,
        output_dtype='bf16' if kind_of(rows[1]) == 'pointwise_narrowing' else 'f32')
    return fuse(adapter, source, choices, '_pointwise_input_widening_groups', 'pointwise_input_widening',
                neighbor_pairs(choices, device_casts(choices), accepts, before=True), prepare=prepared, base=1,
                inputs=lambda group: (group.input_id,),
                cycles='unchanged pointwise program; private input widening omitted')


def export_group(group, compiled, name, *, target_source):
    return export_precision(group.precision, compiled, name, target_source=target_source, input_dtype='bf16',
        output_dtype=group.output_dtype, inputs=(group.input_id,), outputs=group.source.outputs,
        programs=('segments/' + name,), manifest=True, kind='pointwise_input_widening',
        input_node_ids=[group.input_id], output_node_ids=list(group.source.outputs),
        source_request_digest=group.source.digest(),
        widening_proof='raw BF16 validation equals exact widening then the original encoder')
