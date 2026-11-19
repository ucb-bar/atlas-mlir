"""Pack private pointwise BF16 results before they leave the device.

Merlin owns region extraction and escaping-value checks. Atlas retains the
producer's selected numerical policy, rounding and input guards, then applies
its ordinary static raw PACK to the completed tile. Dynamic quantizers and
shared BF16 results retain their separate implementations.
"""
from dataclasses import dataclass

from merlin.compile.semantic_regions import extract_semantic_region

from .composition_adapter import (check_source_request, device_quantizers, fuse, kind_of, neighbor_pairs,
                                  write_source_request)
from .pointwise_narrowing import compile_request, export_precision


@dataclass(frozen=True)
class QuantizedPointwise:
    source: object
    precision: object
    exponent: int
    input_ids: tuple[str, ...]
    input_dtype: str | None


def prepare(source, adapter, kind):
    from .static_quantize import prepare_source as quantizer_source
    from .pointwise import prepare_source, prepare_execution_plan, plan_abi
    from .pointwise_narrowing import prepare as narrowing
    from .pointwise_input_widening import prepare as widening

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) < 3 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('pointwise quantization requires a dense private producer')
    quantizer = extract_semantic_region(source, tuple(node.id for node in pure[-2:]))
    x, codes, exponent, fixed = quantizer_source(quantizer)
    producer = extract_semantic_region(source, tuple(node.id for node in pure[:-2]))
    if (x.type.dtype != 'bf16' or producer.outputs != (x.id,)
            or source.outputs != (codes.id, exponent.id)):
        raise ValueError('pointwise result must be private to both static quantizer projections')
    inputs = tuple(node.id for node in producer.nodes if node.effect in ('input', 'constant'))
    input_dtype = None
    if kind == 'pointwise':
        precision = prepare_source(producer, bf16_policy=adapter.bf16_pointwise)
    elif kind == 'pointwise_narrowing':
        precision = narrowing(producer, adapter).precision
    elif kind == 'pointwise_input_widening':
        previous = widening(producer, adapter, output_dtype='bf16')
        precision, inputs, input_dtype = previous.precision, (previous.input_id,), 'bf16'
    else:
        raise ValueError('unsupported pointwise quantization producer')
    if prepare_execution_plan(precision.request)[2]:
        raise ValueError('packed pointwise output requires a single resident frame')
    plan_abi(precision.request)
    return QuantizedPointwise(source, precision, fixed, inputs, input_dtype)


def alternatives(adapter, source, choices):
    accepts = lambda row: kind_of(row) in ('pointwise', 'pointwise_narrowing', 'pointwise_input_widening')
    return fuse(adapter, source, choices, '_pointwise_quantize_groups', 'pointwise_quantize',
                neighbor_pairs(choices, device_quantizers(adapter, choices), accepts),
                prepare=lambda region, rows: prepare(region, adapter, kind_of(rows[0])),
                inputs=lambda group: group.input_ids,
                cycles='static PACK follows the unchanged BF16 pointwise result in VMEM')


def compile_group(group, snapshot, *, target_source, destination, limits):
    destination = write_source_request(destination, group.source)
    compile_request(snapshot, group.precision.request, target_source=target_source, destination=destination / 'pointwise',
                    limits=limits, static_output_exponent=group.exponent)


def export_group(group, compiled, name, *, target_source):
    compiled = check_source_request(compiled, group.source, 'quantized pointwise publication differs from its source')
    return export_precision(group.precision, compiled / 'pointwise', name, target_source=target_source,
        input_dtype=group.input_dtype, output_dtype='i8', static_output_exponent=group.exponent,
        inputs=group.input_ids, outputs=group.source.outputs, programs=('segments/' + name + '/pointwise',),
        launches=1, kind='pointwise_quantize', input_node_ids=list(group.input_ids),
        output_node_ids=list(group.source.outputs), source_request_digest=group.source.digest(),
        quantization_proof='unchanged rounded BF16 result; ordinary static raw PACK; authored exponent',
        static_exponent=group.exponent)
