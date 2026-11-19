"""Pack private BF16 softmax results in their producing row program.

Merlin extracts the dense region and its escaping values. Atlas proves the
selected softmax arithmetic and uses its ordinary exact static PACK lowering.
Dynamic scales and shared probability tensors retain the separate regions.
"""
from dataclasses import dataclass

from merlin.compile.semantic_regions import extract_semantic_region

from .composition_adapter import (check_source_request, device_quantizers, fuse, kind_of, neighbor_pairs,
                                  write_source_request)
from .softmax_narrowing import export_softmax


@dataclass(frozen=True)
class QuantizedSoftmax:
    source: object
    softmax: object
    quantizer: object
    exponent: int
    input_dtype: str


def prepare(source, adapter, kind, *, input_dtype='f32'):
    from .static_quantize import prepare_source
    from .softmax_narrowing import prepare as narrowing
    from .softmax_input_widening import prepare as widening

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) < 3 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('softmax quantization requires a dense private producer')
    quantizer = extract_semantic_region(source, tuple(node.id for node in pure[-2:]))
    x, codes, exponent, fixed = prepare_source(quantizer)
    producer = extract_semantic_region(source, tuple(node.id for node in pure[:-2]))
    if (x.type.dtype != 'bf16' or producer.outputs != (x.id,)
            or source.outputs != (codes.id, exponent.id)):
        raise ValueError('softmax probability must be private to both static quantizer projections')
    if input_dtype == 'bf16':
        softmax = widening(producer, adapter, kind, output_dtype='bf16')
    elif input_dtype == 'f32':
        softmax = narrowing(producer, adapter, kind)
    else:
        raise ValueError('unsupported softmax score transport')
    return QuantizedSoftmax(source, softmax, quantizer, fixed, input_dtype)


def alternatives(adapter, source, choices):
    pure = [node for node in source.nodes if node.effect == 'pure']

    def prepared(region, rows):
        flavor = kind_of(rows[0])
        previous = extract_semantic_region(source, tuple(pure[index].id for index in rows[0].nodes))
        prior = getattr(adapter, '_' + flavor + '_groups')[previous.digest()]
        if flavor == 'softmax_input_widening' and prior.output_dtype != 'bf16':
            raise ValueError('quantized softmax needs a BF16 probability')
        return prepare(region, adapter, prior.kind, input_dtype='bf16' if flavor == 'softmax_input_widening' else 'f32')
    accepts = lambda row: kind_of(row) in ('softmax_narrowing', 'softmax_input_widening')
    return fuse(adapter, source, choices, '_softmax_quantize_groups', 'softmax_quantize',
                neighbor_pairs(choices, device_quantizers(adapter, choices), accepts), prepare=prepared,
                cycles='static PACK follows the unchanged BF16 softmax result in VMEM')


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .row_loop import compile_row_loop

    destination = write_source_request(destination, group.source)
    compile_row_loop(snapshot, group.softmax.precision.request, target_source=target_source,
        destination=destination / 'row', limits=limits, static_output_exponent=group.exponent)


def export_group(group, compiled, name, *, target_source):
    compiled = check_source_request(compiled, group.source, 'quantized softmax publication differs from its source')
    softmax = group.softmax
    return export_softmax(softmax, compiled / 'row', name, target_source=target_source,
        scores=softmax.input_id if group.input_dtype == 'bf16' else None, input_dtype=group.input_dtype,
        output_dtype='i8', static_output_exponent=group.exponent, outputs=group.source.outputs,
        programs=('segments/' + name + '/row',), kind='softmax_quantize', source_request_digest=group.source.digest(),
        static_exponent=group.exponent,
        quantization_proof='unchanged rounded BF16 probability; ordinary static raw PACK; authored exponent')
