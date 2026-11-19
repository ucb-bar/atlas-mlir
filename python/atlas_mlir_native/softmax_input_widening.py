"""Consume original BF16 scores instead of materializing an exact f32 widening."""
from dataclasses import dataclass

from merlin.compile.input_widening import split_private_bf16_widening
from merlin.compile.semantic_regions import extract_semantic_region

from .composition_adapter import device_casts, fuse, kind_of, neighbor_pairs
from .softmax_narrowing import KINDS, compile_group, export_softmax


@dataclass(frozen=True)
class Widening:
    source: object
    producer: object
    precision: object
    kind: str
    input_id: str
    output_dtype: str
    score_domain: str
    finite_sentinel_bits: int | None


def prepare(source, adapter, kind, *, output_dtype='f32'):
    from .row_softmax import SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN
    from .softmax_precision import lower_semantic_softmax
    from .softmax_narrowing import prepare as prepare_narrowing

    if kind not in KINDS or adapter.already_quantized_score_domain not in (SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN):
        raise ValueError('softmax input widening needs a supported bounded score policy')
    split = split_private_bf16_widening(source)
    if output_dtype == 'bf16':
        narrowed = prepare_narrowing(split.consumer, adapter, kind)
        producer, precision = narrowed.producer, narrowed.precision
    elif output_dtype == 'f32':
        producer = split.consumer
        precision = lower_semantic_softmax(producer, allow_masked=kind != 'softmax',
            allow_guarded=kind == 'guarded_softmax', score_domain=adapter.already_quantized_score_domain)
    else:
        raise ValueError('unsupported softmax publication dtype')
    if precision.source_region.scores != split.widened_id:
        raise ValueError('the private widening must supply the softmax score boundary')
    # The original transport exactly widens BF16 then rounds it back to BF16
    # before checking these finite code ranges. Direct code validation is the
    # same operation, including signed zeros and rejection before device I/O.
    return Widening(source, producer, precision, kind, split.input_id, output_dtype,
        adapter.already_quantized_score_domain,
        adapter.already_quantized_mask_sentinel if kind == 'masked_softmax' else None)


def alternatives(adapter, source, choices):
    pure = [node for node in source.nodes if node.effect == 'pure']

    def prepared(region, rows):
        kind, output_dtype = kind_of(rows[1]), 'f32'
        if kind == 'softmax_narrowing':
            previous = extract_semantic_region(source, tuple(pure[index].id for index in rows[1].nodes))
            kind, output_dtype = adapter._softmax_narrowing_groups[previous.digest()].kind, 'bf16'
        return prepare(region, adapter, kind, output_dtype=output_dtype)
    return fuse(adapter, source, choices, '_softmax_input_widening_groups', 'softmax_input_widening',
                neighbor_pairs(choices, device_casts(choices), lambda row: kind_of(row) in (*KINDS, 'softmax_narrowing'),
                               before=True), prepare=prepared, base=1,
                cycles='unchanged row-softmax program; private exact widening omitted')


def export_group(group, compiled, name, *, target_source):
    return export_softmax(group, compiled, name, target_source=target_source, scores=group.input_id,
        input_dtype='bf16', output_dtype=group.output_dtype, outputs=group.source.outputs,
        programs=('segments/' + name,), kind='softmax_input_widening', source_request_digest=group.source.digest(),
        widening_proof='finite BF16 score codes; exact widening then RNE preserves every admitted code')
