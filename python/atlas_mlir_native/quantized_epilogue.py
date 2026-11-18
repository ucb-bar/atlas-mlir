"""Source-bound multiplication after unchanged-code quantized contractions.

Only the adjacent, dense, lane-wise product is admitted. The contraction's
code/exponent intake and ordered K arithmetic stay unchanged; restoration
precedes checked BF16 multiplication with the original operand order.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .composition_adapter import c_region, device_runs, fuse, kind_of


def prepare_source(source):
    from .already_quantized import OPS, prepare_source as prepare_contraction
    from .pointwise import BF16_POINTWISE_POLICY, prepare_source as prepare_pointwise

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) != 2:
        raise ValueError('quantized epilogue requires exactly two source operations')
    contraction, product = pure
    if (contraction.op not in OPS or product.op not in ('f32_mul', 'bf16_mul')
            or len(product.inputs) != 2 or product.inputs.count(contraction.id) != 1
            or product.type != contraction.type or source.outputs != (product.id,)
            or product.index_maps != pointwise_identity_maps(len(product.type.shape), 2)):
        raise ValueError('quantized epilogue requires one dense nonescaping contraction result and its product')
    operand = product.inputs.index(contraction.id)
    other_id = product.inputs[1 - operand]
    other = source.node(other_id)
    if other.type != product.type or other.effect not in ('input', 'constant'):
        raise ValueError('quantized epilogue other operand must be a complete matching source boundary')
    plan = prepare_contraction(extract_semantic_region(source, (contraction.id,)))
    precision = prepare_pointwise(extract_semantic_region(source, (product.id,)),
        bf16_policy=BF16_POINTWISE_POLICY if product.type.dtype == 'bf16' else None)
    if (precision.request.nodes[-1].op != 'mul_bf16_zero'
            or len(precision.request.nodes[-1].inputs) != 2
            or precision.record()['inputs'] != list(product.inputs)):
        raise ValueError('quantized epilogue requires the original selected bounded BF16 RNE product')
    inputs = tuple(node.id for node in source.nodes if node.effect in ('input', 'constant'))
    if len(inputs) != 5 or set(inputs) != {*plan.input_ids, other_id}:
        raise ValueError('quantized epilogue requires unchanged code/exponent ports and one extra operand')
    return replace(plan, source=source, region_id=product.id, input_ids=inputs,
                   epilogue_precision=precision, epilogue_contract_operand=operand,
                   epilogue_other_id=other_id, epilogue_other_dtype=other.type.dtype)


def alternatives(adapter, source, choices):
    return adjacent_alternatives(adapter, source, choices,
                                 kind='quantized_epilogue', consumer_kind='pointwise', prepare=prepare_source)


def counted_scratch(plan):
    """Choice fields of a counted-route plan, or None for a proved construction refusal.

    Tool failures and unproved bounds raise; only a proved refusal declines.
    """
    from .quantized_loop_sizing import size_counted

    sizing = size_counted(plan).record()
    return dict(scratch_bytes=sizing['counted_frame_bytes']) if sizing['route'] == 'counted' else None


def adjacent_alternatives(adapter, source, choices, *, kind, consumer_kind, prepare):
    """Offer a recognized quantized group only after counted sizing succeeds."""
    pairs = (rows for rows in device_runs(choices, 2)
             if [kind_of(row) for row in rows] == ['already_quantized', consumer_kind])
    return fuse(adapter, source, choices, '_' + kind + '_groups', kind, pairs, prepare=lambda region, _: prepare(region),
                errors=(ValueError, NativeCompilationError), admit=counted_scratch, inputs=lambda plan: plan.input_ids,
                cycles='checked contraction epilogue; no measured cost', demands=(), occupancy=())


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .quantized_loop import compile_source
    compile_source(snapshot, group.source, target_source=target_source, destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source, kind='quantized_epilogue'):
    from .quantized_loop_c import prepare_c_package

    record, files = prepare_c_package(group.source, Path(compiled), name, target_source=Path(target_source))
    record.update(kind=kind, launches_per_invocation=1,
                  native_manifest_sha256=sha256((Path(compiled) / 'manifest.json').read_bytes()).hexdigest())
    return c_region(name, group.input_ids, (group.output_id,), files, ('segments/' + name,), record)
