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


def adjacent_alternatives(adapter, source, choices, *, kind, consumer_kind, prepare):
    """Offer a recognized quantized group only after counted sizing succeeds."""
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval
    from .quantized_loop_sizing import size_counted

    choices = list(choices)
    singles = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    groups = {}
    setattr(adapter, "_" + kind + "_groups", groups)
    for first, last in zip(singles, singles[1:]):
        if (first.placement != 'accelerator' or last.placement != 'accelerator'
                or last.nodes[0] != first.nodes[0] + 1
                or dict(first.metadata).get('kind') != 'already_quantized'
                or dict(last.metadata).get('kind') != consumer_kind):
            continue
        ids = tuple(pure[row.nodes[0]].id for row in (first, last))
        try:
            region = extract_semantic_region(source, ids)
            plan = prepare(region)
        except (ValueError, NativeCompilationError):
            continue
        # Only a proved construction refusal declines an otherwise supported
        # candidate. Tool failures or unproved bounds remain diagnosed errors.
        sizing = size_counted(plan).record()
        if sizing['route'] != 'counted':
            continue
        arena = sizing['counted_frame_bytes']
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        groups[region.digest()] = plan
        choices.append(replace(first, id=kind + '_' + first.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_' + kind,
            inputs=tuple(ports[key] for key in plan.input_ids), outputs=tuple(ports[key] for key in region.outputs),
            demands=(), occupancy=(), scratch_bytes=arena,
            cycles=CycleInterval.unknown('checked contraction epilogue; no measured cost'),
            metadata=(('kind', kind),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .quantized_loop import compile_source
    compile_source(snapshot, group.source, target_source=target_source, destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source, kind='quantized_epilogue'):
    from merlin.compile.semantic_application import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES
    from .quantized_loop_c import prepare_c_package

    record, files = prepare_c_package(group.source, Path(compiled), name, target_source=Path(target_source))
    record.update(kind=kind, launches_per_invocation=1,
                  native_manifest_sha256=sha256((Path(compiled) / 'manifest.json').read_bytes()).hexdigest())
    return CompiledCRegion(name + '_source_dispatch', name + '.h', group.input_ids, (group.output_id,),
                           files, RUNTIME_SHARED_FILES, ('segments/' + name,), record)
