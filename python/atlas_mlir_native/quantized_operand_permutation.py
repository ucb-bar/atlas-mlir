"""Read a private matrix transpose during the existing FP8 tile packing.

Merlin proves complete permutation semantics and ownership. Atlas restricts the
mapping to unchanged batch axes and its counted unchanged-code MXU route.
"""
from dataclasses import replace

from merlin.compile.layout_c import permutation_axes
from merlin.compile.operand_permutations import private_operand_permutations
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def prepare_source(source):
    from .already_quantized import prepare_source as prepare_contraction

    pure = [node for node in source.nodes if node.effect == 'pure']
    views = [node for node in pure if node.op in {'gather', 'transpose'}]
    if len(views) != 1:
        raise ValueError('quantized operand permutation needs one private matrix transpose')
    view = views[0]
    if len(view.inputs) != 1 or view.id in source.outputs:
        raise ValueError('quantized operand permutation must have one input and remain private')
    base = source.node(view.inputs[0])
    axes = permutation_axes(base, view)
    rank = len(view.type.shape)
    if (rank < 2 or axes != (*range(rank - 2), rank - 1, rank - 2)
            or base.effect not in {'input', 'constant'}):
        raise ValueError('quantized operand permutation must preserve every batch axis')
    logical = extract_semantic_region(source, tuple(node.id for node in pure if node != view))
    plan = prepare_contraction(logical)
    if (plan.operand_ids.count(view.id) != 1
            or sum(node.inputs.count(view.id) for node in pure) != 1):
        raise ValueError('quantized operand permutation must feed exactly one code operand')
    inputs = tuple(node.id for node in source.nodes if node.effect in {'input', 'constant'})
    expected = {base.id if key == view.id else key for key in plan.input_ids}
    if set(inputs) != expected or len(inputs) != len(plan.input_ids):
        raise ValueError('quantized operand permutation needs distinct unchanged boundary ports')
    return replace(plan, source=source, input_ids=inputs,
                   operand_permutation=(view.id, base.id, axes))


def alternatives(adapter, source, choices):
    from .quantized_loop_sizing import size_counted

    groups = {}
    adapter._quantized_operand_permutation_groups = groups
    result = list(choices)
    for candidate in private_operand_permutations(source, choices):
        consumer = candidate.consumer
        if (consumer.placement != 'accelerator' or dict(consumer.metadata).get('kind') not in
                {'already_quantized', 'quantized_narrowing', 'quantized_epilogue'}):
            continue
        try:
            plan = prepare_source(candidate.region)
        except (ValueError, NativeCompilationError):
            continue
        sizing = size_counted(plan).record()
        if sizing['route'] != 'counted':
            continue
        groups[plan.source.digest()] = plan
        nodes = tuple(sorted((*candidate.producer.nodes, *consumer.nodes)))
        result.append(replace(consumer, id='quantized_operand_permutation_' + '_'.join(map(str, nodes)),
            nodes=nodes, implementation='source_quantized_operand_permutation',
            cycles=CycleInterval.unknown('counted MXU with folded operand permutation; unmeasured'),
            inputs=candidate.inputs, outputs=candidate.outputs, scratch_bytes=sizing['counted_frame_bytes'],
            metadata=(('kind', 'quantized_operand_permutation'),)))
    return tuple(result)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .quantized_loop import compile_source

    compile_source(snapshot, group.source, target_source=target_source, destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .quantized_epilogue import export_group as export_counted

    return export_counted(group, compiled, name, target_source=target_source,
                          kind='quantized_operand_permutation')
