"""Publish a checked quantized product through its private exact BF16 cast."""
from dataclasses import replace

from merlin.compile.result_narrowing import split_private_bf16_cast
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.target_binding import NativeCompilationError


def prepare_source(source):
    from .quantized_epilogue import prepare_source as prepare_epilogue

    producer = split_private_bf16_cast(source)
    plan = prepare_epilogue(producer)
    # The existing epilogue validates the restored contraction and its other
    # operand before a bounded BF16 RNE product. Neither admitted operand can
    # be NaN or infinity, so the result is non-NaN. Exact widening followed by
    # RNE narrowing preserves every such BF16 code, including signed zero.
    if (plan.output_dtype != 'f32' or
            plan.epilogue_precision.request.nodes[-1].op != 'mul_bf16_zero'):
        raise ValueError('quantized product narrowing requires a non-NaN BF16 result proof')
    return replace(plan, source=source, region_id=source.outputs[0],
                   output_dtype='bf16', narrowing_input_id=producer.outputs[0])


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval
    from .quantized_loop_sizing import size_counted

    choices = list(choices)
    casts = {row.nodes[0]: row for row in choices
             if len(row.nodes) == 1 and row.placement == 'accelerator'
             and dict(row.metadata).get('kind') == 'exact_float_device'}
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._quantized_epilogue_narrowing_groups = {}
    for first in tuple(choices):
        if first.placement != 'accelerator' or dict(first.metadata).get('kind') != 'quantized_epilogue':
            continue
        last = casts.get(max(first.nodes) + 1)
        if last is None:
            continue
        try:
            region = extract_semantic_region(source, tuple(pure[i].id for i in (*first.nodes, *last.nodes)))
            group = prepare_source(region)
        except (ValueError, NativeCompilationError):
            continue
        # Only a proved capacity refusal can decline a valid construction.
        # Tool errors and failed safety proofs must remain visible.
        sizing = size_counted(group).record()
        if sizing['route'] != 'counted':
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        adapter._quantized_epilogue_narrowing_groups[region.digest()] = group
        choices.append(replace(first, id='quantized_epilogue_narrowing_' + first.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_quantized_epilogue_narrowing',
            inputs=tuple(ports[key] for key in group.input_ids),
            outputs=tuple(ports[key] for key in region.outputs),
            scratch_bytes=sizing['counted_frame_bytes'],
            cycles=CycleInterval.unknown('unchanged checked product; private exact cast omitted'),
            metadata=(('kind', 'quantized_epilogue_narrowing'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .quantized_loop import compile_source

    compile_source(snapshot, group.source, target_source=target_source, destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .quantized_epilogue import export_group as export_counted

    return export_counted(group, compiled, name, target_source=target_source,
                          kind='quantized_epilogue_narrowing')
