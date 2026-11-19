"""Publish a checked quantized product through its private exact BF16 cast."""
from dataclasses import replace
from functools import partial

from merlin.compile.result_narrowing import split_private_bf16_cast
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .composition_adapter import device_casts, fuse, kind_of, neighbor_pairs
from .quantized_epilogue import compile_group, export_group as export_counted


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
    from .quantized_epilogue import counted_scratch

    return fuse(adapter, source, choices, '_quantized_epilogue_narrowing_groups', 'quantized_epilogue_narrowing',
                neighbor_pairs(choices, device_casts(choices), lambda row: kind_of(row) == 'quantized_epilogue'),
                prepare=lambda region, _: prepare_source(region), errors=(ValueError, NativeCompilationError),
                admit=counted_scratch, inputs=lambda plan: plan.input_ids,
                cycles='unchanged checked product; private exact cast omitted')


export_group = partial(export_counted, kind='quantized_epilogue_narrowing')
