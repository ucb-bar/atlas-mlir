"""Consume private exact input casts through unchanged pointwise encoders."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.input_widening import split_private_bf16_widening
from merlin.compile.semantic_regions import extract_semantic_region


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
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    pure = [node for node in source.nodes if node.effect == 'pure']
    casts = {row.nodes[0]: row for row in choices if len(row.nodes) == 1
             and row.placement == 'accelerator' and dict(row.metadata).get('kind') == 'exact_float_device'}
    adapter._pointwise_input_widening_groups = {}
    for last in tuple(choices):
        kind = dict(last.metadata).get('kind')
        if last.placement != 'accelerator' or kind not in ('pointwise', 'pointwise_math', 'pointwise_narrowing'):
            continue
        first = casts.get(min(last.nodes)-1)
        if first is None or last.nodes != tuple(range(min(last.nodes), max(last.nodes)+1)):
            continue
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in (*first.nodes, *last.nodes)))
            group = prepare(region, adapter, output_dtype='bf16' if kind == 'pointwise_narrowing' else 'f32')
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        adapter._pointwise_input_widening_groups[region.digest()] = group
        choices.append(replace(last, id='pointwise_input_widening_' + last.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_pointwise_input_widening',
            inputs=(ports[group.input_id],), outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('unchanged pointwise program; private input widening omitted'),
            metadata=(('kind', 'pointwise_input_widening'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise_narrowing import compile_group as compile_pointwise

    compile_pointwise(group, snapshot, target_source=target_source, destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.semantic_application import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES
    from .pointwise_c import prepare_precision_c_package

    compiled = Path(compiled)
    record, files = prepare_precision_c_package(group.precision, compiled, name,
        target_source=Path(target_source), input_dtype='bf16', output_dtype=group.output_dtype)
    record.update(kind='pointwise_input_widening', input_node_ids=[group.input_id],
        output_node_ids=list(group.source.outputs), source_request_digest=group.source.digest(),
        widening_proof='raw BF16 validation equals exact widening then the original encoder',
        context_type=name + '_context', source_context_type=name + '_source_context',
        binding_function=name + '_bind', source_binding_function=name + '_source_bind',
        launches_per_invocation=record.get('streaming', {}).get('launch_count', 1),
        native_manifest_sha256=sha256((compiled/'manifest.json').read_bytes()).hexdigest())
    return CompiledCRegion(name + '_source_dispatch', name + '.h', (group.input_id,),
        group.source.outputs, files, RUNTIME_SHARED_FILES, ('segments/' + name,), record)
