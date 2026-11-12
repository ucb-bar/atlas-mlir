"""Compose square, seeded reduction, checked layout transition and division.

The transition reproduces the removed division encoder on logical reduction
values. Rejected values become admitted padding before device arithmetic, and
the accumulated status is checked before any source result is published.
"""
from dataclasses import replace
from math import prod
import os
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import pointwise_identity_maps

from . import adjacent_square_reduce as square_reduce
from .region_composition import (
    allocate_adjacent_arena,
    compile_pointwise_member,
    compile_adjacent_virtual,
    publish_adjacent_group,
    verify_adjacent_group,
    export_adjacent_package,
)
from .binding import DRAM_BASE


def prepare(region, adapter=None):
    from .pointwise import prepare_source, prepare_execution_plan

    pure = [node for node in region.nodes if node.effect == 'pure']
    if len(pure) != 3:
        raise ValueError('reduction/scale enlargement requires exactly three source operations')
    square, reduced, divided = pure
    if (divided.op != 'f32_div' or len(divided.inputs) != 2 or divided.inputs.count(reduced.id) != 1
            or region.outputs != (divided.id,)):
        raise ValueError('reduction/scale enlargement needs one closed division consumer')
    prefix = square_reduce.prepare(extract_semantic_region(region, (square.id, reduced.id)), adapter)
    precision = prepare_source(extract_semantic_region(region, (divided.id,)), allow_math=True)
    _, plan, streaming = prepare_execution_plan(precision.request)
    rows = prefix['plans'][1].matrix_shape[0]
    if (streaming or plan.compact_broadcasts or plan.shape != (1, rows)
            or any(node.type != divided.type for node in region.nodes if node.id in divided.inputs)
            or divided.index_maps != pointwise_identity_maps(len(divided.type.shape), 2)):
        raise ValueError('division needs dense row-layout operands matching the reduction result')
    keys = precision.record()['inputs']
    operand = keys.index(reduced.id)
    other = keys[1 - operand]
    inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
    if len(inputs) != 3 or set(inputs) != {prefix['data'], prefix['seed'], other}:
        raise ValueError('reduction/scale needs complete distinct data, seed and division boundaries')
    names = {node.id for node in region.nodes}
    def internal(name):
        while name in names:
            name += '_'
        names.add(name)
        return name
    transition, status = internal('__reduction_row'), internal('__boundary_status')
    mutable = (square.id, reduced.id, transition, divided.id, status)
    sizes = {prefix['data']: prefix['sizes'][prefix['data']], prefix['seed']: prefix['sizes'][prefix['seed']],
             other: plan.input_packed_bytes[1 - operand], square.id: prefix['sizes'][square.id],
             reduced.id: prefix['sizes'][reduced.id], transition: plan.input_packed_bytes[operand],
             divided.id: plan.packed_bytes, status: 2048}
    addresses, arena_bytes = allocate_adjacent_arena(sizes, (*inputs, *mutable))
    prefix = {**prefix,
              'addresses': {key: addresses[key] for key in (*prefix['inputs'], *prefix['outputs'])},
              'arena_bytes': addresses[reduced.id] + sizes[reduced.id] - DRAM_BASE}
    spec = precision.operand_specs[operand]
    proof = dict(square_preimage=prefix['proof'], consumer_operand=operand,
                 consumer_conversion=spec.record(), transition='column_to_row',
                 source_matrix=[rows, 1], destination_matrix=[1, rows],
                 pad_code=int.from_bytes(plan.input_pad(operand), 'little'),
                 rejection='invalid logical value becomes consumer padding and sets invocation status; '
                           'C returns INPUT_DOMAIN before assembling or publishing source output',
                 conversion='BF16 exact decode followed by BF16 RNE preserves admitted codes; '
                            'the transition applies original canonicalization and code admission',
                 operation_order='complete square, then unchanged seeded reduction, then byte transition, '
                                 'then unchanged selected reciprocal/product division')
    return dict(region=region, prefix=prefix, precision=precision, plan=plan, rows=rows,
                inputs=inputs, mutable=mutable, transition=transition, status=status,
                data=prefix['data'], seed=prefix['seed'], other=other, operand=operand,
                sizes=sizes, addresses=addresses, arena_bytes=arena_bytes, proof=proof)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    ordered = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._reduction_scale_groups = {}
    for first, middle, last in zip(ordered, ordered[1:], ordered[2:]):
        if (any(row.placement != 'accelerator' for row in (first, middle, last))
                or middle.nodes[0] != first.nodes[0] + 1 or last.nodes[0] != middle.nodes[0] + 1
                or [dict(row.metadata).get('kind') for row in (first, middle, last)]
                    != ['pointwise', 'reduction', 'pointwise_math']):
            continue
        ids = tuple(pure[row.nodes[0]].id for row in (first, middle, last))
        try:
            region = extract_semantic_region(source, ids)
            group = prepare(region, adapter)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, middle, last) for port in (*row.inputs, *row.outputs)}
        adapter._reduction_scale_groups[region.digest()] = group
        choices.append(replace(first, id='reduction_scale_' + first.id,
            nodes=(*first.nodes, *middle.nodes, *last.nodes), implementation='source_adjacent_reduction_scale',
            inputs=tuple(ports[key] for key in group['inputs']), outputs=tuple(ports[key] for key in region.outputs),
            demands=(), occupancy=(), scratch_bytes=group['arena_bytes'],
            cycles=CycleInterval.unknown('checked reduction/scale enlargement; no measured cost'),
            metadata=(('kind', 'adjacent_reduction_scale'),)))
    return tuple(choices)


def transition_source(group):
    from .device_boundary import virtual_transition

    spec = group['precision'].operand_specs[group['operand']]
    first = group['mutable'][0]
    return virtual_transition(source_address=group['addresses'][group['mutable'][1]],
        destination_address=group['addresses'][group['transition']], status_address=group['addresses'][group['status']],
        rows=group['rows'], columns=1, kind='column_to_row', admitted_code_ranges=spec.admitted_code_ranges,
        pad=int.from_bytes(group['plan'].input_pad(group['operand']), 'little'),
        canonicalize_negative_zero=spec.canonicalize_negative_zero,
        output_base=group['addresses'][first], output_bytes=sum(group['sizes'][key] for key in group['mutable']))


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import virtual_program
    from .region_composition import rematerialize_scalar_constants
    from .resident_graph import compose_fragments

    destination, target_source = Path(destination), Path(target_source)
    destination.mkdir()
    square_reduce.compile_group(group['prefix'], snapshot, target_source=target_source,
                                destination=destination / 'square-reduction', limits=limits)
    precision, plan = group['precision'], group['plan']
    keys = precision.record()['inputs']
    addresses = tuple(group['addresses'][group['transition'] if key == group['mutable'][1] else key] for key in keys)
    output = group['addresses'][group['mutable'][3]]
    descriptor, _ = compile_pointwise_member(precision, plan, snapshot, addresses=addresses, output=output,
        target_source=target_source, destination=destination / 'division', limits=limits)
    final = virtual_program(plan, addresses, output, descriptor)
    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR', target_source / 'build/bin'))
    virtual = compose_fragments([(destination / 'square-reduction/atlas-virtual.mlir').read_text(),
                                 transition_source(group), final], tools=tools, boundaries_lowered=True)
    virtual = rematerialize_scalar_constants(virtual)
    compile_adjacent_virtual(virtual, destination=destination, target_source=target_source)
    mutable_bytes = sum(group['sizes'][key] for key in group['mutable'])
    execution = dict(schema='atlas.adjacent_reduction_scale_execution.v1', launches_per_invocation=1,
        input_addresses=[group['addresses'][key] for key in group['inputs']],
        input_buffer_bytes=[group['sizes'][key] for key in group['inputs']],
        output_addresses=[group['addresses'][group['mutable'][0]]], output_bytes=mutable_bytes,
        addresses=group['addresses'], buffer_bytes=group['sizes'], proof=group['proof'])
    publish_adjacent_group(group, destination, kind='adjacent_reduction_scale', execution=execution,
        enlargement=dict(proof=group['proof'],
            precision_plan=[*(p.record() for p in group['prefix']['precisions']), precision.record()],
            source_request_digest=group['region'].digest()))


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.c_conversions import render_f32_array_c
    from merlin.compile.c_pointwise import render_pointwise_c
    from merlin.compile.c_reduction import render_reduction_c

    region, compiled = group['region'], Path(compiled)
    precision_plan = [*(p.record() for p in group['prefix']['precisions']), group['precision'].record()]
    execution = verify_adjacent_group(group, compiled, precision_plan=precision_plan)
    first, second = group['prefix']['plans']
    prefixes = (name + '_square', name + '_reduction', name + '_division')
    helpers = [render_pointwise_c(first, prefixes[0]), render_reduction_c(second, prefixes[1]),
               render_pointwise_c(group['plan'], prefixes[2])]
    helpers = [text.replace('static void ', 'static inline void ') for text in helpers]
    prefix = group['prefix']
    specs = {**square_reduce.outer_input_specs(prefix), group['other']: group['precision'].operand_specs[1 - group['operand']]}
    packers = {group['data']: prefixes[0] + '_pack_0', group['seed']: prefixes[1] + '_pack_1',
               group['other']: prefixes[2] + '_pack_' + str(1 - group['operand'])}
    for index, key in enumerate(group['inputs']):
        count, encoder = prod(region.node(key).type.shape), name + f'_encode_{index}'
        helpers.append(render_f32_array_c(specs[key], count, encoder) if key == group['other']
                       else square_reduce.render_outer_encoder(prefix, key, count, encoder))
        helpers.append(f'static void {name}_pack_{index}(uint8_t *out,const uint8_t *in) {{ {packers[key]}(out,in); }}')
    first_address = group['addresses'][group['mutable'][0]]
    result_offset = group['addresses'][group['mutable'][3]] - first_address
    helpers.append(f'static void {name}_assemble(uint8_t *out,const uint8_t *arena) {{ '
                   f'{prefixes[2]}_assemble(out,arena+{result_offset}u); }}')
    status_offset = sum(group['sizes'][key] for key in group['inputs']) + group['addresses'][group['status']] - first_address
    check = ('  if (' + ' | '.join(f'region->workspace[{status_offset + byte}u]' for byte in range(4))
             + f') return {name.upper()}_INPUT_DOMAIN;')
    guard, input_guard = square_reduce.render_combination_guard(prefix, prefixes[1], group['inputs'])
    return export_adjacent_package(group, compiled, name, kind='adjacent_reduction_scale', target_source=target_source,
        execution=execution, input_sources=group['inputs'], helpers=helpers + guard, completion_check=check,
        input_guard=input_guard,
        packed_sizes=(*(group['sizes'][key] for key in group['inputs']),sum(group['sizes'][key] for key in group['mutable'])),
        metadata=dict(precision_plan=precision_plan, outer_input_conversions={key: spec.record() for key, spec in specs.items()}))
