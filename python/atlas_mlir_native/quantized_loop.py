"""Target-owned counted tile loops for unchanged quantized operands."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path

from merlin.semantic_compiler.target_binding import NativeCompilationError
from .already_quantized import prepare_source
from .binding import (DRAM_BASE, TILE_BYTES, MXU1_CAP448_POLICY, NATIVE_CHECKS,
                      RTL_REVISION, _native_pipeline, _tile_io_base, dram_slots, _run)
from .looped_tiling import _virtual
from .exponent_restoration import exponent_geometry, Restoration, fast_factor_range
from .restoration_budget import fallback_tiles, instruction_bound


def uses_tiled_fallback(plan):
    """Choose from selected DRAM and checked per-launch instruction bounds."""
    from .quantized_loop_sizing import size_counted
    if size_counted(plan).record()['route'] != 'k32_tiles':
        return False
    if plan.epilogue_precision is not None:
        return True
    from .quantized_row_blocks import NoFittingRowBlock, prepare_schedule
    try:
        prepare_schedule(plan.source)
    except NoFittingRowBlock:
        return True
    return False


def _execution_geometry(plan):
    """Derive the single counted frame used by selection and publication."""
    batch = prod(plan.batch_shape)
    mc, nc, kc = ((value + 31)//32 for value in (plan.m, plan.n, plan.k))
    rhs = plan.source.node(plan.operand_ids[1])
    sizes = [batch*mc*kc*1024, (batch if rhs.type.shape[:-2] else 1)*nc*kc*1024]
    output_bytes = batch*mc*nc*2048
    base = _tile_io_base.get()
    addresses = [base, base + ((sizes[0]+2047)//2048)*2048]
    cursor = addresses[1] + ((sizes[1]+2047)//2048)*2048
    exponents = exponent_geometry(plan)
    exponent_bytes = [geometry['bytes'] for geometry in exponents]
    exponent_addresses = [cursor, cursor+exponent_bytes[0]]
    output = exponent_addresses[1]+exponent_bytes[1]
    status = output+output_bytes
    end = status+2048
    epilogue = getattr(plan, 'epilogue_precision', None)
    extra = {}
    if epilogue is not None:
        addresses.append(end)
        sizes.append(output_bytes)
        extra = dict(epilogue=dict(other_id=plan.epilogue_other_id,
                     contract_operand=plan.epilogue_contract_operand,
                     offset_from_output=output_bytes+2048,
                     input_conversion=epilogue.operand_specs[1-plan.epilogue_contract_operand].record(),
                     contraction_conversion=epilogue.operand_specs[plan.epilogue_contract_operand].record()))
        end += output_bytes
    return dict(input_addresses=addresses, input_buffer_bytes=sizes,
                exponent_addresses=exponent_addresses, exponent_buffer_bytes=exponent_bytes,
                exponent_layouts=exponents, output_addresses=[output], output_bytes=output_bytes,
                status_address=status, status_bytes=2048, **extra), end


def execution_layout(plan, *, operand_reuse=None):
    from .quantized_operand_reuse import schedule_record
    geometry, end = _execution_geometry(plan)
    if end > _tile_io_base.get() + dram_slots()*TILE_BYTES:
        raise NativeCompilationError("resource_limit", "quantized loop buffers exceed selected DRAM window")
    return dict(schema="atlas.quantized_loop_execution.v1", operation="mxu1_cap448_looped_ordered",
                **geometry, **schedule_record(plan, operand_reuse),
                status_encoding='little-endian u32: 0=admitted; 1=exponent, readout, or fallback resource/domain refusal',
                restoration_admission=dict(scalar_fallback_tile_budget=fallback_tiles(plan),
                    successful_exponent_domain=('full_scalar_admission' if fallback_tiles(plan) else 'vector_domain_only'),
                    vector_per_operand_range=list(fast_factor_range(plan)),
                    out_of_domain='sticky_status_1_and_atomic_host_output_refusal'),
                )


def virtual_source(plan, layout):
    # The existing emitter consumes only counts/tile_bytes/buffer_bytes here.
    class Geometry:
        counts = tuple((v+31)//32 for v in (plan.m, plan.n, plan.k))
        tile_bytes = (1024, 1024, 2048)
        buffer_bytes = (layout['input_buffer_bytes'][0], layout['input_buffer_bytes'][1], layout['output_bytes'])
    from .region_composition import rematerialize_scalar_constants
    from .quantized_operand_reuse import schedule_record

    schedule = schedule_record(plan, layout.get('operand_reuse_requested', 'ordered'))
    if any(layout.get(key) != schedule.get(key) for key in ('operand_reuse_requested', 'operand_reuse', 'operand_reuse_cached_tiles')):
        raise ValueError('recorded quantized operand reuse schedule differs')
    restoration = Restoration(plan, layout, scalar_fallback=bool(fallback_tiles(plan)))
    virtual = _virtual(Geometry(), tuple(layout['input_addresses'][:2]), layout['output_addresses'][0],
                       unit=1, k_unroll=8 if prod(plan.batch_shape) == 1 else 1,
                       batch_count=prod(plan.batch_shape),
                       shared_weight=not plan.source.node(plan.operand_ids[1]).type.shape[:-2],
                       exponent_pages=restoration.pages(), restoration=restoration,
                       output_window_bytes=layout['output_bytes']+layout['status_bytes'],
                       cache_a=schedule.get('operand_reuse', 'ordered') in ('row-cache', 'prefix-cache'),
                       cache_a_tiles=schedule.get('operand_reuse_cached_tiles'))
    return rematerialize_scalar_constants(virtual)


def compile_source(snapshot, source, *, target_source, destination, limits, io_memory_base=DRAM_BASE):
    """Compile with the counted frame at io_memory_base; only the counted route relocates."""
    from .memory_windows import dma_span_in_window

    if type(io_memory_base) is not int or io_memory_base % TILE_BYTES:
        raise NativeCompilationError('compile_error', 'I/O base must be an aligned integer')
    token = _tile_io_base.set(io_memory_base)
    try:
        plan = prepare_source(source)
        if io_memory_base != DRAM_BASE and not dma_span_in_window(io_memory_base, _execution_geometry(plan)[1]-io_memory_base):
            raise NativeCompilationError('compile_error', 'complete quantized loop frame must fit one DMA window')
        return _compile_source(snapshot, source, plan, target_source=target_source,
                               destination=destination, limits=limits)
    finally:
        _tile_io_base.reset(token)


def _compile_source(snapshot, source, plan, *, target_source, destination, limits):
    from .quantized_loop_sizing import size_counted, publish_route

    sizing = size_counted(plan,target_source=target_source,limits=limits)
    if sizing.record()['route'] != 'counted' and _tile_io_base.get() != DRAM_BASE:
        raise NativeCompilationError('compile_error', 'custom I/O base requires the counted quantized loop route')
    if sizing.record()['route'] == 'k32_tiles':
        if getattr(plan, 'epilogue_precision', None) is not None:
            raise NativeCompilationError('resource_limit', 'overlarge contraction must retain its separate epilogue')
        from .quantized_row_blocks import NoFittingRowBlock, prepare_schedule, compile_source as compile_rows
        try:
            schedule = prepare_schedule(source, target_source=target_source, limits=limits)
        except NoFittingRowBlock as error:
            row_refusals = error.record()
        else:
            if schedule is None:
                raise ValueError('row planning changed the complete counted decision')
            compile_rows(snapshot, source, schedule, target_source=target_source,
                         destination=destination, limits=limits)
            return plan
        from .already_quantized import compile_source as compile_tiles
        result = compile_tiles(snapshot, source, target_source=target_source, destination=destination, limits=limits)
        publish_route(plan,Path(destination),sizing,target_source=target_source)
        (Path(destination)/'row-block-refusals.json').write_text(json.dumps(row_refusals,sort_keys=True,indent=2)+'\n')
        return result
    policy = sizing.record().get('operand_reuse_requested', 'ordered')
    layout = execution_layout(plan, operand_reuse=policy)
    selections = {}
    for role, body in (("reset", plan.reset), ("continue", plan.continuation)):
        selected = snapshot.select(body, limits=limits)
        if selected.status != "selected" or not selected.graph or not selected.rules or not selected.allocation:
            raise NativeCompilationError(selected.status, "quantized loop tile selection: " + selected.reason)
        names = [selected.rules.symbols[v.symbol].get('descriptor', {}).get('name')
                 for v in selected.graph.values if v.kind == 'instruction']
        required = 'atlas.mxu1_cap448_reset_tile' if role == 'reset' else 'atlas.mxu1_cap448_accumulate_tile'
        if required not in names:
            raise NativeCompilationError('unqualified_target', 'quantized loop has no admitted MXU1 body')
        selections[role] = dict(request_digest=body.digest(), check_fingerprint=selected.check_fingerprint,
                                instructions=names, allocation=asdict(selected.allocation))
    epilogue = getattr(plan, 'epilogue_precision', None)
    if epilogue is not None:
        from .pointwise import prepare_plan
        from .binding import AtlasNativeBinding, BF16_BINARY_MODES
        epilogue_plan = prepare_plan(epilogue.request)
        body = epilogue_plan.body_request()
        selected = snapshot.select(body, limits=limits)
        if selected.status != 'selected' or not selected.graph or not selected.rules or not selected.allocation:
            raise NativeCompilationError(selected.status, 'quantized epilogue tile selection: '+selected.reason)
        names = [selected.rules.symbols[value.symbol].get('descriptor',{}).get('name')
                 for value in selected.graph.values if value.kind == 'instruction']
        expected = next(descriptor.name for descriptor in AtlasNativeBinding.profile().descriptors
                        if descriptor.computation == body.nodes[-1].op and descriptor.name in BF16_BINARY_MODES)
        if names.count(expected) != 1 or BF16_BINARY_MODES[expected] != 'mul':
            raise NativeCompilationError('unqualified_target','quantized epilogue has no identical admitted pointwise product body')
        selections['epilogue'] = dict(request_digest=body.digest(),check_fingerprint=selected.check_fingerprint,
                                     instructions=names,descriptor=expected,allocation=asdict(selected.allocation))
    virtual = virtual_source(plan, layout)
    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR', Path(target_source)/'build/bin'))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Sizing is not native publication: semantic body selection above and
    # source attestation below still run for this exact request and snapshot.
    sizing = size_counted(plan,target_source=target_source,limits=limits,operand_reuse=policy)
    if sizing.record()['route'] != 'counted':
        raise NativeCompilationError('resource_limit','counted sizing changed during semantic selection')
    machine = sizing.read('checked/atlas-machine.mlir').decode()
    (destination/'compilation-pipeline.json').write_bytes(sizing.read('checked/compilation-pipeline.json'))
    if 'checked/mandatory-spilled-virtual.mlir' in dict(sizing.files):
        (destination/'mandatory-spilled-virtual.mlir').write_bytes(sizing.read('checked/mandatory-spilled-virtual.mlir'))
    coverage = sizing.record()
    if not (coverage.get('checks_covered_by_immutable_mandatory_result') is True
            and coverage['identity']['pipeline'] == 'mandatory'
            and coverage['identity']['pipeline_arguments'] == ['--atlas-compile-unscheduled']):
        _run([str(tools/'atlas-opt'), *NATIVE_CHECKS, '--verify-atlas-timing', '-'], input_text=machine)
    program = json.loads(sizing.read('checked/physical-program.json'))
    if program.get('selected_rtl_revision') != RTL_REVISION:
        raise NativeCompilationError('unqualified_target', 'quantized loop emitter revision differs')
    binary = b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
    if len(binary) > 32768*4:
        raise NativeCompilationError('resource_limit', 'quantized loop exceeds instruction memory')
    try:
        bound = instruction_bound(virtual, machine, program)
    except (ValueError, KeyError, IndexError) as error:
        raise NativeCompilationError('resource_limit', 'quantized loop instruction bound is unproved: '+str(error)) from error
    if not bound['admitted']:
        raise NativeCompilationError('resource_limit', 'quantized loop worst-case instruction bound exceeds 1000000: '+str(bound['instruction_bound']))
    bound['scalar_fallback_tile_budget'] = fallback_tiles(plan)
    bound['restoration_admission'] = layout['restoration_admission']
    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    artifacts = {'source-request.json': encoded(source.record()),
                 'already-quantized-plan.json': encoded(plan.record()),
                 'execution_plan.json': encoded(layout), 'tile-body-selection.json': encoded(selections),
                 'compilation-pipeline.json': (destination/'compilation-pipeline.json').read_bytes(),
                 'restoration-instruction-bound.json': encoded(bound),
                 'restoration-guard.json': encoded(dict(schema='atlas.exponent_restoration_guard.v1',
                     guard_block='restore_factor_done', vector_block='restore_vector', scalar_block='restore_scalar',
                     factor_body='restore_factor_body', factor_done='restore_factor_done',
                     exponent_modes=[geometry['mode'] for geometry in exponent_geometry(plan)],
                     exponent_roles=['lhs','rhs'],
                     scalar_factor_broadcast='sparse_pow2_s_minus_1_col_sum_then_row_sum',
                     join_block='restore_join', factor_header='restore_factor_header', factor_bound=8,
                     lhs_index_coefficient=4, lhs_index_offsets=[0,1,2,3],
                     rhs_index_coefficient=2, rhs_index_offsets=[0,1,16,17],
                     per_operand_range=list(fast_factor_range(plan)),
                     raw_nonzero_abs_exp2_bounds=[-26,32], exponent_pages=layout['exponent_addresses'])),
                 'atlas-virtual.mlir': virtual.encode(), 'atlas-machine.mlir': machine.encode(),
                 'physical-program.json': encoded(program)}
    if (destination/'mandatory-spilled-virtual.mlir').is_file():
        artifacts['mandatory-spilled-virtual.mlir'] = (destination/'mandatory-spilled-virtual.mlir').read_bytes()
    manifest = dict(schema='atlas.quantized_loop_compilation.v1', artifact_kind='program', engine='merlin_native',
                    request_digest=source.digest(), target_identity=source.target_identity,
                    binary_sha256=sha256(binary).hexdigest(),
                    execution_plan_sha256=sha256(artifacts['execution_plan.json']).hexdigest(),
                    stage_artifacts={key:sha256(data).hexdigest() for key,data in artifacts.items()},
                    target_inventory_sha256=sha256((Path(target_source)/'docs/selected-variant-inventory.json').read_bytes()).hexdigest(),
                    io_memory_bytes=dram_slots()*TILE_BYTES,
                    atlas_opt_sha256=sha256((tools/'atlas-opt').read_bytes()).hexdigest(),
                    atlas_emit_sha256=sha256((tools/'atlas-emit').read_bytes()).hexdigest())
    for name, data in {**artifacts, 'program.bin':binary, 'manifest.json':encoded(manifest)}.items():
        (destination/name).write_bytes(data)
    publish_route(plan,destination,sizing,target_source=target_source)
    return plan
