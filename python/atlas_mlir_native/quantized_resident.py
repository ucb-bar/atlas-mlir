"""Compiler-owned immutable input bindings for counted quantized programs.

The compiler packs and validates immutable bytes once. A binding changes only
the DMA base; the counted program still declares the complete readonly range.
"""
from __future__ import annotations

from math import prod

from merlin.semantic_compiler.target_binding import NativeCompilationError


def _inputs(plan):
    from .exponent_restoration import exponent_geometry

    result = []
    for index, node_id in enumerate(plan.physical_operand_ids):
        node = plan.source.node(node_id)
        logical = plan.source.node(plan.operand_ids[index])
        shape = list(logical.type.shape)
        batches, rows, columns = prod(shape[:-2]), shape[-2], shape[-1]
        layout = dict(schema='atlas.resident_fp8_tiles.v1', source_shape=list(node.type.shape),
                      logical_shape=shape, source_dtype=node.type.dtype,
                      batch_count=batches, row_tiles=(rows+31)//32, k_tiles=(columns+31)//32,
                      tile_bytes=1024, padding='positive_zero')
        if node_id != plan.operand_ids[index]:
            layout['permutation'] = list(plan.operand_permutation[2])
        result.append(dict(node_id=node_id, bytes=batches*layout['row_tiles']*layout['k_tiles']*1024,
                           alignment=2048, kind='fp8', layout=layout,
                           operand_index=index))
    for index, (node_id, geometry) in enumerate(zip(plan.exponent_ids, exponent_geometry(plan), strict=True)):
        node = plan.source.node(node_id)
        layout = dict(schema='atlas.resident_exponent_pages.v1', source_shape=list(node.type.shape),
                      source_dtype=node.type.dtype, **geometry, padding='zero')
        result.append(dict(node_id=node_id, bytes=geometry['bytes'], alignment=2048,
                           kind='exponent', layout=layout, operand_index=index))
    return result


def _inventory(plan):
    result = {}
    for entry in _inputs(plan):
        item = {key: value for key, value in entry.items() if key != 'operand_index'}
        previous = result.setdefault(entry['node_id'], item)
        if previous != item:
            raise NativeCompilationError('unsupported_semantics',
                                         'one immutable input requires incompatible resident layouts')
    return list(result.values())


def inventory_resident_inputs(source):
    """Describe exact device layouts accepted for an outlined counted source."""
    from .already_quantized import prepare_source
    return _inventory(prepare_source(source))


def pack_resident_input(source, node_id, dense_bytes):
    """Validate source bytes and produce the same zero-padded layout as C."""
    import numpy as np
    from .already_quantized import prepare_source, valid_codes

    plan = prepare_source(source)
    entry = next((item for item in _inventory(plan) if item['node_id'] == node_id), None)
    if entry is None:
        raise ValueError('input has no counted resident layout: '+str(node_id))
    layout = entry['layout']
    raw = bytes(dense_bytes)
    count = prod(layout['source_shape'])
    width = 1 if entry['kind'] == 'fp8' else 4
    if len(raw) != count*width:
        raise ValueError('resident input byte length differs from its source tensor')
    packed = bytearray(entry['bytes'])
    if entry['kind'] == 'fp8':
        values = valid_codes(np.frombuffer(raw, dtype=np.uint8).reshape(layout['source_shape']))
        if 'permutation' in layout:
            values = values.transpose(layout['permutation'])
        rows, columns = layout['logical_shape'][-2:]
        values = values.reshape(layout['batch_count'], rows, columns)
        for batch in range(layout['batch_count']):
            for row in range(0, rows, 32):
                for column in range(0, columns, 32):
                    tile = np.zeros((32, 32), dtype=np.uint8)
                    view = values[batch, row:row+32, column:column+32]
                    tile[:view.shape[0], :view.shape[1]] = view
                    offset = ((batch*layout['row_tiles']+row//32)*layout['k_tiles']+column//32)*1024
                    packed[offset:offset+1024] = tile.tobytes()
    elif layout['mode'] == 'row':
        for batch in range(layout['batches']):
            for row in range(layout['rows']):
                source_offset = (batch*layout['rows']+row)*4
                offset = (batch*layout['tiles']+row//32)*2048+(row%32)*4
                packed[offset:offset+4] = raw[source_offset:source_offset+4]
    else:
        for page in range(layout['pages']):
            packed[page*2048:page*2048+4] = raw[page*4:page*4+4]
    return bytes(packed)


def requested_source_bindings(source):
    from merlin.semantic_compiler.codegen import current_codegen_options
    bindings = getattr(current_codegen_options(), 'resident_bindings', None)
    if bindings is None:
        return {}
    if not isinstance(bindings, dict):
        raise NativeCompilationError('compile_error', 'resident bindings must be source-keyed mappings')
    selected = bindings.get(source.digest(), {})
    if not isinstance(selected, dict):
        raise NativeCompilationError('compile_error', 'resident input bindings must be node-keyed mappings')
    return selected


def requested_bindings(plan):
    return requested_source_bindings(plan.source)


def validated_bindings(source, inventory, *, staging_base=None):
    """Validate exact immutable spans without changing the staging reservation."""
    from merlin.semantic_compiler.codegen import current_codegen_options
    from .binding import validate_io_window
    from .memory_windows import dma_span_in_window

    selected = requested_source_bindings(source)
    if not selected:
        return []
    inventory = {entry['node_id']: entry for entry in inventory}
    if set(selected)-set(inventory):
        raise NativeCompilationError('unsupported_semantics', 'resident binding has no accepted input layout')
    staging, staging_bytes = validate_io_window()
    if staging_base is not None:
        from .binding import validate_explicit_io_base
        staging = validate_explicit_io_base(staging_base)
    options = current_codegen_options()
    capacity_base, capacity_bytes = options.io_memory_capacity_base, options.io_memory_capacity_bytes
    if capacity_base is None or capacity_bytes is None:
        raise NativeCompilationError('compile_error', 'resident bindings require declared memory capacity')
    records = []
    for node_id in sorted(selected):
        binding, entry = selected[node_id], inventory[node_id]
        if (not isinstance(binding, dict) or set(binding) != {'address', 'bytes', 'immutable'}
                or binding.get('immutable') is not True or type(binding.get('address')) is not int
                or type(binding.get('bytes')) is not int or binding['bytes'] != entry['bytes']):
            raise NativeCompilationError('compile_error', 'resident binding must name an exact immutable input span')
        address, size = binding['address'], binding['bytes']
        if (address % entry['alignment'] or not dma_span_in_window(address, size)
                or not capacity_base <= address < address+size <= capacity_base+capacity_bytes
                or address < staging+staging_bytes and staging < address+size
                or any(address < other['address']+other['bytes'] and other['address'] < address+size
                       for other in records)):
            raise NativeCompilationError('compile_error', 'resident input span is unaligned, outside capacity, or overlaps another range')
        records.append(dict(**entry, address=address, immutable=True))
    return records


def bind_geometry(plan, geometry):
    """Apply authenticated compiler settings after validating all DMA ranges."""
    selected = requested_bindings(plan)
    if not selected:
        return geometry
    records = validated_bindings(plan.source, _inventory(plan))
    result = dict(geometry, input_addresses=list(geometry['input_addresses']),
                  exponent_addresses=list(geometry['exponent_addresses']), resident_bindings=records)
    for slot in _inputs(plan):
        if slot['node_id'] in selected:
            key = 'input_addresses' if slot['kind'] == 'fp8' else 'exponent_addresses'
            result[key][slot['operand_index']] = selected[slot['node_id']]['address']
    return result
