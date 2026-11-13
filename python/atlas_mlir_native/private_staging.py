"""Keep proved private BF16 staging edges in invocation-local VMEM slots."""
from __future__ import annotations

import io
import os

from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.semantic_compiler.target_binding import NativeCompilationError

_MARKER = 'atlas.private_staging_slot'
# Candidate limit: reserve at most 192 KiB of the selected 240 KiB temporary
# pool, leaving room for ordinary spills. Native placement remains mandatory.
PRIVATE_STAGING_TILE_LIMIT = 96
_TILE_BYTES = 2048


def storage_policy():
    policy = os.environ.get('ATLAS_STAGING_STORAGE', 'dram')
    if policy not in ('dram', 'vmem', 'vmem-inputs'):
        raise NativeCompilationError('unsupported_semantics',
                                     'ATLAS_STAGING_STORAGE must be dram, vmem or vmem-inputs')
    return policy


def _name(op):
    return op.op_name.data if op.name == 'builtin.unregistered' else op.name


def _access(op):
    name = _name(op)
    if name not in ('atlas.virtual_dma_load_bf16', 'atlas.virtual_dma_store_bf16'):
        if name.startswith('atlas.virtual_dma_'):
            raise ValueError('private staging requires only complete BF16 DMA accesses')
        return None
    load = name.endswith('load_bf16')
    pointer, size = op.operands[1 if load else 2], op.operands[-1]
    if (not isinstance(pointer.owner, arith.ConstantOp)
            or not isinstance(size.owner, arith.ConstantOp)
            or size.owner.value.value.data != _TILE_BYTES):
        raise ValueError('private staging requires constant complete-tile accesses')
    address = int(pointer.owner.value.value.data) & 0xffffffff
    if address % _TILE_BYTES:
        raise ValueError('private staging address is not tile aligned')
    return load, address


def mark_fragment(function, *, source, region, slots, sizes, dram_base):
    """Label accesses using the checked semantic cut and its existing allocation.

    A physical staging address may hold a private value early and a public
    output later. Mark individual accesses by their source value, rather than
    promoting a whole DRAM address regardless of its lifetime.
    """
    private = {node.id for node in source.nodes if node.effect == 'pure'} - set(source.outputs)
    reads = [node.id for node in region.nodes if node.effect in ('input', 'constant')]
    writes = list(region.outputs)
    mappings = []
    for keys in (reads, writes):
        if any(sizes[key] != _TILE_BYTES for key in keys):
            raise ValueError('private staging needs complete BF16 boundary tiles')
        mapping = {dram_base + slots[key] * _TILE_BYTES: key for key in keys}
        if len(mapping) != len(keys):
            raise ValueError('staging cut has ambiguous boundary aliases')
        mappings.append(mapping)
    for op in function.walk():
        if _MARKER in op.attributes:
            raise ValueError('private staging marker already exists')
        access = _access(op)
        if access is None:
            continue
        load, address = access
        mapping = mappings[0 if load else 1]
        if address not in mapping:
            raise ValueError('staging access is absent from the checked source boundary')
        if mapping[address] in private:
            op.attributes[_MARKER] = IntegerAttr(address, 64)


def promote(source, *, readonly_addresses=None):
    """Replace marked full-tile transfers; normal native checks prove placement.

    Slots share the original liveness allocation. Public inputs/constants and
    final outputs keep their original DRAM ABI. The source region allocator
    proves value lifetimes; the native slot verifier proves initialized reads
    and dependencies across branches and repeated invocations.
    """
    context = make_context()
    module = parse_mlir_text(source, context)
    functions = tuple(module.body.block.ops)
    if module.attributes or len(functions) != 1 or not isinstance(functions[0], FuncOp):
        raise ValueError('private staging needs one composed function')
    function = functions[0]
    marked = []
    for op in function.walk():
        marker = op.attributes.get(_MARKER)
        if marker is None:
            continue
        access = _access(op)
        if access is None or not isinstance(marker, IntegerAttr) or marker.value.data != access[1]:
            raise ValueError('private staging marker differs from its checked access')
        marked.append((op, *access))
    addresses = sorted({address for _, _, address in marked})
    admitted = 0 < len(addresses) <= PRIVATE_STAGING_TILE_LIMIT
    receipt = dict(schema='atlas.private_staging.v1', requested='vmem',
                   selected='vmem' if admitted else 'dram',
                   slot_addresses=addresses, candidate_tile_limit=PRIVATE_STAGING_TILE_LIMIT,
                   tile_bytes=_TILE_BYTES,
                   loads=sum(load for _, load, _ in marked),
                   stores=sum(not load for _, load, _ in marked))
    if not admitted:
        receipt['reason'] = 'no private accesses' if not addresses else 'candidate slot limit'
        for op, _, _ in marked:
            del op.attributes[_MARKER]
    else:
        entry = function.body.blocks.first
        start = entry.first_op
        if start is None or _name(start) != 'atlas.virtual_start' or start.next_op is None:
            raise ValueError('private staging requires the original entry state')
        anchor = start.next_op
        slot_type = context.get_type('atlas.virtual_spill')('atlas.virtual_spill', True, False, '')
        slots = {}
        for address in addresses:
            declaration = context.get_op('atlas.virtual_slot').create(result_types=[slot_type],
                attributes=dict(kind=StringAttr('bf16'), words=IntegerAttr(_TILE_BYTES // 4, 32),
                                slot_word=IntegerAttr(-1, 32), purpose=StringAttr('scratch')))
            entry.insert_op_before(declaration, anchor)
            slots[address] = declaration.results[0]
        for op, load, address in marked:
            operands = [op.operands[0], slots[address]]
            if not load:
                operands.append(op.operands[1])
            replacement = context.get_op('atlas.virtual_restore' if load else 'atlas.virtual_save').create(
                operands=operands, result_types=[value.type for value in op.results])
            block = op.parent_block()
            block.insert_op_before(replacement, op)
            for old, new in zip(op.results, replacement.results, strict=True):
                old.replace_all_uses_with(new)
            block.erase_op(op)
    if readonly_addresses is not None:
        from .readonly_staging import cache_readonly
        record = cache_readonly(function, context, readonly_addresses,
                                capacity=max(0, PRIVATE_STAGING_TILE_LIMIT - len(addresses)))
        receipt.update(schema='atlas.private_staging.v2', requested='vmem-inputs', input_cache=record)
        if record['cached_addresses']:
            receipt['selected'] = 'vmem-inputs'
            receipt.pop('reason', None)
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue(), receipt
