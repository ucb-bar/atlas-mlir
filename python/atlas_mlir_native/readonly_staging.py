"""Cache repeatedly read immutable BF16 staging inputs within one invocation."""
from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i32
from xdsl.ir import Block

from .private_staging import _access, _name, _TILE_BYTES


def cache_readonly(function, context, addresses, *, capacity):
    """Prefetch declared readonly tiles and retain the original state edges.

    The caller supplies the checked source boundary. Every address must be in
    its immutable DRAM prefix, and the entire composed body must leave that
    prefix unwritten. Repeating the body repeats this initialization; nothing
    is assumed resident across row groups or device invocations.
    """
    if (type(capacity) is not int or capacity < 0 or len(set(addresses)) != len(addresses)
            or any(type(address) is not int or address % _TILE_BYTES for address in addresses)):
        raise ValueError('readonly staging needs distinct aligned tiles and a nonnegative capacity')
    first = function.attributes.get('atlas.input_dram_base')
    last = function.attributes.get('atlas.output_dram_base')
    if (not isinstance(first, IntegerAttr) or not isinstance(last, IntegerAttr)
            or not 0 <= first.value.data < last.value.data <= 1 << 32
            or any(not first.value.data <= address < address + _TILE_BYTES <= last.value.data
                   for address in addresses)):
        raise ValueError('readonly staging tile leaves the declared immutable prefix')
    reads = {address: [] for address in addresses}
    for op in function.walk():
        access = _access(op)
        if access is None:
            continue
        load, address = access
        if address in reads:
            if not load:
                raise ValueError('readonly staging input has a device writer')
            reads[address].append(op)
    candidates = sorted((address for address, users in reads.items() if len(users) > 1),
                        key=lambda address: (-len(reads[address]), address))
    selected = candidates[:capacity]
    record = dict(schema='atlas.readonly_staging.v1', candidate_addresses=candidates,
                  cached_addresses=selected, available_slots=capacity,
                  replaced_loads=sum(len(reads[address]) for address in selected),
                  prefetches=len(selected), initialization='every body invocation')
    if not selected:
        return record
    entry = function.body.blocks.first
    start = entry.first_op
    if start is None or _name(start) != 'atlas.virtual_start' or start.next_op is None:
        raise ValueError('readonly staging requires the original entry state')
    if any(entry in op.successors for op in function.walk()):
        raise ValueError('readonly staging cannot initialize inside an existing entry backedge')
    initial = start.results[0]
    # Counted-loop composition preserves the entry as a preheader. Keep slot
    # declarations there, but place transfers in a real state-taking body so
    # every repeated row initializes its own input cache under the loop guard.
    body = Block(arg_types=[initial.type])
    for op in tuple(entry.ops)[1:]:
        if _name(op) != 'atlas.virtual_slot':
            body.add_op(entry.detach_op(op))
    initial.replace_all_uses_with(body.args[0])
    prefetch = Block(arg_types=[initial.type])
    function.body.insert_block_after(body, entry)
    function.body.insert_block_before(prefetch, body)
    state = prefetch.args[0]
    slot_type = context.get_type('atlas.virtual_spill')('atlas.virtual_spill', True, False, '')
    slots = {}
    for address in selected:
        declaration = context.get_op('atlas.virtual_slot').create(result_types=[slot_type],
            attributes=dict(kind=StringAttr('bf16'), words=IntegerAttr(_TILE_BYTES // 4, 32),
                            slot_word=IntegerAttr(-1, 32), purpose=StringAttr('scratch')))
        pointer = arith.ConstantOp.from_int_and_width(address, i32, truncate_bits=True)
        size = arith.ConstantOp.from_int_and_width(_TILE_BYTES, i32)
        slots[address] = declaration.results[0]
        loaded = context.get_op('atlas.virtual_dma_load_bf16').create(
            operands=[state, pointer.result, size.result],
            result_types=[state.type, reads[address][0].results[1].type])
        saved = context.get_op('atlas.virtual_save').create(
            operands=[loaded.results[0], slots[address], loaded.results[1]], result_types=[state.type])
        entry.add_op(declaration)
        prefetch.add_ops((pointer, size, loaded, saved))
        state = saved.results[0]
    entry.add_op(cf.BranchOp(prefetch, initial))
    prefetch.add_op(cf.BranchOp(body, state))
    for address in selected:
        for op in reads[address]:
            restored = context.get_op('atlas.virtual_restore').create(
                operands=[op.operands[0], slots[address]], result_types=[value.type for value in op.results])
            block = op.parent_block()
            block.insert_op_before(restored, op)
            for old, new in zip(op.results, restored.results, strict=True):
                old.replace_all_uses_with(new)
            block.erase_op(op)
    return record
