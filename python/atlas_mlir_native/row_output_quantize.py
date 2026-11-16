"""Pack a completed BF16 row tile without changing its selected arithmetic."""


def validate_exponent(exponent):
    if exponent is not None and (type(exponent) is not int or not -(1 << 31) <= exponent < (1 << 31)):
        raise ValueError('static row output exponent must be an int32')


def record(exponent):
    validate_exponent(exponent)
    return {} if exponent is None else {'static_output_quantization': {
        'format': 'atlas_e4m3', 'exponent': exponent,
        'physical_layout': 'fp8_column_halves', 'tile_bytes': 1024,
        'frame': 'original BF16 slots; unused upper halves remain private',
    }}


def final_stores(function, output_start, output_bytes):
    """Prove the final full-tile writes on a linear chain of staged blocks.

    Staged softmax reuses output slots for temporary row reductions. Those
    earlier stores must remain BF16 until their readers have completed.
    Branching bodies need a separate path-sensitive proof and are refused.
    """
    from xdsl.dialects import arith, cf, func

    block = function.body.blocks[0]
    visited, last = set(), {}
    while block not in visited:
        visited.add(block)
        for op in block.ops:
            if op.regions:
                raise ValueError('packed row output requires a flat staged body')
            name = op.op_name.data if op.name == 'builtin.unregistered' else op.name
            if name not in ('atlas.virtual_dma_load_bf16', 'atlas.virtual_dma_store_bf16'):
                continue
            index = 1 if name.endswith('load_bf16') else 2
            pointer, size = op.operands[index:]
            if not isinstance(pointer.owner, arith.ConstantOp) or not isinstance(size.owner, arith.ConstantOp):
                raise ValueError('packed row output requires constant DMA spans')
            address = int(pointer.owner.value.value.data) & 0xffffffff
            count = int(size.owner.value.value.data)
            if address < output_start + output_bytes and address + count > output_start:
                if (not output_start <= address < output_start + output_bytes
                        or (address-output_start) % 2048 or count != 2048):
                    raise ValueError('packed row output overlaps an incomplete output tile')
                last[address] = op
        end = block.last_op
        if isinstance(end, func.ReturnOp):
            break
        if not isinstance(end, cf.BranchOp):
            raise ValueError('packed row output requires an unconditional staged chain')
        block = end.successor
    else:
        raise ValueError('packed row output cannot contain a cycle')
    if (visited != set(function.body.blocks)
            or set(last) != set(range(output_start, output_start+output_bytes, 2048))
            or any(op.op_name.data != 'atlas.virtual_dma_store_bf16' for op in last.values())):
        raise ValueError('every output tile must end in a store without later readers')
    return frozenset(last.values())


def pack_store(op, exponent):
    """Replace one full, final BF16 store with the ordinary exact raw PACK.

    The row loop proves this is an output slot, never an intermediate store.
    Raw PACK uses the same scale clamp as the standalone static activation
    quantizer; the host ABI reassembles the tile's two physical column halves. No softmax operation or rounding is changed.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import UnregisteredAttr, UnregisteredOp, i32

    validate_exponent(exponent)
    state, tile, address, size = op.operands
    if not isinstance(size.owner, arith.ConstantOp) or size.owner.value.value.data != 2048:
        raise ValueError('row output quantization requires a complete BF16 tile')
    fp8 = UnregisteredAttr.with_name_and_type('atlas.virtual_fp8', True)(
        'atlas.virtual_fp8', True, False, '')
    scale = arith.ConstantOp.from_int_and_width(max(-127, min(127, exponent)) + 127, i32)
    count = arith.ConstantOp.from_int_and_width(1024, i32)
    packed = UnregisteredOp.with_name('atlas.virtual_raw_pack_fp8').create(
        operands=[tile, scale.result], result_types=[fp8])
    store = UnregisteredOp.with_name('atlas.virtual_dma_store_fp8').create(
        operands=[state, packed.results[0], address, count.result], result_types=[state.type])
    block = op.parent_block()
    block.insert_ops_before([scale, count, packed, store], op)
    op.results[0].replace_all_uses_with(store.results[0])
    block.erase_op(op)
