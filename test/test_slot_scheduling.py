"""Mutable scratch dependencies allow independent transfers to overlap."""

from atlas_mlir_native.binding import DRAM_BASE
from test_virtual_scheduling import (
    S,
    T,
    const,
    dma_load,
    dma_store,
    function,
    schedule,
    scheduled_lines,
    start,
)

P = '!atlas.virtual_spill'


def slot(name):
    return (f'%{name} = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, '
            f'slot_word = -1 : i32, purpose = "scratch"}} : () -> {P}')


def save(before, after, name, value):
    return f'%{after} = "atlas.virtual_save"(%{before}, %{name}, %{value}) : ({S}, {P}, {T}) -> {S}'


def restore(before, after, value, name):
    return f'%{after}, %{value} = "atlas.virtual_restore"(%{before}, %{name}) : ({S}, {P}) -> ({S}, {T})'


def fixture():
    return function('slots', [start(), slot('a'), slot('b'), slot('c'),
        const('x_addr', DRAM_BASE - 2**32), const('y_addr', DRAM_BASE + 2048 - 2**32),
        const('out0', DRAM_BASE + 65536 - 2**32), const('out1', DRAM_BASE + 67584 - 2**32),
        const('out2', DRAM_BASE + 69632 - 2**32), const('size', 2048),
        dma_load('io0', 'load_x', 'x', 'x_addr', 'size', 'bf16'), save('load_x', 'save_a', 'a', 'x'),
        dma_load('save_a', 'load_y', 'y', 'y_addr', 'size', 'bf16'), save('load_y', 'save_b', 'b', 'y'),
        restore('save_b', 'read_a', 'first', 'a'),
        f'%copied = "atlas.virtual_slot_copy"(%read_a, %a, %c) : ({S}, {P}, {P}) -> {S}',
        save('copied', 'overwrite_a', 'a', 'y'), restore('overwrite_a', 'read_c', 'copy', 'c'),
        restore('read_c', 'read_again', 'second', 'a'),
        # A second overwrite also exercises WAW before the final restore.
        save('read_again', 'overwrite_again', 'a', 'x'), restore('overwrite_again', 'read_last', 'third', 'a'),
        dma_store('read_last', 'store0', 'copy', 'out0'), dma_store('store0', 'store1', 'second', 'out1'),
        dma_store('store1', 'store2', 'third', 'out2'), f'return %store2 : {S}'],
        attrs=f'atlas.input_dram_base = {DRAM_BASE} : i64, atlas.output_dram_base = {DRAM_BASE + 65536} : i64, '
              'atlas.output_dram_bytes = 6144 : i64')


def test_every_random_order_retains_slot_hazards_but_allows_independent_dma():
    source = fixture()
    labels = ('save_a', 'load_y', 'read_a', 'copied', 'overwrite_a', 'read_c',
              'read_again', 'overwrite_again', 'read_last')
    lines = {key: next(i for i, row in enumerate(source.splitlines(), 1)
                      if row.lstrip().startswith((f'%{key} =', f'%{key}, '))) for key in labels}
    moved = False
    for options in ((), *((f'random-seed={seed}',) for seed in range(32))):
        order = {line: index for index, line in enumerate(scheduled_lines(source, "enhanced-virtual-scheduling=true", *options))}
        at = {key: order[line] for key, line in lines.items()}
        for before, after in (('save_a', 'read_a'), ('save_a', 'copied'), ('read_a', 'overwrite_a'),
                              ('copied', 'overwrite_a'), ('copied', 'read_c'),
                              ('overwrite_a', 'read_again'), ('read_again', 'overwrite_again'),
                              ('overwrite_a', 'overwrite_again'), ('overwrite_again', 'read_last')):
            assert at[before] < at[after], (options, before, after)
        moved |= at['load_y'] < at['save_a']
    assert moved
    # The ordinary latency/pressure heuristic, not just the random audit,
    # should expose both loads before the first scratch save.
    positions = {line: i for i, line in enumerate(
        scheduled_lines(source, "enhanced-virtual-scheduling=true"))}
    assert positions[lines['load_y']] < positions[lines['save_a']]


def test_overlapping_fixed_slots_are_rejected_before_scheduling():
    source = fixture().replace('slot_word = -1', 'slot_word = 135168')
    result = schedule(source)
    assert result.returncode
    assert 'overlap' in result.stderr


def test_cache_priority_reaches_first_output_through_mutable_slots():
    source = function('priority', [start(), slot('a'), slot('b'),
        const('a_addr', DRAM_BASE - 2**32), const('b_addr', DRAM_BASE + 2048 - 2**32),
        const('out0', DRAM_BASE + 65536 - 2**32), const('out1', DRAM_BASE + 67584 - 2**32),
        const('size', 2048),
        dma_load('io0', 'load_a', 'a_value', 'a_addr', 'size', 'bf16'),
        save('load_a', 'save_a', 'a', 'a_value'),
        dma_load('save_a', 'load_b', 'b_value', 'b_addr', 'size', 'bf16'),
        save('load_b', 'save_b', 'b', 'b_value'),
        restore('save_b', 'read_b', 'first', 'b'),
        restore('read_b', 'read_a', 'second', 'a'),
        dma_store('read_a', 'store0', 'first', 'out0'),
        dma_store('store0', 'store1', 'second', 'out1'), f'return %store1 : {S}'],
        attrs=f'atlas.input_dram_base = {DRAM_BASE} : i64, '
              f'atlas.output_dram_base = {DRAM_BASE + 65536} : i64, '
              'atlas.output_dram_bytes = 4096 : i64')
    lines = {key: next(i for i, row in enumerate(source.splitlines(), 1)
                      if row.lstrip().startswith('%' + key + ','))
             for key in ('load_a', 'load_b')}
    order = {line: index for index, line in enumerate(
        scheduled_lines(source, "enhanced-virtual-scheduling=true"))}
    # Both loads are independent, but B is needed for the first output. The
    # original SSA-only priority could not follow save/restore through B's slot.
    assert order[lines['load_b']] < order[lines['load_a']]
