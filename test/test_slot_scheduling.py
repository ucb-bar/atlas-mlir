"""Mutable scratch dependencies allow independent transfers to overlap."""
import json
import os
import subprocess

import numpy as np
import pytest
from atlas_mlir_native.binding import DRAM_BASE, NATIVE_CHECKS, NATIVE_PIPELINES
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
from test_virtual_ssa import run

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
        order = {line: index for index, line in enumerate(scheduled_lines(source, *options))}
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
    positions = {line: i for i, line in enumerate(scheduled_lines(source))}
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
    order = {line: index for index, line in enumerate(scheduled_lines(source))}
    # Both loads are independent, but B is needed for the first output. The
    # original SSA-only priority could not follow save/restore through B's slot.
    assert order[lines['load_b']] < order[lines['load_a']]


@pytest.mark.skipif(not os.environ.get('ATLAS_RUST_NPU_MODEL'), reason='select Rust simulator')
def test_rust_random_schedules_preserve_whole_memory(tmp_path):
    initial = bytearray([0xa5]) * 73728
    # Raw patterns include NaNs/subnormals; this graph only moves their bytes.
    rng = np.random.default_rng(62019)
    initial[:4096] = rng.integers(0, 65536, 2048, dtype='<u2').tobytes()
    expected = bytearray(initial)
    expected[65536:67584] = initial[:2048]
    expected[67584:69632] = initial[2048:4096]
    expected[69632:71680] = initial[:2048]
    memory = tmp_path / 'memory.bin'
    memory.write_bytes(initial)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'compute': 'cpu', 'dram_base': DRAM_BASE, 'dram_size': len(initial)}))
    for seed in (None, *range(8)):
        source = schedule(fixture(), *((f'random-seed={seed}',) if seed is not None else ()))
        assert source.returncode == 0, source.stderr
        machine = run('atlas-opt', source.stdout, *NATIVE_PIPELINES['mandatory'])
        assert machine.returncode == 0, machine.stderr
        verified = run('atlas-opt', machine.stdout, *NATIVE_CHECKS, '--verify-atlas-timing')
        assert verified.returncode == 0, verified.stderr
        emitted = run('atlas-emit', machine.stdout, '--program-json')
        assert emitted.returncode == 0, emitted.stderr
        physical = json.loads(emitted.stdout)
        work = tmp_path / str(seed)
        work.mkdir()
        program = work / 'program.bin'
        program.write_bytes(b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions']))
        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--config', str(config),
            '--load', f'{DRAM_BASE}:{memory}', '--dump-state', str(work / 'state')],
            check=True, capture_output=True)
        actual = b''.join((work / 'state' / f'dram_{a:016x}.bin').read_bytes()
                         for a in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
        assert actual == expected


@pytest.mark.skipif(not os.environ.get('ATLAS_RUST_NPU_MODEL'), reason='select Rust simulator')
@pytest.mark.parametrize('k', [2048, 5632])
def test_bounded_operand_panels_keep_address_witnesses_and_restores_in_registers(tmp_path, k):
    from atlas_mlir_native.already_quantized import prepare_source
    from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
    from atlas_mlir_native.restoration_budget import instruction_bound
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
    from test_already_quantized import source_request
    from test_exponent_restoration import compile_fixture
    from test_quantized_prefix_reuse import memory_fixture

    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        plan = prepare_source(source_request(m=64, n=65, k=k, batch=()))
        layout = execution_layout(plan, operand_reuse='paired-cache')
        original = virtual_source(plan, layout)
        scheduled = schedule(original)
        assert scheduled.returncode == 0, scheduled.stderr
        programs = []
        for index, virtual in enumerate((original, scheduled.stdout)):
            machine, physical = compile_fixture(virtual)
            assert instruction_bound(virtual, machine, physical)['admitted']
            program = tmp_path / f'{index}.bin'
            program.write_bytes(b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions']))
            assert program.stat().st_size <= 131072
            programs.append(program)
        initial = memory_fixture(plan, layout)
        initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
        config = tmp_path / 'config.json'
        config.write_text(json.dumps({'compute': 'cpu', 'pack_layout': 'register_rows',
                                      'dram_base': DRAM_BASE, 'dram_size': len(initial)}))
        for rejected in (False, True):
            if rejected:
                at = layout['exponent_addresses'][1] - DRAM_BASE
                initial[at:at + 4] = (1000).to_bytes(4, 'little')
            memory = tmp_path / 'memory.bin'
            memory.write_bytes(initial)
            outputs = []
            for index, program in enumerate(programs):
                state = tmp_path / f'state-{index}-{rejected}'
                subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--config', str(config),
                    '--load', f'{DRAM_BASE}:{memory}', '--dump-state', str(state)],
                    check=True, capture_output=True)
                actual = b''.join((state / f'dram_{a:016x}.bin').read_bytes()
                                 for a in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
                start = layout['output_addresses'][0] - DRAM_BASE
                status = layout['status_address'] - DRAM_BASE
                end = status + layout['status_bytes']
                assert actual[:start] == initial[:start]
                assert actual[end:] == initial[end:]
                assert bool(int.from_bytes(actual[status:status + 4], 'little')) == rejected
                outputs.append(actual)
            assert outputs[0] == outputs[1]
