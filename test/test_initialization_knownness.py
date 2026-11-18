"""Unknown tensor-derived scalar data cannot establish memory-safety facts."""
import os
from pathlib import Path
import subprocess

from test_composed_initialization_budget import source


def compile_source(text):
    return subprocess.run([str(Path(os.environ['ATLAS_OOT_BIN_DIR'])/'atlas-opt'),
        '--atlas-compile-unscheduled','-'], input=text, text=True, capture_output=True)


def test_unknown_loop_carried_data_does_not_consume_replay_budget():
    text = source(iterations=512, scalar_facts=0).replace('@proof()', '@proof(%runtime_data: i32)')
    text = text.replace('atlas.input_dram_base =',
        'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
    text = text.replace('^loop(%loaded,%zero : !atlas.virtual_state,i32)',
        '^loop(%loaded,%zero,%runtime_data : !atlas.virtual_state,i32,i32)')
    text = text.replace('%counter: i32):','%counter: i32, %data: i32):')
    facts = []
    value = '%data'
    for i in range(2048):
        facts.append(f'%data_{i} = arith.addi {value}, %counter : i32')
        value = f'%data_{i}'
    text = text.replace('%one = arith.constant 1', '\n'.join(facts)+'\n%one = arith.constant 1')
    text = text.replace('^loop(%next,%n : !atlas.virtual_state,i32)',
        f'^loop(%next,%n,{value} : !atlas.virtual_state,i32,i32)')
    result = compile_source(text)
    assert result.returncode == 0, result.stderr[:2000]
    checked = subprocess.run([str(Path(os.environ['ATLAS_OOT_BIN_DIR'])/'atlas-opt'),
        '--verify-atlas-generated-schedule','-'], input=result.stdout, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr[:2000]


def test_unknown_scalar_address_remains_unproved():
    text = source(iterations=1).replace('@proof()', '@proof(%runtime_address: i32)')
    text = text.replace('atlas.input_dram_base =',
        'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
    text = text.replace('(%body_state, %tile, %out, %bytes)',
        '(%body_state, %tile, %runtime_address, %bytes)')
    result = compile_source(text)
    assert result.returncode != 0
    assert 'DRAM address and byte length must be proven' in result.stderr, result.stderr[:2000]


def test_known_loop_argument_becoming_unknown_still_checks_both_paths():
    text = source(iterations=2).replace('@proof()', '@proof(%runtime_step: i32)')
    text = text.replace('atlas.input_dram_base =',
        'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
    text = text.replace('%n = arith.addi %counter,%one',
        '%n = arith.addi %counter,%runtime_step')
    result = compile_source(text)
    assert result.returncode != 0
    assert 'scalar-state cycle' in result.stderr, result.stderr[:2000]
