"""Paired operand caching composes with the original checked product epilogue."""
import json
import os
import subprocess

import numpy as np
import pytest
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_epilogue import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_exponent_restoration import compile_fixture
from test_quantized_epilogue import case, packed_extra
from test_quantized_prefix_reuse import memory_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('dtype,operand,m,n,k', [
    ('bf16', 0, 33, 35, 1025), ('bf16', 1, 64, 65, 2049),
    ('f32', 0, 97, 35, 1024), ('f32', 1, 64, 128, 2048),
])
def test_cached_epilogue_matches_original_rounding_guards_and_memory(tmp_path, dtype, operand, m, n, k):
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        source, _, _ = case(dtype, operand, m=m, n=n, k=k, batch=())
        plan = prepare_source(source)
        programs = []
        for policy in ('wide-cache', 'paired-cache'):
            layout = execution_layout(plan, operand_reuse=policy)
            if policy == 'paired-cache':
                assert layout['operand_reuse'] == policy
            virtual = virtual_source(plan, layout)
            machine, physical = compile_fixture(virtual)
            assert instruction_bound(virtual, machine, physical)['admitted']
            binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
            assert len(binary) <= 131072
            program = tmp_path / f'{policy}.bin'
            program.write_bytes(binary)
            programs.append(program)
        initial = memory_fixture(plan, layout)
        initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
        codes = np.resize(np.array([0, 0x8000, 0x3f00, 0xbf80, 0x4000], dtype='<u2'), (m, n))
        payload = packed_extra(codes, plan).tobytes()
        at = layout['input_addresses'][-1] - DRAM_BASE
        initial[at:at + len(payload)] = payload
        config = tmp_path / 'config.json'
        config.write_text(json.dumps({'compute': 'cpu', 'pack_layout': 'register_rows',
                                      'dram_base': DRAM_BASE, 'dram_size': len(initial)}))
        for rejected in (False, True):
            if rejected:
                at = layout['exponent_addresses'][1] - DRAM_BASE + ((n - 1) // 32 * 512 + (n - 1) % 32) * 4
                initial[at:at + 4] = (1000).to_bytes(4, 'little')
            memory = tmp_path / 'memory.bin'
            memory.write_bytes(initial)
            outputs, cycles = [], []
            for index, program in enumerate(programs):
                state = tmp_path / f'state-{int(rejected)}-{index}'
                result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
                    '--config', str(config), '--load', f'{DRAM_BASE}:{memory}',
                    '--dump-state', str(state)], check=True, capture_output=True, text=True)
                cycles.append(json.loads(result.stdout)['cycles'])
                actual = b''.join((state / f'dram_{a:016x}.bin').read_bytes()
                                 for a in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
                start = layout['output_addresses'][0] - DRAM_BASE
                end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
                assert actual[:start] == initial[:start]
                assert actual[end:] == initial[end:]
                status = layout['status_address'] - DRAM_BASE
                assert bool(int.from_bytes(actual[status:status + 4], 'little')) == rejected
                outputs.append(actual)
            assert outputs[0] == outputs[1]
            if not rejected:
                assert cycles[1] < cycles[0]
