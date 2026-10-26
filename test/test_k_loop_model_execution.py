"""Execute counted DMA/MXU loops on the explicitly selected npu_model."""

import dataclasses
import importlib.metadata
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from test_bounded_dma_addresses import loop_inputs, loop_source
from test_spill_model_execution import MODEL_REVISION, model_instruction
from test_virtual_spilling import ROOT, compile_program


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1',
                     'set ATLAS_RUN_NPU_MODEL=1 with the documented model pin')
class KLoopModelExecutionTest(unittest.TestCase):
    def test_unrolled_groups_preserve_ordered_rounding_seeds_and_padding(self):
        import torch
        from atlas_mlir_native.looped_tiling import _virtual, prepare_plan
        from merlin.semantic_compiler.ordered_mnk_tiling import materialize_looped_mnk_inputs
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from rtl.make_ordered_tile_fixture import _bf16, _fp8, _round_bf16
        from test_looped_mnk import seeded_source_request, source_request
        from tools.check_program_decode import decode_binary

        inventory = json.loads((ROOT / 'docs/selected-variant-inventory.json').read_text())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            for k in (257, 545):
                shape = (1, 3, k)
                a = bytes([0x38] + [0x18] * (k - 1))
                weights = bytes(0xB8 if i == 64 else 0x38 if i == 0 else 0x18 for i in range(k)) * 3
                for seeded in (False, True):
                    request = seeded_source_request(shape) if seeded else source_request(*shape)
                    plan = prepare_plan(request)
                    inputs = materialize_looped_mnk_inputs(plan, {'a': a, 'w': weights})
                    seed, expected = bytearray(2048), bytearray(2048)
                    for column in range(3):
                        initial = (0x3F80, 0xBF80, 0x3E80)[column] if seeded else 0
                        struct.pack_into('<H', seed, column * 2, initial)
                        value = initial
                        for left, right in zip(a, weights[column*k:(column+1)*k], strict=True):
                            value = _round_bf16(_bf16(value) + _fp8(left) * _fp8(right))
                        struct.pack_into('<H', expected, column * 2, value)
                    addresses = (0x90000000, 0x90010000)
                    if seeded:
                        inputs = (*inputs, bytes(seed))
                        addresses += (0x90020000,)
                    for factor in (1, 2, 4, 8):
                        with self.subTest(k=k, seeded=seeded, unroll=factor):
                            virtual = _virtual(plan, addresses[:2], 0x90030000,
                                               seed_address=addresses[2] if seeded else None,
                                               k_unroll=factor)
                            _, emitted = compile_program(virtual)
                            binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in emitted['instructions'])
                            decoded = decode_binary(inventory, binary)
                            program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
                            program.dram_base = 0x90000000
                            memory = bytearray((i * 17 + 43) % 256 for i in range(0x30900))
                            for address, payload in zip(addresses, inputs, strict=True):
                                offset = address - program.dram_base
                                memory[offset:offset+len(payload)] = payload
                            program.memory_regions = [(0, torch.tensor(list(memory), dtype=torch.uint8))]
                            config = DefaultHardwareConfig()
                            config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics='rtl',
                                                                          randomize_init=True, init_seed=197 + factor)
                            simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / f'{k}-{seeded}-{factor}.log')),
                                                    program, verbose=False)
                            state = simulation.core.arch_state
                            try:
                                state.vmem.fill_(0xCD)
                                simulation.run(max_cycles=100000)
                                self.assertTrue(state.halted)
                                self.assertEqual(simulation.get_stats().runtime_errors, 0)
                                memory[0x30000:0x30800] = expected
                                self.assertEqual(bytes(state.read_dram(program.dram_base, len(memory)).tolist()), bytes(memory))
                            finally:
                                state.close()

    def test_looped_reduction_preserves_dram_inputs_and_guards(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary

        origin = json.loads(importlib.metadata.distribution('npu-model').read_text('direct_url.json') or '{}')
        self.assertTrue(MODEL_REVISION in origin.get('url', '') or
                        origin.get('vcs_info', {}).get('commit_id') == MODEL_REVISION)
        inventory = json.loads((ROOT / 'docs/selected-variant-inventory.json').read_text())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='atlas-k-loop-') as directory:
            for count in (1, 2, 3, 7):
                with self.subTest(k_tiles=count):
                    _, emitted = compile_program(loop_source(count))
                    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in emitted['instructions'])
                    decoded = decode_binary(inventory, binary)
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
                    program.dram_base = 0x90000000
                    initial = bytearray((i*17+43) % 256 for i in range(0x20900))
                    inputs, output = loop_inputs(count)
                    for address, data in inputs:
                        start = address-program.dram_base
                        initial[start:start+len(data)] = data
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics='rtl',
                                                                  randomize_init=True, init_seed=197)
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / f'{count}.json')),
                                            program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        initial[0x20000:0x20800] = output
                        self.assertEqual(bytes(state.read_dram(program.dram_base, len(initial)).tolist()), bytes(initial))
                    finally:
                        state.close()


if __name__ == '__main__':
    unittest.main()
