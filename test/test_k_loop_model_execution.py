"""Execute counted DMA/MXU loops on the explicitly selected npu_model."""

import dataclasses
import importlib.metadata
import json
import os
import tempfile
import unittest
from pathlib import Path

from test_bounded_dma_addresses import loop_inputs, loop_source
from test_spill_model_execution import MODEL_REVISION, model_instruction
from test_virtual_spilling import ROOT, compile_program


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1',
                     'set ATLAS_RUN_NPU_MODEL=1 with the documented model pin')
class KLoopModelExecutionTest(unittest.TestCase):
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
