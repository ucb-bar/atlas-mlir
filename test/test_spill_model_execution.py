"""Optional spill execution on a pinned npu_model, separate from RTL checks."""

import dataclasses
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from test_mxu_spilling import (
    mxu_diamond,
    mxu_diamond_inputs,
    mxu_inputs,
    mxu_loop,
    mxu_loop_inputs,
    mxu_pressure,
)
from test_spill_resources import (
    block_pressure,
    entry_pressure,
    fp8_block_pressure,
    scalar_pressure,
)
from test_virtual_spilling import ROOT, compile_program, pressure

MODEL_REVISION = '337e9f249f938a1dff007ddf8a763e5cd4b3aa2b'


def model_instruction(row, inventory, isa):
    """Bind a decoded instruction and require an identical re-encoded word.

    npu_model executes instruction objects. Decode uses the existing RTL
    inventory, independently of the emitter and its JSON operand annotations.
    """
    from npu_model.isa import SBType, SType, UJType, UType, VIType

    mode = row['mode_id']
    variant = next(v for v in inventory['variants'] if v['id'] == mode)
    name = mode.removesuffix('_ANY') if mode.startswith('DMA_') else variant['model_classes'][0]
    cls, word = getattr(isa, name), row['word_u32']
    operands = {'rd': (word >> 7) & 31, 'rs1': (word >> 15) & 31, 'rs2': (word >> 20) & 31,
                'vd': (word >> 7) & 63, 'vs1': (word >> 13) & 63, 'vs2': (word >> 19) & 63,
                'ch': (word >> 12) & 7}
    if issubclass(cls, SBType):
        immediate = row['fields']['offset_bytes']
    elif issubclass(cls, UJType):
        immediate = row['fields']['offset']
    elif issubclass(cls, SType):
        immediate = ((word >> 25) << 5) | ((word >> 7) & 31)
    elif issubclass(cls, VIType):
        immediate = word >> 16
    elif issubclass(cls, UType):
        immediate = word >> 12
    else:
        immediate = word >> 20
    operands['imm'] = immediate
    instruction = cls(**{name: operands[name] for name in inspect.signature(cls).parameters})
    if instruction.to_bytecode() != word:
        raise ValueError(f'npu_model instruction differs from word {row["word_index"]}')
    return instruction


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'set ATLAS_RUN_NPU_MODEL=1 with the documented model pin')
class SpillModelExecutionTest(unittest.TestCase):
    def test_mxu_spills_restore_before_continuation_on_both_units(self):
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
        with tempfile.TemporaryDirectory(prefix='atlas-mxu-spill-model-') as directory:
            cases = [(unit, continuation, reset, iterations, None)
                     for unit in (0, 1)
                     for continuation, reset, iterations in ((False, False, 0), (True, False, 0),
                                                             (True, True, 0), (True, False, 1),
                                                             (True, False, 3))]
            cases += [(None, True, False, 0, take) for take in (0, 1, 2, 3)]
            for unit, continuation, reset, iterations, take in cases:
                with self.subTest(unit=unit, continuation=continuation, reset=reset, iterations=iterations, take=take):
                    if take is None:
                        tiles, outputs = mxu_inputs(continuation, reset=reset)
                        source = mxu_loop(unit, iterations) if iterations else mxu_pressure(unit, continuation=continuation, reset=reset)
                        if iterations:
                            tiles, outputs = mxu_loop_inputs(iterations)
                    else:
                        source = mxu_diamond()
                        tiles, outputs = mxu_diamond_inputs(take)
                    _, emitted = compile_program(source)
                    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in emitted['instructions'])
                    decoded = decode_binary(inventory, binary)
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
                    program.dram_base = 0x90000000
                    initial = bytearray((i*17+43) % 256 for i in range(0x21900))
                    for i, tile in enumerate(tiles):
                        initial[i*2048:i*2048+len(tile)] = tile
                    if take is not None:
                        initial[0x10000:0x10400] = take.to_bytes(4, 'little') + bytes(1020)
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics='rtl',
                                                                  randomize_init=True, init_seed=187)
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / f'{unit}-{continuation}-{reset}-{iterations}-{take}.json')),
                                            program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        for i, data in enumerate(outputs):
                            initial[0x20000+i*2048:0x20800+i*2048] = data
                        self.assertEqual(bytes(state.read_dram(program.dram_base, len(initial)).tolist()), bytes(initial))
                    finally:
                        state.close()

    def test_spill_programs_execute_with_preserved_inputs_and_guards(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram

        sys.path.insert(0, str(ROOT))
        try:
            from tools.check_program_decode import decode_binary
        finally:
            sys.path.pop(0)

        origin = json.loads(importlib.metadata.distribution('npu-model').read_text('direct_url.json') or '{}')
        self.assertTrue(MODEL_REVISION in origin.get('url', '') or
                        origin.get('vcs_info', {}).get('commit_id') == MODEL_REVISION,
                        'install the documented npu_model revision')
        inventory = json.loads((ROOT / 'docs/selected-variant-inventory.json').read_text())
        cases = [('bf16', pressure(), 32), ('block-loop', block_pressure(True), 16),
                 ('scalar', scalar_pressure(), 2), ('scalar-loop', scalar_pressure(True), 2),
                 ('entry', entry_pressure(), 2), ('entry-i1', entry_pressure(True), 2),
                 ('fp8-loop', fp8_block_pressure(), 33)]
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='atlas-spill-model-') as directory:
            for name, source, count in cases:
                with self.subTest(case=name):
                    _, emitted = compile_program(source)
                    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in emitted['instructions'])
                    decoded = decode_binary(inventory, binary)
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
                    program.dram_base = 0x90000000
                    size = 1024 if name == 'fp8-loop' else 2048
                    tiles = [bytes((j*17+i*29) % 256 for j in range(size)) for i in range(count)]
                    inputs = [(0x90000000+i*2048, tile) for i, tile in enumerate(tiles)]
                    if name == 'block-loop':
                        inputs.append((0x9001f800, bytes([73])*1024))
                    if name.startswith('scalar'):
                        inputs.append((0x90010000, (1).to_bytes(4, 'little')+bytes(1020)))
                    if name.startswith('entry'):
                        controls = b''.join(value.to_bytes(4, 'little') for value in range(1, 19))
                        if name == 'entry-i1':
                            controls += (3).to_bytes(4, 'little')
                        inputs.append((0x90010000, controls+bytes(1024-len(controls))))
                    outputs = count if name in ('bf16', 'block-loop', 'fp8-loop') else 1
                    inputs.append((0x90020000+outputs*size, bytes([91])*64))
                    program.memory_regions = [(address-program.dram_base, torch.tensor(list(data), dtype=torch.uint8)) for address, data in inputs]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics='rtl')
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / f'{name}.json')), program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        for i in range(outputs):
                            expected = tiles[(i+3) % count] if name in ('block-loop', 'fp8-loop') else tiles[i]
                            self.assertEqual(bytes(state.read_dram(0x90020000+i*size, size).tolist()), expected)
                        for address, data in inputs:
                            self.assertEqual(bytes(state.read_dram(address, len(data)).tolist()), data)
                    finally:
                        state.close()


if __name__ == '__main__':
    unittest.main()
