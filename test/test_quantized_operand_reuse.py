"""MXU1 residency preserves transport, K order, exponent checks and memory."""
from dataclasses import replace
import json
from math import prod
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE, NATIVE_PIPELINES
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_operand_reuse import activation_residency
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_exponent_restoration import BIN, ROOT, compile_fixture
from test_quantized_loop import shared, head, packed


def cycle(physical, layout, memory):
    import torch
    import npu_model.simulation as simulation
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.software.program import InstantiableProgram
    from npu_speed_of_light import roofline
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction

    torch.set_num_threads(1)
    inventory = json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
    decoded = decode_binary(inventory, binary)
    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
    program.dram_base = DRAM_BASE
    program.memory_regions = [(0, torch.frombuffer(bytearray(memory), dtype=torch.uint8))]
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics='rtl', randomize_init=True, init_seed=217)
    observed = {}

    class QuietLogger:
        FU_PID = 0
        ARCH_PID = 1
        def __init__(self, *args, **kwargs): pass
        def noop(self, *args, **kwargs): pass
        close = log_cycle = log_insn = log_retire = log_dependency = noop
        log_arch_value = log_stage_start = log_stage_end = noop

    class CheckedSimulation(simulation.Simulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.core.arch_state.vmem.fill_(0xcd)

        def close(self):
            try:
                assert not self.get_stats().runtime_errors
                actual = self.core.arch_state.read_dram(DRAM_BASE, len(memory)).numpy().tobytes()
                begin = layout['output_addresses'][0] - DRAM_BASE
                end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
                assert actual[:begin] == memory[:begin]
                assert actual[end:] == memory[end:]
                observed['memory'] = actual
            finally:
                super().close()

    with patch.object(simulation, 'Logger', QuietLogger), patch.object(roofline, 'Simulation', CheckedSimulation):
        result = roofline.execute(program, config, max_cycles=2_000_000)
    return observed['memory'], result


class QuantizedOperandReuseTest(unittest.TestCase):
    def test_quantized_cache_does_not_select_or_change_mxu0(self):
        from atlas_mlir_native.looped_tiling import _virtual
        from types import SimpleNamespace
        geometry = SimpleNamespace(counts=(1, 2, 1), tile_bytes=(1024, 1024, 2048),
                                   buffer_bytes=(1024, 2048, 4096))
        addresses, output = (DRAM_BASE, DRAM_BASE + 2048), DRAM_BASE + 4096
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered'):
            original = _virtual(geometry, addresses, output, unit=0)
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
            self.assertEqual(_virtual(geometry, addresses, output, unit=0), original)
        with self.assertRaisesRegex(ValueError, 'requires MXU1'):
            _virtual(geometry, addresses, output, unit=0, cache_a=True)

    def test_capacity_single_consumer_and_expansion_refusals_keep_ordered_source(self):
        for k, n, admitted in ((1, 33, True), (2048, 64, True), (2049, 64, False), (32, 32, False)):
            plan = prepare_source(source_request(m=1, n=n, k=k, batch=()))
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
                self.assertEqual(activation_residency(plan) is not None, admitted)
        plan = prepare_source(source_request(m=1, k=2049, batch=()))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
            cached = virtual_source(plan, execution_layout(plan))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered'):
            self.assertEqual(cached, virtual_source(plan, execution_layout(plan)))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='unknown'):
            with self.assertRaises(NativeCompilationError):
                activation_residency(plan)

    def test_missing_cache_writer_is_rejected_by_standard_memory_proofs(self):
        plan = prepare_source(source_request(m=35, n=67, k=65, batch=()))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
            virtual = virtual_source(plan, execution_layout(plan))
        lines = virtual.splitlines()
        writer = next(line for line in lines if '"atlas.virtual_save"' in line and '%cached_a0,' in line)
        # Retain the state chain but erase the only writer of the first slot.
        import re
        output = writer.strip().split(' = ')[0]
        before = re.search(r'"atlas.virtual_save"\((%[^,]+)', writer)[1]
        changed = '\n'.join(line.replace(output, before) for line in lines if line != writer)
        result = subprocess.run([str(BIN/'atlas-opt'), *NATIVE_PIPELINES['mandatory'], '-'],
                                input=changed, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select the pinned cycle model explicitly')
    def test_cycle_model_shared_batched_tails_scales_and_status(self):
        rng = np.random.default_rng(4029)
        for source in (source_request(m=1, n=33, k=1, batch=()), shared, head,
                       source_request(m=33, n=65, k=257, batch=())):
            plan = prepare_source(source)
            layout = execution_layout(plan)
            variants = []
            for policy in ('ordered', 'row-cache'):
                with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE=policy):
                    virtual = virtual_source(plan, layout)
                machine, physical = compile_fixture(virtual)
                self.assertTrue(instruction_bound(virtual, machine, physical)['admitted'])
                variants.append(physical)
            end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
            memory = bytearray([0xa5]) * (end + 4096)
            for node, address, size in zip(plan.operand_ids, layout['input_addresses'], layout['input_buffer_bytes']):
                codes = np.array([0, 0x08, 0x28, 0x38, 0x3f, 0x77, 0x78, 0x7e, 0x88, 0xb8, 0xfe], dtype=np.uint8)
                values = rng.choice(codes, size=source.node(node).type.shape)
                data = packed(values).tobytes()
                self.assertEqual(len(data), size)
                memory[address - DRAM_BASE:address - DRAM_BASE + size] = data
            for geometry, address in zip(layout['exponent_layouts'], layout['exponent_addresses']):
                page = np.zeros(geometry['bytes']//4, dtype='<i4').reshape(-1, 512)
                if geometry['mode'] == 'row':
                    for batch in range(geometry['batches']):
                        for row in range(geometry['rows']):
                            page[batch*geometry['tiles'] + row//32, row%32] = (row//32 + batch) % 3 - 1
                else:
                    page[:, 0] = np.arange(len(page)) % 3 - 1
                memory[address-DRAM_BASE:address-DRAM_BASE+geometry['bytes']] = page.tobytes()
            for refused in (False, True):
                if refused:
                    address = layout['exponent_addresses'][0] - DRAM_BASE
                    memory[address:address+4] = (1000).to_bytes(4, 'little')
                original, baseline = cycle(variants[0], layout, memory)
                actual, cached = cycle(variants[1], layout, memory)
                self.assertEqual(actual, original)
                at = layout['status_address'] - DRAM_BASE
                self.assertEqual(bool(int.from_bytes(actual[at:at+4], 'little')), refused)
                mc, nc, kc = ((value + 31)//32 for value in (plan.m, plan.n, plan.k))
                saved = prod(plan.batch_shape) * mc * (nc - 1) * kc * 1024
                self.assertEqual((baseline.load_beats - cached.load_beats) * 32, saved)
                self.assertEqual(cached.store_beats, baseline.store_beats)


if __name__ == '__main__':
    unittest.main()
