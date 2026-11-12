"""MXU1 residency preserves transport, K order, exponent checks and memory."""
from dataclasses import replace
import json
from math import prod
import os
from pathlib import Path
import subprocess
import tempfile
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


def qualified_functional(physical, layout, memory):
    """Execute decoded instructions with separately authenticated adapters."""
    from dataclasses import fields
    from types import SimpleNamespace
    import torch
    import functional_runtime
    from execute_graph import qualified_execution_options
    from model_runtime import authenticate_model
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.functional import FunctionalConfig
    from npu_model.functional.program import FunctionalInstruction, FunctionalProgram
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction

    model_root = Path(os.environ['NPU_MODEL_ROOT'])
    before = authenticate_model(model_root)
    options = qualified_execution_options(os.environ['ATLAS_NATIVE_QUALIFICATION'],
        os.environ['ATLAS_MEMORY_QUALIFICATION'], Path(os.environ['ATLAS_NATIVE_CACHE']))
    torch.set_num_threads(1)
    inventory = json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
    decoded = decode_binary(inventory, binary)
    instructions = [model_instruction(row, inventory, isa) for row in decoded['instructions']]
    program = FunctionalProgram([FunctionalInstruction(i, instruction, i+1, str(instruction))
                                 for i, instruction in enumerate(instructions)])
    values = {field.name: getattr(DefaultHardwareConfig().arch_state_config, field.name)
              for field in fields(FunctionalConfig)}
    values.update(dram_base=DRAM_BASE, dram_size=len(memory), numerics='rtl')
    stats = SimpleNamespace(load_beats=0, store_beats=0)
    load, store = isa.DMA_LOAD.exec, isa.DMA_STORE.exec

    def counted_load(instruction, state):
        stats.load_beats += (state.read_xrf(instruction.rs2) + 31)//32
        return load(instruction, state)

    def counted_store(instruction, state):
        stats.store_beats += (state.read_xrf(instruction.rs2) + 31)//32
        return store(instruction, state)

    class SeededState(functional_runtime.PhysicalFunctionalState):
        def __init__(self, config):
            super().__init__(config)
            rng = np.random.default_rng(217)
            for register in range(1, 32):
                self.write_xrf(register, int(rng.integers(0, 1 << 32)))

    with patch.object(functional_runtime, 'PhysicalFunctionalState', SeededState), \
            patch.object(isa.DMA_LOAD, 'exec', counted_load), patch.object(isa.DMA_STORE, 'exec', counted_store):
        result = functional_runtime.PhysicalFunctionalInterpreter(FunctionalConfig(**values),
            math_proxy=options['math_proxy'], memory_factory=options['memory_factory']).run(
                program, dram={DRAM_BASE: memory}, vmem={0: b'\xcd'*values['vmem_size']})
    assert result.state.halted
    assert before == authenticate_model(model_root)
    print(json.dumps(dict(backend="qualified_native", program_words=len(instructions),
        instructions=result.instructions_executed, load_beats=stats.load_beats,
        store_beats=stats.store_beats, native_dot_calls=options["math_proxy"].calls), sort_keys=True))
    actual = result.state.read_dram(DRAM_BASE, len(memory)).numpy().tobytes()
    begin = layout['output_addresses'][0] - DRAM_BASE
    end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
    assert actual[:begin] == memory[:begin]
    assert actual[end:] == memory[end:]
    return actual, stats


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

    def test_recorded_schedule_survives_environment_change(self):
        plan = prepare_source(source_request(m=1, n=65, k=257, batch=()))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
            layout = execution_layout(plan)
            virtual = virtual_source(plan, layout)
        self.assertEqual(layout['operand_reuse_requested'], 'row-cache')
        self.assertEqual(layout['operand_reuse'], 'row-cache')
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='unknown'):
            self.assertEqual(virtual_source(plan, layout), virtual)
        damaged = dict(layout, operand_reuse='ordered')
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            virtual_source(plan, damaged)

    def test_default_layout_and_sizing_identity_omit_new_fields(self):
        from atlas_mlir_native.quantized_loop_sizing import _identity
        from merlin.semantic_compiler.search import SearchLimits
        plan = prepare_source(source_request(m=1, n=65, k=257, batch=()))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered'):
            layout = execution_layout(plan)
        self.assertNotIn('operand_reuse_requested', layout)
        self.assertNotIn('operand_reuse', layout)
        identity = _identity(plan, virtual_source(plan, layout), ROOT, BIN, SearchLimits(), 'ordered')
        self.assertNotIn('exact_float_unroll', identity['codegen'])
        self.assertNotIn('operand_reuse_requested', identity)

    def test_default_materialization_identity_omits_ordered_policy(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from types import SimpleNamespace
        choice = SimpleNamespace(to_dict=lambda: dict(id='test', nodes=[], inputs=[], outputs=[],
                                                      resident_inputs=[], resident_outputs=[]))
        adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
        source = source_request(m=1, n=64, k=32, batch=())
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered'):
            original = adapter.materialization_identity(source, choice)
        self.assertNotIn('quantized_operand_reuse', original['compilation_identity'])
        self.assertNotIn('exact_float_unroll', original['compilation_identity'])
        self.assertNotIn('exact_binary_lane_groups', original['compilation_identity'])
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered', ATLAS_EXACT_BINARY_LANE_GROUPS='1'):
            grouped = adapter.materialization_identity(source, choice)
        self.assertIs(grouped['compilation_identity'].pop('exact_binary_lane_groups'), True)
        self.assertEqual(grouped, original)
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='ordered'), \
                codegen_options(CodegenOptions(exact_float_unroll=True)):
            unrolled = adapter.materialization_identity(source, choice)
        self.assertIs(unrolled['compilation_identity'].pop('exact_float_unroll'), True)
        self.assertEqual(unrolled, original)
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
            cached = adapter.materialization_identity(source, choice)
        self.assertEqual(cached['compilation_identity']['quantized_operand_reuse'], 'row-cache')
        del cached['compilation_identity']['quantized_operand_reuse']
        self.assertEqual(cached, original)

    def test_native_sizing_binding_and_export_use_recorded_policy(self):
        from atlas_mlir_native import quantized_loop as loop, quantized_loop_c as native_c
        from atlas_mlir_native.quantized_loop_sizing import size_counted, selected_route
        from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        source = source_request(m=1, n=65, k=65, batch=())
        canonical = replace(source, source_identity='canonical-policy-test')
        plan = prepare_source(canonical)
        with tempfile.TemporaryDirectory(prefix='atlas-row-cache-policy-') as directory, \
                codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            directory = Path(directory)
            capture = SelectionCapture(build_snapshot(directory/'native'))
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='row-cache'):
                sized = size_counted(plan, target_source=ROOT)
                loop.compile_source(capture, canonical, target_source=ROOT,
                                    destination=directory/'materialized', limits=SearchLimits())
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='unknown'):
                self.assertIs(size_counted(plan, target_source=ROOT, operand_reuse='row-cache'), sized)
                self.assertEqual(selected_route(plan, directory/'materialized'), 'counted')
                bind_counted(source, canonical, directory/'materialized', directory/'bound',
                             capture, target_source=ROOT)
                record, files = native_c.prepare_c_package(source, directory/'bound', 'cached', target_source=ROOT)
                self.assertTrue(record)
                self.assertTrue(files)
            self.assertEqual((directory/'bound/atlas-virtual.mlir').read_bytes(),
                             (directory/'materialized/atlas-virtual.mlir').read_bytes())

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
        self.check_cycle_sources((source_request(m=1, n=33, k=1, batch=()), shared, head,
                                  source_request(m=33, n=65, k=257, batch=())))

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select the pinned cycle model explicitly')
    def test_functional_model_tinyllama_k2048_n64_tiles(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            self.check_cycle_sources((source_request(m=1, n=2048, k=2048, batch=()),), execute=qualified_functional)

    def check_cycle_sources(self, sources, execute=cycle):
        rng = np.random.default_rng(4029)
        for source in sources:
            plan = prepare_source(source)
            layout = execution_layout(plan)
            variants = []
            for policy in ('ordered', 'row-cache'):
                with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE=policy):
                    selected_layout = execution_layout(plan)
                    virtual = virtual_source(plan, selected_layout)
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
                original, baseline = execute(variants[0], layout, memory)
                actual, cached = execute(variants[1], layout, memory)
                self.assertEqual(actual, original)
                at = layout['status_address'] - DRAM_BASE
                self.assertEqual(bool(int.from_bytes(actual[at:at+4], 'little')), refused)
                mc, nc, kc = ((value + 31)//32 for value in (plan.m, plan.n, plan.k))
                saved = prod(plan.batch_shape) * mc * (nc - 1) * kc * 1024
                self.assertEqual((baseline.load_beats - cached.load_beats) * 32, saved)
                self.assertEqual(cached.store_beats, baseline.store_beats)


if __name__ == '__main__':
    unittest.main()
