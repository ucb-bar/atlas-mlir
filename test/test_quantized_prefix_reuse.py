"""Bounded prefix residency preserves the complete ordered K reduction."""
import os
import unittest
from math import prod
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_operand_reuse import activation_residency
from atlas_mlir_native.restoration_budget import instruction_bound
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_operand_reuse import cycle
from test_quantized_loop import packed


def memory_fixture(plan, layout):
    """Finite signed operands, nonuniform exponents and protected surroundings."""
    end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
    memory = bytearray([0xa5]) * (end + 4096)
    rng = np.random.default_rng(1873)
    for key, address, size in zip(plan.operand_ids, layout['input_addresses'], layout['input_buffer_bytes']):
        values = rng.choice(np.array([0, 0x20, 0xa0, 0x28, 0xa8], dtype=np.uint8),
                            size=plan.source.node(key).type.shape)
        payload = packed(values).tobytes()
        assert len(payload) == size
        memory[address-DRAM_BASE:address-DRAM_BASE+size] = payload
    for geometry, address in zip(layout['exponent_layouts'], layout['exponent_addresses']):
        page = np.zeros(geometry['bytes']//4, dtype='<i4').reshape(-1, 512)
        if geometry['mode'] == 'row':
            for batch in range(geometry['batches']):
                for row in range(geometry['rows']):
                    page[batch*geometry['tiles']+row//32, row%32] = (row//32+batch)%3-1
        else:
            page[:, 0] = np.arange(len(page))%3-1
        memory[address-DRAM_BASE:address-DRAM_BASE+geometry['bytes']] = page.tobytes()
    return memory


class PrefixReuseTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        # The model workflow explicitly selects its 64 MiB I/O arena.
        options = codegen_options(CodegenOptions(io_memory_bytes=64*1024*1024))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_policy_and_replay(self):
        for k in (32, 2048, 2049, 2337, 5632):
            plan = prepare_source(source_request(m=1, n=65, k=k, batch=()))
            residency = activation_residency(plan, 'prefix-cache')
            self.assertEqual(residency.tile_count, min(64, (k+31)//32))
            layout = execution_layout(plan, operand_reuse='prefix-cache')
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
                virtual_source(plan, layout)  # Replay uses the recorded policy.
            if k > 2048:
                layout['operand_reuse_cached_tiles'] -= 1
                with self.assertRaisesRegex(ValueError, 'schedule differs'):
                    virtual_source(plan, layout)
        single = prepare_source(source_request(m=1, n=32, k=5632, batch=()))
        self.assertIsNone(activation_residency(single, 'prefix-cache'))

    def test_sizing_binding_and_export_replay_the_prefix(self):
        import tempfile
        from dataclasses import replace
        from pathlib import Path
        from atlas_mlir_native import quantized_loop, quantized_loop_c
        from atlas_mlir_native.quantized_loop_sizing import size_counted
        from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
        from merlin.semantic_compiler.search import SearchLimits
        from test_exponent_restoration import ROOT
        from test_looped_mnk import build_snapshot

        source = source_request(m=1, n=33, k=2049, batch=())
        canonical = replace(source, source_identity='canonical-prefix-test')
        plan = prepare_source(canonical)
        with tempfile.TemporaryDirectory(prefix='atlas-prefix-policy-') as directory:
            directory = Path(directory)
            capture = SelectionCapture(build_snapshot(directory/'native'))
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='prefix-cache'):
                sized = size_counted(plan, target_source=ROOT)
                quantized_loop.compile_source(capture, canonical, target_source=ROOT,
                    destination=directory/'materialized', limits=SearchLimits())
            with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
                self.assertIs(size_counted(plan, target_source=ROOT,
                    operand_reuse='prefix-cache'), sized)
                bind_counted(source, canonical, directory/'materialized', directory/'bound',
                    capture, target_source=ROOT)
                record, files = quantized_loop_c.prepare_c_package(source,
                    directory/'bound', 'prefix_cached', target_source=ROOT)
                self.assertTrue(record)
                self.assertTrue(files)
            self.assertEqual((directory/'bound/atlas-virtual.mlir').read_bytes(),
                             (directory/'materialized/atlas-virtual.mlir').read_bytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select pinned model')
    def test_streamed_tail_matches_ordered_arithmetic_and_saved_bytes(self):
        for m, n, k, batch in ((1, 33, 2049, ()), (33, 65, 2337, ()), (1, 33, 5632, ()),
                                (1, 33, 2049, (2,))):
            plan = prepare_source(source_request(m=m, n=n, k=k, batch=batch))
            layouts, programs = [], []
            for policy in ('ordered', 'prefix-cache'):
                layout = execution_layout(plan, operand_reuse=policy)
                virtual = virtual_source(plan, layout)
                machine, physical = compile_fixture(virtual)
                self.assertTrue(instruction_bound(virtual, machine, physical)['admitted'])
                self.assertLessEqual(4*len(physical['instructions']), 131072)
                layouts.append(layout)
                programs.append(physical)
            memory = memory_fixture(plan, layouts[0])
            for refused in (False, True):
                if refused:
                    at = layouts[0]['exponent_addresses'][0]-DRAM_BASE
                    memory[at:at+4] = (1000).to_bytes(4, 'little')
                expected, original = cycle(programs[0], layouts[0], memory)
                actual, reused = cycle(programs[1], layouts[1], memory)
                self.assertEqual(actual, expected)
                at = layouts[0]['status_address']-DRAM_BASE
                self.assertEqual(bool(int.from_bytes(actual[at:at+4], 'little')), refused)
                saved = prod(batch)*((m+31)//32)*((n+31)//32-1)*64*1024
                self.assertEqual(32*(original.load_beats-reused.load_beats), saved)
                self.assertEqual(original.store_beats, reused.store_beats)


if __name__ == '__main__':
    unittest.main()
