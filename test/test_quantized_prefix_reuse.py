"""Bounded prefix residency preserves the complete ordered K reduction."""
from atlas_mlir_native.options import codegen_options, current_codegen_options

import os
import unittest
from dataclasses import replace
from math import prod

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_operand_reuse import activation_residency
from test_already_quantized import source_request
from test_quantized_operand_reuse import LargeArenaTest, admitted_programs, compare_cycles, export_recorded_policy, memory_fixture


class PrefixReuseTest(LargeArenaTest):
    def test_policy_and_replay(self):
        for k in (32, 2048, 2049, 2337, 5632):
            plan = prepare_source(source_request(m=1, n=65, k=k, batch=()))
            residency = activation_residency(plan, 'prefix-cache')
            self.assertEqual(residency.tile_count, min(64, (k+31)//32))
            layout = execution_layout(plan, operand_reuse='prefix-cache')
            with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered')):
                virtual_source(plan, layout)  # Replay uses the recorded policy.
            if k > 2048:
                layout['operand_reuse_cached_tiles'] -= 1
                with self.assertRaisesRegex(ValueError, 'schedule differs'):
                    virtual_source(plan, layout)
        single = prepare_source(source_request(m=1, n=32, k=5632, batch=()))
        self.assertIsNone(activation_residency(single, 'prefix-cache'))

    def test_sizing_binding_and_export_replay_the_prefix(self):
        import tempfile
        from pathlib import Path
        from atlas_mlir_native.quantized_loop_sizing import size_counted
        from test_exponent_restoration import ROOT

        source = source_request(m=1, n=33, k=2049, batch=())
        canonical = replace(source, source_identity='canonical-prefix-test')
        plan = prepare_source(canonical)
        with tempfile.TemporaryDirectory(prefix='atlas-prefix-policy-') as directory:
            export_recorded_policy(self, source, canonical, Path(directory), 'prefix_cached',
                                   quantized_operand_reuse='prefix-cache')
            with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='prefix-cache')):
                sized = size_counted(plan, target_source=ROOT)
            with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered')):
                self.assertIsNot(size_counted(plan, target_source=ROOT, operand_reuse='prefix-cache'), sized)
            with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='prefix-cache')):
                self.assertIs(size_counted(plan, target_source=ROOT), sized)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select pinned model')
    def test_streamed_tail_matches_ordered_arithmetic_and_saved_bytes(self):
        for m, n, k, batch in ((1, 33, 2049, ()), (33, 65, 2337, ()), (1, 33, 5632, ()),
                                (1, 33, 2049, (2,))):
            plan = prepare_source(source_request(m=m, n=n, k=k, batch=batch))
            layouts = [execution_layout(plan, operand_reuse=policy) for policy in ('ordered', 'prefix-cache')]
            programs = admitted_programs(self, [virtual_source(plan, layout) for layout in layouts])
            saved = prod(batch)*((m+31)//32)*((n+31)//32-1)*64*1024
            for _, original, reused in compare_cycles(self, programs, layouts, memory_fixture(plan, layouts[0])):
                self.assertEqual(32*(original.load_beats-reused.load_beats), saved)
                self.assertEqual(original.store_beats, reused.store_beats)


if __name__ == '__main__':
    unittest.main()
